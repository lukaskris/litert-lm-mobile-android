package com.example.qnn_litertlm_gemma

import android.content.Context
import android.os.Build
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.LogSeverity
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

/**
 * Data class for model performance metrics, populated from LiteRT-LM's
 * experimental getBenchmarkInfo() API when available.
 */
data class PerformanceMetrics(
    val initializationTimeMs: Long = 0,
    val timeToFirstTokenMs: Long = 0,
    val prefillTokensPerSecond: Double = 0.0,
    val decodeTokensPerSecond: Double = 0.0,
    val activeBackend: String = "Unknown",
    val memoryUsageMb: Long = 0
)

/**
 * Singleton manager for LiteRT-LM Engine.
 * Handles model initialization, conversation management,
 * and multimodal message sending (text, image, audio).
 *
 * Backend fallback chain: NPU → CPU → GPU
 *
 * GPU is last because libLiteRtGpuAccelerator.so can SIGBUS on some devices,
 * which kills the process before fallback can be attempted.
 */
class LiteRTLMManager private constructor(private val context: Context) {

    @Volatile private var engine: Engine? = null
    @Volatile private var conversation: com.google.ai.edge.litertlm.Conversation? = null
    @Volatile private var isInitialized = false
    @Volatile private var currentBackendName: String = "CPU"

    /** Check if the engine has been initialized. */
    fun isInitialized(): Boolean = isInitialized

    /**
     * Dedicated dispatcher for inference — deliberately isolated from
     * Dispatchers.IO / Default / Main so that:
     *
     *   1. Thread priority is BELOW normal (OS scheduler prefers UI thread)
     *   2. Pool size is capped at half the CPU cores (leaves cores free for UI)
     *   3. UI never starves — inference "steps aside" when the system is busy
     *
     * On an 8-core MediaTek: pool = 4 threads, priority = below normal
     * UI thread (Main) stays at normal priority → gets CPU time first.
     */
    private val inferenceDispatcher: ExecutorCoroutineDispatcher by lazy {
        val coreCount = Runtime.getRuntime().availableProcessors()
        // Use at most half the cores — always leave 2+ for UI / system
        val poolSize = maxOf(1, minOf(coreCount / 2, coreCount - 2))

        Timber.tag(TAG).i("Inference dispatcher: $poolSize threads (device has $coreCount cores)")

        Executors.newFixedThreadPool(poolSize) { runnable ->
            Thread(runnable).apply {
                name = "litertlm-inference-$id"
                // Below normal — OS gives UI thread priority over us
                priority = Thread.NORM_PRIORITY - 1
            }
        }.asCoroutineDispatcher()
    }

    companion object {
        private const val TAG = "LiteRTLMManager"
        private const val NPU_INIT_TIMEOUT_MS = 30_000L

        init {
            // Reduce native logging overhead — only show ERROR+ in production.
            // Default is INFO which adds CPU overhead on low-end devices.
            try {
                Engine.setNativeMinLogSeverity(LogSeverity.ERROR)
            } catch (e: Exception) {
                Timber.tag(TAG).w("Failed to set native log severity: ${e.message}")
            }
            // NOTE: Do NOT manually load LiteRt / GPU / OpenCL libs here.
            // litertlm-android v0.11.0 ships its own libLiteRt.so and
            // libLiteRtClGlAccelerator.so.  The SDK's NativeLibraryLoader
            // loads litertlm_jni.so automatically; the dynamic linker then
            // resolves libLiteRt.so from the same directory.
            // Manually loading an old/wrong version causes ABI mismatch → SIGSEGV.
        }

        /**
         * Load QNN (Qualcomm NPU) native libraries lazily.
         * Only call when NPU backend is actually selected — these libs
         * don't exist on non-Qualcomm devices and waste startup time.
         */
        @Volatile
        private var qnnLibsLoaded = false

        fun loadQnnLibraries(): Boolean {
            if (qnnLibsLoaded) {
                Timber.tag(TAG).d("[LoadModel] QNN libs already loaded — skipping")
                return true
            }
            return synchronized(this) {
                if (qnnLibsLoaded) return true
                try {
                    Timber.tag(TAG).d("[LoadModel] Loading QNN native libs...")
                    System.loadLibrary("QnnSystem")
                    Timber.tag(TAG).d("[LoadModel]   + QnnSystem OK")
                    System.loadLibrary("QnnHtp")
                    Timber.tag(TAG).d("[LoadModel]   + QnnHtp OK")
                    System.loadLibrary("QnnHtpV79Stub")
                    Timber.tag(TAG).d("[LoadModel]   + QnnHtpV79Stub OK")
                    System.loadLibrary("GemmaModelConstraintProvider")
                    Timber.tag(TAG).d("[LoadModel]   + GemmaModelConstraintProvider OK")
                    System.loadLibrary("LiteRtDispatch_Qualcomm")
                    Timber.tag(TAG).d("[LoadModel]   + LiteRtDispatch_Qualcomm OK")
                    qnnLibsLoaded = true
                    Timber.tag(TAG).d("[LoadModel] All QNN native libraries loaded successfully")
                    true
                } catch (e: UnsatisfiedLinkError) {
                    Timber.tag(TAG).e("[LoadModel] QNN lib FAILED: ${e.message}")
                    false
                }
            }
        }

        @Volatile
        private var INSTANCE: LiteRTLMManager? = null

        fun getInstance(context: Context): LiteRTLMManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: LiteRTLMManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    /**
     * Initialize the LiteRT-LM Engine with the specified model.
     * Uses NPU → CPU → GPU fallback chain.
     * GPU is last to avoid native SIGBUS crash that kills the process.
     */
    suspend fun initialize(
        modelPath: String,
        systemPrompt: String? = null,
        isEmbedding: Boolean = false,
        preferredBackend: String? = null
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        Timber.tag(TAG).d("[LoadModel] === START initialize === path=$modelPath, preferred=$preferredBackend, isEmbedding=$isEmbedding")
        if (isInitialized) {
            Timber.tag(TAG).d("[LoadModel] Already initialized — cleaning up previous engine first")
            cleanup()
        }

        try {
            if (isEmbedding) {
                Timber.tag(TAG).w( "Embedding mode not supported in this version")
                currentBackendName = "CPU"
            } else {
                // Build ordered backend list based on preference
                Timber.tag(TAG).d("[LoadModel] Step 1: Building backend list...")
                val backends = buildBackendList(preferredBackend)
                Timber.tag(TAG).d("[LoadModel] Step 2: Backend list = ${backends.map { it.name }}, starting fallback loop...")
                initializeEngineWithFallback(modelPath, backends)
            }
            isInitialized = true
            Timber.tag(TAG).d("[LoadModel] === END initialize OK === backend=$currentBackendName")
            Result.success(true)
        } catch (e: Throwable) {
            Timber.tag(TAG).e(e, "[LoadModel] === END initialize FAILED === ${e.javaClass.simpleName}: ${e.message}")
            Result.failure(Exception(e))
        }
    }

    /**
     * Detect the SoC vendor from Build props.
     * QNN NPU delegate only works on Qualcomm Snapdragon.
     */
    private fun detectSoCVendor(): String {
        val socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.uppercase()
        } else ""
        val hardware = Build.HARDWARE.uppercase()
        val board = Build.BOARD.uppercase()

        return when {
            // Qualcomm Snapdragon
            socModel.contains("SNAPDRAGON") || socModel.contains("QCOM") ||
                hardware.contains("QCOM") || hardware.contains("QUALCOMM") ||
                board.contains("SDM") || board.contains("SM") ||
                socModel.matches(Regex(".*\\bSM-?\\d{4}.*")) -> "QUALCOMM"

            // MediaTek Dimensity / Helio / Kompanio
            socModel.contains("MT") || hardware.contains("MT") ||
                board.contains("MT") -> "MEDIATEK"

            // Samsung Exynos
            socModel.contains("EXYNOS") || hardware.contains("EXYNOS") ||
                board.contains("EXYNOS") -> "SAMSUNG"

            // Unisoc / Spreadtrum
            socModel.contains("UNISOC") || socModel.contains("SPREADTRUM") ||
                hardware.contains("UNISOC") || hardware.contains("SPRD") -> "UNISOC"

            else -> "UNKNOWN"
        }
    }

    /**
     * Build an ordered list of backends to try based on the SoC.
     * QNN NPU delegate is Qualcomm-only — skip on other vendors
     * to avoid wasting 30-120s on a guaranteed failure.
     */
    private fun buildBackendList(preferred: String?): List<BackendFactory> {
        val socVendor = detectSoCVendor()
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        Timber.tag(TAG).i( "SoC vendor detected: $socVendor (model=${if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "N/A"}, hardware=${Build.HARDWARE}, nativeLibDir=$nativeLibDir)")

        val gpuBackend = BackendFactory("GPU") { Backend.GPU() }
        val cpuBackend = BackendFactory("CPU") { Backend.CPU() }

        val backends = when {
            socVendor == "QUALCOMM" && loadQnnLibraries() -> {
                Timber.tag(TAG).i( "Qualcomm SoC with QNN libs: NPU → CPU → GPU")
                val npuBackend = BackendFactory("NPU", nativeLibraryDir = nativeLibDir) {
                    Backend.NPU(nativeLibraryDir = nativeLibDir)
                }
                // CPU before GPU — GPU SIGBUS kills the process (cannot catch native signal).
                // Only try GPU if CPU also fails.
                listOf(npuBackend, cpuBackend, gpuBackend)
            }
            socVendor == "QUALCOMM" -> {
                Timber.tag(TAG).w( "Qualcomm SoC but QNN libs failed to load: GPU → CPU")
                listOf(gpuBackend, cpuBackend)
            }
            else -> {
                Timber.tag(TAG).i( "Non-Qualcomm SoC ($socVendor): CPU → GPU (skipping NPU)")
                listOf(gpuBackend, cpuBackend)
            }
        }

        // If user explicitly requested a specific backend, move it to front
        if (preferred != null) {
            val preferredUpper = preferred.uppercase()
            val preferredIdx = backends.indexOfFirst { it.name == preferredUpper }
            if (preferredIdx > 0) {
                return listOf(backends[preferredIdx]) + backends.filterIndexed { i, _ -> i != preferredIdx }
            }
        }

        return backends
    }

    /**
     * Try each backend in order; stop at the first one that works.
     * NPU gets a bounded timeout — if it doesn't respond in time,
     * we skip it and move to GPU/CPU.
     */
    private fun initializeEngineWithFallback(modelPath: String, backends: List<BackendFactory>) {
        var lastError: Throwable? = null

        Timber.tag(TAG).d("[LoadModel] Fallback loop: will try ${backends.map { it.name }}")

        for ((index, factory) in backends.withIndex()) {
            val attemptStart = System.currentTimeMillis()
            try {
                Timber.tag(TAG).d("[LoadModel] Fallback [${index + 1}/${backends.size}]: trying backend=${factory.name}")

                if (factory.name == "NPU") {
                    // Run NPU init on a separate thread with timeout.
                    // NPU can hang for 60-120s on models without AOT payloads —
                    // we don't want to block the caller indefinitely.
                    Timber.tag(TAG).d("[LoadModel] NPU detected — using timeout wrapper (${NPU_INIT_TIMEOUT_MS}ms)")
                    val result = runWithTimeout(NPU_INIT_TIMEOUT_MS) {
                        initializeEngine(modelPath, factory)
                    }
                    if (result.isFailure) {
                        throw result.exceptionOrNull()
                            ?: Exception("NPU initialization timed out or failed")
                    }
                } else {
                    initializeEngine(modelPath, factory)
                }

                val attemptMs = System.currentTimeMillis() - attemptStart
                Timber.tag(TAG).d("[LoadModel] Backend ${factory.name} SUCCEEDED in ${attemptMs}ms")
                return
            } catch (e: Throwable) {
                val attemptMs = System.currentTimeMillis() - attemptStart
                Timber.tag(TAG).d("[LoadModel] Backend ${factory.name} FAILED after ${attemptMs}ms: ${e.javaClass.simpleName}: ${e.message}")
                if (factory.name == "NPU") {
                    Timber.tag(TAG).d("[LoadModel] NPU failed — falling back to next backend")
                }
                lastError = e
            }
        }

        Timber.tag(TAG).e("[LoadModel] ALL backends failed — throwing last error")
        throw lastError ?: IllegalStateException("All backends failed")
    }

    /**
     * Run a blocking operation with a timeout.
     * Returns Result.success if completed, Result.failure if timed out or threw.
     */
    private fun runWithTimeout(timeoutMs: Long, block: () -> Unit): Result<Unit> {
        val exceptionRef = AtomicReference<Throwable?>(null)
        val thread = Thread {
            try {
                block()
            } catch (e: Throwable) {
                exceptionRef.set(e)
            }
        }
        thread.name = "npu-init-timeout"
        thread.start()
        thread.join(timeoutMs)

        return when {
            thread.isAlive -> {
                // Thread still running — NPU is stuck. Log and move on.
                Timber.tag(TAG).w( "NPU init timed out after ${timeoutMs}ms — interrupting")
                thread.interrupt()
                Result.failure(Exception("NPU initialization timed out after ${timeoutMs}ms"))
            }
            exceptionRef.get() != null -> Result.failure(exceptionRef.get()!!)
            else -> Result.success(Unit)
        }
    }

    @OptIn(ExperimentalApi::class)
    private fun initializeEngine(modelPath: String, factory: BackendFactory) {
        Timber.tag(TAG).d("[LoadModel] --- initializeEngine START --- backend=${factory.name}, path=$modelPath")

        val file = File(modelPath)
        if (!file.exists()) {
            Timber.tag(TAG).e("[LoadModel] Step: File existence check FAILED — $modelPath does not exist")
            throw java.io.FileNotFoundException("Model file not found at $modelPath")
        }
        if (!file.canRead()) {
            Timber.tag(TAG).e("[LoadModel] Step: File readability check FAILED — $modelPath not readable")
            throw java.io.IOException("Model file not readable (permissions?) at $modelPath")
        }

        val fileSize = file.length()
        Timber.tag(TAG).d("[LoadModel] Step: File checks OK — size=${fileSize} bytes (${fileSize / 1024 / 1024}MB), backend=${factory.name}")

        // Validate model file — LiteRT-LM models should be at least 100MB.
        // Corrupt/incomplete downloads or scoped storage issues can produce
        // tiny files that fail with "TF_LITE_PREFILL_DECODE not found".
        if (fileSize < 100_000_000L) {
            Timber.tag(TAG).e("[LoadModel] Step: File size validation FAILED — too small (${fileSize} bytes)")
            throw java.io.IOException(
                "Model file too small ($fileSize bytes / ${fileSize / 1024 / 1024}MB). " +
                "File is likely corrupt or incomplete. Delete and re-download. Path: $modelPath"
            )
        }

        // Quick header check — read first 4 bytes to verify it's a valid model file.
        try {
            file.inputStream().buffered().use { stream ->
                val header = ByteArray(4)
                val read = stream.read(header)
                Timber.tag(TAG).d("[LoadModel] Step: Header bytes = ${header.take(read).map { String.format("%02X", it) }}")
            }
        } catch (e: Exception) {
            Timber.tag(TAG).w("[LoadModel] Step: Could not read model header: ${e.message}")
        }

        Timber.tag(TAG).d("[LoadModel] Step: Creating backend instance for ${factory.name}...")
        val backend = factory.create()
        Timber.tag(TAG).d("[LoadModel] Step: Backend instance created OK: $backend")

        // Set ADSP_LIBRARY_PATH for NPU — QNN needs to find SKEL stubs.
        // Only set when actually trying NPU backend.
        if (factory.name == "NPU") {
            try {
                val libDir = factory.nativeLibraryDir ?: context.applicationInfo.nativeLibraryDir
                android.system.Os.setenv("ADSP_LIBRARY_PATH", libDir, true)
                Timber.tag(TAG).d("[LoadModel] Step: Set ADSP_LIBRARY_PATH=$libDir for NPU")
            } catch (e: Exception) {
                Timber.tag(TAG).w("[LoadModel] Step: Failed to set ADSP_LIBRARY_PATH: ${e.message}")
            }
        }

        // Vision backend strategy:
        //   NPU main → GPU vision  (NPU doesn't use GPU, so GPU is free)
        //   GPU main → CPU vision  (free GPU for UI rendering + token decode)
        //   CPU main → CPU vision  (GPU accelerator can SIGSEGV on some devices)
        //
        // When main=GPU, moving vision to CPU cuts GPU load ~50% — the GPU
        // only runs token decode (~50-100ms per step), leaving gaps for the
        // Android HWUI renderer to push frames.
        val visionBackend = when (factory.name) {
            "CPU" -> {
                Timber.tag(TAG).d("[LoadModel] Step: Vision backend = CPU (GPU skipped — main backend is CPU)")
                Backend.CPU()
            }
            "GPU" -> {
                Timber.tag(TAG).d("[LoadModel] Step: Vision backend = CPU (freeing GPU for UI + decode)")
                Backend.CPU()
            }
            else -> {
                Timber.tag(TAG).d("[LoadModel] Step: Vision backend = GPU (NPU main — GPU is free)")
                Backend.GPU()
            }
        }

        Timber.tag(TAG).d("[LoadModel] Step: Building EngineConfig... modelPath=$modelPath, cacheDir=${context.cacheDir.path}")

        // Enable speculative decoding (MTP) for Gemma 4 — delivers >2x faster decode.
        // Only enable for CPU/NPU backends; GPU does not support speculative decoding
        // and enabling it causes SIGSEGV in native code.
        val enableMtp = factory.name != "GPU"
        ExperimentalFlags.enableSpeculativeDecoding = enableMtp
        Timber.tag(TAG).d("[LoadModel] Step: Speculative decoding (MTP) = $enableMtp (backend=${factory.name})")

        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = backend,
            visionBackend = visionBackend,
            audioBackend = Backend.CPU(),
            cacheDir = context.cacheDir.path
        )

        val startTime = System.currentTimeMillis()
        Timber.tag(TAG).d("[LoadModel] Step: Calling Engine(engineConfig) constructor...")
        val candidateEngine = try {
            Engine(engineConfig)
        } catch (e: Throwable) {
            // Engine constructor can crash with native SIGSEGV on corrupt models,
            // but sometimes it throws a Java exception first — catch those.
            Timber.tag(TAG).e("[LoadModel] Step: Engine constructor threw ${e.javaClass.simpleName}: ${e.message}")
            throw e
        }
        Timber.tag(TAG).d("[LoadModel] Step: Engine constructor OK (${System.currentTimeMillis() - startTime}ms), calling initialize()...")

        try {
            candidateEngine.initialize()
            val duration = System.currentTimeMillis() - startTime
            Timber.tag(TAG).d("[LoadModel] Step: Engine.initialize() SUCCEEDED in ${duration}ms")

            // Verify conversation works — wrap separately so test failure
            // doesn't cascade into closing the engine
            try {
                Timber.tag(TAG).d("[LoadModel] Step: Creating test conversation...")
                val testConv = candidateEngine.createConversation(ConversationConfig())
                testConv.close()
                Timber.tag(TAG).d("[LoadModel] Step: Test conversation OK")
            } catch (e: Throwable) {
                Timber.tag(TAG).w("[LoadModel] Step: Test conversation failed (non-fatal): ${e.message}")
            }
        } catch (e: Throwable) {
            val duration = System.currentTimeMillis() - startTime
            Timber.tag(TAG).e("[LoadModel] Step: Engine.initialize() FAILED after ${duration}ms — ${e.javaClass.simpleName}: ${e.message}")
            if (factory.name == "NPU" && e.message?.contains("TF_LITE_AUX") == true) {
                Timber.tag(TAG).e("[LoadModel] NPU missing AOT payload and JIT compilation failed or was not triggered.")
            }
            try {
                candidateEngine.close()
            } catch (closeErr: Throwable) {
                Timber.tag(TAG).w("[LoadModel] Step: Error closing failed engine: ${closeErr.message}")
            }
            throw e  // Re-throw so fallback chain works
        }

        engine = candidateEngine
        currentBackendName = factory.name
        Timber.tag(TAG).d("[LoadModel] --- initializeEngine END OK --- backend=$currentBackendName")
    }

    /**
     * Start a new conversation.
     */
    fun startConversation(systemPrompt: String? = null) {
        Timber.tag(TAG).d("[LoadModel] startConversation called — initialized=$isInitialized, engine=${engine != null}")
        if (!isInitialized || engine == null) {
            throw IllegalStateException("Engine not initialized.")
        }

        // Close existing conversation — the engine only supports one session at a time.
        try {
            conversation?.close()
        } catch (e: Exception) {
            Timber.tag(TAG).w("Error closing previous conversation: ${e.message}")
        }
        conversation = null

        val conversationConfig = ConversationConfig(
            systemInstruction = if (systemPrompt != null) Contents.of(systemPrompt) else null,
            samplerConfig = SamplerConfig(
                temperature = 0.4,
                topK = 64,
                topP = 0.9
            )
        )
        conversation = engine?.createConversation(conversationConfig)
        Timber.tag(TAG).d("[LoadModel] Conversation created OK — systemPrompt=${systemPrompt != null}")
    }

    /**
     * Send a text-only message and stream the response.
     * Inference runs on [inferenceDispatcher] so it never blocks the UI thread.
     */
    suspend fun sendMessage(text: String): Flow<String> {
        ensureConversation()
        return conversation!!.sendMessageAsync(text)
            .map { msg -> msg.toString() }
            .buffer(capacity = 64)
            .flowOn(inferenceDispatcher)
    }

    /**
     * Send a multimodal message with optional image and/or audio.
     *
     * IMPORTANT: The caller MUST call [startConversation] BEFORE invoking this.
     * This method does NOT reset the conversation — it uses whatever session exists.
     */
    fun sendMultimodalMessage(
        text: String,
        imagePath: String? = null,
        audioBytes: ByteArray? = null
    ): Flow<String> {
        ensureConversation()
        Timber.tag(TAG).i( "Multimodal message: text='${text.take(50)}', image=$imagePath, audio=${audioBytes != null}")
        val conv = conversation!!
        return kotlinx.coroutines.flow.flow {
            // This block runs on inferenceDispatcher (via flowOn below).
            // Content.ImageFile() decodes the bitmap here — off the main thread.
            val contents = buildMultimodalContents(text, imagePath, audioBytes)
            conv.sendMessageAsync(contents)
                .collect { msg ->
                    emit(msg.toString())
                    // Yield between tokens: gives the GPU a brief window
                    // to process pending UI render work before the next
                    // token decode starts. On low-end GPUs (Mali-G52 etc.)
                    // this prevents the UI from starving during inference.
                    kotlinx.coroutines.yield()
                }
        }
            .buffer(capacity = 64)
            .flowOn(inferenceDispatcher)
    }

    /**
     * Build multimodal Contents on the IO thread (called inside flowOn scope).
     * Image decoding inside Content.ImageFile is CPU-heavy — this avoids
     * blocking whatever thread calls sendMultimodalMessage.
     */
    private fun buildMultimodalContents(
        text: String,
        imagePath: String?,
        audioBytes: ByteArray?
    ): Contents {
        val contentParts = mutableListOf<Content>()

        if (imagePath != null) {
            val imgFile = File(imagePath)
            if (imgFile.exists() && imgFile.canRead()) {
                Timber.tag(TAG).i("Adding image: ${imgFile.length() / 1024}KB, ${imgFile.absolutePath}")
                contentParts.add(Content.ImageFile(imagePath))
            } else {
                Timber.tag(TAG).e("Image file not accessible: $imagePath (exists=${imgFile.exists()})")
            }
        }
        if (audioBytes != null) {
            contentParts.add(Content.AudioBytes(audioBytes))
        }

        contentParts.add(Content.Text(text))
        return Contents.of(*contentParts.toTypedArray())
    }

    private fun ensureConversation() {
        if (!isInitialized || engine == null) {
            throw IllegalStateException("Engine not initialized.")
        }
        if (conversation == null) {
            startConversation()
        }
    }

    fun getActiveBackendName(): String = currentBackendName

    /**
     * Get real performance metrics from LiteRT-LM's getBenchmarkInfo() API.
     * Returns null if conversation is not active or API is unavailable.
     */
    @kotlin.OptIn(com.google.ai.edge.litertlm.ExperimentalApi::class)
    fun getPerformanceMetrics(): PerformanceMetrics? {
        if (!isInitialized || conversation == null) return null
        return try {
            val benchmarkInfo = conversation?.getBenchmarkInfo()
            PerformanceMetrics(
                prefillTokensPerSecond = benchmarkInfo?.lastPrefillTokensPerSecond ?: 0.0,
                decodeTokensPerSecond = benchmarkInfo?.lastDecodeTokensPerSecond ?: 0.0,
                activeBackend = currentBackendName,
                memoryUsageMb = getMemoryUsageMb()
            )
        } catch (e: Exception) {
            Timber.tag(TAG).w("getBenchmarkInfo not available: ${e.message}")
            PerformanceMetrics(
                activeBackend = currentBackendName,
                memoryUsageMb = getMemoryUsageMb()
            )
        }
    }

    fun getMemoryUsageMb(): Long {
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
    }
    
    fun cleanup() {
        try {
            conversation?.close()
            engine?.close()
            conversation = null
            engine = null
            isInitialized = false
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Error during cleanup")
        }
    }

    /**
     * Release the inference thread pool. Call when the Activity is destroyed.
     */
    fun shutdown() {
        cleanup()
        // Accessing the lazy creates the pool if it was never used,
        // then immediately closes it — harmless on shutdown.
        inferenceDispatcher.close()
    }
}

/**
 * Factory for creating backends lazily.
 * This avoids constructing NPU/GPU backends that might throw
 * before we're ready to handle the exception.
 */
private data class BackendFactory(
    val name: String,
    val nativeLibraryDir: String? = null,
    val create: () -> Backend
)
