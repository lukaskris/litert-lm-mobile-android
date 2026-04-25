package com.example.qnn_litertlm_gemma

import android.content.Context
import android.os.Build
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors

/**
 * Data class for model performance metrics
 */
data class PerformanceMetrics(
    val initializationTimeMs: Long = 0,
    val timeToFirstTokenMs: Long = 0,
    val tokensPerSecond: Double = 0.0,
    val activeBackend: String = "Unknown",
    val memoryUsageMb: Long = 0
)

/**
 * Singleton manager for LiteRT-LM Engine.
 * Handles model initialization, conversation management,
 * and multimodal message sending (text, image, audio).
 *
 * Backend fallback chain: NPU → GPU → CPU
 */
class LiteRTLMManager private constructor(private val context: Context) {

    private var engine: Engine? = null
    private var conversation: com.google.ai.edge.litertlm.Conversation? = null
    private var isInitialized = false
    private var currentBackendName: String = "CPU"

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

        Log.i(TAG, "Inference dispatcher: $poolSize threads (device has $coreCount cores)")

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
     * Uses NPU → GPU → CPU fallback chain.
     */
    suspend fun initialize(
        modelPath: String,
        systemPrompt: String? = null,
        isEmbedding: Boolean = false,
        preferredBackend: String? = null
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        Log.i(TAG, "Initializing for: $modelPath (preferred: $preferredBackend)")
        if (isInitialized) {
            cleanup()
        }
        
        try {
            if (isEmbedding) {
                Log.w(TAG, "Embedding mode not supported in this version")
                currentBackendName = "CPU"
            } else {
                // Build ordered backend list based on preference
                val backends = buildBackendList(preferredBackend)
                initializeEngineWithFallback(modelPath, backends)
            }
            isInitialized = true
            Log.i(TAG, "Initialization SUCCEEDED on backend: $currentBackendName")
            Result.success(true)
        } catch (e: Throwable) {
            Log.e(TAG, "Initialization FAILED: ${e.message}", e)
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
        Log.i(TAG, "SoC vendor detected: $socVendor (model=${if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "N/A"}, hardware=${Build.HARDWARE})")

        val npuBackend = BackendFactory("NPU") {
            Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        }
        val gpuBackend = BackendFactory("GPU") { Backend.GPU() }
        val cpuBackend = BackendFactory("CPU") { Backend.CPU() }

        val backends = when (socVendor) {
            "QUALCOMM" -> {
                // QNN works — try NPU first as preferred
                Log.i(TAG, "Qualcomm SoC: NPU → GPU → CPU")
                listOf(npuBackend, gpuBackend, cpuBackend)
            }
            else -> {
                // Non-Qualcomm: skip NPU entirely, GPU → CPU
                Log.i(TAG, "Non-Qualcomm SoC ($socVendor): GPU → CPU (skipping NPU)")
                listOf(gpuBackend, cpuBackend)
            }
        }

        // If user explicitly requested a specific backend, respect it
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
     */
    private fun initializeEngineWithFallback(modelPath: String, backends: List<BackendFactory>) {
        var lastError: Throwable? = null

        for (factory in backends) {
            val attemptStart = System.currentTimeMillis()
            try {
                Log.i(TAG, "Trying backend: ${factory.name}")
                initializeEngine(modelPath, factory)
                val attemptMs = System.currentTimeMillis() - attemptStart
                Log.i(TAG, "Backend ${factory.name} SUCCEEDED in ${attemptMs}ms")
                return
            } catch (e: Throwable) {
                val attemptMs = System.currentTimeMillis() - attemptStart
                Log.w(TAG, "Backend ${factory.name} FAILED after ${attemptMs}ms: ${e.message}")
                lastError = e
            }
        }

        throw lastError ?: IllegalStateException("All backends failed")
    }

    private fun initializeEngine(modelPath: String, factory: BackendFactory) {
        val file = File(modelPath)
        if (!file.exists()) {
            throw java.io.FileNotFoundException("Model file not found at $modelPath")
        }
        if (!file.canRead()) {
            throw java.io.IOException("Model file not readable (permissions?)")
        }
        Log.i(TAG, "Model file: ${file.length()} bytes, backend: ${factory.name}")

        val backend = factory.create()

        Log.i(TAG, "Initializing Engine with backend: ${factory.name}")

        // Vision always uses GPU — CPU vision encoding is fundamentally too slow
        // for production use (5+ minutes on non-Qualcomm devices).
        // On MediaTek Mali, this causes a ~12s GPU lockup during vision encoding,
        // but the alternative (CPU) never completes in reasonable time.
        val visionBackend = Backend.GPU()
        Log.i(TAG, "Vision backend: GPU")

        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = backend,
            visionBackend = visionBackend,
            audioBackend = Backend.CPU(),
            cacheDir = context.cacheDir.path
        )

        val startTime = System.currentTimeMillis()
        val candidateEngine = Engine(engineConfig)

        try {
            candidateEngine.initialize()
            val duration = System.currentTimeMillis() - startTime
            Log.i(TAG, "Engine initialization SUCCEEDED in ${duration}ms")

            // Verify conversation works — wrap separately so test failure
            // doesn't cascade into closing the engine
            try {
                val testConv = candidateEngine.createConversation(ConversationConfig())
                testConv.close()
                Log.i(TAG, "Test conversation OK")
            } catch (e: Throwable) {
                Log.w(TAG, "Test conversation failed (non-fatal): ${e.message}")
            }
        } catch (e: Throwable) {
            val duration = System.currentTimeMillis() - startTime
            Log.e(TAG, "Engine initialization FAILED after ${duration}ms: ${e.message}")
            if (factory.name == "NPU" && e.message?.contains("TF_LITE_AUX") == true) {
                Log.e(TAG, "NPU missing AOT payload and JIT compilation failed or was not triggered.")
            }
            try {
                candidateEngine.close()
            } catch (closeErr: Throwable) {
                Log.w(TAG, "Error closing failed engine: ${closeErr.message}")
            }
            throw e
        }

        engine = candidateEngine
        currentBackendName = factory.name
    }

    /**
     * Start a new conversation.
     */
    fun startConversation(systemPrompt: String? = null) {
        if (!isInitialized || engine == null) {
            throw IllegalStateException("Engine not initialized.")
        }
        
        val conversationConfig = ConversationConfig(
            systemInstruction = if (systemPrompt != null) Contents.of(systemPrompt) else null,
            samplerConfig = SamplerConfig(
                temperature = 0.7,
                topK = 40,
                topP = 0.9
            )
        )
        conversation = engine?.createConversation(conversationConfig)
    }

    /**
     * Send a text-only message and stream the response.
     * Inference runs on Dispatchers.IO so it never blocks the UI or Default thread pool.
     * buffer() decouples the native producer from the Kotlin consumer.
     */
    fun sendMessage(text: String): Flow<String> {
        ensureConversation()
        return conversation!!.sendMessageAsync(text)
            .map { msg ->
                // Periodically yield CPU so UI thread can run.
                // Without this, a tight native loop at below-normal priority
                // can still cause scheduler latency on low-end devices.
                Thread.yield()
                msg.toString()
            }
            .buffer(capacity = 64)
            .flowOn(inferenceDispatcher)
    }

    /**
     * Send a multimodal message with optional image and/or audio.
     *
     * buildMultimodalContents() is CPU-heavy (decodes bitmap, prepares GPU input).
     * It is called inside flowOn(Dispatchers.IO) so it NEVER blocks the caller
     * (typically the Main thread).
     */
    fun sendMultimodalMessage(
        text: String,
        imagePath: String? = null,
        audioBytes: ByteArray? = null
    ): Flow<String> {
        ensureConversation()
        Log.i(TAG, "Multimodal message: text='${text.take(50)}', image=$imagePath, audio=${audioBytes != null}")
        val conv = conversation!!
        return kotlinx.coroutines.flow.flow {
            // This block runs on Dispatchers.IO (via flowOn below).
            // Content.ImageFile() decodes the bitmap here — off the main thread.
            val contents = buildMultimodalContents(text, imagePath, audioBytes)
            conv.sendMessageAsync(contents)
                .collect { msg ->
                    Thread.yield()
                    emit(msg.toString())
                }
        }
            .buffer(capacity = 64)
            .flowOn(Dispatchers.IO)
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
            contentParts.add(Content.ImageFile(imagePath))
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
            Log.e(TAG, "Error during cleanup", e)
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
    val create: () -> Backend
)
