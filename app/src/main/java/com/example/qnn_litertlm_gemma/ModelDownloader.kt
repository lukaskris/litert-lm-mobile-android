package com.example.qnn_litertlm_gemma

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.Dispatchers
import timber.log.Timber
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Utility class for loading LiteRT-LM models.
 *
 * Model source: /storage/emulated/0/Download/<filename>
 * Push via: adb push gemma-4-E2B-it.litertlm /sdcard/Download/
 *
 * The model is COPIED to internal storage before loading because:
 *   /storage/emulated/0/ uses FUSE filesystem which does NOT support
 *   mmap alignment required by LiteRT's GPU accelerator.
 *   Reading directly from FUSE causes SIGBUS (BUS_ADRALN) crash.
 *
 * Requires MANAGE_EXTERNAL_STORAGE permission.
 */
class ModelDownloader(private val context: Context) {

    private val prefs: SharedPreferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "ModelDownloader"
        private const val COPY_BUFFER_SIZE = 1024 * 1024 // 1MB buffer for fast copy

        private fun tag() = Timber.tag(TAG)
        private const val KEY_HF_TOKEN = "hf_token"

        val AVAILABLE_MODELS = listOf(
            ModelConfig(
                id = "gemma4-e2b",
                name = "Gemma 4 E2B (Int4)",
                filename = "gemma-4-E2B-it.litertlm",
                url = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
                systemPrompt = "You are Gemma 4, a powerful multimodal AI assistant by Google, running privately on-device. You can understand text, images, and audio.",
                preferredBackend = null
            ),
            ModelConfig(
                id = "gemma-3n",
                name = "Gemma 3n (Int4)",
                filename = "gemma3n.litertlm",
                url = "https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-int4.litertlm",
                systemPrompt = "You are Gemma, a helpful AI assistant powered by Google's LiteRT-LM running on device."
            ),
            ModelConfig(
                id = "qwen3-0.6b",
                name = "Qwen 3 0.6B (Int4)",
                filename = "qwen3-0.6b.litertlm",
                url = "https://huggingface.co/litert-community/Qwen3-0.6B/resolve/main/qwen3-0.6b-int4.litertlm",
                systemPrompt = "You are Qwen, a helpful AI assistant running on device."
            ),
            ModelConfig(
                id = "gemma3-1b",
                name = "Gemma 3 1B (Int4)",
                filename = "gemma3-1b-it-int4.litertlm",
                url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.litertlm",
                systemPrompt = "You are Gemma, a helpful AI assistant running on device."
            )
        )
    }

    fun saveToken(token: String) {
        prefs.edit().putString(KEY_HF_TOKEN, token).apply()
    }

    fun getToken(): String? {
        return prefs.getString(KEY_HF_TOKEN, null)
    }

    /** External source — model MUST be pushed here first */
    private val downloadDir = File("/storage/emulated/0/Download")

    /**
     * Check if a model is ready to use — either already copied to internal,
     * or available in /storage/emulated/0/Download/ for copying.
     */
    fun isModelDownloaded(modelConfig: ModelConfig): Boolean {
        val internal = File(context.filesDir, modelConfig.filename)
        if (internal.exists() && internal.length() > 500_000_000L) return true

        val external = File(downloadDir, modelConfig.filename)
        val exists = external.exists() && external.canRead() && external.length() > 500_000_000L
        tag().i("isModelDownloaded(${modelConfig.filename}): internal=${internal.exists()}, external=$exists (${external.length() / 1024 / 1024}MB)")
        return exists
    }

    /**
     * Get the model path on internal storage (ext4 — safe for mmap).
     * This is a suspend function because copying 2.4GB takes time and
     * must run on Dispatchers.IO to avoid blocking the UI thread.
     *
     * Flow:
     *   1. Check internal storage — if valid (>500MB), use directly (instant)
     *   2. Check disk space before copying
     *   3. Copy from /storage/emulated/0/Download/ → internal storage with progress
     *   4. Verify copied file size
     *   5. Return internal path
     */
    suspend fun getModelPath(modelConfig: ModelConfig): String = withContext(Dispatchers.IO) {
        val internal = File(context.filesDir, modelConfig.filename)

        // 1. Already in internal storage and valid — instant return
        if (internal.exists() && internal.length() > 500_000_000L) {
            tag().i("getModelPath: using internal (${internal.length() / 1024 / 1024}MB): ${internal.absolutePath}")
            return@withContext internal.absolutePath
        }

        // Delete incomplete internal file
        if (internal.exists()) {
            tag().w("getModelPath: internal file too small (${internal.length() / 1024 / 1024}MB), deleting")
            internal.delete()
        }

        // 2. Check external source
        val external = File(downloadDir, modelConfig.filename)
        if (!external.exists() || !external.canRead()) {
            tag().e("getModelPath: external file not found at ${external.absolutePath}")
            return@withContext internal.absolutePath
        }

        // 3. Check disk space before copying
        val sourceSize = external.length()
        val freeSpace = context.filesDir.usableSpace
        tag().i("getModelPath: source=${sourceSize / 1024 / 1024}MB, free=${freeSpace / 1024 / 1024}MB")

        if (freeSpace < sourceSize * 2) {
            tag().e("getModelPath: not enough disk space! need ${sourceSize / 1024 / 1024}MB, have ${freeSpace / 1024 / 1024}MB free")
            return@withContext internal.absolutePath
        }

        // 4. Copy with progress logging
        tag().i("getModelPath: copying ${external.absolutePath} → ${internal.absolutePath}")
        val copied = copyToInternalStorage(external, internal)
        if (copied) {
            tag().i("getModelPath: copy OK, using ${internal.absolutePath}")
        } else {
            tag().e("getModelPath: copy FAILED")
        }

        return@withContext internal.absolutePath
    }

    /**
     * Copy model to internal storage (ext4) so LiteRT can mmap it safely.
     * Uses 1MB buffer for fast copy and logs progress every 500MB.
     */
    private fun copyToInternalStorage(source: File, dest: File): Boolean {
        return try {
            val startTime = System.currentTimeMillis()
            var totalCopied = 0L
            var lastLogTime = startTime

            source.inputStream().use { input ->
                dest.outputStream().use { output ->
                    val buffer = ByteArray(COPY_BUFFER_SIZE)
                    var bytesRead: Int

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        totalCopied += bytesRead

                        // Log progress every 2 seconds
                        val now = System.currentTimeMillis()
                        if (now - lastLogTime > 2000) {
                            val pct = totalCopied * 100 / source.length()
                            val speedMb = (totalCopied / 1024 / 1024) / ((now - startTime) / 1000.0)
                            tag().i("Copy progress: ${totalCopied / 1024 / 1024}/${source.length() / 1024 / 1024}MB ($pct%) — ${String.format("%.1f", speedMb)}MB/s")
                            lastLogTime = now
                        }
                    }
                }
            }

            val duration = System.currentTimeMillis() - startTime

            // Verify
            val sizeOk = dest.exists() && dest.length() > 500_000_000L
            if (!sizeOk) {
                tag().e("Copy verification failed: size=${dest.length()}")
                dest.delete()
                return false
            }

            tag().i("Copy complete: ${dest.length() / 1024 / 1024}MB in ${duration / 1000}s (${String.format("%.1f", dest.length() / 1024.0 / 1024.0 / (duration / 1000.0))}MB/s)")
            true
        } catch (e: Exception) {
            tag().e(e, "Failed to copy model to internal storage")
            dest.delete()
            false
        }
    }

    /**
     * Check if model is already in internal storage and valid.
     * Fast check — no external storage access needed.
     */
    fun isModelInInternal(modelConfig: ModelConfig): Boolean {
        val internal = File(context.filesDir, modelConfig.filename)
        return internal.exists() && internal.length() > 500_000_000L
    }

    /**
     * Copy model from /sdcard/Download/ to internal storage with progress reporting.
     * This is the primary flow for release builds where adb push is not available.
     *
     * Flow:
     *   1. Already in internal → instant Complete
     *   2. Found in /sdcard/Download/ → copy with progress
     *   3. Not found → Error with instructions
     */
    fun copyFromExternalWithProgress(modelConfig: ModelConfig): Flow<DownloadProgress> = flow {
        val internal = File(context.filesDir, modelConfig.filename)

        // Fast path — already in internal
        if (internal.exists() && internal.length() > 500_000_000L) {
            tag().i("Model already in internal: ${internal.length() / 1024 / 1024}MB")
            emit(DownloadProgress.Complete(internal.absolutePath))
            return@flow
        }

        // Delete incomplete internal file
        if (internal.exists()) {
            tag().w("Internal file incomplete (${internal.length() / 1024 / 1024}MB), deleting")
            internal.delete()
        }

        // Check external source
        val external = File(downloadDir, modelConfig.filename)
        if (!external.exists() || !external.canRead() || external.length() < 500_000_000L) {
            val msg = "Model not found. Place ${modelConfig.filename} in /sdcard/Download/ folder (${(external.length() / 1024 / 1024)}MB found, need >500MB)"
            tag().e(msg)
            emit(DownloadProgress.Error(msg))
            return@flow
        }

        // Check disk space
        val sourceSize = external.length()
        val freeSpace = internal.parentFile?.usableSpace ?: 0L
        if (freeSpace < sourceSize * 2) {
            val msg = "Not enough storage. Need ${sourceSize / 1024 / 1024}MB, have ${freeSpace / 1024 / 1024}MB free"
            tag().e(msg)
            emit(DownloadProgress.Error(msg))
            return@flow
        }

        // Copy with progress
        emit(DownloadProgress.Started)
        tag().i("Copying ${external.absolutePath} → ${internal.absolutePath} (${sourceSize / 1024 / 1024}MB)")

        val startTime = System.currentTimeMillis()
        var totalCopied = 0L

        try {
            external.inputStream().use { input ->
                internal.outputStream().use { output ->
                    val buffer = ByteArray(COPY_BUFFER_SIZE)
                    var bytesRead: Int

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        totalCopied += bytesRead

                        val pct = (totalCopied * 100 / sourceSize).toInt()
                        emit(DownloadProgress.Progress(pct, totalCopied, sourceSize))
                    }
                }
            }

            // Verify
            if (!internal.exists() || internal.length() < 500_000_000L) {
                tag().e("Copy verification failed: size=${internal.length()}")
                internal.delete()
                emit(DownloadProgress.Error("Copy verification failed. File may be corrupt."))
                return@flow
            }

            val duration = (System.currentTimeMillis() - startTime) / 1000
            tag().i("Copy complete: ${internal.length() / 1024 / 1024}MB in ${duration}s")
            emit(DownloadProgress.Complete(internal.absolutePath))
        } catch (e: Exception) {
            tag().e(e, "Copy failed at ${totalCopied / 1024 / 1024}MB")
            internal.delete()
            emit(DownloadProgress.Error("Copy failed: ${e.message}"))
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Download model with progress reporting.
     * Downloads to /storage/emulated/0/Download/ then copies to internal.
     */
    fun downloadModel(modelConfig: ModelConfig): Flow<DownloadProgress> = flow {
        try {
            emit(DownloadProgress.Started)

            val externalFile = File(downloadDir, modelConfig.filename)

            // Already exists in Download folder
            if (externalFile.exists() && externalFile.length() > 500_000_000L) {
                tag().i("Model already at ${externalFile.absolutePath} (${externalFile.length() / 1024 / 1024}MB)")
                emit(DownloadProgress.Complete(externalFile.absolutePath))
                return@flow
            }

            // Download from HuggingFace
            tag().i("Downloading from ${modelConfig.url} → ${externalFile.absolutePath}")

            val url = URL(modelConfig.url)
            val connection = url.openConnection() as HttpURLConnection

            val token = getToken()
            if (!token.isNullOrBlank()) {
                tag().d("Using HF Token for authentication")
                connection.setRequestProperty("Authorization", "Bearer $token")
            }

            connection.connect()

            val responseCode = connection.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw Exception("Server returned HTTP $responseCode: ${connection.responseMessage}")
            }

            val fileLength = connection.contentLength

            connection.inputStream.use { input ->
                FileOutputStream(externalFile).use { output ->
                    val buffer = ByteArray(8192)
                    var total: Long = 0
                    var count: Int

                    while (input.read(buffer).also { count = it } != -1) {
                        total += count
                        output.write(buffer, 0, count)

                        if (fileLength > 0) {
                            val progress = (total * 100 / fileLength).toInt()
                            emit(DownloadProgress.Progress(progress, total, fileLength.toLong()))
                        }
                    }
                }
            }

            tag().i("Download complete: ${externalFile.absolutePath} (${externalFile.length() / 1024 / 1024}MB)")
            emit(DownloadProgress.Complete(externalFile.absolutePath))

        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Error downloading model: ${e.message}")
            emit(DownloadProgress.Error(e.message ?: "Unknown error"))
        }
    }.flowOn(Dispatchers.IO)
}

sealed class DownloadProgress {
    object Started : DownloadProgress()
    data class Progress(val percentage: Int, val downloaded: Long, val total: Long) : DownloadProgress()
    data class Complete(val filePath: String) : DownloadProgress()
    data class Error(val message: String) : DownloadProgress()
}
