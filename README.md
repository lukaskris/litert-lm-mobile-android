# Model Garden Android LiteRT

A premium **multimodal** on-device LLM chat application for Android, powered by **Google LiteRT-LM**. Features **Gemma 4 E2B** as the primary model with support for **text, image, and audio** inputs, running entirely on-device with **NPU/GPU/CPU** acceleration.

## Gemma 4 E2B

[Gemma 4](https://ai.google.dev/gemma/docs/core) is Google's latest family of open models, built from the same research as Gemini.

*   **E2B**: Effective ~2 Billion parameters — ideal for on-device deployment
*   **Multimodal**: Understands **text + images + audio** natively
*   **Architecture**: Per-Layer Embeddings (PLE), Shared KV Cache, variable aspect ratio vision encoder
*   **Context**: Up to 32K tokens
*   **License**: Apache 2.0 (fully open)
*   **Model**: [`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)

## Features

*   **Gemma 4 E2B** as the default on-device model (2.58 GB)
*   **Multimodal Input**: Attach images from gallery and record audio directly in-app
*   **NPU → GPU → CPU** backend fallback with automatic SoC detection (Qualcomm, MediaTek, Exynos, Unisoc)
*   **ADB Push Support**: Push the model from PC — no in-app download needed for large files
*   **Multi-Model Support**: Switch between Gemma 4, Gemma 3n, Qwen 3, Gemma 3 1B
*   **Real-time Benchmarks**: TTFT, tokens/sec, token count displayed live in the header
*   **Modern Premium UI**: Deep Blue & Soft Gray aesthetic with streaming responses

## Benchmarks (Samsung S25 Ultra - Snapdragon 8 Elite)

| Metric | Gemma 4 E2B | Gemma 3n | Qwen 3 0.6B |
| :--- | :--- | :--- | :--- |
| **Model Size** | 2.58 GB | ~1.5 GB | ~0.5 GB |
| **Modalities** | Text + Image + Audio | Text | Text |
| **Context Length** | 32K | 8K | 4K |
| **Backend** | NPU/GPU/CPU | GPU/CPU | GPU/CPU |

> **Note**: Performance benchmarks from [the model card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) show excellent throughput on Android with GPU acceleration via XNNPack and ML Drift.

## Setup & Installation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carrycooldude/ModelGarden-QNN-LiteRT/blob/main/google_colab/LiteRT_Gemma4_NPU_AOT_Compilation.ipynb)

### Prerequisites
*   Android Studio Ladybug (or newer)
*   Samsung S25 Ultra (or any Android 10+ device with ARM64)
*   ~3GB free storage for the model

### 1. Clone the Repository
```bash
git clone https://github.com/carrycooldude/ModelGarden-QNN-LiteRT.git
cd ModelGarden-QNN-LiteRT
```

### 2. Build & Install the APK
```bash
./gradlew installDebug
```

### 3. Push the Model via ADB (Recommended)

Download the model on your PC from HuggingFace:
```bash
# Download the model (2.58 GB) — use curl.exe on Windows, curl on Linux/Mac
curl -L -o gemma-4-E2B-it.litertlm "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm"

# Push to phone
adb push gemma-4-E2B-it.litertlm /sdcard/Download/
```

Or use the Hugging Face CLI:
```bash
pip install huggingface_hub
huggingface-cli download litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --local-dir .
adb push gemma-4-E2B-it.litertlm /sdcard/Download/
```

The app will automatically detect the model in `/sdcard/Download/` on launch.

### 4. Usage
1.  **Launch the App**: The app detects the Gemma 4 model and initializes (NPU → GPU → CPU)
2.  **Chat**: Type messages for text-only conversations
3.  **Image Input**: Tap the gallery button to attach an image, then ask about it
4.  **Audio Input**: Tap the mic button to record audio, tap again to stop
5.  **Switch Models**: Settings → Select Model to try other models
6.  **Benchmarks**: Watch real-time TTFT and tokens/sec in the header

## Improving Load Time & Performance

### 1. Push Models to App-Specific Storage

Pushing directly to the app-specific external directory avoids the file-copy step the app performs when models are found in `/sdcard/Download/`. This copy (2.58 GB for Gemma 4) can take 30-60 seconds on internal storage.

```bash
# Fastest path — no copy needed on launch
adb push gemma-4-E2B-it.litertlm /sdcard/Android/data/com.example.qnn_litertlm_gemma/files/
```

### 2. AOT Compile for NPU (Qualcomm Devices)

On Snapdragon devices, use the [Colab notebook](./google_colab/LiteRT_Gemma4_NPU_AOT_Compilation.ipynb) to Ahead-Of-Time compile the model for the NPU. Without AOT, the QNN delegate must JIT-compile at load time, which adds 30-120 seconds to initialization and may fail entirely.

AOT-compiled models:
- Skip JIT compilation on device
- Initialize in seconds instead of minutes
- Achieve the best possible inference throughput on Snapdragon NPU

### 3. Enable R8/ProGuard for Release Builds

The current `build.gradle.kts` has minification disabled. Enabling it shrinks the APK and removes unused code from LiteRT delegates:

```kotlin
// app/build.gradle.kts
buildTypes {
    release {
        isMinifyEnabled = true
        isShrinkResources = true
        proguardFiles(
            getDefaultProguardFile("proguard-android-optimize.txt"),
            "proguard-rules.pro"
        )
    }
}
```

This reduces APK size by removing unused delegate code and can improve cold start time.

### 4. Select the Right Backend for Your SoC

The app auto-detects your SoC vendor and skips backends that won't work:

| SoC | Backend Chain | Notes |
| :--- | :--- | :--- |
| **Qualcomm Snapdragon** | NPU → GPU → CPU | NPU requires AOT-compiled model for best results |
| **MediaTek Dimensity/Helio** | GPU → CPU | NPU skipped — QNN delegate is Qualcomm-only |
| **Samsung Exynos** | GPU → CPU | NPU skipped |
| **Unisoc** | GPU → CPU | NPU skipped |

You can also manually override the backend in Settings → Select Model. For fastest text-only inference on Qualcomm, use NPU. For multimodal (image + text), GPU is used for vision encoding regardless of the text backend.

### 5. Reduce Image Resolution for Faster Vision Encoding

The app automatically downscales attached images to a maximum of **256px** on the longest edge before sending to the model. This keeps the vision patch count low and avoids GPU lockups on Mali GPUs (MediaTek) that can freeze the UI compositor for 15+ seconds with larger images.

If you are modifying the code, do not increase this beyond 512px on non-Qualcomm devices. On Snapdragon with NPU, you may safely raise it to 512px for better visual detail at the cost of slightly longer encoding time.

### 6. Keep the Model in Internal Storage

The LiteRT-LM engine uses `mmap()` to load model weights. Files on internal storage (`context.filesDir`) are accessible without SELinux restrictions and can be memory-mapped directly. If the model is on external storage, the app copies it to internal first — this is a one-time cost, but avoid deleting it from internal storage.

### 7. Use `cacheDir` for Faster Subsequent Loads

The app passes `context.cacheDir` to the LiteRT-LM `EngineConfig`. The engine caches compiled kernels and intermediate artifacts here, so the second and subsequent initializations are faster than the first. Do not clear the app cache between sessions if you want fast reloads.

### 8. Dedicated Inference Thread Pool

Inference runs on a dedicated thread pool (not `Dispatchers.IO` or `Dispatchers.Default`) with:
- **Pool size**: Half the CPU cores (minimum 1, always leaves 2+ cores free for UI)
- **Thread priority**: Below normal — the OS scheduler prioritizes the UI thread over inference
- **Thread.yield()**: Called per token to periodically yield CPU time back to the UI

This ensures the UI remains responsive even during heavy inference. On an 8-core device, inference uses 4 threads at reduced priority while the main thread and system services keep the remaining cores.

### 9. UI Update Throttling

Streaming tokens are throttled to one UI update every **120ms**. Without throttling, each token triggers a RecyclerView rebind and layout pass, which causes jank — especially on devices where the GPU is shared between inference and rendering.

The app uses payload-based `notifyItemChanged()` with a partial-bind payload, so only the text content is rebound — not the entire message layout.

### 10. Use Smaller Models for Faster TTFT

If low latency is more important than response quality, switch to a smaller model:

| Model | Size | Typical TTFT | Use Case |
| :--- | :--- | :--- | :--- |
| **Qwen 3 0.6B** | ~0.5 GB | Fastest | Quick answers, low resource |
| **Gemma 3 1B** | ~1 GB | Fast | General purpose text |
| **Gemma 3n E2B** | ~1.5 GB | Moderate | Text conversations |
| **Gemma 4 E2B** | 2.58 GB | Slowest | Multimodal (text + image + audio) |

## Notes on Hardware Acceleration

*   The app tries **NPU** first (Qualcomm Hexagon on Snapdragon), then falls back to **GPU** (OpenCL/ML Drift), then **CPU** (XNNPack)
*   NPU requires AOT compilation for reliable results — use the Colab notebook above
*   On non-Qualcomm devices, NPU is skipped entirely to avoid wasting 30-120s on a guaranteed failure
*   Vision encoding always uses **GPU** — CPU vision encoding is too slow for production use
*   `cacheDir` is used for faster model reloading on subsequent launches

## References

*   [Gemma 4 Overview](https://ai.google.dev/gemma/docs/core)
*   [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
*   [HuggingFace Gemma 4 Collection](https://huggingface.co/collections/google/gemma-4)
*   [HuggingFace Gemma 4 Blog](https://huggingface.co/blog/gemma4)
*   [LiteRT-LM Android Guide](https://ai.google.dev/edge/litert-lm/android)
*   [LiteRT-LM Model (gemma-4-E2B-it)](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)

## Demo

https://github.com/user-attachments/assets/4c3c494e-a119-45d5-9726-4e43b2351ed9

## License
Apache 2.0
