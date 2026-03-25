# 🔧 MiniGPT — Tech Stack

Every technology choice in this project was made to serve a single constraint: **a useful language model under 16MB that runs without a server.**

---

## Training Stack

### PyTorch 2.2+
The de-facto standard for research-to-production ML. Used for model definition, training, and checkpoint export. Key features used:

- `nn.Module` — clean model definition with weight tying
- `torch.onnx.export` — deterministic graph export with dynamic axes
- `torch.cuda.amp.GradScaler` — automatic mixed precision (FP16 training)
- `torch.optim.AdamW` — Adam with decoupled weight decay (crucial for transformer training)

### HuggingFace Datasets
Provides streaming access to TinyStories without requiring a full 2GB download for tokenizer training. The streaming API lets us sample 20,000 documents for BPE training in seconds.

### HuggingFace Tokenizers (`tokenizers` library)
Fast Rust-based BPE tokenizer training. `ByteLevelBPETokenizer` operates on raw bytes, so it handles any Unicode text without unknown characters — every possible byte sequence maps to something in the vocabulary.

Why custom BPE over GPT-2's tokenizer:
- GPT-2 vocab: 50,257 tokens → 25MB embedding layer alone
- Our vocab: 5,000 tokens → 2.5MB embedding layer

### HuggingFace Transformers
Used only for `PreTrainedTokenizerFast` — a thin Python wrapper that loads our trained `vocab.json`/`merges.txt` and provides the standard `tokenizer(text)` interface used by PyTorch DataLoaders.

---

## Quantization

### ONNX Runtime Quantization (`onnxruntime.quantization`)
Post-training **dynamic INT8 quantization** applied after ONNX export.

- Quantizes weight matrices from FP32 (4 bytes/param) to INT8 (1 byte/param)
- Activations are quantized dynamically at runtime based on observed ranges
- Uses `QuantType.QUInt8` — unsigned INT8, compatible with all ONNX Runtime backends including WASM

Why not static quantization?
Static quantization is more accurate but requires a calibration dataset and is harder to automate. For a 1.4M param model trained on a narrow domain, dynamic INT8 loses <1% perplexity in practice.

---

## Export Format

### ONNX (Open Neural Network Exchange)
ONNX is the **universal model format** — the one export that works everywhere:

| Platform | Runtime | Backend |
|---|---|---|
| Web browser | ONNX Runtime Web | WASM / WebGPU |
| iOS | CoreML (via conversion) | CPU + Neural Engine |
| Android | TFLite (via conversion) | CPU + NNAPI + GPU |
| Desktop | ONNX Runtime | CPU / CUDA / DirectML |
| Edge devices | ONNX Runtime | ARM Cortex / NEON |

ONNX opset 14 is required for correct handling of the dynamic-shape KV-cache concatenation operations.

---

## Browser Runtime

### ONNX Runtime Web (`ort` v1.17+)
The browser build of ONNX Runtime, distributed as a JavaScript library + WebAssembly binary.

**Execution backends (in priority order):**
1. **WebGPU** — GPU acceleration, ~3× faster for larger models. For <16MB models, GPU transfer overhead may outweigh the benefit — benchmarks recommended.
2. **WASM SIMD** — CPU with Single Instruction Multiple Data. Processes 4–8 float operations per CPU clock. **This is our default** — reliably fast on all platforms.
3. **WASM** — Fallback for browsers without SIMD support (pre-2021 hardware).

**ONNX Runtime Web WASM SIMD performance on target hardware:**
- Modern laptop (M2/i7): 100–150 tok/s
- Mid-range phone (A14/Snapdragon 8): 60–100 tok/s
- Budget phone (Snapdragon 680): 30–60 tok/s

### Web Workers API
The inference loop runs in a `Worker` thread, not the main thread. This is **non-negotiable** for a responsive UI — JavaScript is single-threaded, and ONNX inference blocks for 5–15ms per token. Without a Worker, the browser would stutter or freeze during generation.

The Worker communicates via `postMessage` — individual tokens are streamed back to the main thread as they're generated, enabling real-time text streaming identical to ChatGPT's streaming UX.

### Transformers.js (tokenizer, optional)
`@xenova/transformers` provides a browser-compatible BPE tokenizer that can load our custom `tokenizer.json` via `AutoTokenizer.from_pretrained()`. The project includes a `SimpleBPETokenizer` fallback for offline use.

---

## Mobile Runtimes

### CoreML (iOS 16+ / macOS 13+)
Apple's on-device ML framework. The `.mlpackage` format runs on:
- **Apple Neural Engine (ANE)** — dedicated ML accelerator present in A12+ and M1+ chips. ~10× more energy-efficient than CPU for matrix operations.
- CPU fallback for older devices

Conversion path: PyTorch → ONNX → CoreML via `coremltools`. The `ct.precision.FLOAT16` flag keeps the model at ~2.7MB on disk with minimal quality loss.

### TensorFlow Lite (Android)
Google's lightweight inference framework for mobile and edge. The `.tflite` format supports:
- **NNAPI** — Android Neural Networks API, routes to GPU/DSP/NPU on supported chipsets
- **GPU Delegate** — OpenGL ES or Vulkan compute shaders
- CPU fallback with ARM NEON SIMD

Conversion path: PyTorch → ONNX → TensorFlow SavedModel → TFLite via `onnx-tf` + `tf.lite.TFLiteConverter`.

---

## Web UI

### Vanilla HTML/CSS/JS (no framework)
The entire browser app is a single `index.html` file. No React, no Vue, no build step. This is intentional:

- Zero bundle overhead (the model itself is the payload)
- Deployable to GitHub Pages, Netlify, S3, or any static host
- No Node.js required for deployment
- Loads in <50ms before the model download even starts

### Google Fonts: JetBrains Mono + Syne
- **JetBrains Mono** — monospace for all code/output text. Excellent readability for streaming token output; fixed-width prevents layout shift as text appears.
- **Syne** — geometric sans-serif for UI text. Distinctive and technical-feeling without being cold.

---

## Size Budget Accounting

```
Component                        Size
──────────────────────────────────────────────────────
mini_gpt_quant.onnx (INT8)       ~1.42 MB
mini_gpt_tokenizer/vocab.json    ~245 KB
mini_gpt_tokenizer/merges.txt    ~180 KB
index.html (self-contained)      ~28 KB
inference.worker.js              ~8 KB
ort.min.js (CDN, not bundled)    —
ort-wasm-simd.wasm (CDN)         —
──────────────────────────────────────────────────────
Total local assets               ~1.88 MB  ✓ << 16MB
```

ONNX Runtime Web files (`.wasm`, `.js`) are loaded from CDN and cached by the browser. They are not bundled with the project.

---

## Security & Privacy

- **Zero server-side components** — inference runs entirely on the user's device
- **No telemetry** — no analytics, no error reporting, no usage tracking
- **No API keys** — nothing to steal, nothing to rotate
- **Offline capable** — after first load, works without internet (ONNX Runtime WASM files can be self-hosted if needed)
- **Content Security Policy** compatible — no inline eval required by ONNX Runtime WASM

---

## Deployment Options

| Platform | Method | Notes |
|---|---|---|
| GitHub Pages | Push `web/` to main branch | Free, HTTPS, CDN |
| Netlify | Drag `web/` folder | Free tier, instant deploy |
| Vercel | `vercel --cwd web` | Free tier |
| Cloudflare Pages | Git integration | Free, global CDN |
| Self-hosted | Any static file server | `nginx`, `caddy`, `http-server` |
| Mobile PWA | Add `manifest.json` | Installable, offline-capable |

**Note on CORS/MIME types:** All platforms above serve `.onnx` files as `application/octet-stream` correctly. If self-hosting with nginx, add:
```nginx
types { application/octet-stream onnx; }
```
