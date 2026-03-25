# ⚡ MiniGPT — Sub-16MB On-Device Language Model

> A 1.4M parameter GPT-style transformer that runs **entirely in the browser** (via WebAssembly) or **natively on iOS/Android** — no server, no API keys, no network calls after initial load.

---

## 📊 At a Glance

| Property | Value |
|---|---|
| Parameters | ~1.42 Million |
| Model size (INT8) | **~1.42 MB** |
| Inference speed | 60–120+ tokens/sec |
| Architecture | 4L × 128d × 4H Transformer |
| Vocabulary | 5,000 (Byte-Level BPE) |
| Runtime (web) | ONNX Runtime Web (WASM SIMD) |
| Runtime (iOS) | CoreML / Apple Neural Engine |
| Runtime (Android) | TensorFlow Lite / NNAPI |

---

## 🚀 5-Minute Setup

```bash
git clone https://github.com/your-username/minigpt.git && cd minigpt
pip install -r requirements.txt
python main.py info                        # Verify config
python main.py train --n-samples 50000    # Quick train
python main.py export                      # → web/assets/mini_gpt_quant.onnx
npx http-server web/ -p 8080              # Open http://localhost:8080
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Full setup walkthrough |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Architecture, optimization, extending |
| [docs/TECHSTACK.md](docs/TECHSTACK.md) | Technology choices and rationale |
| [docs/README.md](docs/README.md) | Full project README |

---

## 🗂 Project Structure

```
minigpt/
├── main.py                    # CLI: train / export / generate / benchmark
├── requirements.txt
├── src/model/                 # MiniGPT architecture + config
├── src/tokenizer/             # BPE tokenizer + dataset pipeline
├── src/training/              # Training loop + LR schedule
├── src/export/                # ONNX export + INT8 quantization
├── src/inference/             # Python + ONNX Runtime inference
├── web/                       # Browser app (single HTML file + Web Worker)
├── mobile/ios/                # Swift CoreML inference + SwiftUI
├── mobile/android/            # Kotlin TFLite inference engine
├── scripts/                   # Mobile export utilities
├── docs/                      # Full documentation
└── tests/                     # Pytest test suite
```

---

## 📜 License

MIT - Use it, fork it, build with it.
