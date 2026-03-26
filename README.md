# ⚡ MiniGPT — Sub-16MB On-Device Language Model

> A 1.4M parameter GPT-style transformer that runs **entirely in the browser** via WebAssembly, or **natively on iOS/Android** — no server, no API keys, no data leaves your device.

---

## 🚀 5-Minute Setup

```bash
git clone https://github.com/david-spies/minigpt.git && cd minigpt
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

## 📊 At a Glance

| Property | Value |
|---|---|
| Parameters | ~1.46 Million |
| Model size (INT8) | **~2.0 MB** |
| Model size (FP32) | ~8.0 MB |
| Inference speed | 60–120+ tokens/sec |
| Architecture | 4 layers × 128 dim × 4 heads |
| Vocabulary | 5,000 tokens (Byte-Level BPE) |
| Context window | 256 tokens |
| Training data | TinyStories (Eldan & Li, 2023) |
| Runtime (web) | ONNX Runtime Web — WASM SIMD |
| Runtime (iOS) | CoreML / Apple Neural Engine |
| Runtime (Android) | TensorFlow Lite / NNAPI |

---

## 📁 Project Structure

```
MiniGPT/                            ← project root (case-sensitive — capital M and G)
│
├── main.py                         ← CLI: train / export / generate / benchmark / info
├── conftest.py                     ← shared pytest fixtures
├── setup.py                        ← pip install -e . support
├── pyproject.toml                  ← build config, ruff, mypy, pytest settings
├── requirements.txt                ← pip install -r requirements.txt
├── Makefile                        ← make train / make export / make serve / make test
├── fix_install.sh                  ← one-shot script to clear stale installs
├── README.md                       ← this file
├── .gitignore
│
├── .github/workflows/
│   └── ci.yml                      ← GitHub Actions CI pipeline
│
├── minigpt_core/                   ← ⭐ main Python package
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py               ← MiniGPTConfig dataclass
│   │   └── model.py                ← MiniGPT + CausalSelfAttention with KV-cache
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── tokenizer.py            ← BPE training + load_tokenizer
│   │   └── dataset.py              ← TextBlockDataset + DataLoader
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py              ← training loop, LR schedule, checkpointing
│   ├── export/
│   │   ├── __init__.py
│   │   └── onnx_export.py          ← ONNX export + INT8 quantization
│   └── inference/
│       ├── __init__.py
│       └── inference.py            ← PyTorch + ONNX Runtime inference
│
├── tests/
│   ├── test_minigpt.py
│   ├── test_tokenizer.py
│   └── test_training.py
│
├── scripts/
│   ├── serve.py                    ← local dev server (fixes .onnx MIME types)
│   ├── benchmark.py                ← inference benchmark suite
│   └── export_mobile.py            ← CoreML + TFLite export
│
├── docs/
│   ├── QUICKSTART.md               ← step-by-step setup with troubleshooting
│   ├── DEVELOPMENT.md              ← architecture deep-dive
│   └── TECHSTACK.md                ← technology choices and rationale
│
├── web/
│   ├── index.html                  ← production browser app
│   ├── demo.html                   ← standalone demo (no model required)
│   └── workers/
│       └── inference.worker.js     ← Web Worker: off-thread ONNX inference
│
└── mobile/
    ├── ios/
    │   ├── MiniGPTInference.swift
    │   └── ContentView.swift
    └── android/
        ├── MiniGPTInference.kt
        ├── MainActivity.kt
        └── build.gradle
```

**Generated directories** (not in a fresh clone — created by running commands):

| Directory | Created by |
|---|---|
| `checkpoints/` | `python main.py train` |
| `mini_gpt_tokenizer/` | `python main.py train` |
| `web/assets/` | `python main.py export` |
| `venv/` | `python -m venv venv` |

---

## 🚀 Quick Start

```bash
# 1. Clone and enter — note the exact capitalisation
git clone https://github.com/david-spies/MiniGPT.git
cd MiniGPT

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install
pip install -e .
pip install datasets tokenizers transformers tqdm

# 4. Verify
python main.py info

# 5. Train (quick run)
python main.py train --n-samples 50000

# 6. Export
python main.py export

# 7. Serve browser app
python scripts/serve.py         # → http://localhost:8080
```

See **[docs/QUICKSTART.md](docs/QUICKSTART.md)** for the full walkthrough including every known error and its fix.

---

## 🛠️ Commands

```bash
python main.py info                             # verify config + size estimates
python main.py train                            # full training run
python main.py train --n-samples 50000         # quick training run (~20 min CPU)
python main.py train --amp                      # GPU training with mixed precision
python main.py export                           # ONNX + INT8 quantize → web/assets/
python main.py generate --prompt "Once upon"   # generate text from checkpoint
python main.py benchmark                        # benchmark inference speed
python scripts/serve.py                         # serve browser app on :8080
python scripts/benchmark.py --all              # full benchmark suite
make test                                       # run test suite
bash fix_install.sh                             # nuclear clean reinstall
```

---

## ⚡ Performance

| Platform | Speed | Notes |
|---|---|---|
| Browser — Chrome/Edge (WASM SIMD) | 60–120 tok/s | Modern laptop |
| Browser — Firefox | 50–100 tok/s | Enable `wasm_simd` in `about:config` |
| Browser — Safari | 40–90 tok/s | Efficient on Apple Silicon |
| iOS (Apple Neural Engine) | 80–150 tok/s | A14+ chip |
| Android (Snapdragon 8) | 60–100 tok/s | NNAPI accelerated |

---

## ❗ Common Errors — Quick Reference

| Error message | One-line fix |
|---|---|
| `cannot import name 'MiniGPT' from 'src.model'` | `sed -i 's/from src\./from minigpt_core\./g' main.py` |
| `cannot import name 'MiniGPT' from 'minigpt_core.model' (unknown location)` | `__init__.py` files missing — run the block in QUICKSTART §2 |
| `minigpt_core.model.__file__: None` | Same as above |
| `ValueError: Couldn't instantiate the backend tokenizer` | `rm -rf mini_gpt_tokenizer/ && python main.py train ...` |
| `Fatal Python error: PyGILState_Release` / core dump | Side effect of tokenizer error — fix tokenizer first |
| `No such file or directory: minigpt_core` | Wrong directory — check `pwd`; use `MiniGPT` not `minigpt` |
| `BackendUnavailable` during `pip install -e .` | `pip install --upgrade setuptools` then retry |
| Browser: `model not found` | Run `python main.py export` first |
| Browser: worker error / CORS | Use `python scripts/serve.py`, not `file://` |

Full detail for every error with terminal commands: **[docs/QUICKSTART.md](docs/QUICKSTART.md)**

---

## 📜 License

MIT - Use it, fork it, build with it.

## 🔗 References

- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [GPT Architecture](https://openai.com/blog/language-unsupervised) — Radford et al., 2018
- [Byte-Level BPE](https://arxiv.org/abs/1508.07909) — Sennrich et al., 2016
