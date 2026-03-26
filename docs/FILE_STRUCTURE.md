# MiniGPT — File Structure Reference

> **Current as of latest build.** Use this to verify your local directory matches exactly before running any commands.
> 
> ⚠️ **Important:** The internal Python package is named `minigpt_core/` — NOT `src/`. If you have a `src/` directory from an earlier build, delete it: `rm -rf src/`

---

```
minigpt/                                   ← project root (your working directory)
│
├── main.py                                ← CLI entry point  (train/export/generate/benchmark/info)
├── conftest.py                            ← shared pytest fixtures
├── setup.py                              ← legacy pip install fallback
├── pyproject.toml                         ← build config + tool settings (ruff, mypy, pytest)
├── requirements.txt                       ← pip install -r requirements.txt
├── Makefile                               ← make train / make export / make serve / make test
├── README.md                              ← project overview + quick-start summary
├── .gitignore                             ← excludes checkpoints, *.onnx (fp32), venv, __pycache__
│
├── .github/
│   └── workflows/
│       └── ci.yml                         ← GitHub Actions: test matrix, size-gate, lint, web-validate
│
├── minigpt_core/                          ← ⭐ main Python package (was src/ — renamed to avoid PyPI conflict)
│   ├── __init__.py                        ← sets __version__ = "1.0.0"  (no re-exports)
│   │
│   ├── model/
│   │   ├── __init__.py                    ← exports MiniGPT, MiniGPTConfig, DEFAULT_CONFIG, MEDIUM_CONFIG
│   │   ├── config.py                      ← MiniGPTConfig dataclass  (vocab, layers, dims, training params)
│   │   └── model.py                       ← MiniGPT model + CausalSelfAttention with KV-cache
│   │
│   ├── tokenizer/
│   │   ├── __init__.py                    ← exports prepare_tokenizer, build_dataloaders, etc.
│   │   ├── tokenizer.py                   ← ByteLevelBPE training + load_tokenizer + load_tinystories
│   │   └── dataset.py                     ← TextBlockDataset (flat-stream blocks) + DataLoader builder
│   │
│   ├── training/
│   │   ├── __init__.py                    ← exports Trainer
│   │   └── trainer.py                     ← training loop, cosine LR schedule, AMP, checkpointing
│   │
│   ├── export/
│   │   ├── __init__.py                    ← exports export_onnx, quantize_onnx, export_pipeline
│   │   └── onnx_export.py                 ← ONNX export with KV-cache dynamic axes + INT8 quantization
│   │
│   └── inference/
│       ├── __init__.py                    ← exports generate, benchmark_pytorch, OnnxInferenceSession
│       └── inference.py                   ← PyTorch + ONNX Runtime inference + benchmarking utilities
│
├── tests/
│   ├── test_minigpt.py                    ← model shape, KV-cache consistency, weight tying, size gate
│   ├── test_tokenizer.py                  ← TextBlockDataset, DataLoader, BPE training (skipif guarded)
│   └── test_training.py                   ← Trainer end-to-end, LR schedule, checkpoint save/load
│
├── scripts/
│   ├── serve.py                           ← local dev server  (fixes .onnx + .wasm MIME types)
│   ├── benchmark.py                       ← full benchmark suite: PyTorch vs ONNX, KV-cache speedup
│   └── export_mobile.py                   ← CoreML (.mlpackage) + TFLite (.tflite) export pipeline
│
├── docs/
│   ├── README.md                          ← full project README (specs, pipeline, deploy options)
│   ├── QUICKSTART.md                      ← step-by-step: install → train → export → browser
│   ├── DEVELOPMENT.md                     ← architecture deep-dive, KV-cache internals, extending
│   └── TECHSTACK.md                       ← technology choices and rationale
│
├── web/
│   ├── index.html                         ← production browser app (dark UI, streaming output)
│   ├── demo.html                          ← standalone demo (no model needed, simulates pipeline)
│   └── workers/
│       └── inference.worker.js            ← Web Worker: ONNX inference off main thread + KV-cache
│
│   [generated after running: python main.py export]
│   └── assets/                            ← created by export step, NOT committed to git (except quant)
│       ├── mini_gpt_fp32.onnx             ← full precision ~5.5MB  (debug only)
│       ├── mini_gpt_quant.onnx            ← INT8 quantized ~1.5MB  (production ✓)
│       └── mini_gpt_tokenizer/            ← copied from project root by export script
│           ├── vocab.json
│           ├── merges.txt
│           └── special_tokens_map.json
│
├── mobile/
│   ├── ios/
│   │   ├── MiniGPTInference.swift         ← CoreML inference engine + KV-cache (Swift)
│   │   └── ContentView.swift              ← SwiftUI interface
│   └── android/
│       ├── MiniGPTInference.kt            ← TFLite inference engine, Flow-based streaming (Kotlin)
│       ├── MainActivity.kt                ← Jetpack Compose UI
│       └── build.gradle                   ← TFLite + GPU delegate + Compose dependencies
│
│   [generated after running: python scripts/export_mobile.py]
│   ├── ios/
│   │   └── mini_gpt.mlpackage/            ← CoreML package for Xcode (drag into project)
│   └── android/
│       └── app/src/main/assets/
│           └── mini_gpt.tflite            ← TFLite model for Android assets/
│
│   [generated after running: python main.py train]
├── checkpoints/                           ← created by training, NOT committed to git
│   ├── mini_gpt_best.pt                   ← best validation loss checkpoint
│   └── mini_gpt_final.pt                  ← final epoch checkpoint
│
│   [generated after running: python main.py train (tokenizer step)]
└── mini_gpt_tokenizer/                    ← created by training, commit this to git
    ├── vocab.json                         ← 5000-token BPE vocabulary
    ├── merges.txt                         ← BPE merge rules
    └── special_tokens_map.json            ← <s>, </s>, <pad>, <unk> mappings
```

---

## Quick Counts

| Category | Files |
|---|---|
| Python package (`minigpt_core/`) | 13 |
| Tests | 3 |
| Scripts | 3 |
| Web app | 3 |
| Mobile (iOS) | 2 |
| Mobile (Android) | 3 |
| Docs | 4 |
| Config / root | 7 |
| **Total tracked files** | **38** |

---

## What Gets Generated (not committed)

These directories **do not exist** in a fresh clone. They are created when you run the corresponding commands:

| Directory / File | Created by |
|---|---|
| `checkpoints/` | `python main.py train` |
| `mini_gpt_tokenizer/` | `python main.py train` |
| `web/assets/` | `python main.py export` |
| `mobile/ios/mini_gpt.mlpackage/` | `python scripts/export_mobile.py --platform ios` |
| `mobile/android/app/src/main/assets/mini_gpt.tflite` | `python scripts/export_mobile.py --platform android` |
| `venv/` | `python -m venv venv` |

---

## Common Setup Issues

**`src/` directory exists alongside `minigpt_core/`**
Delete it — it's a stale artifact from the original build:
```bash
rm -rf src/
```

**`ImportError: cannot import name 'MiniGPT' from 'src.model' (unknown location)`**
The old `src/` package is shadowing the local code. Run:
```bash
rm -rf src/
pip install -e .
python main.py info
```

**`web/assets/` is missing**
You haven't exported the model yet. Run:
```bash
python main.py train --n-samples 50000   # quick test run
python main.py export
```

**Browser shows "model not found"**
The server must be started from the `web/` directory (or with the serve script), not opened directly as a file:
```bash
python scripts/serve.py        # serves web/ on http://localhost:8080
```
