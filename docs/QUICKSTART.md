# MiniGPT — Quickstart & Troubleshooting Guide

This guide covers every step from a fresh machine to a running browser demo, including every error encountered across real installs and the exact commands to resolve them.

---

## Prerequisites

- Python 3.10, 3.11, or 3.12
- pip 23+
- 2 GB free disk space (dataset download)
- Internet connection (first run only)
- A modern browser (Chrome, Firefox, Safari, Edge)

---

## Part 1 — Setup

### Step 1 — Get into the project directory

The project folder is named `MiniGPT` with a capital M and capital G. This matters on Linux and macOS where paths are case-sensitive.

```bash
cd ~/MiniGPT
```

Confirm you are in the right place before running any other command:

```bash
pwd
# Should print: /home/YOUR_USERNAME/MiniGPT
ls
# Should show: main.py  minigpt_core/  web/  tests/  ...
```

If `ls` shows nothing relevant, you are in the wrong directory. Do not proceed until `pwd` ends with `/MiniGPT`.

---

### Step 2 — Verify all `__init__.py` files exist

This is the most common cause of import errors. Each subdirectory of `minigpt_core/` must contain an `__init__.py` file or Python will not treat it as a package.

Run this check:

```bash
find ~/MiniGPT/minigpt_core -name "__init__.py" | sort
```

Expected output — exactly 6 files:

```
/home/YOU/MiniGPT/minigpt_core/__init__.py
/home/YOU/MiniGPT/minigpt_core/export/__init__.py
/home/YOU/MiniGPT/minigpt_core/inference/__init__.py
/home/YOU/MiniGPT/minigpt_core/model/__init__.py
/home/YOU/MiniGPT/minigpt_core/tokenizer/__init__.py
/home/YOU/MiniGPT/minigpt_core/training/__init__.py
```

If any are missing, paste this entire block to create all 6:

```bash
cat > ~/MiniGPT/minigpt_core/__init__.py << 'EOF'
"""
MiniGPT — Sub-16MB Language Model
"""
__version__ = "1.0.0"
EOF

cat > ~/MiniGPT/minigpt_core/model/__init__.py << 'EOF'
from .config import MiniGPTConfig, DEFAULT_CONFIG, MEDIUM_CONFIG
from .model import MiniGPT

__all__ = ["MiniGPT", "MiniGPTConfig", "DEFAULT_CONFIG", "MEDIUM_CONFIG"]
EOF

cat > ~/MiniGPT/minigpt_core/tokenizer/__init__.py << 'EOF'
from .tokenizer import prepare_tokenizer, train_tokenizer, load_tokenizer, load_tinystories
from .dataset import TextBlockDataset, build_dataloaders

__all__ = [
    "prepare_tokenizer",
    "train_tokenizer",
    "load_tokenizer",
    "load_tinystories",
    "TextBlockDataset",
    "build_dataloaders",
]
EOF

cat > ~/MiniGPT/minigpt_core/training/__init__.py << 'EOF'
from .trainer import Trainer

__all__ = ["Trainer"]
EOF

cat > ~/MiniGPT/minigpt_core/export/__init__.py << 'EOF'
from .onnx_export import export_onnx, quantize_onnx, export_pipeline

__all__ = ["export_onnx", "quantize_onnx", "export_pipeline"]
EOF

cat > ~/MiniGPT/minigpt_core/inference/__init__.py << 'EOF'
from .inference import generate, benchmark_pytorch, OnnxInferenceSession, benchmark_onnx

__all__ = ["generate", "benchmark_pytorch", "OnnxInferenceSession", "benchmark_onnx"]
EOF

echo "Done — verifying:"
find ~/MiniGPT/minigpt_core -name "__init__.py" | sort
```

---

### Step 3 — Remove any stale `src/` directory

An earlier version of this project used a folder called `src/` which conflicts with a PyPI package of the same name. If it exists, delete it:

```bash
ls ~/MiniGPT/src 2>/dev/null && rm -rf ~/MiniGPT/src && echo "Removed stale src/" || echo "No src/ found — OK"
```

---

### Step 4 — Patch any old import references

If `main.py` or other files still contain `from src.` imports, rename them all in one command:

```bash
find ~/MiniGPT -name "*.py" \
  -not -path "*/venv/*" \
  -not -path "*/.git/*" \
  | xargs sed -i 's/from src\./from minigpt_core\./g'

echo "Checking for remaining src. imports:"
grep -rn "from src\." ~/MiniGPT --include="*.py" \
  --exclude-path="*/venv/*" || echo "None found ✓"
```

---

### Step 5 — Create and activate a virtual environment

```bash
cd ~/MiniGPT
python -m venv venv
source venv/bin/activate
```

Your prompt should now start with `(venv)`. All subsequent commands assume the venv is active.

---

### Step 6 — Install the project

```bash
pip install -e .
```

If this fails with `BackendUnavailable` or `No module named 'setuptools.backends'`:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

---

### Step 7 — Install training dependencies

```bash
pip install datasets tokenizers transformers tqdm
```

---

### Step 8 — Verify everything works

```bash
python main.py info
```

Expected output:

```
=======================================================
Config: DEFAULT (1.4M params)
  Parameters  : 1,461,504
  FP32 size   : 8.02 MB
  FP16 size   : 4.01 MB
  INT8 size   : 2.00 MB
  Under 16MB  : ✓ (INT8)
=======================================================
Config: MEDIUM (still <16MB)
  Parameters  : 6,838,784
  ...
  Under 16MB  : ✓ (INT8)
```

If you see this output, setup is complete. Proceed to Part 2.

---

## Part 2 — Training

### Step 9 — Train the model

Quick run (50k samples, ~10–20 min on CPU, ~2–3 min on GPU):

```bash
python main.py train --n-samples 50000
```

Full run (all data, better quality):

```bash
python main.py train
```

GPU with mixed precision (fastest):

```bash
python main.py train --amp
```

Training produces:

```
checkpoints/
├── mini_gpt_best.pt      ← saved whenever val loss improves
└── mini_gpt_final.pt     ← saved at end of training

mini_gpt_tokenizer/
├── vocab.json
├── merges.txt
├── tokenizer.json        ← required for reliable reloading
└── special_tokens_map.json
```

Expected loss curve: starts around `6.5–7.0`, comes down to `3.5–4.5` over 5 epochs on 50k samples.

---

## Part 3 — Export

### Step 10 — Export to ONNX

```bash
python main.py export
```

This produces:

```
web/assets/
├── mini_gpt_fp32.onnx    ← ~8 MB full precision (debug)
└── mini_gpt_quant.onnx   ← ~2 MB INT8 (production ✓)
```

---

## Part 4 — Browser App

### Step 11 — Serve locally

```bash
python scripts/serve.py
```

Open **http://localhost:8080** in your browser.

Custom port:

```bash
python scripts/serve.py --port 3000
```

---

## Part 5 — Troubleshooting

Every error encountered across real installs, with the exact terminal commands to resolve each one.

---

### ❌ `cannot import name 'MiniGPT' from 'src.model' (unknown location)`

**Cause:** `main.py` still has old `from src.` imports from an earlier version of the project.

**Fix:**

```bash
sed -i 's/from src\./from minigpt_core\./g' ~/MiniGPT/main.py
python main.py info
```

To patch every Python file at once:

```bash
find ~/MiniGPT -name "*.py" -not -path "*/venv/*" \
  | xargs sed -i 's/from src\./from minigpt_core\./g'
```

---

### ❌ `cannot import name 'MiniGPT' from 'minigpt_core.model' (unknown location)`

**Cause:** One or more `__init__.py` files are missing from the `minigpt_core/` subdirectories. Without them Python sees each folder as an empty namespace package — `__file__` returns `None` and all imports fail.

**Diagnose:**

```bash
find ~/MiniGPT/minigpt_core -name "__init__.py" | sort
python -c "
import minigpt_core.model as m
print('__file__:', m.__file__)
print('contents:', dir(m))
"
```

If `__file__` is `None` or fewer than 6 `__init__.py` files are listed, run the block from Step 2 above to recreate all of them.

---

### ❌ `minigpt_core.model.__file__: None`

Same cause and fix as the error above — missing `__init__.py` files.

---

### ❌ `No such file or directory: '/home/YOU/minigpt/minigpt_core'`

**Cause:** The project folder is `MiniGPT` (capital M, capital G) but the command used `minigpt` (all lowercase).

**Fix:** Check the exact name of your project folder:

```bash
ls ~/
```

Use the exact name shown. If it is `MiniGPT`:

```bash
cd ~/MiniGPT
# All commands from here use relative paths — no more path confusion
```

Never mix `~/minigpt` and `~/MiniGPT` in the same session.

---

### ❌ `ValueError: Couldn't instantiate the backend tokenizer` + core dump

**Full error:**

```
ValueError: Couldn't instantiate the backend tokenizer from one of:
(1) a `tokenizers` library serialization file,
(2) a slow tokenizer instance to convert or
(3) an equivalent slow tokenizer class to instantiate and convert.
You need to have sentencepiece or tiktoken installed...

Fatal Python error: PyGILState_Release: auto-releasing thread-state
Aborted (core dumped)
```

**Cause:** Two separate issues:

1. The tokenizer was trained and saved, but the `tokenizer.json` file was not generated (older version of `tokenizer.py` only saved `vocab.json` and `merges.txt`). The loader fails, throws an exception, and the crash happens because a HuggingFace streaming download thread was still running when Python's garbage collector kicked in.

2. The core dump is a side effect — it goes away once the tokenizer error is fixed.

**Fix:** Delete the incomplete tokenizer directory and retrain with the updated code:

```bash
rm -rf ~/MiniGPT/mini_gpt_tokenizer/
python main.py train --n-samples 50000
```

After training, verify `tokenizer.json` was created:

```bash
ls ~/MiniGPT/mini_gpt_tokenizer/
# Should show: merges.txt  special_tokens_map.json  tokenizer.json  vocab.json
```

If `tokenizer.json` is still missing, ensure you have the latest `minigpt_core/tokenizer/tokenizer.py` from the outputs folder, then retrain.

---

### ❌ `pip install -e .` fails with `BackendUnavailable`

**Full error:**

```
pip._vendor.pyproject_hooks._impl.BackendUnavailable:
ModuleNotFoundError: No module named 'setuptools.backends'
```

**Cause:** Old setuptools installed in the venv. The `pyproject.toml` `build-backend` declaration requires setuptools 42+.

**Fix:**

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

---

### ❌ Stale editable install — import resolves to wrong location

**Symptom:** `pip install -e .` succeeds, but `python main.py info` still fails with the unknown location error even after all `__init__.py` files are present.

**Cause:** A previous editable install left a `.pth` file in the venv's `site-packages` pointing to a stale location. The new install does not overwrite it cleanly.

**Fix — run the included script:**

```bash
bash ~/MiniGPT/fix_install.sh
```

This script: uninstalls minigpt, deletes all stale `.pth`, `.egg-link`, and `.dist-info` files from the venv, clears `__pycache__` and `.egg-info` from the project, removes any leftover `src/` directory, and performs a fresh `pip install -e .`.

**Manual fix if the script is unavailable:**

```bash
# Find your venv's site-packages path
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo $SITE

# Remove stale editable files
find $SITE -name "__editable__minigpt*" -delete
find $SITE -name "minigpt*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true
find $SITE -name "minigpt*.egg-link" -delete 2>/dev/null || true

# Clean local artifacts
find ~/MiniGPT -name "__pycache__" -not -path "*/venv/*" -exec rm -rf {} + 2>/dev/null || true
find ~/MiniGPT -name "*.egg-info" -not -path "*/venv/*" -exec rm -rf {} + 2>/dev/null || true

# Fresh install
pip install -e .
python main.py info
```

---

### ❌ Browser: `model not found` or blank output

**Cause:** The ONNX model file does not exist at `web/assets/mini_gpt_quant.onnx`.

**Fix:**

```bash
# Confirm the file is missing
ls ~/MiniGPT/web/assets/ 2>/dev/null || echo "web/assets/ does not exist yet"

# Export the model (requires a trained checkpoint)
python main.py export

# Confirm the file was created
ls -lh ~/MiniGPT/web/assets/
```

---

### ❌ Browser: CORS error or Web Worker fails to load

**Cause:** Opening `index.html` directly in the browser using a `file://` URL. Web Workers and WASM files require an HTTP server.

**Fix:**

```bash
python scripts/serve.py
# Then open: http://localhost:8080
```

Do not open the file directly. Do not use double-click. Always go through the serve script or another HTTP server.

---

### ❌ `HuggingFace 429 Too Many Requests` during dataset download

**Cause:** Unauthenticated HuggingFace requests are rate-limited.

**Fix — Option A (free):** Wait a few minutes and retry.

**Fix — Option B (recommended):** Create a free HuggingFace account, generate a token at https://huggingface.co/settings/tokens, then:

```bash
export HF_TOKEN=hf_your_token_here
python main.py train --n-samples 50000
```

---

## Part 6 — Diagnostic Commands

When something goes wrong, run these to gather information before asking for help:

```bash
# Where is the project?
pwd
ls

# What Python and pip are active?
which python && python --version
which pip && pip --version

# Is the venv active?
echo $VIRTUAL_ENV

# What does Python think minigpt_core is?
python -c "
import sys
print('sys.path:', sys.path[:3])
import minigpt_core
print('minigpt_core.__file__:', minigpt_core.__file__)
import minigpt_core.model as m
print('model.__file__:', m.__file__)
print('model contents:', [x for x in dir(m) if not x.startswith('_')])
"

# Are all __init__.py files present?
find ~/MiniGPT/minigpt_core -name "__init__.py" | sort

# Are there any stale src. imports?
grep -rn "from src\." ~/MiniGPT --include="*.py" --exclude-path="*/venv/*"

# What files exist in the tokenizer directory?
ls ~/MiniGPT/mini_gpt_tokenizer/ 2>/dev/null || echo "mini_gpt_tokenizer/ not created yet"

# What files exist in web/assets?
ls ~/MiniGPT/web/assets/ 2>/dev/null || echo "web/assets/ not created yet"

# Run the test suite
cd ~/MiniGPT && python -m pytest tests/ -v --tb=short
```

---

## Part 7 — Full Command Reference

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -e .
pip install datasets tokenizers transformers tqdm

# Verify
python main.py info

# Training
python main.py train --n-samples 50000    # quick run
python main.py train                       # full run
python main.py train --amp                 # GPU + mixed precision

# Export
python main.py export                      # → web/assets/mini_gpt_quant.onnx

# Generate (CLI)
python main.py generate --prompt "Once upon a time"
python main.py generate                    # interactive prompt

# Benchmark
python main.py benchmark
python scripts/benchmark.py --all

# Browser
python scripts/serve.py                   # http://localhost:8080
python scripts/serve.py --port 3000       # custom port

# Tests
python -m pytest tests/ -v
python -m pytest tests/ -v --tb=short -k "not tokenizer_train"

# Clean reinstall
bash fix_install.sh

# Mobile export
pip install coremltools onnx               # iOS
python scripts/export_mobile.py --platform ios

pip install onnx-tf tensorflow             # Android
python scripts/export_mobile.py --platform android
```
