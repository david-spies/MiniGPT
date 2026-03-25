# ⚡ MiniGPT — Quickstart

Get from zero to a running language model in under 5 minutes.

---

## Prerequisites

- Python 3.10+
- Node.js 18+ (for local web server only)
- 2GB free disk space (for dataset download)
- Any modern browser (Chrome, Firefox, Safari, Edge)

---

## Step 1 — Install Python Dependencies

```bash
git clone https://github.com/david-spies/minigpt.git
cd minigpt
pip install -r requirements.txt
```

**GPU users (NVIDIA):** Replace the torch install for CUDA acceleration:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3):** MPS acceleration is auto-detected — no changes needed.

---

## Step 2 — Verify Config & Size Estimates

Before training, confirm the model will fit under 16MB:

```bash
python main.py info
```

Expected output:
```
Config: DEFAULT (1.4M params)
  Parameters  : 1,419,264
  FP32 size   : 5.41 MB
  FP16 size   : 2.71 MB
  INT8 size   : 1.35 MB
  Under 16MB  : ✓ (INT8)
```

---

## Step 3 — Train the Tokenizer + Model

### Option A: Quick test run (~10 minutes, CPU)
```bash
python main.py train --n-samples 50000
```

### Option B: Full training (~2–4 hours, CPU)
```bash
python main.py train
```

### Option C: Full training with GPU acceleration
```bash
python main.py train --amp
```

Training creates:
```
checkpoints/
├── mini_gpt_best.pt    ← Best validation loss checkpoint
└── mini_gpt_final.pt   ← Final epoch checkpoint

mini_gpt_tokenizer/
├── vocab.json
├── merges.txt
└── special_tokens_map.json
```

---

## Step 4 — Export to ONNX

```bash
python main.py export
```

This produces two files in `web/assets/`:

| File | Size | Use |
|---|---|---|
| `mini_gpt_fp32.onnx` | ~5.5 MB | Debugging |
| `mini_gpt_quant.onnx` | ~1.4 MB | **Production** |

The tokenizer files are also copied automatically.

---

## Step 5 — Run in Browser

```bash
# Requires a local server (not file:// due to Web Worker restrictions)
npx http-server web/ -p 8080
```

Open **http://localhost:8080** in your browser. The model loads in ~1 second on most connections.

![MiniGPT Browser UI](https://placeholder.example.com/screenshot)

### What you should see:
1. Click **"Load Model (~1.5MB)"** — progress bar fills
2. Status changes to `model ready ✓`
3. Enter a prompt in the text area
4. Click **"▶ Generate"** — tokens stream in real time
5. Stats bar shows tokens/sec

---

## Step 6 — Test Generation (CLI)

```bash
python main.py generate --prompt "Once upon a time, there was a cat named"
```

Example output:
```
Once upon a time, there was a cat named Luna who lived in a small house near the forest.
She loved to chase butterflies and sleep in the warm sun. One day, she found a tiny bird
with a hurt wing...
```

---

## Step 7 — Benchmark

```bash
# Benchmark ONNX (matches browser performance)
python main.py benchmark --onnx web/assets/mini_gpt_quant.onnx
```

Expected results:
```
Benchmark Results:
  Speed  : 85.3 tokens/sec
  Latency: 586ms for 50 tokens
```

---

## Deploy to GitHub Pages (Free Hosting)

```bash
# Copy docs to root for GitHub Pages
cp docs/README.md web/README.md

git init
git add .
git commit -m "feat: initial MiniGPT deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/minigpt.git
git push -u origin main
```

Then in GitHub → **Settings → Pages → Source: main branch → /web folder**.

Your model is live at `https://YOUR_USERNAME.github.io/minigpt/` in ~60 seconds.

---

## Common Issues

**`Workers are not supported` error in browser:**
You must serve files over HTTP, not open `index.html` directly with `file://`.
Use `npx http-server` or VS Code Live Server.

**`Model not found` error:**
Ensure `web/assets/mini_gpt_quant.onnx` exists. Run `python main.py export` first.

**Slow generation:**
- Chrome/Edge generally have the fastest WASM SIMD implementation
- Firefox: enable `javascript.options.wasm_simd` in `about:config`
- Try using more threads: edit `ort.env.wasm.numThreads` in `inference.worker.js`

**CUDA out of memory during training:**
Reduce batch size: `python main.py train --amp` (AMP uses less VRAM)
Or train on CPU with `--n-samples 50000` for a quick test.
