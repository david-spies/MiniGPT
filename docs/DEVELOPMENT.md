# 🛠 MiniGPT — Development Guide

Deep dive into the architecture, optimization decisions, and how to extend the project.

---

## Architecture Overview

```
Input IDs (B, T)
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  Token Embedding (wte)  5000 × 128           │
  │+ Position Embedding (wpe) 256 × 128          │
  │  Dropout                                     │
  └─────────────────────────────────────────────┘
       │
       ▼   ×4 layers
  ┌─────────────────────────────────────────────┐
  │  Block                                       │
  │  ├─ LayerNorm                                │
  │  ├─ CausalSelfAttention                      │
  │  │   ├─ Fused QKV: Linear(128 → 384)         │
  │  │   ├─ Causal mask                          │
  │  │   ├─ KV-Cache concat (inference only)     │
  │  │   └─ Output projection: Linear(128 → 128) │
  │  ├─ LayerNorm                                │
  │  └─ MLP                                      │
  │      ├─ Linear(128 → 512) + GELU             │
  │      └─ Linear(512 → 128) + Dropout          │
  └─────────────────────────────────────────────┘
       │
       ▼
  LayerNorm → LM Head (tied to wte weights)
       │
       ▼
  Logits (B, T, 5000)
```

**Key design choices explained below.**

---

## Parameter Budget Breakdown

```
Component                    Params       Size (INT8)
─────────────────────────────────────────────────────
wte  (5000 × 128)            640,000      625 KB
wpe  (256  × 128)             32,768       32 KB
× 4 blocks:
  QKV proj  (128 → 384)       49,152
  O   proj  (128 → 128)       16,384
  FFN up    (128 → 512)       65,536
  FFN down  (512 → 128)       65,536
  LayerNorms                     512
  Sub-total per block         197,120      193 KB × 4
lm_head    (tied to wte)            0       0 KB  ← Weight tying!
─────────────────────────────────────────────────────
Total                       1,460,768    ~1.39 MB ✓
```

**Weight tying** is the single highest-impact optimization: the output projection (`lm_head`) is set to share the exact same tensor as the input embedding (`wte`). This saves 640,000 parameters (~625 KB) with no quality loss, since both layers map between the same embedding space and token space.

---

## KV-Cache: How It Works

Without a cache, generating token #100 requires recomputing the full 100-token attention matrix — O(n²) complexity.

With the KV-Cache, each layer stores its past Key and Value tensors. Generating token #100 only requires:
1. Computing Q, K, V for the **single new token**
2. Concatenating new K/V with cached K/V
3. Running one attention operation

```
Step 1 (prompt): feed all N tokens → build initial KV cache
Step 2 onwards : feed 1 token + existing cache → O(1) per token
```

**ONNX I/O structure for KV-cache:**
```
Inputs:  input_ids, past_k_0, past_v_0, past_k_1, past_v_1, ..., past_k_3, past_v_3
Outputs: logits,  present_k_0, present_v_0, ..., present_k_3, present_v_3
```

On the first call, `past_k_i` / `past_v_i` are zero-sized tensors `[1, 4, 0, 32]`.
On subsequent calls, they are the `present_*` outputs from the previous step.

---

## Training Details

### Dataset
**TinyStories** (Eldan & Li, 2023) — ~2M short stories written for children aged 3–4, designed so that small models can learn coherent grammar by focusing on simple, high-frequency narrative patterns.

Why it's critical: a general web corpus would require 100M+ parameters to learn coherent grammar. TinyStories enables grammar learning at 1M parameters.

### Loss Function
Standard cross-entropy next-token prediction:
```
loss = CrossEntropy(logits[:-1], tokens[1:])
```
The model predicts the next token at every position simultaneously.

### LR Schedule
Linear warmup (200 steps) → cosine decay to 10% of peak LR.

```
Peak LR: 5e-4
Min LR:  5e-5
Warmup:  200 steps
```

This follows the GPT-2/GPT-3 schedule and prevents early training instability.

### Gradient Clipping
All gradient norms clipped to 1.0. Critical for stability with small batch sizes.

### Weight Initialization
- Linear layers: `N(0, 0.02)`
- Residual projections: `N(0, 0.02 / sqrt(2 * n_layer))` — GPT-2 style scaled-down init
- LayerNorm: weight=1, bias=0

---

## ONNX Export Details

The export wraps the model in `_ONNXWrapper` which flattens the list-of-tuples KV cache into individual tensor arguments — required because `torch.onnx.export` doesn't support arbitrary Python structures as I/O.

Dynamic axes are set for:
- `input_ids`: seq dimension (handles both prompt and single-token decode)
- All `past_k_*/past_v_*`: past_seq dimension (grows with each decode step)
- All `present_k_*/present_v_*`: total_seq dimension

**ONNX opset 14** is required for correct handling of dynamic shape concatenation in the KV-cache.

---

## Browser Runtime Architecture

```
Main Thread (index.html)
│
│   postMessage({ type: 'generate', ... })
│
▼
Web Worker (inference.worker.js)
│
├─ ONNX Runtime Web (ort.min.js via CDN)
│   └─ WASM SIMD backend
├─ SimpleBPETokenizer (vocab.json + merges.txt)
├─ KV-Cache management (plain JS objects → ort.Tensor)
│
│   postMessage({ type: 'token', text: '...' })  ← streaming
│   postMessage({ type: 'done', tps: 85 })
│
▼
Main Thread updates DOM
```

The Web Worker is essential: ONNX inference blocks the JavaScript thread. Without it, the browser UI would freeze during generation. The Worker runs in a separate thread, posts individual tokens back to the main thread as they're generated, enabling real-time streaming output.

---

## Extending the Model

### Increasing Capacity (while staying under 16MB)

| Config | Params | INT8 Size | Notes |
|---|---|---|---|
| 4L × 128d | 1.4M | 1.4MB | Default |
| 4L × 256d | 3.2M | 3.2MB | Better quality |
| 6L × 256d | 4.6M | 4.6MB | Noticeably better |
| 6L × 512d | 14.2M | **~14.2MB** | Max before hitting 16MB |

Edit `src/model/config.py`:
```python
MEDIUM_CONFIG = MiniGPTConfig(
    vocab_size=8000,
    n_embd=256,
    n_head=8,
    n_layer=6,
    block_size=256,
)
```

### Custom Training Domain

Replace TinyStories with any plain text corpus by modifying `src/tokenizer/dataset.py`:

```python
# In build_dataloaders(), replace:
texts = load_tinystories(...)
# With:
texts = open("your_corpus.txt").read().split("\n\n")
```

Re-train the tokenizer on your corpus first (`prepare_tokenizer()`).

### Adding Grouped-Query Attention (GQA)

GQA uses fewer KV heads than Q heads, reducing KV-cache memory. For a 4-head model, use 2 KV heads:

```python
# In config.py
n_kv_head: int = 2   # Number of KV heads (must divide n_head)
```

Then in `CausalSelfAttention.forward()`, split projections accordingly.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test model output shape
pytest tests/test_model.py

# Test tokenizer round-trip
pytest tests/test_tokenizer.py

# Test ONNX export integrity
pytest tests/test_export.py
```

---

## Performance Profiling

```python
# Profile a forward pass
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True
) as prof:
    logits, _ = model(input_ids)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-improvement`
3. Ensure `python main.py info` still shows INT8 size < 16MB
4. Add tests for any new functionality
5. Open a pull request with a clear description

**Contribution ideas:**
- Grouped-Query Attention (GQA) for smaller KV cache
- ALiBi positional encoding (removes `wpe`, saves 32KB, enables longer contexts)
- GGUF export for llama.cpp compatibility
- React Native wrapper for cross-platform mobile
- Streaming SSE endpoint for server-side deployment
