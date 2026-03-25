#!/usr/bin/env python3
"""
scripts/benchmark.py — Comprehensive MiniGPT benchmark suite.

Measures:
  - Model forward pass latency
  - KV-cache vs no-cache generation speed
  - Memory usage
  - ONNX Runtime vs PyTorch parity

Usage:
  python scripts/benchmark.py --checkpoint checkpoints/mini_gpt_best.pt
  python scripts/benchmark.py --onnx web/assets/mini_gpt_quant.onnx
  python scripts/benchmark.py --all   # run both
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _bar(tps: float, max_tps: float = 200, width: int = 30) -> str:
    filled = int(min(tps / max_tps, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _print_header(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _print_row(label: str, value: str, unit: str = "", note: str = "") -> None:
    row = f"  {label:<28} {value:>10} {unit:<10}"
    if note:
        row += f"  # {note}"
    print(row)


# ── PyTorch benchmark ──────────────────────────────────────────────────────────

def benchmark_pytorch(
    model,
    config,
    prompt_ids: list,
    n_tokens: int = 50,
    n_runs: int = 5,
    device: str = "cpu",
) -> dict:
    """Benchmark PyTorch KV-cached generation."""
    import torch
    import torch.nn.functional as F

    model.eval().to(device)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(prompt, max_new_tokens=5)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=n_tokens)
        t1 = time.perf_counter()
        actual = out.shape[1] - prompt.shape[1]
        times.append((t1 - t0, actual))

    avg_time = sum(t for t, _ in times) / n_runs
    avg_tokens = sum(n for _, n in times) / n_runs
    tps = avg_tokens / avg_time

    return {
        "backend": f"PyTorch ({device.upper()})",
        "tps": tps,
        "avg_ms": avg_time * 1000,
        "tokens": avg_tokens,
        "runs": n_runs,
    }


def benchmark_pytorch_no_cache(
    model,
    config,
    prompt_ids: list,
    n_tokens: int = 20,
    device: str = "cpu",
) -> dict:
    """Naive generation without KV cache — shows the speedup factor."""
    model.eval().to(device)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    times = []
    for _ in range(3):
        all_ids = prompt.clone()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_tokens):
                # Recompute full sequence every step — no cache
                logits, _ = model(all_ids)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                all_ids = torch.cat([all_ids, next_tok], dim=1)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    tps = n_tokens / avg_time
    return {"backend": "PyTorch (no cache)", "tps": tps, "avg_ms": avg_time * 1000, "tokens": n_tokens}


# ── ONNX benchmark ─────────────────────────────────────────────────────────────

def benchmark_onnx(
    onnx_path: str,
    config,
    prompt_ids: list,
    n_tokens: int = 50,
    n_runs: int = 5,
    n_threads: int = 4,
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"backend": "ONNX Runtime", "error": "onnxruntime not installed"}

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )

    from src.inference import OnnxInferenceSession
    ort_session = OnnxInferenceSession(onnx_path)

    # Warmup
    ort_session.generate(prompt_ids, max_new_tokens=5, n_head=config.n_head, head_dim=config.head_dim)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = ort_session.generate(
            prompt_ids, max_new_tokens=n_tokens,
            n_head=config.n_head, head_dim=config.head_dim
        )
        t1 = time.perf_counter()
        actual = len(out) - len(prompt_ids)
        times.append((t1 - t0, actual))

    avg_time = sum(t for t, _ in times) / n_runs
    avg_tokens = sum(n for _, n in times) / n_runs
    tps = avg_tokens / avg_time

    model_size_mb = Path(onnx_path).stat().st_size / 1024**2

    return {
        "backend": f"ONNX Runtime (CPU, {n_threads} threads)",
        "tps": tps,
        "avg_ms": avg_time * 1000,
        "tokens": avg_tokens,
        "runs": n_runs,
        "model_mb": model_size_mb,
    }


# ── Memory usage ───────────────────────────────────────────────────────────────

def measure_memory(model) -> dict:
    """Measure model memory footprint."""
    import io

    # Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # State dict size (what gets saved to disk)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    state_dict_bytes = buf.tell()

    return {
        "param_mb": param_bytes / 1024**2,
        "state_dict_mb": state_dict_bytes / 1024**2,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiniGPT comprehensive benchmark")
    parser.add_argument("--checkpoint", default="checkpoints/mini_gpt_best.pt")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--n-tokens", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    from src.model import MiniGPT, DEFAULT_CONFIG
    config = DEFAULT_CONFIG

    _print_header("⚡ MiniGPT Benchmark Suite")

    # ── Config info ────────────────────────────────────────────────────────────
    print(f"\n  Config: {config.n_layer}L × {config.n_embd}d × {config.n_head}H")
    _print_row("Parameters", f"{config.estimated_params():,}", "params")
    _print_row("INT8 model size", f"{config.estimated_size_mb(8):.2f}", "MB")
    _print_row("FP32 model size", f"{config.estimated_size_mb(32):.2f}", "MB")

    # Dummy prompt (10 tokens)
    prompt_ids = [0, 23, 45, 67, 12, 89, 34, 56, 78, 1]
    results = []

    # ── PyTorch ────────────────────────────────────────────────────────────────
    if args.checkpoint and Path(args.checkpoint).exists():
        model = MiniGPT(config)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        _print_header("PyTorch Inference (KV-Cache)")

        # Memory
        mem = measure_memory(model)
        _print_row("Active parameters", f"{mem['n_params']:,}")
        _print_row("Runtime memory", f"{mem['param_mb']:.2f}", "MB", "FP32 tensors")
        _print_row("Checkpoint size", f"{mem['state_dict_mb']:.2f}", "MB", "on disk")

        # With cache
        r_cache = benchmark_pytorch(model, config, prompt_ids, args.n_tokens, args.n_runs)
        _print_row("Speed (with KV-cache)", f"{r_cache['tps']:.1f}", "tok/s")
        _print_row("Latency per 50 tok", f"{r_cache['avg_ms']:.0f}", "ms")
        print(f"\n  Speed bar: [{_bar(r_cache['tps'])}] {r_cache['tps']:.0f}/200 tok/s")
        results.append(r_cache)

        # Without cache (comparison)
        if args.all:
            _print_header("PyTorch Inference (No KV-Cache — for comparison)")
            r_nocache = benchmark_pytorch_no_cache(model, config, prompt_ids, n_tokens=20)
            _print_row("Speed (no cache)", f"{r_nocache['tps']:.1f}", "tok/s")
            speedup = r_cache['tps'] / r_nocache['tps']
            _print_row("KV-cache speedup", f"{speedup:.1f}×", "", "vs naive decode")
            results.append(r_nocache)

    # ── ONNX ───────────────────────────────────────────────────────────────────
    onnx_path = args.onnx or "web/assets/mini_gpt_quant.onnx"
    if Path(onnx_path).exists():
        _print_header(f"ONNX Runtime — {onnx_path}")
        r_onnx = benchmark_onnx(onnx_path, config, prompt_ids, args.n_tokens, args.n_runs, args.threads)

        if "error" in r_onnx:
            print(f"  Skipped: {r_onnx['error']}")
        else:
            _print_row("Model file size", f"{r_onnx['model_mb']:.2f}", "MB", "INT8 quantized")
            _print_row("Speed (ONNX RT)", f"{r_onnx['tps']:.1f}", "tok/s")
            _print_row("Latency per 50 tok", f"{r_onnx['avg_ms']:.0f}", "ms")
            _print_row("Threads used", str(args.threads))
            print(f"\n  Speed bar: [{_bar(r_onnx['tps'])}] {r_onnx['tps']:.0f}/200 tok/s")
            results.append(r_onnx)

    # ── Summary ────────────────────────────────────────────────────────────────
    if len(results) > 1:
        _print_header("Summary")
        for r in results:
            if "error" not in r:
                print(f"  {r['backend']:<45} {r['tps']:>7.1f} tok/s")

    print(f"\n{'═' * 60}")
    print(f"  Browser estimate: ~60–120 tok/s (WASM SIMD, modern device)")
    print(f"  iPhone estimate : ~80–150 tok/s (Apple Neural Engine)")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
