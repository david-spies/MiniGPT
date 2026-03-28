"""
MiniGPT ONNX Export + INT8 Quantization

Exports two ONNX models:
1. mini_gpt_fp32.onnx  — Full precision, good for debugging (~8MB)
2. mini_gpt_quant.onnx — INT8 quantized, production target (~2MB)

Compatibility: PyTorch 2.2 through 2.11+
  - Uses dynamo=False to force the legacy TorchScript exporter.
  - PyTorch 2.7+ defaults to the dynamo exporter which does not accept
    dynamic_axes with flattened tuple inputs — the legacy exporter handles
    both correctly and produces identical ONNX graphs.

KV-Cache ONNX Structure:
  Inputs : input_ids, past_k_0, past_v_0, ..., past_k_{n-1}, past_v_{n-1}
  Outputs: logits,  present_k_0, present_v_0, ..., present_k_{n-1}, present_v_{n-1}
"""
import os
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX wrapper — flattens list-of-tuple KV cache into individual tensor I/O
# ---------------------------------------------------------------------------

class _ONNXWrapper(torch.nn.Module):
    """
    Wraps MiniGPT to present a flat signature compatible with torch.onnx.export.

    The model's forward() accepts past_key_values as List[Tuple[Tensor, Tensor]]
    and returns (logits, List[Tuple[Tensor, Tensor]]). ONNX export requires all
    inputs and outputs to be individual tensors at the top level, so this wrapper
    unpacks them on the way in and flattens on the way out.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layer = model.config.n_layer

    def forward(self, input_ids, *past_kv_flat):
        # Reconstruct List[Tuple[k, v]] from the flat positional args
        past_key_values = None
        if len(past_kv_flat) > 0 and past_kv_flat[0].shape[2] > 0:
            past_key_values = [
                (past_kv_flat[i * 2], past_kv_flat[i * 2 + 1])
                for i in range(self.n_layer)
            ]

        logits, new_kvs = self.model(input_ids, past_key_values)

        # Flatten List[Tuple[k, v]] back to individual tensors for ONNX output
        flat_kvs = []
        for k, v in new_kvs:
            flat_kvs.extend([k, v])

        return (logits, *flat_kvs)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(
    model,
    config,
    output_path: str = "mini_gpt_fp32.onnx",
    opset: int = 14,
) -> str:
    """
    Export MiniGPT to ONNX using the legacy TorchScript exporter (dynamo=False).

    dynamo=False is explicit to ensure compatibility with PyTorch 2.7–2.11+
    where the default changed to the dynamo exporter, which cannot handle
    dynamic_axes with flattened variadic inputs.
    """
    model.eval()
    model.cpu()

    wrapped = _ONNXWrapper(model)
    wrapped.eval()

    B        = 1
    T        = 1       # single-token incremental decode
    past_T   = 8       # dummy past length for shape tracing
    head_dim = config.n_embd // config.n_head

    input_ids    = torch.zeros((B, T), dtype=torch.long)
    past_kv_dummy = []
    for _ in range(config.n_layer):
        past_kv_dummy.append(torch.zeros(B, config.n_head, past_T, head_dim))
        past_kv_dummy.append(torch.zeros(B, config.n_head, past_T, head_dim))

    # ── I/O names ──────────────────────────────────────────────────────────
    input_names  = ["input_ids"]
    output_names = ["logits"]
    for i in range(config.n_layer):
        input_names  += [f"past_k_{i}",    f"past_v_{i}"]
        output_names += [f"present_k_{i}", f"present_v_{i}"]

    # ── Dynamic axes ────────────────────────────────────────────────────────
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "logits":    {0: "batch", 1: "seq"},
    }
    for i in range(config.n_layer):
        dynamic_axes[f"past_k_{i}"]    = {0: "batch", 2: "past_seq"}
        dynamic_axes[f"past_v_{i}"]    = {0: "batch", 2: "past_seq"}
        dynamic_axes[f"present_k_{i}"] = {0: "batch", 2: "total_seq"}
        dynamic_axes[f"present_v_{i}"] = {0: "batch", 2: "total_seq"}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Export using legacy TorchScript exporter ────────────────────────────
    # dynamo=False bypasses the new dynamo-based exporter introduced in
    # PyTorch 2.7 which breaks dynamic_axes with tuple/variadic inputs.
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (input_ids, *past_kv_dummy),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,           # ← force legacy exporter, required for PyTorch 2.7+
        )

    size_mb = os.path.getsize(output_path) / 1024**2
    logger.info(f"ONNX exported: {output_path} ({size_mb:.2f} MB)")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_onnx(
    fp32_path: str = "mini_gpt_fp32.onnx",
    output_path: str = "mini_gpt_quant.onnx",
    quant_type: str = "uint8",
) -> str:
    """
    INT8 dynamic quantization of an ONNX model.
    Reduces ~8MB FP32 → ~2MB INT8 with minimal quality loss.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError("pip install onnxruntime")

    qt = QuantType.QUInt8 if quant_type == "uint8" else QuantType.QInt8
    quantize_dynamic(fp32_path, output_path, weight_type=qt)

    size_mb = os.path.getsize(output_path) / 1024**2
    logger.info(f"Quantized ONNX: {output_path} ({size_mb:.2f} MB)")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def export_pipeline(
    model,
    config,
    output_dir: str = "web/assets",
) -> dict:
    """
    Full export pipeline: FP32 ONNX → INT8 ONNX.
    Returns dict with paths to both exported files.
    """
    os.makedirs(output_dir, exist_ok=True)
    fp32_path  = os.path.join(output_dir, "mini_gpt_fp32.onnx")
    quant_path = os.path.join(output_dir, "mini_gpt_quant.onnx")

    logger.info("Step 1/2 — Exporting FP32 ONNX...")
    export_onnx(model, config, fp32_path)

    logger.info("Step 2/2 — Quantizing to INT8...")
    quantize_onnx(fp32_path, quant_path)

    fp32_mb  = os.path.getsize(fp32_path)  / 1024**2
    quant_mb = os.path.getsize(quant_path) / 1024**2
    reduction = (1 - quant_mb / fp32_mb) * 100

    # Step 3: copy tokenizer to web/assets so browser can load it
    tok_src = "mini_gpt_tokenizer"
    tok_dst = os.path.join(output_dir, "mini_gpt_tokenizer")
    if os.path.exists(tok_src):
        import shutil
        if os.path.exists(tok_dst):
            shutil.rmtree(tok_dst)
        shutil.copytree(tok_src, tok_dst)
        logger.info(f"Tokenizer copied → {tok_dst}")
    else:
        logger.warning(f"Tokenizer not found at '{tok_src}' — browser may fail to load vocab.")

    logger.info(
        f"\n{'='*50}\n"
        f"Export complete!\n"
        f"  FP32 : {fp32_mb:.2f} MB  → {fp32_path}\n"
        f"  INT8 : {quant_mb:.2f} MB  → {quant_path}\n"
        f"  Tokenizer → {tok_dst}\n"
        f"  Reduction : {reduction:.0f}%\n"
        f"  Under 16MB: {'✓ YES' if quant_mb < 16 else '✗ NO'}\n"
        f"{'='*50}"
    )

    return {"fp32": fp32_path, "quantized": quant_path}


def copy_tokenizer_to_web(
    tokenizer_dir: str = "mini_gpt_tokenizer",
    output_dir: str = "web/assets",
) -> str:
    """
    Copy the trained tokenizer files into web/assets so the browser can load them.
    The browser worker fetches vocab.json and merges.txt from this directory.
    """
    import shutil

    src = Path(tokenizer_dir)
    dst = Path(output_dir) / "mini_gpt_tokenizer"

    if not src.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at '{tokenizer_dir}'. Run 'python main.py train' first."
        )

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    logger.info(f"Tokenizer copied: {src} → {dst}")
    return str(dst)
