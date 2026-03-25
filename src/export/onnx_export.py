"""
MiniGPT ONNX Export + INT8 Quantization

Exports two ONNX models:
1. mini_gpt_fp32.onnx  — Full precision, good for debugging (~5.5MB)
2. mini_gpt_quant.onnx — INT8 quantized, production target (~1.5MB)

KV-Cache ONNX Structure:
  Inputs : input_ids, past_k_0, past_v_0, ..., past_k_{n-1}, past_v_{n-1}
  Outputs: logits,    present_k_0, present_v_0, ..., present_k_{n-1}, present_v_{n-1}
"""
import os
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

class _ONNXWrapper(torch.nn.Module):
    """
    Flattens the list-of-tuples KV-cache into individual tensor args/returns
    so torch.onnx.export can handle them cleanly.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layer = model.config.n_layer

    def forward(self, input_ids, *past_kv_flat):
        # Reconstruct list of (k, v) pairs
        past_key_values = None
        if past_kv_flat and past_kv_flat[0] is not None:
            past_key_values = [
                (past_kv_flat[i * 2], past_kv_flat[i * 2 + 1])
                for i in range(self.n_layer)
            ]

        logits, new_kvs = self.model(input_ids, past_key_values)

        # Flatten new_kvs back to a tuple for ONNX
        flat_kvs = []
        for k, v in new_kvs:
            flat_kvs.extend([k, v])

        return (logits, *flat_kvs)


def export_onnx(
    model,
    config,
    output_path: str = "mini_gpt_fp32.onnx",
    opset: int = 14,
) -> str:
    """
    Export MiniGPT to ONNX with full KV-cache support.

    Parameters
    ----------
    model       : trained MiniGPT instance
    config      : MiniGPTConfig
    output_path : output .onnx file path
    opset       : ONNX opset version (14+ recommended for dynamic shapes)

    Returns
    -------
    Absolute path to exported file
    """
    model.eval()
    model.cpu()

    wrapped = _ONNXWrapper(model)

    B = 1
    T = 1          # Single-token incremental decode
    past_T = 8     # Dummy past length for shape inference

    input_ids = torch.zeros((B, T), dtype=torch.long)

    head_dim = config.n_embd // config.n_head
    past_kv_dummy = []
    for _ in range(config.n_layer):
        k = torch.zeros(B, config.n_head, past_T, head_dim)
        v = torch.zeros(B, config.n_head, past_T, head_dim)
        past_kv_dummy.extend([k, v])

    # Build names
    input_names = ["input_ids"]
    for i in range(config.n_layer):
        input_names += [f"past_k_{i}", f"past_v_{i}"]

    output_names = ["logits"]
    for i in range(config.n_layer):
        output_names += [f"present_k_{i}", f"present_v_{i}"]

    # Dynamic axes
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
    for i in range(config.n_layer):
        dynamic_axes[f"past_k_{i}"] = {0: "batch", 2: "past_seq"}
        dynamic_axes[f"past_v_{i}"] = {0: "batch", 2: "past_seq"}
        dynamic_axes[f"present_k_{i}"] = {0: "batch", 2: "total_seq"}
        dynamic_axes[f"present_v_{i}"] = {0: "batch", 2: "total_seq"}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

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
    )

    size_mb = os.path.getsize(output_path) / 1024**2
    logger.info(f"ONNX exported: {output_path} ({size_mb:.2f} MB)")
    return os.path.abspath(output_path)


def quantize_onnx(
    fp32_path: str = "mini_gpt_fp32.onnx",
    output_path: str = "mini_gpt_quant.onnx",
    quant_type: str = "uint8",
) -> str:
    """
    INT8 dynamic quantization of an ONNX model.
    Reduces ~5.5MB FP32 → ~1.5MB INT8 with minimal quality loss.

    Parameters
    ----------
    fp32_path   : path to full-precision ONNX model
    output_path : output quantized ONNX path
    quant_type  : 'uint8' (default) or 'int8'

    Returns
    -------
    Absolute path to quantized file
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
    fp32_path = os.path.join(output_dir, "mini_gpt_fp32.onnx")
    quant_path = os.path.join(output_dir, "mini_gpt_quant.onnx")

    logger.info("Step 1/2 — Exporting FP32 ONNX...")
    export_onnx(model, config, fp32_path)

    logger.info("Step 2/2 — Quantizing to INT8...")
    quantize_onnx(fp32_path, quant_path)

    fp32_mb = os.path.getsize(fp32_path) / 1024**2
    quant_mb = os.path.getsize(quant_path) / 1024**2
    reduction = (1 - quant_mb / fp32_mb) * 100

    logger.info(
        f"\n{'='*50}\n"
        f"Export complete!\n"
        f"  FP32:  {fp32_mb:.2f} MB → {fp32_path}\n"
        f"  INT8:  {quant_mb:.2f} MB → {quant_path}\n"
        f"  Size reduction: {reduction:.0f}%\n"
        f"  Under 16MB: {'✓ YES' if quant_mb < 16 else '✗ NO'}\n"
        f"{'='*50}"
    )

    return {"fp32": fp32_path, "quantized": quant_path}
