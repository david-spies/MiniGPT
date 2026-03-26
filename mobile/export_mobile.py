"""
scripts/export_mobile.py
Exports MiniGPT to:
  - CoreML (.mlpackage) for iOS/macOS
  - TensorFlow Lite (.tflite) for Android

Usage:
  python scripts/export_mobile.py --checkpoint checkpoints/mini_gpt_best.pt --platform all
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def export_coreml(model, config, output_path: str = "mobile/ios/mini_gpt.mlpackage"):
    """
    Convert ONNX → CoreML for iOS/macOS Neural Engine deployment.
    Requires: pip install coremltools onnx
    """
    try:
        import coremltools as ct
        import onnx
    except ImportError:
        raise ImportError("pip install coremltools onnx")

    # First export to ONNX
    from minigpt_core.export import export_onnx
    onnx_path = "/tmp/mini_gpt_export.onnx"
    export_onnx(model, config, onnx_path)

    logger.info("Converting ONNX → CoreML...")
    onnx_model = onnx.load(onnx_path)

    # Convert with flexible input shapes
    mlmodel = ct.convert(
        onnx_model,
        source="onnx",
        convert_to="mlprogram",  # .mlpackage (neural engine compatible)
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16,
        inputs=[
            ct.TensorType(name="input_ids", shape=ct.Shape(shape=(1, ct.RangeDim(1, config.block_size))), dtype=int),
        ],
    )

    # Metadata
    mlmodel.short_description = "MiniGPT — 1.4M param language model"
    mlmodel.author = "MiniGPT Project"
    mlmodel.license = "MIT"
    mlmodel.version = "1.0"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)

    size_mb = sum(
        f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
    ) / 1024**2
    logger.info(f"CoreML package saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


def export_tflite(model, config, output_path: str = "mobile/android/app/src/main/assets/mini_gpt.tflite"):
    """
    Convert PyTorch → ONNX → TFLite for Android deployment.
    Requires: pip install onnx onnx-tf tensorflow
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        raise ImportError("pip install onnx onnx-tf tensorflow")

    from minigpt_core.export import export_onnx

    onnx_path = "/tmp/mini_gpt_export.onnx"
    tf_path = "/tmp/mini_gpt_tf"

    # Step 1: ONNX export
    logger.info("Step 1/3 — Exporting to ONNX...")
    export_onnx(model, config, onnx_path)

    # Step 2: ONNX → TensorFlow SavedModel
    logger.info("Step 2/3 — Converting to TensorFlow SavedModel...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)

    # Step 3: TF SavedModel → TFLite
    logger.info("Step 3/3 — Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / 1024**2
    logger.info(f"TFLite model saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export MiniGPT to mobile formats")
    parser.add_argument("--checkpoint", default="checkpoints/mini_gpt_best.pt")
    parser.add_argument("--platform", choices=["ios", "android", "all"], default="all")
    parser.add_argument("--output-dir", default="mobile")
    args = parser.parse_args()

    import torch
    from minigpt_core.model import MiniGPT, DEFAULT_CONFIG

    config = DEFAULT_CONFIG
    model = MiniGPT(config)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    if args.platform in ("ios", "all"):
        export_coreml(model, config, f"{args.output_dir}/ios/mini_gpt.mlpackage")

    if args.platform in ("android", "all"):
        export_tflite(
            model, config,
            f"{args.output_dir}/android/app/src/main/assets/mini_gpt.tflite"
        )

    logger.info("Mobile export complete.")


if __name__ == "__main__":
    main()
