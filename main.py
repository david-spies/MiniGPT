"""
MiniGPT — Main Entry Point

Commands:
  python main.py train       — Train model on TinyStories
  python main.py export      — Export trained model to ONNX + quantize
  python main.py generate    — Interactive generation in terminal
  python main.py benchmark   — Benchmark inference speed
  python main.py info        — Show model parameter count and size estimate
"""
import argparse
import logging
import sys
import os

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# MUST run before any project imports.
#
# Problem: pip editable installs register a .pth file that adds the install-time
# directory to sys.path. If the project was ever installed from a different path,
# or if a prior 'src' package install exists, Python resolves minigpt_core to
# "(unknown location)" — an empty namespace package — instead of the local files.
#
# Fix: forcibly evict any previously-imported minigpt_core from sys.modules and
# put the real project root at index 0 so it always wins.

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Remove any stale minigpt_core entries from the module cache
for _key in list(sys.modules.keys()):
    if _key == "minigpt_core" or _key.startswith("minigpt_core."):
        del sys.modules[_key]

# Put project root first, removing any duplicate entries
sys.path = [_ROOT] + [p for p in sys.path if os.path.realpath(p) != os.path.realpath(_ROOT)]

# Sanity-check: the package must be importable as a real directory, not a namespace
_pkg_path = os.path.join(_ROOT, "minigpt_core", "__init__.py")
if not os.path.isfile(_pkg_path):
    print(f"ERROR: minigpt_core/__init__.py not found at {_pkg_path}")
    print(f"       Make sure you are running main.py from the project root: {_ROOT}")
    sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("minigpt")


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_info(args):
    from minigpt_core.model import MiniGPT, DEFAULT_CONFIG, MEDIUM_CONFIG
    for name, cfg in [("DEFAULT (1.4M params)", DEFAULT_CONFIG), ("MEDIUM (still <16MB)", MEDIUM_CONFIG)]:
        model = MiniGPT(cfg)
        params = model.count_parameters()
        print(f"\n{'='*55}")
        print(f"Config: {name}")
        print(f"  Parameters  : {params:,}")
        print(f"  FP32 size   : {cfg.estimated_size_mb(32):.2f} MB")
        print(f"  FP16 size   : {cfg.estimated_size_mb(16):.2f} MB")
        print(f"  INT8 size   : {cfg.estimated_size_mb(8):.2f} MB")
        print(f"  Under 16MB  : {'✓' if cfg.estimated_size_mb(8) < 16 else '✗'} (INT8)")
    print()


def cmd_train(args):
    import torch
    from minigpt_core.model import MiniGPT, DEFAULT_CONFIG
    from minigpt_core.tokenizer import prepare_tokenizer, build_dataloaders
    from minigpt_core.training import Trainer

    config = DEFAULT_CONFIG
    logger.info(f"Config: {config}")

    tokenizer = prepare_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=config.vocab_size,
    )
    train_loader, val_loader = build_dataloaders(
        tokenizer,
        block_size=config.block_size,
        batch_size=config.batch_size,
        n_train_samples=args.n_samples,
    )
    model = MiniGPT(config)
    logger.info(f"Model: {model.count_parameters():,} parameters")

    trainer = Trainer(
        model, config, train_loader, val_loader,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=args.amp,
    )
    trainer.train()


def cmd_export(args):
    import torch
    from minigpt_core.model import MiniGPT, DEFAULT_CONFIG
    from minigpt_core.export import export_pipeline

    config = DEFAULT_CONFIG
    model = MiniGPT(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    export_pipeline(model, config, output_dir=args.output_dir)


def cmd_generate(args):
    import torch
    from minigpt_core.model import MiniGPT, DEFAULT_CONFIG
    from minigpt_core.tokenizer import load_tokenizer
    from minigpt_core.inference import generate

    config = DEFAULT_CONFIG
    model = MiniGPT(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = load_tokenizer(args.tokenizer_dir)

    prompt = args.prompt or input("Enter prompt: ")
    result = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens)
    print("\n" + "="*60)
    print(result)
    print("="*60 + "\n")


def cmd_benchmark(args):
    import torch
    from minigpt_core.model import MiniGPT, DEFAULT_CONFIG
    from minigpt_core.tokenizer import load_tokenizer
    from minigpt_core.inference import benchmark_pytorch, OnnxInferenceSession, benchmark_onnx

    config = DEFAULT_CONFIG
    tokenizer = load_tokenizer(args.tokenizer_dir)
    prompt_tokens = tokenizer.encode("Once upon a time,")

    if args.onnx:
        logger.info(f"Benchmarking ONNX: {args.onnx}")
        session = OnnxInferenceSession(args.onnx)
        result = benchmark_onnx(
            session, prompt_tokens,
            n_tokens=50, n_runs=3,
            n_head=config.n_head, head_dim=config.head_dim,
        )
    else:
        model = MiniGPT(config)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        result = benchmark_pytorch(model, tokenizer, n_tokens=50)

    print(f"\nBenchmark Results:")
    print(f"  Speed  : {result['tps']:.1f} tokens/sec")
    print(f"  Latency: {result['avg_s']*1000:.0f}ms for 50 tokens")


# ── Argument Parser ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiniGPT <16MB Language Model")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show model sizes and parameter counts")

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--tokenizer-dir", default="mini_gpt_tokenizer")
    p_train.add_argument("--checkpoint-dir", default="checkpoints")
    p_train.add_argument("--n-samples", type=int, default=None)
    p_train.add_argument("--amp", action="store_true")

    p_export = sub.add_parser("export", help="Export to ONNX + quantize")
    p_export.add_argument("--checkpoint", default="checkpoints/mini_gpt_best.pt")
    p_export.add_argument("--output-dir", default="web/assets")

    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--checkpoint", default="checkpoints/mini_gpt_best.pt")
    p_gen.add_argument("--tokenizer-dir", default="mini_gpt_tokenizer")
    p_gen.add_argument("--prompt", default=None)
    p_gen.add_argument("--max-tokens", type=int, default=100)

    p_bench = sub.add_parser("benchmark", help="Benchmark inference speed")
    p_bench.add_argument("--checkpoint", default="checkpoints/mini_gpt_best.pt")
    p_bench.add_argument("--tokenizer-dir", default="mini_gpt_tokenizer")
    p_bench.add_argument("--onnx", default=None)

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
