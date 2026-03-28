"""
MiniGPT Tokenizer
- Trains a custom Byte-Level BPE on TinyStories
- Keeps vocab_size=5000 (vs 50k default = 25MB+ embedding)
- Saves a full tokenizer.json via tokenizer.save() for reliable reloading
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
BOS_TOKEN  = "<s>"
PAD_TOKEN  = "<pad>"
EOS_TOKEN  = "</s>"
UNK_TOKEN  = "<unk>"


def train_tokenizer(
    texts: List[str],
    vocab_size: int = 5000,
    save_dir: str = "mini_gpt_tokenizer",
    min_frequency: int = 2,
) -> "ByteLevelBPETokenizer":
    """
    Train a Byte-Level BPE tokenizer and save BOTH formats:
      - vocab.json + merges.txt  (legacy / browser)
      - tokenizer.json           (used by load_tokenizer for reliable reloading)
    """
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        raise ImportError("pip install tokenizers")

    logger.info(f"Training BPE tokenizer | vocab_size={vocab_size} | texts={len(texts):,}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )

    os.makedirs(save_dir, exist_ok=True)

    # Save legacy vocab.json + merges.txt (used by browser / Transformers.js)
    tokenizer.save_model(save_dir)

    # Save full tokenizer.json — this is what load_tokenizer uses to reload
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # Save special tokens map for Transformers compatibility
    special_tokens_map = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "unk_token": UNK_TOKEN,
        "pad_token": PAD_TOKEN,
        "model_max_length": 256,
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    logger.info(f"Tokenizer saved to {save_dir}/")
    return tokenizer


def load_tokenizer(tokenizer_dir: str = "mini_gpt_tokenizer"):
    """
    Load a pre-trained tokenizer from disk.

    Loads from tokenizer.json (the full serialization) via
    PreTrainedTokenizerFast — no sentencepiece or tiktoken required.
    """
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        raise ImportError("pip install transformers")

    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    vocab_file     = os.path.join(tokenizer_dir, "vocab.json")

    if not os.path.exists(tokenizer_file) and not os.path.exists(vocab_file):
        raise FileNotFoundError(
            f"Tokenizer not found at '{tokenizer_dir}'. "
            f"Run 'python main.py train' first."
        )

    # If tokenizer.json exists use it directly — most reliable path
    if os.path.exists(tokenizer_file):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            pad_token=PAD_TOKEN,
        )
    else:
        # Fallback: rebuild from vocab.json + merges.txt via tokenizers library
        logger.warning(
            "tokenizer.json not found — rebuilding from vocab.json + merges.txt. "
            "Re-run training to generate tokenizer.json for faster future loads."
        )
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder

            merges_file = os.path.join(tokenizer_dir, "merges.txt")
            tok = Tokenizer(BPE(vocab=vocab_file, merges=merges_file))
            tok.pre_tokenizer = ByteLevel()
            tok.decoder = ByteLevelDecoder()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tok,
                bos_token=BOS_TOKEN,
                eos_token=EOS_TOKEN,
                unk_token=UNK_TOKEN,
                pad_token=PAD_TOKEN,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load tokenizer from {tokenizer_dir}. "
                f"Delete the directory and re-run training. Original error: {e}"
            )

    logger.info(f"Tokenizer loaded | vocab_size={tokenizer.vocab_size}")
    return tokenizer


def load_tinystories(
    n_samples: Optional[int] = None,
    split: str = "train",
    streaming: bool = True,
) -> List[str]:
    """
    Load TinyStories dataset from HuggingFace.
    Uses streaming=False when loading the full dataset for training
    to avoid the 'Bad file descriptor' crash on interrupted streams.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info(f"Loading TinyStories | split={split} | n_samples={n_samples}")

    if n_samples is not None and streaming:
        # Streaming for small tokenizer-training samples (fast, low memory)
        raw = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        texts = []
        for i, row in enumerate(raw):
            if i >= n_samples:
                break
            texts.append(row["text"])
        return texts
    else:
        # Non-streaming for full training data (avoids GIL/fd crash on cleanup)
        raw = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
        texts = raw["text"]
        return texts[:n_samples] if n_samples else texts


def prepare_tokenizer(
    tokenizer_dir: str = "mini_gpt_tokenizer",
    vocab_size: int = 5000,
    tokenizer_train_samples: int = 20_000,
) -> object:
    """
    High-level: load existing tokenizer or train a new one on TinyStories.
    """
    tokenizer_file = Path(tokenizer_dir) / "tokenizer.json"
    vocab_file     = Path(tokenizer_dir) / "vocab.json"

    if tokenizer_file.exists() or vocab_file.exists():
        logger.info("Found existing tokenizer, loading...")
        return load_tokenizer(tokenizer_dir)

    logger.info("No tokenizer found — training new one on TinyStories...")
    texts = load_tinystories(n_samples=tokenizer_train_samples)
    train_tokenizer(texts, vocab_size=vocab_size, save_dir=tokenizer_dir)
    return load_tokenizer(tokenizer_dir)
