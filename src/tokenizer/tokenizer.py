"""
MiniGPT Tokenizer
- Trains a custom Byte-Level BPE on TinyStories
- Keeps vocab_size=5000 (vs 50k default = 25MB+ embedding)
- Saves/loads as HuggingFace-compatible tokenizer.json
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
BOS_TOKEN = "<s>"
PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


def train_tokenizer(
    texts: List[str],
    vocab_size: int = 5000,
    save_dir: str = "mini_gpt_tokenizer",
    min_frequency: int = 2,
) -> "ByteLevelBPETokenizer":
    """
    Train a Byte-Level BPE tokenizer on provided texts.

    Parameters
    ----------
    texts       : list of raw text strings (e.g. TinyStories subset)
    vocab_size  : target vocabulary size (default 5000 for <16MB)
    save_dir    : directory to save vocab.json and merges.txt
    min_frequency : minimum subword frequency

    Returns
    -------
    Trained tokenizer instance
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
    tokenizer.save_model(save_dir)

    # Also save a full tokenizer.json for Transformers.js compatibility
    tokenizer_json = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "unk_token": UNK_TOKEN,
        "pad_token": PAD_TOKEN,
        "model_max_length": 256,
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w") as f:
        json.dump(tokenizer_json, f, indent=2)

    logger.info(f"Tokenizer saved to {save_dir}/")
    return tokenizer


def load_tokenizer(tokenizer_dir: str = "mini_gpt_tokenizer"):
    """
    Load a pre-trained tokenizer from disk.

    Returns a HuggingFace PreTrainedTokenizerFast instance.
    """
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        raise ImportError("pip install transformers")

    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    merges_file = os.path.join(tokenizer_dir, "merges.txt")

    if not os.path.exists(vocab_file):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_dir}. Run train_tokenizer() first."
        )

    tokenizer = PreTrainedTokenizerFast(
        vocab_file=vocab_file,
        merges_file=merges_file,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
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

    Parameters
    ----------
    n_samples  : if set, only load this many samples (useful for tokenizer training)
    split      : 'train' or 'validation'
    streaming  : use streaming to avoid full download for tokenizer training

    Returns
    -------
    list of text strings
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info(f"Loading TinyStories | split={split} | n_samples={n_samples}")

    if n_samples is not None and streaming:
        raw = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        texts = []
        for i, row in enumerate(raw):
            if i >= n_samples:
                break
            texts.append(row["text"])
        return texts
    else:
        raw = load_dataset("roneneldan/TinyStories", split=split)
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
    if Path(tokenizer_dir).exists() and (Path(tokenizer_dir) / "vocab.json").exists():
        logger.info("Found existing tokenizer, loading...")
        return load_tokenizer(tokenizer_dir)

    logger.info("No tokenizer found — training new one on TinyStories...")
    texts = load_tinystories(n_samples=tokenizer_train_samples)
    train_tokenizer(texts, vocab_size=vocab_size, save_dir=tokenizer_dir)
    return load_tokenizer(tokenizer_dir)
