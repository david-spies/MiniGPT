"""
MiniGPT Dataset Utilities
- Tokenizes TinyStories into fixed-length blocks
- Efficient PyTorch DataLoader setup
"""
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TextBlockDataset(Dataset):
    """
    Concatenates all token IDs into a flat stream, then yields
    non-overlapping fixed-length blocks for next-token prediction.

    This is more memory-efficient than padding each story separately.
    """

    def __init__(self, token_ids: list, block_size: int = 256):
        self.block_size = block_size
        flat = torch.tensor(token_ids, dtype=torch.long)
        # Trim to a multiple of block_size
        n_blocks = len(flat) // block_size
        self.data = flat[: n_blocks * block_size].view(n_blocks, block_size)
        logger.info(
            f"Dataset: {n_blocks:,} blocks × {block_size} tokens "
            f"= {n_blocks * block_size:,} total tokens"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        chunk = self.data[idx]
        x = chunk[:-1]   # (block_size - 1,) input
        y = chunk[1:]    # (block_size - 1,) target (shifted by 1)
        return x, y


def build_dataloaders(
    tokenizer,
    block_size: int = 256,
    batch_size: int = 64,
    val_fraction: float = 0.02,
    num_workers: int = 2,
    n_train_samples: Optional[int] = None,
) -> tuple:
    """
    Build train/val DataLoaders from TinyStories.

    Returns (train_loader, val_loader)
    """
    from .tokenizer import load_tinystories

    logger.info("Loading and tokenizing TinyStories...")
    texts = load_tinystories(n_samples=n_train_samples)

    # Encode all texts in batch
    encoded = tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=True,
    )

    # Flatten all token IDs into a single stream
    all_ids = []
    eos_id = tokenizer.eos_token_id
    for ids in encoded["input_ids"]:
        all_ids.extend(ids)
        all_ids.append(eos_id)  # Story separator

    # Split train / val
    split_point = int(len(all_ids) * (1 - val_fraction))
    train_ids = all_ids[:split_point]
    val_ids = all_ids[split_point:]

    train_ds = TextBlockDataset(train_ids, block_size)
    val_ds = TextBlockDataset(val_ids, block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(
        f"Dataloaders ready | train={len(train_loader)} batches | "
        f"val={len(val_loader)} batches"
    )
    return train_loader, val_loader
