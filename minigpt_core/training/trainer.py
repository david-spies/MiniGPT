"""
MiniGPT Training Loop
- AdamW + cosine LR schedule with linear warmup
- Gradient clipping
- Periodic validation + checkpoint saving
- Mixed precision (FP16) optional
"""
import os
import math
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR Schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training lifecycle for MiniGPT.

    Usage
    -----
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
    trainer.save("checkpoints/mini_gpt_best.pt")
    """

    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # Optimizer
        # Separate params: no weight decay on bias / LayerNorm
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Mixed precision
        self.use_amp = use_amp and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.global_step = 0
        self.best_val_loss = float("inf")

        total_steps = len(train_loader) * config.epochs
        self.max_steps = total_steps
        logger.info(
            f"Training: {config.epochs} epochs × {len(train_loader)} batches = {total_steps:,} steps"
        )

    def train(self) -> None:
        """Full training loop."""
        model = self.model
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Update LR
                lr = get_lr(
                    self.global_step,
                    self.config.warmup_steps,
                    self.max_steps,
                    self.config.learning_rate,
                    self.config.learning_rate * 0.1,
                )
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast():
                        logits, _ = model(x)
                        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits, _ = model(x)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

                epoch_loss += loss.item()
                self.global_step += 1

                if batch_idx % 200 == 0:
                    elapsed = time.time() - t0
                    tokens_per_sec = (
                        batch_idx * self.config.batch_size * (self.config.block_size - 1) / max(elapsed, 1e-6)
                    )
                    logger.info(
                        f"Epoch {epoch:02d} | step {self.global_step:06d} | "
                        f"loss {loss.item():.4f} | lr {lr:.2e} | "
                        f"{tokens_per_sec/1000:.1f}k tok/s"
                    )

            avg_loss = epoch_loss / len(self.train_loader)
            val_loss = self._evaluate()
            elapsed = time.time() - t0
            logger.info(
                f"── Epoch {epoch}/{self.config.epochs} | "
                f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
                f"time={elapsed:.0f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save(self.checkpoint_dir / "mini_gpt_best.pt")
                logger.info(f"✓ New best model saved (val_loss={val_loss:.4f})")

        self.save(self.checkpoint_dir / "mini_gpt_final.pt")
        logger.info("Training complete.")

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(self.val_loader)

    def save(self, path) -> None:
        path = Path(path)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        size_mb = path.stat().st_size / 1024**2
        logger.info(f"Checkpoint saved: {path} ({size_mb:.2f} MB)")

    @classmethod
    def load_checkpoint(cls, path: str, model) -> dict:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")
        return ckpt
