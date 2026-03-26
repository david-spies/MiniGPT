"""
tests/test_training.py — Trainer integration tests
"""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_loader(config, n_samples=32, batch_size=4):
    """Build a tiny synthetic DataLoader."""
    x = torch.randint(0, config.vocab_size, (n_samples, config.block_size))
    inputs = x[:, :-1]
    targets = x[:, 1:]
    ds = TensorDataset(inputs, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


class TestTrainer:

    def test_trainer_runs_one_epoch(self, tmp_path, tiny_config):
        from minigpt_core.model.model import MiniGPT
        from minigpt_core.training.trainer import Trainer

        tiny_config.epochs = 1
        tiny_config.batch_size = 4
        tiny_config.warmup_steps = 2

        model = MiniGPT(tiny_config)
        loader = _make_loader(tiny_config, n_samples=16, batch_size=4)

        trainer = Trainer(
            model, tiny_config, loader, loader,
            checkpoint_dir=str(tmp_path / "ckpts"),
            use_amp=False,
            device="cpu",
        )
        trainer.train()

        # Checkpoint should exist
        assert (tmp_path / "ckpts" / "mini_gpt_best.pt").exists()
        assert (tmp_path / "ckpts" / "mini_gpt_final.pt").exists()

    def test_checkpoint_loads_correctly(self, tmp_path, tiny_config):
        from minigpt_core.model.model import MiniGPT
        from minigpt_core.training.trainer import Trainer

        tiny_config.epochs = 1
        tiny_config.warmup_steps = 1

        model = MiniGPT(tiny_config)
        loader = _make_loader(tiny_config, n_samples=8, batch_size=4)

        trainer = Trainer(
            model, tiny_config, loader, loader,
            checkpoint_dir=str(tmp_path / "ckpts"),
            use_amp=False,
            device="cpu",
        )
        trainer.train()

        # Load checkpoint back into a fresh model
        ckpt_path = str(tmp_path / "ckpts" / "mini_gpt_best.pt")
        fresh_model = MiniGPT(tiny_config)
        ckpt = Trainer.load_checkpoint(ckpt_path, fresh_model)

        assert "global_step" in ckpt
        assert "best_val_loss" in ckpt
        assert ckpt["best_val_loss"] < float("inf")

    def test_lr_schedule_warmup(self):
        from minigpt_core.training.trainer import get_lr

        lr = 5e-4
        # During warmup, LR should grow linearly
        lr_at_0 = get_lr(0, warmup_steps=100, max_steps=1000, lr=lr, min_lr=lr * 0.1)
        lr_at_50 = get_lr(50, warmup_steps=100, max_steps=1000, lr=lr, min_lr=lr * 0.1)
        lr_at_100 = get_lr(100, warmup_steps=100, max_steps=1000, lr=lr, min_lr=lr * 0.1)

        assert lr_at_0 < lr_at_50 < lr_at_100
        assert abs(lr_at_100 - lr) < 1e-8  # Should equal peak LR at end of warmup

    def test_lr_schedule_decay(self):
        from minigpt_core.training.trainer import get_lr

        lr = 5e-4
        min_lr = 5e-5
        # After warmup, LR should decay toward min_lr
        lr_start = get_lr(100, warmup_steps=100, max_steps=1000, lr=lr, min_lr=min_lr)
        lr_end = get_lr(999, warmup_steps=100, max_steps=1000, lr=lr, min_lr=min_lr)

        assert lr_start > lr_end
        assert lr_end >= min_lr - 1e-8

    def test_val_loss_is_finite(self, tmp_path, tiny_config):
        from minigpt_core.model.model import MiniGPT
        from minigpt_core.training.trainer import Trainer

        tiny_config.epochs = 1
        tiny_config.warmup_steps = 1

        model = MiniGPT(tiny_config)
        loader = _make_loader(tiny_config, n_samples=8, batch_size=4)

        trainer = Trainer(
            model, tiny_config, loader, loader,
            checkpoint_dir=str(tmp_path / "ckpts"),
            use_amp=False, device="cpu",
        )
        val_loss = trainer._evaluate()
        assert val_loss > 0
        assert val_loss < 1e6
        assert val_loss == val_loss  # not NaN
