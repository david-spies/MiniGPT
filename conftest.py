"""
conftest.py — Shared pytest fixtures for MiniGPT test suite.
"""
import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def tiny_config():
    """Minimal config for fast unit tests (no disk/network I/O)."""
    from minigpt_core.model.config import MiniGPTConfig
    return MiniGPTConfig(
        vocab_size=100,
        n_embd=32,
        n_head=2,
        n_layer=2,
        block_size=64,
        dropout=0.0,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    """Pre-built tiny model shared across the session."""
    from minigpt_core.model.model import MiniGPT
    model = MiniGPT(tiny_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def default_config():
    from minigpt_core.model.config import DEFAULT_CONFIG
    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def default_model(default_config):
    from minigpt_core.model.model import MiniGPT
    model = MiniGPT(default_config)
    model.eval()
    return model


@pytest.fixture
def random_ids(tiny_config):
    """Random token tensor for shape testing."""
    return torch.randint(0, tiny_config.vocab_size, (1, 16))
