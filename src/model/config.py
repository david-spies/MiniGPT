"""
MiniGPT Model Configuration
Optimized for <16MB deployment on browser/mobile.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MiniGPTConfig:
    # Vocabulary
    vocab_size: int = 5000          # Custom BPE; standard 50k = ~25MB embedding alone

    # Architecture
    n_embd: int = 128               # Embedding dimension (min for coherent English)
    n_head: int = 4                 # Attention heads (32 dims/head)
    n_layer: int = 4                # Transformer depth
    block_size: int = 256           # Context window (tokens)
    dropout: float = 0.1

    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    batch_size: int = 64
    epochs: int = 5
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # Deployment
    quantize: bool = True           # INT8 post-training quantization
    use_kv_cache: bool = True       # KV caching for O(1) incremental decode

    # Derived
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def ffn_dim(self) -> int:
        return 4 * self.n_embd

    def estimated_params(self) -> int:
        """Rough parameter count estimate."""
        emb = self.vocab_size * self.n_embd * 2          # wte + lm_head (tied)
        pos = self.block_size * self.n_embd               # wpe
        per_layer = (
            3 * self.n_embd * self.n_embd +              # QKV projection
            self.n_embd * self.n_embd +                  # output projection
            self.n_embd * self.ffn_dim +                 # FFN up
            self.ffn_dim * self.n_embd +                 # FFN down
            4 * self.n_embd                              # LayerNorm params x2
        )
        return emb + pos + self.n_layer * per_layer

    def estimated_size_mb(self, bits: int = 32) -> float:
        """Estimated file size in MB given quantization bit-width."""
        return (self.estimated_params() * bits / 8) / (1024 ** 2)

    def __repr__(self) -> str:
        params = self.estimated_params()
        fp32 = self.estimated_size_mb(32)
        int8 = self.estimated_size_mb(8)
        return (
            f"MiniGPTConfig(\n"
            f"  vocab={self.vocab_size}, embd={self.n_embd}, "
            f"heads={self.n_head}, layers={self.n_layer}, ctx={self.block_size}\n"
            f"  ~{params/1e6:.2f}M params | FP32: {fp32:.1f}MB | INT8: {int8:.1f}MB\n"
            f")"
        )


# Default production config — verified <16MB
DEFAULT_CONFIG = MiniGPTConfig()

# Larger config that still fits under 16MB with INT8
MEDIUM_CONFIG = MiniGPTConfig(
    vocab_size=8000,
    n_embd=256,
    n_head=8,
    n_layer=6,
    block_size=256,
)
