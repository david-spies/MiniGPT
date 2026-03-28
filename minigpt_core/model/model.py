"""
MiniGPT Model — Production Architecture
- CausalSelfAttention with KV-Cache support
- Pre-norm GPT-style Transformer blocks
- Weight tying (wte == lm_head) saves ~2.5MB
- ONNX-exportable forward signatures
"""
import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniGPTConfig

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]  # [(k, v), ...] per layer


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional KV-cache.

    During training  : pass full sequence, past_key_value=None.
    During inference : pass single new token + past_key_value from previous step.

    Returns
    -------
    y                : (B, T, C) output hidden states
    present_kv       : (k, v) tensors to be fed back as past_key_value next step
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Fused QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask — registered as buffer (not a parameter)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()

        # Compute Q, K, V for current input
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV-Cache: concatenate past keys/values
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # (B, heads, past+T, head_dim)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)  # Store for next decode step

        T_total = k.size(2)  # Full sequence length including past

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T_total)

        # Apply causal mask — only for the new positions
        att = att.masked_fill(
            self.bias[:, :, T_total - T : T_total, :T_total] == 0,
            float("-inf"),
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads
        y = self.resid_drop(self.c_proj(y))

        return y, present_kv


# ---------------------------------------------------------------------------
# Feed-Forward
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.fc   = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.proj = nn.Linear(config.ffn_dim, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Pre-LayerNorm GPT block."""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present_kv = self.attn(self.ln_1(x), past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


# ---------------------------------------------------------------------------
# MiniGPT
# ---------------------------------------------------------------------------

class MiniGPT(nn.Module):
    """
    Full GPT-style language model under 16MB.

    Parameters
    ----------
    config : MiniGPTConfig

    Forward inputs
    --------------
    idx             : (B, T) token IDs
    past_key_values : list of (k, v) per layer — None on first call

    Forward outputs
    ---------------
    logits          : (B, T, vocab_size)
    new_key_values  : list of (k, v) per layer for next step
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        # Shared weight: lm_head tied to wte — saves ~2.5MB for vocab=5000
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)
        # GPT-2 style: scale residual projections by 1/sqrt(n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        past_key_values: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        B, T = idx.size()

        # Position offset accounts for previously-cached tokens
        past_length = past_key_values[0][0].size(2) if past_key_values is not None else 0
        if past_length + T > self.config.block_size: raise ValueError(f"Sequence too long: {past_length + T} > {self.config.block_size}")

        # Embeddings
        pos = torch.arange(past_length, past_length + T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)           # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)           # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        new_key_values: KVCache = []
        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, layer_past)
            new_key_values.append(present)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, new_key_values

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache and top-k sampling.
        Speed: O(1) per token (vs O(n²) without cache).
        """
        self.eval()
        past_kv: Optional[KVCache] = None

        # Process full prompt, build cache
        logits, past_kv = self.forward(idx, None)

        for _ in range(max_new_tokens):
            # Only pass the last generated token + accumulated cache
            logits_last = logits[:, -1, :] / temperature  # (B, vocab)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat([idx, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            # Incremental decode — single token, re-uses cache
            logits, past_kv = self.forward(next_token, past_kv)

        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return f"MiniGPT | {params:,} params | config: {self.config}"
