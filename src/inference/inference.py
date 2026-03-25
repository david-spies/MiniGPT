"""
MiniGPT Python Inference
- KV-cached autoregressive generation
- ONNX Runtime inference (matches browser behavior)
- Benchmarking utilities
"""
import time
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch Native Inference (for development / testing)
# ---------------------------------------------------------------------------

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cpu",
) -> str:
    """
    Generate text using PyTorch model with KV-cache.
    """
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
        )

    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)


def benchmark_pytorch(
    model,
    tokenizer,
    prompt: str = "Once upon a time,",
    n_tokens: int = 50,
    n_runs: int = 3,
    device: str = "cpu",
) -> dict:
    """
    Measure tokens-per-second for PyTorch model.
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        generate(model, tokenizer, prompt, max_new_tokens=n_tokens, device=device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    tps = n_tokens / avg
    logger.info(f"PyTorch | {tps:.1f} tok/s | avg {avg:.3f}s over {n_runs} runs")
    return {"tps": tps, "avg_s": avg, "runs": n_runs}


# ---------------------------------------------------------------------------
# ONNX Runtime Inference (mirrors browser execution)
# ---------------------------------------------------------------------------

class OnnxInferenceSession:
    """
    ONNX Runtime inference session with KV-cache management.
    Matches the JavaScript inference logic exactly.
    """

    def __init__(self, model_path: str, providers: Optional[list] = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=session_options, providers=providers
        )

        # Introspect model to determine n_layers
        input_names = [i.name for i in self.session.get_inputs()]
        self.n_layer = (len(input_names) - 1) // 2
        logger.info(f"ONNX session loaded | layers={self.n_layer} | path={model_path}")

    def _init_empty_cache(self, n_head: int, head_dim: int):
        """Return zero-size past_kv tensors (shape: [1, n_head, 0, head_dim])."""
        return [
            (
                np.zeros((1, n_head, 0, head_dim), dtype=np.float32),
                np.zeros((1, n_head, 0, head_dim), dtype=np.float32),
            )
            for _ in range(self.n_layer)
        ]

    def run_step(
        self,
        token_ids: List[int],
        past_kvs: Optional[list],
        n_head: int = 4,
        head_dim: int = 32,
    ) -> tuple:
        """
        Run one forward pass.

        Parameters
        ----------
        token_ids : list of int token IDs for this step
        past_kvs  : list of (k, v) numpy arrays, or None for first step
        n_head    : model n_head
        head_dim  : model head_dim

        Returns
        -------
        (logits, new_past_kvs)
        """
        if past_kvs is None:
            past_kvs = self._init_empty_cache(n_head, head_dim)

        feeds = {
            "input_ids": np.array([token_ids], dtype=np.int64),
        }
        for i, (k, v) in enumerate(past_kvs):
            feeds[f"past_k_{i}"] = k
            feeds[f"past_v_{i}"] = v

        outputs = self.session.run(None, feeds)

        logits = outputs[0]  # (1, T, vocab)

        # Reconstruct new KV cache
        new_kvs = []
        for i in range(self.n_layer):
            k = outputs[1 + i * 2]
            v = outputs[2 + i * 2]
            new_kvs.append((k, v))

        return logits, new_kvs

    def generate(
        self,
        token_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
        n_head: int = 4,
        head_dim: int = 32,
    ) -> List[int]:
        """
        Full KV-cached generation loop via ONNX Runtime.
        """
        # Process prompt
        logits, past_kvs = self.run_step(token_ids, None, n_head, head_dim)

        output_ids = list(token_ids)

        for _ in range(max_new_tokens):
            last_logits = logits[0, -1, :].astype(np.float64)
            last_logits /= temperature

            # Top-k filtering
            if top_k:
                top_indices = np.argpartition(last_logits, -top_k)[-top_k:]
                mask = np.ones(len(last_logits), dtype=bool)
                mask[top_indices] = False
                last_logits[mask] = -np.inf

            # Softmax + sample
            logits_max = np.max(last_logits)
            exp_l = np.exp(last_logits - logits_max)
            probs = exp_l / exp_l.sum()
            next_token = int(np.random.choice(len(probs), p=probs))

            output_ids.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

            logits, past_kvs = self.run_step([next_token], past_kvs, n_head, head_dim)

        return output_ids


def benchmark_onnx(
    session: OnnxInferenceSession,
    prompt_tokens: List[int],
    n_tokens: int = 50,
    n_runs: int = 3,
    n_head: int = 4,
    head_dim: int = 32,
) -> dict:
    """Measure ONNX inference speed."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.generate(prompt_tokens, max_new_tokens=n_tokens, n_head=n_head, head_dim=head_dim)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    tps = n_tokens / avg
    logger.info(f"ONNX | {tps:.1f} tok/s | avg {avg:.3f}s over {n_runs} runs")
    return {"tps": tps, "avg_s": avg, "runs": n_runs}
