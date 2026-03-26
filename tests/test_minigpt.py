"""
MiniGPT Test Suite
Run: pytest tests/ -v
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from minigpt_core.model import MiniGPT, MiniGPTConfig, DEFAULT_CONFIG


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return MiniGPTConfig(
        vocab_size=100,
        n_embd=32,
        n_head=2,
        n_layer=2,
        block_size=64,
    )

@pytest.fixture
def model(config):
    m = MiniGPT(config)
    m.eval()
    return m


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestModel:

    def test_param_count(self, model, config):
        params = model.count_parameters()
        assert params > 0, "Model has no parameters"
        print(f"\n  Parameters: {params:,}")

    def test_output_shape(self, model, config):
        B, T = 2, 16
        ids = torch.randint(0, config.vocab_size, (B, T))
        logits, kv = model(ids)
        assert logits.shape == (B, T, config.vocab_size), f"Bad shape: {logits.shape}"
        assert len(kv) == config.n_layer, f"Expected {config.n_layer} KV pairs, got {len(kv)}"

    def test_kv_cache_shapes(self, model, config):
        B, T = 1, 8
        ids = torch.randint(0, config.vocab_size, (B, T))
        _, kv = model(ids)
        for i, (k, v) in enumerate(kv):
            assert k.shape == (B, config.n_head, T, config.head_dim), \
                f"Layer {i} K shape wrong: {k.shape}"
            assert v.shape == (B, config.n_head, T, config.head_dim), \
                f"Layer {i} V shape wrong: {v.shape}"

    def test_kv_cache_consistency(self, model, config):
        """KV-cached incremental decode should match full-sequence decode."""
        ids = torch.randint(0, config.vocab_size, (1, 8))

        # Full sequence decode
        with torch.no_grad():
            logits_full, _ = model(ids)

        # Incremental decode with cache
        with torch.no_grad():
            logits_step, kv = model(ids[:, :4])  # Process first half
            logits_cont, _ = model(ids[:, 4:], past_key_values=kv)  # Continue

        # Last 4 positions should match
        assert torch.allclose(
            logits_full[:, 4:, :],
            logits_cont,
            atol=1e-4,
        ), "KV-cached decode diverges from full decode"

    def test_weight_tying(self, model):
        """wte and lm_head must share the same tensor."""
        assert model.transformer.wte.weight is model.lm_head.weight, \
            "Weight tying broken — lm_head and wte are separate tensors"

    def test_generate(self, model, config):
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10)
        assert output.shape[1] > prompt.shape[1], "generate() produced no new tokens"
        assert output.shape[1] <= prompt.shape[1] + 10

    def test_default_config_under_16mb(self):
        cfg = DEFAULT_CONFIG
        size_int8 = cfg.estimated_size_mb(8)
        assert size_int8 < 16.0, \
            f"Default INT8 config exceeds 16MB: {size_int8:.2f}MB"

    def test_no_nan_in_forward(self, model, config):
        ids = torch.randint(0, config.vocab_size, (1, 32))
        logits, _ = model(ids)
        assert not torch.isnan(logits).any(), "NaN in model output"
        assert not torch.isinf(logits).any(), "Inf in model output"


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:

    def test_estimated_params_reasonable(self):
        cfg = MiniGPTConfig()
        params = cfg.estimated_params()
        # Weight-tied model: wte shared with lm_head, ~1.4–2.2M expected range
        assert 1_000_000 < params < 3_000_000, f"Unexpected param count: {params:,}"

    def test_size_estimates_ordered(self):
        cfg = MiniGPTConfig()
        fp32 = cfg.estimated_size_mb(32)
        fp16 = cfg.estimated_size_mb(16)
        int8 = cfg.estimated_size_mb(8)
        assert fp32 > fp16 > int8 > 0

    def test_head_dim_property(self):
        cfg = MiniGPTConfig(n_embd=128, n_head=4)
        assert cfg.head_dim == 32

    def test_invalid_head_count_raises(self):
        with pytest.raises(AssertionError):
            cfg = MiniGPTConfig(n_embd=128, n_head=3)
            model = MiniGPT(cfg)
            model(torch.zeros(1, 4, dtype=torch.long))


# ── ONNX Export Tests ─────────────────────────────────────────────────────────

class TestONNXExport:

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed"
    )
    def test_export_produces_file(self, tmp_path, model, config):
        from minigpt_core.export import export_onnx
        out = str(tmp_path / "test_model.onnx")
        path = export_onnx(model, config, out)
        assert Path(path).exists(), "ONNX file not created"
        size = Path(path).stat().st_size
        assert size > 1000, "ONNX file suspiciously small"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnxruntime") or
        not __import__("importlib").util.find_spec("onnxscript"),
        reason="onnxruntime or onnxscript not installed"
    )
    def test_onnx_inference_matches_pytorch(self, tmp_path, model, config):
        import onnxruntime as ort
        from minigpt_core.export import export_onnx

        model.eval()
        out = str(tmp_path / "test_model.onnx")
        export_onnx(model, config, out)

        session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        ids = torch.randint(0, config.vocab_size, (1, 4))

        # Pytorch
        with torch.no_grad():
            pt_logits, _ = model(ids)

        # ONNX (first step, empty cache)
        empty_kv = {}
        for i in range(config.n_layer):
            shape = [1, config.n_head, 0, config.head_dim]
            empty_kv[f"past_k_{i}"] = np.zeros(shape, dtype=np.float32)
            empty_kv[f"past_v_{i}"] = np.zeros(shape, dtype=np.float32)

        feeds = {"input_ids": ids.numpy().astype(np.int64), **empty_kv}
        outputs = session.run(None, feeds)
        onnx_logits = outputs[0]

        np.testing.assert_allclose(
            pt_logits.numpy(), onnx_logits, atol=1e-4,
            err_msg="ONNX logits diverge from PyTorch"
        )


# ── Training Sanity Tests ─────────────────────────────────────────────────────

class TestTraining:

    def test_loss_decreases(self, model, config):
        """Model should be able to overfit a tiny batch."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randint(0, config.vocab_size, (4, 16))
        y = torch.randint(0, config.vocab_size, (4, 15))

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            logits, _ = model(x[:, :-1])
            loss = criterion(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
