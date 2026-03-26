# MiniGPT Makefile — common development tasks
# Usage: make <target>

.PHONY: help install test test-all lint format size-check train export serve clean

PYTHON    := python
PIP       := pip
PORT      := 8080
CKPT      := checkpoints/mini_gpt_best.pt
ONNX      := web/assets/mini_gpt_quant.onnx

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ⚡ MiniGPT Make Targets"
	@echo "  ─────────────────────────────────────────────"
	@echo "  install      Install all dependencies"
	@echo "  test         Run core test suite (fast)"
	@echo "  test-all     Run full test suite with optional deps"
	@echo "  lint         Run ruff linter"
	@echo "  format       Auto-format with ruff"
	@echo "  size-check   Verify model is under 16MB"
	@echo "  train        Train model on TinyStories (full)"
	@echo "  train-fast   Train on 50k samples (quick test)"
	@echo "  export       Export trained model to ONNX + INT8"
	@echo "  serve        Start local dev server on :$(PORT)"
	@echo "  benchmark    Run inference benchmark"
	@echo "  clean        Remove generated artifacts"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt

install-gpu:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -e ".[all]"

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short \
		-k "not (tokenizer_train or ONNX_inference)"

test-all:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ --ignore E501,E402

format:
	ruff format src/ tests/

typecheck:
	mypy src/model/ --ignore-missing-imports

# ── Size Verification ─────────────────────────────────────────────────────────
size-check:
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from src.model import DEFAULT_CONFIG, MEDIUM_CONFIG; \
configs = [('DEFAULT', DEFAULT_CONFIG), ('MEDIUM', MEDIUM_CONFIG)]; \
[print(f'{n}: {c.estimated_params():,} params | INT8: {c.estimated_size_mb(8):.2f}MB | {\"✓\" if c.estimated_size_mb(8)<16 else \"✗\"}') for n, c in configs]"

# ── Training ──────────────────────────────────────────────────────────────────
train:
	$(PYTHON) main.py train

train-fast:
	$(PYTHON) main.py train --n-samples 50000

train-gpu:
	$(PYTHON) main.py train --amp

# ── Export ────────────────────────────────────────────────────────────────────
export: $(CKPT)
	$(PYTHON) main.py export --checkpoint $(CKPT) --output-dir web/assets
	@ls -lh web/assets/*.onnx

export-mobile: $(CKPT)
	$(PYTHON) scripts/export_mobile.py --checkpoint $(CKPT)

# ── Serve ─────────────────────────────────────────────────────────────────────
serve:
	$(PYTHON) scripts/serve.py --port $(PORT)

serve-root:
	$(PYTHON) scripts/serve.py --port $(PORT) --dir .

# ── Benchmark ─────────────────────────────────────────────────────────────────
benchmark:
	$(PYTHON) scripts/benchmark.py \
		--checkpoint $(CKPT) \
		--onnx $(ONNX) \
		--all

benchmark-onnx:
	$(PYTHON) scripts/benchmark.py --onnx $(ONNX)

# ── Generate ──────────────────────────────────────────────────────────────────
generate:
	$(PYTHON) main.py generate --checkpoint $(CKPT) \
		--prompt "Once upon a time, there was a tiny robot who"

info:
	$(PYTHON) main.py info

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
	@echo "Cleaned."

clean-all: clean
	rm -rf checkpoints/ mini_gpt_tokenizer/ web/assets/*.onnx 2>/dev/null || true
	@echo "All generated artifacts removed."
