"""
tests/test_tokenizer.py — Tokenizer + Dataset pipeline tests
"""
import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

_has_tokenizers   = importlib.util.find_spec("tokenizers")   is not None
_has_transformers = importlib.util.find_spec("transformers") is not None

requires_tokenizers = pytest.mark.skipif(
    not (_has_tokenizers and _has_transformers),
    reason="pip install tokenizers transformers",
)


class TestTokenizerTraining:

    @requires_tokenizers
    def test_train_tokenizer_produces_files(self, tmp_path):
        from minigpt_core.tokenizer.tokenizer import train_tokenizer
        texts = ["Once upon a time there was a tiny cat. " * 100]
        train_tokenizer(texts, vocab_size=200, save_dir=str(tmp_path / "tok"), min_frequency=1)
        assert (tmp_path / "tok" / "vocab.json").exists()
        assert (tmp_path / "tok" / "merges.txt").exists()

    @requires_tokenizers
    def test_special_tokens_present(self, tmp_path):
        from minigpt_core.tokenizer.tokenizer import train_tokenizer, load_tokenizer
        texts = ["hello world " * 200]
        train_tokenizer(texts, vocab_size=200, save_dir=str(tmp_path / "tok"), min_frequency=1)
        tok = load_tokenizer(str(tmp_path / "tok"))
        assert tok.bos_token == "<s>"
        assert tok.eos_token == "</s>"
        assert tok.pad_token == "<pad>"
        assert tok.unk_token == "<unk>"

    @requires_tokenizers
    def test_roundtrip_encoding(self, tmp_path):
        from minigpt_core.tokenizer.tokenizer import train_tokenizer, load_tokenizer
        texts = ["hello world this is a test " * 200]
        train_tokenizer(texts, vocab_size=300, save_dir=str(tmp_path / "tok"), min_frequency=1)
        tok = load_tokenizer(str(tmp_path / "tok"))
        original = "hello world"
        ids = tok.encode(original)
        recovered = tok.decode(ids, skip_special_tokens=True)
        assert original in recovered or recovered.strip() == original.strip()

    @requires_tokenizers
    def test_vocab_size_respected(self, tmp_path):
        from minigpt_core.tokenizer.tokenizer import train_tokenizer, load_tokenizer
        texts = ["the quick brown fox " * 500]
        train_tokenizer(texts, vocab_size=150, save_dir=str(tmp_path / "tok"), min_frequency=1)
        tok = load_tokenizer(str(tmp_path / "tok"))
        assert tok.vocab_size <= 160  # slack for special tokens


class TestTextBlockDataset:

    def test_basic_length(self):
        from minigpt_core.tokenizer.dataset import TextBlockDataset
        ids = list(range(1000))
        ds = TextBlockDataset(ids, block_size=32)
        assert len(ds) == 31

    def test_item_shape(self):
        from minigpt_core.tokenizer.dataset import TextBlockDataset
        ids = list(range(512))
        ds = TextBlockDataset(ids, block_size=64)
        x, y = ds[0]
        assert x.shape == (63,)
        assert y.shape == (63,)

    def test_target_is_shifted(self):
        from minigpt_core.tokenizer.dataset import TextBlockDataset
        ids = list(range(256))
        ds = TextBlockDataset(ids, block_size=16)
        x, y = ds[0]
        assert torch.all(y == x + 1)

    def test_all_tokens_covered(self):
        from minigpt_core.tokenizer.dataset import TextBlockDataset
        ids = list(range(100))
        ds = TextBlockDataset(ids, block_size=10)
        all_x = torch.cat([ds[i][0] for i in range(len(ds))])
        assert len(all_x) == 9 * (100 // 10)

    def test_torch_long_dtype(self):
        from minigpt_core.tokenizer.dataset import TextBlockDataset
        ds = TextBlockDataset(list(range(128)), block_size=32)
        x, y = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long


class TestDataLoader:

    def test_dataloader_batches_correctly(self):
        from minigpt_core.tokenizer.dataset import TextBlockDataset
        from torch.utils.data import DataLoader
        ids = list(range(1024))
        ds = TextBlockDataset(ids, block_size=32)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 31)
        assert batch_y.shape == (4, 31)
