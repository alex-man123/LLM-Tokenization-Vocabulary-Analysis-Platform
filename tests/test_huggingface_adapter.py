"""Unit tests for the Hugging Face `tokenizers` adapter (Phase 7, Task 7.1).

Most tests build a real `tokenizers.Tokenizer` (the actual pip-installed
library, trained on a tiny local corpus) so the suite stays fast and
network-independent. One test (`test_from_pretrained_loads_a_real_hub_tokenizer`)
explicitly exercises `from_pretrained` against a real Hugging Face Hub
model, proving that path works end-to-end; it requires network access on
its first run (the file is cached locally afterward by `huggingface_hub`).
"""

import pytest

from benchmarking.comparator import compare_tokenizers
from tokenizers.adapters.huggingface_tokenizer import (
    HFTokenizer,
    HuggingFaceTokenizerAdapter,
    hf_models,
    hf_pre_tokenizers,
    hf_trainers,
)

_CORPUS = ["low low low low low", "lower lower", "lowest", "hello world", "unbelievable"]


def _train_local_hf_tokenizer() -> HFTokenizer:
    """Build and train a real `tokenizers.Tokenizer` (BPE) entirely offline.

    Uses the real library's submodules re-exported by the adapter module
    (`hf_models`/`hf_trainers`/`hf_pre_tokenizers`) rather than
    `from tokenizers.models import ...` directly, which would hit this
    project's own same-named `tokenizers` package instead of the pip-
    installed library (see `huggingface_tokenizer`'s module docstring).
    """
    tokenizer = HFTokenizer(hf_models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = hf_pre_tokenizers.Whitespace()
    trainer = hf_trainers.BpeTrainer(vocab_size=100, special_tokens=["[UNK]", "[PAD]"])
    tokenizer.train_from_iterator(_CORPUS, trainer)
    return tokenizer


def _adapter() -> HuggingFaceTokenizerAdapter:
    return HuggingFaceTokenizerAdapter(_train_local_hf_tokenizer(), name="huggingface:test-bpe")


# ---------------------------------------------------------------------------
# construction / contract basics
# ---------------------------------------------------------------------------


def test_is_a_tokenizer_subclass():
    from tokenizers.base import Tokenizer

    assert issubclass(HuggingFaceTokenizerAdapter, Tokenizer)


def test_name_reflects_the_constructor_argument():
    assert _adapter().name == "huggingface:test-bpe"


def test_default_name_when_not_given():
    adapter = HuggingFaceTokenizerAdapter(_train_local_hf_tokenizer())

    assert adapter.name == "huggingface"


def test_train_is_a_no_op_and_does_not_raise():
    adapter = _adapter()
    ids_before = adapter.encode("lower hello")

    adapter.train(["some", "unrelated", "corpus"])

    assert adapter.encode("lower hello") == ids_before


def test_using_an_adapter_with_no_tokenizer_loaded_raises_a_clear_error():
    adapter = HuggingFaceTokenizerAdapter()

    with pytest.raises(RuntimeError):
        adapter.encode("hello")
    with pytest.raises(RuntimeError):
        adapter.vocab_size  # noqa: B018 (property access is the point of the test)


# ---------------------------------------------------------------------------
# tokenize / encode / decode / vocab_size
# ---------------------------------------------------------------------------


def test_tokenize_returns_the_real_hf_tokens():
    adapter = _adapter()

    tokens = adapter.tokenize("lower hello")

    assert tokens == adapter._tokenizer.encode("lower hello").tokens


def test_encode_returns_the_real_hf_token_ids():
    adapter = _adapter()

    ids = adapter.encode("lower hello")

    assert ids == adapter._tokenizer.encode("lower hello").ids
    assert all(isinstance(i, int) for i in ids)


def test_encode_and_tokenize_stay_aligned_index_for_index():
    adapter = _adapter()

    tokens = adapter.tokenize("lower hello")
    ids = adapter.encode("lower hello")

    assert len(tokens) == len(ids)


def test_decode_uses_the_real_hf_decode_not_manual_join():
    adapter = _adapter()

    ids = adapter.encode("lower hello")

    assert adapter.decode(ids) == adapter._tokenizer.decode(ids)


def test_vocab_size_is_the_real_hf_vocab_size_not_derived_from_input_text():
    adapter = _adapter()

    assert adapter.vocab_size == adapter._tokenizer.get_vocab_size()
    # Vocab size must not depend on how much text is tokenized afterward.
    adapter.encode("a")
    assert adapter.vocab_size == adapter._tokenizer.get_vocab_size()


def test_encode_ascii_text():
    adapter = _adapter()

    ids = adapter.encode("hello world")

    assert len(ids) > 0


def test_encode_unicode_text_does_not_raise():
    adapter = _adapter()

    ids = adapter.encode("héllo wörld こんにちは")

    assert isinstance(ids, list)
    assert adapter.decode(ids)  # decodes without raising


def test_encode_empty_text():
    adapter = _adapter()

    assert adapter.encode("") == []
    assert adapter.tokenize("") == []


def test_encode_multiple_words():
    adapter = _adapter()

    ids = adapter.encode("hello world lower lowest")

    assert len(ids) >= 4


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    adapter = _adapter()
    path = tmp_path / "hf_tokenizer.json"
    adapter.save(path)

    loaded = HuggingFaceTokenizerAdapter()
    loaded.load(path)

    assert loaded.vocab_size == adapter.vocab_size
    assert loaded.encode("lower hello") == adapter.encode("lower hello")


def test_from_file_classmethod_loads_a_saved_tokenizer(tmp_path):
    adapter = _adapter()
    path = tmp_path / "hf_tokenizer.json"
    adapter.save(path)

    loaded = HuggingFaceTokenizerAdapter.from_file(path, name="huggingface:reloaded")

    assert loaded.name == "huggingface:reloaded"
    assert loaded.encode("lower hello") == adapter.encode("lower hello")


# ---------------------------------------------------------------------------
# Comparator integration (Task 5.3) — no Comparator changes required
# ---------------------------------------------------------------------------


def test_works_with_the_comparator_without_any_comparator_changes():
    from tokenizers.character_tokenizer import CharacterTokenizer

    hf_adapter = _adapter()
    character_tokenizer = CharacterTokenizer()
    character_tokenizer.train(["lower hello"])

    results = compare_tokenizers([character_tokenizer, hf_adapter], "lower hello")

    assert set(results["tokenizer"]) == {"character", "huggingface:test-bpe"}
    assert (results["vocab_size"] > 0).all()


# ---------------------------------------------------------------------------
# a real pretrained Hugging Face Hub tokenizer (network access, if available)
# ---------------------------------------------------------------------------


def test_from_pretrained_loads_a_real_hub_tokenizer():
    """Proves `from_pretrained` works end-to-end against a real Hub-hosted tokenizer.

    Every other test in this file trains a real `tokenizers.Tokenizer`
    locally, so the suite stays fast and deterministic without network
    access. This one test additionally exercises the Hub-download path —
    on first run it fetches `bert-base-uncased`'s `tokenizer.json`
    (cached locally by `huggingface_hub` afterward, so later runs are
    fast). If the Hub is unreachable (offline environment, no cached
    copy), that is an environment limitation, not a defect in this
    adapter, so the test skips instead of failing.
    """
    try:
        adapter = HuggingFaceTokenizerAdapter.from_pretrained("bert-base-uncased")
    except Exception as exc:  # network/cache failure -> skip, not a real test failure
        pytest.skip(f"Hugging Face Hub unreachable and no cached copy available: {exc}")

    assert adapter.vocab_size == 30522  # bert-base-uncased's published vocab size
    assert adapter.name == "huggingface:bert-base-uncased"
    ids = adapter.encode("Hello, world!")
    assert len(ids) > 0
    assert isinstance(adapter.decode(ids), str)
