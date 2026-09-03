"""Unit tests for the SentencePiece adapter (Phase 7, Task 7.3)."""

import pytest

from benchmarking.comparator import compare_tokenizers
from experiments.dataset_loader import load_dataset
from tokenizers.adapters.sentencepiece_tokenizer import SentencePieceAdapter
from tokenizers.base import Tokenizer

# Needs enough lexical diversity for SentencePiece's Unigram trainer to
# support a non-trivial vocab_size -- unlike this project's own BPE/
# WordPiece trainers, SentencePiece's trainer *raises* if `vocab_size` is
# too large for the corpus, rather than silently capping it (see
# `test_train_raises_when_vocab_size_is_too_large_for_the_corpus` below).
_CORPUS = [
    "low low low low low",
    "lower lower",
    "lowest",
    "hello world",
    "unbelievable",
    "the quick brown fox",
] * 5


def _trained(vocab_size: int = 30) -> SentencePieceAdapter:
    tokenizer = SentencePieceAdapter(vocab_size=vocab_size)
    tokenizer.train(_CORPUS)
    return tokenizer


# ---------------------------------------------------------------------------
# construction / contract basics
# ---------------------------------------------------------------------------


def test_is_a_tokenizer_subclass():
    assert issubclass(SentencePieceAdapter, Tokenizer)


def test_name_includes_the_model_type():
    assert SentencePieceAdapter().name == "sentencepiece:unigram"
    assert SentencePieceAdapter(model_type="bpe").name == "sentencepiece:bpe"


def test_using_an_untrained_adapter_raises_a_clear_error():
    adapter = SentencePieceAdapter()

    with pytest.raises(RuntimeError):
        adapter.encode("hello")
    with pytest.raises(RuntimeError):
        adapter.vocab_size  # noqa: B018 (property access is the point of the test)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def test_train_actually_trains_a_model_not_a_no_op():
    # Unlike the pretrained-only HF/tiktoken adapters, this one must train
    # for real: an untrained instance cannot encode, a trained one can.
    tokenizer = SentencePieceAdapter(vocab_size=30)

    with pytest.raises(RuntimeError):
        tokenizer.encode("low")

    tokenizer.train(_CORPUS)

    assert tokenizer.encode("low") != []


def test_train_produces_the_requested_vocab_size():
    tokenizer = _trained(vocab_size=30)

    assert tokenizer.vocab_size == 30


def test_train_is_deterministic():
    first = _trained(vocab_size=30)
    second = _trained(vocab_size=30)

    assert first.encode("lower hello world") == second.encode("lower hello world")


def test_train_raises_when_vocab_size_is_too_large_for_the_corpus():
    tokenizer = SentencePieceAdapter(vocab_size=100000)

    with pytest.raises(RuntimeError):
        tokenizer.train(_CORPUS)


def test_train_raises_on_an_empty_corpus():
    tokenizer = SentencePieceAdapter(vocab_size=30)

    with pytest.raises(RuntimeError):
        tokenizer.train([])


def test_retraining_replaces_the_previous_model():
    # Two real, unrelated datasets, both comfortably supporting the same
    # vocab_size -- unlike the tiny hand-written corpora used elsewhere
    # here, which each need a carefully-picked vocab_size of their own.
    en_text, _ = load_dataset("en")
    ro_text, _ = load_dataset("ro")
    tokenizer = SentencePieceAdapter(vocab_size=100)

    tokenizer.train([en_text])
    ids_after_en = tokenizer.encode("lower hello")

    tokenizer.train([ro_text])
    ids_after_ro = tokenizer.encode("lower hello")

    assert tokenizer.vocab_size == 100  # same target, freshly retrained
    assert ids_after_ro != ids_after_en


# ---------------------------------------------------------------------------
# tokenize / encode / decode
# ---------------------------------------------------------------------------


def test_tokenize_marks_word_starts_with_the_sentencepiece_space_marker():
    tokenizer = _trained()

    tokens = tokenizer.tokenize("lower hello")

    assert any(token.startswith("▁") for token in tokens)  # "▁"


def test_tokenize_and_encode_stay_aligned_index_for_index():
    tokenizer = _trained()

    tokens = tokenizer.tokenize("lower hello world")
    ids = tokenizer.encode("lower hello world")

    assert len(tokens) == len(ids)


def test_encode_returns_integer_ids():
    tokenizer = _trained()

    ids = tokenizer.encode("lower hello")

    assert all(isinstance(i, int) for i in ids)


def test_decode_uses_sentencepieces_own_decode():
    tokenizer = _trained()
    ids = tokenizer.encode("lower hello world")

    assert tokenizer.decode(ids) == "lower hello world"


def test_encode_decode_roundtrip_for_corpus_words():
    tokenizer = _trained()

    for word in ("low", "lower", "lowest", "hello", "world"):
        assert tokenizer.decode(tokenizer.encode(word)) == word


def test_encode_handles_unicode_text_without_raising():
    tokenizer = _trained()

    ids = tokenizer.encode("héllo wörld こんにちは")

    assert isinstance(ids, list)
    assert isinstance(tokenizer.decode(ids), str)


def test_encode_handles_punctuation():
    tokenizer = _trained()

    ids = tokenizer.encode("hello, world!")

    assert isinstance(ids, list)
    assert len(ids) > 0


def test_encode_empty_text():
    tokenizer = _trained()

    assert tokenizer.encode("") == []
    assert tokenizer.tokenize("") == []


def test_encode_whitespace_only_text():
    tokenizer = _trained()

    ids = tokenizer.encode("   ")

    assert isinstance(ids, list)


def test_decode_of_empty_id_list_is_empty_string():
    assert _trained().decode([]) == ""


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    tokenizer = _trained(vocab_size=30)
    path = tmp_path / "sentencepiece.model"
    tokenizer.save(path)

    loaded = SentencePieceAdapter()
    loaded.load(path)

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("lower hello world") == tokenizer.encode("lower hello world")
    assert loaded.decode(loaded.encode("lower hello")) == tokenizer.decode(
        tokenizer.encode("lower hello")
    )


# ---------------------------------------------------------------------------
# Comparator integration (Task 5.3) -- no Comparator changes required
# ---------------------------------------------------------------------------


def test_works_with_the_comparator_without_any_comparator_changes():
    from tokenizers.character_tokenizer import CharacterTokenizer

    sp_adapter = _trained()
    character_tokenizer = CharacterTokenizer()
    character_tokenizer.train(["lower hello"])

    results = compare_tokenizers([character_tokenizer, sp_adapter], "lower hello")

    assert set(results["tokenizer"]) == {"character", "sentencepiece:unigram"}
    assert (results["vocab_size"] > 0).all()
