"""Unit tests for the central `Vocabulary` mapping (Phase 4, Task 4.1)."""

import pytest

from vocabulary.vocab import Vocabulary


def test_add_token_assigns_deterministic_incremental_ids():
    vocab = Vocabulary()

    assert vocab.add_token("hello") == 0
    assert vocab.add_token("world") == 1


def test_add_token_is_idempotent_for_duplicates():
    vocab = Vocabulary()

    first_id = vocab.add_token("hello")
    second_id = vocab.add_token("hello")

    assert first_id == second_id
    assert vocab.vocab_size == 1


def test_get_id_and_get_token_roundtrip():
    vocab = Vocabulary()
    token_id = vocab.add_token("hello")

    assert vocab.get_id("hello") == token_id
    assert vocab.get_token(token_id) == "hello"


def test_get_id_raises_for_unknown_token():
    vocab = Vocabulary()

    with pytest.raises(KeyError):
        vocab.get_id("missing")


def test_get_token_raises_for_unknown_id():
    vocab = Vocabulary()

    with pytest.raises(KeyError):
        vocab.get_token(42)


def test_has_token_and_has_id():
    vocab = Vocabulary()
    token_id = vocab.add_token("hello")

    assert vocab.has_token("hello") is True
    assert vocab.has_token("missing") is False
    assert vocab.has_id(token_id) is True
    assert vocab.has_id(999) is False


def test_contains_operator_matches_has_token():
    vocab = Vocabulary()
    vocab.add_token("hello")

    assert "hello" in vocab
    assert "missing" not in vocab


def test_vocab_size_and_len_match_number_of_distinct_tokens():
    vocab = Vocabulary()
    vocab.add_token("hello")
    vocab.add_token("world")
    vocab.add_token("hello")  # duplicate, must not be counted twice

    assert vocab.vocab_size == 2
    assert len(vocab) == 2


def test_constructor_accepts_initial_tokens_in_order():
    vocab = Vocabulary(["a", "b", "c"])

    assert vocab.get_id("a") == 0
    assert vocab.get_id("b") == 1
    assert vocab.get_id("c") == 2
    assert vocab.vocab_size == 3
