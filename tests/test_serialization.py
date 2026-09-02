"""Tests for the generalized tokenizer serialization (Phase 4, Task 4.4)."""

import json

import pytest

from tokenizers.bpe.tokenizer import BPETokenizer
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer
from tokenizers.wordpiece.tokenizer import WordPieceTokenizer
from vocabulary.serialization import vocabulary_tokens_in_order
from vocabulary.vocab import Vocabulary

TOKENIZER_CLASSES = [CharacterTokenizer, WordTokenizer, BPETokenizer, WordPieceTokenizer]


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_save_produces_a_versioned_json_file_with_metadata(tmp_path, tokenizer_cls):
    tokenizer = tokenizer_cls()
    tokenizer.train(["hello world"])
    path = tmp_path / "state.json"

    tokenizer.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["tokenizer_type"] == tokenizer.name
    assert payload["vocab_size"] == tokenizer.vocab_size
    assert isinstance(payload["version"], str)
    assert isinstance(payload["trained_at"], str)
    assert isinstance(payload["vocabulary"], list)
    assert isinstance(payload["config"], dict)


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_load_restores_identical_encode_decode_behavior(tmp_path, tokenizer_cls):
    tokenizer = tokenizer_cls()
    tokenizer.train(["hello world", "hello there"])
    text = "hello world"
    ids_before = tokenizer.encode(text)
    decoded_before = tokenizer.decode(ids_before)

    path = tmp_path / "state.json"
    tokenizer.save(path)

    reloaded = tokenizer_cls()
    reloaded.load(path)

    assert reloaded.vocab_size == tokenizer.vocab_size
    assert reloaded.encode(text) == ids_before
    assert reloaded.decode(reloaded.encode(text)) == decoded_before


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_load_rejects_a_file_saved_by_a_different_tokenizer_type(tmp_path, tokenizer_cls):
    other_cls = next(cls for cls in TOKENIZER_CLASSES if cls is not tokenizer_cls)
    other = other_cls()
    other.train(["hello world"])
    path = tmp_path / "state.json"
    other.save(path)

    with pytest.raises(ValueError):
        tokenizer_cls().load(path)


def test_vocabulary_tokens_in_order_matches_ids():
    vocab = Vocabulary(["a", "b", "c"])

    tokens = vocabulary_tokens_in_order(vocab)

    assert tokens == ["a", "b", "c"]
    for expected_id, token in enumerate(tokens):
        assert vocab.get_id(token) == expected_id
