"""Contract tests: every concrete tokenizer must satisfy `Tokenizer` (Task 1.4, 2.5)."""

import pytest

from tokenizers.base import Tokenizer
from tokenizers.bpe.tokenizer import BPETokenizer
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer
from tokenizers.wordpiece.tokenizer import WordPieceTokenizer

TOKENIZER_CLASSES = [CharacterTokenizer, WordTokenizer, BPETokenizer, WordPieceTokenizer]


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_is_a_tokenizer_subclass(tokenizer_cls):
    assert issubclass(tokenizer_cls, Tokenizer)
    assert isinstance(tokenizer_cls(), Tokenizer)


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_implements_every_required_method(tokenizer_cls):
    for member in ("train", "tokenize", "encode", "decode", "save", "load"):
        assert callable(getattr(tokenizer_cls, member))


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_exposes_vocab_size_and_name_as_properties(tokenizer_cls):
    tokenizer = tokenizer_cls()

    assert isinstance(tokenizer.vocab_size, int)
    assert isinstance(tokenizer.name, str)
    assert tokenizer.name


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_full_pipeline_runs_without_error(tokenizer_cls):
    tokenizer = tokenizer_cls()
    tokenizer.train(["hello world"])

    tokens = tokenizer.tokenize("hello world")
    ids = tokenizer.encode("hello world")
    text = tokenizer.decode(ids)

    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert isinstance(text, str)
