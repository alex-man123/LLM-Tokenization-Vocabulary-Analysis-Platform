"""Unit tests for the tokenizer Comparator (Phase 5, Task 5.3)."""

import pandas as pd

from benchmarking.comparator import compare_tokenizers
from tokenizers.bpe.tokenizer import BPETokenizer
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer

REQUIRED_COLUMNS = {
    "tokenizer",
    "number_of_tokens",
    "tokens_per_word",
    "characters_per_token",
    "compression_ratio",
    "vocab_size",
    "encoding_time",
    "decoding_time",
}


def _trained(tokenizer_cls, text, **kwargs):
    tokenizer = tokenizer_cls(**kwargs)
    tokenizer.train([text])
    return tokenizer


def test_compare_a_single_tokenizer_returns_one_row_with_required_columns():
    text = "hello world"

    df = compare_tokenizers([_trained(CharacterTokenizer, text)], text)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert REQUIRED_COLUMNS <= set(df.columns)


def test_compare_two_tokenizers_returns_two_rows():
    text = "hello world"
    tokenizers = [_trained(CharacterTokenizer, text), _trained(WordTokenizer, text)]

    df = compare_tokenizers(tokenizers, text)

    assert len(df) == 2
    assert set(df["tokenizer"]) == {"character", "word"}


def test_compare_four_tokenizers_simultaneously():
    text = "hello world hello"
    tokenizers = [
        _trained(CharacterTokenizer, text),
        _trained(WordTokenizer, text),
        _trained(BPETokenizer, text, num_merges=5),
        _trained(BPETokenizer, text, num_merges=20),
    ]

    df = compare_tokenizers(tokenizers, text)

    assert len(df) == 4


def test_compare_reports_vocab_size_matching_each_tokenizer():
    text = "hello world"
    tokenizers = [_trained(CharacterTokenizer, text), _trained(BPETokenizer, text)]

    df = compare_tokenizers(tokenizers, text)

    assert (df["vocab_size"] > 0).all()
    for tokenizer, vocab_size in zip(tokenizers, df["vocab_size"], strict=True):
        assert vocab_size == tokenizer.vocab_size


def test_compare_handles_empty_text():
    df = compare_tokenizers([CharacterTokenizer()], "")

    assert df.loc[0, "number_of_tokens"] == 0
    assert pd.isna(df.loc[0, "tokens_per_word"])
    assert pd.isna(df.loc[0, "characters_per_token"])


def test_compare_handles_unicode_text():
    text = "こんにちは"
    tokenizer = _trained(CharacterTokenizer, text)

    df = compare_tokenizers([tokenizer], text)

    assert df.loc[0, "number_of_tokens"] == 5
    assert df.loc[0, "compression_ratio"] > df.loc[0, "characters_per_token"]


def test_compare_includes_tokens_column_for_ui_display():
    text = "hello"
    tokenizer = _trained(CharacterTokenizer, text)

    df = compare_tokenizers([tokenizer], text)

    assert df.loc[0, "tokens"] == list("hello")
