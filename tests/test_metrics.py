"""Unit tests for tokenizer-agnostic metrics (Phase 5, Task 5.1)."""

from benchmarking.metrics import compute_metrics
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer


def _trained(tokenizer_cls, text):
    tokenizer = tokenizer_cls()
    tokenizer.train([text])
    return tokenizer


def test_number_of_tokens_comes_from_encode_not_tokenize():
    tokenizer = _trained(WordTokenizer, "hello world")

    metrics = compute_metrics(tokenizer, "hello world")

    assert metrics.number_of_tokens == len(tokenizer.encode("hello world"))


def test_tokens_per_word_basic_case():
    tokenizer = _trained(WordTokenizer, "hello world")

    metrics = compute_metrics(tokenizer, "hello world")

    assert metrics.tokens_per_word == 1.0  # 2 tokens / 2 words


def test_tokens_per_word_is_none_when_there_are_no_words():
    tokenizer = _trained(CharacterTokenizer, "   ")

    metrics = compute_metrics(tokenizer, "   ")

    assert metrics.tokens_per_word is None  # "   ".split() == []


def test_characters_per_token_basic_case():
    tokenizer = _trained(CharacterTokenizer, "hello")

    metrics = compute_metrics(tokenizer, "hello")

    assert metrics.characters_per_token == 1.0  # 5 chars / 5 tokens


def test_characters_per_token_is_none_for_empty_text():
    metrics = compute_metrics(CharacterTokenizer(), "")

    assert metrics.number_of_tokens == 0
    assert metrics.characters_per_token is None


def test_compression_ratio_equals_utf8_bytes_over_tokens_for_ascii():
    tokenizer = _trained(CharacterTokenizer, "hello")

    metrics = compute_metrics(tokenizer, "hello")

    assert metrics.compression_ratio == 1.0  # 5 ASCII bytes / 5 tokens


def test_compression_ratio_differs_from_characters_per_token_for_non_ascii_text():
    text = "こんにちは"  # 5 characters, 3 UTF-8 bytes each = 15 bytes
    tokenizer = _trained(CharacterTokenizer, text)

    metrics = compute_metrics(tokenizer, text)

    assert metrics.characters_per_token == 1.0
    assert metrics.compression_ratio == 3.0
    assert metrics.characters_per_token != metrics.compression_ratio


def test_compression_ratio_is_none_when_there_are_no_tokens():
    metrics = compute_metrics(CharacterTokenizer(), "")

    assert metrics.compression_ratio is None


def test_vocab_size_matches_tokenizer_vocab_size_property():
    tokenizer = _trained(CharacterTokenizer, "hello world")

    metrics = compute_metrics(tokenizer, "anything")

    assert metrics.vocab_size == tokenizer.vocab_size


def test_encoding_and_decoding_time_are_reserved_for_task_5_2():
    tokenizer = _trained(CharacterTokenizer, "hello")

    metrics = compute_metrics(tokenizer, "hello")

    assert metrics.encoding_time is None
    assert metrics.decoding_time is None


def test_single_token_input():
    tokenizer = _trained(CharacterTokenizer, "a")

    metrics = compute_metrics(tokenizer, "a")

    assert metrics.number_of_tokens == 1


def test_multiple_and_repeated_tokens():
    tokenizer = _trained(CharacterTokenizer, "aaa")

    metrics = compute_metrics(tokenizer, "aaa")

    assert metrics.number_of_tokens == 3
    assert metrics.characters_per_token == 1.0


def test_whitespace_only_text_can_have_tokens_but_never_words():
    tokenizer = _trained(CharacterTokenizer, "   ")

    metrics = compute_metrics(tokenizer, "   ")

    assert metrics.number_of_tokens == 3  # each space is its own character token
    assert metrics.tokens_per_word is None
    assert metrics.characters_per_token == 1.0
