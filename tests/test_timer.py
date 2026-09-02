"""Unit tests for tokenizer timing (Phase 5, Task 5.2).

None of these tests assert an exact timing value (real durations depend on
the machine running them) — only structural properties: non-negative
durations, correct repetition counts, warm-up exclusion, and encode/decode
being measured separately.
"""

import pytest

from benchmarking.timer import TimingResult, TokenizerTiming, measure_tokenizer_timing
from tokenizers.base import Tokenizer
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer


class _CountingTokenizer(Tokenizer):
    """Wraps a trained tokenizer, counting `encode`/`decode` calls to inspect timing behavior."""

    def __init__(self, inner: Tokenizer) -> None:
        self._inner = inner
        self.encode_calls = 0
        self.decode_calls = 0

    def train(self, corpus: list[str]) -> None:
        self._inner.train(corpus)

    def tokenize(self, text: str) -> list[str]:
        return self._inner.tokenize(text)

    def encode(self, text: str) -> list[int]:
        self.encode_calls += 1
        return self._inner.encode(text)

    def decode(self, ids: list[int]) -> str:
        self.decode_calls += 1
        return self._inner.decode(ids)

    def save(self, path) -> None:
        self._inner.save(path)

    def load(self, path) -> None:
        self._inner.load(path)

    @property
    def vocab_size(self) -> int:
        return self._inner.vocab_size

    @property
    def name(self) -> str:
        return self._inner.name


def _counting_word_tokenizer(text: str) -> _CountingTokenizer:
    inner = WordTokenizer()
    inner.train([text])
    return _CountingTokenizer(inner)


def test_returns_a_tokenizer_timing_with_encode_and_decode_results():
    tokenizer = _counting_word_tokenizer("hello world")

    result = measure_tokenizer_timing(tokenizer, "hello world", n_iterations=3)

    assert isinstance(result, TokenizerTiming)
    assert isinstance(result.encode, TimingResult)
    assert isinstance(result.decode, TimingResult)


def test_warm_up_is_not_included_in_the_reported_samples():
    tokenizer = _counting_word_tokenizer("hello world")

    result = measure_tokenizer_timing(tokenizer, "hello world", n_iterations=5)

    assert len(result.encode.samples_ms) == 5
    assert len(result.decode.samples_ms) == 5


def test_warm_up_is_not_included_but_still_actually_happens():
    tokenizer = _counting_word_tokenizer("hello world")

    measure_tokenizer_timing(tokenizer, "hello world", n_iterations=4)

    # encode: 1 warm-up + 4 measured (for the encode timing loop) + 1 more
    # to produce the IDs decode is measured against = 6 total.
    assert tokenizer.encode_calls == 6
    # decode: 1 warm-up + 4 measured, all against the same pre-computed IDs.
    assert tokenizer.decode_calls == 5


def test_decode_is_measured_against_the_same_ids_not_re_encoded_each_time():
    tokenizer = _counting_word_tokenizer("hello world")

    measure_tokenizer_timing(tokenizer, "hello world", n_iterations=10)

    # If decode's timing loop accidentally re-encoded before each decode,
    # encode_calls would scale with n_iterations far beyond the fixed +2
    # this function actually needs.
    assert tokenizer.encode_calls == 10 + 2


def test_n_iterations_equal_to_one():
    tokenizer = _counting_word_tokenizer("hello")

    result = measure_tokenizer_timing(tokenizer, "hello", n_iterations=1)

    assert len(result.encode.samples_ms) == 1
    assert len(result.decode.samples_ms) == 1
    assert result.encode.mean_ms == result.encode.median_ms == result.encode.samples_ms[0]


def test_n_iterations_greater_than_one():
    tokenizer = _counting_word_tokenizer("hello world foo bar")

    result = measure_tokenizer_timing(tokenizer, "hello world foo bar", n_iterations=20)

    assert len(result.encode.samples_ms) == 20
    assert len(result.decode.samples_ms) == 20


@pytest.mark.parametrize("n_iterations", [0, -1, -100])
def test_invalid_n_iterations_raises_value_error(n_iterations):
    tokenizer = _counting_word_tokenizer("hello")

    with pytest.raises(ValueError):
        measure_tokenizer_timing(tokenizer, "hello", n_iterations=n_iterations)


def test_all_durations_are_non_negative():
    tokenizer = _counting_word_tokenizer("hello world")

    result = measure_tokenizer_timing(tokenizer, "hello world", n_iterations=5)

    assert result.encode.mean_ms >= 0
    assert result.encode.median_ms >= 0
    assert all(sample >= 0 for sample in result.encode.samples_ms)
    assert result.decode.mean_ms >= 0
    assert result.decode.median_ms >= 0
    assert all(sample >= 0 for sample in result.decode.samples_ms)


def test_empty_text():
    tokenizer = _counting_word_tokenizer("")

    result = measure_tokenizer_timing(tokenizer, "", n_iterations=3)

    assert len(result.encode.samples_ms) == 3
    assert len(result.decode.samples_ms) == 3


def test_works_with_a_simple_character_tokenizer():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["hello"])

    result = measure_tokenizer_timing(tokenizer, "hello", n_iterations=3)

    assert result.encode.mean_ms >= 0
    assert result.decode.mean_ms >= 0


def test_works_with_a_tokenizer_producing_many_tokens():
    tokenizer = WordTokenizer()
    text = "the quick brown fox jumps over the lazy dog " * 10
    tokenizer.train([text])

    result = measure_tokenizer_timing(tokenizer, text, n_iterations=3)

    assert result.encode.mean_ms >= 0
    assert result.decode.mean_ms >= 0


def test_result_structure_is_deterministic_across_runs():
    tokenizer = _counting_word_tokenizer("hello world")

    first = measure_tokenizer_timing(tokenizer, "hello world", n_iterations=5)
    second = measure_tokenizer_timing(tokenizer, "hello world", n_iterations=5)

    assert len(first.encode.samples_ms) == len(second.encode.samples_ms) == 5
    assert len(first.decode.samples_ms) == len(second.decode.samples_ms) == 5
