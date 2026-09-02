"""Unit tests for token frequency analysis (Phase 4, Task 4.3)."""

from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer
from vocabulary.frequency_analysis import (
    TokenFrequency,
    compute_token_frequencies,
    rare_tokens,
    to_dataframe,
    top_n_frequent_tokens,
)

# ---------------------------------------------------------------------------
# compute_token_frequencies
# ---------------------------------------------------------------------------


def test_compute_token_frequencies_counts_tokenizer_output_not_raw_substrings():
    tokenizer = WordTokenizer()

    frequencies = compute_token_frequencies(tokenizer, ["hello world", "hello there"])

    assert frequencies == {"hello": 2, "world": 1, "there": 1}


def test_compute_token_frequencies_counts_repeated_token_across_documents():
    tokenizer = CharacterTokenizer()

    frequencies = compute_token_frequencies(tokenizer, ["aa", "a"])

    assert frequencies["a"] == 3


def test_compute_token_frequencies_multiple_distinct_tokens():
    tokenizer = WordTokenizer()

    frequencies = compute_token_frequencies(tokenizer, ["a b c a b a"])

    assert frequencies == {"a": 3, "b": 2, "c": 1}


def test_compute_token_frequencies_tokens_with_equal_frequency():
    tokenizer = WordTokenizer()

    frequencies = compute_token_frequencies(tokenizer, ["a b"])

    assert frequencies == {"a": 1, "b": 1}


def test_compute_token_frequencies_handles_unicode():
    tokenizer = CharacterTokenizer()

    frequencies = compute_token_frequencies(tokenizer, ["héllo"])

    assert frequencies["é"] == 1
    assert frequencies["h"] == 1


def test_compute_token_frequencies_empty_corpus_is_empty():
    assert compute_token_frequencies(WordTokenizer(), []) == {}


def test_compute_token_frequencies_tokenizer_producing_zero_tokens():
    assert compute_token_frequencies(WordTokenizer(), ["", "   "]) == {}


# ---------------------------------------------------------------------------
# top_n_frequent_tokens
# ---------------------------------------------------------------------------


def test_top_n_frequent_tokens_orders_by_frequency_descending():
    frequencies = {"a": 1, "b": 5, "c": 3}

    assert top_n_frequent_tokens(frequencies, n=3) == [
        TokenFrequency("b", 5),
        TokenFrequency("c", 3),
        TokenFrequency("a", 1),
    ]


def test_top_n_frequent_tokens_n_equals_1():
    frequencies = {"a": 1, "b": 5, "c": 3}

    assert top_n_frequent_tokens(frequencies, n=1) == [TokenFrequency("b", 5)]


def test_top_n_frequent_tokens_n_larger_than_distinct_tokens_returns_all():
    frequencies = {"a": 1, "b": 5}

    result = top_n_frequent_tokens(frequencies, n=100)

    assert len(result) == 2


def test_top_n_frequent_tokens_n_zero_returns_empty():
    assert top_n_frequent_tokens({"a": 1}, n=0) == []


def test_top_n_frequent_tokens_negative_n_returns_empty():
    assert top_n_frequent_tokens({"a": 1}, n=-5) == []


def test_top_n_frequent_tokens_breaks_ties_lexicographically():
    frequencies = {"z": 2, "a": 2}

    assert top_n_frequent_tokens(frequencies, n=2) == [
        TokenFrequency("a", 2),
        TokenFrequency("z", 2),
    ]


def test_top_n_frequent_tokens_empty_frequencies():
    assert top_n_frequent_tokens({}, n=5) == []


# ---------------------------------------------------------------------------
# rare_tokens
# ---------------------------------------------------------------------------


def test_rare_tokens_returns_tokens_below_threshold():
    frequencies = {"a": 1, "b": 5, "c": 2}

    assert rare_tokens(frequencies, threshold=3) == [
        TokenFrequency("a", 1),
        TokenFrequency("c", 2),
    ]


def test_rare_tokens_very_low_threshold_excludes_everything():
    frequencies = {"a": 1, "b": 5}

    assert rare_tokens(frequencies, threshold=1) == []


def test_rare_tokens_very_high_threshold_includes_everything():
    frequencies = {"a": 1, "b": 5}

    result = rare_tokens(frequencies, threshold=1000)

    assert {tf.token for tf in result} == {"a", "b"}


def test_rare_tokens_threshold_zero_or_negative_returns_empty():
    frequencies = {"a": 1}

    assert rare_tokens(frequencies, threshold=0) == []
    assert rare_tokens(frequencies, threshold=-1) == []


def test_rare_tokens_breaks_ties_lexicographically():
    frequencies = {"z": 1, "a": 1}

    assert rare_tokens(frequencies, threshold=5) == [
        TokenFrequency("a", 1),
        TokenFrequency("z", 1),
    ]


def test_rare_tokens_empty_frequencies():
    assert rare_tokens({}, threshold=5) == []


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_same_corpus_produces_the_same_frequencies_every_time():
    tokenizer = WordTokenizer()
    corpus = ["hello world", "hello there"]

    first = compute_token_frequencies(tokenizer, corpus)
    second = compute_token_frequencies(tokenizer, corpus)

    assert first == second


def test_same_frequencies_produce_the_same_top_n_every_time():
    frequencies = {"a": 1, "b": 5, "c": 3}

    assert top_n_frequent_tokens(frequencies, n=2) == top_n_frequent_tokens(frequencies, n=2)


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------


def test_to_dataframe_has_token_and_frequency_columns():
    df = to_dataframe([TokenFrequency("a", 3), TokenFrequency("b", 1)])

    assert list(df.columns) == ["token", "frequency"]
    assert df.iloc[0]["token"] == "a"
    assert df.iloc[0]["frequency"] == 3


def test_to_dataframe_empty_list_is_empty_dataframe_with_expected_columns():
    df = to_dataframe([])

    assert list(df.columns) == ["token", "frequency"]
    assert len(df) == 0
