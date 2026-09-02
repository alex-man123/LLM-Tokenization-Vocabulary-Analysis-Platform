"""Unit tests for the BPE training primitives (Phase 2, Task 2.1/2.2)."""

from tokenizers.bpe.trainer import (
    END_OF_WORD,
    apply_merge,
    base_symbols_from,
    build_word_frequencies,
    count_pairs,
    merge_pair_in_word,
    select_best_pair,
    train_bpe,
)


def test_build_word_frequencies_splits_on_whitespace_and_appends_end_of_word():
    word_freqs = build_word_frequencies(["low low", "lower"])

    assert word_freqs[("l", "o", "w", END_OF_WORD)] == 2
    assert word_freqs[("l", "o", "w", "e", "r", END_OF_WORD)] == 1


def test_build_word_frequencies_handles_empty_and_whitespace_only_corpus():
    assert build_word_frequencies([]) == {}
    assert build_word_frequencies([""]) == {}
    assert build_word_frequencies(["   "]) == {}


def test_base_symbols_from_is_first_seen_order():
    word_freqs = build_word_frequencies(["ba ab"])

    assert base_symbols_from(word_freqs) == ["b", "a", END_OF_WORD]


def test_count_pairs_matches_the_conceptual_example():
    word_freqs = {("h", "e", "l", "l", "o"): 1}

    assert count_pairs(word_freqs) == {
        ("h", "e"): 1,
        ("e", "l"): 1,
        ("l", "l"): 1,
        ("l", "o"): 1,
    }


def test_count_pairs_empty_word_freqs():
    assert count_pairs({}) == {}


def test_count_pairs_single_symbol_word_has_no_pairs():
    assert count_pairs({("a",): 5}) == {}


def test_count_pairs_aggregates_frequency_across_words():
    word_freqs = {("a", "b"): 3, ("a", "b", "c"): 2}

    pair_counts = count_pairs(word_freqs)

    assert pair_counts[("a", "b")] == 5
    assert pair_counts[("b", "c")] == 2


def test_select_best_pair_picks_highest_frequency():
    assert select_best_pair({("a", "b"): 3, ("c", "d"): 5}) == ("c", "d")


def test_select_best_pair_breaks_ties_lexicographically():
    assert select_best_pair({("z", "a"): 4, ("a", "z"): 4}) == ("a", "z")


def test_select_best_pair_returns_none_for_empty_counts():
    assert select_best_pair({}) is None


def test_merge_pair_in_word_replaces_non_overlapping_occurrences():
    word = ("l", "o", "w", END_OF_WORD)

    assert merge_pair_in_word(word, ("l", "o")) == ("lo", "w", END_OF_WORD)


def test_merge_pair_in_word_leaves_word_untouched_when_pair_absent():
    word = ("a", "b", "c")

    assert merge_pair_in_word(word, ("b", "d")) == word


def test_apply_merge_updates_every_word():
    result = apply_merge({("a", "b"): 2, ("a", "c"): 1}, ("a", "b"))

    assert result == {("ab",): 2, ("a", "c"): 1}


def test_apply_merge_aggregates_frequencies_of_colliding_words():
    # ("x", "y") merges to ("xy",); the already-merged word ("xy",) has no
    # adjacent pair to merge and stays as-is — both end up as the same key.
    result = apply_merge({("x", "y"): 2, ("xy",): 3}, ("x", "y"))

    assert result == {("xy",): 5}


def test_train_bpe_reproduces_the_classic_low_lower_lowest_example():
    corpus = ["low"] * 5 + ["lower"] * 2 + ["lowest"]

    result = train_bpe(corpus, num_merges=4)

    assert result.merges == [
        ("l", "o"),
        ("lo", "w"),
        ("low", END_OF_WORD),
        ("low", "e"),
    ]
    assert sorted(result.base_symbols) == sorted(["l", "o", "w", "e", "s", "t", "r", END_OF_WORD])


def test_train_bpe_is_deterministic():
    corpus = ["low"] * 5 + ["lower"] * 2 + ["lowest"]

    first = train_bpe(corpus, num_merges=10)
    second = train_bpe(corpus, num_merges=10)

    assert first.merges == second.merges
    assert first.base_symbols == second.base_symbols


def test_train_bpe_stops_early_when_no_pairs_remain():
    result = train_bpe(["a"], num_merges=100)

    assert result.merges == [("a", END_OF_WORD)]


def test_train_bpe_handles_empty_corpus():
    result = train_bpe([], num_merges=10)

    assert result.merges == []
    assert result.base_symbols == []


def test_train_bpe_respects_num_merges_cap():
    corpus = ["low"] * 5 + ["lower"] * 2 + ["lowest"]

    result = train_bpe(corpus, num_merges=2)

    assert len(result.merges) == 2
    assert result.merges == [("l", "o"), ("lo", "w")]
