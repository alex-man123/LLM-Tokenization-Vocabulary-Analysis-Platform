"""Unit tests for the WordPiece training primitives (Phase 3, Task 3.1/3.2/3.3)."""

from tokenizers.wordpiece.trainer import (
    CONTINUATION_PREFIX,
    apply_merge,
    base_symbols_from,
    build_word_frequencies,
    count_pairs,
    count_symbol_frequencies,
    merge_pair_in_word,
    merged_symbol,
    score_pairs,
    select_best_pair,
    train_wordpiece,
    word_to_symbols,
)

# ---------------------------------------------------------------------------
# Task 3.1 — pre-tokenization & ## representation
# ---------------------------------------------------------------------------


def test_word_to_symbols_prefixes_every_character_but_the_first():
    assert word_to_symbols("hello") == ("h", "##e", "##l", "##l", "##o")


def test_word_to_symbols_single_character_word_has_no_prefix():
    assert word_to_symbols("a") == ("a",)


def test_word_to_symbols_empty_string_is_empty_tuple():
    assert word_to_symbols("") == ()


def test_word_to_symbols_handles_unicode_characters():
    assert word_to_symbols("héllo") == ("h", "##é", "##l", "##l", "##o")


def test_build_word_frequencies_reuses_word_tokenizer_split():
    # "hello!" -> words ["hello", "!"] via WordTokenizer.tokenize (Task 1.2).
    word_freqs = build_word_frequencies(["hello! hello!"])

    assert word_freqs[word_to_symbols("hello")] == 2
    assert word_freqs[word_to_symbols("!")] == 2


def test_build_word_frequencies_handles_empty_and_whitespace_only_corpus():
    assert build_word_frequencies([]) == {}
    assert build_word_frequencies([""]) == {}
    assert build_word_frequencies(["   "]) == {}
    assert build_word_frequencies(["\n\n"]) == {}


def test_build_word_frequencies_splits_contractions_like_word_tokenizer():
    word_freqs = build_word_frequencies(["don't"])

    assert word_to_symbols("don") in word_freqs
    assert word_to_symbols("'") in word_freqs
    assert word_to_symbols("t") in word_freqs


def test_base_symbols_from_is_first_seen_order():
    word_freqs = build_word_frequencies(["ba ab"])

    assert base_symbols_from(word_freqs) == ["b", "##a", "a", "##b"]


# ---------------------------------------------------------------------------
# Task 3.2 — WordPiece merge score
# ---------------------------------------------------------------------------


def test_count_symbol_frequencies_matches_manual_calculation():
    word_freqs = {("t", "##h"): 10, ("z", "##q"): 2}

    assert count_symbol_frequencies(word_freqs) == {"t": 10, "##h": 10, "z": 2, "##q": 2}


def test_count_pairs_matches_manual_calculation():
    word_freqs = {("t", "##h"): 10, ("z", "##q"): 2}

    assert count_pairs(word_freqs) == {("t", "##h"): 10, ("z", "##q"): 2}


def test_score_pairs_matches_the_formula_freq_ab_over_freq_a_times_freq_b():
    # freq(t,h)=10, freq(t)=10, freq(h)=10 -> score = 10 / (10*10) = 0.1
    # freq(z,q)=2,  freq(z)=2,  freq(q)=2  -> score = 2  / (2*2)   = 0.5
    word_freqs = {("t", "##h"): 10, ("z", "##q"): 2}
    symbol_freqs = count_symbol_frequencies(word_freqs)
    pair_counts = count_pairs(word_freqs)

    scores = score_pairs(pair_counts, symbol_freqs)

    assert scores[("t", "##h")] == 10 / (10 * 10)
    assert scores[("z", "##q")] == 2 / (2 * 2)


def test_highest_raw_frequency_pair_is_not_necessarily_the_highest_scoring_pair():
    # ("t", "##h") is far more frequent in absolute terms (10 vs 2), but
    # ("z", "##q") scores higher because its components never occur apart
    # from each other -- this is exactly the WordPiece-vs-BPE distinction.
    word_freqs = {("t", "##h"): 10, ("z", "##q"): 2}
    symbol_freqs = count_symbol_frequencies(word_freqs)
    pair_counts = count_pairs(word_freqs)

    most_frequent_pair = max(pair_counts.items(), key=lambda item: item[1])[0]
    scores = score_pairs(pair_counts, symbol_freqs)
    best_scoring_pair = select_best_pair(scores)

    assert most_frequent_pair == ("t", "##h")
    assert best_scoring_pair == ("z", "##q")
    assert best_scoring_pair != most_frequent_pair


def test_score_pairs_skips_pairs_whose_symbol_frequency_is_missing_instead_of_raising():
    pair_counts = {("a", "##b"): 3}
    symbol_freqs = {"a": 5}  # "##b" missing -- would otherwise ZeroDivisionError

    assert score_pairs(pair_counts, symbol_freqs) == {}


def test_score_pairs_empty_input():
    assert score_pairs({}, {}) == {}


def test_select_best_pair_picks_highest_score():
    assert select_best_pair({("a", "##b"): 0.1, ("c", "##d"): 0.5}) == ("c", "##d")


def test_select_best_pair_breaks_ties_lexicographically():
    assert select_best_pair({("z", "##a"): 0.4, ("a", "##z"): 0.4}) == ("a", "##z")


def test_select_best_pair_returns_none_for_empty_scores():
    assert select_best_pair({}) is None


# ---------------------------------------------------------------------------
# merges: merged_symbol / merge_pair_in_word / apply_merge
# ---------------------------------------------------------------------------


def test_merged_symbol_strips_continuation_prefix_only_from_the_right_symbol():
    assert merged_symbol("un", "##believ") == "unbeliev"
    assert merged_symbol("##be", "##lievable") == "##believable"


def test_merged_symbol_left_without_prefix_stays_without_prefix():
    assert merged_symbol("a", "##b") == "ab"


def test_merge_pair_in_word_replaces_non_overlapping_occurrences():
    word = ("h", "##e", "##l", "##l", "##o")

    assert merge_pair_in_word(word, ("##l", "##l")) == ("h", "##e", "##ll", "##o")


def test_merge_pair_in_word_leaves_word_untouched_when_pair_absent():
    word = ("a", "##b", "##c")

    assert merge_pair_in_word(word, ("##b", "##d")) == word


def test_apply_merge_updates_every_word():
    result = apply_merge({("a", "##b"): 2, ("a", "##c"): 1}, ("a", "##b"))

    assert result == {("ab",): 2, ("a", "##c"): 1}


def test_apply_merge_aggregates_frequencies_of_colliding_words():
    result = apply_merge({("x", "##y"): 2, ("xy",): 3}, ("x", "##y"))

    assert result == {("xy",): 5}


# ---------------------------------------------------------------------------
# Task 3.3 — training loop
# ---------------------------------------------------------------------------


def test_train_wordpiece_learns_the_highest_scoring_merge_first():
    corpus = ["th"] * 10 + ["zq"] * 2

    result = train_wordpiece(corpus, vocab_size=100)

    assert "zq" in result.vocabulary_tokens
    zq_index = result.vocabulary_tokens.index("zq")
    th_index = (
        result.vocabulary_tokens.index("th") if "th" in result.vocabulary_tokens else None
    )
    # "zq" (higher score) must be learned before "th" (higher raw frequency
    # but lower score), if "th" is learned at all within this vocab budget.
    assert th_index is None or zq_index < th_index


def test_train_wordpiece_is_deterministic():
    corpus = ["low"] * 5 + ["lower"] * 2 + ["lowest"]

    first = train_wordpiece(corpus, vocab_size=20)
    second = train_wordpiece(corpus, vocab_size=20)

    assert first.vocabulary_tokens == second.vocabulary_tokens


def test_train_wordpiece_stops_early_when_no_pairs_remain():
    result = train_wordpiece(["a"], vocab_size=100)

    assert result.vocabulary_tokens == ["a"]


def test_train_wordpiece_handles_empty_corpus():
    result = train_wordpiece([], vocab_size=10)

    assert result.vocabulary_tokens == []


def test_train_wordpiece_respects_vocab_size_budget():
    corpus = ["low"] * 5 + ["lower"] * 2 + ["lowest"]
    base_alphabet_size = len(base_symbols_from(build_word_frequencies(corpus)))

    result = train_wordpiece(corpus, vocab_size=base_alphabet_size + 2)

    assert len(result.vocabulary_tokens) == base_alphabet_size + 2


def test_train_wordpiece_never_shrinks_below_the_base_alphabet():
    corpus = ["low"] * 5 + ["lower"] * 2 + ["lowest"]
    base_alphabet_size = len(base_symbols_from(build_word_frequencies(corpus)))

    result = train_wordpiece(corpus, vocab_size=1)  # smaller than the alphabet itself

    assert len(result.vocabulary_tokens) == base_alphabet_size


def test_train_wordpiece_on_repetitive_corpus_merges_the_whole_repeated_word():
    result = train_wordpiece(["ab"] * 20, vocab_size=100)

    assert "ab" in result.vocabulary_tokens


def test_train_wordpiece_every_token_covers_only_characters_seen_in_corpus():
    corpus = ["un"] * 6 + ["believable"] * 6
    result = train_wordpiece(corpus, vocab_size=50)

    corpus_chars = {char for word in corpus for char in word}
    for token in result.vocabulary_tokens:
        stripped = (
            token[len(CONTINUATION_PREFIX) :] if token.startswith(CONTINUATION_PREFIX) else token
        )
        assert all(char in corpus_chars for char in stripped)
