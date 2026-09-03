"""Unit tests for `tokenizers.bpe.visualization` (Task 8.11)."""

from tokenizers.bpe.trainer import END_OF_WORD, train_bpe
from tokenizers.bpe.visualization import compute_merge_steps, format_word_freqs

_LOW_FAMILY_CORPUS = ["low"] * 5 + ["lower"] * 2 + ["lowest"]


def test_compute_merge_steps_matches_the_real_trainer_merges():
    result = train_bpe(_LOW_FAMILY_CORPUS, num_merges=4)

    steps = compute_merge_steps(_LOW_FAMILY_CORPUS, result.merges)

    assert [step.pair for step in steps] == result.merges
    assert [step.step for step in steps] == [1, 2, 3, 4]


def test_first_step_before_state_is_the_untouched_initial_corpus():
    result = train_bpe(_LOW_FAMILY_CORPUS, num_merges=4)
    steps = compute_merge_steps(_LOW_FAMILY_CORPUS, result.merges)

    low_word = ("l", "o", "w", END_OF_WORD)
    assert steps[0].before[low_word] == 5  # "low" appears 5 times in the corpus


def test_each_steps_after_state_is_the_next_steps_before_state():
    result = train_bpe(_LOW_FAMILY_CORPUS, num_merges=4)
    steps = compute_merge_steps(_LOW_FAMILY_CORPUS, result.merges)

    for current, following in zip(steps, steps[1:], strict=False):
        assert current.after == following.before


def test_last_step_after_state_has_no_more_of_the_learned_pair():
    result = train_bpe(_LOW_FAMILY_CORPUS, num_merges=4)
    steps = compute_merge_steps(_LOW_FAMILY_CORPUS, result.merges)

    last_pair = steps[-1].pair
    for word in steps[-1].after:
        pairs = set(zip(word, word[1:], strict=False))
        assert last_pair not in pairs


def test_empty_merges_list_produces_no_steps():
    assert compute_merge_steps(_LOW_FAMILY_CORPUS, []) == []


def test_format_word_freqs_is_sorted_by_descending_frequency_then_word():
    word_freqs = {
        ("l", "o", "w", END_OF_WORD): 5,
        ("l", "o", "w", "e", "r", END_OF_WORD): 2,
        ("l", "o", "w", "e", "s", "t", END_OF_WORD): 2,
    }

    lines = format_word_freqs(word_freqs)

    assert lines[0] == f"l o w {END_OF_WORD}  (x5)"
    # the two freq=2 entries tie-break lexicographically by their symbols
    assert lines[1] == f"l o w e r {END_OF_WORD}  (x2)"
    assert lines[2] == f"l o w e s t {END_OF_WORD}  (x2)"


def test_format_word_freqs_of_empty_table_is_empty_list():
    assert format_word_freqs({}) == []
