"""Unit tests for `ui/next_token_demo.py` (Task 8.14).

`ui/` is not on `pythonpath` (only `src/` is, `pyproject.toml`), so this
test inserts it into `sys.path` itself, exactly like every page under
`ui/pages/` already does to import sibling modules such as
`tokenizer_options`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

from next_token_demo import MAX_CANDIDATES, candidate_tokens, random_probabilities  # noqa: E402

from tokenizers.character_tokenizer import CharacterTokenizer  # noqa: E402
from vocabulary.special_tokens import ORDERED_SPECIAL_TOKENS  # noqa: E402


def _trained_special_tokens(text: str):
    tokenizer = CharacterTokenizer()
    tokenizer.train([text])
    return tokenizer.special_tokens


def test_candidate_tokens_excludes_special_tokens():
    special_tokens = _trained_special_tokens("hello world")

    candidates = candidate_tokens(special_tokens)

    assert not any(token in ORDERED_SPECIAL_TOKENS for token in candidates)


def test_candidate_tokens_are_real_vocabulary_entries():
    special_tokens = _trained_special_tokens("hello world")

    candidates = candidate_tokens(special_tokens)

    for token in candidates:
        assert special_tokens.vocabulary.has_token(token)


def test_candidate_tokens_caps_at_max_candidates():
    # A long, varied corpus produces many distinct characters.
    special_tokens = _trained_special_tokens("abcdefghijklmnopqrstuvwxyz0123456789")

    candidates = candidate_tokens(special_tokens)

    assert len(candidates) == MAX_CANDIDATES


def test_candidate_tokens_respects_a_custom_max():
    special_tokens = _trained_special_tokens("abcdefghij")

    candidates = candidate_tokens(special_tokens, max_candidates=3)

    assert len(candidates) == 3


def test_candidate_tokens_returns_fewer_than_max_for_a_tiny_vocabulary():
    special_tokens = _trained_special_tokens("aa")

    candidates = candidate_tokens(special_tokens)

    assert candidates == ["a"]


def test_candidate_tokens_is_deterministic_and_ordered_by_id():
    special_tokens = _trained_special_tokens("hello world")

    first = candidate_tokens(special_tokens)
    second = candidate_tokens(special_tokens)

    assert first == second


def test_random_probabilities_sum_to_one():
    probabilities = random_probabilities(6, seed=1)

    assert probabilities.shape == (6,)
    assert probabilities.sum() == pytest.approx(1.0)


def test_random_probabilities_are_all_non_negative():
    probabilities = random_probabilities(8, seed=42)

    assert (probabilities >= 0).all()


def test_random_probabilities_same_seed_is_deterministic():
    first = random_probabilities(5, seed=7)
    second = random_probabilities(5, seed=7)

    assert np.array_equal(first, second)


def test_random_probabilities_different_seed_differs():
    first = random_probabilities(5, seed=1)
    second = random_probabilities(5, seed=2)

    assert not np.array_equal(first, second)


def test_random_probabilities_of_zero_candidates_is_empty():
    probabilities = random_probabilities(0, seed=1)

    assert probabilities.shape == (0,)
