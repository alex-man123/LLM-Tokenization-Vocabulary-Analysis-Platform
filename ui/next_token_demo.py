"""Illustrative next-token prediction bar chart (Task 8.14).

Extends the "Model -> prediction" stage of the pipeline (Task 8.13) with a
small chart shaped like a real LLM's next-token probability distribution.
Candidates are real entries from the *currently trained* tokenizer's own
vocabulary (Task 4.1) — this ties the demo visually to the tokenizer
actually selected, rather than a hardcoded word list — but the
probabilities themselves are 100% random and have no relationship to
those tokens' real likelihood. No model is loaded, trained, or simulated:
this shows the *shape* of a prediction, not a computed one, the same
real/illustrative distinction Task 8.6/8.12 already draw for embeddings.
"""

from __future__ import annotations

import numpy as np

from vocabulary.serialization import vocabulary_tokens_in_order
from vocabulary.special_tokens import SpecialTokens

MAX_CANDIDATES = 8


def candidate_tokens(
    special_tokens: SpecialTokens, max_candidates: int = MAX_CANDIDATES
) -> list[str]:
    """Up to `max_candidates` real, non-special tokens from `special_tokens`'s vocabulary.

    Ordered by token ID (registration order), so the same trained
    tokenizer always offers the same candidates — only the probabilities
    below are randomized, never which tokens are shown. `<PAD>`/`<UNK>`/
    `<BOS>`/`<EOS>` are excluded: they are not meaningful "next word"
    candidates for this illustration.
    """
    all_tokens = vocabulary_tokens_in_order(special_tokens.vocabulary)
    real_tokens = [token for token in all_tokens if not special_tokens.is_special(token)]
    return real_tokens[:max_candidates]


def random_probabilities(n: int, seed: int) -> np.ndarray:
    """`n` random numbers, seeded by `seed`, normalized to sum to 1.

    Deterministic for a given `(n, seed)` pair — so the chart stays stable
    across unrelated Streamlit reruns (e.g. moving a different widget) —
    but otherwise has no relationship whatsoever to the candidate tokens'
    real likelihood. Calling this again with a different `seed` (e.g. a
    "Reroll" button) produces a completely different distribution, which
    is the point: nothing here is a deterministic model computation.
    """
    if n <= 0:
        return np.zeros(0)
    weights = np.random.default_rng(seed=seed).uniform(0.01, 1.0, size=n)
    return weights / weights.sum()
