"""Token frequency analysis over a trained tokenizer's vocabulary (Task 4.3).

Feeds the planned "Vocabulary" UI page (Phase 8): how often each token a
trained tokenizer actually produces occurs across a training corpus, which
tokens are rare, and which dominate — the practical face of natural
language's Zipfian distribution. A handful of tokens (common words,
frequent subwords) account for most occurrences, while most of a
vocabulary's tokens appear rarely — a "long tail" — which is exactly why
vocabulary size and merge/training choices matter: growing a vocabulary
mostly adds rarely-used long-tail entries, with diminishing returns. This
module only computes the statistics — no Streamlit dependency, mirroring
`benchmarking.metrics`'s "pure function + dataclass" pattern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from tokenizers.base import Tokenizer


@dataclass(frozen=True)
class TokenFrequency:
    """One token's usage count, as actually produced by a tokenizer's `tokenize`."""

    token: str
    frequency: int


def compute_token_frequencies(tokenizer: Tokenizer, corpus: list[str]) -> dict[str, int]:
    """Count how often each token `tokenizer.tokenize` produces appears across `corpus`.

    Uses the tokenizer's real segmentation (`tokenize`), never substring
    counting of a token's string in the raw corpus text — two pieces of
    text can look alike but tokenize differently (e.g. a WordPiece
    continuation piece `"##ing"` is not the same event as the literal text
    `"ing"` appearing somewhere), so this reports actual tokenizer usage.
    No filtering is applied: if `tokenize` produces a special token (e.g.
    `<UNK>` for input it cannot cover), it is counted like any other token,
    consistent with the rest of the project not hiding special tokens from
    output. An empty corpus, or a tokenizer that produces zero tokens for
    every document, returns `{}`.
    """
    frequencies: dict[str, int] = {}
    for text in corpus:
        for token in tokenizer.tokenize(text):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def top_n_frequent_tokens(frequencies: dict[str, int], n: int) -> list[TokenFrequency]:
    """Return the `n` most frequent tokens, highest frequency first.

    Ties are broken lexicographically by token, so the result is
    deterministic regardless of dict/hash ordering. `n <= 0` returns `[]`;
    `n` larger than the number of distinct tokens returns all of them.
    """
    if n <= 0:
        return []
    ordered = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return [TokenFrequency(token=token, frequency=freq) for token, freq in ordered[:n]]


def rare_tokens(frequencies: dict[str, int], threshold: int) -> list[TokenFrequency]:
    """Return every token with `frequency < threshold`, ordered by frequency then token.

    `threshold <= 0` returns `[]`: every recorded frequency is at least 1
    (a token only appears in `frequencies` because `tokenize` produced it
    at least once), so no frequency is ever below a non-positive threshold.
    A `threshold` higher than every frequency returns all tokens; ties are
    broken lexicographically, matching `top_n_frequent_tokens`.
    """
    if threshold <= 0:
        return []
    ordered = sorted(
        (item for item in frequencies.items() if item[1] < threshold),
        key=lambda item: (item[1], item[0]),
    )
    return [TokenFrequency(token=token, frequency=freq) for token, freq in ordered]


def to_dataframe(entries: list[TokenFrequency]) -> pd.DataFrame:
    """Convert a list of `TokenFrequency` (e.g. from `top_n_frequent_tokens`/`rare_tokens`)
    into a table.

    Columns are `token`/`frequency`, in the order `entries` was given —
    ready to display or export (CSV/JSON via Pandas) without further
    reshaping. An empty list produces an empty (but still `token`/
    `frequency`-shaped) DataFrame.
    """
    if not entries:
        return pd.DataFrame(columns=["token", "frequency"])
    return pd.DataFrame([asdict(entry) for entry in entries])
