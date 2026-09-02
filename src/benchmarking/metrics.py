"""Tokenizer-agnostic metrics for benchmarking (Task 5.1).

Metrics are computed purely from a `Tokenizer`'s public API (`encode`,
`vocab_size`) and the raw text — never from a tokenizer's internals, and
never duplicated inside a tokenizer, the Comparator (Task 5.3), or the UI.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizers.base import Tokenizer


@dataclass(frozen=True)
class TokenizationMetrics:
    """One tokenizer's metrics for one piece of text.

    `tokens_per_word`, `characters_per_token`, and `compression_ratio` are
    `None` when their denominator would be zero (no words, or no tokens) —
    there is no meaningful ratio to report, so `None` is used instead of a
    misleading `0.0`.

    `characters_per_token` (chars/token) and `compression_ratio`
    (UTF-8 bytes/token) are deliberately different metrics, not two names
    for the same formula: they read the same for ASCII text (~1 byte per
    character) but diverge for non-ASCII text (e.g. a Japanese character is
    usually 3 UTF-8 bytes), which is exactly the comparison worth reporting
    for multi-language experiments.

    `encoding_time`/`decoding_time` are reserved for Task 5.2 (timing) and
    are always `None` here — this task only prepares the field so timing
    can be added later without changing this shape.
    """

    tokenizer: str
    number_of_tokens: int
    tokens_per_word: float | None
    characters_per_token: float | None
    compression_ratio: float | None
    vocab_size: int
    encoding_time: float | None = None
    decoding_time: float | None = None


def compute_metrics(tokenizer: Tokenizer, text: str) -> TokenizationMetrics:
    """Compute `TokenizationMetrics` for `tokenizer` on `text`.

    `number_of_tokens` comes from `len(tokenizer.encode(text))` — the
    tokenizer's official tokenized-sequence contract — not from
    `tokenize(text)`. `tokens_per_word` uses a whitespace-only
    `text.split()` word count, independent of any tokenizer's own
    word-splitting logic, so it is comparable across tokenizers.
    """
    number_of_tokens = len(tokenizer.encode(text))
    num_words = len(text.split())
    num_characters = len(text)
    original_size_in_bytes = len(text.encode("utf-8"))

    return TokenizationMetrics(
        tokenizer=tokenizer.name,
        number_of_tokens=number_of_tokens,
        tokens_per_word=(number_of_tokens / num_words) if num_words else None,
        characters_per_token=(num_characters / number_of_tokens) if number_of_tokens else None,
        compression_ratio=(
            (original_size_in_bytes / number_of_tokens) if number_of_tokens else None
        ),
        vocab_size=tokenizer.vocab_size,
    )
