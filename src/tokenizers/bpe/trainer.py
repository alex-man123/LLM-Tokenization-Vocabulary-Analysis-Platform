"""BPE training: pair counting, merge selection, and the training loop.

Implements the classical character-level BPE algorithm (Sennrich et al.,
2015): start from characters plus an end-of-word marker, repeatedly merge
the most frequent adjacent pair of symbols, and record the merges in the
order they were learned. Pure functions, independent of `Vocabulary`/`Tokenizer`
— `tokenizers.bpe.tokenizer.BPETokenizer` is what wires this into the rest
of the project.
"""

from __future__ import annotations

from dataclasses import dataclass

END_OF_WORD = "</w>"

WordFrequencies = dict[tuple[str, ...], int]
PairCounts = dict[tuple[str, str], int]


def build_word_frequencies(corpus: list[str]) -> WordFrequencies:
    """Split `corpus` into whitespace-delimited words and count them.

    Each word becomes a tuple of its characters plus a trailing
    `END_OF_WORD` marker (e.g. `"low"` -> `("l", "o", "w", "</w>")`), so a
    merge can never combine symbols across a word boundary. Punctuation
    attached to a word (e.g. `"hello,"`) stays part of that word — corpus
    text is only split on whitespace here, not further tokenized.

    An empty corpus, or a corpus of only whitespace, produces `{}`.
    """
    word_freqs: WordFrequencies = {}
    for text in corpus:
        for word in text.split():
            symbols = (*tuple(word), END_OF_WORD)
            word_freqs[symbols] = word_freqs.get(symbols, 0) + 1
    return word_freqs


def base_symbols_from(word_freqs: WordFrequencies) -> list[str]:
    """Return the initial alphabet appearing in `word_freqs`, in first-seen order.

    Deterministic regardless of hash randomization: `word_freqs` is built
    by iterating the corpus in order, and dict iteration order is
    insertion order, so this only ever depends on the corpus content and
    order, never on Python's hashing of strings/tuples.
    """
    seen: dict[str, None] = {}
    for word in word_freqs:
        for symbol in word:
            seen.setdefault(symbol, None)
    return list(seen)


def count_pairs(word_freqs: WordFrequencies) -> PairCounts:
    """Count every adjacent symbol pair across `word_freqs`, weighted by word frequency.

    An empty `word_freqs` (`{}`) returns `{}`. A word with a single symbol
    contributes no pairs. Repeated words are already folded into `freq` by
    `build_word_frequencies`, so their pairs are counted with that weight,
    not once per raw occurrence.
    """
    pair_counts: PairCounts = {}
    for word, freq in word_freqs.items():
        for left, right in zip(word, word[1:], strict=False):
            pair = (left, right)
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


def select_best_pair(pair_counts: PairCounts) -> tuple[str, str] | None:
    """Pick the next pair to merge: highest frequency, ties broken lexicographically.

    This tie-break rule is what makes training deterministic across
    runs/platforms/Python versions when several pairs share the max
    frequency (common on small corpora) — the choice is never left to
    incidental dict/set ordering.

    Returns `None` when `pair_counts` is empty (nothing left to merge).
    """
    if not pair_counts:
        return None
    return min(pair_counts.items(), key=lambda item: (-item[1], item[0]))[0]


def merge_pair_in_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every non-overlapping, left-to-right occurrence of `pair` in `word`."""
    merged_symbol = pair[0] + pair[1]
    result: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            result.append(merged_symbol)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def apply_merge(word_freqs: WordFrequencies, pair: tuple[str, str]) -> WordFrequencies:
    """Apply `merge_pair_in_word` to every word, summing frequencies of words that collide.

    A collision (two different pre-merge words becoming the same word after
    merging `pair`) is rare but not impossible; frequencies are added, not
    overwritten, so the corpus's total token count is preserved.
    """
    new_word_freqs: WordFrequencies = {}
    for word, freq in word_freqs.items():
        new_word = merge_pair_in_word(word, pair)
        new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
    return new_word_freqs


@dataclass
class BPETrainingResult:
    """Result of `train_bpe`: the learned merges, in learned order, and the initial alphabet."""

    merges: list[tuple[str, str]]
    base_symbols: list[str]


def train_bpe(corpus: list[str], num_merges: int) -> BPETrainingResult:
    """Run the BPE training loop for up to `num_merges` merges.

    Stops early, before `num_merges` is reached, once no adjacent pair is
    left anywhere in the corpus (e.g. a corpus small enough that every word
    has fully collapsed into a single symbol).

    Complexity is O(num_merges * distinct_words) rather than fully
    incremental: pair counts are recomputed from the word-frequency table
    each iteration (not from raw corpus text, and not per raw occurrence —
    `word_freqs` already folds duplicates by frequency). This is a
    deliberate simplicity-over-performance choice for this project's small,
    educational corpora, not the incrementally-updated frequency structure
    a large-scale implementation would use.
    """
    word_freqs = build_word_frequencies(corpus)
    base_symbols = base_symbols_from(word_freqs)

    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pair_counts = count_pairs(word_freqs)
        best_pair = select_best_pair(pair_counts)
        if best_pair is None:
            break
        word_freqs = apply_merge(word_freqs, best_pair)
        merges.append(best_pair)

    return BPETrainingResult(merges=merges, base_symbols=base_symbols)
