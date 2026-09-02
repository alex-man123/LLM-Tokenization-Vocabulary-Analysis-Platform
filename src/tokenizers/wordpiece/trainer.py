"""WordPiece training: likelihood-inspired merge scoring and the training loop.

> **Disclaimer:** the scoring rule implemented here
> (`score(a, b) = freq(a, b) / (freq(a) * freq(b))`) is a **simplified
> WordPiece training implementation inspired by the original WordPiece
> objective** (Schuster & Nakajima, 2012), not a reproduction of the
> training procedure used by BERT/Hugging Face's `tokenizers` library. Real
> WordPiece training optimizes a language-model likelihood over
> segmentations, with additional validation steps this project does not
> implement. Do not describe this module as training "exactly how BERT
> trains its tokenizer" — it approximates the same intuition (prefer pairs
> that co-occur often *relative to* their individual frequencies) with a
> much simpler formula.

Pure functions, independent of `Vocabulary`/`Tokenizer` — one exception is
pre-tokenization, which deliberately reuses `WordTokenizer.tokenize`
(Task 1.2) rather than re-implementing word/punctuation splitting.
`tokenizers.wordpiece.tokenizer.WordPieceTokenizer` is what wires the rest
of this module into the project (vocabulary registration, encode, decode).
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizers.word_tokenizer import WordTokenizer

CONTINUATION_PREFIX = "##"

WordFrequencies = dict[tuple[str, ...], int]
SymbolFrequencies = dict[str, int]
PairCounts = dict[tuple[str, str], int]
PairScores = dict[tuple[str, str], float]

_WORD_SPLITTER = WordTokenizer()


def word_to_symbols(word: str) -> tuple[str, ...]:
    """Represent `word` as characters, marking every non-initial one as a continuation.

    `"hello"` -> `("h", "##e", "##l", "##l", "##o")`: the first character
    carries no prefix (it can start a new word during encode), every other
    character is prefixed with `CONTINUATION_PREFIX` ("##") to mark that it
    only ever continues the symbol before it. A single-character word stays
    a single, unprefixed symbol (`"a"` -> `("a",)`, not `("##a",)`); an
    empty string produces an empty tuple.
    """
    if not word:
        return ()
    return (word[0], *(f"{CONTINUATION_PREFIX}{char}" for char in word[1:]))


def build_word_frequencies(corpus: list[str]) -> WordFrequencies:
    """Pre-tokenize `corpus` into words and count their WordPiece symbol representations.

    Word/punctuation splitting is delegated to `WordTokenizer.tokenize`
    (Task 1.2) instead of being re-implemented here, so WordPiece and the
    word-level tokenizer agree on what counts as a "word" (e.g. `"hello!"`
    splits into the words `"hello"` and `"!"`, each converted to symbols
    independently — punctuation never fuses with an adjacent word). An
    empty corpus, or a corpus of only whitespace, produces `{}`.
    """
    word_freqs: WordFrequencies = {}
    for text in corpus:
        for word in _WORD_SPLITTER.tokenize(text):
            symbols = word_to_symbols(word)
            word_freqs[symbols] = word_freqs.get(symbols, 0) + 1
    return word_freqs


def base_symbols_from(word_freqs: WordFrequencies) -> list[str]:
    """Return the initial alphabet appearing in `word_freqs`, in first-seen order.

    Deterministic regardless of hash randomization, exactly like BPE's
    `base_symbols_from`: `word_freqs` is built by iterating the corpus in
    order, and dict iteration order is insertion order.
    """
    seen: dict[str, None] = {}
    for word in word_freqs:
        for symbol in word:
            seen.setdefault(symbol, None)
    return list(seen)


def count_symbol_frequencies(word_freqs: WordFrequencies) -> SymbolFrequencies:
    """Count how often each symbol occurs across `word_freqs`, weighted by word frequency."""
    symbol_freqs: SymbolFrequencies = {}
    for word, freq in word_freqs.items():
        for symbol in word:
            symbol_freqs[symbol] = symbol_freqs.get(symbol, 0) + freq
    return symbol_freqs


def count_pairs(word_freqs: WordFrequencies) -> PairCounts:
    """Count every adjacent symbol pair across `word_freqs`, weighted by word frequency.

    Identical in shape to BPE's `count_pairs` — this is the raw-frequency
    count that WordPiece's scoring (`score_pairs`) normalizes by individual
    symbol frequency, and that BPE would use directly as its merge
    criterion.
    """
    pair_counts: PairCounts = {}
    for word, freq in word_freqs.items():
        for left, right in zip(word, word[1:], strict=False):
            pair = (left, right)
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


def score_pairs(pair_counts: PairCounts, symbol_freqs: SymbolFrequencies) -> PairScores:
    """Score every candidate pair as `freq(a, b) / (freq(a) * freq(b))`.

    This is the simplified WordPiece merge criterion (see the module
    docstring's disclaimer): a pair scores highest when it co-occurs often
    *relative to* how often its two symbols occur on their own, not simply
    when it is the most frequent pair in absolute terms — that would be the
    BPE criterion (`count_pairs` alone). A pair whose components are
    individually rare can outscore a far more frequent pair whose
    components are common.

    Defensive against division by zero: a pair from `pair_counts` whose
    left or right symbol is missing from `symbol_freqs` (which should not
    happen when `symbol_freqs` was computed from the same `word_freqs` via
    `count_symbol_frequencies`, since a symbol cannot appear in a pair
    without itself having positive frequency) is skipped rather than
    raising `ZeroDivisionError`.
    """
    scores: PairScores = {}
    for pair, pair_freq in pair_counts.items():
        left, right = pair
        left_freq = symbol_freqs.get(left, 0)
        right_freq = symbol_freqs.get(right, 0)
        if left_freq == 0 or right_freq == 0:
            continue
        scores[pair] = pair_freq / (left_freq * right_freq)
    return scores


def select_best_pair(pair_scores: PairScores) -> tuple[str, str] | None:
    """Pick the next pair to merge: highest score, ties broken lexicographically.

    Same tie-break rule as BPE's `select_best_pair` (lexicographically
    smaller `(a, b)` wins), applied to `score` instead of raw frequency, so
    training is deterministic whenever several pairs share the same top
    score — common on small corpora.

    Returns `None` when `pair_scores` is empty (nothing left to merge).
    """
    if not pair_scores:
        return None
    return min(pair_scores.items(), key=lambda item: (-item[1], item[0]))[0]


def merged_symbol(left: str, right: str) -> str:
    """Concatenate `left` and `right` into the symbol their merge produces.

    `right`'s continuation prefix is stripped before concatenating (it only
    marked `right` as continuing `left`; the merged symbol continues
    whatever `left` continued). `left`'s own prefix (present or absent) is
    kept as-is, so the merged symbol's "starts a word" status always
    matches `left`'s: `merged_symbol("un", "##believ") == "unbeliev"`,
    `merged_symbol("##be", "##lievable") == "##believable"`.
    """
    if right.startswith(CONTINUATION_PREFIX):
        right = right[len(CONTINUATION_PREFIX) :]
    return left + right


def merge_pair_in_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every non-overlapping, left-to-right occurrence of `pair` in `word`."""
    merged = merged_symbol(*pair)
    result: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            result.append(merged)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def apply_merge(word_freqs: WordFrequencies, pair: tuple[str, str]) -> WordFrequencies:
    """Apply `merge_pair_in_word` to every word, summing frequencies of words that collide."""
    new_word_freqs: WordFrequencies = {}
    for word, freq in word_freqs.items():
        new_word = merge_pair_in_word(word, pair)
        new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
    return new_word_freqs


@dataclass
class WordPieceTrainingResult:
    """Result of `train_wordpiece`: the final vocabulary (base alphabet + learned merges).

    Unlike BPE, no ordered merge-rule list is kept: WordPiece encode is
    greedy longest-match against this vocabulary, not sequential merge
    replay, so the final vocabulary is all `WordPieceTokenizer` needs
    (Task 3.3/3.4).
    """

    vocabulary_tokens: list[str]


def train_wordpiece(corpus: list[str], vocab_size: int) -> WordPieceTrainingResult:
    """Run the WordPiece training loop until `vocab_size` tokens are learned.

    Starts from the base alphabet (every symbol in `build_word_frequencies`,
    first-seen order) and repeatedly merges the highest-scoring pair
    (`score_pairs`/`select_best_pair`) until the vocabulary reaches
    `vocab_size` or no candidate pair remains.

    `vocab_size` is a target, not a hard cap on this function's own output:
    if the corpus's base alphabet alone already has at least `vocab_size`
    symbols, no merges are performed and the (larger) base alphabet is
    returned as-is — a WordPiece vocabulary must always contain every
    training-corpus character, or greedy longest-match could fail on
    training data itself. This mirrors how BPE's `num_merges` is a budget,
    not a guarantee of an exact resulting vocabulary size.

    A merge that produces a symbol already present in the vocabulary (rare,
    but possible) does not grow the vocabulary that iteration; this keeps
    `len(vocabulary_tokens)` an accurate count of *distinct* tokens rather
    than double-counting a collision.
    """
    word_freqs = build_word_frequencies(corpus)
    vocabulary_tokens = base_symbols_from(word_freqs)
    seen = set(vocabulary_tokens)

    while len(vocabulary_tokens) < vocab_size:
        symbol_freqs = count_symbol_frequencies(word_freqs)
        pair_counts = count_pairs(word_freqs)
        pair_scores = score_pairs(pair_counts, symbol_freqs)
        best_pair = select_best_pair(pair_scores)
        if best_pair is None:
            break

        word_freqs = apply_merge(word_freqs, best_pair)
        new_symbol = merged_symbol(*best_pair)
        if new_symbol not in seen:
            vocabulary_tokens.append(new_symbol)
            seen.add(new_symbol)

    return WordPieceTrainingResult(vocabulary_tokens=vocabulary_tokens)
