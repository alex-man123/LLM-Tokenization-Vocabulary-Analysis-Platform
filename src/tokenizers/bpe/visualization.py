"""Step-by-step replay of a trained `BPETokenizer`'s learned merges (Task 8.11).

Purely a *visualization* helper: it does not reimplement BPE training and
never selects a merge itself (no call to `count_pairs`/`select_best_pair`
here). It replays the already-learned, ordered `merges` list
(`BPETokenizer.merges`, Task 2.3) over the same word-frequency
representation `train_bpe` itself starts from (`build_word_frequencies`),
applying each merge with the exact function training uses to fold a chosen
pair into the corpus (`apply_merge`). Single source of truth:

    BPE trainer (`trainer.py`) -- learns and records `merges`
                |
                v
    this module -- replays `merges` via the trainer's own `apply_merge`
                |
                v
    UI (Tokenize page) -- displays each step
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizers.bpe.trainer import WordFrequencies, apply_merge, build_word_frequencies


@dataclass(frozen=True)
class MergeStep:
    """One learned merge: which pair, and the word-frequency table right before/after it."""

    step: int  # 1-based position in the learned `merges` list
    pair: tuple[str, str]
    before: WordFrequencies
    after: WordFrequencies


def compute_merge_steps(corpus: list[str], merges: list[tuple[str, str]]) -> list[MergeStep]:
    """Replay `merges`, in order, over `corpus`, recording each step's before/after state.

    `merges` is expected to be the ordered list an already-trained
    `BPETokenizer` learned for this same `corpus` (its `.merges` property).
    This function does not verify that correspondence (doing so would mean
    retraining) — it only replays whatever merges it is given, exactly the
    way `train_bpe`'s own loop folds one merge into `word_freqs` at a time.

    An empty `merges` list returns `[]`.
    """
    word_freqs = build_word_frequencies(corpus)
    steps: list[MergeStep] = []
    for position, pair in enumerate(merges, start=1):
        after = apply_merge(word_freqs, pair)
        steps.append(MergeStep(step=position, pair=pair, before=word_freqs, after=after))
        word_freqs = after
    return steps


def format_word_freqs(word_freqs: WordFrequencies) -> list[str]:
    """Render a word-frequency table as sorted, human-readable `"symbol symbol ... (xN)"` lines.

    Ordered by descending frequency then the word's own symbols, so the
    same `word_freqs` always renders identically regardless of dict
    iteration order — determinism for display, matching this project's
    determinism guarantees elsewhere (e.g. BPE's own merge tie-breaking).
    """
    ordered = sorted(word_freqs.items(), key=lambda item: (-item[1], item[0]))
    return [f"{' '.join(word)}  (x{freq})" for word, freq in ordered]
