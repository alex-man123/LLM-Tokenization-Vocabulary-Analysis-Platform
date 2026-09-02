"""WordPiece tokenizer: subword tokenization via greedy longest-match on a trained vocabulary.

> **Disclaimer:** training (`tokenizers.wordpiece.trainer`) uses a
> simplified, didactic scoring rule inspired by the original WordPiece
> objective (Schuster & Nakajima, 2012), not the likelihood-based training
> procedure used by BERT/Hugging Face. See `docs/wordpiece_explained.md`
> and the trainer module's docstring for the full disclaimer.

Uses the shared `Vocabulary`/`SpecialTokens` infrastructure (Task 4.1/4.2)
for the token<->ID mapping, exactly like `CharacterTokenizer`/`WordTokenizer`/
`BPETokenizer`. Unlike BPE, no ordered merge-rule list is kept: encode is
greedy longest-match-first against the trained vocabulary, so the
vocabulary alone is enough to reconstruct tokenization behaviour after
`load` (Task 3.3/3.4).
"""

from __future__ import annotations

from pathlib import Path

from tokenizers.base import Tokenizer
from tokenizers.word_tokenizer import WordTokenizer
from tokenizers.wordpiece.trainer import CONTINUATION_PREFIX, train_wordpiece
from vocabulary.serialization import (
    TokenizerState,
    load_tokenizer_state,
    save_tokenizer_state,
    vocabulary_tokens_in_order,
)
from vocabulary.special_tokens import SpecialTokens
from vocabulary.vocab import Vocabulary

DEFAULT_VOCAB_SIZE = 100

_WORD_SPLITTER = WordTokenizer()


def _greedy_longest_match(word: str, vocabulary: Vocabulary) -> list[str] | None:
    """Greedily split `word` into the longest available vocabulary tokens, left to right.

    At each position, tries the longest remaining substring first (prefixed
    with `CONTINUATION_PREFIX` unless it starts at position 0), shrinking
    until a match is found in `vocabulary`. Returns `None` — signalling
    total failure for the whole word, per standard WordPiece/BERT
    behaviour — as soon as any position has no match at all, rather than
    returning the tokens matched so far.
    """
    if not word:
        return []

    chars = word
    length = len(chars)
    tokens: list[str] = []
    start = 0
    while start < length:
        end = length
        matched: str | None = None
        while start < end:
            candidate = chars[start:end]
            if start > 0:
                candidate = f"{CONTINUATION_PREFIX}{candidate}"
            if vocabulary.has_token(candidate):
                matched = candidate
                break
            end -= 1
        if matched is None:
            return None
        tokens.append(matched)
        start = end
    return tokens


class WordPieceTokenizer(Tokenizer):
    """Subword tokenizer using WordPiece's `##`-continuation convention and greedy encode.

    `vocab_size` is a constructor argument (not a `train()` argument), so
    `train()` keeps exactly the `Tokenizer` contract's `(self, corpus) -> None`
    signature. It targets the size of the *learned* WordPiece vocabulary
    (initial alphabet + merged pieces) — the tokenizer's own `vocab_size`
    property additionally includes the 4 special tokens `SpecialTokens`
    always registers, mirroring how `BPETokenizer.num_merges` relates to
    its `vocab_size` property.

    `tokenize`/`encode` never raise on unseen input: a word with no
    matching vocabulary entry at some position becomes a single `<UNK>`
    token for the whole word (never a partial tokenization), the same
    "whole word fails together" rule BERT's WordPiece tokenizer uses.
    """

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        self._vocab_size_target = vocab_size
        self._special_tokens = SpecialTokens()

    @property
    def special_tokens(self) -> SpecialTokens:
        """The `SpecialTokens` (and underlying `Vocabulary`) this tokenizer uses."""
        return self._special_tokens

    def train(self, corpus: list[str]) -> None:
        result = train_wordpiece(corpus, vocab_size=self._vocab_size_target)
        vocabulary = self._special_tokens.vocabulary
        for token in result.vocabulary_tokens:
            vocabulary.add_token(token)

    def tokenize(self, text: str) -> list[str]:
        vocabulary = self._special_tokens.vocabulary
        unk_token = self._special_tokens.unk_token
        tokens: list[str] = []
        for word in _WORD_SPLITTER.tokenize(text):
            word_tokens = _greedy_longest_match(word, vocabulary)
            tokens.extend(word_tokens if word_tokens is not None else [unk_token])
        return tokens

    def encode(self, text: str) -> list[int]:
        vocabulary = self._special_tokens.vocabulary
        unk_id = self._special_tokens.unk_id
        return [
            vocabulary.get_id(token) if vocabulary.has_token(token) else unk_id
            for token in self.tokenize(text)
        ]

    def decode(self, ids: list[int]) -> str:
        vocabulary = self._special_tokens.vocabulary
        words: list[str] = []
        for id_ in ids:
            token = vocabulary.get_token(id_)
            if token.startswith(CONTINUATION_PREFIX) and words:
                words[-1] += token[len(CONTINUATION_PREFIX) :]
            else:
                words.append(token)
        return " ".join(words)

    def save(self, path: str | Path) -> None:
        state = TokenizerState(
            tokenizer_type=self.name,
            vocabulary_tokens=vocabulary_tokens_in_order(self._special_tokens.vocabulary),
        )
        save_tokenizer_state(path, state)

    def load(self, path: str | Path) -> None:
        state = load_tokenizer_state(path, expected_type=self.name)
        self._special_tokens = SpecialTokens(Vocabulary(tokens=state.vocabulary_tokens))

    @property
    def vocab_size(self) -> int:
        return self._special_tokens.vocabulary.vocab_size

    @property
    def name(self) -> str:
        return "wordpiece"
