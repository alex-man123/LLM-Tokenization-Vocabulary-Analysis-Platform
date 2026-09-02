"""Character-level tokenizer: every character of the text is its own token.

Uses the shared `Vocabulary`/`SpecialTokens` infrastructure (Task 4.1/4.2)
for the token<->ID mapping — this tokenizer keeps no mapping of its own.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers.base import Tokenizer
from vocabulary.serialization import (
    TokenizerState,
    load_tokenizer_state,
    save_tokenizer_state,
    vocabulary_tokens_in_order,
)
from vocabulary.special_tokens import SpecialTokens
from vocabulary.vocab import Vocabulary


class CharacterTokenizer(Tokenizer):
    """Tokenizer where each character of the input text is one token.

    Every character is significant, including spaces, newlines, and other
    whitespace — none of them are stripped, collapsed, or treated as
    separators; they are just characters like any other. Works on any
    Unicode text, not just ASCII.

    Unknown characters (seen at encode time but never seen during `train`)
    are mapped to the `<UNK>` special token instead of raising, so `encode`
    never fails on unseen input. `decode` reconstructs the exact original
    text for any text made only of characters seen during `train`.
    """

    def __init__(self) -> None:
        self._special_tokens = SpecialTokens()

    @property
    def special_tokens(self) -> SpecialTokens:
        """The `SpecialTokens` (and underlying `Vocabulary`) this tokenizer uses."""
        return self._special_tokens

    def train(self, corpus: list[str]) -> None:
        vocabulary = self._special_tokens.vocabulary
        for text in corpus:
            for char in text:
                vocabulary.add_token(char)

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def encode(self, text: str) -> list[int]:
        vocabulary = self._special_tokens.vocabulary
        unk_id = self._special_tokens.unk_id
        return [
            vocabulary.get_id(char) if vocabulary.has_token(char) else unk_id
            for char in self.tokenize(text)
        ]

    def decode(self, ids: list[int]) -> str:
        vocabulary = self._special_tokens.vocabulary
        return "".join(vocabulary.get_token(id_) for id_ in ids)

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
        return "character"
