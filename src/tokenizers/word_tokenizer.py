"""Word-level tokenizer: splits text into words and individual punctuation marks.

Uses the shared `Vocabulary`/`SpecialTokens` infrastructure (Task 4.1/4.2)
for the token<->ID mapping — this tokenizer keeps no mapping of its own.
"""

from __future__ import annotations

import re
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

# A maximal run of Unicode "word" characters (letters/digits/underscore) is
# one token; any other non-whitespace character is its own single-character
# token; whitespace itself is never captured, so it is a pure separator.
_WORD_OR_PUNCTUATION = re.compile(r"\w+|[^\w\s]")


class WordTokenizer(Tokenizer):
    """Tokenizer that splits text into words and punctuation, regex-based.

    Tokenization rule (see `_WORD_OR_PUNCTUATION`):

    - a maximal run of word characters is one token, e.g. `"hello"`,
      `"héllo123"` (Unicode-aware, not ASCII-only);
    - every other non-whitespace character becomes its own one-character
      token — punctuation is never merged with a word, and repeated
      punctuation like `"..."` becomes three separate `"."` tokens rather
      than one `"..."` token;
    - whitespace (spaces, tabs, newlines, and runs of any of these) is a
      pure separator: it never becomes a token itself, so `"a  b"`,
      `"a\\nb"`, and `"a b"` all tokenize to `["a", "b"]`;
    - the empty string and a whitespace-only string both tokenize to `[]`.

    This is a simple, deterministic split, not a linguistically-aware one:
    contractions are not special-cased, so `"don't"` tokenizes as
    `["don", "'", "t"]`.

    `decode` joins tokens with a single space. This does not reconstruct
    original spacing/adjacency exactly — e.g. `"hello!"` round-trips to
    `"hello !"` — which is a documented limitation of this tokenizer, not a
    bug.
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
            for token in self.tokenize(text):
                vocabulary.add_token(token)

    def tokenize(self, text: str) -> list[str]:
        return _WORD_OR_PUNCTUATION.findall(text)

    def encode(self, text: str) -> list[int]:
        vocabulary = self._special_tokens.vocabulary
        unk_id = self._special_tokens.unk_id
        return [
            vocabulary.get_id(token) if vocabulary.has_token(token) else unk_id
            for token in self.tokenize(text)
        ]

    def decode(self, ids: list[int]) -> str:
        vocabulary = self._special_tokens.vocabulary
        return " ".join(vocabulary.get_token(id_) for id_ in ids)

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
        return "word"
