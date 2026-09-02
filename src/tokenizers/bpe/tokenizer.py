"""BPE tokenizer: character-level Byte Pair Encoding with a word-boundary marker.

Educational/didactic variant (Sennrich et al., 2015): operates on Unicode
characters, not raw UTF-8 bytes, and uses an explicit end-of-word marker
(`</w>`) rather than treating a leading space as part of a token. This is
**not** the byte-level BPE used by GPT-style models / `tiktoken`, which
operates on bytes (256 possible values) and therefore never encounters an
"unknown" character — see `docs/architecture.md` for the full comparison.

Uses the shared `Vocabulary`/`SpecialTokens` infrastructure (Task 4.1/4.2)
for the token<->ID mapping, exactly like `CharacterTokenizer`/`WordTokenizer`
— the only BPE-specific state is the ordered list of learned merge rules.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers.base import Tokenizer
from tokenizers.bpe.trainer import END_OF_WORD, train_bpe
from vocabulary.serialization import (
    TokenizerState,
    load_tokenizer_state,
    save_tokenizer_state,
    vocabulary_tokens_in_order,
)
from vocabulary.special_tokens import SpecialTokens
from vocabulary.vocab import Vocabulary

DEFAULT_NUM_MERGES = 50


class BPETokenizer(Tokenizer):
    """Character-level BPE tokenizer with a `</w>` word-boundary marker.

    `num_merges` is a constructor argument (not a `train()` argument), so
    `train()` keeps exactly the `Tokenizer` contract's `(self, corpus) -> None`
    signature; it caps how many merge rules `train` learns (training may
    stop earlier, if the corpus runs out of pairs to merge first).

    `encode` never raises on unseen input: any symbol not present in the
    vocabulary (a character never seen during training, or — before the
    relevant merge was learned — an intermediate merged symbol that never
    occurs) falls back to `<UNK>`, the same mechanism
    `CharacterTokenizer`/`WordTokenizer` use. There is no separate
    byte-level fallback.
    """

    def __init__(self, num_merges: int = DEFAULT_NUM_MERGES) -> None:
        self._num_merges = num_merges
        self._special_tokens = SpecialTokens()
        self._merges: list[tuple[str, str]] = []
        self._merge_ranks: dict[tuple[str, str], int] = {}

    @property
    def special_tokens(self) -> SpecialTokens:
        """The `SpecialTokens` (and underlying `Vocabulary`) this tokenizer uses."""
        return self._special_tokens

    @property
    def merges(self) -> tuple[tuple[str, str], ...]:
        """Merge rules learned during training, in the order they were learned."""
        return tuple(self._merges)

    def train(self, corpus: list[str]) -> None:
        result = train_bpe(corpus, num_merges=self._num_merges)
        vocabulary = self._special_tokens.vocabulary
        for symbol in result.base_symbols:
            vocabulary.add_token(symbol)
        for left, right in result.merges:
            vocabulary.add_token(left + right)
        self._merges = result.merges
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self._merges)}

    def _apply_learned_merges(self, symbols: list[str]) -> list[str]:
        """Greedily merge the lowest-rank (earliest-learned) applicable pair, until none apply."""
        while True:
            best_rank: int | None = None
            best_index: int | None = None
            for i in range(len(symbols) - 1):
                rank = self._merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, i
            if best_index is None:
                return symbols
            symbols[best_index : best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in text.split():
            symbols = [*word, END_OF_WORD]
            tokens.extend(self._apply_learned_merges(symbols))
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
        text = "".join(vocabulary.get_token(id_) for id_ in ids)
        return text.replace(END_OF_WORD, " ").strip()

    def save(self, path: str | Path) -> None:
        state = TokenizerState(
            tokenizer_type=self.name,
            vocabulary_tokens=vocabulary_tokens_in_order(self._special_tokens.vocabulary),
            config={"merges": self._merges},
        )
        save_tokenizer_state(path, state)

    def load(self, path: str | Path) -> None:
        state = load_tokenizer_state(path, expected_type=self.name)
        self._special_tokens = SpecialTokens(Vocabulary(tokens=state.vocabulary_tokens))
        self._merges = [tuple(pair) for pair in state.config.get("merges", [])]
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self._merges)}

    @property
    def vocab_size(self) -> int:
        return self._special_tokens.vocabulary.vocab_size

    @property
    def name(self) -> str:
        return "bpe"
