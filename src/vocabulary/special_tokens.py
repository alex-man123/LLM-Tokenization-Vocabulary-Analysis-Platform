"""Centralized special-token definitions shared by every tokenizer.

Any tokenizer (character, word, BPE, WordPiece, or a future external
adapter) must go through `SpecialTokens` instead of hardcoding its own
`<PAD>`/`<UNK>`/`<BOS>`/`<EOS>` strings or IDs, so that every tokenizer in
the project agrees on the same tokens and the same IDs.
"""

from __future__ import annotations

from vocabulary.vocab import Vocabulary

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"

# Deterministic registration order: fixes PAD=0, UNK=1, BOS=2, EOS=3 in any
# Vocabulary that registers them through SpecialTokens. Do not reorder this
# tuple once a saved vocabulary depends on it.
ORDERED_SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, BOS, EOS)


class SpecialTokens:
    """Registers PAD/UNK/BOS/EOS into a `Vocabulary` and exposes their IDs.

    Wraps a `Vocabulary` instance (creating a new one if none is given) and
    guarantees the special tokens above are registered in it, in the fixed
    order defined by `ORDERED_SPECIAL_TOKENS`. Registering the same
    `SpecialTokens`/`Vocabulary` pair more than once, or constructing this
    class around a `Vocabulary` that already has the special tokens, does
    not create duplicate entries or change existing IDs (`Vocabulary.add_token`
    is idempotent).
    """

    def __init__(self, vocabulary: Vocabulary | None = None) -> None:
        self.vocabulary = vocabulary if vocabulary is not None else Vocabulary()
        for token in ORDERED_SPECIAL_TOKENS:
            self.vocabulary.add_token(token)

    @property
    def tokens(self) -> tuple[str, ...]:
        """The special tokens, in their fixed registration order."""
        return ORDERED_SPECIAL_TOKENS

    def is_special(self, token: str) -> bool:
        """Return whether `token` is one of the special tokens."""
        return token in ORDERED_SPECIAL_TOKENS

    def is_special_id(self, id_: int) -> bool:
        """Return whether `id_` is the ID of one of the special tokens."""
        if not self.vocabulary.has_id(id_):
            return False
        return self.vocabulary.get_token(id_) in ORDERED_SPECIAL_TOKENS

    @property
    def pad_id(self) -> int:
        return self.vocabulary.get_id(PAD)

    @property
    def unk_id(self) -> int:
        return self.vocabulary.get_id(UNK)

    @property
    def bos_id(self) -> int:
        return self.vocabulary.get_id(BOS)

    @property
    def eos_id(self) -> int:
        return self.vocabulary.get_id(EOS)

    @property
    def unk_token(self) -> str:
        """The `<UNK>` token string, for tokenizers that need the literal token."""
        return UNK
