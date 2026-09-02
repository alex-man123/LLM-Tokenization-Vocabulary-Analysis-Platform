"""Abstract contract shared by every tokenizer in this project.

Both from-scratch implementations (character-level, word-level, BPE,
WordPiece, ...) and thin adapters over external libraries (Hugging Face
`tokenizers`, `tiktoken`, `sentencepiece`) must subclass `Tokenizer` and
honour this contract, so that the vocabulary manager, benchmarking layer,
experiments runner and UI can treat every tokenizer uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Tokenizer(ABC):
    """Common interface for all tokenizers (custom implementations or external adapters)."""

    @abstractmethod
    def train(self, corpus: list[str]) -> None:
        """Learn a vocabulary from `corpus`.

        Args:
            corpus: raw text documents/lines used to build the vocabulary
                (e.g. merge rules for BPE, word frequencies for a
                word-level tokenizer).

        Populates whatever internal state `tokenize`/`encode`/`decode` rely
        on. Calling `train` again is expected to replace the previous
        vocabulary, not merge into it. For adapters wrapping an already
        pretrained external tokenizer, loading should go through `load`
        instead — `train` may be a no-op in that case.
        """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Split `text` into its string subword/word/character units.

        Returns human-readable tokens (e.g. `["un", "believ", "able"]`)
        without converting them to IDs. Distinct from `encode`: `tokenize`
        exists mainly for visualization/debugging, so a caller can inspect
        how a piece of text was segmented.
        """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert `text` into a list of vocabulary IDs.

        Equivalent to tokenizing `text` and looking up each token's ID in
        the vocabulary (an unknown token maps to the tokenizer's UNK id, if
        one is defined). This is the representation the benchmarking layer
        measures (token counts, compression ratio, encode/decode timing).
        """

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of vocabulary IDs back into text.

        Must be the inverse of `encode` for any ID sequence produced by
        this tokenizer instance: `decode(encode(text))` should reconstruct
        `text` as closely as the tokenizer's algorithm allows (exact for
        lossless tokenizers; some information, such as original
        whitespace, may not round-trip for tokenizers that do not
        explicitly encode it).
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the trained/loaded vocabulary and tokenizer state to `path`.

        Must write everything `load` needs to fully reconstruct this
        tokenizer's behaviour (vocabulary, merge rules, special tokens,
        configuration) without requiring `train` to be called again.
        """

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore vocabulary and tokenizer state previously written by `save`.

        After `load` returns, `tokenize`/`encode`/`decode` must behave
        identically to the tokenizer instance that produced the file at
        `path`.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of entries in the tokenizer's vocabulary.

        Reflects the vocabulary actually available after `train` or `load`
        has been called.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier for this tokenizer (e.g. "bpe", "tiktoken:cl100k_base").

        Used to label results in benchmarking output and the UI; should be
        human-readable and distinguish between different configurations of
        the same algorithm where relevant.
        """
