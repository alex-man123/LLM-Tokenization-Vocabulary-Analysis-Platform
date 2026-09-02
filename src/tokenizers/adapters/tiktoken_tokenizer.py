"""Adapter over `tiktoken` (Task 7.2), the byte-level BPE tokenizer OpenAI's models use.

Wraps a `tiktoken.Encoding` behind this project's `Tokenizer` interface
(Task 0.2), so it can sit next to `BPETokenizer`/`WordPieceTokenizer`/the
Hugging Face adapter in the Comparator (Task 5.3) without the Comparator
knowing the difference. This module only translates `tiktoken`'s existing
API — it does not re-implement its BPE algorithm.

> **Byte-level, not character-level — see `docs/limitations.md`.**
> `tiktoken` merges over UTF-8 *bytes*, not Unicode *characters*: a single
> token's raw bytes are not necessarily a complete, independently valid
> UTF-8 character (a multi-byte character can be split across adjacent
> tokens). This is a different representation from this project's own
> character-level `BPETokenizer` (Phase 2/`Phase 2`), not merely a
> different set of merges — do not describe the two as the same algorithm
> operating on the same units. `tokenize()` below decodes each token's
> bytes independently for a human-readable string (`errors="replace"`,
> since that can split a character); `decode()` never does this per-token
> decoding — it always uses `tiktoken`'s own `decode`, which concatenates
> raw bytes first and decodes UTF-8 once at the end, so it reconstructs
> the original text correctly even when individual tokens could not be
> decoded on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import tiktoken

from tokenizers.base import Tokenizer


class TiktokenAdapter(Tokenizer):
    """Wraps a `tiktoken.Encoding` as a project `Tokenizer`.

    `train()` is a deliberate no-op: this adapter wraps an
    *already-trained/pretrained* external tokenizer, exactly the case
    `Tokenizer.train`'s own docstring documents ("for adapters wrapping an
    already pretrained external tokenizer, loading should go through
    `load` instead — `train` may be a no-op in that case"). Build one via
    `from_encoding_name` (e.g. `"cl100k_base"`), `for_model` (e.g.
    `"gpt-4"`), or by passing an already-constructed `tiktoken.Encoding` to
    the constructor.
    """

    def __init__(
        self, encoding: tiktoken.Encoding | None = None, *, name: str | None = None
    ) -> None:
        self._encoding = encoding
        self._name = name or (f"tiktoken:{encoding.name}" if encoding is not None else "tiktoken")

    @classmethod
    def from_encoding_name(cls, encoding_name: str) -> TiktokenAdapter:
        """Load a named `tiktoken` encoding (e.g. `"cl100k_base"`, `"o200k_base"`)."""
        encoding = tiktoken.get_encoding(encoding_name)
        return cls(encoding, name=f"tiktoken:{encoding.name}")

    @classmethod
    def for_model(cls, model_name: str) -> TiktokenAdapter:
        """Load whichever `tiktoken` encoding `model_name` uses (e.g. `"gpt-4"`)."""
        encoding = tiktoken.encoding_for_model(model_name)
        return cls(encoding, name=f"tiktoken:{encoding.name}")

    def _require_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            raise RuntimeError(
                "No underlying tiktoken encoding is loaded. Construct via "
                "`from_encoding_name(...)`/`for_model(...)`, pass one to the "
                "constructor, or call `load(path)` before using this adapter."
            )
        return self._encoding

    def train(self, corpus: list[str]) -> None:
        """No-op — see the class docstring: this adapter wraps a pretrained external tokenizer."""
        return

    def _token_string(self, encoding: tiktoken.Encoding, token_id: int) -> str:
        """Best-effort human-readable string for one token, for `tokenize()`/visualization only.

        Decodes that single token's raw bytes as UTF-8 with
        `errors="replace"`: a byte-level token's bytes are not guaranteed
        to be a complete UTF-8 character on their own (see the module
        docstring), so this can legitimately show a replacement character
        for part of a multi-byte character split across tokens. `decode()`
        never uses this path — it always decodes the whole ID sequence's
        concatenated bytes at once, which is lossless.
        """
        token_bytes = encoding.decode_single_token_bytes(token_id)
        return token_bytes.decode("utf-8", errors="replace")

    def tokenize(self, text: str) -> list[str]:
        encoding = self._require_encoding()
        return [self._token_string(encoding, token_id) for token_id in encoding.encode(text)]

    def encode(self, text: str) -> list[int]:
        return self._require_encoding().encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._require_encoding().decode(ids)

    def save(self, path: str | Path) -> None:
        """Persist just enough to reconstruct this encoding via `load`.

        `tiktoken.Encoding` has no native save format of its own (unlike
        the Hugging Face `tokenizers` library) — its byte-pair merges live
        in a small, versioned registry the library ships and downloads by
        name, so the only state worth persisting is which named encoding
        this adapter wraps; `load` looks that name up again through
        `tiktoken.get_encoding`, not from data written here.
        """
        payload = {"tokenizer_type": self.name, "encoding_name": self._require_encoding().name}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Restore a tokenizer previously written by `save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._encoding = tiktoken.get_encoding(payload["encoding_name"])
        self._name = payload.get("tokenizer_type", self._name)

    @property
    def vocab_size(self) -> int:
        return self._require_encoding().n_vocab

    @property
    def name(self) -> str:
        return self._name
