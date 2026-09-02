"""Generalized save/load for any tokenizer built on the shared Vocabulary Manager.

Used by `CharacterTokenizer`, `WordTokenizer`, and `BPETokenizer` so each one
does not reinvent its own JSON format; the only thing that differs between
them is the `config` document (e.g. BPE's merge rules).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vocabulary.vocab import Vocabulary

FORMAT_VERSION = "1.0"


@dataclass
class TokenizerState:
    """Everything needed to reconstruct a tokenizer's vocabulary and config.

    `vocabulary_tokens` must be ordered by ID (index == ID), matching what
    `vocabulary_tokens_in_order` produces and what `Vocabulary(tokens=...)`
    expects to reproduce the same IDs on reload.
    """

    tokenizer_type: str
    vocabulary_tokens: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    trained_at: str | None = None


def vocabulary_tokens_in_order(vocabulary: Vocabulary) -> list[str]:
    """Return `vocabulary`'s tokens as a list ordered by ID (index == ID)."""
    return [vocabulary.get_token(i) for i in range(vocabulary.vocab_size)]


def save_tokenizer_state(path: str | Path, state: TokenizerState) -> None:
    """Write `state` to `path` as a versioned, human-readable JSON document."""
    payload = {
        "version": FORMAT_VERSION,
        "tokenizer_type": state.tokenizer_type,
        "vocab_size": len(state.vocabulary_tokens),
        "trained_at": state.trained_at or datetime.now(UTC).isoformat(),
        "vocabulary": state.vocabulary_tokens,
        "config": state.config,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tokenizer_state(path: str | Path, *, expected_type: str) -> TokenizerState:
    """Read a `TokenizerState` from `path`.

    Raises:
        ValueError: if the file was saved by a different `tokenizer_type`
            than `expected_type` — loading a word-tokenizer file into a
            character tokenizer (for example) is a caller error, not
            something to silently paper over.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tokenizer_type = payload.get("tokenizer_type")
    if tokenizer_type != expected_type:
        raise ValueError(
            f"Cannot load a {tokenizer_type!r} vocabulary file into a {expected_type!r} tokenizer"
        )
    return TokenizerState(
        tokenizer_type=tokenizer_type,
        vocabulary_tokens=payload["vocabulary"],
        config=payload.get("config", {}),
        trained_at=payload.get("trained_at"),
    )
