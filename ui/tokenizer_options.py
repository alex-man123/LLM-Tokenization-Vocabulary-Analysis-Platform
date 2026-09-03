"""Shared tokenizer-selection helpers for the Compare (Task 8.3) and Benchmark
(Task 8.5) pages.

Both pages let a user pick from this project's own trainable tokenizers
(`tokenizers.registry.AVAILABLE_TOKENIZERS`) plus a couple of real,
optional extras, and both need the exact same "build a ready tokenizer,
never let one failure discard every other result" logic — defined once
here instead of duplicated per page.

Three kinds of tokenizer, each built differently:

- **This project's own** (character/word/BPE/WordPiece): trained live on
  the page's input text, essentially never fails.
- **Pretrained external** (Hugging Face `tokenizers`, `tiktoken` —
  Task 7.1/7.2): loaded once and cached (`st.cache_resource`, including
  failures, with a retry TTL) since construction can mean a network
  fetch; never (re)trained on the page's text.
- **Trainable external** (SentencePiece — Task 7.3): trained live on the
  page's text, like this project's own tokenizers, but not part of
  `AVAILABLE_TOKENIZERS` since it wraps an external library rather than
  being one of this project's from-scratch algorithms. Training can
  legitimately fail on very short input text (SentencePiece requires
  enough text to support its target vocabulary size) — that is a real
  constraint of the algorithm, surfaced as a per-tokenizer error, not
  hidden.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from tokenizers.adapters.huggingface_tokenizer import HuggingFaceTokenizerAdapter
from tokenizers.adapters.sentencepiece_tokenizer import SentencePieceAdapter
from tokenizers.adapters.tiktoken_tokenizer import TiktokenAdapter
from tokenizers.base import Tokenizer
from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer

PRETRAINED_EXTERNAL_FACTORIES: dict[str, Callable[[], Tokenizer]] = {
    "huggingface:bert-base-uncased": lambda: HuggingFaceTokenizerAdapter.from_pretrained(
        "bert-base-uncased"
    ),
    "tiktoken:cl100k_base": lambda: TiktokenAdapter.from_encoding_name("cl100k_base"),
}

#: Trained live on the page's text, like this project's own tokenizers —
#: not cached, since (unlike the pretrained ones above) there is no
#: network fetch to avoid repeating, and the result legitimately depends
#: on the text.
TRAINABLE_EXTRA_NAMES: tuple[str, ...] = ("sentencepiece:unigram",)

# Smaller than SentencePieceAdapter's own DEFAULT_VOCAB_SIZE (200, aimed at
# real corpus-sized training, e.g. scripts/run_experiments.py on
# data/raw/): a live demo page's input is often just a sentence or two, and
# SentencePiece's trainer raises if vocab_size exceeds what the text can
# support. This is still not guaranteed to fit very short input — that is
# a real property of the algorithm (see `build_tokenizers`'s docstring),
# surfaced as a per-tokenizer error rather than hidden.
_SENTENCEPIECE_UI_VOCAB_SIZE = 40

# A failed pretrained-tokenizer load is retried after this long, in case
# network access has come back — short enough that a genuinely fixed
# connection is noticed within a session, long enough that a still-offline
# environment doesn't retry the network on every rerun (every keystroke in
# the text box, every widget change).
EXTERNAL_TOKENIZER_RETRY_SECONDS = 300


def all_tokenizer_names() -> list[str]:
    """Every selectable tokenizer name, from all three sources above, sorted."""
    return sorted(
        {*AVAILABLE_TOKENIZERS, *PRETRAINED_EXTERNAL_FACTORIES, *TRAINABLE_EXTRA_NAMES}
    )


@dataclass(frozen=True)
class _ExternalLoadResult:
    """Either a successfully loaded external tokenizer, or why it failed."""

    tokenizer: Tokenizer | None
    error: str | None


@st.cache_resource(
    show_spinner="Loading external tokenizer...", ttl=EXTERNAL_TOKENIZER_RETRY_SECONDS
)
def _load_pretrained_external_tokenizer(name: str) -> _ExternalLoadResult:
    """Build one of the real, pretrained external tokenizers, caching success *and* failure.

    Caching only the success case would still leave a *failure* (e.g. no
    network) retried on every single rerun for as long as that tokenizer
    stays selected — caching the failure too, for
    `EXTERNAL_TOKENIZER_RETRY_SECONDS`, avoids that retry storm while
    still eventually trying again.
    """
    try:
        return _ExternalLoadResult(
            tokenizer=PRETRAINED_EXTERNAL_FACTORIES[name](), error=None
        )
    except Exception as exc:  # network/cache/model failure -> report, don't crash the page
        return _ExternalLoadResult(tokenizer=None, error=str(exc))


def build_tokenizers(names: list[str], text: str) -> tuple[list[Tokenizer], dict[str, str]]:
    """Build every tokenizer in `names`; return the ones that succeeded plus `{name: error}`.

    One tokenizer that fails to load or train must not discard results for
    every *other* tokenizer in the selection.
    """
    tokenizers: list[Tokenizer] = []
    errors: dict[str, str] = {}
    for name in names:
        if name in PRETRAINED_EXTERNAL_FACTORIES:
            result = _load_pretrained_external_tokenizer(name)
            if result.error is not None:
                errors[name] = result.error
            else:
                tokenizers.append(result.tokenizer)
            continue
        try:
            if name == "sentencepiece:unigram":
                tokenizer: Tokenizer = SentencePieceAdapter(vocab_size=_SENTENCEPIECE_UI_VOCAB_SIZE)
            else:
                tokenizer = create_tokenizer(name)
            tokenizer.train([text])
            tokenizers.append(tokenizer)
        except Exception as exc:  # a tokenizer failure must not crash the page
            errors[name] = str(exc)
    return tokenizers, errors
