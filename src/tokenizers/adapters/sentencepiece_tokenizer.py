"""Adapter over `sentencepiece` (Task 7.3), Google's Unigram/BPE subword tokenizer library.

Unlike the Hugging Face `tokenizers` and `tiktoken` adapters (Task 7.1/7.2),
which wrap an already-pretrained external tokenizer and treat `train()` as
a no-op, this adapter genuinely **trains** a fresh SentencePiece model on
whatever corpus is passed to `train(corpus)` — entirely in memory, via
`SentencePieceTrainer.Train(sentence_iterator=..., model_writer=io.BytesIO())`,
no temporary files. This is deliberate: Task 7.3 asks for training on this
project's *own* corpus (e.g. `data/raw/`), for a fair, same-corpus
comparison against `BPETokenizer`/`WordPieceTokenizer`, not for wrapping a
model trained on unrelated external data.

Default `model_type="unigram"`: SentencePiece's Unigram model is the
concrete reason this adapter exists — this project's own trainers
implement BPE and a WordPiece-like scorer (Task 2/3), but not the
Unigram algorithm itself (see `docs/unigram_notes.md`, Task 2.6, for why).
Wrapping SentencePiece's Unigram is how this project can still show real
Unigram segmentation behavior without reimplementing EM training.

SentencePiece marks word/space boundaries with a literal `▁` (U+2581)
prefix on the piece that starts a new word (e.g. `"▁lower"`), instead of
BPE's trailing `</w>` or WordPiece's `##` continuation prefix. This
adapter does not strip or alter that marker — `tokenize()` returns
SentencePiece's own piece strings exactly as produced.
"""

from __future__ import annotations

import io
from pathlib import Path

import sentencepiece as spm

from tokenizers.base import Tokenizer

DEFAULT_VOCAB_SIZE = 200

# Fixed so this adapter's special-token IDs are stable across trainings,
# mirroring (in spirit, not by sharing code) the fixed PAD/UNK/BOS/EOS order
# `vocabulary.special_tokens.SpecialTokens` uses for this project's own
# tokenizers. SentencePiece's special tokens are intrinsic to its trained
# model, not routed through this project's shared `Vocabulary` — exactly
# like the Hugging Face/tiktoken adapters, whose special tokens are simply
# whatever the wrapped library itself defines.
_PAD_ID, _UNK_ID, _BOS_ID, _EOS_ID = 0, 1, 2, 3


class SentencePieceAdapter(Tokenizer):
    """Trains and wraps a real `sentencepiece` model as a project `Tokenizer`.

    `train(corpus)` actually trains (unlike the pretrained-only Hugging
    Face/tiktoken adapters): each string in `corpus` is split into
    non-empty lines (SentencePiece's trainer expects one sentence per
    line), and a fresh Unigram (by default) model is trained to
    `vocab_size` pieces. Training on a very small corpus with too large a
    `vocab_size` raises `RuntimeError` from the underlying library (e.g.
    "Vocabulary size too high") — this is real, useful feedback, not
    something this adapter should hide or work around.
    """

    def __init__(
        self, vocab_size: int = DEFAULT_VOCAB_SIZE, *, model_type: str = "unigram"
    ) -> None:
        self._vocab_size_target = vocab_size
        self._model_type = model_type
        self._processor: spm.SentencePieceProcessor | None = None

    def _require_processor(self) -> spm.SentencePieceProcessor:
        if self._processor is None:
            raise RuntimeError(
                "No SentencePiece model is trained/loaded yet. Call `train(corpus)` "
                "or `load(path)` before using this adapter."
            )
        return self._processor

    def train(self, corpus: list[str]) -> None:
        sentences = [line for text in corpus for line in text.splitlines() if line.strip()]
        model_writer = io.BytesIO()
        spm.SentencePieceTrainer.Train(
            sentence_iterator=iter(sentences),
            model_writer=model_writer,
            vocab_size=self._vocab_size_target,
            model_type=self._model_type,
            character_coverage=1.0,
            pad_id=_PAD_ID,
            unk_id=_UNK_ID,
            bos_id=_BOS_ID,
            eos_id=_EOS_ID,
        )
        self._processor = spm.SentencePieceProcessor(model_proto=model_writer.getvalue())

    def tokenize(self, text: str) -> list[str]:
        return self._require_processor().encode(text, out_type=str)

    def encode(self, text: str) -> list[int]:
        return self._require_processor().encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._require_processor().decode(ids)

    def save(self, path: str | Path) -> None:
        """Write the trained model to `path`, in SentencePiece's own binary `.model` format.

        SentencePiece has no JSON-based serialization the way the Hugging
        Face `tokenizers` library does; `serialized_model_proto()` is its
        native save format, restorable via `SentencePieceProcessor(model_proto=...)`.
        """
        Path(path).write_bytes(self._require_processor().serialized_model_proto())

    def load(self, path: str | Path) -> None:
        """Load a model previously written by `save` (or any SentencePiece `.model` file)."""
        self._processor = spm.SentencePieceProcessor(model_proto=Path(path).read_bytes())

    @property
    def vocab_size(self) -> int:
        return self._require_processor().get_piece_size()

    @property
    def name(self) -> str:
        return f"sentencepiece:{self._model_type}"
