"""Adapter over the pip-installed Hugging Face `tokenizers` library (Task 7.1).

Wraps an already-trained/pretrained `tokenizers.Tokenizer` (the external
library — e.g. the BPE tokenizer GPT-2 uses, or the WordPiece tokenizer
BERT uses) behind this project's `Tokenizer` interface (Task 0.2), so a
real, production-grade tokenizer can sit next to
`BPETokenizer`/`WordPieceTokenizer`/etc. in the Comparator (Task 5.3)
without the Comparator knowing the difference. This module only translates
the existing HF API into this project's contract — it does not
re-implement BPE, WordPiece, vocabulary training, or merge rules.

> **Name collision note:** this project's own core package is also named
> `tokenizers` (`src/tokenizers/`), and `src` is on `sys.path`
> (`pythonpath = ["src"]`, `pyproject.toml`) for every test/run in this
> repository — so a plain `import tokenizers` anywhere else in this
> project means *this project's* package, never the pip-installed
> library. Renaming this project's package to resolve that would be a
> large, unrelated refactor, so instead this module is the one place that
> needs the real library, and it loads it via
> `_import_installed_tokenizers_library()` below: a temporary, local
> `sys.path`/`sys.modules` adjustment that is fully reverted immediately
> after, leaving every other module's `import tokenizers` unaffected.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import ModuleType

_SRC_DIR = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def _prefer_installed_tokenizers_package():
    """Temporarily hide this project's own `tokenizers` package from the import system.

    Removes `_SRC_DIR` from `sys.path` and drops any already-cached
    `tokenizers`/`tokenizers.*` entries from `sys.modules` for the
    duration of the `with` block, so a plain `import tokenizers` inside it
    resolves to the pip-installed library instead of `src/tokenizers`.
    Restores both exactly as they were afterward, so this project's own
    `tokenizers` package — and every module that already imported from it
    — is completely unaffected once the block exits.
    """
    saved_path = sys.path[:]
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "tokenizers" or name.startswith("tokenizers.")
    }
    for name in saved_modules:
        del sys.modules[name]
    sys.path = [p for p in sys.path if Path(p).resolve() != _SRC_DIR]
    try:
        yield
    finally:
        sys.path = saved_path
        for name in list(sys.modules):
            if name == "tokenizers" or name.startswith("tokenizers."):
                del sys.modules[name]
        sys.modules.update(saved_modules)


def _import_installed_tokenizers_library() -> ModuleType:
    """Import and return the real, pip-installed `tokenizers` library module."""
    with _prefer_installed_tokenizers_package():
        try:
            import tokenizers as installed_library
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'tokenizers' package (pip install tokenizers) is required by "
                "HuggingFaceTokenizerAdapter but is not installed."
            ) from exc
        return installed_library


_hf_tokenizers = _import_installed_tokenizers_library()
HFTokenizer = _hf_tokenizers.Tokenizer
# Re-exported so other code that needs the real library's submodules (e.g.
# to build/train a `tokenizers.Tokenizer` from scratch) does not have to
# repeat the same `sys.path`/`sys.modules` workaround itself — a plain
# `from tokenizers.models import ...` elsewhere in this project would hit
# the exact same name collision described above.
hf_models = _hf_tokenizers.models
hf_trainers = _hf_tokenizers.trainers
hf_pre_tokenizers = _hf_tokenizers.pre_tokenizers

from tokenizers.base import Tokenizer  # noqa: E402 (must follow the workaround above)


class HuggingFaceTokenizerAdapter(Tokenizer):
    """Wraps a pip-installed `tokenizers.Tokenizer` as a project `Tokenizer`.

    `train()` is a deliberate no-op: this adapter wraps an
    *already-trained/pretrained* external tokenizer, exactly the case
    `Tokenizer.train`'s own docstring documents ("for adapters wrapping an
    already pretrained external tokenizer, loading should go through
    `load` instead — `train` may be a no-op in that case"). Load a
    tokenizer via `from_pretrained` (a Hugging Face Hub identifier),
    `from_file` (a local `tokenizer.json`), or by passing an
    already-constructed `tokenizers.Tokenizer` to the constructor.

    `tokenize`/`encode`/`decode`/`vocab_size` all delegate directly to the
    wrapped tokenizer's own API (`Encoding.tokens`/`.ids`, `.decode`,
    `.get_vocab_size()`) — token IDs are the HF tokenizer's real IDs, never
    re-numbered or invented.
    """

    def __init__(self, tokenizer: HFTokenizer | None = None, *, name: str | None = None) -> None:
        self._tokenizer = tokenizer
        self._name = name or "huggingface"

    @classmethod
    def from_pretrained(cls, identifier: str) -> HuggingFaceTokenizerAdapter:
        """Load a tokenizer published on the Hugging Face Hub (e.g. `"bert-base-uncased"`)."""
        return cls(HFTokenizer.from_pretrained(identifier), name=f"huggingface:{identifier}")

    @classmethod
    def from_file(
        cls, path: str | Path, *, name: str = "huggingface"
    ) -> HuggingFaceTokenizerAdapter:
        """Load a tokenizer from a local `tokenizer.json` file (the HF library's native format)."""
        return cls(HFTokenizer.from_file(str(path)), name=name)

    def _require_tokenizer(self) -> HFTokenizer:
        if self._tokenizer is None:
            raise RuntimeError(
                "No underlying Hugging Face tokenizer is loaded. Construct via "
                "`from_pretrained(...)`/`from_file(...)`, pass one to the "
                "constructor, or call `load(path)` before using this adapter."
            )
        return self._tokenizer

    def train(self, corpus: list[str]) -> None:
        """No-op — see the class docstring: this adapter wraps a pretrained external tokenizer."""
        return

    def tokenize(self, text: str) -> list[str]:
        return self._require_tokenizer().encode(text).tokens

    def encode(self, text: str) -> list[int]:
        return self._require_tokenizer().encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._require_tokenizer().decode(ids)

    def save(self, path: str | Path) -> None:
        """Write the wrapped tokenizer to `path`, in the HF library's native JSON format."""
        self._require_tokenizer().save(str(path))

    def load(self, path: str | Path) -> None:
        """Load a tokenizer previously written by `save` (or any HF `tokenizer.json` file)."""
        self._tokenizer = HFTokenizer.from_file(str(path))

    @property
    def vocab_size(self) -> int:
        return self._require_tokenizer().get_vocab_size()

    @property
    def name(self) -> str:
        return self._name
