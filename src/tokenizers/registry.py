"""Registry of tokenizer implementations available to the UI and other callers.

A single source of truth for "which tokenizers exist", so the Streamlit UI
(Tokenize, Compare, ...) never hardcodes this list in more than one place.
"""

from __future__ import annotations

from tokenizers.base import Tokenizer
from tokenizers.bpe.tokenizer import BPETokenizer
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer

AVAILABLE_TOKENIZERS: dict[str, type[Tokenizer]] = {
    "character": CharacterTokenizer,
    "word": WordTokenizer,
    "bpe": BPETokenizer,
}


def create_tokenizer(name: str) -> Tokenizer:
    """Instantiate a fresh tokenizer by its registered name.

    Raises:
        KeyError: if `name` is not in `AVAILABLE_TOKENIZERS`.
    """
    return AVAILABLE_TOKENIZERS[name]()
