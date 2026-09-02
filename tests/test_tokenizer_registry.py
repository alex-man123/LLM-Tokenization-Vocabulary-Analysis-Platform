"""Unit tests for the tokenizer registry used by the UI (Task 8.1/8.2)."""

import pytest

from tokenizers.base import Tokenizer
from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer


def test_all_registered_tokenizers_are_tokenizer_subclasses():
    for tokenizer_cls in AVAILABLE_TOKENIZERS.values():
        assert issubclass(tokenizer_cls, Tokenizer)


def test_create_tokenizer_returns_a_working_instance():
    tokenizer = create_tokenizer("character")

    assert tokenizer.name == "character"


def test_create_tokenizer_raises_for_unknown_name():
    with pytest.raises(KeyError):
        create_tokenizer("does-not-exist")


def test_registry_keys_match_each_tokenizer_own_name():
    for key, tokenizer_cls in AVAILABLE_TOKENIZERS.items():
        assert tokenizer_cls().name == key
