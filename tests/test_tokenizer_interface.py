"""Contract tests for the abstract `Tokenizer` interface (Phase 0, Task 0.2).

These only verify the shape of the contract itself — no concrete tokenizer
exists yet.
"""

import inspect
from abc import ABC

import pytest

from tokenizers.base import Tokenizer


def test_tokenizer_is_abstract_and_cannot_be_instantiated():
    assert issubclass(Tokenizer, ABC)
    with pytest.raises(TypeError):
        Tokenizer()


def test_tokenizer_declares_required_contract():
    required_methods = {"train", "tokenize", "encode", "decode", "save", "load"}
    required_properties = {"vocab_size", "name"}

    abstract_members = Tokenizer.__abstractmethods__
    assert required_methods <= abstract_members
    assert required_properties <= abstract_members


def test_tokenizer_vocab_size_and_name_are_properties():
    assert isinstance(inspect.getattr_static(Tokenizer, "vocab_size"), property)
    assert isinstance(inspect.getattr_static(Tokenizer, "name"), property)
