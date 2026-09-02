"""Special-token integration tests for the tokenizers (Task 1.3, 2.4).

Verifies that CharacterTokenizer/WordTokenizer/BPETokenizer use the
centralized `SpecialTokens`/`Vocabulary` infrastructure from Task 4.1/4.2,
rather than defining their own special tokens or IDs.
"""

import pytest

from tokenizers.bpe.tokenizer import BPETokenizer
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer
from vocabulary.special_tokens import BOS, EOS, ORDERED_SPECIAL_TOKENS, PAD, UNK

TOKENIZER_CLASSES = [CharacterTokenizer, WordTokenizer, BPETokenizer]


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_all_special_tokens_are_registered_from_construction(tokenizer_cls):
    tokenizer = tokenizer_cls()

    for token in (PAD, UNK, BOS, EOS):
        assert tokenizer.special_tokens.vocabulary.has_token(token)


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_special_token_ids_come_from_the_shared_vocabulary_manager(tokenizer_cls):
    tokenizer = tokenizer_cls()
    special = tokenizer.special_tokens

    assert special.pad_id == special.vocabulary.get_id(PAD)
    assert special.unk_id == special.vocabulary.get_id(UNK)
    assert special.bos_id == special.vocabulary.get_id(BOS)
    assert special.eos_id == special.vocabulary.get_id(EOS)


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_special_tokens_are_not_duplicated_after_training(tokenizer_cls):
    tokenizer = tokenizer_cls()
    size_before_training = tokenizer.vocab_size

    tokenizer.train(["hello world"])

    for token in ORDERED_SPECIAL_TOKENS:
        assert tokenizer.special_tokens.vocabulary.has_token(token)
    assert tokenizer.vocab_size >= size_before_training


@pytest.mark.parametrize("tokenizer_cls", TOKENIZER_CLASSES)
def test_unk_used_for_unknown_input(tokenizer_cls):
    tokenizer = tokenizer_cls()
    tokenizer.train(["a"])

    ids = tokenizer.encode("unseen-content-not-in-corpus")

    assert tokenizer.special_tokens.unk_id in ids
