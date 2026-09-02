"""Unit tests for the centralized special tokens (Phase 4, Task 4.2)."""

from vocabulary.special_tokens import BOS, EOS, PAD, UNK, SpecialTokens
from vocabulary.vocab import Vocabulary


def test_special_tokens_registered_in_fixed_deterministic_order():
    special = SpecialTokens()

    assert special.pad_id == 0
    assert special.unk_id == 1
    assert special.bos_id == 2
    assert special.eos_id == 3


def test_all_four_special_tokens_exist_in_the_vocabulary():
    special = SpecialTokens()

    for token in (PAD, UNK, BOS, EOS):
        assert special.vocabulary.has_token(token)


def test_unk_token_and_unk_id_are_accessible():
    special = SpecialTokens()

    assert special.unk_token == UNK
    assert special.vocabulary.get_id(special.unk_token) == special.unk_id


def test_is_special_true_for_special_tokens_false_for_others():
    special = SpecialTokens()

    assert special.is_special(PAD) is True
    assert special.is_special(UNK) is True
    assert special.is_special(BOS) is True
    assert special.is_special(EOS) is True
    assert special.is_special("hello") is False


def test_is_special_id_true_for_special_ids_false_for_others():
    special = SpecialTokens()
    special.vocabulary.add_token("hello")

    assert special.is_special_id(special.unk_id) is True
    assert special.is_special_id(special.vocabulary.get_id("hello")) is False
    assert special.is_special_id(999) is False


def test_tokens_property_lists_all_special_tokens_in_order():
    special = SpecialTokens()

    assert special.tokens == (PAD, UNK, BOS, EOS)


def test_special_tokens_do_not_duplicate_on_an_already_populated_vocabulary():
    vocab = Vocabulary()
    vocab.add_token(PAD)

    special = SpecialTokens(vocab)

    assert special.vocabulary is vocab
    assert vocab.vocab_size == 4
    assert special.pad_id == 0


def test_reconstructing_special_tokens_on_same_vocabulary_is_idempotent():
    vocab = Vocabulary()
    first = SpecialTokens(vocab)
    second = SpecialTokens(vocab)

    assert first.pad_id == second.pad_id
    assert first.unk_id == second.unk_id
    assert first.bos_id == second.bos_id
    assert first.eos_id == second.eos_id
    assert vocab.vocab_size == 4
