"""Unit tests for `BPETokenizer` (Phase 2, Task 2.3/2.4/2.5)."""

from tokenizers.bpe.tokenizer import BPETokenizer
from vocabulary.special_tokens import ORDERED_SPECIAL_TOKENS, UNK

_LOW_FAMILY_CORPUS = ["low"] * 5 + ["lower"] * 2 + ["lowest"]


def _train_low_family(num_merges: int = 4) -> BPETokenizer:
    tokenizer = BPETokenizer(num_merges=num_merges)
    tokenizer.train(_LOW_FAMILY_CORPUS)
    return tokenizer


def test_can_be_instantiated_with_default_and_explicit_num_merges():
    BPETokenizer()
    BPETokenizer(num_merges=5)


def test_name_is_bpe():
    assert BPETokenizer().name == "bpe"


def test_vocab_size_starts_with_only_special_tokens():
    assert BPETokenizer().vocab_size == len(ORDERED_SPECIAL_TOKENS)


def test_train_grows_vocabulary_with_base_symbols_and_merges():
    tokenizer = _train_low_family(num_merges=4)

    # special tokens + base alphabet {l,o,w,e,s,t,r,</w>} (8) + 4 merges
    assert tokenizer.vocab_size == len(ORDERED_SPECIAL_TOKENS) + 8 + 4
    assert len(tokenizer.merges) == 4


def test_train_is_deterministic():
    first = _train_low_family(num_merges=4)
    second = _train_low_family(num_merges=4)

    assert first.merges == second.merges


def test_tokenize_applies_learned_merges_in_rank_order():
    tokenizer = _train_low_family(num_merges=4)

    assert tokenizer.tokenize("low") == ["low</w>"]


def test_tokenize_partially_merges_a_word_never_seen_at_training():
    tokenizer = _train_low_family(num_merges=4)

    tokens = tokenizer.tokenize("lowly")

    assert tokens[0] == "low"


def test_tokenize_empty_and_whitespace_only_strings():
    tokenizer = _train_low_family()

    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []


def test_untrained_tokenizer_does_not_raise_on_encode():
    tokenizer = BPETokenizer()

    ids = tokenizer.encode("hello")

    assert all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == UNK * len(ids)


def test_encode_returns_ids_for_trained_text():
    tokenizer = _train_low_family(num_merges=4)

    ids = tokenizer.encode("low")

    assert len(ids) == 1
    assert isinstance(ids[0], int)


def test_encode_unknown_character_falls_back_to_unk_without_raising():
    tokenizer = _train_low_family(num_merges=4)

    ids = tokenizer.encode("é")  # never seen during training

    assert tokenizer.decode(ids) == UNK


def test_decode_reconstructs_multiple_trained_words_with_boundaries():
    tokenizer = _train_low_family(num_merges=4)

    assert tokenizer.decode(tokenizer.encode("low lower")) == "low lower"


def test_encode_decode_roundtrip_for_every_word_in_the_training_corpus():
    tokenizer = BPETokenizer(num_merges=10)
    tokenizer.train(_LOW_FAMILY_CORPUS)

    for word in ("low", "lower", "lowest"):
        assert tokenizer.decode(tokenizer.encode(word)) == word


def test_encode_decode_roundtrip_empty_string():
    tokenizer = _train_low_family()

    assert tokenizer.decode(tokenizer.encode("")) == ""


def test_decode_of_empty_id_list_is_empty_string():
    assert _train_low_family().decode([]) == ""


def test_encode_is_deterministic():
    tokenizer = _train_low_family(num_merges=4)

    assert tokenizer.encode("lower") == tokenizer.encode("lower")


def test_repeated_patterns_tokenize_identically():
    tokenizer = _train_low_family(num_merges=4)

    assert tokenizer.tokenize("low low") == tokenizer.tokenize("low") * 2


def test_save_and_load_roundtrip_preserves_merges_and_behavior(tmp_path):
    tokenizer = _train_low_family(num_merges=4)
    path = tmp_path / "bpe_vocab.json"
    tokenizer.save(path)

    loaded = BPETokenizer()
    loaded.load(path)

    assert loaded.merges == tokenizer.merges
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("lower") == tokenizer.encode("lower")
    assert loaded.decode(loaded.encode("lower")) == tokenizer.decode(tokenizer.encode("lower"))
