"""Unit tests for `WordPieceTokenizer` (Phase 3, Task 3.1/3.3/3.4/3.5)."""

from tokenizers.wordpiece.tokenizer import WordPieceTokenizer
from vocabulary.special_tokens import ORDERED_SPECIAL_TOKENS, UNK

_LOW_FAMILY_CORPUS = ["low"] * 5 + ["lower"] * 2 + ["lowest"]


def _train_low_family(vocab_size: int = 20) -> WordPieceTokenizer:
    tokenizer = WordPieceTokenizer(vocab_size=vocab_size)
    tokenizer.train(_LOW_FAMILY_CORPUS)
    return tokenizer


def _vocab_with_tokens(*tokens: str) -> WordPieceTokenizer:
    """A tokenizer whose vocabulary is populated directly, to test encode/decode in isolation."""
    tokenizer = WordPieceTokenizer()
    for token in tokens:
        tokenizer.special_tokens.vocabulary.add_token(token)
    return tokenizer


# ---------------------------------------------------------------------------
# construction / naming / contract basics
# ---------------------------------------------------------------------------


def test_can_be_instantiated_with_default_and_explicit_vocab_size():
    WordPieceTokenizer()
    WordPieceTokenizer(vocab_size=30)


def test_name_is_wordpiece():
    assert WordPieceTokenizer().name == "wordpiece"


def test_vocab_size_starts_with_only_special_tokens():
    assert WordPieceTokenizer().vocab_size == len(ORDERED_SPECIAL_TOKENS)


def test_train_grows_vocabulary_up_to_the_target_size():
    # 7 base symbols (l, ##o, ##w, ##e, ##r, ##s, ##t) + 2 reachable merges.
    tokenizer = _train_low_family(vocab_size=9)

    assert tokenizer.vocab_size == len(ORDERED_SPECIAL_TOKENS) + 9


def test_train_is_deterministic():
    first = _train_low_family(vocab_size=15)
    second = _train_low_family(vocab_size=15)

    assert first.encode("lower") == second.encode("lower")
    assert first.vocab_size == second.vocab_size


# ---------------------------------------------------------------------------
# Task 3.4 — greedy longest-match encode
# ---------------------------------------------------------------------------


def test_tokenize_picks_the_longest_available_match_not_the_first_one():
    # "u", "un", "unb", "unbe" are all valid vocabulary entries; the longest
    # one ("unbe") must be selected for the word's first piece.
    tokenizer = _vocab_with_tokens("u", "un", "unb", "unbe", "##lievable")

    assert tokenizer.tokenize("unbelievable") == ["unbe", "##lievable"]


def test_tokenize_processes_left_to_right_greedily():
    # Only "ab" (whole) and "##c" are in the vocabulary -- "a"/"##b" alone
    # are not -- proving the match starts at position 0 and consumes the
    # longest available span before moving on.
    tokenizer = _vocab_with_tokens("ab", "##c")

    assert tokenizer.tokenize("abc") == ["ab", "##c"]


def test_tokenize_unknown_word_becomes_a_single_unk_token():
    tokenizer = _vocab_with_tokens("a", "##b")

    assert tokenizer.tokenize("xyz") == [UNK]


def test_tokenize_partial_match_failure_falls_back_to_unk_for_the_whole_word():
    # "a" and "##b" match, but nothing matches the remaining "c" -- the
    # word must become a single <UNK>, not ["a", "##b", "<UNK>"].
    tokenizer = _vocab_with_tokens("a", "##b")

    assert tokenizer.tokenize("abc") == [UNK]


def test_tokenize_empty_and_whitespace_only_strings():
    tokenizer = _train_low_family()

    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []


def test_tokenize_known_word_from_training_corpus_has_no_unk():
    tokenizer = _train_low_family(vocab_size=20)

    assert UNK not in tokenizer.tokenize("low")


def test_untrained_tokenizer_does_not_raise_on_encode():
    tokenizer = WordPieceTokenizer()

    ids = tokenizer.encode("hello")

    assert all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == UNK


def test_encode_returns_matching_ids_for_tokenize_output():
    tokenizer = _vocab_with_tokens("un", "##believ", "##able")

    tokens = tokenizer.tokenize("unbelievable")
    ids = tokenizer.encode("unbelievable")

    assert tokens == ["un", "##believ", "##able"]
    assert [tokenizer.special_tokens.vocabulary.get_token(i) for i in ids] == tokens


def test_encode_unknown_word_falls_back_to_unk_without_raising():
    tokenizer = _train_low_family(vocab_size=20)

    ids = tokenizer.encode("é")  # never seen during training

    assert tokenizer.decode(ids) == UNK


def test_encode_is_deterministic():
    tokenizer = _train_low_family(vocab_size=20)

    assert tokenizer.encode("lower") == tokenizer.encode("lower")


# ---------------------------------------------------------------------------
# Task 3.5 — decode
# ---------------------------------------------------------------------------


def test_decode_reassembles_continuation_pieces_without_spaces():
    tokenizer = _vocab_with_tokens("un", "##believ", "##able")

    assert tokenizer.decode(tokenizer.encode("unbelievable")) == "unbelievable"


def test_decode_adds_spaces_only_between_words():
    tokenizer = _vocab_with_tokens("hello", "world")

    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"


def test_decode_of_empty_id_list_is_empty_string():
    assert _train_low_family().decode([]) == ""


def test_encode_decode_roundtrip_for_every_word_in_the_training_corpus():
    tokenizer = WordPieceTokenizer(vocab_size=30)
    tokenizer.train(_LOW_FAMILY_CORPUS)

    for word in ("low", "lower", "lowest"):
        assert tokenizer.decode(tokenizer.encode(word)) == word


def test_encode_decode_roundtrip_empty_string():
    tokenizer = _train_low_family()

    assert tokenizer.decode(tokenizer.encode("")) == ""


def test_decode_handles_unicode_roundtrip():
    tokenizer = WordPieceTokenizer(vocab_size=30)
    tokenizer.train(["héllo wörld"] * 3)

    assert tokenizer.decode(tokenizer.encode("héllo wörld")) == "héllo wörld"


def test_very_long_rare_word_does_not_raise_and_falls_back_gracefully():
    tokenizer = _train_low_family(vocab_size=20)

    long_word = "x" * 200
    ids = tokenizer.encode(long_word)

    assert tokenizer.decode(ids) == UNK


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip_preserves_vocabulary_and_behavior(tmp_path):
    tokenizer = _train_low_family(vocab_size=20)
    path = tmp_path / "wordpiece_vocab.json"
    tokenizer.save(path)

    loaded = WordPieceTokenizer()
    loaded.load(path)

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("lower") == tokenizer.encode("lower")
    assert loaded.decode(loaded.encode("lower")) == tokenizer.decode(tokenizer.encode("lower"))
