"""Unit tests for `WordTokenizer` (Phase 1, Task 1.2)."""

from tokenizers.word_tokenizer import WordTokenizer
from vocabulary.special_tokens import ORDERED_SPECIAL_TOKENS, UNK


def test_can_be_instantiated():
    WordTokenizer()


def test_name_is_word():
    assert WordTokenizer().name == "word"


def test_vocab_size_starts_with_only_special_tokens():
    tokenizer = WordTokenizer()
    assert tokenizer.vocab_size == len(ORDERED_SPECIAL_TOKENS)


def test_train_builds_vocabulary_from_corpus_words():
    tokenizer = WordTokenizer()
    tokenizer.train(["hello world"])

    assert tokenizer.vocab_size == len(ORDERED_SPECIAL_TOKENS) + 2  # "hello" and "world"


def test_train_does_not_create_duplicate_ids_for_repeated_words():
    tokenizer = WordTokenizer()
    tokenizer.train(["hello hello", "hello"])

    assert tokenizer.vocab_size == len(ORDERED_SPECIAL_TOKENS) + 1  # just "hello"


def test_tokenize_splits_on_whitespace():
    tokenizer = WordTokenizer()

    assert tokenizer.tokenize("hello world") == ["hello", "world"]
    assert tokenizer.tokenize("  hello   world  ") == ["hello", "world"]
    assert tokenizer.tokenize("hello\nworld") == ["hello", "world"]


def test_tokenize_separates_punctuation_from_words():
    tokenizer = WordTokenizer()

    assert tokenizer.tokenize("hello!") == ["hello", "!"]
    assert tokenizer.tokenize("don't") == ["don", "'", "t"]


def test_tokenize_empty_and_whitespace_only_strings():
    tokenizer = WordTokenizer()

    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []


def test_encode_returns_ids_for_trained_words():
    tokenizer = WordTokenizer()
    tokenizer.train(["hello world"])

    ids = tokenizer.encode("hello world")

    assert len(ids) == 2
    assert all(isinstance(i, int) for i in ids)


def test_encode_unknown_word_falls_back_to_unk_without_raising():
    tokenizer = WordTokenizer()
    tokenizer.train(["hello world"])

    ids = tokenizer.encode("banana")

    assert tokenizer.decode(ids) == UNK


def test_decode_joins_tokens_with_single_space():
    tokenizer = WordTokenizer()
    tokenizer.train(["hello world"])

    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"


def test_encode_decode_roundtrip_for_trained_inputs():
    tokenizer = WordTokenizer()
    corpus = ["hello world", "hello", ""]
    tokenizer.train(corpus)

    for text in corpus:
        assert tokenizer.decode(tokenizer.encode(text)) == " ".join(tokenizer.tokenize(text))


def test_encode_is_deterministic():
    tokenizer = WordTokenizer()
    tokenizer.train(["hello world"])

    assert tokenizer.encode("hello world") == tokenizer.encode("hello world")


def test_save_and_load_roundtrip(tmp_path):
    tokenizer = WordTokenizer()
    tokenizer.train(["hello world"])
    path = tmp_path / "word_vocab.json"
    tokenizer.save(path)

    loaded = WordTokenizer()
    loaded.load(path)

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("hello world") == tokenizer.encode("hello world")
