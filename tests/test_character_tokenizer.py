"""Unit tests for `CharacterTokenizer` (Phase 1, Task 1.1)."""

from tokenizers.character_tokenizer import CharacterTokenizer
from vocabulary.special_tokens import ORDERED_SPECIAL_TOKENS, UNK


def test_can_be_instantiated():
    CharacterTokenizer()


def test_name_is_character():
    assert CharacterTokenizer().name == "character"


def test_vocab_size_starts_with_only_special_tokens():
    tokenizer = CharacterTokenizer()
    assert tokenizer.vocab_size == len(ORDERED_SPECIAL_TOKENS)


def test_train_builds_vocabulary_from_corpus_characters():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["ab"])

    assert tokenizer.vocab_size == len(ORDERED_SPECIAL_TOKENS) + 2  # "a" and "b"


def test_train_does_not_create_duplicate_ids_for_repeated_characters():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["aaa", "aaa"])

    assert tokenizer.vocab_size == len(ORDERED_SPECIAL_TOKENS) + 1  # just "a"


def test_tokenize_returns_list_of_characters():
    tokenizer = CharacterTokenizer()

    assert tokenizer.tokenize("hello") == ["h", "e", "l", "l", "o"]
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize(" ") == [" "]


def test_tokenize_treats_whitespace_and_newlines_as_ordinary_characters():
    tokenizer = CharacterTokenizer()

    assert tokenizer.tokenize("a b") == ["a", " ", "b"]
    assert tokenizer.tokenize("a\nb") == ["a", "\n", "b"]


def test_tokenize_supports_unicode_text():
    tokenizer = CharacterTokenizer()

    assert tokenizer.tokenize("ăâîșț") == ["ă", "â", "î", "ș", "ț"]


def test_encode_returns_ids_for_trained_characters():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["hello"])

    ids = tokenizer.encode("hello")

    assert len(ids) == 5
    assert all(isinstance(i, int) for i in ids)


def test_encode_unknown_character_falls_back_to_unk_without_raising():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["abc"])

    ids = tokenizer.encode("z")

    assert tokenizer.decode(ids) == UNK


def test_decode_reconstructs_trained_text_exactly():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["hello world"])

    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"


def test_encode_decode_roundtrip_for_various_trained_inputs():
    tokenizer = CharacterTokenizer()
    corpus = ["hello", "hello world", "ăâîșț", "Hello!", "123", ""]
    tokenizer.train(corpus)

    for text in corpus:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_is_deterministic():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["hello world"])

    assert tokenizer.encode("hello") == tokenizer.encode("hello")


def test_save_and_load_roundtrip(tmp_path):
    tokenizer = CharacterTokenizer()
    tokenizer.train(["hello world"])
    path = tmp_path / "character_vocab.json"
    tokenizer.save(path)

    loaded = CharacterTokenizer()
    loaded.load(path)

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("hello") == tokenizer.encode("hello")
    assert loaded.decode(loaded.encode("hello")) == "hello"
