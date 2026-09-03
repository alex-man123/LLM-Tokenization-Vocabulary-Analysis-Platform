"""Unit tests for the `tiktoken` adapter (Phase 7, Task 7.2)."""

import pytest
import tiktoken

from benchmarking.comparator import compare_tokenizers
from tokenizers.adapters.tiktoken_tokenizer import TiktokenAdapter
from tokenizers.base import Tokenizer

_ENCODING_NAME = "cl100k_base"


def _adapter() -> TiktokenAdapter:
    return TiktokenAdapter.from_encoding_name(_ENCODING_NAME)


# ---------------------------------------------------------------------------
# construction / contract basics
# ---------------------------------------------------------------------------


def test_is_a_tokenizer_subclass():
    assert issubclass(TiktokenAdapter, Tokenizer)


def test_name_includes_the_encoding_name():
    assert _adapter().name == f"tiktoken:{_ENCODING_NAME}"


def test_explicit_name_overrides_the_default():
    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    adapter = TiktokenAdapter(encoding, name="custom-name")

    assert adapter.name == "custom-name"


def test_for_model_loads_whichever_encoding_that_model_uses():
    adapter = TiktokenAdapter.for_model("gpt-4")

    assert adapter.name == f"tiktoken:{tiktoken.encoding_for_model('gpt-4').name}"
    assert adapter.encode("hello world") == tiktoken.encoding_for_model("gpt-4").encode(
        "hello world"
    )


def test_train_is_a_no_op_and_does_not_raise():
    adapter = _adapter()
    ids_before = adapter.encode("hello world")

    adapter.train(["some", "unrelated", "corpus"])

    assert adapter.encode("hello world") == ids_before


def test_using_an_adapter_with_no_encoding_loaded_raises_a_clear_error():
    adapter = TiktokenAdapter()

    with pytest.raises(RuntimeError):
        adapter.encode("hello")
    with pytest.raises(RuntimeError):
        adapter.vocab_size  # noqa: B018 (property access is the point of the test)


# ---------------------------------------------------------------------------
# encode / decode / vocab_size
# ---------------------------------------------------------------------------


def test_encode_returns_real_tiktoken_ids():
    adapter = _adapter()
    encoding = tiktoken.get_encoding(_ENCODING_NAME)

    assert adapter.encode("hello world") == encoding.encode("hello world")


def test_vocab_size_is_the_real_tiktoken_vocab_size():
    adapter = _adapter()
    encoding = tiktoken.get_encoding(_ENCODING_NAME)

    assert adapter.vocab_size == encoding.n_vocab
    # Must not be derived from how much text has been tokenized.
    adapter.encode("a")
    assert adapter.vocab_size == encoding.n_vocab


def test_decode_uses_tiktokens_own_decode_not_manual_join():
    adapter = _adapter()
    ids = adapter.encode("hello world")

    assert adapter.decode(ids) == tiktoken.get_encoding(_ENCODING_NAME).decode(ids)


def test_encode_decode_roundtrip_ascii():
    adapter = _adapter()

    assert adapter.decode(adapter.encode("hello world")) == "hello world"


def test_encode_decode_roundtrip_unicode():
    adapter = _adapter()
    text = "héllo wörld こんにちは"

    assert adapter.decode(adapter.encode(text)) == text


def test_encode_empty_text():
    adapter = _adapter()

    assert adapter.encode("") == []
    assert adapter.tokenize("") == []


def test_encode_multiple_words():
    adapter = _adapter()

    ids = adapter.encode("the quick brown fox jumps over the lazy dog")

    assert len(ids) > 1


def test_tokenize_produces_one_string_per_token_id():
    adapter = _adapter()

    ids = adapter.encode("hello world")
    tokens = adapter.tokenize("hello world")

    assert len(tokens) == len(ids)
    assert all(isinstance(t, str) for t in tokens)


# ---------------------------------------------------------------------------
# Task 7.2's required test: byte-level vs. character-level
# ---------------------------------------------------------------------------


def test_len_text_is_not_len_utf8_bytes_for_unicode():
    text = "こんにちは"

    assert len(text) != len(text.encode("utf-8"))


def test_a_single_token_can_split_a_multi_byte_character_across_tokens():
    # "👍🏽" (thumbs up + skin tone modifier) is built from multi-byte UTF-8
    # sequences; cl100k_base's byte-level BPE does not guarantee a merge
    # exists that keeps every character's bytes inside one token.
    adapter = _adapter()
    encoding = tiktoken.get_encoding(_ENCODING_NAME)

    ids = adapter.encode("👍🏽")
    per_token_bytes = [encoding.decode_single_token_bytes(i) for i in ids]

    at_least_one_token_is_not_valid_utf8_alone = any(
        _not_valid_utf8(token_bytes) for token_bytes in per_token_bytes
    )
    assert at_least_one_token_is_not_valid_utf8_alone
    # Despite that, the full decode is lossless:
    assert adapter.decode(ids) == "👍🏽"


def test_tokenize_may_show_replacement_characters_for_split_multi_byte_tokens():
    # tokenize()'s per-token string is a best-effort visualization (see the
    # module docstring) and can legitimately contain U+FFFD for a token
    # whose bytes are not independently valid UTF-8 -- this is expected
    # byte-level behavior, not a bug, and decode() is unaffected by it.
    adapter = _adapter()

    tokens = adapter.tokenize("👍🏽")

    assert any("�" in token for token in tokens) or len(tokens) == 1


def _not_valid_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    adapter = _adapter()
    path = tmp_path / "tiktoken_state.json"
    adapter.save(path)

    loaded = TiktokenAdapter()
    loaded.load(path)

    assert loaded.vocab_size == adapter.vocab_size
    assert loaded.name == adapter.name
    assert loaded.encode("hello world") == adapter.encode("hello world")


# ---------------------------------------------------------------------------
# Comparator integration (Task 5.3) — no Comparator changes required
# ---------------------------------------------------------------------------


def test_works_with_the_comparator_without_any_comparator_changes():
    from tokenizers.character_tokenizer import CharacterTokenizer

    tiktoken_adapter = _adapter()
    character_tokenizer = CharacterTokenizer()
    character_tokenizer.train(["hello world"])

    results = compare_tokenizers([character_tokenizer, tiktoken_adapter], "hello world")

    assert set(results["tokenizer"]) == {"character", f"tiktoken:{_ENCODING_NAME}"}
    assert (results["vocab_size"] > 0).all()
