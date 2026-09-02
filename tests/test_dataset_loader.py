"""Unit/integration tests for the dataset loader & preprocessing (Phase 6, Task 6.1/6.2)."""

import unicodedata

import pytest

from experiments.dataset_loader import (
    DATASET_CATEGORIES,
    DatasetMetadata,
    load_all_datasets,
    load_dataset,
    normalize_text,
)

# ---------------------------------------------------------------------------
# Task 6.1 — the 9 raw datasets exist and have real content
# ---------------------------------------------------------------------------


def test_exactly_nine_categories_are_defined():
    assert len(DATASET_CATEGORIES) == 9
    assert set(DATASET_CATEGORIES) == {
        "en",
        "ro",
        "es",
        "ja",
        "code_python",
        "numbers",
        "urls",
        "emoji",
        "technical",
    }


@pytest.mark.parametrize("category", list(DATASET_CATEGORIES))
def test_every_category_loads_real_utf8_content_of_a_reasonable_size(category):
    text, metadata = load_dataset(category)

    assert isinstance(text, str)
    assert text.strip()  # not empty/whitespace-only
    assert "TODO" not in text
    assert "placeholder" not in text.lower()
    assert "lorem ipsum" not in text.lower()
    # ~2-5 KB is the target; a little slack on both ends is fine.
    assert 1500 <= metadata.length_bytes <= 6000


def test_code_python_dataset_is_syntactically_valid_python():
    text, _ = load_dataset("code_python")

    compile(text, "code_python.txt", "exec")  # raises SyntaxError if invalid


def test_code_python_dataset_preserves_mixed_case_identifiers():
    text, _ = load_dataset("code_python")

    assert "warehouseName" in text
    assert "WarehouseManager" in text
    assert "MAX_RETRIES" in text


def test_urls_dataset_preserves_case_in_paths():
    text, _ = load_dataset("urls")

    assert "CaseSensitivePath" in text


def test_romanian_dataset_contains_the_five_romanian_diacritics():
    text, _ = load_dataset("ro")

    for letter in "ăâîșț":
        assert letter in text


def test_spanish_dataset_contains_accented_characters_and_enye():
    text, _ = load_dataset("es")

    for letter in "áéíóúñ":
        assert letter in text


def test_japanese_dataset_contains_japanese_characters():
    text, _ = load_dataset("ja")

    assert any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in text)


# ---------------------------------------------------------------------------
# Task 6.2 — loader metadata
# ---------------------------------------------------------------------------


def test_load_dataset_returns_text_and_metadata_tuple():
    text, metadata = load_dataset("en")

    assert isinstance(text, str)
    assert isinstance(metadata, DatasetMetadata)
    assert metadata.name == "en"
    assert metadata.language_or_type == "english"
    assert metadata.length_chars == len(text)
    assert metadata.length_bytes == len(text.encode("utf-8"))


def test_load_all_datasets_returns_all_nine_categories():
    results = load_all_datasets()

    assert set(results.keys()) == set(DATASET_CATEGORIES)
    for category, (text, metadata) in results.items():
        assert metadata.name == category
        assert text  # non-empty


def test_load_dataset_raises_for_unknown_category():
    with pytest.raises(ValueError):
        load_dataset("klingon")


def test_load_dataset_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset("en", raw_dir=tmp_path)


def test_load_dataset_handles_an_empty_file_without_crashing(tmp_path):
    (tmp_path / "en.txt").write_text("", encoding="utf-8")

    text, metadata = load_dataset("en", raw_dir=tmp_path)

    assert text == ""
    assert metadata.length_chars == 0
    assert metadata.length_bytes == 0


# ---------------------------------------------------------------------------
# Task 6.2 — NFC normalization
# ---------------------------------------------------------------------------


def test_normalize_text_unifies_decomposed_and_precomposed_forms():
    decomposed = "é"  # "e" + COMBINING ACUTE ACCENT
    precomposed = "é"  # "é"

    assert normalize_text(decomposed) == precomposed
    assert unicodedata.is_normalized("NFC", normalize_text(decomposed))


def test_normalize_text_unifies_japanese_decomposed_dakuten():
    decomposed = "が"  # "か" + COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK
    precomposed = "が"  # "が"

    assert normalize_text(decomposed) == precomposed


def test_normalize_text_fixes_romanian_cedilla_variants_to_comma_below():
    cedilla_variants = "şŞţŢ"  # ş Ş ţ Ţ
    comma_below = "șȘțȚ"  # ș Ș ț Ț

    assert normalize_text(cedilla_variants) == comma_below


def test_normalize_text_does_not_alter_already_correct_romanian_text():
    text = "Aș vrea să înțeleg mai bine țara aceasta."

    assert normalize_text(text) == text


def test_normalize_text_handles_empty_string():
    assert normalize_text("") == ""


def test_normalize_text_never_lowercases():
    assert normalize_text("MyVariable_CONST") == "MyVariable_CONST"
    assert normalize_text("Example.com/CaseSensitivePath") == "Example.com/CaseSensitivePath"


def test_loaded_code_dataset_was_actually_normalized_not_bypassed():
    # Sanity check that load_dataset really routes through normalize_text,
    # by confirming the loaded text is already in NFC form.
    text, _ = load_dataset("code_python")

    assert unicodedata.is_normalized("NFC", text)
