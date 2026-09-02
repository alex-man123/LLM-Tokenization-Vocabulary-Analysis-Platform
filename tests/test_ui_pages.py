"""Streamlit UI regression tests (Phase 8, Task 8.1/8.2/8.3).

Uses `streamlit.testing.v1.AppTest` to actually execute each page script
in-process (simulating widget input) rather than just importing it — this
is what proves the pages run without runtime errors, not merely that they
parse.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

UI_DIR = Path(__file__).resolve().parents[1] / "ui"


def test_home_page_runs_without_error():
    at = AppTest.from_file(str(UI_DIR / "streamlit_app.py"))
    at.run(timeout=15)

    assert not at.exception


def test_tokenize_page_runs_with_default_input():
    at = AppTest.from_file(str(UI_DIR / "pages" / "1_Tokenize.py"))
    at.run(timeout=15)

    assert not at.exception
    assert sorted(at.selectbox[0].options) == ["bpe", "character", "word", "wordpiece"]
    assert len(at.dataframe) == 1
    assert set(at.dataframe[0].value.columns) == {"token", "token_id"}


def test_tokenize_page_handles_bpe_and_non_ascii_text():
    at = AppTest.from_file(str(UI_DIR / "pages" / "1_Tokenize.py"))
    at.run(timeout=15)
    at.selectbox[0].set_value("bpe").run(timeout=15)
    at.text_area[0].set_value("hello world こんにちは").run(timeout=15)

    assert not at.exception
    assert len(at.dataframe[0].value) > 0


def test_tokenize_page_handles_empty_text_without_error():
    at = AppTest.from_file(str(UI_DIR / "pages" / "1_Tokenize.py"))
    at.run(timeout=15)
    at.text_area[0].set_value("").run(timeout=15)

    assert not at.exception
    assert len(at.info) > 0


def test_compare_page_runs_with_default_selection():
    at = AppTest.from_file(str(UI_DIR / "pages" / "2_Compare.py"))
    at.run(timeout=15)

    assert not at.exception
    assert sorted(at.multiselect[0].value) == ["bpe", "character", "word", "wordpiece"]
    assert len(at.dataframe) == 1

    df = at.dataframe[0].value
    required_columns = {
        "tokenizer",
        "number_of_tokens",
        "tokens_per_word",
        "characters_per_token",
        "compression_ratio",
        "vocab_size",
        "encoding_time",
        "decoding_time",
    }
    assert required_columns <= set(df.columns)
    assert len(df) == 4
    assert (df["vocab_size"] > 0).all()


def test_compare_page_shows_fair_comparison_disclaimer():
    at = AppTest.from_file(str(UI_DIR / "pages" / "2_Compare.py"))
    at.run(timeout=15)

    assert len(at.warning) > 0
    assert "vocab_size" in at.warning[0].value


def test_compare_page_handles_empty_text_without_error():
    at = AppTest.from_file(str(UI_DIR / "pages" / "2_Compare.py"))
    at.run(timeout=15)
    at.text_area[0].set_value("").run(timeout=15)

    assert not at.exception
    assert len(at.info) > 0


def test_compare_page_handles_no_tokenizer_selected():
    at = AppTest.from_file(str(UI_DIR / "pages" / "2_Compare.py"))
    at.run(timeout=15)
    at.multiselect[0].set_value([]).run(timeout=15)

    assert not at.exception
    assert len(at.info) > 0


def test_placeholder_pages_run_without_error():
    for filename in ("3_Vocabulary.py", "4_Benchmark.py", "5_Experiments.py"):
        at = AppTest.from_file(str(UI_DIR / "pages" / filename))
        at.run(timeout=15)

        assert not at.exception, f"{filename} raised an exception"
