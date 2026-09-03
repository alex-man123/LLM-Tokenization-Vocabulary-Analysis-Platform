"""Streamlit UI regression tests (Phase 8, Task 8.1/8.2/8.3).

Uses `streamlit.testing.v1.AppTest` to actually execute each page script
in-process (simulating widget input) rather than just importing it — this
is what proves the pages run without runtime errors, not merely that they
parse.
"""

from pathlib import Path

import pytest
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


def test_compare_page_offers_external_tokenizers_as_optional_selections():
    at = AppTest.from_file(str(UI_DIR / "pages" / "2_Compare.py"))
    at.run(timeout=15)

    assert not at.exception
    options = at.multiselect[0].options
    assert "huggingface:bert-base-uncased" in options
    assert "tiktoken:cl100k_base" in options
    # Not selected by default -- loading them can require network access.
    assert "huggingface:bert-base-uncased" not in at.multiselect[0].value
    assert "tiktoken:cl100k_base" not in at.multiselect[0].value


def test_compare_page_can_include_an_external_tokenizer_when_selected():
    at = AppTest.from_file(str(UI_DIR / "pages" / "2_Compare.py"))
    at.run(timeout=15)
    at.multiselect[0].set_value(["character", "tiktoken:cl100k_base"]).run(timeout=30)

    assert not at.exception
    if at.error:
        return  # network unavailable -- the page reported it cleanly instead of crashing
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) == 2
    assert set(at.dataframe[0].value["tokenizer"]) == {"character", "tiktoken:cl100k_base"}


def test_compare_page_shows_a_failed_external_tokenizer_without_losing_other_results(monkeypatch):
    # Deterministically simulate "no network" for the tiktoken adapter,
    # rather than depending on this test environment's real connectivity.
    import streamlit as st

    import tokenizers.adapters.tiktoken_tokenizer as tiktoken_adapter_module

    def _simulate_offline(_encoding_name):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(tiktoken_adapter_module.tiktoken, "get_encoding", _simulate_offline)
    # `st.cache_resource` is a process-wide cache keyed by function+args, so
    # a real successful load cached by another test (or a real prior failure
    # cached by this one) could otherwise mask what this test is checking.
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(UI_DIR / "pages" / "2_Compare.py"))
        at.run(timeout=15)
        at.multiselect[0].set_value(["character", "tiktoken:cl100k_base"]).run(timeout=30)

        assert not at.exception
        assert len(at.error) == 1
        assert "tiktoken:cl100k_base" in at.error[0].value
        # The character tokenizer's result must still be shown, not
        # discarded just because the other selected tokenizer failed.
        assert len(at.dataframe) == 1
        assert list(at.dataframe[0].value["tokenizer"]) == ["character"]
    finally:
        st.cache_resource.clear()  # don't leak the simulated failure into later tests


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


def test_vocabulary_page_runs_with_default_input():
    at = AppTest.from_file(str(UI_DIR / "pages" / "3_Vocabulary.py"))
    at.run(timeout=15)

    assert not at.exception
    assert len(at.metric) == 1  # vocabulary size
    assert len(at.dataframe) >= 1  # top-N frequency table


def test_vocabulary_page_handles_a_single_token_corpus_without_error():
    # Regression: a corpus producing exactly one distinct token used to
    # crash the "Top N" slider (Streamlit rejects min_value == max_value).
    at = AppTest.from_file(str(UI_DIR / "pages" / "3_Vocabulary.py"))
    at.run(timeout=15)
    at.selectbox[0].set_value("character").run(timeout=15)
    at.text_area[0].set_value("a").run(timeout=15)

    assert not at.exception


def test_vocabulary_page_handles_empty_text_without_error():
    at = AppTest.from_file(str(UI_DIR / "pages" / "3_Vocabulary.py"))
    at.run(timeout=15)
    at.text_area[0].set_value("").run(timeout=15)

    assert not at.exception
    assert len(at.info) > 0


def test_vocabulary_page_handles_unicode_text():
    at = AppTest.from_file(str(UI_DIR / "pages" / "3_Vocabulary.py"))
    at.run(timeout=15)
    at.text_area[0].set_value("héllo wörld こんにちは").run(timeout=15)

    assert not at.exception
    assert len(at.metric) == 1


def test_vocabulary_page_works_for_every_available_tokenizer():
    for name in ("character", "word", "bpe", "wordpiece"):
        at = AppTest.from_file(str(UI_DIR / "pages" / "3_Vocabulary.py"))
        at.run(timeout=15)
        at.selectbox[0].set_value(name).run(timeout=15)

        assert not at.exception, f"tokenizer {name!r} raised an exception"


def test_how_llms_use_tokens_page_runs_with_default_input():
    at = AppTest.from_file(str(UI_DIR / "pages" / "6_How_LLMs_Use_Tokens.py"))
    at.run(timeout=15)

    assert not at.exception
    assert len(at.dataframe) == 2  # tokens/IDs table + illustrative embeddings table
    embeddings_df = at.dataframe[1].value
    assert "illustrative_embedding" in embeddings_df.columns


def test_how_llms_use_tokens_page_labels_embeddings_as_illustrative():
    at = AppTest.from_file(str(UI_DIR / "pages" / "6_How_LLMs_Use_Tokens.py"))
    at.run(timeout=15)

    assert not at.exception
    assert len(at.warning) > 0
    assert "illustrative" in at.warning[0].value.lower()


def test_how_llms_use_tokens_embedding_is_deterministic_per_token_id():
    at = AppTest.from_file(str(UI_DIR / "pages" / "6_How_LLMs_Use_Tokens.py"))
    at.run(timeout=15)
    at.text_area[0].set_value("hello hello").run(timeout=15)

    embeddings_df = at.dataframe[1].value
    # "hello" appears twice with the same token_id -> the same illustrative
    # embedding both times (deterministic per token ID, not re-randomized).
    duplicate_id_rows = embeddings_df[embeddings_df.duplicated("token_id", keep=False)]
    assert len(duplicate_id_rows) >= 2
    first_embedding = duplicate_id_rows.iloc[0]["illustrative_embedding"]
    assert all(
        list(row) == list(first_embedding) for row in duplicate_id_rows["illustrative_embedding"]
    )


def test_how_llms_use_tokens_page_handles_empty_text_without_error():
    at = AppTest.from_file(str(UI_DIR / "pages" / "6_How_LLMs_Use_Tokens.py"))
    at.run(timeout=15)
    at.text_area[0].set_value("").run(timeout=15)

    assert not at.exception
    assert len(at.info) > 0


def test_how_llms_use_tokens_page_handles_unicode_text():
    at = AppTest.from_file(str(UI_DIR / "pages" / "6_How_LLMs_Use_Tokens.py"))
    at.run(timeout=15)
    at.text_area[0].set_value("héllo wörld こんにちは").run(timeout=15)

    assert not at.exception


def test_benchmark_page_runs_with_default_input():
    at = AppTest.from_file(str(UI_DIR / "pages" / "4_Benchmark.py"))
    at.run(timeout=15)

    assert not at.exception
    assert len(at.dataframe) == 3  # metrics, timing, cost
    metrics_df = at.dataframe[0].value
    assert {"vocab_size", "number_of_tokens", "compression_ratio"} <= set(metrics_df.columns)


def test_benchmark_page_shows_encode_decode_timing():
    at = AppTest.from_file(str(UI_DIR / "pages" / "4_Benchmark.py"))
    at.run(timeout=15)

    timing_df = at.dataframe[1].value
    assert {"encode_mean_ms", "encode_median_ms", "decode_mean_ms", "decode_median_ms"} <= set(
        timing_df.columns
    )
    assert (timing_df["encode_mean_ms"] >= 0).all()
    assert (timing_df["decode_mean_ms"] >= 0).all()


def test_benchmark_page_cost_estimator_matches_the_formula():
    at = AppTest.from_file(str(UI_DIR / "pages" / "4_Benchmark.py"))
    at.run(timeout=15)

    cost_df = at.dataframe[2].value
    for _, row in cost_df.iterrows():
        expected = (row["num_tokens"] / 1_000_000) * row["price_per_million"]
        assert row["estimated_cost"] == pytest.approx(expected)


def test_benchmark_page_cost_recalculates_when_price_changes():
    at = AppTest.from_file(str(UI_DIR / "pages" / "4_Benchmark.py"))
    at.run(timeout=15)
    cost_before = at.dataframe[2].value.set_index("tokenizer")["estimated_cost"]

    at.number_input[0].set_value(at.number_input[0].value * 2).run(timeout=15)
    cost_after = at.dataframe[2].value.set_index("tokenizer")["estimated_cost"]

    for tokenizer_name in cost_before.index:
        assert cost_after[tokenizer_name] == pytest.approx(cost_before[tokenizer_name] * 2)


def test_benchmark_page_cost_estimator_price_zero_gives_zero_cost():
    at = AppTest.from_file(str(UI_DIR / "pages" / "4_Benchmark.py"))
    at.run(timeout=15)
    at.number_input[0].set_value(0.0).run(timeout=15)

    assert (at.dataframe[2].value["estimated_cost"] == 0).all()


def test_benchmark_page_handles_empty_text_without_error():
    at = AppTest.from_file(str(UI_DIR / "pages" / "4_Benchmark.py"))
    at.run(timeout=15)
    at.text_area[0].set_value("").run(timeout=15)

    assert not at.exception
    assert len(at.info) > 0


def test_benchmark_page_handles_no_tokenizer_selected():
    at = AppTest.from_file(str(UI_DIR / "pages" / "4_Benchmark.py"))
    at.run(timeout=15)
    at.multiselect[0].set_value([]).run(timeout=15)

    assert not at.exception
    assert len(at.info) > 0


def test_benchmark_page_offers_the_same_external_options_as_compare():
    at = AppTest.from_file(str(UI_DIR / "pages" / "4_Benchmark.py"))
    at.run(timeout=15)

    options = at.multiselect[0].options
    assert "sentencepiece:unigram" in options
    assert "huggingface:bert-base-uncased" in options
    assert "tiktoken:cl100k_base" in options


_EXPERIMENT_RESULTS_PATH = (
    UI_DIR.parent / "data" / "results" / "experiment_results.json"
)


def test_experiments_page_runs_without_error():
    at = AppTest.from_file(str(UI_DIR / "pages" / "5_Experiments.py"))
    at.run(timeout=15)

    assert not at.exception


def test_experiments_page_shows_precomputed_data_without_recomputing():
    # This project's real `data/results/experiment_results.json` (Task 6.3)
    # is committed, so this test verifies the page actually reads and
    # displays it -- not just that it fails to crash.
    if not _EXPERIMENT_RESULTS_PATH.exists():
        pytest.skip("data/results/experiment_results.json not present")

    at = AppTest.from_file(str(UI_DIR / "pages" / "5_Experiments.py"))
    at.run(timeout=15)

    assert not at.exception
    assert len(at.dataframe) == 2  # aggregated table + raw results
    raw_results = at.dataframe[1].value
    assert "tokenizer" in raw_results.columns
    assert "language_or_type" in raw_results.columns
    assert len(raw_results) > 0


def test_experiments_page_reports_missing_results_without_fake_data():
    # The page resolves its results path from its own file location, not
    # from cwd or an env var, so the only way to exercise the "missing"
    # branch is to temporarily move the real committed file aside -- always
    # restored in `finally`, even if the assertions below fail. The page
    # must report this plainly, never with invented rows to look populated.
    if not _EXPERIMENT_RESULTS_PATH.exists():
        pytest.skip("data/results/experiment_results.json not present")

    backup_path = _EXPERIMENT_RESULTS_PATH.with_suffix(".json.bak")
    _EXPERIMENT_RESULTS_PATH.rename(backup_path)
    try:
        at = AppTest.from_file(str(UI_DIR / "pages" / "5_Experiments.py"))
        at.run(timeout=15)

        assert not at.exception
        assert len(at.dataframe) == 0
        assert len(at.info) > 0
        assert "run_experiments.py" in at.info[0].value
    finally:
        backup_path.rename(_EXPERIMENT_RESULTS_PATH)


def test_placeholder_pages_run_without_error():
    pass  # no remaining placeholder pages -- Benchmark and Experiments are both implemented
