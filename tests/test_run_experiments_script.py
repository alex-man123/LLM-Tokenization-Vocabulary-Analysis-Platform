"""Unit tests for the pure helper functions in `scripts/run_experiments.py` (Task 6.3/6.4).

Only the pure formatting/highlight helpers are tested here — `main()` writes
to the real `data/results/`/`docs/experiment_results.md` paths by design
(it is the single script a human runs deliberately to regenerate them, per
Task 6.3's Definition of Done) and must not run as a side effect of `pytest`.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_experiments_script", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_module = _load_script_module()


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tokenizer": "character",
                "dataset": "en",
                "language_or_type": "english",
                "vocab_size": 30,
                "num_tokens": 100,
                "tokens_per_word": 5.0,
                "characters_per_token": 1.0,
                "compression_ratio": 1.0,
                "encode_time_ms": None,
                "decode_time_ms": None,
            },
            {
                "tokenizer": "character",
                "dataset": "ja",
                "language_or_type": "japanese",
                "vocab_size": 30,
                "num_tokens": 80,
                "tokens_per_word": 20.0,
                "characters_per_token": 1.0,
                "compression_ratio": 1.0,
                "encode_time_ms": None,
                "decode_time_ms": None,
            },
        ]
    )


def test_dataframe_to_markdown_table_has_header_separator_and_rows():
    df = pd.DataFrame([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])

    table = _module._dataframe_to_markdown_table(df)
    lines = table.splitlines()

    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert len(lines) == 4  # header + separator + 2 rows


def test_dataframe_to_markdown_table_formats_floats_compactly():
    df = pd.DataFrame([{"value": 3.141592653589793}])

    table = _module._dataframe_to_markdown_table(df)

    assert "3.14" in table
    assert "3.141592653589793" not in table


def test_highlight_extremes_names_the_max_and_min_category_per_tokenizer():
    by_group_and_tokenizer = _module.aggregate_by_group_and_tokenizer(
        _sample_results(), metrics=["tokens_per_word"]
    )

    highlights = _module._highlight_extremes(by_group_and_tokenizer)

    assert len(highlights) == 1  # only one tokenizer ("character") in the sample
    assert "'japanese'" in highlights[0]  # highest tokens_per_word (20.0)
    assert "'english'" in highlights[0]  # lowest tokens_per_word (5.0)


def test_build_report_includes_every_expected_section():
    report = _module._build_report(_sample_results())

    assert "# Experiment results" in report
    assert "## Raw results" in report
    assert "## Aggregated by language/type and tokenizer" in report
    assert "## Highlights" in report
    assert "## All observations" in report
    assert "in this experiment's dataset" in report.lower()


def test_build_report_does_not_crash_on_a_single_row():
    single_row = _sample_results().iloc[[0]]

    report = _module._build_report(single_row)

    assert "1 rows: 1 tokenizers x 1 datasets." in report
