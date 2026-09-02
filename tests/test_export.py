"""Unit tests for benchmark result export (Phase 5, Task 5.4)."""

import json

import pandas as pd
import pytest

from benchmarking.comparator import compare_tokenizers
from benchmarking.export import (
    export_results_csv,
    export_results_json,
    export_schema_csv,
    export_schema_json,
    load_results_json,
    to_experiment_schema,
)
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.word_tokenizer import WordTokenizer


def _compare_result(text: str = "hello world") -> pd.DataFrame:
    tokenizers = [CharacterTokenizer(), WordTokenizer()]
    for tokenizer in tokenizers:
        tokenizer.train([text])
    return compare_tokenizers(tokenizers, text)


# ---------------------------------------------------------------------------
# to_experiment_schema
# ---------------------------------------------------------------------------


def test_to_experiment_schema_renames_columns_to_the_task_0_3_schema():
    schema_df = to_experiment_schema(_compare_result(), dataset="demo")

    assert "num_tokens" in schema_df.columns
    assert "encode_time_ms" in schema_df.columns
    assert "decode_time_ms" in schema_df.columns
    assert "number_of_tokens" not in schema_df.columns
    assert "encoding_time" not in schema_df.columns
    assert "decoding_time" not in schema_df.columns


def test_to_experiment_schema_adds_dataset_and_timestamp():
    schema_df = to_experiment_schema(_compare_result(), dataset="my_dataset")

    assert (schema_df["dataset"] == "my_dataset").all()
    assert schema_df["timestamp"].notna().all()


def test_to_experiment_schema_respects_explicit_timestamp():
    schema_df = to_experiment_schema(
        _compare_result(), dataset="demo", timestamp="2026-01-01T00:00:00+00:00"
    )

    assert (schema_df["timestamp"] == "2026-01-01T00:00:00+00:00").all()


def test_to_experiment_schema_preserves_tokens_column():
    schema_df = to_experiment_schema(_compare_result(), dataset="demo")

    assert "tokens" in schema_df.columns


def test_to_experiment_schema_preserves_vocab_size():
    schema_df = to_experiment_schema(_compare_result(), dataset="demo")

    assert (schema_df["vocab_size"] > 0).all()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def test_export_results_csv_creates_a_loadable_file(tmp_path):
    path = tmp_path / "results.csv"

    export_results_csv(_compare_result(), path, dataset="demo")
    loaded = pd.read_csv(path)

    assert len(loaded) == 2  # character + word tokenizer
    assert "num_tokens" in loaded.columns
    assert "dataset" in loaded.columns


def test_export_results_csv_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "results.csv"

    export_results_csv(_compare_result(), path, dataset="demo")

    assert path.exists()


def test_export_results_csv_preserves_tokens_as_json_string(tmp_path):
    path = tmp_path / "results.csv"
    export_results_csv(_compare_result("hello world"), path, dataset="demo")

    loaded = pd.read_csv(path)
    tokens = json.loads(loaded.iloc[0]["tokens"])

    assert isinstance(tokens, list)


def test_export_results_csv_handles_unicode_text(tmp_path):
    path = tmp_path / "results.csv"

    export_results_csv(_compare_result("こんにちは world"), path, dataset="unicode_demo")
    loaded = pd.read_csv(path)

    assert len(loaded) == 2


def test_export_results_csv_single_tokenizer(tmp_path):
    tokenizer = CharacterTokenizer()
    tokenizer.train(["hi"])
    results = compare_tokenizers([tokenizer], "hi")
    path = tmp_path / "results.csv"

    export_results_csv(results, path, dataset="demo")
    loaded = pd.read_csv(path)

    assert len(loaded) == 1


def test_export_results_csv_default_overwrites_existing_file(tmp_path):
    path = tmp_path / "results.csv"
    export_results_csv(_compare_result(), path, dataset="first")
    export_results_csv(_compare_result(), path, dataset="second")

    loaded = pd.read_csv(path)

    assert (loaded["dataset"] == "second").all()


def test_export_results_csv_overwrite_false_raises_if_file_exists(tmp_path):
    path = tmp_path / "results.csv"
    export_results_csv(_compare_result(), path, dataset="first")

    with pytest.raises(FileExistsError):
        export_results_csv(_compare_result(), path, dataset="second", overwrite=False)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def test_export_results_json_creates_a_loadable_file(tmp_path):
    path = tmp_path / "results.json"

    export_results_json(_compare_result(), path, dataset="demo")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(payload, list)
    assert len(payload) == 2


def test_export_results_json_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "results.json"

    export_results_json(_compare_result(), path, dataset="demo")

    assert path.exists()


def test_export_results_json_keeps_tokens_as_a_native_list(tmp_path):
    path = tmp_path / "results.json"
    export_results_json(_compare_result(), path, dataset="demo")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(payload[0]["tokens"], list)


def test_export_results_json_handles_unicode_text(tmp_path):
    path = tmp_path / "results.json"
    text = "héllo wörld こんにちは"

    export_results_json(_compare_result(text), path, dataset="unicode_demo")
    payload = json.loads(path.read_text(encoding="utf-8"))

    reconstructed = "".join(payload[0]["tokens"])  # CharacterTokenizer row
    assert "こんにちは" in reconstructed


def test_export_results_json_overwrite_false_raises_if_file_exists(tmp_path):
    path = tmp_path / "results.json"
    export_results_json(_compare_result(), path, dataset="first")

    with pytest.raises(FileExistsError):
        export_results_json(_compare_result(), path, dataset="second", overwrite=False)


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------


def test_json_roundtrip_preserves_every_column(tmp_path):
    path = tmp_path / "results.json"
    fixed_timestamp = "2026-01-01T00:00:00+00:00"
    original = to_experiment_schema(_compare_result(), dataset="demo", timestamp=fixed_timestamp)
    export_results_json(_compare_result(), path, dataset="demo", timestamp=fixed_timestamp)

    reloaded = load_results_json(path)

    assert set(reloaded.columns) == set(original.columns)
    assert len(reloaded) == len(original)


def test_json_roundtrip_preserves_vocab_size_and_num_tokens(tmp_path):
    path = tmp_path / "results.json"
    export_results_json(_compare_result(), path, dataset="demo")

    reloaded = load_results_json(path)

    assert (reloaded["vocab_size"] > 0).all()
    assert (reloaded["num_tokens"] >= 0).all()


def test_json_roundtrip_preserves_encoding_and_decoding_time_fields():
    # encoding_time/decoding_time are always None until Task 5.2's Timer is
    # wired into a caller's own results, but the schema/roundtrip must still
    # carry the fields through untouched, whatever their value.
    results = pd.DataFrame(
        [
            {
                "tokenizer": "character",
                "number_of_tokens": 5,
                "tokens_per_word": 1.0,
                "characters_per_token": 1.0,
                "compression_ratio": 1.0,
                "vocab_size": 10,
                "encoding_time": 1.23,
                "decoding_time": 0.87,
                "tokens": ["h", "e", "l", "l", "o"],
            }
        ]
    )

    schema_df = to_experiment_schema(results, dataset="demo")

    assert schema_df.iloc[0]["encode_time_ms"] == 1.23
    assert schema_df.iloc[0]["decode_time_ms"] == 0.87


# ---------------------------------------------------------------------------
# Low-level schema-writing primitives (used by the Experiment Runner, Task 6.3,
# to combine per-dataset stamped rows into one multi-dataset file)
# ---------------------------------------------------------------------------


def test_export_schema_csv_writes_an_already_shaped_dataframe_as_is(tmp_path):
    first = to_experiment_schema(_compare_result("hello"), dataset="ds_a")
    second = to_experiment_schema(_compare_result("world"), dataset="ds_b")
    combined = pd.concat([first, second], ignore_index=True)
    path = tmp_path / "combined.csv"

    export_schema_csv(combined, path)
    loaded = pd.read_csv(path)

    assert set(loaded["dataset"]) == {"ds_a", "ds_b"}
    assert len(loaded) == 4  # 2 tokenizers x 2 datasets


def test_export_schema_json_writes_an_already_shaped_dataframe_as_is(tmp_path):
    first = to_experiment_schema(_compare_result("hello"), dataset="ds_a")
    second = to_experiment_schema(_compare_result("world"), dataset="ds_b")
    combined = pd.concat([first, second], ignore_index=True)
    path = tmp_path / "combined.json"

    export_schema_json(combined, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert {row["dataset"] for row in payload} == {"ds_a", "ds_b"}
    assert len(payload) == 4


def test_export_schema_csv_overwrite_false_raises_if_file_exists(tmp_path):
    schema_df = to_experiment_schema(_compare_result(), dataset="demo")
    path = tmp_path / "results.csv"
    export_schema_csv(schema_df, path)

    with pytest.raises(FileExistsError):
        export_schema_csv(schema_df, path, overwrite=False)


def test_multiple_tokenizers_produce_multiple_rows(tmp_path):
    path = tmp_path / "results.json"
    export_results_json(_compare_result(), path, dataset="demo")

    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_names = {row["tokenizer"] for row in payload}

    assert tokenizer_names == {"character", "word"}
