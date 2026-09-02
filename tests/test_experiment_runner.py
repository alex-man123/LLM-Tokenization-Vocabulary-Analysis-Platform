"""Unit/integration tests for the Experiment Runner (Phase 6, Task 6.3)."""

import pandas as pd

from benchmarking.export import load_results_json
from experiments.dataset_loader import DATASET_CATEGORIES
from experiments.runner import ExperimentConfig, run_and_export_experiment, run_experiment
from tokenizers.registry import AVAILABLE_TOKENIZERS


def _without_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the wall-clock `timestamp` column before comparing two runs for determinism."""
    return df.drop(columns=["timestamp"])


def test_default_config_covers_every_registered_tokenizer_and_dataset():
    config = ExperimentConfig()

    assert set(config.tokenizer_names) == set(AVAILABLE_TOKENIZERS)
    assert set(config.dataset_categories) == set(DATASET_CATEGORIES)


def test_single_dataset_single_tokenizer():
    config = ExperimentConfig(tokenizer_names=("character",), dataset_categories=("en",))

    results = run_experiment(config)

    assert len(results) == 1
    assert results.iloc[0]["tokenizer"] == "character"
    assert results.iloc[0]["dataset"] == "en"
    assert results.iloc[0]["language_or_type"] == "english"


def test_multiple_datasets_produce_one_row_per_dataset_per_tokenizer():
    config = ExperimentConfig(tokenizer_names=("character",), dataset_categories=("en", "ro"))

    results = run_experiment(config)

    assert len(results) == 2
    assert set(results["dataset"]) == {"en", "ro"}


def test_multiple_tokenizers_produce_one_row_each_per_dataset():
    config = ExperimentConfig(tokenizer_names=("character", "word"), dataset_categories=("en",))

    results = run_experiment(config)

    assert len(results) == 2
    assert set(results["tokenizer"]) == {"character", "word"}


def test_full_default_matrix_runs_without_error():
    results = run_experiment(ExperimentConfig())

    assert len(results) == len(AVAILABLE_TOKENIZERS) * len(DATASET_CATEGORIES)
    assert set(results["dataset"]) == set(DATASET_CATEGORIES)
    assert set(results["tokenizer"]) == set(AVAILABLE_TOKENIZERS)
    assert (results["vocab_size"] > 0).all()
    assert (results["num_tokens"] >= 0).all()


def test_language_or_type_metadata_is_preserved_per_dataset():
    config = ExperimentConfig(
        tokenizer_names=("word",), dataset_categories=("ja", "code_python")
    )

    results = run_experiment(config)

    by_dataset = results.set_index("dataset")["language_or_type"]
    assert by_dataset["ja"] == "japanese"
    assert by_dataset["code_python"] == "python_code"


def test_running_the_same_config_twice_is_deterministic():
    config = ExperimentConfig(tokenizer_names=("bpe", "wordpiece"), dataset_categories=("en", "ro"))

    first = run_experiment(config)
    second = run_experiment(config)

    pd.testing.assert_frame_equal(
        _without_timestamp(first).reset_index(drop=True),
        _without_timestamp(second).reset_index(drop=True),
    )


def test_empty_config_returns_an_empty_dataframe():
    config = ExperimentConfig(tokenizer_names=(), dataset_categories=())

    results = run_experiment(config)

    assert len(results) == 0


def test_run_and_export_experiment_writes_a_loadable_csv(tmp_path):
    config = ExperimentConfig(
        tokenizer_names=("character", "word"), dataset_categories=("en", "ro")
    )
    csv_path = tmp_path / "experiment_results.csv"

    results = run_and_export_experiment(config, csv_path=csv_path)
    loaded = pd.read_csv(csv_path)

    assert len(loaded) == len(results) == 4
    assert set(loaded["dataset"]) == {"en", "ro"}


def test_run_and_export_experiment_writes_a_loadable_json(tmp_path):
    config = ExperimentConfig(tokenizer_names=("character",), dataset_categories=("en", "es"))
    json_path = tmp_path / "experiment_results.json"

    results = run_and_export_experiment(config, json_path=json_path)
    loaded = load_results_json(json_path)

    assert len(loaded) == len(results) == 2
    assert set(loaded["dataset"]) == {"en", "es"}


def test_run_and_export_experiment_can_write_both_formats_at_once(tmp_path):
    config = ExperimentConfig(tokenizer_names=("character",), dataset_categories=("en",))
    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "results.json"

    run_and_export_experiment(config, csv_path=csv_path, json_path=json_path)

    assert csv_path.exists()
    assert json_path.exists()


def test_run_and_export_experiment_without_paths_only_returns_the_dataframe(tmp_path):
    config = ExperimentConfig(tokenizer_names=("character",), dataset_categories=("en",))

    results = run_and_export_experiment(config)

    assert len(results) == 1
