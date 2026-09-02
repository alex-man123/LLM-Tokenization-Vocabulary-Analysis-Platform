"""Unit tests for experiment result aggregation (Phase 6, Task 6.4)."""

import pandas as pd
import pytest

from experiments.aggregation import (
    aggregate_by_group,
    aggregate_by_group_and_tokenizer,
    describe_observations,
)


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
            {
                "tokenizer": "word",
                "dataset": "en",
                "language_or_type": "english",
                "vocab_size": 50,
                "num_tokens": 60,
                "tokens_per_word": 3.0,
                "characters_per_token": 2.0,
                "compression_ratio": 2.0,
                "encode_time_ms": None,
                "decode_time_ms": None,
            },
            {
                "tokenizer": "word",
                "dataset": "ja",
                "language_or_type": "japanese",
                "vocab_size": 50,
                "num_tokens": 40,
                "tokens_per_word": 10.0,
                "characters_per_token": 2.0,
                "compression_ratio": 2.0,
                "encode_time_ms": None,
                "decode_time_ms": None,
            },
        ]
    )


# ---------------------------------------------------------------------------
# aggregate_by_group_and_tokenizer
# ---------------------------------------------------------------------------


def test_aggregate_by_group_and_tokenizer_computes_mean_and_median():
    results = _sample_results()

    aggregated = aggregate_by_group_and_tokenizer(results, metrics=["tokens_per_word"])

    row = aggregated[
        (aggregated["language_or_type"] == "japanese") & (aggregated["tokenizer"] == "character")
    ].iloc[0]
    assert row["tokens_per_word_mean"] == 20.0
    assert row["tokens_per_word_median"] == 20.0


def test_aggregate_by_group_and_tokenizer_only_uses_available_metric_columns():
    results = _sample_results()

    aggregated = aggregate_by_group_and_tokenizer(results)

    # encode_time_ms/decode_time_ms are all-None here; pandas can still
    # aggregate them (mean/median of all-NaN is NaN), so the columns exist.
    assert "tokens_per_word_mean" in aggregated.columns
    assert "compression_ratio_mean" in aggregated.columns


def test_aggregate_by_group_and_tokenizer_can_group_by_dataset_instead():
    results = _sample_results()

    aggregated = aggregate_by_group_and_tokenizer(
        results, group_column="dataset", metrics=["num_tokens"]
    )

    assert set(aggregated["dataset"]) == {"en", "ja"}


def test_aggregate_by_group_and_tokenizer_separates_tokenizers():
    results = _sample_results()

    aggregated = aggregate_by_group_and_tokenizer(results, metrics=["tokens_per_word"])

    assert len(aggregated) == 4  # 2 language_or_type values x 2 tokenizers


# ---------------------------------------------------------------------------
# aggregate_by_group
# ---------------------------------------------------------------------------


def test_aggregate_by_group_averages_across_tokenizers():
    results = _sample_results()

    aggregated = aggregate_by_group(results, metrics=["tokens_per_word"])

    english_row = aggregated[aggregated["language_or_type"] == "english"].iloc[0]
    # (5.0 + 3.0) / 2 == 4.0, averaged across both tokenizers for "english".
    assert english_row["tokens_per_word_mean"] == 4.0


def test_aggregate_by_group_has_one_row_per_group_value():
    results = _sample_results()

    aggregated = aggregate_by_group(results, metrics=["tokens_per_word"])

    assert len(aggregated) == 2  # "english", "japanese"


# ---------------------------------------------------------------------------
# describe_observations
# ---------------------------------------------------------------------------


def test_describe_observations_produces_one_sentence_per_group_and_tokenizer():
    results = _sample_results()

    observations = describe_observations(results, metric="tokens_per_word")

    assert len(observations) == 4


def test_describe_observations_uses_dataset_scoped_language_not_universal_claims():
    results = _sample_results()

    observations = describe_observations(results, metric="tokens_per_word")

    for sentence in observations:
        assert "in this experiment's dataset" in sentence.lower()
    # Must not phrase it as a universal claim about the language/type itself.
    joined = " ".join(observations).lower()
    assert "japanese is" not in joined
    assert "english is" not in joined


def test_describe_observations_includes_the_actual_computed_value():
    results = _sample_results()

    observations = describe_observations(results, metric="tokens_per_word")

    assert any("20" in sentence and "'character'" in sentence for sentence in observations)


def test_describe_observations_raises_for_unknown_metric():
    results = _sample_results()

    with pytest.raises(ValueError):
        describe_observations(results, metric="not_a_real_metric")


def test_describe_observations_raises_when_metric_is_entirely_missing():
    results = _sample_results()

    with pytest.raises(ValueError):
        describe_observations(results, metric="encode_time_ms")


def test_describe_observations_skips_rows_with_missing_metric_values():
    results = _sample_results()
    results.loc[0, "tokens_per_word"] = None

    observations = describe_observations(results, metric="tokens_per_word")

    # 4 rows total, 1 has a missing value -> at most 3 (group,tokenizer) pairs.
    assert len(observations) <= 3
