"""Analysis & aggregation of Experiment Runner results (Task 6.4).

Consumes the combined table `experiments.runner.run_experiment` produces
(or any DataFrame shaped like it — the Task 0.3 experiment schema plus a
`language_or_type` column) and aggregates it with `pandas.groupby`/`.agg`
— no manual aggregation loops, per Task 6.4's explicit instruction that
Pandas already does this directly.

**Interpretation guardrail (Task 6.4.5, load-bearing):** every sentence
`describe_observations` produces is phrased as an observation about *this
experiment's dataset* ("in this experiment's dataset, tokenizer X
produced..."), never as a claim about a language or text type in general.
A single small corpus per category does not generalize to "Japanese is
harder to tokenize" — only to "in this run, on this corpus, tokenizer X
needed more tokens per word for the Japanese-labeled text." Callers
building further text (e.g. `docs/experiment_results.md`) from this
module's output must preserve that framing, not strip it for brevity.
"""

from __future__ import annotations

import pandas as pd

#: Every numeric metric column the Task 0.3 schema/Comparator can produce,
#: in the order they should be reported. Aggregation only ever operates on
#: whichever of these are actually present in a given results table (e.g.
#: `encode_time_ms`/`decode_time_ms` are `None` until Task 5.2's Timer is
#: wired into a caller's own results).
METRIC_COLUMNS: tuple[str, ...] = (
    "num_tokens",
    "tokens_per_word",
    "characters_per_token",
    "compression_ratio",
    "vocab_size",
    "encode_time_ms",
    "decode_time_ms",
)


def _available_metrics(results: pd.DataFrame, metrics: list[str] | None) -> list[str]:
    if metrics is not None:
        return metrics
    return [m for m in METRIC_COLUMNS if m in results.columns]


def aggregate_by_group_and_tokenizer(
    results: pd.DataFrame,
    group_column: str = "language_or_type",
    *,
    metrics: list[str] | None = None,
    stats: tuple[str, ...] = ("mean", "median"),
) -> pd.DataFrame:
    """Group `results` by `(group_column, "tokenizer")` and aggregate each metric column.

    `group_column` defaults to `"language_or_type"` (set by
    `experiments.dataset_loader`/`experiments.runner`), covering both the
    "by language" and "by text type" groupings the Definition of Done asks
    for — this project's dataset metadata uses one field for both concepts
    (e.g. `"japanese"`, `"python_code"`). Pass `group_column="dataset"` to
    aggregate per raw dataset file instead.

    Columns are named `"{metric}_{stat}"` (e.g. `"tokens_per_word_mean"`),
    flattened from pandas' `.agg` MultiIndex result so the returned table
    is immediately usable for display or as a Markdown table, without a
    caller having to flatten it itself.
    """
    metrics = _available_metrics(results, metrics)
    grouped = results.groupby([group_column, "tokenizer"])[metrics].agg(list(stats))
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def aggregate_by_group(
    results: pd.DataFrame,
    group_column: str = "language_or_type",
    *,
    metrics: list[str] | None = None,
    stats: tuple[str, ...] = ("mean", "median"),
) -> pd.DataFrame:
    """Group `results` by `group_column` alone (across every tokenizer) and aggregate.

    Useful for a coarser summary than `aggregate_by_group_and_tokenizer`
    (e.g. "average compression ratio per language, across all tokenizers
    tested"). Same column-flattening behavior.
    """
    metrics = _available_metrics(results, metrics)
    grouped = results.groupby(group_column)[metrics].agg(list(stats))
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def describe_observations(
    results: pd.DataFrame,
    *,
    group_column: str = "language_or_type",
    metric: str = "tokens_per_word",
) -> list[str]:
    """One dataset-scoped sentence per `(group_value, tokenizer)` pair's mean `metric`.

    Every sentence follows the same template: "In this experiment's
    dataset, tokenizer '<name>' produced a mean <metric> of <value> for
    <group_column>='<value>'." — deliberately not "Language X has Y
    property", since a single small corpus per category cannot support
    that broader claim (see the module docstring).

    Raises:
        ValueError: if `metric` is not a column in `results` (including
            when it exists but every value is missing, e.g. the
            `encode_time_ms`/`decode_time_ms` columns before Task 5.2's
            Timer has been wired into the caller's own results).
    """
    if metric not in results.columns:
        raise ValueError(f"Unknown metric column: {metric!r}")
    if results[metric].isna().all():
        raise ValueError(f"Metric column {metric!r} has no values to aggregate")

    grouped = results.dropna(subset=[metric]).groupby([group_column, "tokenizer"])[metric].mean()
    return [
        f"In this experiment's dataset, tokenizer {tokenizer!r} produced a mean "
        f"{metric} of {value:.3g} for {group_column}={group_value!r}."
        for (group_value, tokenizer), value in grouped.items()
    ]
