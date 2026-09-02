"""Experiment Runner: the Tokenizer x Dataset matrix (Task 6.3).

Automates running the existing benchmarking pipeline over every
combination of tokenizer and dataset, instead of manually invoking it once
per pair. Every step is reused, not reimplemented:

    Dataset Loader (Task 6.2, experiments.dataset_loader)
          |
          v
    Tokenizer (tokenizers.registry, trained live on that dataset's text)
          |
          v
    Comparator (Task 5.3, benchmarking.comparator.compare_tokenizers)
          |
          v
    Experiment schema (Task 5.4, benchmarking.export.to_experiment_schema)
          |
          v
    Combined results table  ->  optionally exported (Task 5.4)

Each dataset's `compare_tokenizers` call and schema-stamping happens
separately (so every row gets its *own* dataset's name, not one value
shared across the whole matrix), and the per-dataset tables are then
concatenated into one combined `pandas.DataFrame` — this is why
`export_schema_csv`/`export_schema_json` (not `export_results_csv`/
`export_results_json`) are the export primitives used for the final
write: the combined table is already fully schema-shaped and must not be
re-stamped with a single shared `dataset` value.

Reproducibility: for a fixed `ExperimentConfig`, running this twice
produces identical results, because every step it calls is itself
deterministic — the tokenizers' own training/encoding (Phase 2/3), the
Comparator (Task 5.3), and the schema reshaping (Task 5.4) introduce no
randomness. No seeding is needed because nothing here is random.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from benchmarking.comparator import compare_tokenizers
from benchmarking.export import export_schema_csv, export_schema_json, to_experiment_schema
from experiments.dataset_loader import DATASET_CATEGORIES, load_dataset
from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer

#: `language_or_type` is metadata (Task 6.2), not part of the Task 0.3
#: experiment schema itself, but the Aggregation step (Task 6.4) needs it on
#: every row to group by language/text-type, so the runner carries it
#: through as an extra column rather than making 6.4 re-load every dataset
#: just to look it up again.
_LANGUAGE_OR_TYPE_COLUMN = "language_or_type"


@dataclass(frozen=True)
class ExperimentConfig:
    """Which tokenizers and datasets make up one experiment run.

    Defaults to every tokenizer in `tokenizers.registry.AVAILABLE_TOKENIZERS`
    (this project's own, deterministic, network-free tokenizers) and every
    category in `experiments.dataset_loader.DATASET_CATEGORIES` — the full
    matrix the Definition of Done describes. Pass a narrower
    `tokenizer_names`/`dataset_categories` to run a subset (e.g. for a
    quick, fast test).
    """

    tokenizer_names: tuple[str, ...] = field(
        default_factory=lambda: tuple(sorted(AVAILABLE_TOKENIZERS))
    )
    dataset_categories: tuple[str, ...] = DATASET_CATEGORIES


def run_experiment(config: ExperimentConfig | None = None) -> pd.DataFrame:
    """Run every (tokenizer, dataset) combination in `config` and return one combined table.

    For each dataset: load and normalize its text (Task 6.2), train a
    fresh instance of every configured tokenizer on that text, run them
    through the Comparator (Task 5.3), and stamp the result with that
    dataset's name and metadata via `to_experiment_schema` (Task 5.4).
    Every dataset's rows are then concatenated into one table — this is
    the "matrix" the Definition of Done asks for, with metadata preserved
    so it can later be grouped by language/text-type/tokenizer (Task 6.4).
    """
    config = config or ExperimentConfig()
    per_dataset_tables: list[pd.DataFrame] = []
    for dataset_name in config.dataset_categories:
        text, metadata = load_dataset(dataset_name)

        tokenizers = []
        for tokenizer_name in config.tokenizer_names:
            tokenizer = create_tokenizer(tokenizer_name)
            tokenizer.train([text])
            tokenizers.append(tokenizer)

        comparison = compare_tokenizers(tokenizers, text)
        schema_df = to_experiment_schema(comparison, dataset=dataset_name)
        schema_df[_LANGUAGE_OR_TYPE_COLUMN] = metadata.language_or_type
        per_dataset_tables.append(schema_df)

    if not per_dataset_tables:
        return pd.DataFrame()
    return pd.concat(per_dataset_tables, ignore_index=True)


def run_and_export_experiment(
    config: ExperimentConfig | None = None,
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Run `run_experiment` and, if given, write the combined table via Task 5.4's exporter.

    Returns the combined `pandas.DataFrame` either way, so a caller can use
    the in-memory result (e.g. for Task 6.4 aggregation) without re-reading
    the file it was just written to.
    """
    results = run_experiment(config)
    if csv_path is not None:
        export_schema_csv(results, csv_path, overwrite=overwrite)
    if json_path is not None:
        export_schema_json(results, json_path, overwrite=overwrite)
    return results
