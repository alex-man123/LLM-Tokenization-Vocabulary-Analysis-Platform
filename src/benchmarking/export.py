"""Export benchmark results to CSV/JSON (Task 5.4).

Consumes whatever the Comparator (`benchmarking.comparator.compare_tokenizers`,
Task 5.3) already computed — this module never retokenizes or recomputes a
metric, it only reshapes and writes. Column names are aligned to the
documented Experiment Result Schema (Task 0.3, see the "Experiment result
schema" section of `docs/architecture.md`): `number_of_tokens` ->
`num_tokens`, `encoding_time`/`decoding_time` -> `encode_time_ms`/
`decode_time_ms`, plus the schema's `dataset`/`timestamp` fields, which the
Comparator's output does not carry on its own (they describe the run, not
a per-tokenizer metric).

The Comparator's `tokens` column is not part of the documented schema, but
Task 5.4 explicitly requires it to be preserved (never dropped to simplify
export), so it is kept as an additional field: a native list in JSON, a
JSON-encoded string in CSV (CSV has no native list type).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_COLUMN_RENAMES = {
    "number_of_tokens": "num_tokens",
    "encoding_time": "encode_time_ms",
    "decoding_time": "decode_time_ms",
}

# Column order for the documented schema fields that are present; any other
# column present in `results` (e.g. `tokens`) is appended after these, in
# its original order, rather than dropped.
_SCHEMA_COLUMN_ORDER = [
    "tokenizer",
    "dataset",
    "vocab_size",
    "num_tokens",
    "tokens_per_word",
    "characters_per_token",
    "compression_ratio",
    "encode_time_ms",
    "decode_time_ms",
    "timestamp",
]


def to_experiment_schema(
    results: pd.DataFrame, *, dataset: str, timestamp: str | None = None
) -> pd.DataFrame:
    """Reshape a Comparator-style DataFrame into the Task 0.3 experiment result schema.

    `dataset` is required (the schema marks it required — it identifies
    which text/corpus these rows were measured on, and the Comparator has
    no notion of "dataset" of its own to fill it in). `timestamp` defaults
    to the current UTC time in ISO 8601 format, matching how
    `vocabulary.serialization.save_tokenizer_state` defaults `trained_at`
    when not given explicitly.

    Any column from `results` other than the ones renamed/added here
    (e.g. `tokens`) is kept, appended after the schema's own columns —
    never silently dropped.
    """
    schema_df = results.rename(columns=_COLUMN_RENAMES).copy()
    schema_df["dataset"] = dataset
    schema_df["timestamp"] = timestamp or datetime.now(UTC).isoformat()

    ordered = [c for c in _SCHEMA_COLUMN_ORDER if c in schema_df.columns]
    remaining = [c for c in schema_df.columns if c not in ordered]
    return schema_df[ordered + remaining]


def _write_schema_csv(schema_df: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists and overwrite=False")

    csv_df = schema_df
    if "tokens" in csv_df.columns:
        csv_df = csv_df.copy()
        csv_df["tokens"] = csv_df["tokens"].apply(json.dumps)

    path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(path, index=False)


def _write_schema_json(schema_df: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists and overwrite=False")

    records = schema_df.to_dict(orient="records")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def export_schema_csv(
    schema_df: pd.DataFrame, path: str | Path, *, overwrite: bool = True
) -> None:
    """Write a DataFrame already shaped by `to_experiment_schema` to `path` as CSV.

    For a caller that already has a fully-stamped schema DataFrame — e.g.
    the Experiment Runner (Task 6.3), which stamps each dataset's rows with
    its own `dataset` value via `to_experiment_schema` *before* combining
    every dataset's results into one table — calling `to_experiment_schema`
    again here would overwrite those per-row values with a single shared
    one. `export_results_csv` below is the convenience path for the common
    case of exporting results for a single dataset in one call; this is the
    lower-level primitive it (and the Experiment Runner) both build on.
    Same directory-creation/`overwrite` behavior as `export_results_csv`.
    """
    _write_schema_csv(schema_df, Path(path), overwrite=overwrite)


def export_schema_json(
    schema_df: pd.DataFrame, path: str | Path, *, overwrite: bool = True
) -> None:
    """Write a DataFrame already shaped by `to_experiment_schema` to `path` as JSON.

    See `export_schema_csv` for why this lower-level primitive exists
    separately from `export_results_json`.
    """
    _write_schema_json(schema_df, Path(path), overwrite=overwrite)


def export_results_csv(
    results: pd.DataFrame,
    path: str | Path,
    *,
    dataset: str,
    timestamp: str | None = None,
    overwrite: bool = True,
) -> None:
    """Write `results` to `path` as CSV, matching the Task 0.3 schema.

    Creates any missing parent directories (e.g. `data/results/`) so a
    caller never has to create them manually first. If `path` already
    exists and `overwrite` is `False`, raises `FileExistsError` instead of
    silently replacing it; `overwrite=True` (the default) always writes.

    A `tokens` column (list of strings per row), if present, is JSON-encoded
    per cell — CSV has no native list type, and this keeps the information
    instead of discarding it.
    """
    schema_df = to_experiment_schema(results, dataset=dataset, timestamp=timestamp)
    _write_schema_csv(schema_df, Path(path), overwrite=overwrite)


def export_results_json(
    results: pd.DataFrame,
    path: str | Path,
    *,
    dataset: str,
    timestamp: str | None = None,
    overwrite: bool = True,
) -> None:
    """Write `results` to `path` as JSON (a list of row objects), matching the Task 0.3 schema.

    Same directory-creation and `overwrite` behavior as `export_results_csv`.
    Unlike CSV, a `tokens` column (if present) is kept as a native JSON
    array per row — no serialized-string workaround needed.
    """
    schema_df = to_experiment_schema(results, dataset=dataset, timestamp=timestamp)
    _write_schema_json(schema_df, Path(path), overwrite=overwrite)


def load_results_json(path: str | Path) -> pd.DataFrame:
    """Read back a DataFrame previously written by `export_results_json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(payload)
