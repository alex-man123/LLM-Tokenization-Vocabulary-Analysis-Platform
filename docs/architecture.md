# Architecture (draft)

This document tracks the project's data conventions and schemas as they are
established. It grows with each phase; Phase 0 only covers the raw text
convention and the experiment result schema.

## Raw text data (`data/raw/*.txt`)

- Encoding: UTF-8, no BOM.
- One file per dataset/category (e.g. `en.txt`, `ro.txt`, `es.txt`, `ja.txt`,
  `code_python.txt`, `urls.txt`, `numbers.txt`, `emoji.txt`, `technical.txt`).
- File name: lowercase, `snake_case`, no spaces, `.txt` extension.
- Line endings: `\n` (LF). Files should not rely on a specific number of
  lines or trailing newline behaviour — tokenizers must treat the raw text
  as-is.
- Raw files are input data, not build output: nothing in the codebase may
  modify files under `data/raw/` automatically. Adding or replacing a
  dataset is a manual, reviewed change.

## Experiment result schema (JSON)

One JSON object per experiment run (one tokenizer × one dataset).

| Field                | Type   | Required | Unit / format        | Meaning                                                             |
|----------------------|--------|----------|-----------------------|----------------------------------------------------------------------|
| `tokenizer`          | string | yes      | —                      | Value of the tokenizer's `name` property (e.g. `"bpe_custom_v1"`).   |
| `dataset`             | string | yes      | —                      | Name of the raw dataset used, without extension (e.g. `"en"`).      |
| `vocab_size`          | int    | yes      | count                  | Tokenizer's `vocab_size` at the time of the run.                    |
| `num_tokens`          | int    | yes      | count                  | Total number of tokens produced by `encode` on the dataset.         |
| `compression_ratio`   | float  | yes      | characters / token     | `len(raw_text) / num_tokens`. Higher means fewer tokens per character. |
| `encode_time_ms`      | float  | no       | milliseconds           | Wall-clock time to encode the dataset once.                          |
| `decode_time_ms`      | float  | no       | milliseconds           | Wall-clock time to decode the produced IDs back to text.             |
| `timestamp`           | string | no       | ISO 8601 (UTC)          | When the experiment was run.                                        |

Example (see [`data/results/example_result.json`](../data/results/example_result.json),
a dummy file kept only to illustrate the schema):

```json
{
  "tokenizer": "bpe_custom_v1",
  "dataset": "en",
  "vocab_size": 500,
  "num_tokens": 128,
  "compression_ratio": 3.42,
  "encode_time_ms": 1.23,
  "decode_time_ms": 0.87,
  "timestamp": "2026-09-02T00:00:00Z"
}
```

## Experiment result schema (CSV)

Same fields as the JSON schema, flattened to columns, for consumption in
Pandas/Streamlit. One row per experiment run.

```csv
tokenizer,dataset,vocab_size,num_tokens,compression_ratio,encode_time_ms,decode_time_ms,timestamp
bpe_custom_v1,en,500,128,3.42,1.23,0.87,2026-09-02T00:00:00Z
```

Optional fields (`encode_time_ms`, `decode_time_ms`, `timestamp`) may be
empty cells when not measured, but the column must still be present so that
results from different runs can be concatenated into one table.
