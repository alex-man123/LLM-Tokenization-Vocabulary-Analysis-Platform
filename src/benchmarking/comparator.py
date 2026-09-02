"""Side-by-side comparison of multiple tokenizers on the same text (Task 5.3)."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from benchmarking.metrics import compute_metrics
from tokenizers.base import Tokenizer


def compare_tokenizers(tokenizers: list[Tokenizer], text: str) -> pd.DataFrame:
    """Run `text` through every tokenizer in `tokenizers` and return one row of metrics each.

    Every tokenizer must already be trained (or otherwise ready to
    `encode`/`tokenize`) — this function does not train anything. Each row
    also carries a `tokens` column (`tokenizer.tokenize(text)`) so a caller
    (e.g. the Streamlit Compare page) can display per-tokenizer tokens
    without retokenizing.

    `vocab_size` is always one of the returned columns. Custom tokenizers
    trained on a small corpus and production tokenizers with much larger
    vocabularies are not directly comparable on token count/compression
    ratio alone — reporting `vocab_size` alongside every result is what
    makes that visible instead of hidden (see "Fair comparison" in
    docs/architecture.md).
    """
    rows = []
    for tokenizer in tokenizers:
        row = asdict(compute_metrics(tokenizer, text))
        row["tokens"] = tokenizer.tokenize(text)
        rows.append(row)
    return pd.DataFrame(rows)
