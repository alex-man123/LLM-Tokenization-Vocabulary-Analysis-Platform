"""Experiments page (Task 8.5): pre-computed results from `data/results/`, never live.

**This page never recomputes anything.** Unlike Benchmark/Compare (live,
on whatever text the user types right now), Experiments only loads
whatever `scripts/run_experiments.py` (Task 6.3) already wrote to
`data/results/experiment_results.json`, and aggregates it with the
existing Task 6.4 functions (`experiments.aggregation`) — no tokenizer is
ever trained or run from this page. If that file does not exist yet, this
page says so plainly and stops; it never fabricates rows to make a chart
look populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs each page as an independent script, so every page must set
# this up itself rather than relying on streamlit_app.py having run first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st  # noqa: E402

from benchmarking.export import load_results_json  # noqa: E402
from experiments.aggregation import (  # noqa: E402
    aggregate_by_group,
    aggregate_by_group_and_tokenizer,
    describe_observations,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_PATH = _PROJECT_ROOT / "data" / "results" / "experiment_results.json"

st.set_page_config(page_title="Experiments", page_icon="🧪")
st.title("Experiments")
st.caption(
    "Pre-computed results from this project's own fixed datasets "
    "(`data/raw/`, Task 6.1), produced by `scripts/run_experiments.py` "
    "(Task 6.3) — reproducible, and never recalculated live from this page. "
    "For an interactive, live run on your own text, see **Benchmark** or "
    "**Compare** instead."
)

if not _RESULTS_PATH.exists():
    st.info(
        "No experiment results are available yet. Run `python "
        "scripts/run_experiments.py` from the project root to generate "
        f"`{_RESULTS_PATH.relative_to(_PROJECT_ROOT)}`, then reload this page."
    )
else:
    try:
        results = load_results_json(_RESULTS_PATH)
    except Exception as exc:  # a malformed results file must not crash the page
        st.error(f"Could not load experiment results: {exc}")
    else:
        if results.empty:
            st.info("The experiment results file exists but contains no rows.")
        else:
            st.caption(
                f"{len(results)} rows: {results['tokenizer'].nunique()} tokenizers x "
                f"{results['dataset'].nunique()} datasets "
                f"({', '.join(sorted(results['language_or_type'].unique()))})."
            )

            st.subheader("Compression ratio per language/type")
            by_group = aggregate_by_group(results, metrics=["compression_ratio", "tokens_per_word"])
            st.bar_chart(by_group.set_index("language_or_type")["compression_ratio_mean"])

            st.subheader("Tokens per word per language/type, by tokenizer")
            by_group_and_tokenizer = aggregate_by_group_and_tokenizer(
                results, metrics=["tokens_per_word"]
            )
            pivoted = by_group_and_tokenizer.pivot(
                index="language_or_type", columns="tokenizer", values="tokens_per_word_mean"
            )
            st.bar_chart(pivoted)

            st.subheader("Aggregated table")
            st.dataframe(by_group_and_tokenizer, width="stretch")

            st.subheader("Observations")
            st.caption(
                "Every sentence below describes what happened in *this* experiment's "
                "dataset — not a general claim about a language or text type "
                "(see `docs/experiment_results.md`)."
            )
            for sentence in describe_observations(results, metric="tokens_per_word"):
                st.markdown(f"- {sentence}")

            st.subheader("Raw results")
            st.dataframe(results.drop(columns=["tokens"]), width="stretch")
