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
_UI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UI_DIR.parent / "src"))
sys.path.insert(0, str(_UI_DIR))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from theme import TOKENIZER_FALLBACK_COLOR, color_map_for, inject_theme  # noqa: E402

from benchmarking.export import load_results_json  # noqa: E402
from experiments.aggregation import (  # noqa: E402
    aggregate_by_group,
    aggregate_by_group_and_tokenizer,
    describe_observations,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_PATH = _PROJECT_ROOT / "data" / "results" / "experiment_results.json"

st.set_page_config(page_title="Experiments", page_icon="🧪")
inject_theme()
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
            st.caption(
                "Averaged across every tokenizer in this run — not tied to any one "
                "tokenizer, so this chart uses a neutral accent color rather than one "
                "of the tokenizer-specific colors from the chart below."
            )
            by_group = aggregate_by_group(results, metrics=["compression_ratio", "tokens_per_word"])
            fig_by_group = px.bar(
                by_group,
                x="language_or_type",
                y="compression_ratio_mean",
                color_discrete_sequence=[TOKENIZER_FALLBACK_COLOR],
                labels={
                    "language_or_type": "Language / type",
                    "compression_ratio_mean": "Mean compression ratio",
                },
            )
            fig_by_group.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 10, "b": 0})
            st.plotly_chart(fig_by_group, width="stretch")

            st.subheader("Tokens per word per language/type, by tokenizer")
            by_group_and_tokenizer = aggregate_by_group_and_tokenizer(
                results, metrics=["tokens_per_word"]
            )
            fig_by_tokenizer = px.bar(
                by_group_and_tokenizer,
                x="language_or_type",
                y="tokens_per_word_mean",
                color="tokenizer",
                barmode="group",
                color_discrete_map=color_map_for(
                    sorted(by_group_and_tokenizer["tokenizer"].unique())
                ),
                labels={
                    "language_or_type": "Language / type",
                    "tokens_per_word_mean": "Mean tokens per word",
                },
            )
            fig_by_tokenizer.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0})
            st.plotly_chart(fig_by_tokenizer, width="stretch")

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
