"""Benchmark page (Task 8.5): a live, interactive run of the Comparator on user input.

**Benchmark vs. Compare vs. Experiments — three different things:**

- **Compare** (Task 8.3): live, on whatever text the user types, tokens +
  metrics side by side.
- **Benchmark** (this page): also live, on whatever text the user types,
  but additionally times each tokenizer's `encode`/`decode`
  (`benchmarking.timer`, Task 5.2) and turns token counts into an
  estimated cost at a user-chosen price (Task 8.7) — the two things
  Compare does not show.
- **Experiments** (`5_Experiments.py`, Task 8.5): the opposite of "live" —
  it loads *pre-computed* results from `data/results/` (Task 6.3/6.4) and
  never recomputes anything on the fly.

Tokenizer selection/construction is shared with Compare via
`ui/tokenizer_options.py`, so this page does not duplicate that logic —
only the timing and cost sections below are specific to Benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs each page as an independent script, so every page must set
# this up itself rather than relying on streamlit_app.py having run first.
_UI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UI_DIR.parent / "src"))
sys.path.insert(0, str(_UI_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from tokenizer_options import all_tokenizer_names, build_tokenizers  # noqa: E402

from benchmarking.comparator import compare_tokenizers  # noqa: E402
from benchmarking.timer import measure_tokenizer_timing  # noqa: E402
from tokenizers.registry import AVAILABLE_TOKENIZERS  # noqa: E402

_TIMING_ITERATIONS = 5

st.set_page_config(page_title="Benchmark", page_icon="📊")
st.title("Benchmark")
st.caption(
    "Live benchmark: pick tokenizers, enter text, and see metrics, timing, and "
    "an estimated cost recomputed on every change — nothing here is pre-computed. "
    "For results on this project's own fixed, reproducible datasets instead, see "
    "the **Experiments** page."
)

selected_names = st.multiselect(
    "Tokenizers to benchmark",
    all_tokenizer_names(),
    default=sorted(AVAILABLE_TOKENIZERS),
)
text = st.text_area("Text", value="Education is power.")

if not selected_names:
    st.info("Select at least one tokenizer above.")
elif not text:
    st.info("Enter some text above to run the benchmark.")
else:
    tokenizers, load_errors = build_tokenizers(selected_names, text)
    for name, error in load_errors.items():
        st.error(f"Could not load {name!r}: {error}")

    if not tokenizers:
        st.info("None of the selected tokenizers could be loaded.")
    else:
        try:
            results = compare_tokenizers(tokenizers, text)
        except Exception as exc:  # a comparator failure must not crash the page
            st.error(f"Could not benchmark these tokenizers: {exc}")
        else:
            st.subheader("Metrics")
            st.dataframe(results.drop(columns=["tokens"]), width="stretch")

            st.subheader("Timing")
            st.caption(
                f"{_TIMING_ITERATIONS} measured repetitions per tokenizer, after one "
                "unmeasured warm-up call each (see `benchmarking.timer`, Task 5.2)."
            )
            timing_rows = []
            for tokenizer in tokenizers:
                timing = measure_tokenizer_timing(tokenizer, text, n_iterations=_TIMING_ITERATIONS)
                timing_rows.append(
                    {
                        "tokenizer": tokenizer.name,
                        "encode_mean_ms": timing.encode.mean_ms,
                        "encode_median_ms": timing.encode.median_ms,
                        "decode_mean_ms": timing.decode.mean_ms,
                        "decode_median_ms": timing.decode.median_ms,
                    }
                )
            st.dataframe(pd.DataFrame(timing_rows), width="stretch")

            st.subheader("Cost estimator")
            st.caption(
                "Generic calculator: `estimated_cost = (num_tokens / 1_000_000) * "
                "price_per_million`. `price_per_million` is whatever you set below — "
                "not tied to any specific provider's real pricing."
            )
            price_per_million = st.number_input(
                "Price per 1,000,000 tokens (any currency/unit you like)",
                min_value=0.0,
                value=5.0,
                step=0.5,
            )
            cost_df = pd.DataFrame(
                {
                    "tokenizer": results["tokenizer"],
                    "characters": len(text),
                    "num_tokens": results["number_of_tokens"],
                    "price_per_million": price_per_million,
                    "estimated_cost": (results["number_of_tokens"] / 1_000_000)
                    * price_per_million,
                }
            )
            st.dataframe(cost_df, width="stretch")
