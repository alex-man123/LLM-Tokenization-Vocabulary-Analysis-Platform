"""Compare page (Task 8.3): run several tokenizers on the same text, side by side.

Thin UI only — no metrics or comparison logic here. Everything comes from
`benchmarking.comparator.compare_tokenizers` (Task 5.3), which itself uses
`benchmarking.metrics` (Task 5.1); this page never retokenizes or
recomputes a metric on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs each page as an independent script, so every page must set
# this up itself rather than relying on streamlit_app.py having run first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st  # noqa: E402

from benchmarking.comparator import compare_tokenizers  # noqa: E402
from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer  # noqa: E402

st.set_page_config(page_title="Compare", page_icon="⚖️")
st.title("Compare")

st.warning(
    "**Fair comparison:** a tokenizer's vocabulary size strongly affects its "
    "token count and compression ratio — a bigger vocabulary tends to produce "
    "fewer, longer tokens. The tokenizers below are trained live on the text "
    "you enter (not on a large corpus), so their `vocab_size` is small; a "
    "production tokenizer (e.g. GPT's ~100k-token vocabulary) is not directly "
    "comparable to these numbers. Always read `vocab_size` alongside every "
    "other metric here, never a metric alone."
)

selected_names = st.multiselect(
    "Tokenizers to compare",
    sorted(AVAILABLE_TOKENIZERS),
    default=sorted(AVAILABLE_TOKENIZERS),
)
text = st.text_area("Text", value="Education is power.")

if not selected_names:
    st.info("Select at least one tokenizer above.")
elif not text:
    st.info("Enter some text above to compare.")
else:
    try:
        tokenizers = []
        for name in selected_names:
            tokenizer = create_tokenizer(name)
            tokenizer.train([text])
            tokenizers.append(tokenizer)
        results = compare_tokenizers(tokenizers, text)
    except Exception as exc:  # a tokenizer/comparator failure must not crash the page
        st.error(f"Could not compare these tokenizers: {exc}")
    else:
        st.subheader("Metrics")
        st.dataframe(results.drop(columns=["tokens"]), width="stretch")

        st.subheader("Tokens")
        for row in results.itertuples():
            st.markdown(f"**{row.tokenizer}** (vocab size: {row.vocab_size})")
            st.write(row.tokens if row.tokens else "_(no tokens)_")
