"""Streamlit entry point (Task 8.1): thin app shell, no tokenization logic here.

Streamlit auto-discovers the sibling `pages/` directory when this script is
the one passed to `streamlit run`, and lists each page in the sidebar. This
script and every page only call into `src/` (tokenizers, benchmarking) —
none of them re-implement tokenization, metrics, or comparison.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="LLM Tokenization Lab", page_icon="🧩")
st.title("LLM Tokenization & Vocabulary Analysis Platform")
st.write(
    "Use the sidebar to navigate.\n\n"
    "- **Tokenize** — run one tokenizer over a piece of text.\n"
    "- **Compare** — run several tokenizers (including real external ones) "
    "on the same text, side by side.\n"
    "- **Vocabulary** — inspect a trained tokenizer's vocabulary size, "
    "most-frequent tokens, and rare tokens.\n"
    "- **Benchmark** — live comparison plus encode/decode timing and a "
    "generic tokenization cost estimator.\n"
    "- **Experiments** — pre-computed results from this project's own "
    "fixed datasets, never recalculated live.\n"
    "- **How LLMs Use Tokens** — a purely illustrative walk through "
    "text → tokens → token IDs → (illustrative) embeddings."
)
