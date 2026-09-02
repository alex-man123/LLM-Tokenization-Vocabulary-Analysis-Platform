"""Tokenize page (Task 8.2): text in, tokens + token IDs out.

Thin UI only — no tokenization logic here. Everything comes from
`tokenizers.registry` and the selected tokenizer's own public API
(`train`, `tokenize`, `encode`).
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Streamlit runs each page as an independent script, so every page must set
# this up itself rather than relying on streamlit_app.py having run first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer  # noqa: E402

_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#f0c419",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46b8b8",
    "#e83e8c",
    "#8bc34a",
    "#7986cb",
]

st.set_page_config(page_title="Tokenize", page_icon="🔤")
st.title("Tokenize")
st.caption(
    "This demo trains the selected tokenizer live on the text you enter, so it "
    "can tokenize anything without unknown tokens — it is not pretrained on a "
    "large corpus, so its vocabulary size will be small."
)

tokenizer_name = st.selectbox("Tokenizer", sorted(AVAILABLE_TOKENIZERS))
text = st.text_area("Text", value="Education is power.")

if not text:
    st.info("Enter some text above to see its tokens.")
else:
    try:
        tokenizer = create_tokenizer(tokenizer_name)
        tokenizer.train([text])
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
    except Exception as exc:  # a tokenizer failure must not crash the page
        st.error(f"Could not tokenize this input: {exc}")
    else:
        if not tokens:
            st.info("This input produced no tokens.")
        else:
            st.subheader("Tokens")
            spans = "".join(
                f'<span style="background:{_PALETTE[i % len(_PALETTE)]};'
                f"color:#000;padding:2px 6px;margin:2px;border-radius:4px;"
                f'display:inline-block;font-family:monospace;">'
                f"{html.escape(token)}</span>"
                for i, token in enumerate(tokens)
            )
            st.markdown(spans, unsafe_allow_html=True)

            st.subheader("Token IDs")
            st.dataframe(
                pd.DataFrame({"token": tokens, "token_id": token_ids}),
                width="stretch",
            )
            st.caption(f"Vocabulary size: {tokenizer.vocab_size}")
