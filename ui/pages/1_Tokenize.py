"""Tokenize page (Task 8.2): text in, tokens + token IDs out.

Thin UI only — no tokenization logic here. Everything comes from
`tokenizers.registry` and the selected tokenizer's own public API
(`train`, `tokenize`, `encode`). Token pill colors (Task 8.9/8.8) and the
BPE merge-by-merge walkthrough (Task 8.11) reuse existing central
modules (`ui/theme.py`, `tokenizers/bpe/visualization.py`) rather than
duplicating either concern here.
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
from theme import inject_theme, render_token_pills  # noqa: E402

from tokenizers.bpe.tokenizer import BPETokenizer  # noqa: E402
from tokenizers.bpe.visualization import compute_merge_steps, format_word_freqs  # noqa: E402
from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer  # noqa: E402

st.set_page_config(page_title="Tokenize", page_icon="🔤")
inject_theme()
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
            st.markdown(render_token_pills(tokens, tokenizer_name), unsafe_allow_html=True)

            st.subheader("Token IDs")
            st.dataframe(
                pd.DataFrame({"token": tokens, "token_id": token_ids}),
                width="stretch",
            )
            st.caption(f"Vocabulary size: {tokenizer.vocab_size}")

            if isinstance(tokenizer, BPETokenizer) and tokenizer.merges:
                with st.expander("BPE Merge Visualization"):
                    st.caption(
                        "Replays this tokenizer's own learned merges, in order, over the "
                        "text above — the same `apply_merge` function training itself "
                        "uses, not a second implementation of BPE."
                    )
                    merges = list(tokenizer.merges)
                    steps = compute_merge_steps([text], merges)
                    if len(steps) == 1:
                        step_index = 1
                    else:
                        step_index = st.slider(
                            "Merge step", min_value=1, max_value=len(steps), value=1
                        )
                    step = steps[step_index - 1]
                    st.markdown(
                        f"**Step {step.step} / {len(steps)} — selected pair:** "
                        f"`{step.pair[0]}` + `{step.pair[1]}` → `{step.pair[0]}{step.pair[1]}`"
                    )
                    before_col, after_col = st.columns(2)
                    with before_col:
                        st.markdown("**Corpus before merge**")
                        st.code("\n".join(format_word_freqs(step.before)), language=None)
                    with after_col:
                        st.markdown("**Corpus after merge**")
                        st.code("\n".join(format_word_freqs(step.after)), language=None)
