"""Vocabulary page (Task 8.4): statistics for a trained tokenizer's vocabulary.

Thin UI only — no frequency-analysis logic here. Everything comes from
`vocabulary.frequency_analysis` (Task 4.3); this page never counts tokens
or ranks frequencies on its own. The selected tokenizer is trained live on
the text entered below, the same deliberate demo simplification used by
the Tokenize/Compare pages (Task 8.2/8.3) — it needs a corpus to compute
frequencies over anyway, so training and frequency-counting share the same
input text.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs each page as an independent script, so every page must set
# this up itself rather than relying on streamlit_app.py having run first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st  # noqa: E402

from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer  # noqa: E402
from vocabulary.frequency_analysis import (  # noqa: E402
    compute_token_frequencies,
    rare_tokens,
    to_dataframe,
    top_n_frequent_tokens,
)

st.set_page_config(page_title="Vocabulary", page_icon="📖")
st.title("Vocabulary")
st.caption(
    "This demo trains the selected tokenizer live on the text you enter, then "
    "reports how often each resulting token actually occurs in that same "
    "text — it is not pretrained on a large corpus, so treat the numbers as "
    "illustrative of the *method*, not of real-world token frequencies."
)

tokenizer_name = st.selectbox("Tokenizer", sorted(AVAILABLE_TOKENIZERS))
text = st.text_area(
    "Training corpus",
    value=(
        "Education is power. The quick brown fox jumps over the lazy dog. "
        "Education, education, education."
    ),
    height=120,
)

if not text:
    st.info("Enter some text above to train a tokenizer and see its vocabulary statistics.")
else:
    try:
        tokenizer = create_tokenizer(tokenizer_name)
        tokenizer.train([text])
        frequencies = compute_token_frequencies(tokenizer, [text])
    except Exception as exc:  # a tokenizer/analysis failure must not crash the page
        st.error(f"Could not analyze this tokenizer's vocabulary: {exc}")
    else:
        st.subheader("Vocabulary size")
        st.metric("Total registered tokens", tokenizer.vocab_size)
        st.caption(
            f"{len(frequencies)} of those tokens actually appear in the corpus above; "
            "the rest (e.g. special tokens, or characters/merges never used here) do not."
        )

        if not frequencies:
            st.info("This corpus produced no tokens to analyze.")
        else:
            st.subheader("Most frequent tokens")
            if len(frequencies) <= 1:
                top_n = len(frequencies)  # nothing meaningful to slide between
            else:
                top_n = st.slider(
                    "Top N",
                    min_value=1,
                    max_value=len(frequencies),
                    value=min(10, len(frequencies)),
                )
            top_entries = top_n_frequent_tokens(frequencies, top_n)
            top_df = to_dataframe(top_entries).set_index("token")
            st.bar_chart(top_df["frequency"])
            st.dataframe(to_dataframe(top_entries), width="stretch")

            st.caption(
                "Natural-language token frequencies are approximately **Zipfian**: a "
                "handful of tokens account for most occurrences, while most tokens in "
                "a vocabulary appear rarely — the 'long tail' visible below."
            )

            st.subheader("Rare tokens")
            rare_threshold = st.slider(
                "Rare threshold (frequency strictly below this)",
                min_value=1,
                max_value=max(2, max(frequencies.values())),
                value=2,
            )
            rare_entries = rare_tokens(frequencies, rare_threshold)
            if not rare_entries:
                st.info(f"No tokens with frequency below {rare_threshold} in this corpus.")
            else:
                st.dataframe(to_dataframe(rare_entries), width="stretch")
