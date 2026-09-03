"""Compare page (Task 8.3): run several tokenizers on the same text, side by side.

Thin UI only — no metrics or comparison logic here. Everything comes from
`benchmarking.comparator.compare_tokenizers` (Task 5.3), which itself uses
`benchmarking.metrics` (Task 5.1); this page never retokenizes or
recomputes a metric on its own. Tokenizer selection/construction is shared
with the Benchmark page (Task 8.5) via `ui/tokenizer_options.py`, so the
"don't crash on one bad tokenizer" logic is defined exactly once.

Alongside this project's own trainable tokenizers (character/word/BPE/
WordPiece, from `tokenizers.registry`) and SentencePiece (Task 7.3, trained
live like the others), two real production tokenizers are offered as
optional, explicitly opt-in selections: a Hugging Face `tokenizers`
adapter and a `tiktoken` adapter (Task 7.1/7.2). They are not selected by
default — loading a real pretrained tokenizer can require network access
on first use, and this page's default view should stay fast and work
offline; selecting one is a deliberate user action.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs each page as an independent script, so every page must set
# this up itself rather than relying on streamlit_app.py having run first.
_UI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UI_DIR.parent / "src"))
sys.path.insert(0, str(_UI_DIR))

import streamlit as st  # noqa: E402
from tokenizer_options import all_tokenizer_names, build_tokenizers  # noqa: E402

from benchmarking.comparator import compare_tokenizers  # noqa: E402
from tokenizers.registry import AVAILABLE_TOKENIZERS  # noqa: E402

st.set_page_config(page_title="Compare", page_icon="⚖️")
st.title("Compare")

st.warning(
    "**Fair comparison:** using the same input text for every tokenizer "
    "below does not by itself make this an apples-to-apples algorithm "
    "comparison. `vocab_size` strongly affects token count and compression "
    "ratio — a bigger vocabulary tends to produce fewer, longer tokens. "
    "This project's own tokenizers are trained live on the text you enter "
    "(small `vocab_size`); the Hugging Face/tiktoken options are real "
    "production tokenizers with vocabularies trained on massive external "
    "corpora (tens of thousands to ~100k tokens) — comparing them directly "
    "is useful for seeing the order of magnitude involved, not for "
    "concluding which *algorithm* is better. For an algorithm-level "
    "comparison, train tokenizers to the same vocabulary size on the same "
    "corpus instead. Always read `vocab_size` alongside every other metric "
    "here, never alone. See `docs/benchmarking_methodology.md` for the "
    "full methodology."
)

selected_names = st.multiselect(
    "Tokenizers to compare",
    all_tokenizer_names(),
    default=sorted(AVAILABLE_TOKENIZERS),
    help=(
        "The Hugging Face/tiktoken options are real, pretrained production "
        "tokenizers (not trained on the text below) and may need network "
        "access the first time they are selected. SentencePiece trains "
        "live on the text below like this project's own tokenizers, but "
        "needs enough text to support its vocabulary size — very short "
        "input may show an error."
    ),
)
text = st.text_area("Text", value="Education is power.")

if not selected_names:
    st.info("Select at least one tokenizer above.")
elif not text:
    st.info("Enter some text above to compare.")
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
            st.error(f"Could not compare these tokenizers: {exc}")
        else:
            st.subheader("Metrics")
            st.dataframe(results.drop(columns=["tokens"]), width="stretch")

            st.subheader("Tokens")
            for row in results.itertuples():
                st.markdown(f"**{row.tokenizer}** (vocab size: {row.vocab_size})")
                st.write(row.tokens if row.tokens else "_(no tokens)_")
