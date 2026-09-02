"""Compare page (Task 8.3): run several tokenizers on the same text, side by side.

Thin UI only — no metrics or comparison logic here. Everything comes from
`benchmarking.comparator.compare_tokenizers` (Task 5.3), which itself uses
`benchmarking.metrics` (Task 5.1); this page never retokenizes or
recomputes a metric on its own.

Alongside this project's own trainable tokenizers (character/word/BPE/
WordPiece, from `tokenizers.registry`), two real production tokenizers are
offered as optional, explicitly opt-in selections: a Hugging Face
`tokenizers` adapter and a `tiktoken` adapter (Task 7.1/7.2). They are not
selected by default — loading a real pretrained tokenizer can require
network access on first use, and this page's default view should stay
fast and work offline; selecting one is a deliberate user action.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Streamlit runs each page as an independent script, so every page must set
# this up itself rather than relying on streamlit_app.py having run first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st  # noqa: E402

from benchmarking.comparator import compare_tokenizers  # noqa: E402
from tokenizers.adapters.huggingface_tokenizer import HuggingFaceTokenizerAdapter  # noqa: E402
from tokenizers.adapters.tiktoken_tokenizer import TiktokenAdapter  # noqa: E402
from tokenizers.base import Tokenizer  # noqa: E402
from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer  # noqa: E402

_EXTERNAL_TOKENIZER_FACTORIES: dict[str, Callable[[], Tokenizer]] = {
    "huggingface:bert-base-uncased": lambda: HuggingFaceTokenizerAdapter.from_pretrained(
        "bert-base-uncased"
    ),
    "tiktoken:cl100k_base": lambda: TiktokenAdapter.from_encoding_name("cl100k_base"),
}

# A failed load is retried after this long, in case network access has come
# back — short enough that a genuinely fixed connection is noticed within a
# session, long enough that a still-offline environment doesn't retry the
# network on every rerun (every keystroke in the text box, every widget
# change).
_EXTERNAL_TOKENIZER_RETRY_SECONDS = 300


@dataclass(frozen=True)
class _ExternalLoadResult:
    """Either a successfully loaded external tokenizer, or why it failed."""

    tokenizer: Tokenizer | None
    error: str | None


@st.cache_resource(
    show_spinner="Loading external tokenizer...", ttl=_EXTERNAL_TOKENIZER_RETRY_SECONDS
)
def _load_external_tokenizer(name: str) -> _ExternalLoadResult:
    """Build one of the real, pretrained external tokenizers, caching success *and* failure.

    Unlike this project's own tokenizers (cheaply retrained live on every
    rerun), constructing these can mean a Hugging Face Hub download or a
    `tiktoken` encoding-file fetch on first use. Caching only the success
    case would still leave a *failure* (e.g. no network) retried on every
    single rerun for as long as that tokenizer stays selected — caching the
    failure too, for `_EXTERNAL_TOKENIZER_RETRY_SECONDS`, avoids that retry
    storm while still eventually trying again.
    """
    try:
        return _ExternalLoadResult(tokenizer=_EXTERNAL_TOKENIZER_FACTORIES[name](), error=None)
    except Exception as exc:  # network/cache/model failure -> report, don't crash the page
        return _ExternalLoadResult(tokenizer=None, error=str(exc))


def _build_tokenizers(names: list[str], text: str) -> tuple[list[Tokenizer], dict[str, str]]:
    """Build every tokenizer in `names`; return the ones that succeeded plus `{name: error}`.

    One tokenizer that fails to load (typically an external one, offline)
    must not discard results for every *other* tokenizer in the selection
    — this project's own tokenizers train live and essentially never fail,
    so a comparison should still show them even if, say, the Hugging Face
    option couldn't be reached.
    """
    tokenizers: list[Tokenizer] = []
    errors: dict[str, str] = {}
    for name in names:
        if name in _EXTERNAL_TOKENIZER_FACTORIES:
            result = _load_external_tokenizer(name)
            if result.error is not None:
                errors[name] = result.error
            else:
                tokenizers.append(result.tokenizer)
            continue
        try:
            tokenizer = create_tokenizer(name)
            tokenizer.train([text])
            tokenizers.append(tokenizer)
        except Exception as exc:  # a tokenizer failure must not crash the page
            errors[name] = str(exc)
    return tokenizers, errors


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

all_tokenizer_names = sorted(AVAILABLE_TOKENIZERS) + sorted(_EXTERNAL_TOKENIZER_FACTORIES)
selected_names = st.multiselect(
    "Tokenizers to compare",
    all_tokenizer_names,
    default=sorted(AVAILABLE_TOKENIZERS),
    help=(
        "The Hugging Face/tiktoken options are real, pretrained production "
        "tokenizers (not trained on the text below) and may need network "
        "access the first time they are selected."
    ),
)
text = st.text_area("Text", value="Education is power.")

if not selected_names:
    st.info("Select at least one tokenizer above.")
elif not text:
    st.info("Enter some text above to compare.")
else:
    tokenizers, load_errors = _build_tokenizers(selected_names, text)
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
