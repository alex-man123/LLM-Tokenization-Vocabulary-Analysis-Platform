"""How LLMs Use Tokens page (Task 8.6): a purely illustrative walk through the pipeline.

    Text -> Tokenization -> Tokens -> Token IDs -> Embedding lookup -> Vectors -> Model

Only the first three arrows (text -> tokens -> token IDs) use real
components from this project (`tokenizers.registry`) — everything after
"Token IDs" is explicitly illustrative. This page does not implement, load,
or approximate a real embedding model: the "embeddings" shown are random,
fixed-seed-per-token-ID vectors generated with the standard library `random`
module, labeled as illustrative everywhere they appear. No `next_token`
prediction is actually computed; the pipeline stops at "a model would take
it from here."
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Streamlit runs each page as an independent script, so every page must set
# this up itself rather than relying on streamlit_app.py having run first.
_UI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UI_DIR.parent / "src"))
sys.path.insert(0, str(_UI_DIR))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from embeddings_3d import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSIONS,
    generate_illustrative_embeddings,
    pca_3d,
)
from theme import inject_theme  # noqa: E402

from tokenizers.registry import AVAILABLE_TOKENIZERS, create_tokenizer  # noqa: E402

_EMBEDDING_DIMENSIONS = 8

_PIPELINE_DIAGRAM = """\
┌─────────────────┐
│       TEXT       │
└─────────┬─────────┘
          ↓
┌─────────────────┐
│   TOKENIZATION    │   <- real: the tokenizer you picked below
└─────────┬─────────┘
          ↓
┌─────────────────┐
│      TOKENS       │   <- real
└─────────┬─────────┘
          ↓
┌─────────────────┐
│    TOKEN IDs       │   <- real
└─────────┬─────────┘
          ↓
┌───────────────────────────┐
│      EMBEDDING LOOKUP       │   <- ILLUSTRATIVE ONLY (random vectors)
└─────────────┬───────────────┘
              ↓
┌───────────────────────────┐
│          VECTORS             │   <- ILLUSTRATIVE ONLY
└─────────────┬───────────────┘
              ↓
┌───────────────────────────┐
│  MODEL -> NEXT-TOKEN PREDICTION │   <- not computed here at all
└───────────────────────────┘\
"""


def _illustrative_embedding(token_id: int, dimensions: int = _EMBEDDING_DIMENSIONS) -> list[float]:
    """A fixed, reproducible-per-token-ID vector of small random floats — nothing more.

    Seeding a private `random.Random` instance with `token_id` means the
    same token always shows the same "embedding" across reruns (a nicer
    demo than reshuffling on every interaction), without needing a model,
    a training run, or `numpy`. This is not how real embeddings are
    produced — real ones are learned parameters of a trained model.
    """
    rng = random.Random(token_id)
    return [round(rng.uniform(-1.0, 1.0), 4) for _ in range(dimensions)]


st.set_page_config(page_title="How LLMs Use Tokens", page_icon="🧠")
inject_theme()
st.title("How LLMs Use Tokens")

st.write(
    "LLMs don't read words directly — they process **token IDs**, which get "
    "mapped to **vectors (embeddings)** learned during the model's training. "
    "Those vectors are what the model actually computes with to predict the "
    "next token."
)

st.code(_PIPELINE_DIAGRAM, language=None)

st.warning(
    "**Everything from \"Embedding lookup\" onward on this page is "
    "illustrative.** The vectors shown are random numbers, not embeddings "
    "from any trained model, and no next-token prediction is actually "
    "computed here. Only the tokenization step (text → tokens → token IDs) "
    "uses this project's real tokenizers."
)

tokenizer_name = st.selectbox("Tokenizer", sorted(AVAILABLE_TOKENIZERS))
text = st.text_area("Text", value="Hello world")

if not text:
    st.info("Enter some text above to walk through the pipeline.")
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
            st.subheader("1. Text → Tokens → Token IDs (real)")
            st.dataframe(
                pd.DataFrame({"token": tokens, "token_id": token_ids}),
                width="stretch",
            )

            st.subheader("2. Token IDs → Illustrative embeddings")
            st.caption(
                f"{_EMBEDDING_DIMENSIONS}-dimensional random vectors, one per token ID — "
                "ILLUSTRATIVE ONLY, not a trained model's embeddings."
            )
            embedding_rows = [
                {
                    "token": token,
                    "token_id": token_id,
                    "illustrative_embedding": _illustrative_embedding(token_id),
                }
                for token, token_id in zip(tokens, token_ids, strict=True)
            ]
            st.dataframe(pd.DataFrame(embedding_rows), width="stretch")

            st.subheader("3. Vectors → Model → Next-token prediction")
            st.info(
                "A real LLM would feed these (real, learned) embedding vectors "
                "through its layers to predict the next token's ID. This page "
                "stops here — no model is implemented or simulated."
            )

            st.subheader("4. Illustrative 3D embedding space")
            st.warning(
                "Illustrative embeddings — randomly generated and not trained on "
                "language data. Token IDs → embeddings → vectors → this 3D plot; the "
                "distances between points say nothing about real semantic "
                "relationships between tokens."
            )
            st.caption(
                f"{DEFAULT_EMBEDDING_DIMENSIONS}-dimensional random vectors, one per "
                "distinct token ID above, reduced to 3 dimensions with a plain NumPy "
                "PCA (via SVD) purely so they can be plotted here."
            )
            unique_ids = sorted(set(token_ids))
            id_to_token = dict(zip(token_ids, tokens, strict=True))
            vectors = generate_illustrative_embeddings(unique_ids)
            coords = pca_3d(vectors)
            labels = [id_to_token[token_id] for token_id in unique_ids]
            scatter = go.Figure(
                data=[
                    go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode="markers+text",
                        text=labels,
                        textposition="top center",
                        customdata=unique_ids,
                        marker={"size": 6, "color": coords[:, 0], "colorscale": "Viridis"},
                        hovertemplate=(
                            "token=%{text}<br>token_id=%{customdata}<br>"
                            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                        ),
                    )
                ]
            )
            scatter.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=500)
            st.plotly_chart(scatter, width="stretch")
