"""Centralized visual theme (Task 8.9/8.8): tokenizer colors, CSS, token pills.

Single source of truth for "what color represents this tokenizer" and for
the CSS applied across every page. Every other UI component (token pill
markup here, Plotly `color_discrete_map`s in Vocabulary/Experiments, Task
8.10) reads colors through `get_tokenizer_color`/`color_map_for` instead of
hardcoding a hex value of its own — change a color once, here, and it is
consistent everywhere.
"""

from __future__ import annotations

import html

#: One accent color per tokenizer *family*. Keys are the short family name
#: (see `tokenizer_family`), not a full selectable name like
#: "tiktoken:cl100k_base" or "huggingface:bert-base-uncased" — every
#: pretrained-model variant of the same underlying library shares one color.
TOKENIZER_COLORS: dict[str, str] = {
    "bpe": "#4F9DDE",
    "character": "#F2994A",
    "word": "#27AE60",
    "wordpiece": "#BB6BD9",
    "huggingface": "#EB5757",
    "tiktoken": "#F2C94C",
    "sentencepiece": "#56CCF2",
}

#: Used for any tokenizer name not in `TOKENIZER_COLORS` (e.g. a future
#: tokenizer added to the registry before this dict is updated) — the app
#: must never crash or raise just because a name is unrecognized.
TOKENIZER_FALLBACK_COLOR = "#8A94A6"

#: Category colors for the "How LLMs Use Tokens" pipeline diagram (Task
#: 8.13) — kept separate from `TOKENIZER_COLORS` (which is keyed by
#: tokenizer *identity*, not pipeline stage), but still one central place
#: so the diagram, its legend, and nothing else define these colors.
#: `"pipeline_real"` marks a stage this project actually computes (text ->
#: tokens -> token IDs); `"pipeline_illustrative"` marks a stage that is
#: only a labeled illustration (embedding lookup onward) — the same
#: real/illustrative split the page's existing text disclaimer states.
PIPELINE_COLORS: dict[str, str] = {
    "pipeline_real": "#4F9DDE",
    "pipeline_illustrative": "#E0A836",
}


def tokenizer_family(name: str) -> str:
    """The color-lookup key for a tokenizer name, e.g. `"tiktoken:cl100k_base"` -> `"tiktoken"`.

    This project's own tokenizers (`"bpe"`, `"character"`, ...) have no
    `":"` and are their own family name.
    """
    return name.split(":", 1)[0]


def get_tokenizer_color(name: str) -> str:
    """The accent color for tokenizer `name`, falling back to `TOKENIZER_FALLBACK_COLOR`."""
    return TOKENIZER_COLORS.get(tokenizer_family(name), TOKENIZER_FALLBACK_COLOR)


def color_map_for(names: list[str]) -> dict[str, str]:
    """A `{name: color}` map covering exactly `names`, for Plotly's `color_discrete_map`.

    Building this per chart (rather than passing `TOKENIZER_COLORS`
    directly) guarantees every category gets an explicit color — including
    the fallback for an unrecognized name — instead of silently falling
    back to Plotly's own default color cycle for anything missing from
    `TOKENIZER_COLORS`.
    """
    return {name: get_tokenizer_color(name) for name in names}


def _blend_with_white(hex_color: str, amount: float) -> str:
    """Blend `hex_color` toward white by `amount` (0 = unchanged, 1 = white).

    Used to give adjacent token pills of the *same* tokenizer two
    distinguishable shades of that tokenizer's one color, instead of
    either a same-color wall of pills or an unrelated rainbow palette.
    """
    stripped = hex_color.lstrip("#")
    r, g, b = (int(stripped[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def render_token_pills(tokens: list[str], tokenizer_name: str) -> str:
    """HTML for `tokens` as colored pills, all in `tokenizer_name`'s color (Task 8.8).

    Adjacent tokens alternate between the tokenizer's base color and a
    lighter tint of that *same* color, purely so individual tokens stay
    visually distinguishable — the hue itself is still a single source of
    truth (`get_tokenizer_color`), not a second, independent palette.
    """
    base = get_tokenizer_color(tokenizer_name)
    shades = (base, _blend_with_white(base, 0.35))
    return "".join(
        f'<span class="token-pill" style="background:{shades[i % 2]};">'
        f"{html.escape(token)}</span>"
        for i, token in enumerate(tokens)
    )


_FONT_IMPORT_URL = (
    "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"
)

#: Global CSS injected once per page (Task 8.8): a monospace font for
#: tokens/IDs/code/merge rules, token-pill styling driven by whatever color
#: `render_token_pills` computed inline (never a second, hardcoded color
#: here), and a small set of accent variables so the app reads as one
#: coherent "technical AI/NLP dashboard" rather than Streamlit's default
#: generic dark theme.
_THEME_CSS = f"""
<style>
@import url('{_FONT_IMPORT_URL}');

:root {{
    --accent-primary: {TOKENIZER_COLORS["bpe"]};
    --accent-secondary: {TOKENIZER_COLORS["wordpiece"]};
    --surface: rgba(255, 255, 255, 0.04);
    --text-muted: #9aa4b2;
}}

code, pre, .token-pill, [data-testid="stDataFrame"] * {{
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace !important;
}}

h1 {{
    border-bottom: 3px solid var(--accent-primary);
    padding-bottom: 0.3rem;
}}

.token-pill {{
    display: inline-block;
    padding: 2px 10px;
    margin: 2px;
    border-radius: 999px;
    color: #0b0e14;
    font-weight: 600;
    font-size: 0.92rem;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}}

.token-pill:hover {{
    transform: translateY(-1px);
    box-shadow: 0 2px 8px var(--accent-secondary);
}}

/* Task 8.15: the sidebar's own vertical stack (collapse-button header,
   auto-generated page nav, this page's own `st.sidebar` content) is a
   plain block layout by default; making it a flex column lets `order`
   below move our custom brand block between the header and the nav,
   without touching the header itself. */
[data-testid="stSidebarContent"] {{
    display: flex !important;
    flex-direction: column !important;
}}

/* Push the auto-generated page list after our brand block (order 0,
   default, wins the tie with the header via DOM order) instead of
   pulling the brand above the header too. */
[data-testid="stSidebarNav"] {{
    order: 1 !important;
}}

/* Hides only the nav entry for the main script (`streamlit_app.py`),
   whose auto-generated label is the literal, generic "streamlit app" —
   Streamlit derives it from the entry-point filename, underscores
   replaced with spaces (see `streamlit.source_util.page_icon_and_name`).
   Matched by its href, which always ends in exactly "/" for the app's
   root/home page regardless of which page is currently active — every
   *other* nav link's href ends in "/PageName", never a bare "/". */
a[data-testid="stSidebarNavLink"][href$="/"] {{
    display: none !important;
}}

.sidebar-brand {{
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    padding: 14px 4px 2px 4px;
}}

.sidebar-brand-subtitle {{
    padding: 2px 4px 12px 4px;
    font-size: 0.75rem;
    color: var(--text-muted);
}}
</style>
"""

#: (pill label, tokenizer family) for the sidebar brand (Task 8.15) —
#: colors come from `TOKENIZER_COLORS`, the same palette every token pill
#: elsewhere in the app already uses, so the project's own name is built
#: from its own central visual concept instead of a one-off logo asset.
_BRAND_PILLS: tuple[tuple[str, str], ...] = (
    ("To", "bpe"),
    ("ken", "wordpiece"),
    ("Lab", "tiktoken"),
)

_BRAND_SUBTITLE = "LLM Tokenization Lab"


def render_sidebar_brand() -> str:
    """HTML for the sidebar brand: the project name as colored token pills.

    Reuses the same `.token-pill` CSS class and `TOKENIZER_COLORS` palette
    as every other token pill in the app (Tokenize/Compare, Task 8.9/8.8)
    — literally the app's own tokenization concept, applied to its own
    name, not a separate design.
    """
    pills = "".join(
        f'<span class="token-pill" style="background:{TOKENIZER_COLORS[family]};">'
        f"{html.escape(label)}</span>"
        for label, family in _BRAND_PILLS
    )
    return (
        f'<div class="sidebar-brand">{pills}</div>'
        f'<div class="sidebar-brand-subtitle">{html.escape(_BRAND_SUBTITLE)}</div>'
    )


def inject_theme() -> None:
    """Apply the shared theme CSS and the sidebar brand block. Call once, near the top of each page.

    Folding the sidebar brand into this single call (rather than a second
    function every page would have to remember to call separately)
    guarantees it renders identically on every page — every page already
    calls `inject_theme()` right after `st.set_page_config()`.
    """
    import streamlit as st

    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    st.sidebar.markdown(render_sidebar_brand(), unsafe_allow_html=True)
