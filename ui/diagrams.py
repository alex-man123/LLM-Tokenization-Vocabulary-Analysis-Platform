"""Pipeline diagram for the "How LLMs Use Tokens" page (Task 8.13).

Replaces the old ASCII `st.code()` block with a static, color-coded SVG:
the same seven-stage pipeline (Text -> Tokenization -> Tokens -> Token IDs
-> Embedding lookup -> Vectors -> Model), but color-coded *by category*
rather than per individual step — the first four stages (this project
actually computes them) in `theme.PIPELINE_COLORS["pipeline_real"]`, the
last three (labeled illustrations only) in
`theme.PIPELINE_COLORS["pipeline_illustrative"]`. This is the same
real/illustrative split the page's existing text disclaimer already
states; the diagram makes it visible by color alone, without replacing
that disclaimer.
"""

from __future__ import annotations

import html

from theme import PIPELINE_COLORS

#: (label, category) for each of the 7 pipeline stages, in order. Category
#: is a suffix of a `PIPELINE_COLORS` key (`"pipeline_" + category`) — see
#: `render_pipeline_diagram`.
_STAGES: tuple[tuple[str, str], ...] = (
    ("Text", "real"),
    ("Tokenization", "real"),
    ("Tokens", "real"),
    ("Token IDs", "real"),
    ("Embedding lookup", "illustrative"),
    ("Vectors", "illustrative"),
    ("Model → prediction", "illustrative"),
)

_BOX_WIDTH = 300
_BOX_HEIGHT = 46
_ARROW_HEIGHT = 28
_ARROWHEAD_HEIGHT = 8
_TEXT_COLOR = "#0b0e14"
_ARROW_COLOR = "#6b7280"


def render_pipeline_diagram() -> str:
    """An SVG string for the full pipeline, boxes color-coded real vs. illustrative.

    Pure string building (no external SVG/plotting dependency) — rendered
    by the caller via `st.markdown(..., unsafe_allow_html=True)`. Every
    label is HTML-escaped, and the only colors used come from
    `theme.PIPELINE_COLORS`, never a hardcoded hex value of its own.
    """
    total_height = (
        len(_STAGES) * _BOX_HEIGHT + (len(_STAGES) - 1) * _ARROW_HEIGHT + 2 * 10
    )
    parts = [
        f'<svg viewBox="0 0 {_BOX_WIDTH + 40} {total_height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Text to tokens to token IDs are real; embedding lookup, '
        f'vectors, and model prediction are illustrative only.">'
    ]
    y = 10
    for index, (label, category) in enumerate(_STAGES):
        color = PIPELINE_COLORS[f"pipeline_{category}"]
        parts.append(
            f'<rect x="20" y="{y}" width="{_BOX_WIDTH}" height="{_BOX_HEIGHT}" '
            f'rx="10" fill="{color}" />'
        )
        parts.append(
            f'<text x="{20 + _BOX_WIDTH / 2}" y="{y + _BOX_HEIGHT / 2 + 5}" '
            f'text-anchor="middle" font-family="JetBrains Mono, Fira Code, monospace" '
            f'font-size="14" font-weight="600" fill="{_TEXT_COLOR}">'
            f"{html.escape(label)}</text>"
        )
        y += _BOX_HEIGHT
        if index < len(_STAGES) - 1:
            arrow_x = 20 + _BOX_WIDTH / 2
            shaft_end = y + _ARROW_HEIGHT - _ARROWHEAD_HEIGHT
            parts.append(
                f'<line x1="{arrow_x}" y1="{y}" x2="{arrow_x}" y2="{shaft_end}" '
                f'stroke="{_ARROW_COLOR}" stroke-width="2" />'
            )
            parts.append(
                f'<polygon points="{arrow_x - 6},{shaft_end} '
                f"{arrow_x + 6},{shaft_end} {arrow_x},{y + _ARROW_HEIGHT}\" "
                f'fill="{_ARROW_COLOR}" />'
            )
            y += _ARROW_HEIGHT
    parts.append("</svg>")
    return "".join(parts)


def render_pipeline_legend() -> str:
    """HTML for a minimal legend: one colored circle + short label per pipeline category."""
    items = (
        (PIPELINE_COLORS["pipeline_real"], "Real pipeline"),
        (PIPELINE_COLORS["pipeline_illustrative"], "Illustrative only"),
    )
    dots = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'margin-right:18px;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'border-radius:50%;background:{color};"></span>'
        f'<span style="font-size:0.85rem;color:var(--text-muted);">'
        f"{html.escape(label)}</span></span>"
        for color, label in items
    )
    return f'<div style="margin:8px 0 14px 0;">{dots}</div>'
