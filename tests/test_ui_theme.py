"""Unit tests for `ui/theme.py` (Task 8.9/8.8): tokenizer color lookup and pill rendering.

`ui/` is not on `pythonpath` (only `src/` is, `pyproject.toml`), so this
test inserts it into `sys.path` itself, exactly like every page under
`ui/pages/` already does to import sibling modules such as
`tokenizer_options`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

from theme import (  # noqa: E402
    TOKENIZER_COLORS,
    TOKENIZER_FALLBACK_COLOR,
    color_map_for,
    get_tokenizer_color,
    render_token_pills,
    tokenizer_family,
)


def test_tokenizer_family_splits_on_first_colon():
    assert tokenizer_family("tiktoken:cl100k_base") == "tiktoken"
    assert tokenizer_family("huggingface:bert-base-uncased") == "huggingface"


def test_tokenizer_family_of_a_name_without_a_colon_is_itself():
    assert tokenizer_family("bpe") == "bpe"


def test_get_tokenizer_color_matches_the_central_dict_for_known_names():
    for name, color in TOKENIZER_COLORS.items():
        assert get_tokenizer_color(name) == color


def test_get_tokenizer_color_resolves_a_pretrained_variant_to_its_family_color():
    assert get_tokenizer_color("tiktoken:cl100k_base") == TOKENIZER_COLORS["tiktoken"]
    assert (
        get_tokenizer_color("huggingface:bert-base-uncased") == TOKENIZER_COLORS["huggingface"]
    )


def test_get_tokenizer_color_falls_back_for_an_unknown_tokenizer_without_raising():
    assert get_tokenizer_color("some_future_tokenizer") == TOKENIZER_FALLBACK_COLOR


def test_color_map_for_covers_every_requested_name_including_unknown_ones():
    names = ["bpe", "character", "totally_unknown"]
    color_map = color_map_for(names)

    assert color_map == {
        "bpe": TOKENIZER_COLORS["bpe"],
        "character": TOKENIZER_COLORS["character"],
        "totally_unknown": TOKENIZER_FALLBACK_COLOR,
    }


def test_render_token_pills_includes_every_token_and_the_tokenizers_color():
    html = render_token_pills(["low", "er"], "bpe")

    assert html.count('class="token-pill"') == 2
    assert ">low<" in html
    assert ">er<" in html
    assert TOKENIZER_COLORS["bpe"] in html


def test_render_token_pills_escapes_html_in_token_text():
    html = render_token_pills(["<script>"], "character")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_token_pills_of_empty_list_is_empty_string():
    assert render_token_pills([], "bpe") == ""
