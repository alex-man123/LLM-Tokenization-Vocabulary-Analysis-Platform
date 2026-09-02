"""Dataset loader & preprocessing for `data/raw/` (Task 6.1/6.2).

Loads one of the 9 raw-text categories documented in `docs/architecture.md`
("Raw text data") and returns `(normalized_text, DatasetMetadata)`. This is
the single place category names, file paths, and normalization rules are
defined — the Experiment Runner (Task 6.3) and any other consumer import
from here rather than re-deriving them.

## Normalization: NFC is mandatory, first, and the only cleaning step

`unicodedata.normalize("NFC", text)` runs before anything else, on every
category uniformly. Two concrete reasons this matters (not just a
theoretical concern):

- **Japanese composed vs. decomposed forms**: a character like "が" can be
  encoded either as one precomposed codepoint or as its base character
  ("か") plus a separate combining voiced-sound-mark codepoint. These are
  canonically equivalent, and NFC unifies them into the single precomposed
  form — without it, the "same" character could tokenize as two different
  symbol sequences depending on which form a given source file happened to
  use.
- **Romanian composed accents**: a character like "â" can likewise arrive
  either precomposed or as "a" + a combining circumflex; NFC unifies these
  the same way.

> **Correction to a common claim about Romanian ș/ț:** Romanian text in
> the wild sometimes uses the Unicode *cedilla* forms (U+015F ş, U+0163 ţ —
> originally Turkish letters, historically substituted for Romanian due to
> old font/encoding limitations) instead of the correct *comma-below*
> forms (U+0219 ș, U+021B ț). These look nearly identical but are **not
> canonically equivalent** — `unicodedata.normalize("NFC", ...)` does
> **not** unify them (verified: their Unicode decompositions are
> `0073 0327` (s + COMBINING CEDILLA) vs. `0073 0326` (s + COMBINING COMMA
> BELOW) — two different base+combiner pairs, so NFC has nothing to fold
> together). Do not repeat the claim that plain NFC fixes this; it does
> not. This module fixes it anyway, as an explicit, separate,
> project-specific step applied right after NFC (`_ROMANIAN_CEDILLA_FIX`,
> below) — narrow enough (four characters) to apply uniformly to every
> category without any risk of altering non-Romanian text.

**No lowercasing, ever, for any category.** Case is meaningful information
this project must not destroy: Python identifiers are case-sensitive
(`myVariable` vs `MyVariable` vs `MY_VARIABLE` are different names), and
URL path/query components can be case-sensitive too. NFC normalization
(plus the narrow Romanian fix above) is the *only* text transformation
this loader performs — uniformly, on every category, with no per-category
exceptions.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# category (== filename stem in data/raw/, and the `dataset` identifier used
# throughout benchmarking/export) -> (language_or_type, source description)
_CATEGORY_INFO: dict[str, tuple[str, str]] = {
    "en": ("english", "original text written for this project"),
    "ro": ("romanian", "original text written for this project"),
    "es": ("spanish", "original text written for this project"),
    "ja": ("japanese", "original text written for this project"),
    "code_python": ("python_code", "original Python code written for this project"),
    "numbers": ("numbers", "original synthetic examples written for this project"),
    "urls": ("urls", "synthetic example.com/example.org URLs written for this project"),
    "emoji": ("emoji", "original text written for this project"),
    "technical": ("technical", "original text written for this project"),
}

DATASET_CATEGORIES: tuple[str, ...] = tuple(_CATEGORY_INFO)

# Legacy Turkish-cedilla forms sometimes found in Romanian text -> the
# correct Romanian comma-below forms. See the module docstring: NFC alone
# does not perform this mapping (the two are not canonically equivalent).
_ROMANIAN_CEDILLA_FIX = str.maketrans(
    {
        "ş": "ș",  # ş -> ș
        "Ş": "Ș",  # Ş -> Ș
        "ţ": "ț",  # ţ -> ț
        "Ţ": "Ț",  # Ţ -> Ț
    }
)


@dataclass(frozen=True)
class DatasetMetadata:
    """Descriptive metadata for one loaded dataset category."""

    name: str
    language_or_type: str
    source: str
    length_chars: int
    length_bytes: int


def normalize_text(text: str) -> str:
    """Apply this project's one mandatory cleaning step: NFC, then the Romanian cedilla fix.

    No lowercasing, no whitespace collapsing, no stripping — anything
    beyond Unicode normalization would risk destroying information a
    tokenizer should see (see the module docstring).
    """
    return unicodedata.normalize("NFC", text).translate(_ROMANIAN_CEDILLA_FIX)


def _dataset_path(name: str, raw_dir: Path) -> Path:
    if name not in _CATEGORY_INFO:
        raise ValueError(
            f"Unknown dataset category {name!r}; expected one of {DATASET_CATEGORIES}"
        )
    return raw_dir / f"{name}.txt"


def load_dataset(name: str, *, raw_dir: Path = DEFAULT_RAW_DIR) -> tuple[str, DatasetMetadata]:
    """Load and normalize one dataset category, returning `(text, metadata)`.

    Raises:
        ValueError: if `name` is not one of `DATASET_CATEGORIES`.
        FileNotFoundError: if the category is known but its file is
            missing from `raw_dir` — propagated as-is (not swallowed into
            an empty result), since a missing dataset is a setup problem,
            not a valid empty dataset.
    """
    path = _dataset_path(name, raw_dir)
    raw_text = path.read_text(encoding="utf-8")
    text = normalize_text(raw_text)
    language_or_type, source = _CATEGORY_INFO[name]
    metadata = DatasetMetadata(
        name=name,
        language_or_type=language_or_type,
        source=source,
        length_chars=len(text),
        length_bytes=len(text.encode("utf-8")),
    )
    return text, metadata


def load_all_datasets(
    *, raw_dir: Path = DEFAULT_RAW_DIR
) -> dict[str, tuple[str, DatasetMetadata]]:
    """Load every category in `DATASET_CATEGORIES`, keyed by category name."""
    return {name: load_dataset(name, raw_dir=raw_dir) for name in DATASET_CATEGORIES}
