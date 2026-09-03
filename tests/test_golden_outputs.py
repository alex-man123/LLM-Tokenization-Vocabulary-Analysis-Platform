"""Golden-output regression tests (Phase 9, Task 9.3).

Fixes this project's own tokenizers' *exact* known-correct behavior for a
small set of fixed inputs (including the classic `low`/`lower`/`lowest`
example) as fixture files under `tests/golden/*.json`, separate from the
tokenizer implementations themselves. A test here failing means training
or encode/decode behavior changed — deliberately (update the golden file
after confirming the new behavior is correct) or accidentally (a
regression to fix).

Manually verified once (not committed as a permanent change): temporarily
inverting `select_best_pair`'s comparison in `tokenizers/bpe/trainer.py`
(from `min(..., key=lambda item: (-item[1], item[0]))` — highest frequency
wins — to `(item[1], item[0])` — *lowest* frequency wins) made 4 of the 13
`test_bpe_low_family_golden_cases` parametrizations fail with clear
token/ID mismatches, exactly as expected; reverting the change restored
all 13 to a pass. (An earlier attempt, flipping the encode-time rank
comparison in `BPETokenizer._apply_learned_merges` instead, produced no
failures — this specific golden corpus never presents more than one
simultaneously-applicable ranked pair per step, so that particular
comparison never actually gets exercised by these cases. The
training-time break above is the one that was confirmed to work.) This
confirms these tests actually detect a behavior regression, not just
execute without error.
"""

import json
from pathlib import Path

import pytest

from tokenizers.bpe.tokenizer import BPETokenizer
from tokenizers.wordpiece.tokenizer import WordPieceTokenizer

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _load_golden(filename: str) -> dict:
    return json.loads((_GOLDEN_DIR / filename).read_text(encoding="utf-8"))


_BPE_GOLDEN = _load_golden("bpe_low_family_golden.json")
_WORDPIECE_GOLDEN = _load_golden("wordpiece_low_family_golden.json")


def _trained_bpe() -> BPETokenizer:
    tokenizer = BPETokenizer(num_merges=_BPE_GOLDEN["num_merges"])
    tokenizer.train(_BPE_GOLDEN["training_corpus"])
    return tokenizer


def _trained_wordpiece() -> WordPieceTokenizer:
    tokenizer = WordPieceTokenizer(vocab_size=_WORDPIECE_GOLDEN["vocab_size"])
    tokenizer.train(_WORDPIECE_GOLDEN["training_corpus"])
    return tokenizer


@pytest.mark.parametrize("case", _BPE_GOLDEN["cases"], ids=lambda c: c["input"])
def test_bpe_low_family_golden_cases(case):
    tokenizer = _trained_bpe()

    assert tokenizer.tokenize(case["input"]) == case["expected_tokens"]
    assert tokenizer.encode(case["input"]) == case["expected_ids"]


@pytest.mark.parametrize("case", _WORDPIECE_GOLDEN["cases"], ids=lambda c: c["input"])
def test_wordpiece_low_family_golden_cases(case):
    tokenizer = _trained_wordpiece()

    assert tokenizer.tokenize(case["input"]) == case["expected_tokens"]
    assert tokenizer.encode(case["input"]) == case["expected_ids"]


def test_bpe_golden_corpus_matches_the_fixture_exactly():
    # Guards against silently editing the fixture's training_corpus without
    # updating the expected outputs (or vice versa) -- both must move together.
    assert _BPE_GOLDEN["training_corpus"] == ["low"] * 5 + ["lower"] * 2 + ["lowest"]


def test_wordpiece_golden_corpus_matches_the_fixture_exactly():
    assert _WORDPIECE_GOLDEN["training_corpus"] == ["low"] * 5 + ["lower"] * 2 + ["lowest"]


def test_bpe_golden_training_is_still_deterministic():
    first = _trained_bpe()
    second = _trained_bpe()

    for case in _BPE_GOLDEN["cases"]:
        assert first.encode(case["input"]) == second.encode(case["input"])
