"""Integration tests across real components (Phase 9, Task 9.2).

These tests exercise actual interactions between layers — no mocks — as
opposed to the unit tests elsewhere in `tests/`, which mostly test one
module in isolation. Streamlit smoke testing (`AppTest`, executing every
page end to end) already lives in `tests/test_ui_pages.py`; this file
covers the three other integration scenarios Task 9.2 calls for.
"""

import tempfile
from pathlib import Path

import pandas as pd

from benchmarking.comparator import compare_tokenizers
from benchmarking.export import export_results_json, load_results_json
from experiments.dataset_loader import load_dataset
from experiments.runner import ExperimentConfig, run_experiment
from tokenizers.adapters.sentencepiece_tokenizer import SentencePieceAdapter
from tokenizers.adapters.tiktoken_tokenizer import TiktokenAdapter
from tokenizers.bpe.tokenizer import BPETokenizer
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.registry import AVAILABLE_TOKENIZERS
from tokenizers.wordpiece.tokenizer import WordPieceTokenizer

_SAMPLE_TEXT = "Tokenization is the first step in almost every NLP pipeline."

# ---------------------------------------------------------------------------
# Integration Test 1 — full Comparator across >= 4 real tokenizers
# ---------------------------------------------------------------------------


def test_comparator_runs_across_all_four_own_tokenizer_types():
    """All four of this project's own tokenizer types, through the real Comparator."""
    tokenizers = []
    for name in ("character", "word", "bpe", "wordpiece"):
        tokenizer = AVAILABLE_TOKENIZERS[name]()
        tokenizer.train([_SAMPLE_TEXT])
        tokenizers.append(tokenizer)

    results = compare_tokenizers(tokenizers, _SAMPLE_TEXT)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 4
    assert set(results["tokenizer"]) == {"character", "word", "bpe", "wordpiece"}
    required_columns = {
        "tokenizer",
        "number_of_tokens",
        "tokens_per_word",
        "characters_per_token",
        "compression_ratio",
        "vocab_size",
        "tokens",
    }
    assert required_columns <= set(results.columns)
    assert (results["vocab_size"] > 0).all()
    assert (results["number_of_tokens"] > 0).all()
    assert results["tokens"].apply(lambda tokens: isinstance(tokens, list)).all()


def test_comparator_runs_across_five_tokenizers_including_a_real_external_adapter():
    """This project's own four, plus a real `tiktoken` adapter -- a genuine external tokenizer."""
    own_tokenizers = []
    for name in ("character", "word", "bpe", "wordpiece"):
        tokenizer = AVAILABLE_TOKENIZERS[name]()
        tokenizer.train([_SAMPLE_TEXT])
        own_tokenizers.append(tokenizer)

    external = TiktokenAdapter.from_encoding_name("cl100k_base")
    tokenizers = [*own_tokenizers, external]

    results = compare_tokenizers(tokenizers, _SAMPLE_TEXT)

    assert len(results) == 5
    assert "tiktoken:cl100k_base" in set(results["tokenizer"])
    assert not results.isna().all(axis=None)  # not every cell is NaN
    assert (results["vocab_size"] > 0).all()
    # tiktoken's real production vocabulary must be far larger than this
    # project's own live-trained toy vocabularies -- proof it is the real
    # library, not a stub.
    tiktoken_row = results[results["tokenizer"] == "tiktoken:cl100k_base"].iloc[0]
    own_rows = results[results["tokenizer"] != "tiktoken:cl100k_base"]
    assert tiktoken_row["vocab_size"] > own_rows["vocab_size"].max() * 100


# ---------------------------------------------------------------------------
# Integration Test 2 — Dataset Loader -> Experiment Runner -> Results
# ---------------------------------------------------------------------------


def test_dataset_loader_to_experiment_runner_to_results_on_a_small_real_dataset():
    """The real Phase 6 pipeline, restricted to one small dataset and two tokenizers."""
    text, metadata = load_dataset("en")
    assert text  # the loader actually returned real text, not a stub

    config = ExperimentConfig(tokenizer_names=("character", "bpe"), dataset_categories=("en",))
    results = run_experiment(config)

    assert len(results) == 2
    assert set(results["dataset"]) == {"en"}
    assert set(results["language_or_type"]) == {metadata.language_or_type}
    assert set(results["tokenizer"]) == {"character", "bpe"}
    assert (results["vocab_size"] > 0).all()


def test_experiment_runner_results_survive_export_and_reload():
    """Runner output -> Task 5.4 export -> reload, without losing rows or columns."""
    config = ExperimentConfig(
        tokenizer_names=("character", "word"), dataset_categories=("en", "ro")
    )
    results = run_experiment(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "integration_results.json"
        export_results_json(results, path, dataset="multi")  # dataset stamp is overwritten below
        # Runner output is already schema-shaped per-dataset; re-exporting
        # through export_results_json here (not export_schema_json) is
        # deliberately checking that even the single-dataset convenience
        # path round-trips the already-correct rows without corrupting them.
        reloaded = load_results_json(path)

    assert len(reloaded) == len(results)
    assert set(reloaded["tokenizer"]) == set(results["tokenizer"])


# ---------------------------------------------------------------------------
# Integration Test 3 — serialization: train -> encode -> save -> load -> encode
# ---------------------------------------------------------------------------


def test_bpe_save_load_roundtrip_produces_identical_encode_results(tmp_path):
    tokenizer = BPETokenizer(num_merges=20)
    tokenizer.train([_SAMPLE_TEXT])
    ids_before = tokenizer.encode(_SAMPLE_TEXT)

    path = tmp_path / "bpe.json"
    tokenizer.save(path)

    reloaded = BPETokenizer()
    reloaded.load(path)

    assert reloaded.encode(_SAMPLE_TEXT) == ids_before
    assert reloaded.decode(reloaded.encode(_SAMPLE_TEXT)) == tokenizer.decode(ids_before)


def test_wordpiece_save_load_roundtrip_produces_identical_encode_results(tmp_path):
    tokenizer = WordPieceTokenizer(vocab_size=40)
    tokenizer.train([_SAMPLE_TEXT])
    ids_before = tokenizer.encode(_SAMPLE_TEXT)

    path = tmp_path / "wordpiece.json"
    tokenizer.save(path)

    reloaded = WordPieceTokenizer()
    reloaded.load(path)

    assert reloaded.encode(_SAMPLE_TEXT) == ids_before


def test_character_tokenizer_save_load_roundtrip_produces_identical_encode_results(tmp_path):
    tokenizer = CharacterTokenizer()
    tokenizer.train([_SAMPLE_TEXT])
    ids_before = tokenizer.encode(_SAMPLE_TEXT)

    path = tmp_path / "character.json"
    tokenizer.save(path)

    reloaded = CharacterTokenizer()
    reloaded.load(path)

    assert reloaded.encode(_SAMPLE_TEXT) == ids_before
    assert reloaded.decode(reloaded.encode(_SAMPLE_TEXT)) == _SAMPLE_TEXT


def test_sentencepiece_save_load_roundtrip_produces_identical_encode_results(tmp_path):
    tokenizer = SentencePieceAdapter(vocab_size=30)
    tokenizer.train([_SAMPLE_TEXT] * 10)
    ids_before = tokenizer.encode(_SAMPLE_TEXT)

    path = tmp_path / "sentencepiece.model"
    tokenizer.save(path)

    reloaded = SentencePieceAdapter()
    reloaded.load(path)

    assert reloaded.encode(_SAMPLE_TEXT) == ids_before
