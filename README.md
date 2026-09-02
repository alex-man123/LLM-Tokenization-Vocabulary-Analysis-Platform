# LLM-Tokenization-Vocabulary-Analysis-Platform

[![CI](https://github.com/alex-man123/LLM-Tokenization-Vocabulary-Analysis-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/alex-man123/LLM-Tokenization-Vocabulary-Analysis-Platform/actions/workflows/ci.yml)

An educational platform implementing LLM tokenization algorithms (character-level,
word-level, BPE, WordPiece) from scratch — including vocabulary training, not just
encode/decode — with a benchmarking dashboard to compare the custom implementations
against production tokenizers (Hugging Face `tokenizers`, `tiktoken`, `sentencepiece`).

## Project structure

```text
src/            # core tokenizers, vocabulary manager, external adapters, benchmarking, experiments
scripts/        # run_experiments.py: the single entry point for the full experiment matrix
ui/             # Streamlit dashboard (thin UI layer over src/)
tests/          # unit / integration / regression tests
data/raw/       # raw text datasets used as experiment input
data/results/   # experiment results (JSON/CSV), see docs/architecture.md
docs/           # architecture and methodology documentation
```

See [docs/architecture.md](docs/architecture.md) for the data schemas.

## Getting started

Requires Python 3.11+.

```bash
pip install -r requirements-dev.txt
```

## Running the tests

```bash
pytest
```

## Running the dashboard

```bash
streamlit run ui/streamlit_app.py
```

## Status

Implemented so far: project foundations, the abstract `Tokenizer` contract, the
central `Vocabulary`/special-tokens manager and generalized serialization, the
character-level, word-level, BPE, and WordPiece tokenizers, two external
adapters (Hugging Face `tokenizers`, `tiktoken`), benchmarking (metrics, an
encode/decode Timer, a tokenizer Comparator, CSV/JSON export), vocabulary
frequency analysis, a Streamlit dashboard (Tokenize, Compare, Vocabulary, and
"How LLMs Use Tokens" pages; Benchmark/Experiments are placeholders), and the
Phase 6 dataset pipeline: 9 raw-text categories in `data/raw/`, a loader with
mandatory Unicode NFC normalization, an Experiment Runner over the full
Tokenizer x Dataset matrix, and result aggregation
(`scripts/run_experiments.py`, `docs/experiment_results.md`). The BPE
implementation is the classical **character-level BPE with a `</w>`
word-boundary marker** (Sennrich et al., 2015) — not the byte-level BPE used
by GPT-style models/`tiktoken`; see
[docs/architecture.md](docs/architecture.md),
[docs/limitations.md](docs/limitations.md), and
[docs/benchmarking_methodology.md](docs/benchmarking_methodology.md) for why
that distinction (and fair-comparison methodology in general) matters.
