# LLM-Tokenization-Vocabulary-Analysis-Platform

[![CI](https://github.com/alex-man123/LLM-Tokenization-Vocabulary-Analysis-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/alex-man123/LLM-Tokenization-Vocabulary-Analysis-Platform/actions/workflows/ci.yml)

An educational platform implementing LLM tokenization algorithms (character-level,
word-level, BPE, WordPiece) from scratch — including vocabulary training, not just
encode/decode — with a benchmarking dashboard to compare the custom implementations
against production tokenizers (Hugging Face `tokenizers`, `tiktoken`, `sentencepiece`).

## Project structure

```text
src/            # core tokenizers, vocabulary manager, external adapters, benchmarking
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
character-level, word-level, and BPE tokenizers, benchmarking metrics + a
tokenizer Comparator, and a Streamlit dashboard (Tokenize and Compare pages;
Vocabulary/Benchmark/Experiments are placeholders). The BPE implementation is
the classical **character-level BPE with a `</w>` word-boundary marker**
(Sennrich et al., 2015) — not the byte-level BPE used by GPT-style
models/`tiktoken`; see [docs/architecture.md](docs/architecture.md) for why
that distinction matters. WordPiece, external tokenizer adapters, and
encode/decode timing (Task 5.2) are not implemented yet.
