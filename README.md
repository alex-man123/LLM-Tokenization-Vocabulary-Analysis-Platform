# LLM-Tokenization-Vocabulary-Analysis-Platform

[![CI](https://github.com/alex-man123/LLM-Tokenization-Vocabulary-Analysis-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/alex-man123/LLM-Tokenization-Vocabulary-Analysis-Platform/actions/workflows/ci.yml)

An educational platform implementing LLM tokenization algorithms (character-level,
word-level, BPE, WordPiece) from scratch — including vocabulary training, not just
encode/decode — with a benchmarking dashboard to compare the custom implementations
against production tokenizers (Hugging Face `tokenizers`, `tiktoken`, `sentencepiece`).

## Project structure

```text
src/            # core tokenizers, vocabulary manager, external adapters, benchmarking
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

## Status

Project setup is in progress (Phase 0 — foundations). Tokenization algorithms and
the benchmarking dashboard are not implemented yet.
