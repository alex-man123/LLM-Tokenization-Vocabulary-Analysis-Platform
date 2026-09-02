# Benchmarking methodology: comparing tokenizers correctly

This document explains how to compare this project's own tokenizers
(`CharacterTokenizer`, `WordTokenizer`, `BPETokenizer`, `WordPieceTokenizer`)
against production tokenizers (the Hugging Face `tokenizers` adapter,
`tiktoken`) **without drawing conclusions the data does not support**. It
does not introduce a new comparison mechanism — every comparison here goes
through the existing Comparator (`benchmarking.comparator.compare_tokenizers`,
Task 5.3), which the Compare UI page (Task 8.3) also uses directly.

## Two different kinds of comparison

### 1. Same vocabulary size — for comparing *algorithms*

Train this project's own tokenizer (e.g. `BPETokenizer`) and an external
one to approximately the **same vocabulary size**, ideally on the **same
training corpus**. Any difference in token count, compression ratio, or
segmentation then mostly reflects a difference in the *merge/scoring
algorithm* itself, not a difference in how much vocabulary each one was
given to work with.

This is the methodology to use when the question is "does WordPiece's
likelihood-inspired scoring segment this text differently than BPE's raw
frequency, at a comparable vocabulary size?" — an algorithmic question.

### 2. Same production model — for orders of magnitude, not verdicts

Compare this project's own small, locally-trained tokenizer directly
against a real production tokenizer (`tiktoken`'s `cl100k_base`, a
pretrained Hugging Face `tokenizers.Tokenizer` such as BERT's WordPiece)
as-is, with no attempt to match vocabulary size or training corpus.

This is useful for:

- demonstrations and teaching — showing what a production-scale tokenizer
  looks like next to an educational one;
- observing practical, real-world differences;
- getting a sense of the *order of magnitude* involved (a ~50-token toy
  vocabulary vs. `cl100k_base`'s ~100k tokens).

**It must never be used to conclude "algorithm X is better than algorithm
Y."** A production tokenizer's vocabulary size, training corpus, training
objective, and preprocessing are all different from a tokenizer trained
live on a short demo string — any metric difference is expected and
uninformative about the underlying algorithm's merits.

## Always report `vocab_size` next to compression ratio

**`vocab_size` must be reported alongside every `compression_ratio` /
`characters_per_token` / `tokens_per_word` value, never on its own.** A
larger vocabulary mechanically tends to produce fewer, longer tokens
(better-looking compression) regardless of how good the underlying merge
algorithm is. `benchmarking.comparator.compare_tokenizers` already returns
`vocab_size` as one of its columns for exactly this reason (see "Fair
comparison" in `docs/architecture.md`); the Compare UI page (Task 8.3)
surfaces it next to every other metric, and this document is the
methodology that requirement follows from.

## Confounding variables

Two tokenizers can produce different results on the same text for reasons
that have nothing to do with which one is the "better algorithm." Before
attributing a metric difference to the tokenization algorithm, rule out:

- **`vocab_size`** — see above.
- **Training corpus** — a tokenizer trained on English news text will
  segment Japanese or source code differently than one trained on a
  matching corpus, independent of its algorithm.
- **Language distribution of the training corpus** — a multilingual corpus
  vs. a monolingual one changes which subwords are common enough to merge.
- **Domain of the text being tokenized** — code, URLs, and prose stress
  a tokenizer's merges very differently.
- **Normalization** — whether Unicode normalization (see
  `docs/architecture.md`'s dataset loader section, Task 6.2) or
  lowercasing was applied before training/encoding.
- **Pre-tokenization rules** — this project's `WordTokenizer`-based
  splitting, whitespace-only splitting, and a production tokenizer's own
  pre-tokenizer (e.g. GPT-2's byte-level regex) are not the same rules.
- **Training objective** — this project's from-scratch trainers optimize
  simple, explicit criteria (raw pair frequency for BPE, a simplified
  frequency ratio for WordPiece — see `docs/wordpiece_explained.md`'s
  disclaimer); production tokenizers may optimize different or more
  elaborate objectives.
- **Byte-level vs. character/symbol-level tokenization** — this project's
  own `BPETokenizer`/`WordPieceTokenizer` operate on Unicode characters;
  `tiktoken` operates on UTF-8 bytes. These are different base units, not
  just different merges over the same units — see "Character-level BPE
  vs. byte-level BPE" in `docs/limitations.md` for the full explanation
  (including why `len(text) != len(text.encode("utf-8"))` for non-ASCII
  text matters here).

None of this means comparisons are meaningless — it means every reported
difference needs to be read together with which of these variables
differed, not treated as a clean, isolated measurement of "algorithm
quality."
