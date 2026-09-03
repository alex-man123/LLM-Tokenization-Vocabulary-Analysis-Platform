# Unigram & SentencePiece (Task 2.6)

> **This project does not implement the Unigram algorithm from scratch.**
> This document explains the algorithm conceptually — enough to understand
> how it differs from the BPE/WordPiece training loops this project *does*
> implement (`src/tokenizers/bpe/trainer.py`,
> `src/tokenizers/wordpiece/trainer.py`) — and why. Real Unigram
> segmentation is still available in this project through the
> `SentencePieceAdapter` (Task 7.3, `src/tokenizers/adapters/sentencepiece_tokenizer.py`),
> which wraps Google's `sentencepiece` library rather than reimplementing it.

## 1. What Unigram is

Unigram (Kudo, 2018) is a subword tokenization algorithm built on a
**probabilistic language model** over subword units, rather than on a
sequence of merge rules. Every piece in the vocabulary has a probability;
the probability of a segmentation (a way of splitting a word into pieces)
is the product of the probabilities of its pieces (hence "unigram" — each
piece's probability is treated as independent of its neighbors). Training
picks the vocabulary that maximizes the likelihood of the training corpus
under this model.

## 2. Difference from BPE (and this project's WordPiece)

- **BPE** (`bpe/trainer.py`) is **merge-based / additive**: start from
  individual characters and repeatedly *add* a merge rule for whichever
  adjacent pair is most valuable (highest frequency for BPE, highest
  frequency-ratio for this project's simplified WordPiece score). The
  vocabulary only ever grows, one merge at a time.
- **Unigram** is **pruning-based / subtractive**: start from a large
  candidate vocabulary (e.g. every substring that occurs often enough) and
  repeatedly *remove* the pieces that contribute least to the corpus's
  likelihood, until the target vocabulary size is reached.

```text
BPE / WordPiece:  {chars} --add merge--> --add merge--> ... --> final vocab
Unigram:          {large candidate vocab} --prune--> --prune--> ... --> final vocab
```

## 3. EM (Expectation-Maximization), conceptually

Unigram training alternates two steps, without needing heavy mathematics
to understand *why*:

- **Expectation**: given the current piece probabilities, figure out how
  each training-corpus word is most likely segmented (and how much
  "credit" each candidate piece gets across all the likely segmentations).
- **Maximization**: given those credit assignments, re-estimate each
  piece's probability as its (weighted) frequency of use.

Repeating this a few times lets piece probabilities and segmentations
converge on each other — a piece that turns out to be rarely useful in the
best segmentations ends up with a low probability, marking it as a good
pruning candidate.

## 4. Segmentation: multiple candidates compete

For a single word, there is usually more than one way to split it into
known pieces. Unigram scores *every* candidate segmentation by the product
of its pieces' probabilities and picks the highest-scoring one (in
practice, via a Viterbi-style search, not by literally enumerating every
possibility). This is a genuinely different mechanism from BPE's
segmentation: BPE has one deterministic answer (replay the learned merges
in order); Unigram is picking a winner among competing candidates every
time it segments a word.

## 5. Pruning

At each pruning round, Unigram estimates how much the training corpus's
overall likelihood would *drop* if a given piece were removed (its pieces
can usually be re-spelled with shorter, already-present pieces, at some
likelihood cost). Pieces whose removal costs the least likelihood are
pruned first. This directly ties vocabulary size to a likelihood
objective, rather than to a frequency count or a frequency ratio — the
loss function BPE/WordPiece's simplified scores only approximate.

## 6. Where SentencePiece fits in

**SentencePiece is not Unigram** — it is a tokenizer *library/framework*
that can train several different model types, Unigram (its default) and a
BPE variant included. What SentencePiece additionally standardizes,
independent of which model type is used, is treating the input as a raw
Unicode stream and **encoding spaces explicitly** as a literal `▁`
(U+2581) prefix on the piece that starts a new word — e.g. `"▁lower"` —
instead of leaving whitespace as an implicit separator the way this
project's own tokenizers do (see `SentencePieceAdapter`'s docstring).
This project's adapter uses SentencePiece's Unigram model type by default,
specifically to demonstrate real Unigram segmentation.

## 7. Why not implemented from scratch here

- **Complexity**: a correct Unigram trainer needs a suffix-array-based
  substring extraction step, a Viterbi/forward-backward segmentation
  search, and several rounds of EM re-estimation and pruning — substantially
  more machinery than the from-scratch BPE/WordPiece trainers in this
  project, whose entire value is in being simple enough to read
  start-to-finish (see their own docstrings/disclaimers).
- **Project scope**: this project's explicit from-scratch focus (Phase 2/3)
  is BPE and a WordPiece-inspired scorer, deliberately chosen as the two
  algorithms simple enough to implement transparently while still covering
  merge-based subword tokenization end to end.
- **A real implementation already exists to wrap**: `sentencepiece` is a
  mature, widely-used, correctly-implemented library — Task 7.3's adapter
  gives this project genuine Unigram segmentation behavior (trained on its
  own corpus, for a fair comparison) without duplicating that
  implementation effort or risking a subtly-wrong reimplementation.

**This project does not claim to implement Unigram.** Anywhere Unigram
segmentation is shown (e.g. via `SentencePieceAdapter`), it comes from the
real `sentencepiece` library, not from project code.

## A real comparison: SentencePiece (Unigram) vs. this project's BPE

Both trained on `data/raw/en.txt` (this project's own English dataset, Task
6.1) to a comparable vocabulary size (SentencePiece: 200 pieces; BPE: 204
tokens including the base alphabet), then run on the same sentence:

```text
Input: "tokenization is unbelievably useful for understanding"

SentencePiece (Unigram, vocab=200):
['▁t', 'okenization', '▁is', '▁', 'un', 'believ', 'a', 'b', 'ly',
 '▁', 'u', 's', 'e', 'f', 'u', 'l', '▁', 'f', 'or', '▁under', 'st', 'an', 'ding']

BPE (character-level, vocab=204):
['tokeniz', 'ati', 'on</w>', 'is</w>', 'un', 'b', 'el', 'i', 'ev', 'ab',
 'ly</w>', 'us', 'e', 'f', 'u', 'l</w>', 'for', '</w>', 'un', 'd', 'er',
 'st', 'an', 'd', 'ing</w>']
```

Both segmentations are plausible and neither is "correct" in an absolute
sense — this is exactly the point of `docs/benchmarking_methodology.md`'s
guidance: two different algorithms (pruning-based Unigram vs. additive
BPE), trained on the same corpus at a comparable vocabulary size, still
produce different, individually-reasonable segmentations. This is a
qualitative illustration of the algorithms' difference on one sentence,
not a benchmark claim about which produces "better" tokens in general.
