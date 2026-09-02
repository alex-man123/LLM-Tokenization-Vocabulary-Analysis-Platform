# WordPiece, explained

> **Disclaimer:** this is a **simplified WordPiece training implementation
> inspired by the original WordPiece objective** (Schuster & Nakajima,
> 2012), built for this project's educational purposes — not a
> reproduction of the training procedure BERT or Hugging Face's
> `tokenizers` library actually use. Real WordPiece training optimizes a
> language-model likelihood over segmentations, with additional validation
> steps this implementation does not have. Nothing here should be read as
> "this is exactly how BERT trains its tokenizer."

## Pre-tokenization & the `##` convention (Task 3.1)

Text is first split into words using the same word-level split as
`WordTokenizer` (Task 1.2) — `tokenizers/wordpiece/trainer.py` reuses
`WordTokenizer.tokenize` rather than re-implementing splitting, so both
tokenizers agree on what counts as a "word" (punctuation included, as its
own word-like unit).

Each word is then represented as a tuple of symbols: its first character
carries no prefix, every other character is prefixed with `##` to mark
that it only ever continues the symbol before it:

```text
"hello" -> ("h", "##e", "##l", "##l", "##o")
```

This is WordPiece's key visual difference from BPE: BPE marks the *end* of
a word (`</w>`), WordPiece marks the *continuation* of one.

## Merge score (Task 3.2)

Where BPE always merges the most frequent adjacent pair, WordPiece here
scores each candidate pair as:

```text
score(a, b) = freq(a, b) / (freq(a) * freq(b))
```

A pair scores highest when it co-occurs often *relative to* how often its
two symbols occur on their own — not simply when it is the most frequent
pair in absolute terms. A pair whose components barely ever occur apart
can outscore a far more frequent pair whose components are common
individually. `tests/test_wordpiece_trainer.py` includes a worked example
where the pair with the highest raw frequency is *not* the pair with the
highest score, to make this concrete.

Ties (same score for multiple candidate pairs, common on small corpora)
are broken the same way as BPE: the lexicographically smaller pair `(a, b)`
wins, so training is deterministic.

## Training loop (Task 3.3)

Starting from the base alphabet, the highest-scoring pair is merged
repeatedly until the vocabulary reaches a target `vocab_size` or no
candidate pair remains. Unlike BPE, **no ordered merge-rule list is kept**
— the final vocabulary alone is enough to encode new text, because encode
is greedy longest-match against the vocabulary, not sequential replay of
learned merge rules.

## Encode: greedy longest-match (Task 3.4)

To tokenize a word, WordPiece finds the longest prefix of the remaining
characters that exists in the vocabulary, adds it as a token, and repeats
on what's left (prefixing every continuation piece with `##`). If any
position has no match at all, the *whole word* becomes a single `<UNK>`
token — a partial tokenization is never returned, matching the standard
WordPiece/BERT convention.

```text
"unbelievable" -> ["un", "##believ", "##able"]   (if those pieces are in the vocabulary)
```

## Decode (Task 3.5)

Decoding concatenates tokens, stripping `##` from continuation pieces so
they reattach to the previous token without a space, and joins distinct
words with a single space:

```text
["un", "##believ", "##able"] -> "unbelievable"
["hello", "world"]           -> "hello world"
```

## `<UNK>` and other special tokens

`<PAD>`/`<UNK>`/`<BOS>`/`<EOS>` are registered through the shared
`SpecialTokens` (`src/vocabulary/special_tokens.py`), exactly like every
other tokenizer in this project — WordPiece does not define its own
special-token convention.
