# BPE, explained

> **Scope note:** this project's `BPETokenizer` (`src/tokenizers/bpe/`)
> implements the classical **character-level BPE with a `</w>`
> word-boundary marker** (Sennrich et al., 2015) — not the **byte-level**
> BPE used by GPT-style models / `tiktoken`. Both are BPE (the same merge
> strategy — repeatedly combine the most frequent adjacent pair), applied
> to different base units. See "Character-level BPE vs. byte-level BPE" in
> [`docs/limitations.md`](limitations.md) for the full distinction, and the
> `TiktokenAdapter` (`src/tokenizers/adapters/tiktoken_tokenizer.py`) for
> the byte-level implementation this project wraps rather than
> reimplements.

## 1. What is BPE?

Byte Pair Encoding (BPE) is a subword tokenization algorithm that starts
from individual symbols (here, Unicode characters) and iteratively
**merges** the most frequent adjacent pair of symbols into a new, single
symbol. Repeating this a fixed number of times grows a vocabulary of
progressively longer subword units — common character sequences (like
`"ing"` or, as below, `"low"`) end up as single tokens, while rare or
unseen character sequences stay split into smaller pieces.

## 2. Starting vocabulary

Training starts from the corpus, not from a fixed alphabet. The corpus is
split on **whitespace only** into words — punctuation attached to a word
(`"hello,"`) stays part of that word, since the corpus is not further
tokenized before BPE runs. Each word becomes a tuple of its individual
characters plus a trailing end-of-word marker `</w>`:

```text
"low" -> ("l", "o", "w", "</w>")
```

`</w>` prevents a merge from ever crossing a word boundary and marks where
a token ends a word — the opposite convention from WordPiece's `##`, which
marks where a token *continues* one (see
[`docs/wordpiece_explained.md`](wordpiece_explained.md)). Identical words
are counted once, weighted by how many times they occur (`word_freqs`), so
a word appearing 1000 times in the corpus is not iterated 1000 times.

## 3. Pair counting

At each training step, every adjacent pair of symbols across every word is
counted, weighted by that word's frequency:

```text
count_pairs(word_freqs):
    for each (word, freq) in word_freqs:
        for each adjacent (left, right) in word:
            pair_counts[(left, right)] += freq
```

A word with a single remaining symbol contributes no pairs.

## 4. Most frequent pair

The pair with the highest total count is selected to merge next. On small
corpora, several pairs often tie for the highest count — the tie is broken
by taking the **lexicographically smallest pair**, so that training is
100% deterministic (the same corpus and `num_merges` always produce the
same merges, regardless of platform, Python version, or hash
randomization). This is not an implementation detail to gloss over: without
a defined tie-break, "the most frequent pair" is ambiguous whenever a tie
occurs.

## 5. Merge

Every non-overlapping, left-to-right occurrence of the winning pair
`(a, b)` is replaced by the single new symbol `a + b`, in every word:

```text
merge ("l", "o") in ("l", "o", "w", "</w>")  ->  ("lo", "w", "</w>")
```

The new symbol is added to the vocabulary, and the merge rule
`("l", "o")` is appended to an **ordered** list of learned merges — the
order matters, and is reused at encode time (step 8).

## 6. Training loop

```text
train_bpe(corpus, num_merges):
    word_freqs = split corpus on whitespace, count word frequencies
    base_symbols = every distinct character seen, in first-seen order
    merges = []
    repeat up to num_merges times:
        pair_counts = count_pairs(word_freqs)
        if pair_counts is empty:
            stop early                      # nothing left to merge
        best_pair = most frequent pair, ties broken lexicographically
        word_freqs = merge best_pair everywhere it occurs
        merges.append(best_pair)
    return merges, base_symbols
```

`num_merges` is a cap, not a guarantee: training stops early once no
adjacent pair remains anywhere in the corpus (e.g. every word has fully
collapsed into one symbol). Pair counts are recomputed from `word_freqs`
from scratch on every iteration rather than incrementally updated — a
deliberate simplicity-over-performance choice appropriate for this
project's small, educational corpora (see `train_bpe`'s docstring in
`src/tokenizers/bpe/trainer.py`), not the approach a large-scale
implementation would take.

## 7. Example: `low` / `lower` / `lowest`

This is a **real, verified run** of `BPETokenizer`, not an illustrative
approximation — it matches `tests/golden/bpe_low_family_golden.json`
exactly. Training corpus: `"low"` x5, `"lower"` x2, `"lowest"` x1, with
`num_merges=4`.

Starting word frequencies:

```text
("l","o","w","</w>")          freq 5   # low
("l","o","w","e","r","</w>")  freq 2   # lower
("l","o","w","e","s","t","</w>") freq 1  # lowest
```

**Step 1** — pair counts: `(l,o)`: 8, `(o,w)`: 8, `(w,</w>)`: 5,
`(w,e)`: 3, `(e,r)`: 2, `(r,</w>)`: 2, `(e,s)`: 1, `(s,t)`: 1,
`(t,</w>)`: 1. `(l,o)` and `(o,w)` tie at 8 — `(l,o)` wins the
lexicographic tie-break. **Merge 1: `(l, o) -> "lo"`.**

**Step 2** — now `(lo,w)`: 8 is the clear maximum. **Merge 2:
`(lo, w) -> "low"`.**

**Step 3** — now `(low,</w>)`: 5 is the clear maximum (the `"low"` word
has nothing else left to merge). **Merge 3: `(low, </w>) -> "low</w>"`.**

**Step 4** — now `(low,e)`: 3 (from `"lower"`/`"lowest"`, which still
have a `"low"` symbol followed by `"e"`) is the maximum. **Merge 4:
`(low, e) -> "lowe"`.**

`num_merges=4` is reached, so training stops. Learned merges, in order:
`(l,o)`, `(lo,w)`, `(low,</w>)`, `(low,e)`. Final vocabulary: the 8-symbol
base alphabet `{l, o, w, </w>, e, r, s, t}` (in first-seen order) plus
these 4 merged symbols, on top of the 4 special tokens
(`<PAD>`/`<UNK>`/`<BOS>`/`<EOS>`) every tokenizer in this project starts
with.

Re-tokenizing each training word with these merges:

```text
"low"    -> ["low</w>"]
"lower"  -> ["lowe", "r", "</w>"]
"lowest" -> ["lowe", "s", "t", "</w>"]
```

## 8. Encoding

Encoding never replays training — it applies the already-learned merges to
new text. `tokenize(text)` splits `text` on whitespace, turns each word
into characters + `</w>`, then repeatedly finds *whichever adjacent pair in
the word has the lowest rank* (i.e. was learned earliest) and merges it,
until no learned pair applies anywhere in the word:

```text
tokenize(word):
    symbols = [*word, "</w>"]
    loop:
        find the adjacent pair in symbols with the lowest merge rank
        if no such pair exists: return symbols
        merge that pair in symbols
```

Applying the lowest rank first (not just the first pair found, left to
right) reproduces exactly the segmentation training's own merge order
would have produced for that word — this is what lets a word **never seen
during training** still merge partially. Continuing the example above,
`"lowering"` (never in the training corpus) tokenizes to
`["lowe", "r", "i", "n", "g", "</w>"]`: `(l,o)`, `(lo,w)`, then `(low,e)`
all apply in rank order, but `i`, `n`, `g` were never part of any learned
merge, so they stay as single characters.

`encode(text)` looks each resulting token up in the vocabulary and returns
its ID; a token absent from the vocabulary (a character never seen during
training, e.g. `i`/`n`/`g` above) falls back to `<UNK>`'s ID rather than
raising — the same mechanism `CharacterTokenizer`/`WordTokenizer` use.
Encoding a completely unrelated word, e.g. `"newword"`, produces no merges
at all (none of its adjacent pairs were ever learned) and falls back to
`<UNK>` for every character not in the training alphabet (here, all of
`n`, `w`, `d` — only `e`, `w`, `o`, `r` happen to be in the trained
alphabet, but never adjacent in a learned pair starting from `"newword"`'s
own characters).

## 9. Decoding

Tokens are concatenated directly (no separator), then every `</w>` marker
is replaced with a single space and the result is stripped:

```text
"low</w>" + "lowe" + "r" + "</w>"  ->  "low</w>lower</w>"  ->  "low lower"
```

This exactly reconstructs whitespace-delimited text for words seen during
training, because `</w>` can only ever end up at the end of the token it
was merged into (merges only ever combine *adjacent* symbols, and `</w>`
only ever appears as a word's trailing symbol to begin with). Like
`WordTokenizer`, repeated whitespace in the original input collapses to a
single space on decode — decode does not reproduce exact original spacing.

## 10. Edge cases

- **Unknown characters**: a character never seen during training has no
  vocabulary entry; `encode` maps it to `<UNK>` rather than raising. There
  is no byte-level fallback (contrast with `tiktoken`, which can always
  fall back to raw bytes — see `docs/limitations.md`).
- **Empty input**: `tokenize("")` and `tokenize("   ")` (whitespace-only)
  both return `[]` — `text.split()` on whitespace-only text yields no
  words, so there is nothing to tokenize. `decode([])` returns `""`.
- **Untrained tokenizer**: a `BPETokenizer` that has never called `train`
  has an empty merge list and a vocabulary containing only the four
  special tokens — every character falls back to `<UNK>`, but `encode`
  still does not raise.
- **Punctuation**: punctuation attached to a word (`"hello,"`) is part of
  that word's symbol tuple like any other character — it is never split
  off before training, so `,` can itself take part in a learned merge
  (e.g. `("o", "</w>")` if the corpus is dominated by punctuation, or more
  commonly just falls back to `<UNK>` if the comma character was never
  part of the training alphabet).
- **Unicode**: any Unicode character can be a symbol — `tokenize` operates
  on `str` characters (Python's Unicode code points), not bytes, so e.g.
  Japanese or emoji characters are valid symbols if seen during training,
  but (like any other character) fall back to `<UNK>` if not.
- **Vocabulary limits**: `num_merges` is a cap on the constructor, not a
  target vocabulary size — the actual `vocab_size` after training is
  `4 (special tokens) + |base alphabet| + (merges actually learned)`,
  which can be less than `4 + |base alphabet| + num_merges` if training
  stops early (no pairs left to merge).

## BPE vs. WordPiece

Both algorithms in this project build a subword vocabulary by repeatedly
merging symbol pairs starting from individual characters, and both use the
same lexicographic tie-break for determinism — but they differ in what
they optimize for and how they encode new text. See
[`docs/wordpiece_explained.md`](wordpiece_explained.md) for the WordPiece
side in full, and [`docs/unigram_notes.md`](unigram_notes.md) for a third,
structurally different algorithm (Unigram) this project wraps via
SentencePiece instead of implementing from scratch.

| Aspect | BPE (`src/tokenizers/bpe/`) | WordPiece (`src/tokenizers/wordpiece/`) |
| --- | --- | --- |
| Word-boundary convention | Trailing `</w>` marks the **end** of a word | Leading `##` marks a **continuation** of a word |
| Training principle | Merge the pair with the highest raw frequency | Merge the pair with the highest `freq(a,b) / (freq(a) * freq(b))` score (this project's simplified, likelihood-inspired scorer — see that doc's disclaimer) |
| Tie-break | Lexicographically smallest pair | Lexicographically smallest pair (same rule) |
| What training records | An **ordered** list of merge rules (`merges`) | Only the **final vocabulary** — no ordered merge list |
| Encoding new text | Replays learned merges in rank order (lowest rank first) until none apply | Greedy longest-match-first against the vocabulary |
| Unmatched input | Falls back to `<UNK>` **per character** that has no vocabulary entry | Falls back to `<UNK>` for the **entire word** if any position has no match |
| Typical usage in this project | `BPETokenizer`, and (as byte-level BPE) the `TiktokenAdapter` | `WordPieceTokenizer` only |

Neither is "better" in the abstract — see
[`docs/benchmarking_methodology.md`](benchmarking_methodology.md) for how
to compare them (and production tokenizers) without drawing conclusions
the data doesn't support.
