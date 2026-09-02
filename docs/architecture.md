# Architecture (draft)

This document tracks the project's data conventions and schemas as they are
established. It grows with each phase; so far it covers the raw text
convention, the experiment result schema, the vocabulary manager, and the
character/word/BPE tokenizers built on top of it.

## Raw text data (`data/raw/*.txt`)

- Encoding: UTF-8, no BOM.
- One file per dataset/category (e.g. `en.txt`, `ro.txt`, `es.txt`, `ja.txt`,
  `code_python.txt`, `urls.txt`, `numbers.txt`, `emoji.txt`, `technical.txt`).
- File name: lowercase, `snake_case`, no spaces, `.txt` extension.
- Line endings: `\n` (LF). Files should not rely on a specific number of
  lines or trailing newline behaviour — tokenizers must treat the raw text
  as-is.
- Raw files are input data, not build output: nothing in the codebase may
  modify files under `data/raw/` automatically. Adding or replacing a
  dataset is a manual, reviewed change.

## Vocabulary Manager (`src/vocabulary/`)

`Vocabulary` ([`src/vocabulary/vocab.py`](../src/vocabulary/vocab.py)) is the
single, central token↔ID mapping used by every tokenizer in the project
(character, word, BPE, WordPiece, and future external adapters). No
tokenizer should keep its own `token -> id` / `id -> token` dictionaries —
they all register their tokens into a `Vocabulary` instance instead, so the
mapping logic is implemented and tested exactly once.

- `add_token(token) -> id`: registers a token and returns its ID; calling it
  again with a token that is already registered is a no-op that returns the
  existing ID (no duplicates, no ID reuse/reassignment).
- `get_id(token)` / `get_token(id)`: raise `KeyError` for a token/ID that was
  never registered — there is no silent fallback.
- `has_token(token)` / `has_id(id)` / `token in vocab`: existence checks.
- `vocab_size` (and `len(vocab)`): number of distinct registered tokens.
- IDs are assigned deterministically, in registration order, starting at 0.

### Special tokens (`src/vocabulary/special_tokens.py`)

`SpecialTokens` wraps a `Vocabulary` and registers the four special tokens
in a fixed, documented order, so their IDs are the same across every
tokenizer and every run:

| Order | Token   | ID |
|-------|---------|----|
| 1     | `<PAD>` | 0  |
| 2     | `<UNK>` | 1  |
| 3     | `<BOS>` | 2  |
| 4     | `<EOS>` | 3  |

API: `SpecialTokens().pad_id` / `.unk_id` / `.bos_id` / `.eos_id` for the
IDs, `.unk_token` for the literal `<UNK>` string, `.tokens` for the ordered
tuple of all four, and `.is_special(token)` / `.is_special_id(id)` to check
whether a token or ID is one of the special ones. Tokenizers must import the
`PAD`/`UNK`/`BOS`/`EOS` constants (or use `SpecialTokens`) instead of
hardcoding these strings or IDs themselves.

`WordPieceTokenizer` registers its tokens into `Vocabulary`/`SpecialTokens`
the same way as every other tokenizer — see the dedicated section below.

### Generalized tokenizer serialization (`src/vocabulary/serialization.py`, Task 4.4)

`save_tokenizer_state`/`load_tokenizer_state` (backed by `TokenizerState`)
are the single save/load mechanism used by every tokenizer in this repo
(`CharacterTokenizer`, `WordTokenizer`, `BPETokenizer`) — none of them
writes its own ad hoc JSON. The file format:

```json
{
  "version": "1.0",
  "tokenizer_type": "bpe",
  "vocab_size": 16,
  "trained_at": "2026-09-02T00:00:00+00:00",
  "vocabulary": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "l", "o", "w", "..."],
  "config": {"merges": [["l", "o"], ["lo", "w"], ["low", "</w>"], ["low", "e"]]}
}
```

- `vocabulary` is every registered token, ordered by ID (index == ID) — a
  `Vocabulary` can be rebuilt from it exactly (`Vocabulary(tokens=...)`
  reassigns the same IDs, since `add_token` assigns IDs in iteration order).
- `config` holds whatever is specific to that tokenizer type; only
  `BPETokenizer` currently uses it, for its ordered `merges` list.
  Character/word/WordPiece tokenizers save `config: {}` — WordPiece needs
  no merge-rule list, since its encode is greedy longest-match against the
  vocabulary (see the WordPiece section below).
- `tokenizer_type` must match the loading tokenizer's `name` — `load`
  raises `ValueError` rather than silently loading a mismatched vocabulary
  (e.g. a `"word"` file into a `CharacterTokenizer`).
- `version`/`trained_at` are metadata for traceability, not currently used
  to change loading behavior.

## Character, Word, BPE & WordPiece tokenizers (`src/tokenizers/`)

`CharacterTokenizer`, `WordTokenizer`, `BPETokenizer`, and
`WordPieceTokenizer` are the concrete implementations of the `Tokenizer`
contract (Task 0.2) built so far. All four use the shared
`Vocabulary`/`SpecialTokens` for every token<->ID mapping — none keeps a
mapping of its own. `tokenize()` returns string tokens; `encode()` returns
the IDs for those tokens (looked up through `Vocabulary`), with an unknown
token mapped to `<UNK>` instead of raising.

- **`CharacterTokenizer`**: `tokenize(text)` is `list(text)` — every
  character (including spaces/newlines, and any Unicode character) is its
  own token. `decode()` joins with `""`, so it reconstructs the original
  text exactly for any text made only of characters seen during `train`.
- **`WordTokenizer`**: `tokenize(text)` splits on the regex `\w+|[^\w\s]` —
  a run of word characters is one token, every other non-whitespace
  character (punctuation) is its own one-character token, and whitespace is
  a pure separator (never a token, and repeated whitespace collapses).
  Contractions are not special-cased (`"don't"` → `["don", "'", "t"]`).
  `decode()` joins tokens with a single space, so it does **not** perfectly
  reconstruct original spacing/punctuation adjacency (`"hello!"` round-trips
  to `"hello !"`) — a documented limitation, not a bug.
- **`BPETokenizer`**: see the dedicated section below.

### BPE tokenizer (`src/tokenizers/bpe/`)

> **Scope note:** this is **character-level BPE with a word-boundary
> marker** (`</w>`) — the classical, didactic algorithm (Sennrich et al.,
> 2015), not the byte-level BPE used by GPT-style models/`tiktoken`.
> Byte-level BPE operates on UTF-8 bytes (256 possible values), so it never
> encounters an "unknown" character; this implementation operates on
> Unicode characters and falls back to `<UNK>` for characters never seen
> during training. That is a deliberate, documented scope choice for an
> educational implementation, not an oversight.

**Representation** (`src/tokenizers/bpe/trainer.py`): the corpus is split
on whitespace into words; each word becomes a tuple of its characters plus
a trailing `END_OF_WORD` (`"</w>"`) marker, e.g. `"low"` →
`("l", "o", "w", "</w>")`, counted by frequency in a `word_freqs` dict.
Punctuation attached to a word (`"hello,"`) stays part of that word — the
corpus is only split on whitespace, not further tokenized.

**Pair counting**: `count_pairs(word_freqs)` counts every adjacent symbol
pair across all words, weighted by each word's frequency.

**Training loop** (`train_bpe`): repeatedly picks the best pair
(`select_best_pair`), merges it everywhere it occurs (`apply_merge`), and
records it in `merges`, until either `num_merges` merges have been learned
or no adjacent pair is left. Two determinism-critical choices:

- **Tie-breaking**: when multiple pairs share the top frequency (common on
  small corpora), the lexicographically smaller pair `(a, b)` wins. Without
  this rule, training would not be guaranteed reproducible across runs.
- **Complexity**: pair counts are recomputed from `word_freqs` (not from
  raw corpus text, and not per raw occurrence — frequencies are already
  folded) on every iteration, rather than incrementally updating only the
  counts touched by the last merge. This trades some performance for
  simplicity/correctness, which is the right tradeoff for this project's
  small, educational corpora — not the approach a large-scale
  implementation would take.

**Vocabulary integration**: `BPETokenizer.train` registers, into the same
`Vocabulary` used by every other tokenizer, the base alphabet first (in
first-seen order), then each merge's concatenated symbol in the order
merges were learned — so IDs stay deterministic and reproducible. The
`merges` list itself (needed to encode new text) is BPE-specific state kept
on the tokenizer, not inside `Vocabulary`.

**Encode**: `tokenize` re-splits each word into characters + `</w>`, then
greedily applies the learned merges — at each step, merging whichever
adjacent pair present in the word has the *lowest rank* (earliest learned),
until no learned pair applies. This reproduces, for any word (including
one never seen during training), the same segmentation training's own
merge sequence would have produced. Encode never retrains and never
raises: any resulting symbol absent from the vocabulary falls back to
`<UNK>`, the same mechanism `CharacterTokenizer`/`WordTokenizer` use — no
byte-level fallback.

**Decode**: tokens are concatenated directly, then every `</w>` is replaced
with a single space and the result is stripped — since `</w>` only ever
appears as the trailing marker of a word's symbols (merges only ever
combine adjacent symbols, so it can only end up at the end of whatever
token it was merged into), this exactly reconstructs whitespace-delimited
text for words seen during training. Like `WordTokenizer`, repeated
whitespace in the input collapses to a single space in the output.

**Serialization**: uses the generalized `TokenizerState` (Task 4.4, above);
its `config` holds `{"merges": [[a, b], ...]}`.

### WordPiece tokenizer (`src/tokenizers/wordpiece/`)

> **Disclaimer:** see `docs/wordpiece_explained.md` for the full writeup
> and disclaimer. In short: the merge score implemented here is a
> **simplified WordPiece training implementation inspired by the original
> WordPiece objective** (Schuster & Nakajima, 2012), not a reproduction of
> the training procedure BERT/Hugging Face's `tokenizers` library use.

**Representation** (`src/tokenizers/wordpiece/trainer.py`): the corpus is
split into words by reusing `WordTokenizer.tokenize` (Task 1.2) — WordPiece
does not re-implement word/punctuation splitting. Each word becomes a tuple
of symbols: its first character unprefixed, every other character prefixed
with `##` to mark it as a continuation, e.g. `"hello"` ->
`("h", "##e", "##l", "##l", "##o")`. This is the opposite convention from
BPE's trailing `</w>`: WordPiece marks the *continuation* of a word, BPE
marks its *end*.

**Merge score**: instead of BPE's raw pair frequency, each candidate pair
is scored `freq(a, b) / (freq(a) * freq(b))` (`score_pairs`) — a pair
scores highest when it co-occurs often *relative to* its components'
individual frequencies, not simply when it is the most frequent pair in
absolute terms. The same lexicographic tie-break as BPE
(`select_best_pair`) keeps training deterministic on small corpora.

**Training loop** (`train_wordpiece`): repeatedly merges the
highest-scoring pair until a target `vocab_size` is reached or no candidate
pair remains. Unlike BPE, **no ordered merge-rule list is recorded** — only
the final vocabulary is kept, because WordPiece encode does not replay
merges in order.

**Encode**: `tokenize`/`encode` use greedy longest-match-first
(`_greedy_longest_match` in `tokenizers/wordpiece/tokenizer.py`): for each
word, the longest available vocabulary entry is matched at each position
(continuation pieces tried with the `##` prefix), left to right. If any
position has no match, the *whole word* becomes a single `<UNK>` token — a
partial tokenization is never returned, the standard WordPiece/BERT
behaviour.

**Decode**: tokens are concatenated, stripping `##` from continuation
pieces so they reattach without a space, and distinct words are joined
with a single space — the inverse of the encode convention above.

**Serialization**: uses the generalized `TokenizerState` (Task 4.4, above)
with `config: {}` — no merge-rule list is needed (see "Training loop").

## Experiment result schema (JSON)

One JSON object per experiment run (one tokenizer × one dataset).

> **Note:** an earlier draft of this schema defined `compression_ratio` as
> `characters / token` — the same formula as `characters_per_token` below,
> under two different names. Task 5.1 (see "Benchmarking: metrics" further
> down) corrected this: the two are genuinely different metrics
> (characters vs. UTF-8 bytes), and this table now matches that correction.

| Field                   | Type   | Required | Unit / format        | Meaning                                                             |
|-------------------------|--------|----------|-----------------------|----------------------------------------------------------------------|
| `tokenizer`             | string | yes      | —                      | Value of the tokenizer's `name` property (e.g. `"bpe_custom_v1"`).   |
| `dataset`               | string | yes      | —                      | Name of the raw dataset used, without extension (e.g. `"en"`).      |
| `vocab_size`            | int    | yes      | count                  | Tokenizer's `vocab_size` at the time of the run.                    |
| `num_tokens`            | int    | yes      | count                  | Total number of tokens produced by `encode` on the dataset.         |
| `tokens_per_word`       | float  | no       | tokens / word          | `num_tokens / len(raw_text.split())`; `null` if the text has no words. |
| `characters_per_token`  | float  | no       | characters / token     | `len(raw_text) / num_tokens`; `null` if `num_tokens == 0`. Granularity: how many characters one token covers on average. |
| `compression_ratio`     | float  | no       | UTF-8 bytes / token    | `len(raw_text.encode("utf-8")) / num_tokens`; `null` if `num_tokens == 0`. **Distinct from `characters_per_token`** — reads the same for ASCII text, but diverges for non-ASCII (e.g. a Japanese character is ~3 UTF-8 bytes), which is the interesting result to report for multi-language experiments. |
| `encode_time_ms`        | float  | no       | milliseconds           | Wall-clock time to encode the dataset once (Task 5.2, not yet implemented). |
| `decode_time_ms`        | float  | no       | milliseconds           | Wall-clock time to decode the produced IDs back to text (Task 5.2, not yet implemented). |
| `timestamp`             | string | no       | ISO 8601 (UTC)          | When the experiment was run.                                        |

Example (see [`data/results/example_result.json`](../data/results/example_result.json),
a dummy file kept only to illustrate the schema):

```json
{
  "tokenizer": "bpe_custom_v1",
  "dataset": "en",
  "vocab_size": 500,
  "num_tokens": 128,
  "tokens_per_word": 1.28,
  "characters_per_token": 3.4,
  "compression_ratio": 3.42,
  "encode_time_ms": 1.23,
  "decode_time_ms": 0.87,
  "timestamp": "2026-09-02T00:00:00Z"
}
```

## Experiment result schema (CSV)

Same fields as the JSON schema, flattened to columns, for consumption in
Pandas/Streamlit. One row per experiment run.

```csv
tokenizer,dataset,vocab_size,num_tokens,tokens_per_word,characters_per_token,compression_ratio,encode_time_ms,decode_time_ms,timestamp
bpe_custom_v1,en,500,128,1.28,3.4,3.42,1.23,0.87,2026-09-02T00:00:00Z
```

Optional fields (`tokens_per_word`, `characters_per_token`, `compression_ratio`,
`encode_time_ms`, `decode_time_ms`, `timestamp`) may be empty cells when not
measured/defined, but the column must still be present so that results from
different runs can be concatenated into one table.

## Benchmarking: metrics (`src/benchmarking/metrics.py`, Task 5.1)

`compute_metrics(tokenizer, text) -> TokenizationMetrics` is the single
place these metrics are computed — never inside a tokenizer, the
Comparator, or the UI. It reads only a tokenizer's public API
(`encode`, `vocab_size`) plus the raw text:

- `number_of_tokens` — `len(tokenizer.encode(text))` (the official encoded
  sequence, not `tokenize(text)`).
- `tokens_per_word` — `number_of_tokens / len(text.split())`; `None` when
  the text has no words. The word count is a plain whitespace split,
  independent of any tokenizer, so this is comparable across tokenizers.
- `characters_per_token` — `len(text) / number_of_tokens`; `None` when
  `number_of_tokens == 0`.
- `compression_ratio` — `len(text.encode("utf-8")) / number_of_tokens`;
  `None` when `number_of_tokens == 0`. See the note above: intentionally
  not the same formula as `characters_per_token`.
- `vocab_size` — `tokenizer.vocab_size`, unrelated to the current text.
- `encoding_time`/`decoding_time` — reserved fields, always `None` until
  Task 5.2 (timing) is implemented.

## Benchmarking: Comparator (`src/benchmarking/comparator.py`, Task 5.3)

`compare_tokenizers(tokenizers, text) -> pandas.DataFrame` runs `text`
through every (already-trained) tokenizer in `tokenizers` and returns one
row of `TokenizationMetrics` per tokenizer, plus a `tokens` column
(`tokenizer.tokenize(text)`) so a caller can display per-tokenizer tokens
without retokenizing.

**Fair comparison:** `vocab_size` is always one of the returned columns,
deliberately not tucked away. A custom tokenizer trained on a small corpus
and a production tokenizer with a much larger vocabulary (e.g. GPT's
~100k tokens) are not directly comparable on token count or compression
ratio alone — a larger vocabulary tends to produce fewer, longer tokens.
Any consumer of this DataFrame (the Streamlit Compare page included) must
show `vocab_size` alongside the other metrics, not report them in
isolation.

## Token frequency analysis (`src/vocabulary/frequency_analysis.py`, Task 4.3)

Feeds the planned "Vocabulary" UI page: how often each token a trained
tokenizer actually produces occurs across a training corpus, which tokens
are rare, and which dominate. Mirrors `benchmarking.metrics`'s "pure
function + dataclass" pattern — no Streamlit dependency.

- `compute_token_frequencies(tokenizer, corpus) -> dict[str, int]` — counts
  tokens from `tokenizer.tokenize(text)` for each document, **not**
  substring occurrences of a token's text in the raw corpus (two different
  pieces of text can look alike but tokenize differently). No special
  tokens are filtered out; if `tokenize` produces one (e.g. `<UNK>`), it is
  counted like any other token.
- `top_n_frequent_tokens(frequencies, n) -> list[TokenFrequency]` — the `n`
  most frequent tokens, highest first, ties broken lexicographically by
  token for determinism. `n <= 0` returns `[]`.
- `rare_tokens(frequencies, threshold) -> list[TokenFrequency]` — every
  token with `frequency < threshold`, ordered by frequency then token.
  `threshold <= 0` returns `[]` (frequencies are always >= 1).
- `to_dataframe(entries) -> pandas.DataFrame` — a `token`/`frequency` table
  ready to display or export, for either of the two functions above.

**Zipfian distribution**: natural-language token frequencies are
approximately Zipfian — a handful of tokens (common words, frequent
subwords) account for most occurrences, while most of a vocabulary's
entries appear rarely, a long tail. This is why growing a vocabulary size
has diminishing returns: most of the added capacity goes to rarely-used
tokens, not to better covering the common case.

## Streamlit UI (`ui/`, Phase 8)

`ui/streamlit_app.py` is the entry point; `ui/pages/` holds the pages
Streamlit's classic `pages/` convention auto-discovers (numbered filenames
control sidebar order: `1_Tokenize.py`, `2_Compare.py`, `3_Vocabulary.py`,
`4_Benchmark.py`, `5_Experiments.py`). The UI is intentionally thin — pages
only call `tokenizers.registry`, a tokenizer's own `train`/`tokenize`/`encode`,
and `benchmarking.comparator.compare_tokenizers`; no tokenization, metrics,
or comparison logic lives in `ui/`.

- **`tokenizers/registry.py`**: `AVAILABLE_TOKENIZERS` / `create_tokenizer(name)`
  — the single list of tokenizers the UI offers, so it is not hardcoded in
  more than one page.
- **Tokenize** (`1_Tokenize.py`, Task 8.2): text in, colored tokens + a
  token→ID table out. To work for *any* input without pretraining on a
  large corpus, the selected tokenizer is trained live on the text the
  user enters — this is a deliberate demo simplification (documented on
  the page itself), not a hidden default.
- **Compare** (`2_Compare.py`, Task 8.3): the same idea, but for several
  tokenizers (multi-select) trained on one shared input text, displayed as
  a metrics table (via the Comparator) plus each tokenizer's token list.
  Always shows the fair-comparison disclaimer and `vocab_size`.
- **Vocabulary**, **Benchmark**, **Experiments**: placeholder pages for
  later phases.

Pages are tested with `streamlit.testing.v1.AppTest`
(`tests/test_ui_pages.py`), which actually executes each page script
(including simulated widget input), not just imports it.
