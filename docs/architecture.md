# Architecture (draft)

This document tracks the project's data conventions and schemas as they are
established. It grows with each phase; so far it covers the raw text
convention, the experiment result schema, the vocabulary manager, the
character/word/BPE/WordPiece tokenizers and external adapters built on top
of it, benchmarking (metrics, timing, export), and the Phase 6 dataset
loader/Experiment Runner/aggregation pipeline built on top of all of that.

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

All 9 categories above exist (Task 6.1), each ~2-5 KB of original text
written for this project (not copied verbatim from an external source, to
avoid licensing concerns) — `ro.txt` additionally includes a few
deliberate instances of the legacy Turkish-cedilla ş/ţ mixed in among the
correct comma-below ș/ț, to give the loader's normalization step (below) a
real, representative case to fix, not just a synthetic unit-test string.

### Dataset loader & preprocessing (`src/experiments/dataset_loader.py`, Task 6.2)

`load_dataset(name) -> (text, DatasetMetadata)` / `load_all_datasets() ->
dict[str, (text, DatasetMetadata)]` are the single way any code in this
project reads `data/raw/`. `DatasetMetadata` carries `name` (the category/
file stem, e.g. `"ro"`), `language_or_type` (e.g. `"romanian"`,
`"python_code"` — one field covers both concepts, since this project's
categories are language *or* text-type, not both at once for any given
category), `source`, `length_chars`, and `length_bytes`.

**Normalization — `unicodedata.normalize("NFC", text)` — is mandatory,
first, and applied uniformly to every category**, with no per-category
exceptions and, critically, **no lowercasing**: Python identifiers and URL
paths are case-sensitive, so a global lowercase step would destroy
information a tokenizer should see. NFC correctly unifies composed vs.
decomposed Unicode forms (e.g. Japanese "が" as one precomposed codepoint
vs. base + combining voiced-sound-mark; Romanian "â" as one codepoint vs.
"a" + combining circumflex).

> **Correction to a common claim about Romanian ș/ț:** plain NFC does
> **not** unify the legacy Turkish-cedilla forms (ş U+015F, ţ U+0163) with
> the correct Romanian comma-below forms (ș U+0219, ț U+021B) — verified:
> their canonical decompositions are `s + COMBINING CEDILLA` vs.
> `s + COMBINING COMMA BELOW`, two different base+combiner pairs, so NFC
> has nothing to fold together. `dataset_loader.normalize_text` fixes this
> anyway, as an explicit, separate, four-character translation table
> applied right after NFC — narrow enough to apply uniformly to every
> category with no risk to non-Romanian text. See the module's docstring
> for the full explanation; do not repeat the claim that NFC alone solves
> this.

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
- `encoding_time`/`decoding_time` — reserved fields; `compute_metrics`
  itself always leaves them `None`. Task 5.2 (`src/benchmarking/timer.py`,
  below) implements the actual timing separately, so a caller that wants
  both metrics and timing for the same run merges the two rather than
  `compute_metrics` calling the timer itself — timing every `encode`/
  `decode` call would slow down the common case (a metrics-only
  comparison) for no benefit.

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

## Benchmarking: timing (`src/benchmarking/timer.py`, Task 5.2)

`measure_tokenizer_timing(tokenizer, text, n_iterations=10) -> TokenizerTiming`
measures `encode`/`decode` reliably: one unmeasured warm-up call per
operation, then `n_iterations` measured repetitions, summarized as
`TimingResult(mean_ms, median_ms, samples_ms)` for each of `.encode`/
`.decode`. Always in **milliseconds**, matching the `encode_time_ms`/
`decode_time_ms` fields of the experiment result schema above — the one
place these numbers are computed, never duplicated in the Comparator,
export, or UI.

- Uses `time.perf_counter()` (monotonic, high-resolution), never
  `time.time()`.
- `decode` is timed against IDs from a single `encode` call made once,
  up front — its timing loop never re-encodes, so it never measures
  encode+decode as one operation.
- `n_iterations < 1` raises `ValueError` rather than producing a
  meaningless empty/NaN result.
- Works with any `Tokenizer`, including the external adapters below —
  there is no separate timer for adapters.

## Benchmarking: result export (`src/benchmarking/export.py`, Task 5.4)

`export_results_csv`/`export_results_json` persist a Comparator-style
DataFrame to `data/results/` (or any path; parent directories are created
automatically), reshaped to match the experiment result schema above via
`to_experiment_schema`: `number_of_tokens` -> `num_tokens`,
`encoding_time`/`decoding_time` -> `encode_time_ms`/`decode_time_ms`, plus
the schema's `dataset` (required) and `timestamp` (defaults to now, UTC)
fields, which the Comparator itself has no notion of. The Comparator's
`tokens` column — not part of the documented schema, but explicitly not to
be dropped — is preserved: a native list per row in JSON, a JSON-encoded
string per cell in CSV (CSV has no list type). `overwrite=True` by default
(matching every other `save`/serialization path in this project, e.g.
`vocabulary.serialization.save_tokenizer_state`); `overwrite=False` raises
`FileExistsError` instead of silently replacing an existing file.
`load_results_json` reads a file `export_results_json` wrote, back into a
DataFrame with the same columns.

Neither function retokenizes or recomputes anything — they only consume
results the Comparator already produced.

## External tokenizer adapters (`src/tokenizers/adapters/`, Phase 7)

`HuggingFaceTokenizerAdapter` (Task 7.1) and `TiktokenAdapter` (Task 7.2)
wrap a real, pretrained external tokenizer behind this project's
`Tokenizer` interface, so either can be passed to the Comparator exactly
like `BPETokenizer`/`WordPieceTokenizer` — **no Comparator changes were
needed**. Both translate an existing library's API rather than
reimplementing any tokenization algorithm, and both treat `train()` as a
deliberate no-op (per `Tokenizer.train`'s own docstring: an adapter
wrapping an already-pretrained external tokenizer loads state via `load`
instead).

- **`HuggingFaceTokenizerAdapter`**: wraps a `tokenizers.Tokenizer` (the
  pip-installed Hugging Face library). Load via `from_pretrained(id)` (a
  Hugging Face Hub identifier, e.g. `"bert-base-uncased"`), `from_file`/
  `load` (a local `tokenizer.json`), or by passing an already-built
  `tokenizers.Tokenizer` to the constructor. `tokenize`/`encode`/`decode`/
  `vocab_size` all delegate to the wrapped tokenizer's own API — IDs are
  the library's real IDs, never re-numbered. See `docs/limitations.md` for
  why this module has to work around a name collision between this
  project's own `tokenizers` package and the pip-installed one of the same
  name.
- **`TiktokenAdapter`**: wraps a `tiktoken.Encoding`. Load via
  `from_encoding_name` (e.g. `"cl100k_base"`), `for_model` (e.g.
  `"gpt-4"`), or by passing an already-built `tiktoken.Encoding` to the
  constructor. `decode` always uses `tiktoken`'s own (lossless) decode;
  `tokenize`'s per-token strings are a best-effort visualization that can
  show `�` for a token whose raw bytes are not independently valid UTF-8 —
  expected byte-level BPE behavior, not a bug. See "Character-level BPE
  vs. byte-level BPE" in `docs/limitations.md` for the full explanation of
  why `tiktoken` and this project's own `BPETokenizer` are not directly
  comparable algorithms operating on the same units.
- **`SentencePieceAdapter`** (Task 7.3): wraps `sentencepiece`, Google's
  Unigram/BPE tokenizer library. Unlike the two adapters above, **`train()`
  is not a no-op** — it genuinely trains a fresh model (Unigram by
  default) on whatever corpus is passed to it, entirely in memory via
  `SentencePieceTrainer.Train(sentence_iterator=..., model_writer=io.BytesIO())`
  (no temp files), so it can be trained on this project's own datasets for
  a fair, same-corpus comparison against `BPETokenizer`/`WordPieceTokenizer`.
  `tokenize()` returns SentencePiece's own pieces unmodified, including its
  `▁` (U+2581) explicit word-start marker. `save`/`load` use
  `serialized_model_proto()`, SentencePiece's own binary format — there is
  no JSON-based serialization to integrate with, the same reasoning as the
  Hugging Face adapter's native-format `save`/`load`. SentencePiece's
  trainer *raises* (does not silently cap) if `vocab_size` is too large for
  the given corpus — real, useful feedback this adapter does not hide. See
  `docs/unigram_notes.md` (Task 2.6) for why this project wraps Unigram via
  SentencePiece instead of implementing it from scratch.

## Token frequency analysis (`src/vocabulary/frequency_analysis.py`, Task 4.3)

Feeds the "Vocabulary" UI page (below): how often each token a trained
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

## Experiment Runner (`src/experiments/runner.py`, Task 6.3)

`ExperimentConfig(tokenizer_names, dataset_categories)` (defaults: every
`tokenizers.registry.AVAILABLE_TOKENIZERS` entry x every
`experiments.dataset_loader.DATASET_CATEGORIES`) and
`run_experiment(config) -> pandas.DataFrame` automate running the existing
pipeline over the whole Tokenizer x Dataset matrix, reusing every prior
task rather than reimplementing any of them:

```text
Dataset Loader (6.2) -> Tokenizer (trained live) -> Comparator (5.3) -> schema (5.4) -> combined table
```

Each dataset's `compare_tokenizers` call is stamped with *that* dataset's
name and `language_or_type` via `benchmarking.export.to_experiment_schema`
**separately**, before all datasets' rows are concatenated — calling
`export_results_csv`/`export_results_json` (which stamp one shared
`dataset` value on every row) on the combined multi-dataset table would
have overwritten each row's correct dataset. This is why
`benchmarking.export` also exposes lower-level `export_schema_csv`/
`export_schema_json` (write an already-schema-shaped DataFrame as-is,
Task 5.4): `run_and_export_experiment(config, csv_path=..., json_path=...)`
uses those to write the combined table without re-stamping it.

**Reproducibility**: every step (`load_dataset`, a tokenizer's own
`train`/`encode`, `compare_tokenizers`, `to_experiment_schema`) is
deterministic, so running the same `ExperimentConfig` twice produces
identical rows (aside from the `timestamp` field, which is wall-clock by
design) — no seeding is needed because nothing here is random.

## Result aggregation (`src/experiments/aggregation.py`, Task 6.4)

`aggregate_by_group_and_tokenizer(results, group_column="language_or_type")`
and `aggregate_by_group(results, group_column="language_or_type")` group an
Experiment Runner-shaped DataFrame with plain `pandas.groupby(...).agg(...)`
(mean/median by default) — no manual aggregation loops. `group_column`
defaults to `"language_or_type"` but accepts e.g. `"dataset"` for
finer-grained grouping.

**Interpretation guardrail (load-bearing):** `describe_observations(results,
group_column=..., metric=...)` produces one sentence per `(group value,
tokenizer)` pair, always phrased as *"in this experiment's dataset,
tokenizer 'X' produced a mean \<metric\> of \<value\> for
\<group_column\>='\<value\>'"* — never as a universal claim about a
language or text type ("Japanese is harder to tokenize"). A few KB of text
per category cannot support that broader claim; every consumer of this
module's output (`docs/experiment_results.md`, any future UI) must
preserve that scoped framing rather than shortening it away.

`scripts/run_experiments.py` is the single script that ties the whole
pipeline together end to end: it runs the full default matrix, writes
`data/results/experiment_results.{csv,json}` (Task 5.4), computes the
aggregations above, and regenerates `docs/experiment_results.md` from
those real, just-computed numbers — never from invented ones. Re-run it
after changing a tokenizer, a dataset, or the aggregation logic.

## Streamlit UI (`ui/`, Phase 8)

`ui/streamlit_app.py` is the entry point; `ui/pages/` holds the pages
Streamlit's classic `pages/` convention auto-discovers (numbered filenames
control sidebar order: `1_Tokenize.py`, `2_Compare.py`, `3_Vocabulary.py`,
`4_Benchmark.py`, `5_Experiments.py`, `6_How_LLMs_Use_Tokens.py`). The UI
is intentionally thin — pages
only call `tokenizers.registry`, a tokenizer's own `train`/`tokenize`/`encode`,
and `benchmarking.comparator.compare_tokenizers`; no tokenization, metrics,
or comparison logic lives in `ui/`.

- **`tokenizers/registry.py`**: `AVAILABLE_TOKENIZERS` / `create_tokenizer(name)`
  — the single list of tokenizers the UI offers, so it is not hardcoded in
  more than one page.
- **`ui/tokenizer_options.py`**: tokenizer-selection helpers shared by
  Compare and Benchmark (below), so neither page duplicates the "which
  tokenizers can a user pick, and how do I build one without letting a
  single failure discard every other result" logic. Three sources: this
  project's own (`AVAILABLE_TOKENIZERS`, trained live), a pretrained
  external one (Hugging Face/tiktoken, Task 7.1/7.2 — loaded once and
  cached via `@st.cache_resource`, including *failures*, for
  `EXTERNAL_TOKENIZER_RETRY_SECONDS` = 5 minutes, so an offline
  environment doesn't retry the network on every rerun), and a trainable
  external one (SentencePiece, Task 7.3 — trained live like this
  project's own tokenizers, at a smaller UI-specific `vocab_size` than the
  adapter's own default, since a live demo's input is often just a
  sentence or two).
- **Tokenize** (`1_Tokenize.py`, Task 8.2): text in, colored tokens + a
  token→ID table out. To work for *any* input without pretraining on a
  large corpus, the selected tokenizer is trained live on the text the
  user enters — this is a deliberate demo simplification (documented on
  the page itself), not a hidden default.
- **Compare** (`2_Compare.py`, Task 8.3): several tokenizers (multi-select)
  on one shared input text, displayed as a metrics table (via the
  Comparator) plus each tokenizer's token list. Always shows the
  fair-comparison disclaimer (referencing `docs/benchmarking_methodology.md`,
  Task 7.4) and `vocab_size`. External tokenizers (`ui/tokenizer_options.py`)
  are opt-in, not selected by default. Passed to `compare_tokenizers`
  exactly like any other `Tokenizer`, with no Comparator changes.
- **Vocabulary** (`3_Vocabulary.py`, Task 8.4): selects one of this
  project's own tokenizers, trains it live on an entered corpus (the same
  text doubles as the corpus `vocabulary.frequency_analysis` counts over),
  and shows vocabulary size, a top-N frequent-token bar chart, and a rare-
  token table — all computed by Task 4.3's functions, never recomputed
  here.
- **Benchmark** (`4_Benchmark.py`, Task 8.5): the same live/interactive
  idea as Compare, plus two things Compare does not show: encode/decode
  timing per tokenizer (`benchmarking.timer.measure_tokenizer_timing`,
  Task 5.2) and a generic tokenization cost estimator (Task 8.7) —
  `estimated_cost = (num_tokens / 1_000_000) * price_per_million`, with
  `price_per_million` a plain `st.number_input` the user sets themselves
  (never a hardcoded real provider price). Reuses `ui/tokenizer_options.py`,
  so tokenizer selection/construction is not duplicated between Compare
  and Benchmark.
- **Experiments** (`5_Experiments.py`, Task 8.5): the opposite of
  live — loads `data/results/experiment_results.json` (written by
  `scripts/run_experiments.py`, Task 6.3) via
  `benchmarking.export.load_results_json` and aggregates it with the
  existing Task 6.4 functions (`experiments.aggregation`); it never trains
  or runs a tokenizer itself. Shows compression-ratio-per-language/type
  and tokens-per-word-per-type bar charts, the aggregated table,
  `describe_observations`'s dataset-scoped sentences, and the raw rows. If
  the results file does not exist yet, it says so plainly (with the
  command to generate it) instead of inventing rows.
- **How LLMs Use Tokens** (`6_How_LLMs_Use_Tokens.py`, Task 8.6): a purely
  illustrative walk through `Text -> Tokens -> Token IDs -> Embedding
  lookup -> Vectors -> Model`. Only the first three steps use real
  components (`tokenizers.registry`); the "embeddings" are `random`-module
  vectors seeded by token ID (fixed per token ID, not real learned
  parameters), and the page never claims or implements a next-token
  prediction — everything past "Token IDs" is labeled illustrative.

Pages are tested with `streamlit.testing.v1.AppTest`
(`tests/test_ui_pages.py`), which actually executes each page script
(including simulated widget input), not just imports it.
