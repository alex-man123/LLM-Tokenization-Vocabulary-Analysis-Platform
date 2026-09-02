# Known limitations & scope notes

## Character-level BPE vs. byte-level BPE (Task 7.2)

This project's own `BPETokenizer` (Phase 2, `src/tokenizers/bpe/`) operates
on **Unicode characters**: the initial alphabet is every distinct character
seen in the training corpus, plus an end-of-word marker, and merges combine
adjacent characters/character-groups. A character never seen during
training has no vocabulary entry and falls back to `<UNK>`.

`tiktoken` (Task 7.2, `src/tokenizers/adapters/tiktoken_tokenizer.py`)
operates on **UTF-8 bytes**: its initial alphabet is the 256 possible byte
values, and merges combine adjacent bytes/byte-groups. Because every
possible byte is already in the vocabulary from the start, byte-level BPE
never has an "unknown" character — it can always fall back to raw bytes.

**Consequence:** for ASCII text, a character and a UTF-8 byte are the same
thing, so the two approaches look similar. For non-ASCII text they are not:

```text
len("こんにちは")               == 5   # 5 Unicode characters
len("こんにちは".encode("utf-8")) == 15  # 15 UTF-8 bytes (3 per character)
```

A `tiktoken` token's raw bytes are therefore not guaranteed to be a
complete, independently-decodable UTF-8 character on their own — a
multi-byte character can be split across adjacent tokens. `TiktokenAdapter.
tokenize()` decodes each token's bytes independently for a human-readable
string (`errors="replace"`), so it can legitimately show a replacement
character (`�`) for one piece of a split character; `decode()` never does
this per-token decoding, and is lossless, because it concatenates every
token's raw bytes first and decodes the whole result as UTF-8 once at the
end (see `tests/test_tiktoken_adapter.py` for a worked example with an
emoji + skin-tone modifier).

**Do not describe these as the same algorithm with different results.**
Both use BPE as a *merge strategy*, but over different base units
(characters vs. bytes) — token counts, granularity, and `<UNK>` behavior
differ for that reason, not because one implementation is "better BPE"
than the other. When comparing them (e.g. via the Comparator, Task 5.3),
also read `vocab_size` alongside token count/compression ratio, for the
same fair-comparison reason `docs/architecture.md` already documents for
the Comparator in general — `tiktoken`'s `cl100k_base` has a ~100k-token
vocabulary, several orders of magnitude larger than anything this
project's own tokenizers are trained to on a small demo corpus.

## The `tokenizers` package name collision (Task 7.1)

This project's own core package is named `tokenizers`
(`src/tokenizers/`), and `src` is on `sys.path` for every run/test in this
repository (`pythonpath = ["src"]`, `pyproject.toml`). The pip-installed
Hugging Face library used by Task 7.1 is *also* named `tokenizers`. Since
`src` is on `sys.path`, a plain `import tokenizers` anywhere in this
project resolves to this project's own package, never the installed
library.

`src/tokenizers/adapters/huggingface_tokenizer.py` works around this
locally: it temporarily removes `src` from `sys.path` and clears cached
`tokenizers`/`tokenizers.*` entries from `sys.modules` just long enough to
import the real library, then restores everything exactly as it was. It
also re-exports the library's `models`/`trainers`/`pre_tokenizers`
submodules (as `hf_models`/`hf_trainers`/`hf_pre_tokenizers`) so other code
that needs them does not have to repeat the same workaround. Renaming this
project's own `tokenizers` package to avoid the collision entirely would
be a large, unrelated refactor and was intentionally not done here.
