"""Reliable encode/decode timing for any `Tokenizer` (Task 5.2).

A single timing measurement is unreliable: OS scheduling, other processes,
cache state, CPU frequency scaling, and one-off interpreter warm-up costs
all add noise a single sample cannot average out. This module applies the
standard fix — one unmeasured warm-up call, then several measured
repetitions, summarized by mean and median — and is meant to be reused by
the Comparator, other benchmarking code, and result export, never
reimplemented per caller.

Uses `time.perf_counter()` (a monotonic, high-resolution clock intended for
benchmarking), never `time.time()` (wall-clock: can jump when the system
clock is adjusted, and has coarser resolution on some platforms). Every
reported duration is in **milliseconds**, matching the `encode_time_ms`/
`decode_time_ms` fields of the documented experiment result schema
(Task 0.3, see `docs/architecture.md`) — this module is the one place
these numbers are computed, so that unit stays consistent wherever they end
up (Comparator, export, UI).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from tokenizers.base import Tokenizer


@dataclass(frozen=True)
class TimingResult:
    """Aggregate timing (milliseconds) over the *measured* repetitions of one operation.

    The warm-up call is never included in `mean_ms`/`median_ms`/`samples_ms`
    — `samples_ms` has exactly `n_iterations` entries, one per measured
    repetition, in the order they ran.
    """

    mean_ms: float
    median_ms: float
    samples_ms: tuple[float, ...]


@dataclass(frozen=True)
class TokenizerTiming:
    """Separate `encode`/`decode` timing results for one tokenizer on one text."""

    encode: TimingResult
    decode: TimingResult


def _time_calls(fn: Callable[[], object], n_iterations: int) -> TimingResult:
    """Run `fn` once (unmeasured, warm-up) then `n_iterations` times, timed with `perf_counter`.

    Raises:
        ValueError: if `n_iterations < 1` — there is no meaningful mean/median
            over zero measured repetitions, so this is rejected up front
            rather than silently producing an empty or NaN result.
    """
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

    fn()  # warm-up: deliberately not timed, not counted in the result

    samples_ms: list[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - start) * 1000)

    return TimingResult(
        mean_ms=statistics.mean(samples_ms),
        median_ms=statistics.median(samples_ms),
        samples_ms=tuple(samples_ms),
    )


def measure_tokenizer_timing(
    tokenizer: Tokenizer, text: str, *, n_iterations: int = 10
) -> TokenizerTiming:
    """Measure `tokenizer.encode(text)` and the matching `decode` call, timed separately.

    Encode and decode are two different operations with two different
    costs and are never combined into a single measurement. `text` is
    encoded once, up front, to obtain the IDs `decode` is measured against
    for every repetition — decode's timing loop never re-encodes, so it
    never accidentally measures encode+decode together.

    Raises:
        ValueError: if `n_iterations < 1` (see `_time_calls`).
    """
    encode_timing = _time_calls(lambda: tokenizer.encode(text), n_iterations)

    ids = tokenizer.encode(text)
    decode_timing = _time_calls(lambda: tokenizer.decode(ids), n_iterations)

    return TokenizerTiming(encode=encode_timing, decode=decode_timing)
