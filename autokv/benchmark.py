"""vLLM capacity parsing and benchmark matrix construction."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from autokv.config import Profile


@dataclass(frozen=True)
class Capacity:
    tokens: int
    model_length: int | None
    max_concurrency: float | None


@dataclass(frozen=True)
class BenchmarkCase:
    input_length: int
    output_length: int
    repeat: int


def parse_capacity_tokens(log: str) -> Capacity:
    capacity_match = re.search(
        r"GPU KV cache size:\s*([\d,]+)\s*tokens", log, flags=re.IGNORECASE
    )
    if capacity_match is None:
        raise ValueError("server log does not contain GPU KV cache size")
    tokens = int(capacity_match.group(1).replace(",", ""))
    concurrency_match = re.search(
        r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x",
        log,
        flags=re.IGNORECASE,
    )
    if concurrency_match is None:
        return Capacity(tokens=tokens, model_length=None, max_concurrency=None)
    return Capacity(
        tokens=tokens,
        model_length=int(concurrency_match.group(1).replace(",", "")),
        max_concurrency=float(concurrency_match.group(2)),
    )


def build_benchmark_matrix(profile: Profile) -> tuple[BenchmarkCase, ...]:
    return tuple(
        BenchmarkCase(input_length, output_length, repeat)
        for repeat, input_length, output_length in itertools.product(
            range(profile.benchmark.repeats),
            profile.benchmark.input_lengths,
            profile.benchmark.output_lengths,
        )
    )
