"""Deterministic Needle-in-a-Haystack case generation."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Callable, Literal

from autokv.config import Profile


FILLER_SENTENCE = (
    "The archive contains routine notes about weather routes tools and ordinary events. "
)
INTRODUCTION = (
    "Read the entire archive. One line contains a verification code. "
    "Ignore all other identifiers and remember that exact code.\n\n"
)
QUESTION = (
    "\n\nWhat is the verification code? Output it exactly five times separated "
    "by |. Do not add an explanation."
)


@dataclass(frozen=True)
class NiahCase:
    sample_id: str
    target_tokens: int
    depth: float
    seed: int
    code: str

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if not 0.0 <= self.depth <= 1.0:
            raise ValueError("depth must be between zero and one")
        if not self.code.strip():
            raise ValueError("code cannot be empty")


@dataclass(frozen=True)
class MaterializedPrompt:
    prompt: str
    token_count: int
    needle_depth: float


def expected_answer(code: str) -> str:
    if not code.strip():
        raise ValueError("code cannot be empty")
    return "|".join([code.strip()] * 5)


def _code_for(phase: str, length: int, depth: float, seed: int) -> tuple[str, str]:
    raw = f"{phase}:{length}:{depth:.6f}:{seed}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    code = f"KV-{digest[:4].upper()}-{int(digest[4:8], 16) % 10000:04d}"
    sample_id = f"{phase}-{length}-{round(depth * 100):02d}-{seed}-{digest[:8]}"
    return sample_id, code


def make_cases(
    profile: Profile, phase: Literal["probe", "final"]
) -> tuple[NiahCase, ...]:
    if phase == "probe":
        lengths = profile.quality.probe_lengths
        depths = profile.quality.probe_depths
        seeds = profile.quality.probe_seeds
    elif phase == "final":
        lengths = profile.quality.final_lengths
        depths = profile.quality.final_depths
        seeds = profile.quality.final_seeds
    else:
        raise ValueError("phase must be probe or final")

    cases: list[NiahCase] = []
    for length, depth, seed in itertools.product(lengths, depths, seeds):
        sample_id, code = _code_for(phase, length, depth, seed)
        cases.append(NiahCase(sample_id, length, depth, seed, code))
    return tuple(cases)


def _render(case: NiahCase, repetitions: int) -> tuple[str, str]:
    before_count = round(repetitions * case.depth)
    before = FILLER_SENTENCE * before_count
    after = FILLER_SENTENCE * (repetitions - before_count)
    needle = f"VERIFICATION-CODE: {case.code}\n"
    prefix = INTRODUCTION + before
    return prefix + needle + after + QUESTION, prefix


def fit_prompt(
    case: NiahCase,
    count_tokens: Callable[[str], int],
) -> MaterializedPrompt:
    tolerance = max(1, round(case.target_tokens * 0.005))
    low = 0
    high = 1

    for _ in range(32):
        prompt, _ = _render(case, high)
        if count_tokens(prompt) >= case.target_tokens:
            break
        low = high + 1
        high *= 2
    else:
        raise ValueError("token counter did not reach the requested prompt length")

    best: tuple[int, int, str, str] | None = None
    while low <= high:
        middle = (low + high) // 2
        prompt, prefix = _render(case, middle)
        token_count = count_tokens(prompt)
        candidate = (abs(token_count - case.target_tokens), token_count, prompt, prefix)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        if token_count < case.target_tokens:
            low = middle + 1
        elif token_count > case.target_tokens:
            high = middle - 1
        else:
            break

    if best is None or best[0] > tolerance:
        observed = "none" if best is None else str(best[1])
        raise ValueError(
            f"could not fit prompt within 0.5%; target={case.target_tokens}, observed={observed}"
        )
    _, token_count, prompt, prefix = best
    prefix_tokens = count_tokens(prefix)
    return MaterializedPrompt(
        prompt=prompt,
        token_count=token_count,
        needle_depth=prefix_tokens / max(token_count, 1),
    )
