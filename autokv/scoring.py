"""Quality metrics for deterministic long-context recall."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationScore:
    exact_match: float
    edit_distance: int
    normalized_output: str
    normalized_expected: str


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (
                left_character != right_character
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def score_generation(output: str, expected: str) -> GenerationScore:
    normalized_output = normalize_answer(output)
    normalized_expected = normalize_answer(expected)
    exact_match = float(normalized_expected in normalized_output)
    edit_distance = levenshtein_distance(normalized_output, normalized_expected)
    return GenerationScore(
        exact_match=exact_match,
        edit_distance=edit_distance,
        normalized_output=normalized_output,
        normalized_expected=normalized_expected,
    )


def answer_nll_from_echo(
    response: Mapping[str, Any], answer_start_offset: int
) -> float | None:
    try:
        choice = response["choices"][0]
        logprobs = choice["logprobs"]
        token_logprobs = logprobs["token_logprobs"]
        text_offsets = logprobs["text_offset"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(token_logprobs, list) or not isinstance(text_offsets, list):
        return None
    selected: list[float] = []
    for offset, logprob in zip(text_offsets, token_logprobs):
        if (
            isinstance(offset, int)
            and offset >= answer_start_offset
            and isinstance(logprob, (int, float))
            and math.isfinite(float(logprob))
        ):
            selected.append(float(logprob))
    if not selected:
        return None
    return -sum(selected) / len(selected)


def quality_score(
    exact_match: float,
    answer_nll: float | None,
    edit_distance: int,
    expected_length: int,
) -> float:
    if not 0.0 <= exact_match <= 1.0:
        raise ValueError("exact_match must be between zero and one")
    if edit_distance < 0 or expected_length < 0:
        raise ValueError("edit distance and expected length cannot be negative")
    if answer_nll is not None:
        probability = math.exp(-min(max(answer_nll, 0.0), 20.0))
        return 0.8 * exact_match + 0.2 * probability
    normalized = edit_distance / max(expected_length, 1)
    return exact_match - 0.001 * normalized
