"""AutoKV-Skip v2.0 的任务评分、三层聚合与配对区间。"""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from typing import Any, Mapping, Sequence

from autokv.scoring import score_generation
from autokv.v2_config import V2QualityConfig, V2_TIERS


def _normalize_qa_answer(value: str) -> str:
    """LongBench 英文 QA F1 使用的标准 SQuAD 风格归一化。"""

    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    predicted_tokens = _normalize_qa_answer(prediction).split()
    truth_tokens = _normalize_qa_answer(ground_truth).split()
    common = Counter(predicted_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if not predicted_tokens or not truth_tokens:
        return float(predicted_tokens == truth_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, answers: Sequence[str]) -> float:
    if not answers:
        raise ValueError("QA 样本至少需要一个参考答案")
    return max(qa_f1_score(prediction, answer) for answer in answers)


def set_f1_score(output: str, expected: Sequence[str], pattern: str) -> float:
    expected_set = {item.upper() for item in expected}
    if not expected_set:
        raise ValueError("集合答案不能为空")
    try:
        predicted_set = {item.upper() for item in re.findall(pattern, output.upper())}
    except re.error as exc:
        raise ValueError(f"无效答案 token_pattern：{pattern}") from exc
    if not predicted_set:
        return 0.0
    overlap = len(predicted_set & expected_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_set)
    recall = overlap / len(expected_set)
    return 2 * precision * recall / (precision + recall)


def score_v2_output(output: str, sample: Mapping[str, Any]) -> float:
    answers = sample.get("expected_answers")
    if (
        not isinstance(answers, list)
        or not answers
        or any(not isinstance(answer, str) for answer in answers)
    ):
        raise ValueError("v2 样本 expected_answers 无效")
    mode = sample.get("answer_mode")
    if mode == "contains":
        return score_generation(output, answers[0]).exact_match
    if mode == "qa_f1":
        return best_qa_f1(output, answers)
    if mode == "set_f1":
        metadata = sample.get("metadata")
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("token_pattern"), str
        ):
            raise ValueError("集合任务缺少 token_pattern")
        return set_f1_score(output, answers, str(metadata["token_pattern"]))
    raise ValueError(f"未知 v2 answer_mode：{mode}")


def aggregate_v2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("不能聚合空结果")
    scores: dict[str, list[float]] = {tier: [] for tier in V2_TIERS}
    easy_passes = 0
    sample_ids: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        tier = row.get("tier")
        score = row.get("task_score")
        if not isinstance(sample_id, str) or sample_id in sample_ids:
            raise ValueError("结果含无效或重复 sample_id")
        if tier not in scores:
            raise ValueError(f"结果含未知 tier：{tier}")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(f"{sample_id} 的 task_score 无效")
        if row.get("error") is not None:
            raise ValueError(f"{sample_id} 含失败结果，不能聚合")
        sample_ids.add(sample_id)
        scores[str(tier)].append(float(score))
        if tier == "easy" and float(score) == 1.0:
            easy_passes += 1
    if any(not values for values in scores.values()):
        raise ValueError("结果必须同时包含 easy、hard、natural")
    tier_scores = {tier: sum(values) / len(values) for tier, values in scores.items()}
    return {
        "rows": len(rows),
        "tier_rows": {tier: len(values) for tier, values in scores.items()},
        "scores": tier_scores,
        "s_v2": sum(tier_scores.values()) / len(V2_TIERS),
        "easy_passes": easy_passes,
        "easy_total": len(scores["easy"]),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("百分位输入不能为空")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_gap_summary(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """按样本配对并在各 tier 内分层重采样，差值方向为 reference-candidate。"""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples 必须为正数")
    reference = {str(row.get("sample_id")): row for row in reference_rows}
    candidate = {str(row.get("sample_id")): row for row in candidate_rows}
    if len(reference) != len(reference_rows) or len(candidate) != len(candidate_rows):
        raise ValueError("配对结果含重复 sample_id")
    if set(reference) != set(candidate):
        raise ValueError("两个策略的样本集合不同，不能配对")
    tier_differences: dict[str, list[float]] = {tier: [] for tier in V2_TIERS}
    for sample_id in sorted(reference):
        left = reference[sample_id]
        right = candidate[sample_id]
        if left.get("tier") != right.get("tier"):
            raise ValueError(f"{sample_id} 的 tier 不一致")
        tier = str(left["tier"])
        tier_differences[tier].append(
            float(left["task_score"]) - float(right["task_score"])
        )
    if any(not values for values in tier_differences.values()):
        raise ValueError("配对结果缺少一个或多个 tier")

    observed = {
        tier: sum(values) / len(values) for tier, values in tier_differences.items()
    }
    observed["global"] = sum(observed[tier] for tier in V2_TIERS) / len(V2_TIERS)
    samples: dict[str, list[float]] = {**{tier: [] for tier in V2_TIERS}, "global": []}
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_samples):
        replicate: dict[str, float] = {}
        for tier, values in tier_differences.items():
            replicate[tier] = sum(rng.choice(values) for _ in values) / len(values)
            samples[tier].append(replicate[tier])
        samples["global"].append(
            sum(replicate[tier] for tier in V2_TIERS) / len(V2_TIERS)
        )
    return {
        key: {
            "gap": observed[key],
            "ci95_low": _percentile(values, 0.025),
            "ci95_high": _percentile(values, 0.975),
        }
        for key, values in samples.items()
    }


def quality_constraints(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    config: V2QualityConfig,
    *,
    endpoint: bool,
) -> dict[str, Any]:
    reference_scores = reference.get("scores")
    candidate_scores = candidate.get("scores")
    if not isinstance(reference_scores, Mapping) or not isinstance(
        candidate_scores, Mapping
    ):
        raise ValueError("聚合结果缺少 tier scores")
    numerical_tolerance = 1e-12
    checks = {
        "global": float(candidate["s_v2"])
        >= float(reference["s_v2"]) - config.epsilon_global - numerical_tolerance,
        "hard": float(candidate_scores["hard"])
        >= float(reference_scores["hard"]) - config.epsilon_tier - numerical_tolerance,
        "natural": float(candidate_scores["natural"])
        >= float(reference_scores["natural"])
        - config.epsilon_tier
        - numerical_tolerance,
        "easy": (
            int(candidate["easy_passes"]) == int(candidate["easy_total"])
            if endpoint
            else int(candidate["easy_passes"]) == int(reference["easy_passes"])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "epsilon_global": config.epsilon_global,
            "epsilon_tier": config.epsilon_tier,
            "easy_rule": "all_pass" if endpoint else "same_pass_count_as_p32",
        },
    }
