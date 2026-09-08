"""四任务评分、约束排序和按来源组配对重采样。"""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from statistics import mean

from autokv.v2_metrics import best_qa_f1
from autokv.v3_config import TASKS

KEY = r"\bK[0-9A-F]{8}\b"
VALUE = r"\bV[0-9A-F]{8}\b"


def set_f1(predicted, expected):
    predicted, expected = set(predicted), set(expected)
    return 2 * len(predicted & expected) / (len(predicted) + len(expected)) if expected else 0.0


def score_output(output, sample):
    task = sample["task"]
    if task == "multi_key":
        pairs = re.findall(f"({KEY})[\\s\"':=,>-]+({VALUE})", output.upper())
        return set_f1(pairs, [tuple(pair) for pair in sample["expected_answers"]])
    if task == "multi_value":
        return set_f1(re.findall(VALUE, output.upper()), sample["expected_answers"])
    if task in {"qa_single", "qa_multi"}:
        return best_qa_f1(output, sample["expected_answers"])
    raise ValueError(f"未知任务：{task}")


def aggregate(rows):
    cells = defaultdict(list)
    for row in rows:
        score = row["task_score"]
        if row.get("error") or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("未完成请求或无效分数不能参与质量汇总")
        cells[(row["task"], row["length_bucket"])].append(score)
    if set(t for t, _ in cells) != set(TASKS):
        raise ValueError("质量评估缺少主任务")
    lengths = sorted({l for _, l in cells})
    if len(lengths) != 4 or any((t, l) not in cells for t in TASKS for l in lengths):
        raise ValueError("质量评估缺少长度单元")
    cell_means = {f"{t}:{l}": mean(cells[t, l]) for t in TASKS for l in lengths}
    tasks = {t: mean(cell_means[f"{t}:{l}"] for l in lengths) for t in TASKS}
    return {"all": mean(tasks.values()), **tasks, "cells": cell_means,
            "lengths": {str(l): mean(cell_means[f"{t}:{l}"] for t in TASKS) for l in lengths},
            "count": len(rows)}


def align(reference, candidate):
    ref = {r["sample_id"]: r for r in reference}
    cand = {r["sample_id"]: r for r in candidate}
    if len(ref) != len(reference) or len(cand) != len(candidate) or ref.keys() != cand.keys():
        raise ValueError("配对比较必须具有完全相同且不重复的样本 ID")
    candidate = [cand[r["sample_id"]] for r in reference]
    for a, b in zip(reference, candidate):
        if any(a.get(k) != b.get(k) for k in ("split", "task", "length_bucket", "source_group_id")):
            raise ValueError("配对样本的来源或分层不一致")
    return list(reference), candidate


def comparison(reference, candidate, config):
    reference, candidate = align(reference, candidate)
    return compare_summary(aggregate(reference), aggregate(candidate), config)


def compare_summary(ref, cand, config):
    gaps = {j: ref[j] - cand[j] for j in config.epsilons}
    violations = [max(0.0, (gaps[j] - eps) / eps) for j, eps in config.epsilons.items()]
    return {"gaps": gaps, "passed": all(gaps[j] <= eps + 1e-12 for j, eps in config.epsilons.items()),
            "key": (max(violations), sum(violations), -cand["all"]), "scores": cand}


def reference_valid(rows):
    scores = aggregate(rows)
    return all(scores[t] > 0 for t in TASKS)


def keys_equal(a, b):
    return all(abs(x-y) <= 1e-12 for x, y in zip(a, b))


def bootstrap_indices(rows, repeats, seed):
    """同一个来源组不得跨单元；每次共同抽取所有配置的同一组行。"""
    cells = defaultdict(lambda: defaultdict(list))
    owner = {}
    for i, row in enumerate(rows):
        cell = (row["task"], row["length_bucket"])
        group = row.get("source_group_id", row["sample_id"])
        if group in owner and owner[group] != cell:
            raise ValueError("来源组跨越分层单元，不能按单元独立重采样")
        owner[group] = cell
        cells[cell][group].append(i)
    if any(len(groups) < 2 for groups in cells.values()):
        return None
    rng = random.Random(seed)
    groups = [list(cells[cell].values()) for cell in sorted(cells)]
    return [[i for pool in groups for group in rng.choices(pool, k=len(pool)) for i in group]
            for _ in range(repeats)]


def resampled_summaries(rows, draws):
    # 只重采样分数，不复制长提示和文本；同批 draws 可被所有候选复用。
    return [aggregate([rows[i] for i in indices]) for indices in draws]


def paired_interval(reference, candidate, config):
    reference, candidate = align(reference, candidate)
    point = comparison(reference, candidate, config)
    draws = bootstrap_indices(reference, config.bootstrap_samples, config.raw["scoring"]["bootstrap_seed"])
    result = {"gaps": point["gaps"], "intervals": None, "method": "paired_stratified_source_group_bootstrap",
              "groups": len({r.get("source_group_id", r["sample_id"]) for r in reference}),
              "repeats": config.bootstrap_samples, "simultaneous_coverage": False}
    if draws is None:
        result["note"] = "至少一个单元只有一个独立来源组，区间不可可靠估计"
        return result
    values = {j: [] for j in config.epsilons}
    for a, b in zip(resampled_summaries(reference, draws), resampled_summaries(candidate, draws)):
        for j in values:
            values[j].append(a[j] - b[j])
    def percentile(values, p):
        ordered = sorted(values)
        pos = (len(ordered)-1)*p
        lo = int(pos)
        return ordered[lo] + (ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(pos-lo)
    result["intervals"] = {j: [percentile(v, .025), percentile(v, .975)] for j, v in values.items()}
    result["note"] = "区间未经多重比较校正；点估计达标不等于已经证明非劣性"
    return result
