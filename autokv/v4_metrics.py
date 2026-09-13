"""短代码评分、配对统计与两批构造判定。"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from statistics import mean

from autokv.v3_metrics import align, bootstrap_indices
from autokv.v4_config import SCORE_VERSION, TASKS

CODE = re.compile(r"(?<!\w)[0-9]{6}(?!\w)")


def codes(output):
    return CODE.findall(output)


def score_output(output, sample):
    if sample["task"] not in TASKS or sample["scorer"] != SCORE_VERSION:
        raise ValueError("未知的 v4 任务或评分版本")
    return float(set(codes(output)) == set(sample["expected_answers"]) and len(sample["expected_answers"]) == 1)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p+z*z/(2*count))/(1+z*z/count)
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0.0, center-half), min(1.0, center+half)]


def diagnostics(rows):
    found = [codes(r["output_text"]) for r in rows]
    return {"incorrect": sum(r["task_score"] == 0 for r in rows),
            "no_code": sum(not x for x in found), "multiple_codes": sum(len(set(x)) > 1 for x in found),
            "repeated_code": sum(len(x) > len(set(x)) for x in found),
            "extra_code_occurrences": sum(len(x)-len(set(x)) for x in found),
            "empty": sum(not r["output_text"].strip() for r in rows),
            "length_finished": sum(r.get("finish_reason") == "length" for r in rows)}


def paired_summary(reference, candidate, config):
    reference, candidate = align(reference, candidate)
    n = len(reference)
    if not n or any(r.get("error") or r["task_score"] not in (0, 1) for r in (*reference, *candidate)):
        raise ValueError("构造比较需要完整的二值评分，通信错误不能当作答错")
    a, b = sum(r["task_score"] for r in reference), sum(r["task_score"] for r in candidate)
    differences = [x["task_score"]-y["task_score"] for x, y in zip(reference, candidate)]
    interval, status = None, "degenerate"
    if len(set(differences)) > 1:
        draws = bootstrap_indices(reference, config.bootstrap_samples, config.raw["scoring"]["bootstrap_seed"])
        if draws is None:
            status = "insufficient_groups"
        else:
            values = sorted(mean(differences[i] for i in indices) for indices in draws)
            def percentile(p):
                index = (len(values)-1)*p
                lo = int(index)
                return values[lo]+(values[min(lo+1, len(values)-1)]-values[lo])*(index-lo)
            interval, status = [percentile(.025), percentile(.975)], "estimated"
    limits = config.raw["construction"]
    bf16_ok = Fraction(int(a), n) >= Fraction(str(limits["min_bf16_score"]))
    gap_ok = Fraction(int(a-b), n) > Fraction(str(limits["min_fp8_gap"]))
    return {"count": n, "bf16_successes": int(a), "fp8_successes": int(b),
            "bf16_score": a/n, "fp8_score": b/n, "gap": (a-b)/n,
            "bf16_interval95": wilson(a, n), "gap_interval95": interval, "gap_interval_status": status,
            "bf16_solvable": bf16_ok, "fp8_recovery_needed": gap_ok, "passed": bf16_ok and gap_ok,
            "bf16_errors": diagnostics(reference), "fp8_errors": diagnostics(candidate),
            "note": "点判据与区间分开；退化区间不能解释为总体误差为零，区间未经同时覆盖校正"}


def confirmation_passed(batches, config):
    return len(batches) == config.raw["construction"]["confirmation_batches"] and all(b["passed"] for b in batches)
