"""固定晋级名额的束搜索：复筛决定方向，完整实验集只确认少量候选。"""
from __future__ import annotations

import random

from autokv.io import append_jsonl
from autokv.v2_policy import endpoint_policies
from autokv.v3_metrics import aggregate, comparison, reference_valid
from autokv.v3_runtime import SearchBudgetExceeded, utc_now
from autokv.v3_search import policy_for


def search(config, experiment, runner, trace_path):
    ids = [r["sample_id"] for r in experiment]
    if (len(ids) != config.experiment_size or len(set(ids)) != len(ids)
            or any(r["split"] != "experiment" for r in experiment)):
        raise ValueError("选层需要完整、不重复的实验集，不能使用测试集")
    coarse, medium, full = config.fidelities
    limits = config.raw["search"]
    p32, p0 = endpoint_policies(config.num_layers, config.low_kv_dtype)
    runner.phase = "endpoints"
    reference = runner.evaluate(p32, "experiment", ids)
    low = runner.evaluate(p0, "experiment", ids)
    checked, pool, summaries, priority = {}, {}, {}, {}
    depth, early_used = 0, 0

    def log(event, **fields):
        append_jsonl(trace_path, {"event": event, "timestamp": utc_now(), **fields})

    def result(status, candidate=None, **extra):
        return {"status": status, "candidate": candidate.record() if candidate else None,
                "reference": aggregate(reference, config), "p0": aggregate(low, config),
                "depth_reached": depth, "full_evaluated": [p.record() for p in checked.values()],
                "full_candidates_used": len(checked), "early_full_candidates_used": early_used,
                "full_candidates_limit": limits["full_candidates"], "pruning_complete": False,
                "pruning_status": "not_requested", **extra}

    def evaluate(policy, count, parents=()):
        before = runner.statistics()["total"]["requests"]
        rows = runner.evaluate(policy, "experiment", ids[:count])
        comp = comparison(reference[:count], rows, config)
        log("evaluation", depth=policy.k, policy=policy.record(), samples=count, comparison=comp,
            parents=[p.config_id for p in parents],
            new_requests=runner.statistics()["total"]["requests"]-before)
        return comp

    def rank(policy):
        return (*summaries[policy.config_id]["key"], policy.k, priority[policy.config_id])

    def confirm(policy, stage):
        # 续跑按同一固定前缀重放决策；缓存补齐，不会新增第七个完整候选。
        log("full_confirmation", depth=depth, policy=policy.record(), stage=stage,
            slot=len(checked)+1, limit=limits["full_candidates"])
        comp = evaluate(policy, full)
        checked[policy.config_id] = policy
        return comp["passed"]

    if not reference_valid(reference, config):
        return result("reference_degenerate")
    if comparison(reference, low, config)["passed"]:
        log("selected_endpoint", policy=p0.record())
        return result("selected", p0)
    runner.phase = "search"
    beam = [p0]
    try:
        for depth in range(1, config.max_layers+1):
            candidates, parents = {}, {}
            for parent in beam:
                for layer in range(config.num_layers):
                    if layer in parent.bf16_layers:
                        continue
                    p = policy_for((*parent.bf16_layers, layer), config.num_layers, config.low_kv_dtype)
                    candidates[p.config_id] = p
                    parents.setdefault(p.config_id, []).append(parent)
            active = sorted(candidates.values(), key=lambda p: p.bf16_layers)
            shuffled = list(active)
            random.Random(config.seed+2+depth).shuffle(shuffled)
            priority.update({p.config_id: i for i, p in enumerate(shuffled)})
            log("depth", depth=depth, candidates=[p.record() for p in active],
                tie_priority=[p.config_id for p in shuffled])
            coarse_scores = {p.config_id: evaluate(p, coarse, parents[p.config_id]) for p in active}
            ordered = sorted(active, key=lambda p: (*coarse_scores[p.config_id]["key"], priority[p.config_id]))
            # 每条父路径获得复筛名额；共享子组合只计算一次，空余名额按全局排名补齐。
            quota = limits["medium_candidates"] // len(beam)
            chosen = {}
            for parent in beam:
                branch = [p for p in ordered if parent in parents[p.config_id]][:quota]
                chosen.update({p.config_id: p for p in branch})
            for p in ordered:
                if len(chosen) >= limits["medium_candidates"]:
                    break
                chosen[p.config_id] = p
            for p in active:
                log("promotion", depth=depth, samples=coarse, policy=p.record(),
                    reason="fixed_quota" if p.config_id in chosen else "quota_eliminated")
            for p in chosen.values():
                summaries[p.config_id] = evaluate(p, medium, parents[p.config_id])
                pool[p.config_id] = p
            beam = sorted(chosen.values(), key=rank)[:config.beam_width]
            log("beam", depth=depth, samples=medium, policies=[p.record() for p in beam])
            eligible = [p for p in chosen.values() if summaries[p.config_id]["passed"]]
            if eligible and early_used < limits["early_full_candidates"]:
                early_used += 1
                p = min(eligible, key=rank)
                if confirm(p, "early"):
                    return result("selected", p)

        # 即使复筛点估计没有达标，也验证最佳未完整评估者；不把小样本阴性当成结论。
        finalists = sorted((p for pid, p in pool.items() if pid not in checked), key=rank)
        for p in finalists:
            if len(checked) >= limits["full_candidates"]:
                break
            if confirm(p, "final"):
                return result("selected", p)
        return result("no_feasible_within_budget", reason="已完成容量允许深度及预定完整评估；预算内未找到可行解，不代表不存在")
    except SearchBudgetExceeded as exc:
        log("budget_exhausted", reason=str(exc))
        return result("no_feasible_within_budget", reason=str(exc))
