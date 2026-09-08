"""只在实验集上运行的多保真束搜索与反向删层。"""
from __future__ import annotations

import math
import random

from autokv.io import append_jsonl
from autokv.v2_policy import Policy, endpoint_policies
from autokv.v3_metrics import (aggregate, bootstrap_indices, compare_summary, comparison,
                              keys_equal, reference_valid, resampled_summaries)
from autokv.v3_runtime import SearchBudgetExceeded, utc_now


def policy_for(layers, num_layers):
    layers = tuple(sorted(layers))
    return Policy(f"p{len(layers)}-" + ("-".join(map(str, layers)) or "empty"), layers, num_layers)


def promote(config, reference, rows_by_policy, policies, count, seed):
    """保留基础前列、完全并列和重采样中仍可能不差于边界的候选。"""
    summaries = {p.config_id: comparison(reference, rows_by_policy[p.config_id], config) for p in policies}
    ordered = sorted(policies, key=lambda p: summaries[p.config_id]["key"])
    count = min(len(ordered), count)
    boundary = ordered[count-1]
    key = summaries[boundary.config_id]["key"]
    chosen = {p.config_id for p in ordered[:count]}
    reasons = {p.config_id: {"reason": "base_rank"} for p in ordered[:count]}
    draws = bootstrap_indices(reference, config.bootstrap_samples, seed)
    ref_draws = resampled_summaries(reference, draws) if draws is not None else None
    boundary_draws = resampled_summaries(rows_by_policy[boundary.config_id], draws) if draws is not None else None
    for p in ordered[count:]:
        pid = p.config_id
        if keys_equal(summaries[pid]["key"], key) or draws is None:
            chosen.add(pid)
            reasons[pid] = {"reason": "point_tie" if draws is not None else "insufficient_groups"}
            continue
        candidate_draws = resampled_summaries(rows_by_policy[pid], draws)
        wins = 0
        for a, b, c in zip(ref_draws, boundary_draws, candidate_draws):
            kb, kc = compare_summary(a, b, config)["key"], compare_summary(a, c, config)["key"]
            wins += kc < kb or keys_equal(kc, kb)
        fraction = wins / len(draws)
        keep = fraction >= config.raw["search"]["promotion_win_fraction"]
        if keep:
            chosen.add(pid)
        reasons[pid] = {"reason": "uncertain_promoted" if keep else "early_eliminated", "win_fraction": fraction}
    return [p for p in policies if p.config_id in chosen], reasons


def search(config, experiment, runner, trace_path):
    if any(r["split"] != "experiment" for r in experiment):
        raise ValueError("搜索只能接收实验集")
    ids = [r["sample_id"] for r in experiment]
    if len(ids) != config.experiment_size:
        raise ValueError("实验集不完整")
    p32, p0 = endpoint_policies(config.num_layers)
    runner.phase = "endpoints"
    reference = runner.evaluate(p32, "experiment", ids)
    p0_rows = runner.evaluate(p0, "experiment", ids)
    full_rows = {p0.config_id: p0_rows, p32.config_id: reference}

    def log(event, **fields):
        append_jsonl(trace_path, {"event": event, "timestamp": utc_now(), **fields})

    def result(status, candidate=None, **extra):
        return {"status": status, "candidate": candidate.record() if candidate else None,
                "reference": aggregate(reference), "p0": aggregate(p0_rows), **extra}

    if not reference_valid(reference):
        return result("reference_degenerate")
    if comparison(reference, p0_rows, config)["passed"]:
        log("selected_endpoint", policy=p0.record())
        return result("selected", p0, pruning_complete=True)
    runner.phase = "search"
    beam, selected = [p0], None
    feasible, removable = [], []
    try:
        for depth in range(1, config.max_layers+1):
            parents = {}
            candidates = {}
            for parent in beam:
                for layer in range(config.num_layers):
                    if layer in parent.bf16_layers:
                        continue
                    policy = policy_for((*parent.bf16_layers, layer), config.num_layers)
                    candidates[policy.config_id] = policy
                    parents.setdefault(policy.config_id, []).append(parent)
            active = sorted(candidates.values(), key=lambda p: p.bf16_layers)
            rng = random.Random(config.seed+2+depth)
            tie_order = list(active)
            rng.shuffle(tie_order)
            priority = {p.config_id: i for i, p in enumerate(tie_order)}
            log("depth", depth=depth, candidates=[p.record() for p in active],
                tie_priority=[p.config_id for p in tie_order])
            levels = config.fidelities if config.raw["search"]["schedule"] == "progressive" else (len(ids),)
            feasible = []
            for stage, count in enumerate(levels):
                rows_by_policy = {}
                for p in active:
                    before = runner.statistics()["total"]["requests"]
                    rows = runner.evaluate(p, "experiment", ids[:count])
                    rows_by_policy[p.config_id] = rows
                    comp = comparison(reference[:count], rows, config)
                    gains = {parent.config_id: {j: comp["scores"][j]-aggregate(full_rows[parent.config_id][:count])[j]
                                                for j in config.epsilons} for parent in parents[p.config_id]}
                    log("evaluation", depth=depth, policy=p.record(), samples=count, comparison=comp,
                        parents=[parent.config_id for parent in parents[p.config_id]], marginal_gains=gains,
                        new_requests=runner.statistics()["total"]["requests"]-before)
                    if count == len(ids):
                        full_rows[p.config_id] = rows
                        if comp["passed"]:
                            feasible.append(p)
                if count < len(ids):
                    keep = max(config.beam_width, math.ceil(len(active)/4)) if stage == 0 else config.beam_width
                    promoted, reasons = promote(config, reference[:count], rows_by_policy, active, keep,
                                                config.raw["scoring"]["bootstrap_seed"]+depth*10+stage)
                    for p in active:
                        log("promotion", depth=depth, samples=count, policy=p.record(), **reasons[p.config_id])
                    active = promoted
            rank = lambda p: (*comparison(reference, full_rows[p.config_id], config)["key"], priority[p.config_id])
            if feasible:
                selected = min(feasible, key=rank)
                break
            beam = sorted(active, key=rank)[:config.beam_width]
            log("beam", depth=depth, policies=[p.record() for p in beam])
        if selected is None:
            return result("no_feasible_within_budget", reason="达到容量允许的层数或预设层数上限")
        while selected.k:
            deletions = [policy_for(set(selected.bf16_layers)-{layer}, config.num_layers) for layer in selected.bf16_layers]
            random.Random(config.seed+3+sum(selected.bf16_layers)).shuffle(deletions)
            removable = []
            for p in deletions:
                rows = full_rows.get(p.config_id)
                if rows is None:
                    rows = runner.evaluate(p, "experiment", ids)
                    full_rows[p.config_id] = rows
                comp = comparison(reference, rows, config)
                log("deletion", parent=selected.record(), policy=p.record(), samples=len(ids), comparison=comp)
                if comp["passed"]:
                    removable.append(p)
            if not removable:
                break
            selected = min(removable, key=lambda p: comparison(reference, full_rows[p.config_id], config)["key"])
            log("delete_accepted", policy=selected.record())
        return result("selected", selected, pruning_complete=True)
    except SearchBudgetExceeded as exc:
        if removable:
            selected = min(removable, key=lambda p: comparison(reference, full_rows[p.config_id], config)["key"])
        if selected is None and feasible:
            selected = min(feasible, key=lambda p: comparison(reference, full_rows[p.config_id], config)["key"])
        log("budget_exhausted", reason=str(exc), candidate=selected.record() if selected else None)
        return result("selected" if selected else "no_feasible_within_budget", selected,
                      pruning_complete=False, reason=str(exc))
