"""有限条件扫描与两批确认；只冻结生成规则，不选择 KV 层。"""
from __future__ import annotations

import json

from autokv.io import atomic_write_json, atomic_write_text, read_json
from autokv.v2_policy import endpoint_policies
from autokv.v3_runtime import utc_now
from autokv.v4_data import GENERATOR_RULES, background_usage, ensure_samples, validate_samples
from autokv.v4_metrics import confirmation_passed, paired_summary


def choose_confirmation(conditions, maximum):
    available = [r for r in conditions if r.get("discovery")]
    solvable = [r for r in available if r["discovery"]["bf16_solvable"]]
    if solvable:
        key = lambda r: (-r["discovery"]["gap"], -r["discovery"]["bf16_score"],
                         r["condition"]["length_bucket"], r["condition"]["record_count"], r["condition"]["condition_id"])
    else:
        key = lambda r: (-r["discovery"]["bf16_score"], -r["discovery"]["gap"],
                         r["condition"]["length_bucket"], r["condition"]["record_count"], r["condition"]["condition_id"])
    return [r["condition"] for r in sorted(solvable or available, key=key)[:maximum]]


def choose_rule(conditions, config):
    passed = [r for r in conditions if confirmation_passed(r.get("confirmation", []), config)]
    return min(passed, key=lambda r: (-min(b["bf16_score"] for b in r["confirmation"]),
                                      r["condition"]["length_bucket"], r["condition"]["record_count"],
                                      r["condition"]["condition_id"])) if passed else None


def register(runner, split, rows, config):
    validate_samples(config, {**{s: list(v.values()) for s, v in runner.samples.items()}, split: rows})
    runner.samples[split] = {r["sample_id"]: r for r in rows}


def render_construction(directory, config, result, cost):
    limits = config.raw["construction"]
    low = config.low_precision
    lines = [f"# AutoKV-Skip v{config.experiment_version} 数据构造报告", "", f"运行：`{directory.name}`；状态：`{result['status']}`。",
             f"构造阶段完成：`{result['complete']}`；两批确认通过：`{result['construction_passed']}`。", "",
             f"每批独立要求 BF16 正确率 ≥ {limits['min_bf16_score']}，P32−P0 严格 > {limits['min_'+low+'_gap']}（绝对分差）。",
             "错误答案保留在完整分母内；通信失败表示未完成。没有按单题模型对错筛选数据。",
             "确认通过后才冻结一个生成条件；正式数据全部重新生成。", "",
             f"| 条件 | 状态 | 发现题数 | BF16 | {low.upper()} | 分差 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in result.get("conditions", []):
        summary = row.get("discovery")
        values = " | ".join(f"{summary[k]:.4f}" for k in ("bf16_score", low+"_score", "gap")) if summary else "— | — | —"
        lines.append(f"| {row['condition']['condition_id']} | {row['status']} | {summary['count'] if summary else '—'} | {values} |")
    for row in result.get("conditions", []):
        name = row["condition"]["condition_id"]
        if row.get("reason"):
            lines += ["", f"`{name}`：{row['reason']}。"]
        for label, summary in [("发现", row.get("discovery")), *[(f"确认 {b['batch_id']}", b) for b in row.get("confirmation", [])],
                               ("确认合并（仅供描述，不代替逐批判据）", row.get("confirmation_combined"))]:
            if summary:
                lines += ["", f"## {name}：{label}", "",
                          f"题数 {summary['count']}；BF16 {summary['bf16_score']:.6f}；{low.upper()} {summary[low+'_score']:.6f}；分差 {summary['gap']:.6f}。",
                          f"BF16 95% Wilson 区间：`{summary['bf16_interval95']}`；配对分差区间：`{summary['gap_interval95']}`（`{summary['gap_interval_status']}`）。",
                          f"BF16 可解：`{summary['bf16_solvable']}`；{low.upper()} 恢复需求：`{summary[low+'_recovery_needed']}`；两项合并：`{summary['passed']}`。",
                          f"BF16 错误/格式/截断：`{summary['bf16_errors']}`；{low.upper()}：`{summary[low+'_errors']}`。"]
    if result.get("selected_rule"):
        lines += ["", "## 冻结规则", "", f"条件：`{result['selected_rule']['condition']}`。",
                  "完整词表、模板、位置规则、种子、来源与两批确认统计见 `construction/selected-rule.json`。"]
    else:
        lines += ["", "当前有限范围与预算内未确认满足条件的数据；未完成时只能等待续跑，不能解释为负结论。",
                  "完整扫描仍未找到时，应保留负结果；本轮不自动改变数据分布或精度。"]
    if result.get("error"):
        lines += ["", "```text", result["error"], "```"]
    lines += ["", "## 成本与解释边界", "", f"实际累计开销（含正式阶段，如已运行）：`{cost}`。",
              f"背景复用统计：`{result.get('background_reuse', {})}`。",
              GENERATOR_RULES["background_reuse"]+"。",
              "发现集用于选择条件，确认的两个批次分别判断；没有将发现/确认并入独立测试。",
              "区间未经同时覆盖校正，点判据通过不等于已证明总体性质；退化 bootstrap 不解释为误差为零。", ""]
    path = directory / "report/CONSTRUCTION-v4.zh-CN.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def construct(config, directory, codec, get_sources, runner, source_info, progress=None):
    folder = directory / "construction"
    result_path = folder / "result.json"
    if result_path.exists() and read_json(result_path).get("complete"):
        result = read_json(result_path)
        render_construction(directory, config, result, runner.statistics())
        return result
    result = {"schema_version": 4, "status": "constructing", "complete": False,
              "construction_passed": False, "conditions": [], "selected_rule": None}
    p32, p0 = endpoint_policies(config.num_layers, config.low_kv_dtype)
    limits = config.raw["construction"]
    low = config.low_precision

    def save():
        atomic_write_text(folder / "conditions.jsonl", "".join(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n" for r in result["conditions"]))
        atomic_write_json(result_path, result)

    def endpoints(split, rows):
        register(runner, split, rows, config)
        runner.phase = split
        ids = [r["sample_id"] for r in rows]
        return (runner.evaluate(p32, split, ids), runner.evaluate(p0, split, ids)) if ids else ([], [])

    try:
        conditions = config.conditions()
        result["conditions"] = [{"condition": c, "status": "pending", "discovery": None, "confirmation": []} for c in conditions]
        save()
        specs = [{"split": "discovery", "condition": c, "batch_id": "discovery", "count": limits["discovery_per_condition"]} for c in conditions]
        rows, metadata = ensure_samples(folder / "discovery.jsonl", specs, config, codec, get_sources,
                                        allow_unconstructible=True, progress=progress)
        for row in result["conditions"]:
            row.update(metadata["conditions"][row["condition"]["condition_id"]+":discovery"])
        save()
        reference, candidate = endpoints("discovery", rows)
        result["background_reuse"] = background_usage({"discovery": rows})
        for row in result["conditions"]:
            cid = row["condition"]["condition_id"]
            if row["status"] == "unconstructible":
                continue
            row["discovery"] = paired_summary([r for r in reference if r["condition_id"] == cid],
                                              [r for r in candidate if r["condition_id"] == cid], config)
            row["status"] = "discovery_complete"
        save()
        selected_path = folder / "confirmation-plan.json"
        proposed = choose_confirmation(result["conditions"], limits["max_confirmation_conditions"])
        if selected_path.exists() and read_json(selected_path) != proposed:
            raise ValueError("冻结的确认条件与既有发现结果不一致")
        atomic_write_json(selected_path, proposed)
        specs = [{"split": "confirmation", "condition": c, "batch_id": f"batch-{b+1}", "count": limits["confirmation_per_batch"]}
                 for c in proposed for b in range(limits["confirmation_batches"])]
        rows, metadata = ensure_samples(folder / "confirmation.jsonl", specs, config, codec, get_sources,
                                        allow_unconstructible=True, progress=progress)
        reference, candidate = endpoints("confirmation", rows)
        result["background_reuse"].update(background_usage({"confirmation": rows}))
        for row in result["conditions"]:
            if row["condition"] not in proposed:
                continue
            cid = row["condition"]["condition_id"]
            for b in range(limits["confirmation_batches"]):
                batch = f"batch-{b+1}"
                state = metadata["conditions"][cid+":"+batch]
                if state["status"] == "unconstructible":
                    row.update(status="confirmation_unconstructible", reason=state["reason"])
                    continue
                summary = paired_summary([r for r in reference if r["condition_id"] == cid and r["batch_id"] == batch],
                                         [r for r in candidate if r["condition_id"] == cid and r["batch_id"] == batch], config)
                row["confirmation"].append({"batch_id": batch, **summary})
            if row["status"] != "confirmation_unconstructible":
                row["status"] = "confirmation_passed" if confirmation_passed(row["confirmation"], config) else "confirmation_failed"
                row["confirmation_combined"] = paired_summary([r for r in reference if r["condition_id"] == cid],
                                                               [r for r in candidate if r["condition_id"] == cid], config)
        chosen = choose_rule(result["conditions"], config)
        if chosen:
            rule = {"schema_version": 4, "run_id": directory.name, "frozen_at": utc_now(),
                    "condition": chosen["condition"], "generator": GENERATOR_RULES,
                    "data_config": config.raw["data"], "construction_config": limits,
                    "scoring": config.raw["scoring"], "sources": source_info,
                    "source_allocation_path": "inputs/source-allocation.json",
                    "confirmation": chosen["confirmation"], "chat_template": codec.template_text}
            atomic_write_json(folder / "selected-rule.json", rule)
            result.update(construction_passed=True, selected_rule=rule, status="construction_passed")
        else:
            result["status"] = "construction_not_found"
        result["complete"] = True
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "runtime_failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["updated_at"] = utc_now()
        save()
        render_construction(directory, config, result, runner.statistics())
    return result
