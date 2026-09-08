"""v3 端点、选层、测试和报告；测试分数不反馈到选层。"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from autokv.io import atomic_write_json, atomic_write_text, read_json
from autokv.v2_policy import Policy, endpoint_policies, theoretical_capacity
from autokv.v3_config import identity, load_config
from autokv.v3_data import DATA_ROOT, TransformersPromptCodec, load_dataset
from autokv.v3_metrics import aggregate, comparison, paired_interval, reference_valid
from autokv.v3_runtime import V3PolicyRunner, load_environment, recover_rows, utc_now
from autokv.v3_search import search


def prepare_run(root, config, manifest, environment, development):
    source_root = Path(__file__).resolve().parents[1]
    code = {p.relative_to(source_root).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted((source_root / "autokv").glob("*.py"))}
    run_id = "v3-" + identity({"config": config.raw, "dataset": manifest["dataset_id"],
                               "runtime": environment, "code": code, "development": development})[:16]
    directory = root / "runs" / run_id
    path = directory / "run-manifest.json"
    if not path.exists():
        inputs = directory / "inputs"
        for relative, content in code.items():
            atomic_write_text(inputs / relative, content)
        for relative in ("scripts", "configs/v3.0", "docs/v3.0"):
            source = root / relative
            if source.exists():
                shutil.copytree(source, inputs / relative, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for relative in ("pyproject.toml", "CONTEXT.md", "README.md"):
            if (root / relative).exists():
                shutil.copy2(root / relative, inputs / relative)
        data_relative = DATA_ROOT / ("development" if development else "quality")
        shutil.copytree(root / data_relative, inputs / data_relative, dirs_exist_ok=True)
        atomic_write_json(inputs / "runtime.json", environment)
        atomic_write_json(path, {"schema_version": 3, "run_id": run_id, "created_at": utc_now(),
                                "development": development, "dataset_id": manifest["dataset_id"],
                                "runtime": environment, "config": config.raw,
                                "state": "ready", "source": "实际源码/配置/数据副本见 inputs；Git 提交不是运行条件"})
    return directory


def measured_capacity(runner, reference, candidate, minimum):
    observed = {"reference": runner.capacities(reference), "candidate": runner.capacities(candidate)}
    def single(values):
        known = set(v for v in values if isinstance(v, int) and v > 0)
        return next(iter(known)) if len(known) == 1 else None
    ref, cand = single(observed["reference"]), single(observed["candidate"])
    ratio = cand/ref if ref and cand else None
    return {"status": "unverified" if ratio is None else "passed" if ratio+1e-12 >= minimum else "below_target",
            "reference_tokens": ref, "candidate_tokens": cand, "ratio": ratio, "minimum": minimum,
            "observations": observed, "note": "只采用测试阶段可解析且一致的容量；缺失或冲突保留原始记录"}


def test_use(directory, rows):
    """跨运行保留测试使用记录，不把换 run_id 后的重用称为独立测试。"""
    current = {"questions": sorted({r["source_question_id"] for r in rows}),
               "evidence": sorted({d for r in rows for d in r["evidence_document_ids"]})}
    reused = []
    for path in sorted(directory.parent.glob("v3-*/test-use.json")):
        if path.parent == directory or not any(p.stat().st_size for p in path.parent.glob("policies/*/test.jsonl")):
            continue
        prior = read_json(path)
        if any(set(current[key]) & set(prior[key]) for key in ("questions", "evidence")):
            reused.append(path.parent.name)
    atomic_write_json(directory / "test-use.json", current)
    return reused


def coverage(directory, config):
    result = {"full_experiment_configs": 0, "partial_experiment_configs": 0,
              "successful_answers": 0, "input_tokens": 0, "output_tokens": 0,
              "length_finished": 0, "empty_answers": 0}
    for path in sorted(directory.glob("policies/*/*.jsonl")):
        if path.stem not in {"experiment", "test", "development"}:
            continue
        rows = recover_rows(path)
        if path.stem == "experiment":
            key = "full_experiment_configs" if len(rows) == config.experiment_size else "partial_experiment_configs"
            result[key] += 1
        result["successful_answers"] += len(rows)
        result["input_tokens"] += sum(r["prompt_tokens"] for r in rows)
        result["output_tokens"] += sum(r["output_tokens"] for r in rows)
        result["length_finished"] += sum(r.get("finish_reason") == "length" for r in rows)
        result["empty_answers"] += sum(not r["output_text"].strip() for r in rows)
    return result


def render_report(directory, config, result):
    lines = ["# AutoKV-Skip v3.0 质量与容量报告", "", f"运行：`{directory.name}`；状态：`{result['status']}`。",
             f"流程完成：`{result['complete']}`；主实验达标：`{result['technical_goal_passed']}`。", "",
             f"质量判断采用预定点估计约束：总分下降不超过 {config.epsilons['all']}、每任务不超过 {config.raw['thresholds']['epsilon_task']}。",
             "置信区间未经同时覆盖校正；点估计达标不等于已经证明 1% 非劣性。", "",
             f"实际容差：`{config.epsilons}`；容量目标：`{config.capacity_ratio}×`。",
             "FP8 scale：固定 1.0，未作数据校准；prefix caching 关闭。", ""]
    if result.get("error"):
        lines += ["运行错误：", "", "```text", result["error"], "```", ""]
    candidate = result.get("candidate")
    lines += [f"冻结候选：`{candidate['bf16_layers'] if candidate else None}`（层号从 0 开始）。",
              "选择只使用 experiment；测试失败不会自动换层。P32 回退不是容量优化成功。", ""]
    if result.get("test_reused_from"):
        lines += [f"该测试的问题或证据已被其他运行使用：`{result['test_reused_from']}`。本次仅作重跑诊断，不计独立测试达标。", ""]
    data_manifest = directory / "inputs" / DATA_ROOT / ("development" if result.get("development") else "quality") / "dataset-manifest.json"
    if data_manifest.exists():
        data = read_json(data_manifest)
        lines += [f"数据题数：`{data['counts']}`；来源组数：`{data['source_groups']}`。",
                  f"来源划分：{data.get('source_allocation', '见数据清单')}。",
                  f"背景文档数：`{data.get('background_documents')}`；背景复用和原始来源见输入元数据。", ""]
    for split, summaries in result.get("scores", {}).items():
        lines += [f"## {split} 分数", "", "| 配置 | 总分 | multi_key | multi_value | qa_single | qa_multi |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, scores in summaries.items():
            lines.append("| " + name + " | " + " | ".join(f"{scores[j]:.6f}" for j in config.epsilons) + " |")
        lines += ["", "| 配置 | 任务与长度 | 分数 |", "|---|---|---:|"]
        for name, scores in summaries.items():
            for cell, value in scores["cells"].items():
                lines.append(f"| {name} | {cell} | {value:.6f} |")
        lines.append("")
    for name, interval in result.get("intervals", {}).items():
        lines += [f"## P32 − {name} 配对分差", "", "| 指标 | 点估计 | 95% 区间 |", "|---|---:|---|"]
        for j, gap in interval["gaps"].items():
            bounds = interval["intervals"][j] if interval["intervals"] else None
            lines.append(f"| {j} | {gap:.6f} | {bounds} |")
        lines += ["", interval["note"], ""]
    if candidate:
        p = Policy(candidate["name"], tuple(candidate["bf16_layers"]), config.num_layers)
        theory = theoretical_capacity(p, num_kv_heads=config.raw["model"]["num_kv_heads"], head_dim=config.raw["model"]["head_dim"])
        lines += ["## 容量", "", f"候选理论容量：`{theory}`。", f"测试实测：`{result.get('capacity')}`。", ""]
    lines += ["## 计算开销", "", "| 阶段 | 请求 | 重试 | 服务启动 | 墙钟秒 |", "|---|---:|---:|---:|---:|"]
    for phase, row in result["cost"]["phases"].items():
        lines.append(f"| {phase} | {row['requests']} | {row['retries']} | {row['server_starts']} | {row['seconds']:.3f} |")
    lines += ["", f"合计：`{result['cost']['total']}`。", f"样本与配置统计：`{result.get('coverage')}`。", "",
              f"存在未写完结束时间的启动记录：`{result['cost'].get('timing_incomplete', False)}`；若为 true，墙钟统计只是已记录的下界。", "",
              "仅能解释本次预先确定的任务分布。有限束宽与早期淘汰不保证全局最优；1280 题不自动保证统计检验能力。",
              "未执行方法对照或吞吐实验时，不声称搜索优于其他算法或推理更快。", ""]
    path = directory / "report/QUALITY-v3.zh-CN.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def execute(config, splits, runner, directory, *, development=False):
    completed = directory / "completed-manifest.json"
    if completed.exists() and read_json(completed).get("complete"):
        return read_json(completed)
    result = {"schema_version": 3, "run_id": directory.name, "development": development,
              "complete": False, "quality_passed": False, "technical_goal_passed": False,
              "status": "running", "scores": {}, "candidate": None}
    p32, p0 = endpoint_policies(config.num_layers)
    try:
        if development:
            runner.phase = "development"
            rows = runner.evaluate(p32, "development", [r["sample_id"] for r in splits["development"]])
            result.update(status="development_complete", complete=True, scores={"development": {"P32": aggregate(rows)}})
        else:
            selection_path = directory / "selection.json"
            if selection_path.exists():
                selection = read_json(selection_path)
            else:
                selection = search(config, splits["experiment"], runner, directory / "search-trace.jsonl")
                selection.update(frozen_at=utc_now(), run_id=directory.name)
                atomic_write_json(selection_path, selection)
            result["selection"] = selection
            result["scores"]["experiment"] = {"P32": selection["reference"], "P0": selection["p0"]}
            record = selection["candidate"]
            if record is None:
                result.update(status=selection["status"], complete=True)
            else:
                candidate = Policy(record["name"], tuple(record["bf16_layers"]), config.num_layers)
                if candidate.config_id != record["config_id"] or selection["run_id"] != directory.name:
                    raise ValueError("冻结的策略记录与运行不一致")
                result["candidate"] = record
                exp_rows = runner.evaluate(candidate, "experiment", [r["sample_id"] for r in splits["experiment"]])
                result["scores"]["experiment"]["selected"] = aggregate(exp_rows)
                reused = test_use(directory, splits["test"])
                runner.phase = "test"
                policies = {p.config_id: p for p in (p32, p0, candidate)}
                tests = {pid: runner.evaluate(p, "test", [r["sample_id"] for r in splits["test"]]) for pid, p in policies.items()}
                reference, chosen = tests[p32.config_id], tests[candidate.config_id]
                names = [("P32", p32), ("P0", p0), ("selected", candidate)]
                result["scores"]["test"] = {name: aggregate(tests[p.config_id]) for name, p in names}
                comparisons = {name: comparison(reference, tests[p.config_id], config) for name, p in names[1:]}
                result["comparisons"] = comparisons
                interval_cache = {pid: paired_interval(reference, rows, config) for pid, rows in tests.items() if pid != p32.config_id}
                result["intervals"] = {name: interval_cache[p.config_id] for name, p in names[1:]}
                valid = reference_valid(reference)
                quality = valid and comparisons["selected"]["passed"]
                capacity = measured_capacity(runner, p32, candidate, config.capacity_ratio)
                passed = quality and capacity["status"] == "passed" and not reused
                status = "test_not_independent" if reused else "reference_degenerate" if not valid else "test_quality_failed" if not quality else (
                    "passed" if passed else "capacity_unverified" if capacity["status"] == "unverified" else "capacity_below_target")
                result.update(status=status, complete=True, quality_passed=quality, technical_goal_passed=passed,
                              reference_valid=valid, capacity=capacity, test_reused_from=reused)
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "runtime_failed",
                      error=f"{type(exc).__name__}: {exc}", complete=False)
        raise
    finally:
        result.update(cost=runner.statistics(), finished_at=utc_now())
        # 诊断不能因为尾行未写完而消失；统计与报告错误单独保留。
        try:
            result["coverage"] = coverage(directory, config)
            result["report"] = str(render_report(directory, config, result))
        except (ValueError, OSError, KeyError) as exc:
            result["report_error"] = str(exc)
            if result["complete"]:
                result.update(complete=False, technical_goal_passed=False, status="runtime_failed")
        atomic_write_json(completed, result)
        manifest_path = directory / "run-manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifest.update(state=result["status"], complete=result["complete"], updated_at=utc_now())
            atomic_write_json(manifest_path, manifest)
    return result


def run_pipeline(root, *, port=8010, development=False):
    if not sys.platform.startswith("linux"):
        raise ValueError("真实 v3 GPU 实验请在 Linux 服务器执行；本地测试不能替代 GPU 结果")
    config = load_config(root)
    manifest, splits = load_dataset(root, config, development)
    environment = load_environment(root, config, observe_runtime=True)
    codec = TransformersPromptCodec(Path(environment["model_path"]))
    if codec.template_sha256 != manifest["template_sha256"]:
        raise ValueError("实际 chat template 与数据生成时不同；请使用匹配模板或重新构造数据")
    environment["chat_template"] = codec.template_text
    directory = prepare_run(root, config, manifest, environment, development)
    runner = V3PolicyRunner(config, root, directory, environment, splits, port=port)
    return execute(config, splits, runner, directory, development=development)
