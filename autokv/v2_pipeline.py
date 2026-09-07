"""AutoKV-Skip v2.1 质量与容量主实验，以及独立的可选随机分析。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from autokv.commands import runtime_identity
from autokv.config import Profile, load_profile
from autokv.io import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_text_file,
)
from autokv.v2_config import (
    V2_CONFIG_RELATIVE_PATH,
    V2_DATA_RELATIVE_ROOT,
    V2QualityConfig,
    load_v2_config,
)
from autokv.v2_data import (
    TransformersPromptCodec,
    load_frozen_v2_dataset,
    make_hard_rows,
)
from autokv.v2_metrics import aggregate_v2, paired_gap_summary, quality_constraints, reference_is_valid
from autokv.v2_policy import (
    Policy,
    endpoint_policies,
    group_policies,
    layer_policies,
    nested_budget_policy,
    random_control_policies,
    rank_by_recovery,
    theoretical_capacity,
)
from autokv.v2_runtime import V2PolicyRunner


@dataclass(frozen=True)
class V2RunContext:
    root: Path
    config_path: Path
    config: V2QualityConfig
    profile: Profile
    lock: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]
    calibration: tuple[Mapping[str, Any], ...]
    heldout: tuple[Mapping[str, Any], ...]
    source: Mapping[str, Any]
    run_id: str


def recommend_pilot_difficulty(
    task_means: Mapping[str, float], config: V2QualityConfig
) -> tuple[str, str]:
    if set(task_means) != set(config.hard_families):
        raise ValueError("pilot 必须包含全部三个 Hard 任务族")
    if min(task_means.values()) < config.pilot_score_floor:
        return "easy", "at_least_one_family_below_floor"
    if all(score >= config.pilot_score_ceiling for score in task_means.values()):
        return "hard", "all_families_at_or_above_ceiling"
    return "standard", "bf16_within_preregistered_range"


def run_v2_pilot(root: Path, *, port: int = 8000) -> Mapping[str, Any]:
    """在任何 P0 或正式数据产生前，最多用 9 条 BF16 样本决定难度档位。"""

    root = root.resolve()
    config_path = root / V2_CONFIG_RELATIVE_PATH
    config = load_v2_config(config_path)
    if config.hard_difficulty != "standard":
        raise ValueError("pilot 只能对初始 standard 难度运行一次；调整后不得重跑")
    if (root / V2_DATA_RELATIVE_ROOT / "dataset-manifest.json").exists():
        raise ValueError("正式 v2 数据已经冻结，禁止事后运行 pilot")
    profile = load_profile(root / "configs" / f"{config.profile}.json")
    lock = _load_v2_lock(root, config)
    if lock.get("backend") != "local_vllm":
        raise ValueError("v2 pilot 当前只支持项目已验证的 local_vllm 环境")
    source = _source_identity(root)
    _require_linux()
    codec = TransformersPromptCodec(Path(str(lock["model_path"])))
    pilot_samples = tuple(
        make_hard_rows(
            config,
            codec,
            split_seeds=(("calibration", (config.pilot_seed,)),),
        )
    )
    if len(pilot_samples) != config.pilot_max_requests:
        raise ValueError(
            f"pilot 必须恰好 {config.pilot_max_requests} 条，实际 {len(pilot_samples)}"
        )
    pilot_json = json.dumps(
        pilot_samples,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    pilot_sha256 = hashlib.sha256(pilot_json).hexdigest()
    run_id = (
        "pilot-"
        + hashlib.sha256(
            "\n".join(
                (
                    sha256_text_file(config_path),
                    pilot_sha256,
                    runtime_identity(lock),
                    str(source["tree_sha256"]),
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
    )
    runner = V2PolicyRunner(
        config,
        profile,
        root,
        lock,
        run_id,
        port=port,
    )
    p32, _ = endpoint_policies(config.num_layers)
    result_path = runner.run_policy(
        p32,
        pilot_samples,
        split="calibration",
        split_sha256=pilot_sha256,
        relative_directory=Path("quality/pilot"),
    )
    result_by_id = {row["sample_id"]: row for row in read_jsonl(result_path)}
    task_scores: dict[str, list[float]] = {
        family: [] for family in config.hard_families
    }
    case_scores: list[dict[str, Any]] = []
    for sample in pilot_samples:
        score = float(result_by_id[sample["sample_id"]]["task_score"])
        task_scores[str(sample["task"])].append(score)
        case_scores.append(
            {
                "sample_id": sample["sample_id"],
                "task": sample["task"],
                "target_tokens": sample["target_tokens"],
                "task_score": score,
            }
        )
    task_means = {
        task: sum(values) / len(values) for task, values in task_scores.items()
    }
    overall = sum(task_means.values()) / len(task_means)
    recommendation, reason = recommend_pilot_difficulty(task_means, config)
    decision = {
        "schema_version": 2,
        "run_id": run_id,
        "policy": p32.record(),
        "pilot_sha256": pilot_sha256,
        "requests": len(pilot_samples),
        "current_difficulty": "standard",
        "recommended_difficulty": recommendation,
        "reason_code": reason,
        "score_floor": config.pilot_score_floor,
        "score_ceiling": config.pilot_score_ceiling,
        "task_means": task_means,
        "overall_s_hard": overall,
        "cases": case_scores,
        "next_action": (
            "保持 standard，直接冻结正式数据"
            if recommendation == "standard"
            else f"只把 data.hard.difficulty 改为 {recommendation}，不重跑 pilot，然后冻结正式数据"
        ),
    }
    decision_path = root / "runs" / run_id / "pilot-decision.json"
    atomic_write_json(decision_path, decision)
    return {
        "complete": True,
        "run_id": run_id,
        "requests": len(pilot_samples),
        "recommended_difficulty": recommendation,
        "reason_code": reason,
        "decision_path": decision_path.relative_to(root).as_posix(),
    }


def _load_v2_lock(root: Path, config: V2QualityConfig) -> Mapping[str, Any]:
    """读取已有运行路径；也可直接指定本地 vLLM 与模型，无需重新生成环境锁。"""
    path = root / "runs" / "_environment" / "lock.json"
    value = read_json(path) if path.is_file() else {}
    if not isinstance(value, Mapping):
        raise ValueError("运行环境配置必须是 JSON 对象")
    lock = dict(value)
    overrides = {
        key: os.environ[name]
        for key, name in (
            ("vllm", "AUTOKV_VLLM_BIN"),
            ("model_path", "AUTOKV_MODEL_PATH"),
        )
        if os.environ.get(name)
    }
    lock.update(overrides)
    lock.setdefault("backend", "local_vllm")
    lock.setdefault("model_revision", config.model_revision)
    if lock["model_revision"] != config.model_revision:
        raise ValueError("环境锁的模型 revision 与 v2 冻结配置不一致")
    if lock.get("model_id", config.model_id) != config.model_id:
        raise ValueError("运行环境的模型与 v2 配置不一致")
    if lock["backend"] == "local_vllm":
        for key, variable in (("vllm", "AUTOKV_VLLM_BIN"), ("model_path", "AUTOKV_MODEL_PATH")):
            if not isinstance(lock.get(key), str) or not lock[key]:
                raise ValueError(f"缺少 {key}；设置 {variable} 或复用 runs/_environment/lock.json")
        if not Path(lock["model_path"]).is_dir():
            raise ValueError(f"本地模型目录不存在：{lock['model_path']}")
        lock.setdefault("python", sys.executable)
        identity = {
            key: lock.get(key)
            for key in (
                "runtime_id", "vllm", "model_path", "versions",
                "ld_library_path", "cuda_home", "nvcc",
            )
        }
        lock["runtime_id"] = "local-vllm-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    elif lock["backend"] == "docker":
        for key in ("image_ref", "image_digest"):
            if not isinstance(lock.get(key), str) or not lock[key]:
                raise ValueError(f"容器运行环境缺少 {key}")
    else:
        raise ValueError(f"不支持的运行方式：{lock['backend']}")
    return lock


def _source_identity(root: Path) -> Mapping[str, Any]:
    from autokv.cli import _source_identity as v1_source_identity

    return v1_source_identity(root)


def _v2_run_id(
    config_sha256: str,
    dataset_sha256: str,
    runtime_id: str,
    model_revision: str,
    source_tree_sha256: str,
) -> str:
    payload = "\n".join(
        (
            "autokv-v2",
            config_sha256,
            dataset_sha256,
            runtime_id,
            model_revision,
            source_tree_sha256,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _require_linux() -> None:
    if not sys.platform.startswith("linux"):
        raise ValueError("v2 GPU 运行需要 Linux；请在实验服务器执行")


def load_v2_run_context(root: Path) -> V2RunContext:
    root = root.resolve()
    config_path = root / V2_CONFIG_RELATIVE_PATH
    config = load_v2_config(config_path)
    profile = load_profile(root / "configs" / f"{config.profile}.json")
    if (
        profile.model.model_id != config.model_id
        or profile.model.num_layers != config.num_layers
        or profile.model.max_model_len != config.max_model_len
        or profile.model.attention_backend != "FLASHINFER"
        or profile.kv_cache_memory != "16G"
    ):
        raise ValueError("v1 full profile 与 v2 冻结运行设置不一致")
    lock = _load_v2_lock(root, config)
    dataset_manifest, calibration, heldout = load_frozen_v2_dataset(
        config,
        root / V2_DATA_RELATIVE_ROOT,
        config_path=config_path,
    )
    source = _source_identity(root)
    run_id = _v2_run_id(
        sha256_text_file(config_path),
        str(dataset_manifest["dataset_sha256"]),
        runtime_identity(lock),
        config.model_revision,
        str(source["tree_sha256"]),
    )
    context = V2RunContext(
        root,
        config_path,
        config,
        profile,
        lock,
        dataset_manifest,
        calibration,
        heldout,
        source,
        run_id,
    )
    _ensure_run_manifest(context)
    return context


def _ensure_run_manifest(context: V2RunContext) -> Path:
    path = context.root / "runs" / context.run_id / "run-manifest.json"
    expected = {
        "schema_version": 2,
        "experiment_version": context.config.version,
        "run_id": context.run_id,
        "git_commit": context.source.get("git_commit"),
        "git_dirty": context.source.get("git_dirty"),
        "source_tree_sha256": context.source["tree_sha256"],
        "source_files": context.source["files"],
        "config_path": V2_CONFIG_RELATIVE_PATH.as_posix(),
        "config_sha256": sha256_text_file(context.config_path),
        "dataset_manifest_path": (
            V2_DATA_RELATIVE_ROOT / "dataset-manifest.json"
        ).as_posix(),
        "dataset_sha256": context.dataset_manifest["dataset_sha256"],
        "model_id": context.config.model_id,
        "model_revision": context.config.model_revision,
        "runtime_backend": str(context.lock.get("backend", "docker")),
        "runtime_id": runtime_identity(context.lock),
        "enable_prefix_caching": False,
        "storage_timezone": "UTC",
        "display_timezone": "Asia/Shanghai",
    }
    if path.is_file():
        observed = read_json(path)
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != value
            for key, value in expected.items()
            if key not in {"git_commit", "git_dirty"}
        ):
            raise ValueError("已有 v2 run manifest 与当前冻结输入不一致")
        return path
    # 运行前保存实际输入，发布时无需从 Git 找回服务器上的未提交源码。
    inputs = path.parent / "inputs"
    relative_files = [Path(record["path"]) for record in context.source["files"]]
    relative_files.extend(
        V2_DATA_RELATIVE_ROOT / name
        for name in ("calibration.jsonl", "heldout.jsonl", "dataset-manifest.json")
    )
    for relative in relative_files:
        destination = inputs / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(context.root / relative, destination)
    atomic_write_json(inputs / "runtime.json", {
        key: context.lock[key]
        for key in (
            "backend", "runtime_id", "python", "vllm", "model_path", "model_revision",
            "versions", "host", "ld_library_path", "cuda_home", "nvcc",
            "flashinfer_workspace_base", "torch_extensions_dir", "image_ref", "image_digest",
        )
        if key in context.lock
    })
    atomic_write_json(
        path,
        {**expected, "created_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return path


def _split_sha(context: V2RunContext, split: str) -> str:
    return str(context.dataset_manifest["splits"][split]["sha256"])


def _policy_summary(policy: Policy, path: Path) -> dict[str, Any]:
    return {"policy": policy.record(), "aggregate": aggregate_v2(read_jsonl(path))}


def _bootstrap_seed(config: V2QualityConfig) -> int:
    return int(config.raw["scoring"]["bootstrap_seed"])


def _gap(
    context: V2RunContext, reference_path: Path, candidate_path: Path
) -> Mapping[str, Any]:
    return paired_gap_summary(
        read_jsonl(reference_path),
        read_jsonl(candidate_path),
        bootstrap_samples=context.config.bootstrap_samples,
        bootstrap_seed=_bootstrap_seed(context.config),
    )


def _read_policy_manifest(result_path: Path) -> Mapping[str, Any]:
    path = result_path.with_name(result_path.stem + ".policy-manifest.json")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"policy manifest 无效：{path}")
    return value


def _render_quality_report(context: V2RunContext, selection: Mapping[str, Any]) -> Path:
    report_path = context.root / "runs" / context.run_id / "report/QUALITY-v2.zh-CN.md"
    candidate, final = selection['candidate'], selection['final']
    candidate_text = f"P{candidate['k']}，BF16 层 {candidate['bf16_layers']}" if candidate else '无有效候选'
    final_text = f"P{final['k']}，BF16 层 {final['bf16_layers']}" if final else '基线无效，无法推荐'
    lines = [
        "# AutoKV-Skip v2.1 质量与容量报告", "",
        f"- 运行 ID：`{context.run_id}`",
        f"- 状态：`{selection['status']}`",
        f"- Calibration 候选：{candidate_text}",
        f"- 最终策略：{final_text}",
        f"- 主实验达标：{'是' if selection['technical_goal_passed'] else '否'}",
        f"- 留出集质量达标：{'是' if selection['quality_passed'] else '否'}",
        f"- 容量验收：`{selection['capacity']['status']}`，目标至少 {context.config.min_capacity_ratio:.1f}×",
        "- 实际源码、配置、数据和运行路径副本：`inputs/`",
        "- KV 显存预算：16G；prefix caching 显式关闭；模型权重精度保持原样。",
        "", str(selection['conclusion']), "",
        "`complete` 表示本次流程已给出结果，不等于主实验达标。", "",
    ]
    for title, records in (("Calibration 端点", list(selection['endpoint'].values())), ("Held-out", selection['heldout'])):
        lines.extend([f"## {title}", "", "| 策略 | k | Easy | Hard | Natural | S_v2 |", "|---|---:|---:|---:|---:|---:|"])
        for record in records:
            policy, aggregate = record['policy'], record['aggregate']
            scores = aggregate['scores']
            lines.append(f"| {policy['name']} | {policy['k']} | {scores['easy']:.4f} | {scores['hard']:.4f} | {scores['natural']:.4f} | {aggregate['s_v2']:.4f} |")
        lines.extend(["", "| 策略 | 任务 | 绝对分数 |", "|---|---|---:|"])
        for record in records:
            for task, score in record['aggregate']['task_scores'].items():
                lines.append(f"| {record['policy']['name']} | {task} | {score:.4f} |")
        lines.append("")
    lines.extend(["## 配对差与不确定性", "", "差值方向按名称中的左项减右项；区间用于解释不确定性，不参与运行门禁。", "", "| 比较 | 类别 | 分差 | 95% CI |", "|---|---|---:|---|"])
    for name, differences in selection['paired_differences'].items():
        for tier, value in differences.items():
            lines.append(f"| {name} | {tier} | {value['gap']:+.6f} | [{value['ci95_low']:+.6f}, {value['ci95_high']:+.6f}] |")
    lines.extend(["", "## 质量验收", "", f"全局下降 ≤ {context.config.epsilon_global}，Hard/Natural 各下降 ≤ {context.config.epsilon_tier}；BF16 与候选基础题均须全过。", "", "| 检查 | 通过 |", "|---|---|"])
    for name, passed in selection['heldout_constraints'].get('checks', {}).items():
        lines.append(f"| {name} | {'是' if passed else '否'} |")
    lines.extend(["", "## 已评估候选", "", "只在 calibration 上选择；同预算按总分高、层号小排序。每个层集合在同一 split 只运行一次。", "", "| 策略 | BF16 层 | k | S_v2 | 质量通过 |", "|---|---|---:|---:|---|"])
    for record in selection['candidate_pool']:
        policy = record['policy']
        lines.append(f"| {policy['name']} | {policy['bf16_layers']} | {policy['k']} | {record['aggregate']['s_v2']:.4f} | {'是' if record['constraints']['passed'] else '否'} |")
    lines.extend(["", "## KV 容量", "", "| 策略 | k | bytes/token | 理论倍率 | 实测 tokens | 实测倍率 |", "|---|---:|---:|---:|---:|---:|"])
    for row in selection['capacity_rows']:
        measured = row['measured_tokens'] if row['measured_tokens'] is not None else '未记录'
        ratio = f"{row['measured_ratio_vs_p32']:.4f}×" if row['measured_ratio_vs_p32'] is not None else '未验证'
        lines.append(f"| {row['name']} | {row['k']} | {row['bytes_per_token']} | {row['capacity_ratio_vs_p32']:.4f}× | {measured} | {ratio} |")
    lines.extend(["", "## 结论范围", "", "质量判定仅覆盖这份固定留出集与当前模型/运行时。选择的是已评估候选中的较小 BF16 预算，不保证全局最优。容量指同样 KV 显存预算下的 KV token 数，不等于模型上下文窗口或吞吐倍率。", "", "随机对照使用可选命令 `v2-random-controls`，不影响本报告的主验收；吞吐、TTFT、TPOT/ITL 属于后续性能扩展。", ""])
    atomic_write_text(report_path, "\n".join(lines))
    return report_path


def _write_completed_manifest(context: V2RunContext, selection: Mapping[str, Any]) -> Path:
    run_root = context.root / "runs" / context.run_id
    path = run_root / "completed-manifest.json"
    artifacts = [
        {"path": artifact.relative_to(run_root).as_posix()}
        for artifact in sorted(run_root.rglob("*"))
        if artifact.is_file() and artifact != path
        and "_incomplete" not in artifact.relative_to(run_root).parts
        and not artifact.name.endswith(".working.jsonl")
    ]
    atomic_write_json(path, {
        "schema_version": 2, "experiment_version": "2.1", "complete": True,
        "run_id": context.run_id, "status": selection['status'],
        "technical_goal_passed": selection['technical_goal_passed'],
        "source_tree_sha256": context.source['tree_sha256'],
        "dataset_sha256": context.dataset_manifest['dataset_sha256'],
        "final_policy": selection['final'], "artifacts": artifacts,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return path


def run_v2_pipeline(root: Path, *, port: int = 8000) -> Mapping[str, Any]:
    """用已有候选搜索较小 BF16 预算，主验收只要求质量与实测容量。"""
    context = load_v2_run_context(root)
    _require_linux()
    run_root = context.root / "runs" / context.run_id
    (run_root / "completed-manifest.json").unlink(missing_ok=True)
    runner = V2PolicyRunner(context.config, context.profile, context.root, context.lock, context.run_id, port=port)
    p32, p0 = endpoint_policies(context.config.num_layers)
    # 同一层集合在同一 split 复用；跨 split 必须独立评估。
    results: dict[tuple[str, str], Path] = {}

    def evaluate(policy: Policy, split: str, directory: str) -> Path:
        key = (split, policy.config_id)
        if key not in results:
            results[key] = runner.run_policy(
                policy, context.calibration if split == 'calibration' else context.heldout,
                split=split, split_sha256=_split_sha(context, split), relative_directory=Path(directory),
            )
        return results[key]

    def record(policy: Policy, path: Path) -> dict[str, Any]:
        return {**_policy_summary(policy, path), 'result_path': path.relative_to(context.root).as_posix()}

    selection: dict[str, Any] = {
        'schema_version': 2, 'experiment_version': '2.1', 'run_id': context.run_id,
        'endpoint': {}, 'heldout': [], 'paired_differences': {}, 'candidate_pool': [],
        'group_ranking': [], 'layer_ranking': [], 'budget_trace': [],
        'candidate': None, 'final': None, 'quality_passed': False,
        'heldout_constraints': {'passed': False, 'checks': {}},
        'calibration_decision': 'invalid_reference', 'search_status': 'not_started',
    }
    candidate: Policy | None = None

    def finish(status: str, conclusion: str, final: Policy | None) -> Mapping[str, Any]:
        targets = [p32, p0] + ([candidate] if candidate is not None else [])
        capacities: dict[str, dict[str, Any]] = {}
        for policy in targets:
            if policy.config_id in capacities:
                continue
            paths = [results[(split, policy.config_id)] for split in ('calibration', 'heldout') if (split, policy.config_id) in results]
            if not paths:
                continue
            tokens = None
            for path in paths:
                capacity = _read_policy_manifest(path).get('capacity') or {}
                observed = capacity.get('tokens')
                if isinstance(observed, int) and not isinstance(observed, bool) and observed > 0:
                    tokens = observed
                    break
            capacities[policy.config_id] = {'name': policy.name, 'k': policy.k, **theoretical_capacity(policy), 'measured_tokens': tokens}
        reference_tokens = capacities.get(p32.config_id, {}).get('measured_tokens')
        for row in capacities.values():
            row['measured_ratio_vs_p32'] = row['measured_tokens'] / reference_tokens if reference_tokens and row['measured_tokens'] else None
        candidate_capacity = capacities.get(candidate.config_id, {}) if candidate is not None else {}
        ratio = candidate_capacity.get('measured_ratio_vs_p32')
        capacity_status = 'unverified' if ratio is None else ('passed' if ratio + 1e-12 >= context.config.min_capacity_ratio else 'below_target')
        goal_passed = selection['quality_passed'] and capacity_status == 'passed'
        if status == 'quality_passed':
            status = 'passed' if goal_passed else ('capacity_unverified' if ratio is None else 'capacity_below_target')
            conclusion += (' 实测 KV 容量达到目标。' if goal_passed else ' 实测 KV 容量尚未验证。' if ratio is None else ' 实测 KV 容量未达到目标。')
        selection.update({
            'status': status, 'conclusion': conclusion, 'final': final.record() if final else None,
            'candidate': candidate.record() if candidate else None, 'technical_goal_passed': goal_passed,
            'capacity_rows': list(capacities.values()),
            'capacity': {'status': capacity_status, 'reference_tokens': reference_tokens,
                         'candidate_tokens': candidate_capacity.get('measured_tokens'), 'ratio_vs_p32': ratio,
                         'min_ratio': context.config.min_capacity_ratio},
            'server_starts_this_invocation': runner.server_starts, 'requests_this_invocation': runner.requests,
        })
        selection_path = run_root / 'selection.json'
        atomic_write_json(selection_path, selection)
        atomic_write_json(run_root / 'decision.json', {
            'schema_version': 2, 'run_id': context.run_id, 'decision': selection['calibration_decision'],
            'status': status, 'technical_goal_passed': goal_passed,
        })
        report_path = _render_quality_report(context, selection)
        completed_path = _write_completed_manifest(context, selection)
        return {
            'complete': True, 'run_id': context.run_id, 'status': status,
            'decision': selection['calibration_decision'], 'technical_goal_passed': goal_passed,
            'quality_passed': selection['quality_passed'], 'capacity': selection['capacity'],
            'candidate': selection['candidate'], 'final': selection['final'],
            'selection_path': selection_path.relative_to(context.root).as_posix(),
            'report_path': report_path.relative_to(context.root).as_posix(),
            'completed_manifest_path': completed_path.relative_to(context.root).as_posix(),
            'server_starts_this_invocation': runner.server_starts, 'requests_this_invocation': runner.requests,
        }

    p32_cal = evaluate(p32, 'calibration', 'quality/calibration/endpoints')
    reference = aggregate_v2(read_jsonl(p32_cal))
    selection['endpoint']['p32'] = record(p32, p32_cal)
    if not reference_is_valid(reference):
        return finish('invalid_reference', 'Calibration 的 BF16 基础题未全部通过，当前数据与运行无法用于判断量化质量；已保存结果，未运行 P0 或层搜索。', None)
    p0_cal = evaluate(p0, 'calibration', 'quality/calibration/endpoints')
    selection['endpoint']['p0'] = record(p0, p0_cal)
    p0_aggregate = selection['endpoint']['p0']['aggregate']
    selection['endpoint_constraints'] = quality_constraints(reference, p0_aggregate, context.config, endpoint=True)
    selection['paired_differences']['calibration_P32-P0'] = _gap(context, p32_cal, p0_cal)
    selection['calibration_decision'] = 'no_quality_gap' if selection['endpoint_constraints']['passed'] else 'search_required'
    candidate = p0

    if selection['calibration_decision'] == 'search_required':
        pool: dict[str, tuple[Policy, dict[str, Any]]] = {}

        def consider(policy: Policy, directory: str) -> dict[str, Any]:
            path = evaluate(policy, 'calibration', directory)
            if policy.config_id not in pool:
                item = record(policy, path)
                item['constraints'] = quality_constraints(reference, item['aggregate'], context.config, endpoint=False)
                pool[policy.config_id] = policy, item
                selection['candidate_pool'].append(item)
            return pool[policy.config_id][1]

        def best() -> Policy | None:
            valid = [(policy, item) for policy, item in pool.values() if item['constraints']['passed']]
            return min(valid, key=lambda pair: (pair[0].k, -pair[1]['aggregate']['s_v2'], pair[0].bf16_layers))[0] if valid else None

        group_scores = {}
        for policy in group_policies(context.config.num_layers, context.config.group_size):
            group_scores[policy] = consider(policy, 'quality/calibration/groups')['aggregate']['s_v2']
        selection['group_ranking'] = list(rank_by_recovery(group_scores, p0_aggregate['s_v2']))
        top_groups = sorted(group_scores, key=lambda policy: (-group_scores[policy], policy.bf16_layers))[:context.config.top_groups]
        layer_scores = {}
        for policy in layer_policies(top_groups):
            layer_scores[policy] = consider(policy, 'quality/calibration/layers')['aggregate']['s_v2']
        selection['layer_ranking'] = list(rank_by_recovery(layer_scores, p0_aggregate['s_v2']))
        ordered_layers = [policy.bf16_layers[0] for policy in sorted(layer_scores, key=lambda policy: (-layer_scores[policy], policy.bf16_layers))]
        for k in (2, 4, 8):
            current = best()
            if current is not None and current.k <= k:
                break
            policy = nested_budget_policy(ordered_layers, k, context.config.num_layers)
            item = consider(policy, 'quality/calibration/budgets')
            selection['budget_trace'].append(item)
        candidate = best() or p32
        selection['search_status'] = 'completed'
    else:
        selection['search_status'] = 'skipped_no_quality_gap'
    selection['candidate'] = candidate.record()

    p32_held = evaluate(p32, 'heldout', 'quality/heldout/endpoints')
    selection['heldout'].append(record(p32, p32_held))
    reference_held = selection['heldout'][0]['aggregate']
    if not reference_is_valid(reference_held):
        return finish('invalid_reference', 'Held-out 的 BF16 基础题未全部通过，候选质量无法验证；保留 calibration 选择，不利用 held-out 重选。', None)
    p0_held = evaluate(p0, 'heldout', 'quality/heldout/endpoints')
    selection['heldout'].append(record(p0, p0_held))
    selection['paired_differences']['heldout_P32-P0'] = _gap(context, p32_held, p0_held)
    selection['p0_heldout_constraints'] = quality_constraints(reference_held, selection['heldout'][1]['aggregate'], context.config, endpoint=True)
    candidate_held = evaluate(candidate, 'heldout', 'quality/heldout/selected')
    if candidate.k not in {0, 32}:
        selection['heldout'].append(record(candidate, candidate_held))
        selection['paired_differences']['heldout_P32-candidate'] = _gap(context, p32_held, candidate_held)
        selection['paired_differences']['heldout_candidate-P0'] = _gap(context, candidate_held, p0_held)
    constraints = quality_constraints(reference_held, aggregate_v2(read_jsonl(candidate_held)), context.config, endpoint=candidate.k == 0)
    selection['heldout_constraints'] = constraints
    selection['quality_passed'] = constraints['passed']
    if not constraints['passed']:
        return finish('heldout_failed', 'Calibration 候选未通过 held-out，安全回退 P32；本次技术目标未达成，不重新选层或改预算。', p32)
    if candidate.k == 32:
        return finish('no_qualifying_mixed_policy', '已评估的低 BF16 预算候选均未通过 calibration，回退 P32；本次没有取得容量收益。', p32)
    conclusion = f'P{candidate.k} 在 calibration 与 held-out 满足预定质量约束。'
    if candidate.k > 0 and selection['p0_heldout_constraints']['passed']:
        conclusion += ' 留出集上的 P0 也满足约束，混合方案的必要性未得到独立支持；不据此重新选择。'
    return finish('quality_passed', conclusion, candidate)


def run_v2_random_controls(root: Path, *, port: int = 8000) -> Mapping[str, Any]:
    """主实验之后的可选分析；不改候选、主验收或完成状态。"""
    context = load_v2_run_context(root)
    _require_linux()
    run_root = context.root / 'runs' / context.run_id
    selection = read_json(run_root / 'selection.json')
    if not (run_root / 'completed-manifest.json').is_file():
        raise ValueError('请先完成当前配置的 v2-run 主实验')
    chosen = selection.get('candidate')
    if not chosen or chosen['k'] not in {1, 2, 4, 8} or not selection['quality_passed']:
        return {'complete': True, 'run_id': context.run_id, 'skipped': True, 'reason': '没有通过留出集的中间候选，无需随机对照'}
    selected = Policy(chosen['name'], tuple(chosen['bf16_layers']), context.config.num_layers)
    selected_record = next(item for item in selection['heldout'] if item['policy']['config_id'] == selected.config_id)
    selected_path = context.root / selected_record['result_path']
    runner = V2PolicyRunner(context.config, context.profile, context.root, context.lock, context.run_id, port=port)
    output_path = run_root / 'random-controls.json'
    atomic_write_json(output_path, {'complete': False, 'run_id': context.run_id, 'candidate': chosen})
    records = []
    for policy in random_control_policies(selected, context.config.random_seeds):
        path = runner.run_policy(policy, context.heldout, split='heldout', split_sha256=_split_sha(context, 'heldout'), relative_directory=Path('quality/heldout/random'))
        records.append({**_policy_summary(policy, path), 'candidate_minus_random': _gap(context, selected_path, path)})
    median = statistics.median(item['aggregate']['s_v2'] for item in records)
    result = {
        'complete': True, 'run_id': context.run_id, 'candidate': chosen,
        'controls': records, 'random_s_v2_median': median,
        'observed_above_random_median': selected_record['aggregate']['s_v2'] > median,
        'conclusion': '中位数比较仅描述本次观察；需结合配对差和区间解释，不自动宣称层排序有效，不影响主实验验收。',
        'server_starts_this_invocation': runner.server_starts, 'requests_this_invocation': runner.requests,
    }
    atomic_write_json(output_path, result)
    lines = ['# v2.1 可选随机对照', '', result['conclusion'], '', f"所选策略 S_v2={selected_record['aggregate']['s_v2']:.6f}；三组随机中位数={median:.6f}。", '', '| 随机策略 | 类别 | 候选减随机 | 95% CI |', '|---|---|---:|---|']
    for item in records:
        for tier, value in item['candidate_minus_random'].items():
            lines.append(f"| {item['policy']['name']} | {tier} | {value['gap']:+.6f} | [{value['ci95_low']:+.6f}, {value['ci95_high']:+.6f}] |")
    atomic_write_text(run_root / 'report/RANDOM-v2.zh-CN.md', '\n'.join(lines) + '\n')
    return result
