"""AutoKV-Skip v2.0 阶段 3–4 的条件式端到端编排。"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokv.commands import run_command, runtime_identity
from autokv.config import Profile, load_profile
from autokv.doctor import DoctorError, parse_gpu_csv, validate_host
from autokv.io import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
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
from autokv.v2_metrics import aggregate_v2, paired_gap_summary, quality_constraints
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
    lock = _load_v2_lock(root, profile, config)
    if lock.get("backend") != "local_vllm":
        raise ValueError("v2 pilot 当前只支持项目已验证的 local_vllm 环境")
    source = _source_identity(root)
    if source.get("git_commit") is None or source.get("git_dirty") is not False:
        raise ValueError("pilot 要求 autokv/scripts/configs/pyproject 已提交且干净")
    _assert_linux_a6000(profile)
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
                    sha256_file(config_path),
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


def _load_v2_lock(
    root: Path, profile: Profile, config: V2QualityConfig
) -> Mapping[str, Any]:
    path = root / "runs" / "_environment" / "lock.json"
    if not path.is_file():
        raise ValueError("缺少已有本地 vLLM 环境锁：runs/_environment/lock.json")
    value = read_json(path)
    # 复用 v1 已验证的锁结构，不创建 v2 doctor/lock 链。
    from autokv.cli import _validate_lock

    lock = _validate_lock(value, profile)
    if lock.get("model_revision") != config.model_revision:
        raise ValueError("环境锁的模型 revision 与 v2 冻结配置不一致")
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


def _assert_linux_a6000(profile: Profile) -> None:
    if not sys.platform.startswith("linux"):
        raise ValueError("v2 GPU 运行只能在目标 Linux A6000 服务器执行")
    result = run_command(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ),
        timeout=30,
    )
    if not result.ok:
        raise DoctorError("nvidia-smi 只读查询失败")
    facts = parse_gpu_csv(result.stdout)
    failures = [
        gate for gate in validate_host(facts, profile.hardware.driver) if not gate.ok
    ]
    if failures:
        details = "; ".join(
            f"{gate.name}={gate.observed}, expected={gate.expected}"
            for gate in failures
        )
        raise DoctorError(f"目标服务器身份不匹配：{details}")


def load_v2_run_context(
    root: Path, *, require_clean_source: bool = True
) -> V2RunContext:
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
    lock = _load_v2_lock(root, profile, config)
    dataset_manifest, calibration, heldout = load_frozen_v2_dataset(
        config,
        root / V2_DATA_RELATIVE_ROOT,
        config_path=config_path,
    )
    source = _source_identity(root)
    if require_clean_source and (
        source.get("git_commit") is None or source.get("git_dirty") is not False
    ):
        raise ValueError(
            "正式 v2 运行要求 autokv/scripts/configs/pyproject 已提交且干净"
        )
    run_id = _v2_run_id(
        sha256_file(config_path),
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
        "run_id": context.run_id,
        "git_commit": context.source.get("git_commit"),
        "source_tree_sha256": context.source["tree_sha256"],
        "source_files": context.source["files"],
        "config_path": V2_CONFIG_RELATIVE_PATH.as_posix(),
        "config_sha256": sha256_file(context.config_path),
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
            observed.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("已有 v2 run manifest 与当前冻结输入不一致")
        return path
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


def _render_quality_report(
    context: V2RunContext,
    selection: Mapping[str, Any],
    heldout_summaries: Sequence[Mapping[str, Any]],
    capacity_rows: Sequence[Mapping[str, Any]],
) -> Path:
    report_path = (
        context.root / "runs" / context.run_id / "report" / "QUALITY-v2.zh-CN.md"
    )
    lines = [
        "# AutoKV-Skip v2.0 质量选择报告",
        "",
        f"- 运行 ID：`{context.run_id}`",
        f"- 源码提交：`{context.source.get('git_commit')}`",
        f"- 数据身份：`{context.dataset_manifest['dataset_sha256']}`",
        "- Prefix caching：关闭（每个策略的实际参数与日志均已验证）",
        f"- Calibration 决策：`{selection['calibration_decision']}`",
        f"- Calibration 候选：`{selection['candidate']['name']}`",
        f"- 最终策略 P*：`{selection['final']['name']}`",
        "",
        "## Calibration 端点",
        "",
        "| 策略 | S_easy | S_hard | S_natural | S_v2 |",
        "|---|---:|---:|---:|---:|",
    ]
    endpoint = selection["endpoint"]
    for name in ("p32", "p0"):
        aggregate = endpoint[name]["aggregate"]
        scores = aggregate["scores"]
        lines.append(
            f"| {name} | {scores['easy']:.4f} | {scores['hard']:.4f} | "
            f"{scores['natural']:.4f} | {aggregate['s_v2']:.4f} |"
        )
    gap = endpoint["gap_p32_minus_p0"]
    lines.extend(
        [
            "",
            "| P32 − P0 缺口 | 点估计 | 95% CI |",
            "|---|---:|---:|",
            *[
                f"| {name} | {gap[name]['gap']:.4f} | "
                f"[{gap[name]['ci95_low']:.4f}, {gap[name]['ci95_high']:.4f}] |"
                for name in ("easy", "hard", "natural", "global")
            ],
            "",
            "## 搜索轨迹",
            "",
        ]
    )
    if selection["search_status"] == "skipped_no_quality_gap":
        lines.append("端点满足质量约束，组、单层和预算搜索均未运行。")
    else:
        group_text = "，".join(
            f"{item['policy']['name']}({item['recovery_vs_p0']:+.4f})"
            for item in selection["group_ranking"]
        )
        layer_text = "，".join(
            f"L{item['policy']['bf16_layers'][0]}({item['recovery_vs_p0']:+.4f})"
            for item in selection["layer_ranking"]
        )
        budget_text = "，".join(
            f"P{item['policy']['k']}={'通过' if item['constraints']['passed'] else '未通过'}"
            for item in selection["budget_trace"]
        )
        lines.extend(
            [
                f"- 八组恢复量排名：{group_text}",
                f"- 八个候选层恢复量排名：{layer_text}",
                f"- 预算早停轨迹：{budget_text}",
            ]
        )
    lines.extend(
        [
            "",
            "## Held-out 质量",
            "",
            "| 策略 | k | S_easy | S_hard | S_natural | S_v2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in heldout_summaries:
        policy = item["policy"]
        aggregate = item["aggregate"]
        scores = aggregate["scores"]
        lines.append(
            f"| {policy['name']} | {policy['k']} | {scores['easy']:.4f} | "
            f"{scores['hard']:.4f} | {scores['natural']:.4f} | {aggregate['s_v2']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Held-out 约束：{'通过' if selection['heldout_constraints']['passed'] else '未通过'}；"
            f"Random-k 的 S_v2 中位数：{selection['random_s_v2_median'] if selection['random_s_v2_median'] is not None else '不适用'}。",
            "",
            "## 容量",
            "",
            "| 策略 | k | 理论 bytes/token | 理论容量 / P32 | vLLM 实测 tokens |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in capacity_rows:
        lines.append(
            f"| {row['name']} | {row['k']} | {row['bytes_per_token']} | "
            f"{row['capacity_ratio_vs_p32']:.4f}× | {row['measured_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            str(selection["conclusion"]),
            "",
            "本报告只覆盖质量选择与容量。吞吐、TTFT、TPOT 和 ITL 属于后续阶段 5，"
            "不能由本报告推断。Answer NLL 在当前 API 路径不可无额外请求地取得，因此记录为 null，"
            "且从未进入 S_v2。",
            "",
        ]
    )
    atomic_write_text(report_path, "\n".join(lines))
    return report_path


def _write_completed_manifest(context: V2RunContext, final: Policy) -> Path:
    run_root = context.root / "runs" / context.run_id
    path = run_root / "completed-manifest.json"
    artifacts: list[dict[str, str]] = []
    for artifact in sorted(run_root.rglob("*")):
        if (
            not artifact.is_file()
            or artifact == path
            or "_incomplete" in artifact.relative_to(run_root).parts
            or artifact.name.endswith(".working.jsonl")
        ):
            continue
        artifacts.append(
            {
                "path": artifact.relative_to(run_root).as_posix(),
                "sha256": sha256_file(artifact),
            }
        )
    atomic_write_json(
        path,
        {
            "schema_version": 2,
            "complete": True,
            "run_id": context.run_id,
            "source_tree_sha256": context.source["tree_sha256"],
            "dataset_sha256": context.dataset_manifest["dataset_sha256"],
            "final_policy": final.record(),
            "artifacts": artifacts,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def run_v2_pipeline(root: Path, *, port: int = 8000) -> Mapping[str, Any]:
    """严格执行端点判断；只有有缺口时才做 coarse-to-fine 搜索。"""

    context = load_v2_run_context(root)
    _assert_linux_a6000(context.profile)
    runner = V2PolicyRunner(
        context.config,
        context.profile,
        context.root,
        context.lock,
        context.run_id,
        port=port,
    )
    p32, p0 = endpoint_policies(context.config.num_layers)
    calibration_sha = _split_sha(context, "calibration")
    heldout_sha = _split_sha(context, "heldout")
    calibration_endpoint_dir = Path("quality/calibration/endpoints")
    p32_cal = runner.run_policy(
        p32,
        context.calibration,
        split="calibration",
        split_sha256=calibration_sha,
        relative_directory=calibration_endpoint_dir,
    )
    p0_cal = runner.run_policy(
        p0,
        context.calibration,
        split="calibration",
        split_sha256=calibration_sha,
        relative_directory=calibration_endpoint_dir,
    )
    p32_cal_aggregate = aggregate_v2(read_jsonl(p32_cal))
    p0_cal_aggregate = aggregate_v2(read_jsonl(p0_cal))
    endpoint_constraints = quality_constraints(
        p32_cal_aggregate, p0_cal_aggregate, context.config, endpoint=True
    )
    endpoint_gap = _gap(context, p32_cal, p0_cal)
    endpoint_summary = {
        "schema_version": 2,
        "p32": {"policy": p32.record(), "aggregate": p32_cal_aggregate},
        "p0": {"policy": p0.record(), "aggregate": p0_cal_aggregate},
        "gap_p32_minus_p0": endpoint_gap,
        "constraints": endpoint_constraints,
    }
    endpoint_summary_path = (
        context.root
        / "runs"
        / context.run_id
        / "quality"
        / "calibration"
        / "endpoint-summary.json"
    )
    atomic_write_json(endpoint_summary_path, endpoint_summary)
    calibration_decision = (
        "no_quality_gap" if endpoint_constraints["passed"] else "search_required"
    )
    failed_checks = [
        name for name, passed in endpoint_constraints["checks"].items() if not passed
    ]
    decision = {
        "schema_version": 2,
        "run_id": context.run_id,
        "dataset_sha256": context.dataset_manifest["dataset_sha256"],
        "decision": calibration_decision,
        "reason_code": (
            "p0_within_frozen_quality_bounds"
            if endpoint_constraints["passed"]
            else "p0_failed_" + "_and_".join(failed_checks)
        ),
        "gap_p32_minus_p0": endpoint_gap,
        "thresholds": endpoint_constraints["thresholds"],
        "failed_checks": failed_checks,
    }
    atomic_write_json(
        context.root / "runs" / context.run_id / "decision.json", decision
    )

    group_ranking: tuple[dict[str, object], ...] = ()
    layer_ranking: tuple[dict[str, object], ...] = ()
    budget_trace: list[dict[str, Any]] = []
    candidate = p0
    candidate_cal_path = p0_cal

    if calibration_decision == "search_required":
        group_scores: dict[Policy, float] = {}
        for policy in group_policies(
            context.config.num_layers, context.config.group_size
        ):
            path = runner.run_policy(
                policy,
                context.calibration,
                split="calibration",
                split_sha256=calibration_sha,
                relative_directory=Path("quality/calibration/groups"),
            )
            group_scores[policy] = float(aggregate_v2(read_jsonl(path))["s_v2"])
        group_ranking = rank_by_recovery(group_scores, float(p0_cal_aggregate["s_v2"]))
        ordered_groups = sorted(
            group_scores,
            key=lambda policy: (
                -(group_scores[policy] - float(p0_cal_aggregate["s_v2"])),
                policy.bf16_layers,
            ),
        )
        top_groups = ordered_groups[: context.config.top_groups]

        layer_scores: dict[Policy, float] = {}
        for policy in layer_policies(top_groups):
            path = runner.run_policy(
                policy,
                context.calibration,
                split="calibration",
                split_sha256=calibration_sha,
                relative_directory=Path("quality/calibration/layers"),
            )
            layer_scores[policy] = float(aggregate_v2(read_jsonl(path))["s_v2"])
        layer_ranking = rank_by_recovery(layer_scores, float(p0_cal_aggregate["s_v2"]))
        ordered_layers = [
            policy.bf16_layers[0]
            for policy in sorted(
                layer_scores,
                key=lambda policy: (
                    -(layer_scores[policy] - float(p0_cal_aggregate["s_v2"])),
                    policy.bf16_layers,
                ),
            )
        ]
        candidate = p32
        candidate_cal_path = p32_cal
        for k in (2, 4, 8):
            policy = nested_budget_policy(ordered_layers, k, context.config.num_layers)
            path = runner.run_policy(
                policy,
                context.calibration,
                split="calibration",
                split_sha256=calibration_sha,
                relative_directory=Path("quality/calibration/budgets"),
            )
            aggregate = aggregate_v2(read_jsonl(path))
            constraints = quality_constraints(
                p32_cal_aggregate, aggregate, context.config, endpoint=False
            )
            budget_trace.append(
                {
                    "policy": policy.record(),
                    "aggregate": aggregate,
                    "constraints": constraints,
                }
            )
            if constraints["passed"]:
                candidate = policy
                candidate_cal_path = path
                break

    heldout_endpoint_dir = Path("quality/heldout/endpoints")
    p32_held = runner.run_policy(
        p32,
        context.heldout,
        split="heldout",
        split_sha256=heldout_sha,
        relative_directory=heldout_endpoint_dir,
    )
    p0_held = runner.run_policy(
        p0,
        context.heldout,
        split="heldout",
        split_sha256=heldout_sha,
        relative_directory=heldout_endpoint_dir,
    )
    p32_held_aggregate = aggregate_v2(read_jsonl(p32_held))
    p0_held_aggregate = aggregate_v2(read_jsonl(p0_held))
    heldout_records: list[tuple[Policy, Path]] = [(p32, p32_held), (p0, p0_held)]
    random_records: list[tuple[Policy, Path]] = []
    random_median: float | None = None
    layer_selection_supported: bool | None = None

    if candidate.k in {2, 4, 8}:
        selected_held = runner.run_policy(
            candidate,
            context.heldout,
            split="heldout",
            split_sha256=heldout_sha,
            relative_directory=Path("quality/heldout/selected"),
        )
        heldout_records.append((candidate, selected_held))
        for random_policy in random_control_policies(
            candidate, context.config.random_seeds
        ):
            path = runner.run_policy(
                random_policy,
                context.heldout,
                split="heldout",
                split_sha256=heldout_sha,
                relative_directory=Path("quality/heldout/random"),
            )
            random_records.append((random_policy, path))
            heldout_records.append((random_policy, path))
        candidate_held_aggregate = aggregate_v2(read_jsonl(selected_held))
        heldout_constraints = quality_constraints(
            p32_held_aggregate,
            candidate_held_aggregate,
            context.config,
            endpoint=False,
        )
        random_median = statistics.median(
            float(aggregate_v2(read_jsonl(path))["s_v2"]) for _, path in random_records
        )
        layer_selection_supported = (
            float(candidate_held_aggregate["s_v2"]) > random_median
        )
        final = candidate if heldout_constraints["passed"] else p32
    elif candidate.k == 0:
        heldout_constraints = quality_constraints(
            p32_held_aggregate, p0_held_aggregate, context.config, endpoint=True
        )
        final = p0 if heldout_constraints["passed"] else p32
    else:
        heldout_constraints = quality_constraints(
            p32_held_aggregate, p32_held_aggregate, context.config, endpoint=False
        )
        final = p32

    if candidate.k == 0 and final.k == 0:
        conclusion = "P0 在 calibration 与 held-out 均满足冻结质量约束；自动选择器正确停止，未进行层搜索。"
    elif candidate.k in {2, 4, 8} and final == candidate:
        conclusion = f"P{candidate.k} 在 held-out 满足质量约束；" + (
            "且高于三组同预算随机策略中位数，层排序得到支持。"
            if layer_selection_supported
            else "但未高于三组同预算随机策略中位数，只能支持预算选择，不能声称层排序有附加价值。"
        )
    elif candidate.k in {0, 2, 4, 8} and final.k == 32:
        conclusion = "Calibration 候选未通过 held-out，最终安全回退 P32；不得用 held-out 重新选层。"
    else:
        conclusion = (
            "P2、P4、P8 均未在 calibration 满足质量约束，候选空间内只有 P32 通过。"
        )

    selection = {
        "schema_version": 2,
        "run_id": context.run_id,
        "calibration_decision": calibration_decision,
        "endpoint": endpoint_summary,
        "candidate": candidate.record(),
        "final": final.record(),
        "group_ranking": list(group_ranking),
        "layer_ranking": list(layer_ranking),
        "budget_trace": budget_trace,
        "heldout_constraints": heldout_constraints,
        "random_s_v2_median": random_median,
        "layer_selection_supported": layer_selection_supported,
        "search_status": (
            "skipped_no_quality_gap"
            if calibration_decision == "no_quality_gap"
            else "completed"
        ),
        "server_starts_this_invocation": runner.server_starts,
        "requests_this_invocation": runner.requests,
        "conclusion": conclusion,
    }
    selection_path = context.root / "runs" / context.run_id / "selection.json"
    atomic_write_json(selection_path, selection)

    heldout_summaries = [
        _policy_summary(policy, path) for policy, path in heldout_records
    ]
    capacity_records: list[dict[str, Any]] = []
    capacity_inputs: list[tuple[Policy, Path]] = [(p32, p32_cal), (p0, p0_cal)]
    if candidate.k not in {0, 32}:
        capacity_inputs.append((candidate, candidate_cal_path))
    seen_capacity: set[str] = set()
    for policy, result_path in capacity_inputs:
        if policy.config_id in seen_capacity:
            continue
        seen_capacity.add(policy.config_id)
        theoretical = theoretical_capacity(policy)
        manifest = _read_policy_manifest(result_path)
        capacity_records.append(
            {
                "name": policy.name,
                "k": policy.k,
                **theoretical,
                "measured_tokens": manifest["capacity"]["tokens"],
            }
        )
    report_path = _render_quality_report(
        context, selection, heldout_summaries, capacity_records
    )
    completed_path = _write_completed_manifest(context, final)
    return {
        "complete": True,
        "run_id": context.run_id,
        "decision": calibration_decision,
        "candidate": candidate.record(),
        "final": final.record(),
        "selection_path": selection_path.relative_to(context.root).as_posix(),
        "report_path": report_path.relative_to(context.root).as_posix(),
        "completed_manifest_path": completed_path.relative_to(context.root).as_posix(),
        "server_starts_this_invocation": runner.server_starts,
        "requests_this_invocation": runner.requests,
    }
