"""v4 构造成功后接续既有选层算法；同一运行保存全部阶段。"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from autokv.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from autokv.v2_data import TransformersPromptCodec
from autokv.v2_policy import endpoint_policies
from autokv.v3_config import identity
from autokv.v3_pipeline import coverage, execute
from autokv.v3_runtime import V3PolicyRunner, load_environment, recover_rows, utc_now
from autokv.v4_config import CONFIG_PATH, V4Config, load_config
from autokv.v4_construction import construct, register, render_construction
from autokv.v4_data import GENERATOR_RULES, data_manifest, ensure_samples, source_pool, validate_samples
from autokv.v4_metrics import paired_summary, score_output


def source_inputs(directory, config):
    manifest = read_json(directory / "source-manifest.json")
    records = {r["name"]: r for r in manifest["sources"]}
    files = []
    for name in config.raw["data"]["sources"]:
        if name not in records:
            raise ValueError(f"来源清单缺少 {name}")
        relative = Path(records[name]["file"])
        path = (directory / relative).resolve()
        if relative.is_absolute() or directory.resolve() not in path.parents:
            raise ValueError("来源文件必须位于离线来源目录内")
        files.append({"name": name, "file": relative.as_posix(), "sha256": sha256_file(path)})
    return {"manifest": manifest, "files": files}


def actual_code():
    root = Path(__file__).resolve().parents[1]
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in sorted((root / "autokv").glob("*.py"))}


def prepare_run(root, config, environment, codec, source_dir, source_info, *, run_id=None):
    code = actual_code()
    inputs_id = identity({"config": config.raw, "sources": source_info.get("files", source_info), "runtime": environment,
                          "code": code, "template": codec.template_text})
    directory = root / "runs" / (run_id or "v4-"+inputs_id[:16])
    path = directory / "run-manifest.json"
    if path.exists():
        if read_json(path)["inputs_id"] != inputs_id:
            raise ValueError("既有运行的实际代码、模型、环境或输入已经改变；恢复原输入后续跑，或不指定 run-id 开始新运行")
        return directory
    if run_id is not None:
        raise ValueError(f"找不到既有运行 {run_id}")
    inputs = directory / "inputs"
    for relative, content in code.items():
        atomic_write_text(inputs / relative, content)
    for relative in ("scripts", "docs/v4.0"):
        if (root / relative).exists():
            shutil.copytree(root / relative, inputs / relative, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in ("pyproject.toml", "CONTEXT.md", "README.md", "data/v4.0/README.md", "data/v3.0/README.md"):
        if (root / relative).exists():
            destination = inputs / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, destination)
    atomic_write_json(inputs / CONFIG_PATH, config.raw)
    atomic_write_json(inputs / "runtime.json", environment)
    atomic_write_json(inputs / "source-manifest.json", source_info)
    atomic_write_json(inputs / "generator-rules.json", GENERATOR_RULES)
    atomic_write_text(inputs / "chat-template.txt", codec.template_text)
    atomic_write_json(path, {"schema_version": 4, "run_id": directory.name, "inputs_id": inputs_id,
                            "created_at": utc_now(), "config": config.raw, "runtime": environment,
                            "source_dir": str(source_dir), "source_info": source_info,
                            "state": "constructing", "complete": False,
                            "source": "实际源码、配置、来源说明见 inputs；构造实例见 construction；Git 不是运行条件"})
    return directory


def finish_flags(result, construction, config, runner):
    result.update(construction_passed=construction["construction_passed"],
                  construction_status=construction["status"], data_conditions_reproduced=None,
                  recovery_demonstrated=False, test_data_conditions=None)
    if "test" not in result.get("scores", {}):
        return
    p32, p0 = endpoint_policies(config.num_layers)
    reference = list(runner.cached(p32, "test").values())
    fp8 = list(runner.cached(p0, "test").values())
    summary = paired_summary(reference, fp8, config)
    result.update(test_data_conditions=summary, data_conditions_reproduced=summary["passed"])
    candidate = result.get("candidate")
    mixed = candidate is not None and 0 < len(candidate["bf16_layers"]) < config.num_layers
    result["recovery_demonstrated"] = bool(construction["construction_passed"] and summary["passed"]
                                           and result["technical_goal_passed"] and mixed)
    # 一任务一长度；全配对差相同的 bootstrap 没有可估计的变异。
    for name, interval in result.get("intervals", {}).items():
        if interval["intervals"] and all(a == b for a, b in interval["intervals"].values()):
            interval.update(intervals=None, interval_status="degenerate",
                            note="配对重采样退化，不能可靠估计总体不确定性；不能据此声称误差为零或证明非劣性")


def execute_v4(config, root, directory, codec, get_sources, runner, source_info, *, construction_only=False, progress=None):
    completed = directory / "completed-manifest.json"
    if completed.exists():
        previous = read_json(completed)
        if previous.get("complete") and previous.get("stage") == "quality":
            return previous
    result = {"schema_version": 4, "run_id": directory.name, "stage": "construction", "complete": False,
              "status": "constructing", "construction_passed": False, "quality_passed": False,
              "technical_goal_passed": False, "data_conditions_reproduced": None, "recovery_demonstrated": False}
    delegated = False
    try:
        construction = construct(config, directory, codec, get_sources, runner, source_info, progress)
        result.update(status=construction["status"], construction_passed=construction["construction_passed"],
                      complete=True, construction_report=str(directory / "report/CONSTRUCTION-v4.zh-CN.md"))
        if not construction["construction_passed"] or construction_only:
            # 已进入正式阶段后再要求只看构造，不覆盖正式阶段的断点诊断。
            if completed.exists() and read_json(completed).get("stage") == "quality":
                return read_json(completed)
            return result
        result.update(stage="data", complete=False, status="generating_formal_data")
        chosen = construction["selected_rule"]["condition"]
        config = config.for_condition(chosen, data_directory=f"data/v4.0/{directory.name}")
        runner.config = config
        data_dir = directory / "inputs" / config.data_root / "quality"
        splits = {}
        for split in ("experiment", "test"):
            specs = [{"split": split, "condition": chosen, "batch_id": split, "count": config.per_cell(split)}]
            rows, _ = ensure_samples(data_dir / f"{split}.jsonl", specs, config, codec, get_sources, progress=progress)
            register(runner, split, rows, config)
            splits[split] = rows
        # 包括已完成构造输入，检验四阶段不复用实例/背景；不读取构造分数来筛正式题。
        all_splits = {s: recover_rows(directory / "construction" / f"{s}.jsonl") for s in ("discovery", "confirmation")}
        validate_samples(config, {**all_splits, **splits})
        manifest = data_manifest(config, splits, source_info, construction["selected_rule"])
        manifest["template_sha256"] = codec.template_sha256
        atomic_write_json(data_dir / "dataset-manifest.json", manifest)
        shutil.copytree(data_dir, root / config.data_root / "quality", dirs_exist_ok=True)
        run_path = directory / "run-manifest.json"
        if run_path.exists():
            run = read_json(run_path)
            run.update(dataset_id=manifest["dataset_id"], selected_condition=chosen, state="quality")
            atomic_write_json(run_path, run)

        def finalize(quality):
            finish_flags(quality, construction, config, runner)
            quality["construction_report"] = str(directory / "report/CONSTRUCTION-v4.zh-CN.md")
            render_construction(directory, config, construction, runner.statistics())

        delegated = True
        return execute(config, splits, runner, directory, finalize_result=finalize)
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "runtime_failed",
                      complete=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if not delegated:
            existing = read_json(completed) if completed.exists() else {}
            if not (construction_only and existing.get("stage") == "quality"):
                result.update(cost=runner.statistics(), finished_at=utc_now())
                try:
                    result["coverage"] = coverage(directory, config)
                except (ValueError, OSError, KeyError) as exc:
                    result["report_error"] = str(exc)
                    if result["complete"]:
                        result.update(complete=False, technical_goal_passed=False, status="runtime_failed")
                atomic_write_json(completed, result)
                path = directory / "run-manifest.json"
                if path.exists():
                    manifest = read_json(path)
                    manifest.update(state=result["status"], complete=result["complete"], updated_at=utc_now())
                    atomic_write_json(path, manifest)


def run_pipeline(root, *, source_dir=None, port=8010, run_id=None, construction_only=False):
    root = Path(root).resolve()
    if run_id is not None and not re.fullmatch(r"v4-[A-Za-z0-9_-]+", run_id):
        raise ValueError("请输入 v4 运行 ID")
    saved = read_json(root / "runs" / run_id / "run-manifest.json") if run_id else None
    config = V4Config.from_dict(saved["config"]) if saved else load_config(root)
    if not sys.platform.startswith("linux"):
        raise ValueError("真实 v4 GPU 实验请在 Linux 服务器执行；本地模拟不能替代 GPU 结果")
    environment = load_environment(root, config, observe_runtime=True)
    codec = TransformersPromptCodec(Path(environment["model_path"]))
    environment["chat_template"] = codec.template_text
    source_dir = Path(source_dir or (saved["source_dir"] if saved else "data/v3.0/source"))
    source_dir = (root / source_dir).resolve()
    # 续跑优先已保存输入；只有补生成题时才需要再次读取原始来源。
    info = saved["source_info"] if saved else source_inputs(source_dir, config)
    directory = prepare_run(root, config, environment, codec, source_dir, info, run_id=run_id)
    print(f"运行 ID：{directory.name}；输入与诊断目录：{directory}", file=sys.stderr, flush=True)
    loaded = None

    def get_sources():
        nonlocal loaded
        if loaded is None:
            if source_inputs(source_dir, config).get("files") != info.get("files"):
                raise ValueError("离线来源已改变；续跑需恢复原来源包或指定其新路径")
            loaded = source_pool(source_dir, config)
            atomic_write_json(directory / "inputs/source-allocation.json", {"counts": loaded["allocation"], "pools": loaded["pools"]})
        return loaded

    runner = V3PolicyRunner(config, root, directory, environment, {}, port=port, scorer=score_output)
    return execute_v4(config, root, directory, codec, get_sources, runner, info, construction_only=construction_only,
                      progress=lambda message: print(message, file=sys.stderr, flush=True))
