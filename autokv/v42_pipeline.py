"""v4.2 直接导入 v4.1 已生成的数据，不重新构造，也不导入旧策略回答。"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from autokv.io import atomic_write_json, atomic_write_text, read_json, sha256_file
from autokv.v3_pipeline import execute
from autokv.v3_runtime import V3PolicyRunner
from autokv.v4_data import validate_samples
from autokv.v4_metrics import score_output
from autokv.v4_pipeline import finish_flags, prepare_run


def describe_source(source, config, codec):
    source = Path(source).resolve()
    old = read_json(source / "run-manifest.json")
    previous = old["config"]
    if previous.get("experiment_version") != "4.1":
        raise ValueError("--reuse-run 必须指向 v4.1 原始运行目录")
    for key in ("model", "data", "construction", "scoring"):
        if previous[key] != config.raw[key]:
            raise ValueError(f"复用数据要求 {key} 与 v4.1 保持一致，不重新抽题或改变评分")
    folder = source / "inputs/data/v4.1" / source.name / "quality"
    manifest = read_json(folder / "dataset-manifest.json")
    construction = read_json(source / "construction/result.json")
    if not construction.get("construction_passed") or not construction.get("complete"):
        raise ValueError("源运行尚未完成数据确认，不能作为本版冻结数据")
    if manifest["template_sha256"] != codec.template_sha256:
        raise ValueError("实际模型模板与 v4.1 数据不一致，请复用原模型与 tokenizer")
    info = {"source_run_id": source.name, "dataset_id": manifest["dataset_id"],
            "selected_rule": manifest["selected_rule"],
            "files": [{"file": name, "sha256": sha256_file(folder / name)}
                      for name in ("experiment.jsonl", "test.jsonl", "dataset-manifest.json")],
            "test_previously_used": any(p.stat().st_size for p in source.glob("policies/*/test.jsonl"))}
    return info, folder, construction


def import_dataset(directory, config, source, folder, construction, info):
    destination = directory / "inputs/data/v4.2" / directory.name / "quality"
    destination.mkdir(parents=True, exist_ok=True)
    for record in info["files"]:
        shutil.copy2(folder / record["file"], destination / record["file"])
    atomic_write_json(directory / "construction/result.json", construction)
    atomic_write_json(directory / "inputs/reused-v41-config.json", read_json(source / "run-manifest.json")["config"])
    for path in (source / "report").glob("CONSTRUCTION-*.zh-CN.md"):
        target = directory / "report" / ("SOURCE-v4.1-"+path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    atomic_write_text(directory / "report/DATA-REUSE.zh-CN.md", "\n".join([
        "# v4.2 数据复用", "", f"来源运行：`{info['source_run_id']}`。",
        f"数据集：`{info['dataset_id']}`；条件：`{info['selected_rule']['condition']}`。", "",
        "逐字节复制原 experiment/test 及数据清单，保留原顺序、种子、答案、背景和划分。",
        "本次不执行 discovery/confirmation，不重新生成题目；构造报告是 v4.1 历史证据。",
        "本次重新评估端点与候选，不导入旧策略回答或旧运行耗时。",
        f"源运行已经使用测试集：`{info['test_previously_used']}`；若为 true，本次只作重跑诊断。", ""]))
    atomic_write_json(directory / "inputs/reused-data.json", info)


def load_reused(directory, config):
    info = read_json(directory / "inputs/reused-data.json")
    config = config.for_condition(info["selected_rule"]["condition"], data_directory=f"data/v4.2/{directory.name}")
    folder = directory / "inputs" / config.data_root / "quality"
    # 冻结输入的自动缓存身份检查；不要求操作者运行额外校验命令。
    for record in info["files"]:
        if sha256_file(folder / record["file"]) != record["sha256"]:
            raise ValueError("复制的数据已改变，请恢复原文件后续跑")
    splits = {}
    for split in ("experiment", "test"):
        with (folder / f"{split}.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != config.per_cell(split):
            raise ValueError(f"{split} 题数不完整，不能重新生成补题")
        if any(row["condition_id"] != config.selected_condition["condition_id"] for row in rows):
            raise ValueError("复用题目与冻结条件不一致")
        splits[split] = rows
    validate_samples(config, splits)
    return config, splits, info


def execute_reused(config, directory, runner):
    config, splits, info = load_reused(directory, config)
    runner.config = config
    runner.samples = {split: {r["sample_id"]: r for r in rows} for split, rows in splits.items()}
    construction = read_json(directory / "construction/result.json")
    manifest_path = directory / "run-manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest.update(dataset_id=info["dataset_id"], selected_condition=config.selected_condition)
        atomic_write_json(manifest_path, manifest)

    def finalize(result):
        finish_flags(result, construction, config, runner)
        result["data_reuse"] = info
        if info["test_previously_used"] and "test" in result.get("scores", {}):
            result.update(status="test_not_independent", technical_goal_passed=False, recovery_demonstrated=False)
            result["test_reused_from"] = sorted(set([*result.get("test_reused_from", []), info["source_run_id"]]))

    return execute(config, splits, runner, directory, finalize_result=finalize)


def run_reused(root, config, environment, codec, *, port=8010, run_id=None, reuse_run=None):
    if run_id:
        saved = read_json(root / "runs" / run_id / "run-manifest.json")
        directory = prepare_run(root, config, environment, codec, saved["source_dir"], saved["source_info"], run_id=run_id)
        if not (directory / "inputs/reused-data.json").exists():
            source = Path(saved["source_dir"])
            info, folder, construction = describe_source(source, config, codec)
            if info != saved["source_info"]:
                raise ValueError("未完成复制的源数据已改变，请恢复原数据后续跑")
            import_dataset(directory, config, source, folder, construction, info)
    else:
        if not reuse_run:
            raise ValueError("v4.2 首次运行请用 --reuse-run 指定 v4.1 原始运行目录；不需要离线来源库")
        source = (root / reuse_run).resolve()
        info, folder, construction = describe_source(source, config, codec)
        directory = prepare_run(root, config, environment, codec, source, info)
        if not (directory / "inputs/reused-data.json").exists():
            import_dataset(directory, config, source, folder, construction, info)
    print(f"运行 ID：{directory.name}；直接复用 v4.1 数据；诊断目录：{directory}", file=sys.stderr, flush=True)
    runner = V3PolicyRunner(config, root, directory, environment, {}, port=port, scorer=score_output)
    return execute_reused(config, directory, runner)
