"""生成网页上传用的小型分析包；长提示与完整原始日志保留在服务器。"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from autokv.io import atomic_write_text, read_json

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FIELDS = ("sample_id", "split", "index", "task", "condition_id", "batch_id", "length_bucket",
                 "record_count", "expected_answers", "source_group_id", "source_question_id",
                 "base_instance_id", "evidence_document_ids", "background_document_ids",
                 "evidence_positions", "prompt_tokens", "max_tokens", "scorer")
ANSWER_FIELDS = ("sample_id", "split", "policy_config_id", "task_score", "output_text", "finish_reason",
                 "prompt_tokens", "output_tokens", "e2e_ms", "retry_count", "attempt", "error", "timestamp")


def rows(path, warnings):
    """导出中断现场，不修改或截断源文件；坏行位置写入分析包。"""
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("不是 JSON 对象")
                yield row
            except (ValueError, UnicodeDecodeError) as exc:
                warnings.append({"file": str(path), "line": number, "error": str(exc)})


def json_line(row):
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n").encode("utf-8")


def excerpt(path, limit=8192):
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(limit//2)
        if size <= limit:
            content = head+stream.read()
        else:
            stream.seek(-limit//2, 2)
            content = head+f"\n[中间省略 {size-limit} 字节；完整日志保留在服务器]\n".encode("utf-8")+stream.read()
    return content.decode("utf-8", errors="replace")


def export_run(run, output):
    warnings, counts = [], {"samples": {}, "answers": {}, "attempts": 0}
    destination = output / f"{run.name}-evidence.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        # 精确运行代码和配置很小；保留它们以解释实际评分、后端与搜索行为。
        files = [p for name in ("run-manifest.json", "completed-manifest.json", "selection.json", "test-use.json")
                 if (p := run / name).is_file()]
        for pattern in ("inputs/*.json", "inputs/chat-template.txt", "inputs/autokv/*.py",
                        "inputs/configs/**/*.json", "inputs/scripts/*.py", "inputs/scripts/*.sh",
                        "inputs/runtime-overlay/**/*", "inputs/data/**/dataset-manifest.json",
                        "construction/result.json", "construction/selected-rule.json"):
            files.extend(p for p in run.glob(pattern) if p.is_file())
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(run).as_posix())
        with archive.open("samples.jsonl", "w") as stream:
            paths = [*run.glob("inputs/data/**/quality/*.jsonl"), *run.glob("construction/*.jsonl")]
            for path in sorted(paths):
                if path.stem not in {"experiment", "test", "discovery", "confirmation"}:
                    continue
                count = 0
                for row in rows(path, warnings):
                    stream.write(json_line({k: row[k] for k in SAMPLE_FIELDS if k in row}))
                    count += 1
                counts["samples"][path.stem] = counts["samples"].get(path.stem, 0)+count
        with archive.open("answers.jsonl", "w") as stream:
            for path in sorted(run.glob("policies/*/*.jsonl")):
                if path.stem not in {"experiment", "test", "discovery", "confirmation"}:
                    continue
                count = 0
                for row in rows(path, warnings):
                    stream.write(json_line({k: row[k] for k in ANSWER_FIELDS if k in row}))
                    count += 1
                counts["answers"][f"{path.parent.name}/{path.stem}"] = count
        with archive.open("attempts.jsonl", "w") as stream:
            for path in sorted(run.glob("policies/*/attempts/*.json")):
                try:
                    row = read_json(path)
                    row["sample_count"] = len(row.pop("sample_ids", []))
                    row["missing_count"] = len(row.pop("missing_ids", []))
                    row["attempt"] = path.stem
                    stream.write(json_line(row))
                    counts["attempts"] += 1
                except (ValueError, OSError) as exc:
                    warnings.append({"file": str(path), "error": str(exc)})
        trace = run / "search-trace.jsonl"
        if trace.exists():
            with archive.open("search-trace.jsonl", "w") as stream:
                for row in rows(trace, warnings):
                    stream.write(json_line(row))
        with archive.open("request-errors.jsonl", "w") as stream:
            for path in sorted(run.glob("policies/*/request-errors.jsonl")):
                for row in rows(path, warnings):
                    stream.write(json_line({"policy_config_id": path.parent.name, **row}))
        with archive.open("diagnostics.txt", "w") as stream:
            for path in sorted(run.glob("policies/*/attempts/*.log")):
                stream.write((f"\n===== {path.relative_to(run).as_posix()} =====\n"+excerpt(path)).encode("utf-8"))
        archive.writestr("export-summary.json", json_line({"format_version": 1, "run_id": run.name,
            "counts": counts, "warnings": warnings, "raw_inputs_included": False, "full_logs_included": False,
            "note": "可复算正式质量、配对分差和预算；不能从本包重建完整长提示或替代全部故障日志"}))
    for report in (run / "report").glob("*.zh-CN.md"):
        shutil.copy2(report, output / f"{run.name}-{report.name}")
    return destination, counts, warnings


def export_results(root, run_id=None):
    root = Path(root).resolve()
    if run_id is not None and not re.fullmatch(r"v4[12]-[A-Za-z0-9_-]+", run_id):
        raise ValueError("请输入 v42 运行 ID；也支持转换既有 v41 运行")
    runs = [root / "runs" / run_id] if run_id else sorted(p for p in (root / "runs").glob("v42-*") if p.is_dir())
    if any(not p.is_dir() for p in runs):
        raise ValueError(f"找不到运行 {run_id}")
    prefix = run_id.split("-")[0] if run_id else "v42"
    logs = [p for p in (root / "runs").glob(prefix+"-*") if p.is_file() and p.suffix in {".log", ".exitcode"}]
    if not runs and not logs:
        raise ValueError("没有结果或 CLI 诊断；请按手册保存命令输出后再导出")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / "results" / f"v42-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    lines = ["# 网页上传用分析结果", "", f"导出时间（UTC）：{stamp}。", "",
             "上传本目录的说明、报告和 evidence.zip 即可；不上传原始归档或分片。",
             "ZIP 内包含精简逐题输入索引与真值、原始短回答与分数、逐次服务元数据、搜索轨迹、配置与实际代码，以及日志首尾摘要。",
             "原始长提示、完整日志及运行目录留在服务器。分析包可复算正式质量、配对区间、容量和耗时，但不能替代完整数据复现或全部故障现场。", "",
             "| 运行 | 分析包 MiB | 样本索引数 | 保存回答数 | 导出坏行/文件数 |",
             "|---|---:|---:|---:|---:|"]
    for run in runs:
        path, counts, warnings = export_run(run, output)
        lines.append(f"| {run.name} | {path.stat().st_size/1024**2:.3f} | {sum(counts['samples'].values())} | {sum(counts['answers'].values())} | {len(warnings)} |")
    if logs:
        atomic_write_text(output / "CLI-DIAGNOSTICS.txt", "\n".join(f"===== {p.name} =====\n{excerpt(p, 32768)}" for p in sorted(logs)))
    if not runs:
        lines += ["", "尚未建立运行目录，仅导出 CLI 失败摘要，不能解释为质量实验结果。"]
    lines += ["", "坏行或不完整文件的位置见 ZIP 内 export-summary.json；导出不修改源数据，也不把中断算作成功。",
              "在 Linux 登录设备浏览器选择对应实验分支（v4.2 正式运行选择 v4.2），进入 results 后上传本目录并手动提交。",
              "服务器和 Linux 登录设备均不运行 git push。", "",
              "下载分析包后可用 Python 标准库解压：", "", "```bash",
              "python3 -m zipfile -e v42-实际运行ID-evidence.zip analysis", "```", ""]
    atomic_write_text(output / "README.zh-CN.md", "\n".join(lines))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(export_results(args.project_root, args.run_id))


if __name__ == "__main__":
    main()
