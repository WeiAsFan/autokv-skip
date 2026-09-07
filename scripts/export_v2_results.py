"""离线导出 v2 结果和失败日志，生成可从 GitHub 网页上传的文件。"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PART_BYTES = 20 * 1024 * 1024


def _is_v2_run(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("_"):
        return False
    if (path / "quality/calibration").is_dir() or path.name.startswith("pilot-"):
        return True
    try:
        value = json.loads((path / "run-manifest.json").read_text(encoding="utf-8"))
        return isinstance(value, dict) and value.get("schema_version") == 2
    except (OSError, ValueError):
        return (path / "inputs").is_dir()


def export_results(root: Path, run_id: str | None = None) -> Path:
    root = root.resolve()
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id 含无效字符")
    runs = sorted(path for path in (root / "runs").glob("*") if _is_v2_run(path))
    if run_id is not None:
        runs = [path for path in runs if path.name == run_id]
        if not runs:
            raise ValueError(f"找不到 v2 运行：{run_id}")
    logs = sorted(
        path for path in (root / "runs").glob("v2-*")
        if path.is_file() and path.suffix in {".json", ".log", ".exitcode"}
    )
    if not runs and not logs:
        raise ValueError("没有 v2 结果或 CLI 日志；先按手册保存一次运行的输出")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / "results" / f"v2-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "autokv-v2.tar.gz"
    # 成功和失败都可导出，不要求 completed-manifest，也不重跑哈希门禁。
    sources = [
        root / name for name in
        ("autokv", "scripts", "configs", "pyproject.toml", "data/v2/quality", "docs/v2.0", "README.md")
    ]
    files: set[Path] = set(logs)
    for source in [*sources, *runs]:
        if source.is_file():
            files.add(source)
        elif source.is_dir():
            files.update(path for path in source.rglob("*") if path.is_file())
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(files):
            relative = path.relative_to(root)
            if any(part in {".git", ".cache", "__pycache__"} for part in relative.parts):
                continue
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
                continue
            archive.add(path, arcname=f"autokv-skip/{relative.as_posix()}", recursive=False)

    archive_names = [archive_path.name]
    if archive_path.stat().st_size > PART_BYTES:
        archive_names = []
        with archive_path.open("rb") as source:
            index = 0
            while chunk := source.read(PART_BYTES):
                part = output / f"{archive_path.name}.part-{index:04d}"
                part.write_bytes(chunk)
                archive_names.append(part.name)
                index += 1
        archive_path.unlink()

    lines = [
        "# AutoKV-Skip v2 运行归档", "",
        f"导出时间（UTC）：{stamp}", "",
        "归档包含原始结果、server 日志、未完成策略、CLI 输出和源码/配置/数据。",
        "正式运行的实际输入位于 `runs/<run-id>/inputs/`；归档根目录的源码是导出时的工作区副本。", "",
        "| 运行 ID | 状态 |", "|---|---|",
    ]
    for run in runs:
        status = "未完成（含诊断）"
        try:
            completed = json.loads((run / "completed-manifest.json").read_text(encoding="utf-8"))
            if isinstance(completed, dict) and completed.get("complete") is True:
                status = "已完成"
        except (OSError, ValueError):
            pass
        lines.append(f"| {run.name} | {status} |")
        report = run / "report/QUALITY-v2.zh-CN.md"
        if report.is_file():
            shutil.copy2(report, output / f"{run.name}-QUALITY-v2.zh-CN.md")
    if not runs:
        lines.append("| 尚未建立运行目录 | 仅 CLI 诊断 |")
    lines.extend(["", "## 解包", "", "```bash"])
    if len(archive_names) > 1:
        lines.append("cat autokv-v2.tar.gz.part-* > autokv-v2.tar.gz")
    lines.extend(["tar -xzf autokv-v2.tar.gz", "```", ""])
    (output / "README.zh-CN.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-id", help="只导出指定 v2 运行；默认包含所有 v2 运行及失败日志")
    args = parser.parse_args()
    print(export_results(args.project_root, args.run_id))


if __name__ == "__main__":
    main()
