"""导出 v3 成功/失败证据，供 Linux 设备 GitHub 网页手动提交。"""
from __future__ import annotations

import argparse
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from autokv.io import atomic_write_text, read_json

PART_BYTES = 20 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


def export_results(root, run_id=None, *, part_bytes=PART_BYTES):
    root = root.resolve()
    if run_id is not None and not re.fullmatch(r"v3-[A-Za-z0-9_-]+", run_id):
        raise ValueError("请输入 v3 运行 ID")
    runs = sorted(p for p in (root / "runs").glob("v3-*") if p.is_dir())
    if run_id:
        runs = [p for p in runs if p.name == run_id]
        if not runs:
            raise ValueError(f"找不到运行 {run_id}")
    logs = [p for p in (root / "runs").glob("v3-*") if p.is_file() and p.suffix in {".log", ".json", ".exitcode"}]
    if not runs and not logs:
        raise ValueError("没有 v3 结果或 CLI 日志；按运行手册保存命令输出后再导出")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / "results" / f"v3-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    sources = [root / p for p in ("autokv", "scripts", "configs/v3.0", "docs/v3.0", "data/v3.0/README.md",
                                  "CONTEXT.md", "README.md", "pyproject.toml")]
    if not runs:
        sources.extend(root / p for p in ("data/v3.0/development", "data/v3.0/quality"))
    files = set(logs)
    for path in [*sources, *runs]:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file())
    archive = output / "autokv-v3.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(files):
            relative = path.relative_to(root)
            if any(p in {".git", ".cache", "__pycache__"} for p in relative.parts):
                continue
            if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
                continue
            stream.add(path, arcname="autokv-skip/"+relative.as_posix(), recursive=False)
    names = [archive.name]
    if archive.stat().st_size > part_bytes:
        names = []
        with archive.open("rb") as stream:
            while block := stream.read(part_bytes):
                part = output / f"{archive.name}.part-{len(names):04d}"
                part.write_bytes(block)
                names.append(part.name)
        archive.unlink()
    lines = ["# AutoKV-Skip v3 运行归档", "", f"导出时间（UTC）：{stamp}", "",
             "包含原始回答、部分结果、每次服务日志、搜索轨迹、运行输入与 CLI 失败日志。",
             "`runs/<run_id>/inputs/` 为运行时实际输入；根目录源码是导出时副本。",
             "来源和许可说明位于 inputs 的 dataset-manifest.json；自定义提示与评分不是官方榜单协议。", "",
             "| 运行 | 状态 | 主实验达标 |", "|---|---|---|"]
    for run in runs:
        try:
            status = read_json(run / "completed-manifest.json")
        except (OSError, ValueError):
            status = {"status": "未完成，含诊断", "technical_goal_passed": False}
        lines.append(f"| {run.name} | {status['status']} | {status.get('technical_goal_passed', False)} |")
        for report in (run / "report").glob("*.zh-CN.md"):
            shutil.copy2(report, output / f"{run.name}-{report.name}")
    if not runs:
        lines.append("| 尚未建立运行目录 | 仅 CLI 诊断 | false |")
    lines += ["", "在 Linux 登录设备浏览器选择 GitHub 的 v3.0 分支，上传本目录文件并手动提交。服务器与 Linux 设备均不需要 git push。",
              "", "## 解包", "", "```bash"]
    if len(names) > 1:
        lines.append("cat autokv-v3.tar.gz.part-* > autokv-v3.tar.gz")
    lines += ["tar -xzf autokv-v3.tar.gz", "```", ""]
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
