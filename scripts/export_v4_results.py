"""导出 v4 构造、正式实验及失败日志，供 Linux 设备从 GitHub 网页提交。"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.export_v3_results import PART_BYTES, ROOT, export_results as export_archive


def export_results(root, run_id=None, *, part_bytes=PART_BYTES):
    return export_archive(root, run_id, part_bytes=part_bytes, version=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(export_results(args.project_root, args.run_id))


if __name__ == "__main__":
    main()
