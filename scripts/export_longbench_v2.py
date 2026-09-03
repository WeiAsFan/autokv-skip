"""把固定 revision 的两个 LongBench-E 子集导出为 v2 离线源文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokv.io import atomic_write_json, atomic_write_text, read_json, sha256_file


REPOSITORY = "THUDM/LongBench"
REVISION = "92b6c5fbfb0c97b91e92d9ef79802f95ce74b05e"
DATASETS = ("qasper_e", "hotpotqa_e")


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def export(output_dir: Path) -> Mapping[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "source-manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            raise ValueError("已有 source-manifest.json 无效")
        files = manifest.get("files")
        if (
            manifest.get("repository") != REPOSITORY
            or manifest.get("revision") != REVISION
            or not isinstance(files, Mapping)
            or any(
                not (output_dir / f"{dataset}.jsonl").is_file()
                or sha256_file(output_dir / f"{dataset}.jsonl")
                != files.get(f"{dataset}.jsonl")
                for dataset in DATASETS
            )
        ):
            raise ValueError(
                "已有 LongBench 源目录不完整或 hash 不匹配；请保留后另建目录"
            )
        return manifest

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "缺少 datasets；请在一次性的 data venv 中安装 datasets 与 pyarrow"
        ) from exc

    rows_by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for dataset in DATASETS:
        loaded = load_dataset(
            REPOSITORY,
            dataset,
            split="test",
            revision=REVISION,
        )
        rows = [dict(row) for row in loaded]
        if not rows:
            raise ValueError(f"LongBench {dataset} 返回空数据")
        ids = [row.get("_id") for row in rows]
        if any(not isinstance(item, str) or not item for item in ids) or len(
            ids
        ) != len(set(ids)):
            raise ValueError(f"LongBench {dataset} 的 _id 缺失或重复")
        rows_by_dataset[dataset] = rows

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for dataset in DATASETS:
        filename = f"{dataset}.jsonl"
        path = output_dir / filename
        atomic_write_text(path, _jsonl(rows_by_dataset[dataset]))
        files[filename] = sha256_file(path)
        counts[dataset] = len(rows_by_dataset[dataset])
    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "split": "test",
        "datasets": list(DATASETS),
        "rows": counts,
        "files": files,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="导出 AutoKV-Skip v2 使用的 LongBench-E 数据"
    )
    parser.add_argument(
        "--output-dir",
        default="data/v2/source/LongBench",
        help="离线源文件输出目录",
    )
    args = parser.parse_args(argv)
    manifest = export(Path(args.output_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
