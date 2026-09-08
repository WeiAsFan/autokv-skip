"""在联网 Linux 设备准备来源；已有文件可通过 --input-dir 导入。"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from autokv.io import atomic_write_json

SOURCES = {
    "squad-v2.0-train": ("train-v2.0.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"),
    "squad-v2.0-dev": ("dev-v2.0.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"),
    "hotpot-v1.1-train": ("hotpot_train_v1.1.json", "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"),
    "hotpot-v1.1-dev-distractor": ("hotpot_dev_distractor_v1.json", "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"),
}

HF_ROWS = "https://datasets-server.huggingface.co/rows"


def _json_url(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def hotpot_row(row):
    """恢复原始 HotpotQA JSON 结构，不改变问题、答案和证据句。"""
    facts, context = row["supporting_facts"], row["context"]
    if len(facts["title"]) != len(facts["sent_id"]) or len(context["title"]) != len(context["sentences"]):
        raise ValueError("HotpotQA 镜像的证据字段长度不一致")
    return {"_id": row["id"], "question": row["question"], "answer": row["answer"],
            "type": row.get("type"), "level": row.get("level"),
            "supporting_facts": list(zip(facts["title"], facts["sent_id"])),
            "context": list(zip(context["title"], context["sentences"]))}


def download_hotpot_mirror(path, split):
    """标准库读取公开镜像；逐页验证完整性，拒绝静默使用截断的证据。"""
    cache = path.with_name(path.name+".pages")
    cache.mkdir(parents=True, exist_ok=True)
    def page(offset, length=100):
        cached = cache / f"{offset}-{length}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        url = HF_ROWS + "?" + urllib.parse.urlencode({"dataset": "hotpotqa/hotpot_qa", "config": "distractor",
                                                     "split": split, "offset": offset, "length": length})
        value = _json_url(url)
        atomic_write_json(cached, value)
        return value
    first = page(0)
    total = first["num_rows_total"]
    if not isinstance(total, int) or total <= 0:
        raise ValueError("镜像没有有效题数")
    seen = set()
    with path.open("w", encoding="utf-8", newline="\n") as stream, ThreadPoolExecutor(max_workers=4) as pool:
        stream.write("[")
        for batch_start in range(0, total, 400):
            offsets = list(range(batch_start, min(total, batch_start+400), 100))
            pages = list(pool.map(lambda offset: first if offset == 0 else page(offset), offsets))
            for offset, payload in zip(offsets, pages):
                rows = payload["rows"]
                if payload["num_rows_total"] != total or len(rows) != min(100, total-offset):
                    raise ValueError("镜像分页题数改变或缺失，不能发布不完整来源")
                for index, item in enumerate(rows):
                    expected = offset+index
                    if item["row_idx"] != expected:
                        raise ValueError("镜像分页行号不连续")
                    if item.get("truncated_cells"):
                        item = page(expected, 1)["rows"][0]
                    if item.get("truncated_cells") or item["row_idx"] != expected:
                        raise ValueError("镜像仍截断原始证据；请用 --input-dir 提供完整 HotpotQA JSON")
                    row = hotpot_row(item["row"])
                    if row["_id"] in seen:
                        raise ValueError("镜像存在重复问题 ID")
                    if seen:
                        stream.write(",\n")
                    json.dump(row, stream, ensure_ascii=False)
                    seen.add(row["_id"])
            print(f"HotpotQA {split}：{len(seen)}/{total}", flush=True)
        stream.write("]\n")
    # 仅清理由本下载器建立的页缓存；失败时保留以供下次继续。
    for cached in cache.glob("*-*.json"):
        if re.fullmatch(r"\d+-\d+\.json", cached.name):
            cached.unlink()
    if not any(cache.iterdir()):
        cache.rmdir()
    return {"download_url": HF_ROWS, "repository": "hotpotqa/hotpot_qa", "config": "distractor", "split": split,
            "rows": total, "conversion": "镜像字段还原为原始 HotpotQA JSON；正文与证据未裁剪"}


def prepare(output: Path, input_dir: Path | None = None, hotpot_source="auto"):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (filename, url) in SOURCES.items():
        path = output / filename
        origin_path = output / (filename + ".origin.json")
        if not path.exists():
            temporary = path.with_suffix(".download")
            origin = {"download_url": url}
            if input_dir is not None:
                shutil.copyfile(input_dir / filename, temporary)
                origin = {"imported_file": filename}
            elif name.startswith("hotpot") and hotpot_source == "huggingface":
                origin = download_hotpot_mirror(temporary, "train" if name.endswith("train") else "validation")
            else:
                print(f"下载 {name}: {url}", flush=True)
                try:
                    with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                except OSError as exc:
                    if name.startswith("hotpot") and hotpot_source == "auto":
                        print("原站不可达，改用 hotpotqa/hotpot_qa 公开镜像。", flush=True)
                        origin = download_hotpot_mirror(temporary, "train" if name.endswith("train") else "validation")
                    else:
                        raise RuntimeError(f"下载失败：{url}；可在浏览器下载同名原始文件后用 --input-dir 导入：{exc}") from exc
            # 只在来源准备时检查 JSON，避免把错误网页当成数据。
            with temporary.open(encoding="utf-8") as stream:
                json.load(stream)
            temporary.replace(path)
            atomic_write_json(origin_path, origin)
        origin = json.loads(origin_path.read_text(encoding="utf-8")) if origin_path.exists() else {"imported_file": filename}
        records.append({"name": name, "file": filename, "url": url,
                        "bytes": path.stat().st_size, "license": "CC BY-SA 4.0", "origin": origin})
    manifest = {"schema_version": 3, "sources": records,
                "attribution": ["SQuAD 2.0: Rajpurkar, Jia, Liang (2018), https://rajpurkar.github.io/SQuAD-explorer/",
                                "HotpotQA: Yang et al. (2018), https://hotpotqa.github.io/"],
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "usage": "本项目重新划分来源，构造长上下文和自定义评分；不属于官方榜单评估"}
    atomic_write_json(output / "source-manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/v3.0/source"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--hotpot-source", choices=("auto", "original", "huggingface"), default="auto")
    args = parser.parse_args()
    print(prepare(args.output, args.input_dir, args.hotpot_source))


if __name__ == "__main__":
    main()
