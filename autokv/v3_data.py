"""离线来源划分、证据保护和固定前缀数据构造。"""
from __future__ import annotations

import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from autokv.io import atomic_write_json, atomic_write_text, read_json, read_jsonl
from autokv.v2_data import TransformersPromptCodec
from autokv.v3_config import TASKS, SCORE_VERSION, identity

SPLITS = ("development", "experiment", "test")
DATA_ROOT = Path("data/v3.0")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def document_id(title):
    return "wiki:" + normalized(unquote(title).replace("_", " "))


def load_sources(directory, config):
    manifest = read_json(directory / "source-manifest.json")
    records = {r["name"]: r for r in manifest["sources"]}
    docs, questions, squad_docs, text_owner, parent = {}, [], set(), {}, {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def add(title, text):
        key = document_id(title)
        content = normalized(text)
        if not content:
            raise ValueError("来源包含空文档")
        if content in text_owner:
            a, b = find(key), find(text_owner[content])
            parent[max(a, b)] = min(a, b)
        text_owner[content] = key
        find(key)
        if key not in docs or len(text) > len(docs[key]["text"]):
            docs[key] = {"title": title, "text": text}
        return key

    for name in config.raw["data"]["sources"]:
        if name not in records:
            raise ValueError(f"来源缺少 {name}；在 Linux 设备重新准备来源包")
        raw = read_json(directory / records[name]["file"])
        if name.startswith("squad"):
            for article in raw["data"]:
                for paragraph in article["paragraphs"]:
                    content = paragraph["context"]
                    doc = add(article["title"], content)
                    squad_docs.add(doc)
                    for qa in paragraph["qas"]:
                        if qa.get("is_impossible"):
                            continue
                        answers = sorted({a["text"] for a in qa["answers"] if a["text"] and
                                          content[a["answer_start"]:a["answer_start"]+len(a["text"])] == a["text"]})
                        if not answers:
                            continue
                        questions.append({"id": "squad:"+qa["id"], "task": "qa_single", "question": qa["question"],
                                          "answers": answers, "evidence": [(doc, article["title"], content)]})
        elif name.startswith("hotpot"):
            for qa in raw:
                contexts = {title: pieces for title, pieces in qa["context"]}
                for title, pieces in contexts.items():
                    if "".join(pieces).strip():
                        add(title, "".join(pieces))
                support = qa.get("supporting_facts", [])
                titles = sorted({item[0] for item in support})
                if len(titles) != 2 or not qa.get("answer") or any(
                    title not in contexts or type(index) is not int or not 0 <= index < len(contexts[title])
                    for title, index in support
                ):
                    continue
                evidence = [(document_id(t), t, "".join(contexts[t])) for t in titles]
                if any(not text.strip() for _, _, text in evidence):
                    continue
                questions.append({"id": "hotpot:"+qa["_id"], "task": "qa_multi", "question": qa["question"],
                                  "answers": [qa["answer"]], "evidence": evidence})
        else:
            raise ValueError(f"没有来源适配器：{name}")
    canonical_docs = {}
    for key, doc in docs.items():
        key = find(key)
        if key not in canonical_docs or len(doc["text"]) > len(canonical_docs[key]["text"]):
            canonical_docs[key] = doc
    unique, seen = [], set()
    for question in questions:
        question["evidence"] = [(find(d), title, text) for d, title, text in question["evidence"]]
        key = (normalized(question["question"]), tuple(sorted(d for d, _, _ in question["evidence"])))
        if key not in seen:
            seen.add(key)
            unique.append(question)
    # 来源种类分别平衡，避免 SQuAD 的几百个文章被大背景池的波动耗尽。
    assignment = {}
    rng = random.Random(config.seed)
    squad_docs = {find(d) for d in squad_docs}
    for group in (squad_docs, set(canonical_docs)-squad_docs):
        keys = sorted(group)
        rng.shuffle(keys)
        n = len(keys)
        for i, key in enumerate(keys):
            assignment[key] = "development" if i < int(n*.10) else "experiment" if i < int(n*.28) else "test"
    return canonical_docs, unique, assignment, manifest


class EvidenceTooLong(ValueError):
    pass


def fit_prompt(codec, target, tolerance, instruction, question, evidence, background):
    """二分裁剪非证据文本，完整插入所有证据块。"""
    def compose(size):
        text = background[:size]
        positions = [int(size*(i+1)/(len(evidence)+1)) for i in range(len(evidence))]
        chunks, start = [], 0
        for pos, block in zip(positions, evidence):
            chunks.extend((text[start:pos], "\n"+block+"\n"))
            start = pos
        chunks.append(text[start:])
        return instruction + "\n\n" + "".join(chunks) + "\n\n" + question
    prompt = compose(0)
    _, count = codec.render_and_count(prompt)
    if count > target + tolerance:
        raise EvidenceTooLong("必要证据与问题超过该长度；不能截断证据")
    low, high = 0, len(background)
    while low <= high:
        mid = (low+high)//2
        prompt = compose(mid)
        _, count = codec.render_and_count(prompt)
        if abs(count-target) <= tolerance:
            return prompt, count, mid
        if count < target:
            low = mid+1
        else:
            high = mid-1
    raise ValueError(f"自然背景不足或无法拟合长度 {target}，最后计数 {count}")


def _retrieval(task, rng, parameters):
    def token(prefix, used):
        while True:
            value = prefix + f"{rng.getrandbits(32):08X}"
            if value not in used:
                used.add(value)
                return value
    used = set()
    if task == "multi_key":
        pairs = [(token("K", used), token("V", used)) for _ in range(parameters["key_records"])]
        targets = rng.sample(pairs, parameters["query_keys"])
        question = "Return only the requested key=value pairs, one per line. Keys: " + ", ".join(k for k, _ in targets)
        answers = [list(p) for p in targets]
    else:
        key = token("K", used)
        answers = [token("V", used) for _ in range(parameters["values"])]
        pairs = [(key, value) for value in answers]
        pairs += [(token("K", used), token("V", used)) for _ in range(parameters["distractors"])]
        question = f"Return only all distinct values associated with {key}, separated by commas."
    rng.shuffle(pairs)
    return question, answers, [f"Record: {key} = {value}" for key, value in pairs]


def generate_dataset(config, codec, docs, questions, assignment, requested_splits=SPLITS, progress=None):
    """来源划分统一；只构造本次需要的 split，组内生成不受其他 split 影响。"""
    cells, used_questions, used_evidence = {}, set(), set()
    pools = {split: sorted(d for d in docs if assignment[d] == split) for split in SPLITS}
    qa_pools = defaultdict(list)
    for q in questions:
        splits = {assignment[d] for d, _, _ in q["evidence"]}
        if len(splits) == 1:
            qa_pools[next(iter(splits)), q["task"]].append(q)
    rng = random.Random(config.seed+1)
    for pool in qa_pools.values():
        rng.shuffle(pool)
    skipped = Counter()
    # 先给 QA 分配互不重复的证据；背景可复用，但始终留在来源 split 内。
    for split in requested_splits:
        for task in ("qa_single", "qa_multi", "multi_key", "multi_value"):
            for length in config.lengths:
                rows = []
                cell = (split, task, length)
                rng = random.Random(int(identity([config.seed+1, cell]), 16))
                cells[cell] = rows
                pool = qa_pools[split, task] if task.startswith("qa_") else list(pools[split])
                if not task.startswith("qa_"):
                    rng.shuffle(pool)
                for source in pool:
                    if len(rows) == config.per_cell(split):
                        break
                    if task.startswith("qa_"):
                        evidence_ids = {d for d, _, _ in source["evidence"]}
                        if source["id"] in used_questions or evidence_ids & used_evidence:
                            continue
                        qid, question, answers = source["id"], source["question"], source["answers"]
                        blocks = [f"Document: {title}\n{text}" for _, title, text in source["evidence"]]
                        instruction = "Answer the question using the documents. Return only a short answer; do not explain."
                        question = "Question: " + question + "\nAnswer:"
                    else:
                        if source in used_evidence:
                            continue
                        evidence_ids = {source}
                        qid = f"retrieval:{task}:{source}"
                        question, answers, blocks = _retrieval(task, rng, config.raw["data"]["retrieval"])
                        instruction = "Read the records in the documents and return exactly the requested information."
                    # 每题独立取样自然背景，禁止跨 split，避免对照中改变背景。
                    bg_ids = rng.sample(pools[split], min(512, len(pools[split])))
                    bg_ids = [d for d in bg_ids if d not in evidence_ids]
                    if not task.startswith("qa_"):
                        bg_ids.insert(0, source)
                    parts, starts, size = [], [], 0
                    for d in bg_ids:
                        piece = f"\nDocument: {docs[d]['title']}\n{docs[d]['text']}\n"
                        starts.append((size, d))
                        parts.append(piece)
                        size += len(piece)
                        if size >= length*12:
                            break
                    try:
                        prompt, tokens, used_chars = fit_prompt(codec, length, config.tolerance, instruction,
                                                               question, blocks, "".join(parts))
                    except EvidenceTooLong:
                        skipped[f"{split}:{task}:{length}:evidence_too_long"] += 1
                        continue
                    if not all(block in prompt for block in blocks):
                        raise ValueError("数据拟合丢失证据")
                    background_ids = [d for start, d in starts if start < used_chars]
                    if task.startswith("multi_"):
                        # 首个自然背景锚点代表基础场景，不能换值后重复计算为独立题。
                        group = "retrieval:"+source
                    else:
                        group = "evidence:"+identity(sorted(evidence_ids))
                    row = {"sample_id": f"v3:{split}:{task}:{length}:{len(rows):03d}",
                           "source_question_id": qid, "source_group_id": group, "split": split,
                           "task": task, "length_bucket": length, "user_prompt": prompt,
                           "expected_answers": answers, "prompt_tokens": tokens, "max_tokens": config.max_tokens,
                           "scorer": SCORE_VERSION, "source_document_ids": sorted(evidence_ids | set(background_ids)),
                           "evidence_document_ids": sorted(evidence_ids), "background_document_ids": background_ids,
                           "evidence_positions": [prompt.index(block) for block in blocks], "generation_seed": config.seed+1}
                    rows.append(row)
                    used_questions.add(qid)
                    used_evidence.update(evidence_ids)
                    if progress is not None and (len(rows) % 16 == 0 or len(rows) == config.per_cell(split)):
                        progress(f"{split}/{task}/{length}：{len(rows)}/{config.per_cell(split)}")
                if len(rows) != config.per_cell(split):
                    raise ValueError(f"来源不足：{cell} 需要 {config.per_cell(split)} 题，得到 {len(rows)}；不能用重复题补齐")
    splits = {split: [cells[split, task, length][i] for i in range(config.per_cell(split))
                      for task in TASKS for length in config.lengths] for split in requested_splits}
    validate_splits(config, splits)
    return splits, dict(skipped)


def validate_splits(config, splits):
    ids, questions, owners, groups = set(), set(), {}, {}
    for split, rows in splits.items():
        counts = Counter()
        for row in rows:
            sid, qid = row["sample_id"], row["source_question_id"]
            if sid in ids or qid in questions:
                raise ValueError("重复问题或样本 ID，不能重复计算长度变体")
            ids.add(sid)
            questions.add(qid)
            cell = (split, row["task"], row["length_bucket"])
            if row["split"] != split or row["task"] not in TASKS or row["length_bucket"] not in config.lengths:
                raise ValueError("样本 split/任务/长度与协议不一致")
            counts[cell[1:]] += 1
            if not row["expected_answers"] or row["scorer"] != SCORE_VERSION:
                raise ValueError("缺少标准答案或评分版本不一致")
            if abs(row["prompt_tokens"]-row["length_bucket"]) > config.tolerance or row["max_tokens"] != config.max_tokens:
                raise ValueError("样本长度或输出预算不一致")
            if row["prompt_tokens"]+row["max_tokens"] > config.raw["model"]["max_model_len"]:
                raise ValueError("模型上下文空间不足")
            for doc in row["source_document_ids"]:
                if doc in owners and owners[doc] != split:
                    raise ValueError("来源文档跨 split 泄漏")
                owners[doc] = split
            group = row["source_group_id"]
            if group in groups and groups[group] != cell:
                raise ValueError("来源组跨任务或长度单元")
            groups[group] = cell
        expected = {(t, l): config.per_cell(split) for t in TASKS for l in config.lengths}
        if counts != expected:
            raise ValueError(f"{split} 的单元题数不完整")
        for n in config.fidelities if split == "experiment" else ():
            prefix = Counter((r["task"], r["length_bucket"]) for r in rows[:n])
            if prefix != {cell: n//config.cells for cell in expected}:
                raise ValueError("实验集前缀未按分层交织")


def data_protocol(config):
    return {"model": config.raw["model"], "data": config.raw["data"], "scorer": SCORE_VERSION}


def make_data(root, source_dir, mode="formal", codec=None):
    from autokv.v3_config import load_config
    from autokv.v3_runtime import load_environment
    config = load_config(root)
    if codec is None:
        codec = TransformersPromptCodec(Path(load_environment(root, config)["model_path"]))
    directory = root / DATA_ROOT / ("development" if mode == "development" else "quality")
    manifest_path = directory / "dataset-manifest.json"
    if manifest_path.exists():
        manifest, _ = load_dataset(root, config, mode == "development")
        if manifest["template_sha256"] != codec.template_sha256:
            raise ValueError("当前 chat template 与已生成数据不同；新模板需要新的数据版本")
        return {"directory": str(directory), "reused": True, "counts": manifest["counts"]}
    docs, questions, assignment, source_manifest = load_sources(source_dir, config)
    print("正在离线构造按来源隔离的数据；不启动 GPU 服务", file=sys.stderr, flush=True)
    requested = ("development",) if mode == "development" else ("experiment", "test")
    selected, skipped = generate_dataset(config, codec, docs, questions, assignment, requested,
                                         progress=lambda message: print(message, file=sys.stderr, flush=True))
    for split, rows in selected.items():
        atomic_write_text(directory / f"{split}.jsonl", "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    manifest = {"schema_version": 3, "protocol": data_protocol(config), "template_sha256": codec.template_sha256,
                "chat_template": getattr(codec, "template_text", None),
                "sources": source_manifest, "counts": {s: len(rows) for s, rows in selected.items()},
                "dataset_id": identity(selected), "skipped": skipped,
                "source_groups": {s: len({r["source_group_id"] for r in rows}) for s, rows in selected.items()},
                "background_documents": {s: len({d for r in rows for d in r["background_document_ids"]}) for s, rows in selected.items()},
                "source_allocation": "文档按来源种类固定分配 10% development、18% experiment、其余 test；正式题数为 20%/80%"}
    atomic_write_json(manifest_path, manifest)
    return {"directory": str(directory), "reused": False, "counts": manifest["counts"]}


def load_dataset(root, config, development=False):
    directory = root / DATA_ROOT / ("development" if development else "quality")
    if not (directory / "dataset-manifest.json").exists():
        raise ValueError("缺少 v3 数据；先执行 v3-make-data --mode " + ("development" if development else "formal"))
    manifest = read_json(directory / "dataset-manifest.json")
    if manifest["protocol"] != data_protocol(config):
        raise ValueError("数据与当前任务/模型协议不一致；新协议请使用新项目目录重新构造数据")
    splits = {s: read_jsonl(directory / f"{s}.jsonl") for s in (("development",) if development else ("experiment", "test"))}
    validate_splits(config, splits)
    if manifest["dataset_id"] != identity(splits):
        raise ValueError("实际数据已改变；不能使用旧数据清单或旧回答")
    return manifest, splits
