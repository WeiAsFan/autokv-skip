"""生成并冻结 AutoKV-Skip v2.0 的三层质量数据。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from autokv.io import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
)
from autokv.v2_config import V2QualityConfig


LONG_BENCH_PROMPTS = {
    "qasper_e": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, write "
        '"unanswerable". If the question is a yes/no question, answer "yes", "no", '
        'or "unanswerable". Do not provide any explanation.\n\nArticle: {context}'
        "\n\nAnswer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be "
        'answered based on the information in the article, write "unanswerable". '
        'If the question is a yes/no question, answer "yes", "no", or '
        '"unanswerable". Do not provide any explanation.\n\nQuestion: {input}'
        "\n\nAnswer:"
    ),
    "hotpotqa_e": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}"
        "\n\nAnswer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
}

FILLER = (
    "ARCHIVE-NOTE: routine observations describe ordinary routes, weather, tools, "
    "schedules, and maintenance events; this note contains no answer token.\n"
)
PADDING = " padding"


class PromptCodec(Protocol):
    """用冻结 tokenizer/chat template 渲染并计数一条用户消息。"""

    template_sha256: str

    def render_and_count(self, user_prompt: str) -> tuple[str, int]: ...


class TransformersPromptCodec:
    """延迟依赖 transformers，避免把 GPU 运行环境依赖加入控制器。"""

    def __init__(self, model_path: Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - 只在远程运行环境触发
            raise RuntimeError(
                "冻结 v2 数据需要 transformers；请使用环境锁中的 vLLM Python 执行"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        template = getattr(self._tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            raise ValueError("冻结 tokenizer 没有 chat_template")
        self.template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    def render_and_count(self, user_prompt: str) -> tuple[str, int]:
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = self._tokenizer.encode(rendered, add_special_tokens=False)
        return str(rendered), len(token_ids)


@dataclass(frozen=True)
class FittedPrompt:
    user_prompt: str
    rendered_prompt: str
    prompt_tokens: int


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest_token(prefix: str, *parts: object, size: int = 8) -> str:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:size].upper()
    return f"{prefix}-{digest}"


def _spread_lines(lines: Sequence[str], repetitions: int, padding_units: int) -> str:
    if repetitions < 0 or padding_units < 0:
        raise ValueError("填充次数不能为负数")
    quotient, remainder = divmod(repetitions, len(lines) + 1)
    pieces: list[str] = []
    for index, line in enumerate(lines):
        count = quotient + int(index < remainder)
        pieces.append(FILLER * count)
        pieces.append(line.rstrip() + "\n")
    pieces.append(FILLER * (quotient + int(len(lines) < remainder)))
    pieces.append(PADDING * padding_units)
    return "".join(pieces)


def fit_synthetic_prompt(
    target_tokens: int,
    tolerance_tokens: int,
    compose: Callable[[int, int], str],
    codec: PromptCodec,
) -> FittedPrompt:
    """对可单调填充的任务做二分，使 chat-template 后长度命中目标。"""
    if target_tokens <= 0 or tolerance_tokens < 0:
        raise ValueError("目标长度必须为正数，误差不能为负数")

    cache: dict[tuple[int, int], FittedPrompt] = {}

    def observe(repetitions: int, padding_units: int = 0) -> FittedPrompt:
        key = (repetitions, padding_units)
        if key not in cache:
            user_prompt = compose(repetitions, padding_units)
            rendered, count = codec.render_and_count(user_prompt)
            cache[key] = FittedPrompt(user_prompt, rendered, count)
        return cache[key]

    low, high = 0, 1
    while observe(high).prompt_tokens < target_tokens:
        low = high + 1
        high *= 2
        if high > 1_048_576:
            raise ValueError("tokenizer 计数未随填充增长")

    candidates: list[FittedPrompt] = []
    below: tuple[int, FittedPrompt] | None = None
    while low <= high:
        middle = (low + high) // 2
        current = observe(middle)
        candidates.append(current)
        if current.prompt_tokens <= target_tokens and (
            below is None or current.prompt_tokens > below[1].prompt_tokens
        ):
            below = (middle, current)
        if current.prompt_tokens < target_tokens:
            low = middle + 1
        elif current.prompt_tokens > target_tokens:
            high = middle - 1
        else:
            break

    best = min(
        candidates,
        key=lambda item: (abs(item.prompt_tokens - target_tokens), item.prompt_tokens),
    )
    if abs(best.prompt_tokens - target_tokens) <= tolerance_tokens:
        return best

    if below is not None:
        repetitions, current = below
        pad_low, pad_high = 0, max(64, (target_tokens - current.prompt_tokens) * 4)
        while observe(repetitions, pad_high).prompt_tokens < target_tokens:
            pad_high *= 2
            if pad_high > target_tokens * 8:
                break
        while pad_low <= pad_high:
            middle = (pad_low + pad_high) // 2
            current = observe(repetitions, middle)
            if (
                abs(current.prompt_tokens - target_tokens),
                current.prompt_tokens,
            ) < (abs(best.prompt_tokens - target_tokens), best.prompt_tokens):
                best = current
            if current.prompt_tokens < target_tokens:
                pad_low = middle + 1
            elif current.prompt_tokens > target_tokens:
                pad_high = middle - 1
            else:
                break

    if abs(best.prompt_tokens - target_tokens) > tolerance_tokens:
        raise ValueError(
            f"无法把输入控制在 ±{tolerance_tokens} tokens："
            f"target={target_tokens}, observed={best.prompt_tokens}"
        )
    return best


def _base_row(
    *,
    sample_id: str,
    split: str,
    tier: str,
    task: str,
    fitted: FittedPrompt,
    target_tokens: int | None,
    max_tokens: int,
    expected_answers: Sequence[str],
    answer_mode: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    content_identity = {
        "tier": tier,
        "task": task,
        "prompt": fitted.user_prompt,
        "expected_answers": list(expected_answers),
    }
    return {
        "schema_version": 2,
        "sample_id": sample_id,
        "split": split,
        "tier": tier,
        "task": task,
        "prompt": fitted.user_prompt,
        "rendered_prompt_sha256": hashlib.sha256(
            fitted.rendered_prompt.encode("utf-8")
        ).hexdigest(),
        "prompt_tokens": fitted.prompt_tokens,
        "target_tokens": target_tokens,
        "max_tokens": max_tokens,
        "expected_answers": list(expected_answers),
        "answer_mode": answer_mode,
        "metadata": dict(metadata),
        "content_sha256": _sha256_json(content_identity),
    }


def make_easy_rows(config: V2QualityConfig, codec: PromptCodec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, seed in (
        ("calibration", config.easy_calibration_seed),
        ("heldout", config.easy_heldout_seed),
    ):
        for target in config.target_lengths:
            code = _digest_token("KV", "easy", split, target, seed)
            answer = "|".join([code] * 5)
            line = f"VERIFICATION-CODE: {code}"

            def compose(
                repetitions: int, padding_units: int, *, clue: str = line
            ) -> str:
                body = _spread_lines((clue,), repetitions, padding_units)
                return (
                    "Read the whole archive. Exactly one line contains the verification "
                    "code; all ARCHIVE-NOTE lines are distractors.\n\n"
                    + body
                    + "\nWhat is the verification code? Output it exactly five times "
                    "separated by | and add no explanation."
                )

            fitted = fit_synthetic_prompt(
                target, config.tolerance_tokens, compose, codec
            )
            sample_id = f"v2-easy-{split}-{target}-{seed}-{code[-8:].lower()}"
            rows.append(
                _base_row(
                    sample_id=sample_id,
                    split=split,
                    tier="easy",
                    task="niah",
                    fitted=fitted,
                    target_tokens=target,
                    max_tokens=config.max_tokens_for("niah"),
                    expected_answers=(answer,),
                    answer_mode="contains",
                    metadata={"seed": seed, "depth": config.easy_depth, "code": code},
                )
            )
    return rows


def _hard_parameters(config: V2QualityConfig) -> dict[str, int]:
    hard = config.raw["data"]["hard"]
    values = {key: int(value) for key, value in hard["parameters"].items()}
    if config.hard_difficulty == "easy":
        values["multi_key_query_count"] = max(2, values["multi_key_query_count"] - 1)
        values["variable_steps"] = max(8, values["variable_steps"] - 8)
        values["aggregation_label_count"] = max(
            4, values["aggregation_label_count"] - 1
        )
    elif config.hard_difficulty == "hard":
        values["multi_key_query_count"] += 1
        values["variable_steps"] += 8
        values["aggregation_label_count"] += 1
    return values


def _multi_key_task(
    seed: int, target: int, query_count: int
) -> tuple[list[str], list[str], str, dict[str, Any]]:
    rng = random.Random(f"multi-key:{seed}:{target}")
    record_count = query_count * 4
    pairs = [
        (
            _digest_token("KEY", "mk", seed, target, index),
            _digest_token("VAL", "mv", seed, target, index),
        )
        for index in range(record_count)
    ]
    rng.shuffle(pairs)
    selected = sorted(rng.sample(pairs, query_count), key=lambda item: item[0])
    lines = [f"DATA-RECORD key={key} value={value}" for key, value in pairs]
    keys = [key for key, _ in selected]
    expected = [value for _, value in selected]
    question = (
        "Using only DATA-RECORD lines, return the values for these keys in any order: "
        + ", ".join(keys)
        + ". Output only the value tokens separated by |."
    )
    return (
        lines,
        expected,
        question,
        {"query_keys": keys, "token_pattern": r"VAL-[A-F0-9]{8}"},
    )


def _variable_task(
    seed: int, target: int, variable_count: int, steps: int
) -> tuple[list[str], list[str], str, dict[str, Any]]:
    rng = random.Random(f"variables:{seed}:{target}")
    names = [f"VAR-{index:02d}" for index in range(variable_count)]
    values = {
        name: _digest_token("VALV", "initial", seed, target, name) for name in names
    }
    lines = [
        f"STEP {index:02d}: SET {name} = {values[name]}"
        for index, name in enumerate(names)
    ]
    for step in range(variable_count, variable_count + steps):
        destination, source = rng.sample(names, 2)
        values[destination] = values[source]
        lines.append(f"STEP {step:02d}: SET {destination} = VALUE-OF {source}")
    query_names = tuple(names[index] for index in (0, 2, 5, 7) if index < len(names))
    expected = [values[name] for name in query_names]
    question = (
        "Execute STEP lines in numeric order. Return the final values of "
        + ", ".join(query_names)
        + " in any order. Output only value tokens separated by |."
    )
    return (
        lines,
        expected,
        question,
        {
            "query_variables": list(query_names),
            "steps": steps,
            "token_pattern": r"VALV-[A-F0-9]{8}",
        },
    )


def _aggregation_task(
    seed: int, target: int, label_count: int, top_k: int
) -> tuple[list[str], list[str], str, dict[str, Any]]:
    rng = random.Random(f"aggregation:{seed}:{target}")
    labels = [
        _digest_token("TAG", "aggregate", seed, target, index)
        for index in range(label_count)
    ]
    counts = {
        label: 3 + (label_count - index) * 2 for index, label in enumerate(labels)
    }
    events = [
        f"TAGGED-EVENT label={label}" for label in labels for _ in range(counts[label])
    ]
    rng.shuffle(events)
    expected = labels[:top_k]
    question = (
        f"Count only TAGGED-EVENT lines. Return the {top_k} labels with the highest "
        "frequency in any order. Output only label tokens separated by |."
    )
    return (
        events,
        expected,
        question,
        {
            "top_k": top_k,
            "frequencies": counts,
            "token_pattern": r"TAG-[A-F0-9]{8}",
        },
    )


def make_hard_rows(
    config: V2QualityConfig,
    codec: PromptCodec,
    *,
    split_seeds: Sequence[tuple[str, Sequence[int]]] | None = None,
) -> list[dict[str, Any]]:
    params = _hard_parameters(config)
    rows: list[dict[str, Any]] = []
    requested = split_seeds or (
        ("calibration", config.hard_calibration_seeds),
        ("heldout", config.hard_heldout_seeds),
    )
    for split, seeds in requested:
        if split not in {"calibration", "heldout"} or not seeds:
            raise ValueError("Hard split_seeds 无效")
        for family in config.hard_families:
            for target in config.target_lengths:
                for seed in seeds:
                    if family == "multi_key_value":
                        lines, expected, question, metadata = _multi_key_task(
                            seed, target, params["multi_key_query_count"]
                        )
                    elif family == "variable_tracking":
                        lines, expected, question, metadata = _variable_task(
                            seed,
                            target,
                            params["variable_count"],
                            params["variable_steps"],
                        )
                    else:
                        lines, expected, question, metadata = _aggregation_task(
                            seed,
                            target,
                            params["aggregation_label_count"],
                            params["aggregation_top_k"],
                        )

                    def compose(
                        repetitions: int,
                        padding_units: int,
                        *,
                        fixed_lines: tuple[str, ...] = tuple(lines),
                        fixed_question: str = question,
                    ) -> str:
                        return (
                            "Read the long evidence ledger. ARCHIVE-NOTE lines are "
                            "distractors and must not be treated as evidence.\n\n"
                            + _spread_lines(fixed_lines, repetitions, padding_units)
                            + "\n"
                            + fixed_question
                        )

                    fitted = fit_synthetic_prompt(
                        target, config.tolerance_tokens, compose, codec
                    )
                    digest = hashlib.sha256(
                        f"{family}:{target}:{seed}".encode("utf-8")
                    ).hexdigest()[:8]
                    rows.append(
                        _base_row(
                            sample_id=f"v2-hard-{family}-{split}-{target}-{seed}-{digest}",
                            split=split,
                            tier="hard",
                            task=family,
                            fitted=fitted,
                            target_tokens=target,
                            max_tokens=config.max_tokens_for(family),
                            expected_answers=expected,
                            answer_mode="set_f1",
                            metadata={
                                "seed": seed,
                                "difficulty": config.hard_difficulty,
                                **metadata,
                            },
                        )
                    )
    return rows


def _source_bucket(length: int, boundaries: Sequence[int | None]) -> int:
    for index, upper in enumerate(boundaries):
        if upper is None or length <= upper:
            return index
    raise ValueError("Natural 长度桶必须以 null 结束")


def _validate_longbench_row(row: Any, dataset: str) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{dataset} 源数据含非对象行")
    for field in ("_id", "input", "context"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"{dataset} 源数据缺少 {field}")
    answers = row.get("answers")
    if (
        not isinstance(answers, list)
        or not answers
        or any(not isinstance(answer, str) or not answer for answer in answers)
    ):
        raise ValueError(f"{dataset} 源数据 answers 无效")
    length = row.get("length")
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError(f"{dataset} 源数据 length 无效")
    return row


def make_natural_rows(
    config: V2QualityConfig,
    codec: PromptCodec,
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    natural_config = config.raw["data"]["natural"]
    boundaries = tuple(natural_config["source_length_buckets"])
    rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_contexts: set[str] = set()
    used_questions: set[str] = set()

    for dataset in config.natural_datasets:
        if dataset not in sources:
            raise ValueError(f"缺少 LongBench 源数据：{dataset}")
        buckets: dict[int, list[tuple[str, Mapping[str, Any], FittedPrompt]]] = {
            index: [] for index in range(len(boundaries))
        }
        observed_source_ids: set[str] = set()
        for raw in sources[dataset]:
            source = _validate_longbench_row(raw, dataset)
            source_id = str(source["_id"])
            if source_id in observed_source_ids:
                raise ValueError(f"{dataset} 源数据含重复 _id：{source_id}")
            observed_source_ids.add(source_id)
            user_prompt = LONG_BENCH_PROMPTS[dataset].format(
                context=source["context"], input=source["input"]
            )
            rendered, prompt_tokens = codec.render_and_count(user_prompt)
            if prompt_tokens > config.natural_max_input_tokens:
                continue
            fitted = FittedPrompt(user_prompt, rendered, prompt_tokens)
            bucket = _source_bucket(int(source["length"]), boundaries)
            rank = hashlib.sha256(f"{dataset}\0{source_id}".encode("utf-8")).hexdigest()
            buckets[bucket].append((rank, source, fitted))

        for bucket, candidates in buckets.items():
            candidates.sort(key=lambda item: (item[0], str(item[1]["_id"])))
            needed = config.natural_per_bucket_per_split * 2
            selected: list[tuple[str, Mapping[str, Any], FittedPrompt]] = []
            for candidate in candidates:
                source = candidate[1]
                source_id = str(source["_id"])
                context_hash = hashlib.sha256(
                    str(source["context"]).encode("utf-8")
                ).hexdigest()
                question_hash = hashlib.sha256(
                    str(source["input"]).encode("utf-8")
                ).hexdigest()
                if (
                    source_id in used_ids
                    or context_hash in used_contexts
                    or question_hash in used_questions
                ):
                    continue
                selected.append(candidate)
                used_ids.add(source_id)
                used_contexts.add(context_hash)
                used_questions.add(question_hash)
                if len(selected) == needed:
                    break
            if len(selected) != needed:
                raise ValueError(
                    f"{dataset} 长度桶 {bucket} 在 24576-token 过滤后不足 {needed} 条"
                )
            for offset, (_, source, fitted) in enumerate(selected):
                split = (
                    "calibration"
                    if offset < config.natural_per_bucket_per_split
                    else "heldout"
                )
                source_id = str(source["_id"])
                digest = hashlib.sha256(
                    f"{dataset}:{source_id}".encode("utf-8")
                ).hexdigest()[:12]
                rows.append(
                    _base_row(
                        sample_id=f"v2-natural-{dataset}-{digest}",
                        split=split,
                        tier="natural",
                        task=dataset,
                        fitted=fitted,
                        target_tokens=None,
                        max_tokens=config.max_tokens_for(dataset),
                        expected_answers=tuple(str(item) for item in source["answers"]),
                        answer_mode="qa_f1",
                        metadata={
                            "source_id": source_id,
                            "source_length": int(source["length"]),
                            "source_length_bucket": bucket,
                            "source_dataset": dataset,
                        },
                    )
                )
    return rows


def validate_v2_rows(
    config: V2QualityConfig, rows: Sequence[Mapping[str, Any]]
) -> None:
    expected_counts = {
        ("calibration", "easy"): 3,
        ("calibration", "hard"): 18,
        ("calibration", "natural"): 6,
        ("heldout", "easy"): 3,
        ("heldout", "hard"): 9,
        ("heldout", "natural"): 6,
    }
    counts: Counter[tuple[str, str]] = Counter()
    sample_ids: set[str] = set()
    content_hashes: set[str] = set()
    hard_combinations: set[tuple[str, int, int]] = set()
    natural_counts: Counter[str] = Counter()
    source_ids_by_split: dict[str, set[str]] = {"calibration": set(), "heldout": set()}
    for row in rows:
        split = row.get("split")
        tier = row.get("tier")
        if split not in {"calibration", "heldout"} or tier not in {
            "easy",
            "hard",
            "natural",
        }:
            raise ValueError("数据行 split/tier 无效")
        counts[(str(split), str(tier))] += 1
        sample_id = row.get("sample_id")
        content_hash = row.get("content_sha256")
        if not isinstance(sample_id, str) or sample_id in sample_ids:
            raise ValueError("数据含无效或重复 sample_id")
        if not isinstance(content_hash, str) or content_hash in content_hashes:
            raise ValueError("calibration/held-out 含重复内容")
        sample_ids.add(sample_id)
        content_hashes.add(content_hash)
        prompt = row.get("prompt")
        prompt_tokens = row.get("prompt_tokens")
        max_tokens = row.get("max_tokens")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{sample_id} 缺少 prompt")
        if not isinstance(prompt_tokens, int) or not isinstance(max_tokens, int):
            raise ValueError(f"{sample_id} token 字段无效")
        if prompt_tokens + max_tokens > config.max_model_len:
            raise ValueError(f"{sample_id} 超过模型上下文窗口")
        answers = row.get("expected_answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{sample_id} 缺少答案")
        if tier in {"easy", "hard"}:
            target = row.get("target_tokens")
            if (
                target not in config.target_lengths
                or abs(prompt_tokens - int(target)) > config.tolerance_tokens
            ):
                raise ValueError(f"{sample_id} 未命中冻结长度")
        if tier == "hard":
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{sample_id} 缺少 Hard metadata")
            hard_combinations.add(
                (str(row.get("task")), int(row["target_tokens"]), int(metadata["seed"]))
            )
        if tier == "natural":
            if prompt_tokens > config.natural_max_input_tokens:
                raise ValueError(f"{sample_id} 超过 Natural 输入上限")
            natural_counts[str(row.get("task"))] += 1
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping) or not isinstance(
                metadata.get("source_id"), str
            ):
                raise ValueError(f"{sample_id} 缺少 LongBench _id")
            source_id = str(metadata["source_id"])
            if source_id in source_ids_by_split[str(split)]:
                raise ValueError(f"{sample_id} 的 LongBench _id 重复")
            source_ids_by_split[str(split)].add(source_id)

    if dict(counts) != expected_counts:
        raise ValueError(f"v2 数据规模错误：{dict(counts)}")
    expected_hard = {
        (family, length, seed)
        for family in config.hard_families
        for length in config.target_lengths
        for seed in (*config.hard_calibration_seeds, *config.hard_heldout_seeds)
    }
    if hard_combinations != expected_hard:
        raise ValueError("Hard 数据未完整覆盖 3 任务 × 3 长度 × 3 seed")
    if natural_counts != Counter({dataset: 6 for dataset in config.natural_datasets}):
        raise ValueError(f"Natural 数据规模错误：{dict(natural_counts)}")
    if source_ids_by_split["calibration"] & source_ids_by_split["heldout"]:
        raise ValueError("Natural 两个 split 共享 _id")


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def _load_source_directory(
    config: V2QualityConfig, source_root: Path
) -> tuple[Mapping[str, Any], dict[str, list[Mapping[str, Any]]]]:
    manifest_path = source_root / "source-manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("LongBench source-manifest.json 不是对象")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("repository") != "THUDM/LongBench"
        or manifest.get("revision") != config.natural_source_revision
        or manifest.get("split") != "test"
        or manifest.get("datasets") != list(config.natural_datasets)
    ):
        raise ValueError("LongBench source manifest 身份与冻结配置不一致")
    files = manifest.get("files")
    row_counts = manifest.get("rows")
    if not isinstance(files, Mapping) or not isinstance(row_counts, Mapping):
        raise ValueError("LongBench source manifest 缺少 files 或 rows")
    sources: dict[str, list[Mapping[str, Any]]] = {}
    for dataset in config.natural_datasets:
        filename = f"{dataset}.jsonl"
        path = source_root / filename
        expected_hash = files.get(filename)
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            raise ValueError(f"LongBench 源文件 hash 不匹配：{filename}")
        rows = read_jsonl(path)
        if row_counts.get(dataset) != len(rows):
            raise ValueError(f"LongBench 源文件行数不匹配：{filename}")
        sources[dataset] = rows
    return manifest, sources


def freeze_v2_dataset(
    config: V2QualityConfig,
    codec: PromptCodec,
    source_root: Path,
    output_root: Path,
    *,
    config_path: Path,
) -> Mapping[str, Any]:
    source_manifest, sources = _load_source_directory(config, source_root)
    rows = [
        *make_easy_rows(config, codec),
        *make_hard_rows(config, codec),
        *make_natural_rows(config, codec, sources),
    ]
    rows.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["tier"]),
            str(row["task"]),
            str(row["sample_id"]),
        )
    )
    validate_v2_rows(config, rows)
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    calibration_path = output_root / "calibration.jsonl"
    heldout_path = output_root / "heldout.jsonl"
    atomic_write_text(calibration_path, _jsonl_text(calibration))
    atomic_write_text(heldout_path, _jsonl_text(heldout))
    split_records = {
        "calibration": {
            "path": "calibration.jsonl",
            "rows": len(calibration),
            "sha256": sha256_file(calibration_path),
        },
        "heldout": {
            "path": "heldout.jsonl",
            "rows": len(heldout),
            "sha256": sha256_file(heldout_path),
        },
    }
    source_identity = {
        key: source_manifest.get(key)
        for key in ("repository", "revision", "split", "datasets", "rows", "files")
    }
    identity = {
        "config_sha256": sha256_file(config_path),
        "source_identity_sha256": _sha256_json(source_identity),
        "chat_template_sha256": codec.template_sha256,
        "model_revision": config.model_revision,
        "longbench_revision": config.natural_source_revision,
        "splits": split_records,
    }
    manifest = {
        "schema_version": 2,
        "generator": "autokv-v2-data-v1",
        **identity,
        "dataset_sha256": _sha256_json(identity),
        "rows": len(rows),
        "tier_rows": {
            tier: sum(row["tier"] == tier for row in rows)
            for tier in ("easy", "hard", "natural")
        },
        "source": source_identity,
    }
    atomic_write_json(output_root / "dataset-manifest.json", manifest)
    return manifest


def load_frozen_v2_dataset(
    config: V2QualityConfig, output_root: Path, *, config_path: Path
) -> tuple[
    Mapping[str, Any], tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]
]:
    manifest = read_json(output_root / "dataset-manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 2:
        raise ValueError("v2 dataset manifest 无效")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("v2 dataset 使用了不同的质量配置")
    if manifest.get("model_revision") != config.model_revision:
        raise ValueError("v2 dataset 的模型 revision 不一致")
    split_records = manifest.get("splits")
    if not isinstance(split_records, Mapping):
        raise ValueError("v2 dataset manifest 缺少 split 记录")
    loaded: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for split, expected_rows in (("calibration", 27), ("heldout", 18)):
        record = split_records.get(split)
        if not isinstance(record, Mapping):
            raise ValueError(f"v2 dataset manifest 缺少 {split}")
        path = output_root / str(record.get("path", ""))
        if (
            path.parent.resolve() != output_root.resolve()
            or not path.is_file()
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"v2 dataset 的 {split} hash 不匹配")
        rows = tuple(read_jsonl(path))
        if len(rows) != expected_rows or record.get("rows") != expected_rows:
            raise ValueError(f"v2 dataset 的 {split} 行数错误")
        loaded[split] = rows
    all_rows = (*loaded["calibration"], *loaded["heldout"])
    validate_v2_rows(config, all_rows)
    identity = {
        key: manifest.get(key)
        for key in (
            "config_sha256",
            "source_identity_sha256",
            "chat_template_sha256",
            "model_revision",
            "longbench_revision",
            "splits",
        )
    }
    if manifest.get("dataset_sha256") != _sha256_json(identity):
        raise ValueError("v2 dataset 总身份 hash 不匹配")
    return manifest, loaded["calibration"], loaded["heldout"]
