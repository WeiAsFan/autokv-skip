"""Deterministic Markdown, CSV and SVG experiment reports."""

from __future__ import annotations

import csv
import html
import io
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokv.benchmark import Capacity
from autokv.config import Profile
from autokv.io import atomic_write_text
from autokv.memory import (
    ideal_capacity_gain,
    kv_bytes_per_token,
    mixed_kv_bytes_per_token,
)


PERFORMANCE_CSV_FIELDS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p90_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p90_e2el_ms",
    "p99_e2el_ms",
)


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class QualityAggregate:
    samples: int
    exact_match: float
    quality_score: float
    answer_nll: float | None


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    samples: int = 10000,
) -> BootstrapCI:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must be non-empty and have equal length")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    generator = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        draw = [differences[generator.randrange(len(differences))] for _ in differences]
        bootstrapped.append(statistics.fmean(draw))
    return BootstrapCI(
        mean=statistics.fmean(differences),
        lower=_percentile(bootstrapped, 0.025),
        upper=_percentile(bootstrapped, 0.975),
    )


def aggregate_quality(rows: Sequence[Mapping[str, Any]]) -> QualityAggregate:
    if not rows:
        raise ValueError("quality rows cannot be empty")
    exact = [float(row["exact_match"]) for row in rows]
    quality = [float(row["quality_score"]) for row in rows]
    nll = [
        float(row["answer_nll"])
        for row in rows
        if isinstance(row.get("answer_nll"), (int, float))
        and math.isfinite(float(row["answer_nll"]))
    ]
    return QualityAggregate(
        samples=len(rows),
        exact_match=statistics.fmean(exact),
        quality_score=statistics.fmean(quality),
        answer_nll=statistics.fmean(nll) if nll else None,
    )


def _paired_quality(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> tuple[list[float], list[float]]:
    left_by_id = {str(row["sample_id"]): float(row["quality_score"]) for row in left}
    right_by_id = {
        str(row["sample_id"]): float(row["quality_score"]) for row in right
    }
    shared = sorted(set(left_by_id) & set(right_by_id))
    if len(shared) != len(left_by_id) or len(shared) != len(right_by_id):
        raise ValueError("quality configurations do not contain the same sample IDs")
    return [left_by_id[key] for key in shared], [right_by_id[key] for key in shared]


def _config_order(name: str) -> tuple[int, str]:
    preferred = {"bf16": 0, "fp8": 1, "auto-4": 2}
    return preferred.get(name, 3), name


def _format_float(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_capacity_svg(
    path: Path, capacities: Mapping[str, Capacity], baseline_name: str = "bf16"
) -> None:
    if baseline_name not in capacities:
        raise ValueError("capacity chart requires a BF16 baseline")
    names = sorted(capacities, key=_config_order)
    maximum = max(capacity.tokens for capacity in capacities.values())
    width = 860
    row_height = 72
    height = 80 + row_height * len(names)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="32" font-family="sans-serif" font-size="20" font-weight="bold">KV cache token capacity</text>',
    ]
    baseline = capacities[baseline_name].tokens
    for index, name in enumerate(names):
        capacity = capacities[name].tokens
        y = 58 + index * row_height
        bar_width = 600 * capacity / maximum
        color = "#2563eb" if name == "auto-4" else "#64748b"
        elements.extend(
            (
                f'<text x="24" y="{y + 24}" font-family="sans-serif" font-size="15">{html.escape(name)}</text>',
                f'<rect x="130" y="{y}" width="{bar_width:.1f}" height="30" rx="4" fill="{color}"/>',
                f'<text x="{140 + bar_width:.1f}" y="{y + 21}" font-family="sans-serif" font-size="14">{capacity:,} ({capacity / baseline:.2f}×)</text>',
            )
        )
    elements.append("</svg>")
    atomic_write_text(path, "\n".join(elements) + "\n")


def _summary_csv(
    aggregates: Mapping[str, QualityAggregate],
    capacities: Mapping[str, Capacity],
    performance: Mapping[str, Mapping[str, Any]],
) -> str:
    stream = io.StringIO(newline="")
    fields = (
        "config",
        "samples",
        "exact_match",
        "quality_score",
        "answer_nll",
        "capacity_tokens",
        "capacity_vs_bf16",
        *PERFORMANCE_CSV_FIELDS,
    )
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    baseline = capacities.get("bf16")
    for name in sorted(aggregates, key=_config_order):
        aggregate = aggregates[name]
        capacity = capacities.get(name)
        perf = performance.get(name, {})
        row = {
            "config": name,
            "samples": aggregate.samples,
            "exact_match": f"{aggregate.exact_match:.6f}",
            "quality_score": f"{aggregate.quality_score:.6f}",
            "answer_nll": "" if aggregate.answer_nll is None else f"{aggregate.answer_nll:.6f}",
            "capacity_tokens": "" if capacity is None else capacity.tokens,
            "capacity_vs_bf16": (
                ""
                if capacity is None or baseline is None
                else f"{capacity.tokens / baseline.tokens:.6f}"
            ),
        }
        row.update({field: perf.get(field, "") for field in PERFORMANCE_CSV_FIELDS})
        writer.writerow(row)
    return stream.getvalue()


def _format_metric(metrics: Mapping[str, Any], key: str) -> str:
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "N/A"
    return f"{float(value):.3f}"


def render_report(
    output_dir: Path,
    profile: Profile,
    lock: Mapping[str, Any],
    selection: Mapping[str, Any],
    quality: Mapping[str, Sequence[Mapping[str, Any]]],
    capacities: Mapping[str, Capacity],
    performance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    required = {"bf16", "fp8", "auto-4"}
    if not required.issubset(quality):
        raise ValueError("report requires bf16, fp8 and auto-4 quality rows")
    if not required.issubset(capacities):
        raise ValueError("report requires bf16, fp8 and auto-4 capacities")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = {name: aggregate_quality(rows) for name, rows in quality.items()}
    bf16_q = aggregates["bf16"].quality_score
    fp8_q = aggregates["fp8"].quality_score
    auto_q = aggregates["auto-4"].quality_score
    quality_gap = bf16_q - fp8_q
    recovery = None if quality_gap <= 0.01 else (auto_q - fp8_q) / quality_gap

    auto_left, fp8_right = _paired_quality(quality["auto-4"], quality["fp8"])
    auto_vs_fp8 = paired_bootstrap_ci(
        auto_left, fp8_right, seed=42, samples=10000
    )
    bf16_left, fp8_again = _paired_quality(quality["bf16"], quality["fp8"])
    bf16_vs_fp8 = paired_bootstrap_ci(
        bf16_left, fp8_again, seed=42, samples=10000
    )

    bf16_capacity = capacities["bf16"].tokens
    fp8_capacity_gain = capacities["fp8"].tokens / bf16_capacity
    auto_capacity_gain = capacities["auto-4"].tokens / bf16_capacity
    random_scores = [
        aggregate.quality_score
        for name, aggregate in aggregates.items()
        if name.startswith("random-4")
    ]
    random_median = statistics.median(random_scores) if random_scores else None
    inverted_q = (
        aggregates["inverted-4"].quality_score
        if "inverted-4" in aggregates
        else None
    )

    auto_layers = [int(layer) for layer in selection.get("auto_layers", [])]
    image_ref = str(lock.get("image_ref", "missing"))
    model_revision = str(lock.get("model_revision", "missing"))
    host = lock.get("host", {}) if isinstance(lock.get("host"), Mapping) else {}
    versions = (
        lock.get("versions", {}) if isinstance(lock.get("versions"), Mapping) else {}
    )

    lines = [
        "# AutoKV-Skip 实验报告",
        "",
        f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 结论摘要",
        "",
        f"- Auto-4 保留的 BF16 KV 层：{', '.join(map(str, auto_layers)) or '未记录'}。",
        f"- FP8 实测 KV token capacity / BF16：{fp8_capacity_gain:.3f}×。",
        f"- Auto-4 实测 KV token capacity / BF16：{auto_capacity_gain:.3f}×；理论值 {ideal_capacity_gain(profile.model.num_layers, profile.selection.k):.3f}×。",
        f"- Auto-4 相对全 FP8 的配对 Q 差：{auto_vs_fp8.mean:.4f}，95% CI [{auto_vs_fp8.lower:.4f}, {auto_vs_fp8.upper:.4f}]。",
    ]
    if recovery is None:
        lines.append(
            "- Gap recovery：BF16 与 FP8 的 Q 缺口不超过 0.01，本任务未观察到足够缺口，因此不计算恢复率。"
        )
    else:
        lines.append(f"- Gap recovery：{recovery * 100:.1f}%。")
    lines.extend(
        (
            "- A6000 不具备原生 FP8 Tensor Core；本项目的主结论是 KV 容量/带宽与质量折中，不预设延迟必然下降。",
            "",
            "## 不可变环境",
            "",
            f"- GPU：{host.get('gpu_name', 'missing')}",
            f"- 驱动：{host.get('driver', 'missing')}（项目不更新驱动）",
            f"- 镜像：`{image_ref}`",
            f"- 模型 revision：`{model_revision}`",
            f"- vLLM：{versions.get('vllm', 'missing')}；容器 CUDA：{versions.get('cuda', 'missing')}",
            "- 注意力后端：FLASHINFER；KV 预算：16G；随机 token scales：启用。",
            f"- 全实验质量评分模式：{selection.get('quality_mode', 'missing')}。",
            "",
            "## KV 显存公式",
            "",
            f"- BF16：{kv_bytes_per_token(32, 8, 128, 2):,} bytes/token。",
            f"- FP8：{kv_bytes_per_token(32, 8, 128, 1):,} bytes/token。",
            f"- Auto-4：{mixed_kv_bytes_per_token(32, 4, 8, 128):,} bytes/token。",
            "",
            "## 质量结果",
            "",
            "| 配置 | 样本数 | EM | Q | Answer NLL |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for name in sorted(aggregates, key=_config_order):
        aggregate = aggregates[name]
        lines.append(
            f"| {name} | {aggregate.samples} | {aggregate.exact_match:.4f} | {aggregate.quality_score:.4f} | {_format_float(aggregate.answer_nll)} |"
        )
    lines.extend(
        (
            "",
            "### 配对区间",
            "",
            f"- BF16 − FP8：{bf16_vs_fp8.mean:.4f}，95% CI [{bf16_vs_fp8.lower:.4f}, {bf16_vs_fp8.upper:.4f}]。",
            f"- Auto-4 − FP8：{auto_vs_fp8.mean:.4f}，95% CI [{auto_vs_fp8.lower:.4f}, {auto_vs_fp8.upper:.4f}]。",
            "",
            "## KV 容量",
            "",
            "| 配置 | KV tokens | 相对 BF16 | 最大并发 |",
            "|---|---:|---:|---:|",
        )
    )
    for name in sorted(capacities, key=_config_order):
        capacity = capacities[name]
        lines.append(
            f"| {name} | {capacity.tokens:,} | {capacity.tokens / bf16_capacity:.3f}× | {_format_float(capacity.max_concurrency, 2)} |"
        )
    lines.extend(
        (
            "",
            "## 服务性能",
            "",
            "以下值是固定输入/输出长度矩阵与重复运行的算术聚合；逐场景原始 JSON 保存在 `perf/`，不要把不同长度的聚合当成单一请求的延迟。",
            "",
            "### 吞吐",
            "",
            "| 配置 | Request throughput (req/s) | Output throughput (tok/s) | Total token throughput (tok/s) |",
            "|---|---:|---:|---:|",
        )
    )
    for name in ("bf16", "fp8", "auto-4"):
        metrics = performance.get(name, {})
        lines.append(
            f"| {name} | {_format_metric(metrics, 'request_throughput')} | "
            f"{_format_metric(metrics, 'output_throughput')} | "
            f"{_format_metric(metrics, 'total_token_throughput')} |"
        )
    lines.extend(
        (
            "",
            "### 首 token 与端到端延迟",
            "",
            "| 配置 | Median TTFT (ms) | P99 TTFT (ms) | Median E2E (ms) | P99 E2E (ms) |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for name in ("bf16", "fp8", "auto-4"):
        metrics = performance.get(name, {})
        lines.append(
            f"| {name} | {_format_metric(metrics, 'median_ttft_ms')} | "
            f"{_format_metric(metrics, 'p99_ttft_ms')} | "
            f"{_format_metric(metrics, 'median_e2el_ms')} | "
            f"{_format_metric(metrics, 'p99_e2el_ms')} |"
        )
    lines.extend(
        (
            "",
            "### Decode 延迟",
            "",
            "| 配置 | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | P99 ITL (ms) |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for name in ("bf16", "fp8", "auto-4"):
        metrics = performance.get(name, {})
        lines.append(
            f"| {name} | {_format_metric(metrics, 'median_tpot_ms')} | "
            f"{_format_metric(metrics, 'p99_tpot_ms')} | "
            f"{_format_metric(metrics, 'median_itl_ms')} | "
            f"{_format_metric(metrics, 'p99_itl_ms')} |"
        )

    checks = [
        ("FP8 capacity ≥ 1.85× BF16", fp8_capacity_gain >= 1.85),
        ("Auto-4 capacity ≥ 1.65× BF16", auto_capacity_gain >= 1.65),
        ("可测缺口时 recovery ≥ 50%", recovery is None or recovery >= 0.5),
        (
            "Auto-4 Q 高于 Random-4 中位数",
            random_median is None or auto_q > random_median,
        ),
        ("Auto-4 Q 高于 Inverted-4", inverted_q is None or auto_q > inverted_q),
    ]
    lines.extend(("", "## 预注册验收", "", "| 条件 | 结果 |", "|---|---|"))
    for label, passed in checks:
        lines.append(f"| {label} | {'通过' if passed else '未通过'} |")
    lines.extend(
        (
            "",
            "## 局限与诚实表述",
            "",
            "- `--calculate-kv-scales` 只从固定 seed 的 warmup random tokens 估计一次 scales，不代表数据集标定的最佳 FP8 上限。",
            "- quick profile 的 coarse-to-fine 搜索可能漏掉分散在低分组中的敏感层；full profile 用于检验这一点。",
            "- 单层边际敏感度近似忽略层间非加性交互，联合 Auto-4 的实测结果才是最终依据。",
            "- 如果 Auto-4 不优于随机或反向对照，应报告负结果，而不是更换指标或筛选运行。",
            "- FP8 在 Ampere 上需要存储转换；若 TTFT/ITL 变慢，应作为真实硬件结果报告。",
            "",
        )
    )

    markdown_path = output_dir / "REPORT.zh-CN.md"
    csv_path = output_dir / "summary.csv"
    svg_path = output_dir / "capacity.svg"
    atomic_write_text(markdown_path, "\n".join(lines))
    atomic_write_text(csv_path, _summary_csv(aggregates, capacities, performance))
    render_capacity_svg(svg_path, capacities)
    return {"markdown": markdown_path, "csv": csv_path, "svg": svg_path}
