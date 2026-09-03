"""AutoKV-Skip v2.0 唯一质量配置及其严格校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


V2_CONFIG_RELATIVE_PATH = Path("configs/v2/quality.json")
V2_DATA_RELATIVE_ROOT = Path("data/v2/quality")
V2_TIERS = ("easy", "hard", "natural")
V2_HARD_FAMILIES = (
    "multi_key_value",
    "variable_tracking",
    "aggregation_extraction",
)
V2_NATURAL_DATASETS = ("qasper_e", "hotpotqa_e")
V2_MODEL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
V2_LONGBENCH_REVISION = "92b6c5fbfb0c97b91e92d9ef79802f95ce74b05e"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是对象")
    return value


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} 必须是非空整数数组")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{field} 必须只包含整数")
    return tuple(value)


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} 必须是非空字符串数组")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} 必须只包含非空字符串")
    return tuple(value)


@dataclass(frozen=True)
class V2QualityConfig:
    """阶段 2–4 所有会影响结果的冻结输入。"""

    raw: Mapping[str, Any]
    model_id: str
    model_revision: str
    profile: str
    num_layers: int
    max_model_len: int
    target_lengths: tuple[int, ...]
    tolerance_tokens: int
    easy_calibration_seed: int
    easy_heldout_seed: int
    easy_depth: float
    hard_families: tuple[str, ...]
    hard_calibration_seeds: tuple[int, ...]
    hard_heldout_seeds: tuple[int, ...]
    hard_difficulty: str
    pilot_seed: int
    pilot_max_requests: int
    pilot_score_floor: float
    pilot_score_ceiling: float
    natural_datasets: tuple[str, ...]
    natural_source_revision: str
    natural_max_input_tokens: int
    natural_per_bucket_per_split: int
    candidate_budgets: tuple[int, ...]
    group_size: int
    top_groups: int
    random_seeds: tuple[int, ...]
    epsilon_global: float
    epsilon_tier: float
    bootstrap_samples: int
    calculate_kv_scales: bool
    enable_prefix_caching: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V2QualityConfig":
        root = _mapping(value, "配置")
        if root.get("schema_version") != 2 or root.get("name") != "quality-v2":
            raise ValueError("v2 配置必须为 schema_version=2、name=quality-v2")

        model = _mapping(root.get("model"), "model")
        runtime = _mapping(root.get("runtime"), "runtime")
        data = _mapping(root.get("data"), "data")
        easy = _mapping(data.get("easy"), "data.easy")
        hard = _mapping(data.get("hard"), "data.hard")
        hard_seeds = _mapping(hard.get("seeds"), "data.hard.seeds")
        pilot = _mapping(hard.get("pilot"), "data.hard.pilot")
        natural = _mapping(data.get("natural"), "data.natural")
        selection = _mapping(root.get("selection"), "selection")
        thresholds = _mapping(root.get("thresholds"), "thresholds")
        scoring = _mapping(root.get("scoring"), "scoring")

        model_id = str(model.get("id", ""))
        revision = str(model.get("revision", ""))
        profile = str(model.get("profile", ""))
        num_layers = int(model.get("num_layers", 0))
        max_model_len = int(model.get("max_model_len", 0))
        if model_id != "mistralai/Mistral-7B-Instruct-v0.3":
            raise ValueError("v2 模型必须固定为 Mistral-7B-Instruct-v0.3")
        if revision != V2_MODEL_REVISION:
            raise ValueError("model.revision 必须与 v1.0 实际模型 revision 一致")
        if profile != "full" or num_layers != 32 or max_model_len != 32768:
            raise ValueError("v2 必须使用 full 配置、32 层和 32768 上下文")

        if runtime.get("attention_backend") != "FLASHINFER":
            raise ValueError("runtime.attention_backend 必须是 FLASHINFER")
        if runtime.get("kv_cache_dtype") != "fp8_e4m3":
            raise ValueError("runtime.kv_cache_dtype 必须是 fp8_e4m3")
        if runtime.get("kv_cache_memory") != "16G":
            raise ValueError("runtime.kv_cache_memory 必须是 16G")
        if runtime.get("seed") != 42:
            raise ValueError("runtime.seed 必须固定为 42")
        calculate_scales = runtime.get("calculate_kv_scales")
        prefix_caching = runtime.get("enable_prefix_caching")
        if calculate_scales is not False:
            raise ValueError("runtime.calculate_kv_scales 必须固定为 false")
        if prefix_caching is not False:
            raise ValueError("v2 必须显式设置 enable_prefix_caching=false")

        target_lengths = _integer_tuple(
            data.get("target_lengths"), "data.target_lengths"
        )
        if target_lengths != (8192, 16384, 24576):
            raise ValueError("v2 上下文长度必须严格为 8192、16384、24576")
        tolerance = int(data.get("input_tolerance_tokens", 0))
        if tolerance != 32:
            raise ValueError("v2 输入长度误差必须固定为 32 tokens")

        easy_calibration_seed = int(easy.get("calibration_seed", -1))
        easy_heldout_seed = int(easy.get("heldout_seed", -1))
        easy_depth = float(easy.get("depth", -1.0))
        if (
            easy_calibration_seed != 101
            or easy_heldout_seed != 202
            or easy_depth != 0.5
            or easy.get("max_tokens") != 96
        ):
            raise ValueError("Easy 必须固定 seed=101/202、depth=0.5、max_tokens=96")

        families = _string_tuple(hard.get("families"), "data.hard.families")
        calibration_seeds = _integer_tuple(
            hard_seeds.get("calibration"), "data.hard.seeds.calibration"
        )
        heldout_seeds = _integer_tuple(
            hard_seeds.get("heldout"), "data.hard.seeds.heldout"
        )
        difficulty = str(hard.get("difficulty", ""))
        if families != V2_HARD_FAMILIES:
            raise ValueError("Hard 任务族或顺序与冻结设计不一致")
        if calibration_seeds != (41, 42) or heldout_seeds != (43,):
            raise ValueError("Hard seed 必须为 calibration=41/42、held-out=43")
        if difficulty not in {"easy", "standard", "hard"}:
            raise ValueError("Hard difficulty 只能是 easy、standard 或 hard")
        if hard.get("max_tokens") != 128:
            raise ValueError("Hard max_tokens 必须固定为 128")
        parameters = _mapping(hard.get("parameters"), "data.hard.parameters")
        if dict(parameters) != {
            "multi_key_query_count": 4,
            "variable_count": 8,
            "variable_steps": 24,
            "aggregation_label_count": 6,
            "aggregation_top_k": 3,
        }:
            raise ValueError("Hard 基准任务参数与冻结设计不一致")
        pilot_seed = int(pilot.get("seed", -1))
        pilot_max_requests = int(pilot.get("max_requests", 0))
        pilot_score_floor = float(pilot.get("score_floor", -1))
        pilot_score_ceiling = float(pilot.get("score_ceiling", -1))
        if (
            pilot_seed != 41
            or pilot_max_requests != 9
            or pilot_score_floor != 0.6
            or pilot_score_ceiling != 0.98
        ):
            raise ValueError(
                "BF16-only pilot 必须固定为 seed=41、9 请求、分数区间 [0.6,0.98)"
            )

        natural_datasets = _string_tuple(
            natural.get("datasets"), "data.natural.datasets"
        )
        source_revision = str(natural.get("source_revision", ""))
        max_input = int(natural.get("max_input_tokens", 0))
        per_bucket = int(natural.get("per_bucket_per_split", 0))
        if natural_datasets != V2_NATURAL_DATASETS:
            raise ValueError("Natural 数据集必须严格为 qasper_e、hotpotqa_e")
        if natural.get("repository") != "THUDM/LongBench":
            raise ValueError("Natural repository 必须固定为 THUDM/LongBench")
        if source_revision != V2_LONGBENCH_REVISION:
            raise ValueError("LongBench source_revision 与冻结数据源不一致")
        if natural.get("source_length_buckets") != [4000, 8000, None]:
            raise ValueError("Natural 长度桶必须固定为 4000、8000、无上界")
        if max_input != 24576 or per_bucket != 1:
            raise ValueError(
                "Natural 必须过滤到 24576 tokens，并在每个长度桶每 split 取 1 条"
            )
        natural_max_tokens = _mapping(
            natural.get("max_tokens"), "data.natural.max_tokens"
        )
        if dict(natural_max_tokens) != {"qasper_e": 128, "hotpotqa_e": 32}:
            raise ValueError("Natural 输出上限必须固定为 qasper_e=128、hotpotqa_e=32")

        budgets = _integer_tuple(
            selection.get("candidate_budgets"), "selection.candidate_budgets"
        )
        group_size = int(selection.get("group_size", 0))
        top_groups = int(selection.get("top_groups", 0))
        random_seeds = _integer_tuple(
            selection.get("random_seeds"), "selection.random_seeds"
        )
        if budgets != (0, 2, 4, 8, 32):
            raise ValueError("候选预算必须严格为 0、2、4、8、32")
        if group_size != 4 or top_groups != 2 or random_seeds != (11, 23, 37):
            raise ValueError("搜索必须使用 4 层组、前 2 组和固定的 3 个随机对照")

        epsilon_global = float(thresholds.get("epsilon_global", -1))
        epsilon_tier = float(thresholds.get("epsilon_tier", -1))
        if epsilon_global != 0.01 or epsilon_tier != 0.02:
            raise ValueError("质量阈值必须固定为 global=0.01、tier=0.02")
        if scoring.get("version") != "autokv-v2-score-v1":
            raise ValueError("不支持的 v2 评分版本")
        bootstrap_samples = int(scoring.get("bootstrap_samples", 0))
        if bootstrap_samples != 2000 or scoring.get("bootstrap_seed") != 20260902:
            raise ValueError("bootstrap 必须固定为 2000 次、seed=20260902")

        return cls(
            raw=dict(root),
            model_id=model_id,
            model_revision=revision,
            profile=profile,
            num_layers=num_layers,
            max_model_len=max_model_len,
            target_lengths=target_lengths,
            tolerance_tokens=tolerance,
            easy_calibration_seed=easy_calibration_seed,
            easy_heldout_seed=easy_heldout_seed,
            easy_depth=easy_depth,
            hard_families=families,
            hard_calibration_seeds=calibration_seeds,
            hard_heldout_seeds=heldout_seeds,
            hard_difficulty=difficulty,
            pilot_seed=pilot_seed,
            pilot_max_requests=pilot_max_requests,
            pilot_score_floor=pilot_score_floor,
            pilot_score_ceiling=pilot_score_ceiling,
            natural_datasets=natural_datasets,
            natural_source_revision=source_revision,
            natural_max_input_tokens=max_input,
            natural_per_bucket_per_split=per_bucket,
            candidate_budgets=budgets,
            group_size=group_size,
            top_groups=top_groups,
            random_seeds=random_seeds,
            epsilon_global=epsilon_global,
            epsilon_tier=epsilon_tier,
            bootstrap_samples=bootstrap_samples,
            calculate_kv_scales=calculate_scales,
            enable_prefix_caching=prefix_caching,
        )

    def max_tokens_for(self, task: str) -> int:
        data = _mapping(self.raw["data"], "data")
        if task == "niah":
            return int(_mapping(data["easy"], "data.easy")["max_tokens"])
        if task in self.hard_families:
            return int(_mapping(data["hard"], "data.hard")["max_tokens"])
        natural = _mapping(data["natural"], "data.natural")
        max_tokens = _mapping(natural["max_tokens"], "data.natural.max_tokens")
        if task not in max_tokens:
            raise ValueError(f"未知任务：{task}")
        return int(max_tokens[task])


def load_v2_config(path: Path) -> V2QualityConfig:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 v2 配置 {path}: {exc}") from exc
    return V2QualityConfig.from_dict(_mapping(value, "配置"))
