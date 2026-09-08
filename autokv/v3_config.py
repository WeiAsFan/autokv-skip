"""v3 质量协议；只校验结果正确性所需的配置关系。"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from autokv.config import Model, Profile
from autokv.io import read_json

TASKS = ("multi_key", "multi_value", "qa_single", "qa_multi")
CONFIG_PATH = Path("configs/v3.0/quality.json")
SCORE_VERSION = "autokv-v3-score-v1"


def identity(value) -> str:
    """用于自动识别输入/缓存，不要求操作者核对摘要。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class V3Config:
    raw: dict

    @classmethod
    def from_dict(cls, raw):
        raw = json.loads(json.dumps(raw, allow_nan=False))
        if raw.get("schema_version") != 3 or raw["scoring"]["version"] != SCORE_VERSION:
            raise ValueError("不支持的 v3 配置/评分版本")
        obj = cls(raw)
        if tuple(raw["data"]["tasks"]) != TASKS:
            raise ValueError("v3 必须包含四个主任务")
        for value in (*obj.lengths, obj.num_layers, obj.per_cell("experiment"),
                      obj.per_cell("test"), obj.per_cell("development"), obj.max_tokens,
                      obj.beam_width, obj.bootstrap_samples, raw["model"]["num_kv_heads"],
                      raw["model"]["head_dim"]):
            if type(value) is not int or value <= 0:
                raise ValueError("题数、长度、模型维度和计算预算必须为正整数")
        if len(obj.lengths) != 4 or tuple(sorted(set(obj.lengths))) != obj.lengths:
            raise ValueError("需要四个递增的长度单元")
        if obj.tolerance < 0 or max(obj.lengths) + obj.tolerance + obj.max_tokens > raw["model"]["max_model_len"]:
            raise ValueError("输入长度与输出空间超过模型上限")
        if any(type(n) is not int or n <= 0 or n % obj.cells for n in obj.fidelities):
            raise ValueError("实验前缀须为分层单元数的正整数倍")
        if len(obj.fidelities) != 3 or tuple(sorted(set(obj.fidelities))) != obj.fidelities or obj.fidelities[-1] != obj.experiment_size:
            raise ValueError("需要三个递增前缀，末级等于完整实验集")
        if raw["search"]["schedule"] not in {"progressive", "full"}:
            raise ValueError("未知评估调度")
        if not 1 < obj.capacity_ratio <= 2 or any(not 0 < v <= 1 for v in obj.epsilons.values()):
            raise ValueError("质量容差或容量目标超出有效范围")
        if not 0 < raw["search"]["promotion_win_fraction"] <= 1:
            raise ValueError("晋级比例应处于 (0,1]")
        runtime = raw["runtime"]
        if runtime["enable_prefix_caching"] is not False or runtime["calculate_kv_scales"] is not False:
            raise ValueError("首版协议关闭 prefix caching，使用固定未校准 scale")
        if runtime["attention_backend"] != "FLASHINFER" or runtime["kv_cache_dtype"] != "fp8_e4m3" or runtime["seed"] != 42:
            raise ValueError("首版使用 FLASHINFER、FP8 E4M3 和生成 seed=42")
        for key in ("max_requests", "max_seconds", "max_bf16_layers"):
            value = raw["search"].get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"search.{key} 必须为非负预算或 null")
        if raw["search"].get("max_bf16_layers") is not None and type(raw["search"]["max_bf16_layers"]) is not int:
            raise ValueError("层数上限必须为整数")
        params = raw["data"]["retrieval"]
        if not 1 <= params["query_keys"] <= params["key_records"] or params["values"] < 1 or params["distractors"] < 0:
            raise ValueError("检索任务结构无效")
        return obj

    @property
    def num_layers(self): return self.raw["model"]["num_layers"]
    @property
    def model_id(self): return self.raw["model"]["id"]
    @property
    def model_revision(self): return self.raw["model"]["revision"]
    @property
    def lengths(self): return tuple(self.raw["data"]["target_lengths"])
    @property
    def tolerance(self): return self.raw["data"]["input_tolerance_tokens"]
    @property
    def max_tokens(self): return self.raw["data"]["max_tokens"]
    @property
    def cells(self): return len(TASKS) * len(self.lengths)
    def per_cell(self, split): return self.raw["data"]["per_cell"][split]
    @property
    def experiment_size(self): return self.cells * self.per_cell("experiment")
    @property
    def fidelities(self): return tuple(self.raw["search"]["fidelity_samples"])
    @property
    def beam_width(self): return self.raw["search"]["beam_width"]
    @property
    def bootstrap_samples(self): return self.raw["scoring"]["bootstrap_samples"]
    @property
    def seed(self): return self.raw["data"]["seed"]
    @property
    def capacity_ratio(self): return self.raw["thresholds"]["min_capacity_ratio"]
    @property
    def epsilons(self):
        return {"all": self.raw["thresholds"]["epsilon_global"],
                **{t: self.raw["thresholds"]["epsilon_task"] for t in TASKS}}
    @property
    def max_layers(self):
        limit = math.floor(2 * self.num_layers / self.capacity_ratio - self.num_layers + 1e-12)
        requested = self.raw["search"].get("max_bf16_layers")
        return min(limit, requested if requested is not None else limit)

    def profile(self):
        model = self.raw["model"]
        return replace(Profile.from_dict(Profile.default_dict("full")),
                       model=Model(self.model_id, self.num_layers, model["num_kv_heads"],
                                   model["head_dim"], "bfloat16", "FLASHINFER", model["max_model_len"]),
                       kv_cache_memory=self.raw["runtime"]["kv_cache_memory"],
                       calculate_kv_scales=False, seed=42)


def load_config(root: Path) -> V3Config:
    return V3Config.from_dict(read_json(root / CONFIG_PATH))
