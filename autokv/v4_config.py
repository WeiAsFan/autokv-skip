"""v4 构造参数与冻结后的单条件正式协议。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from autokv.io import read_json
from autokv.v3_config import V3Config

CONFIG_PATH = Path("configs/v4.0/quality.json")
SCORE_VERSION = "autokv-v4-short-code-v1"
TASKS = ("single_lookup", "two_hop_lookup")


def condition(task, length, records):
    return {"condition_id": f"{task}-l{length}-r{records}", "task": task,
            "length_bucket": length, "record_count": records}


@dataclass(frozen=True)
class V4Config(V3Config):
    selected_condition: dict | None = None
    data_directory: str = "data/v4.0"

    @classmethod
    def from_dict(cls, raw):
        raw = json.loads(json.dumps(raw, allow_nan=False))
        if raw.get("schema_version") != 4 or raw["scoring"]["version"] != SCORE_VERSION:
            raise ValueError("不支持的 v4 配置或评分版本")
        obj = cls(raw)
        c, d, s = raw["construction"], raw["data"], raw["search"]
        if not c["tasks"] or len(set(c["tasks"])) != len(c["tasks"]) or not set(c["tasks"]) <= set(TASKS):
            raise ValueError("构造任务必须是明确且不重复的 v4 任务")
        for values in (c["target_lengths"], c["record_counts"]):
            if not values or any(type(n) is not int or n <= 0 for n in values) or sorted(set(values)) != values:
                raise ValueError("长度和记录量必须为递增的正整数")
        if min(c["record_counts"]) < 2:
            raise ValueError("检索至少需要目标与一个干扰记录")
        positive = (obj.num_layers, raw["model"]["num_kv_heads"], raw["model"]["head_dim"],
                    raw["model"]["max_model_len"], obj.max_tokens, d["experiment_samples"], d["test_samples"],
                    c["discovery_per_condition"], c["confirmation_per_batch"], obj.beam_width, obj.bootstrap_samples)
        if any(type(n) is not int or n <= 0 for n in positive):
            raise ValueError("模型维度、样本数和计算次数必须是正整数")
        if c["confirmation_batches"] != 2 or type(c["max_confirmation_conditions"]) is not int or not 1 <= c["max_confirmation_conditions"] <= 2:
            raise ValueError("v4 使用两批新确认样本，最多确认两个条件")
        if not 0 < c["min_bf16_score"] <= 1 or not 0.10 <= c["min_fp8_gap"] < 1:
            raise ValueError("BF16 目标应处于 (0,1]；FP8 恢复需求至少为 0.10 且小于 1")
        if type(obj.tolerance) is not int or obj.tolerance < 0 or max(obj.lengths)+obj.tolerance+obj.max_tokens > raw["model"]["max_model_len"]:
            raise ValueError("实际输入与输出空间超过模型长度上限")
        if (len(obj.fidelities) != 3 or any(type(n) is not int or n <= 0 for n in obj.fidelities)
                or tuple(sorted(set(obj.fidelities))) != obj.fidelities or obj.fidelities[-1] != obj.experiment_size):
            raise ValueError("需要三个递增前缀，末级等于完整实验集")
        if s["schedule"] not in {"progressive", "full"} or not 0 < s["promotion_win_fraction"] <= 1:
            raise ValueError("搜索调度或晋级比例无效")
        if not 1 < obj.capacity_ratio <= 2 or any(not 0 < v <= 1 for v in obj.epsilons.values()):
            raise ValueError("质量容差或容量目标无效")
        for key in ("max_requests", "max_seconds", "max_bf16_layers"):
            value = s.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"search.{key} 必须为非负预算或 null")
            if key != "max_seconds" and value is not None and type(value) is not int:
                raise ValueError(f"search.{key} 必须为整数或 null")
        if not d["sources"] or type(obj.seed) is not int or type(raw["scoring"]["bootstrap_seed"]) is not int:
            raise ValueError("需要来源名称与整数种子")
        runtime = raw["runtime"]
        if (runtime["enable_prefix_caching"] is not False or runtime["calculate_kv_scales"] is not False
                or runtime["attention_backend"] != "FLASHINFER" or runtime["kv_cache_dtype"] != "fp8_e4m3" or runtime["seed"] != 42):
            raise ValueError("v4 沿用 FLASHINFER、固定 scale 的 FP8 E4M3、关闭 prefix caching 和 seed=42")
        return obj

    @property
    def tasks(self):
        return (self.selected_condition["task"],) if self.selected_condition else tuple(self.raw["construction"]["tasks"])

    @property
    def lengths(self):
        return (self.selected_condition["length_bucket"],) if self.selected_condition else tuple(self.raw["construction"]["target_lengths"])

    @property
    def data_root(self): return Path(self.data_directory)

    @property
    def experiment_size(self): return self.raw["data"]["experiment_samples"]

    def per_cell(self, split):
        return self.raw["data"][f"{split}_samples"]

    def conditions(self):
        c = self.raw["construction"]
        return [condition(t, l, n) for t in c["tasks"] for l in c["target_lengths"] for n in c["record_counts"]]

    def for_condition(self, chosen, *, data_directory=None):
        if chosen not in self.conditions():
            raise ValueError("冻结条件不属于本轮构造范围")
        return replace(self, selected_condition=dict(chosen), data_directory=data_directory or self.data_directory)


def load_config(root):
    return V4Config.from_dict(read_json(Path(root) / CONFIG_PATH))
