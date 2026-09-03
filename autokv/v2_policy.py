"""AutoKV-Skip v2.0 策略表示与确定性搜索构造。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from autokv.selection import Variant, canonical_config_id, group_layers, random_controls


@dataclass(frozen=True)
class Policy:
    name: str
    bf16_layers: tuple[int, ...]
    num_layers: int = 32

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(self.bf16_layers)))
        if normalized != self.bf16_layers:
            raise ValueError("BF16 层必须唯一并按升序排列")
        if any(layer < 0 or layer >= self.num_layers for layer in normalized):
            raise ValueError("BF16 层超出模型范围")
        if not self.name:
            raise ValueError("策略名不能为空")

    @property
    def k(self) -> int:
        return len(self.bf16_layers)

    @property
    def variant(self) -> Variant:
        if self.k == 0:
            return Variant.fp8()
        if self.k == self.num_layers:
            return Variant.bf16()
        return Variant.mixed(self.name, self.bf16_layers)

    @property
    def config_id(self) -> str:
        return canonical_config_id(self.variant)

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "config_id": self.config_id,
            "k": self.k,
            "bf16_layers": list(self.bf16_layers),
            "fp8_layers": [
                layer
                for layer in range(self.num_layers)
                if layer not in self.bf16_layers
            ],
        }


def endpoint_policies(num_layers: int = 32) -> tuple[Policy, Policy]:
    return (
        Policy("p32", tuple(range(num_layers)), num_layers),
        Policy("p0", (), num_layers),
    )


def group_policies(num_layers: int = 32, group_size: int = 4) -> tuple[Policy, ...]:
    return tuple(
        Policy(f"group-{index:02d}-p{group_size}", group, num_layers)
        for index, group in enumerate(group_layers(num_layers, group_size))
    )


def layer_policies(groups: Sequence[Policy]) -> tuple[Policy, ...]:
    layers = sorted({layer for group in groups for layer in group.bf16_layers})
    return tuple(
        Policy(f"layer-{layer:02d}-p1", (layer,), groups[0].num_layers)
        for layer in layers
    )


def rank_by_recovery(
    policy_scores: Mapping[Policy, float], p0_score: float
) -> tuple[dict[str, object], ...]:
    ranked = sorted(
        policy_scores,
        key=lambda policy: (
            -(float(policy_scores[policy]) - p0_score),
            policy.bf16_layers,
        ),
    )
    return tuple(
        {
            "rank": index,
            "policy": policy.record(),
            "s_v2": float(policy_scores[policy]),
            "recovery_vs_p0": float(policy_scores[policy]) - p0_score,
        }
        for index, policy in enumerate(ranked, start=1)
    )


def nested_budget_policy(
    ranked_layers: Sequence[int], k: int, num_layers: int = 32
) -> Policy:
    if k not in {2, 4, 8}:
        raise ValueError("v2 中间预算只能是 2、4、8")
    if len(ranked_layers) < k or len(set(ranked_layers)) != len(ranked_layers):
        raise ValueError("层排名不足或包含重复层")
    layers = tuple(sorted(ranked_layers[:k]))
    return Policy(f"selected-p{k}", layers, num_layers)


def random_control_policies(
    selected: Policy, seeds: Sequence[int]
) -> tuple[Policy, ...]:
    if selected.k not in {2, 4, 8}:
        raise ValueError("只有中间策略需要随机同预算对照")
    layer_sets = random_controls(
        selected.num_layers,
        selected.k,
        seeds,
        forbidden={selected.bf16_layers},
    )
    return tuple(
        Policy(f"random-p{selected.k}-s{seed}", layers, selected.num_layers)
        for seed, layers in zip(seeds, layer_sets)
    )


def theoretical_capacity(
    policy: Policy, *, num_kv_heads: int = 8, head_dim: int = 128
) -> dict[str, float | int]:
    bf16_bytes = 2
    fp8_bytes = 1
    per_layer_elements = 2 * num_kv_heads * head_dim
    bytes_per_token = per_layer_elements * (
        policy.k * bf16_bytes + (policy.num_layers - policy.k) * fp8_bytes
    )
    bf16_total = per_layer_elements * policy.num_layers * bf16_bytes
    return {
        "bytes_per_token": bytes_per_token,
        "capacity_ratio_vs_p32": bf16_total / bytes_per_token,
    }
