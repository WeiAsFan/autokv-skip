"""Deterministic AutoKV-Skip probe construction and layer selection."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Variant:
    name: str
    kv_dtype: str
    skip_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kv_dtype not in {"bfloat16", "fp8_e4m3"}:
            raise ValueError(f"unsupported KV dtype: {self.kv_dtype}")
        if any(layer < 0 for layer in self.skip_layers):
            raise ValueError("skip layers must be non-negative")
        normalized = tuple(sorted(set(self.skip_layers)))
        if len(normalized) != len(self.skip_layers):
            raise ValueError("skip layers must be unique")
        object.__setattr__(self, "skip_layers", normalized)
        if self.kv_dtype == "bfloat16" and normalized:
            raise ValueError("BF16 variant cannot specify quantization skip layers")

    @classmethod
    def bf16(cls) -> "Variant":
        return cls("bf16", "bfloat16")

    @classmethod
    def fp8(cls) -> "Variant":
        return cls("fp8", "fp8_e4m3")

    @classmethod
    def mixed(cls, name: str, skip_layers: Sequence[int]) -> "Variant":
        if not skip_layers:
            raise ValueError("mixed variant requires at least one BF16 skip layer")
        return cls(name, "fp8_e4m3", tuple(skip_layers))

    def validate_for_model(self, num_layers: int) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if any(layer >= num_layers for layer in self.skip_layers):
            raise ValueError(f"skip layer is outside model range 0..{num_layers - 1}")


def canonical_config_id(variant: Variant) -> str:
    payload = {
        "kv_dtype": variant.kv_dtype,
        "skip_layers": list(variant.skip_layers),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def group_layers(num_layers: int, group_size: int) -> tuple[tuple[int, ...], ...]:
    if num_layers <= 0 or group_size <= 0 or num_layers % group_size:
        raise ValueError("num_layers must be evenly divisible by group_size")
    return tuple(
        tuple(range(start, start + group_size))
        for start in range(0, num_layers, group_size)
    )


def group_probe_variants(num_layers: int, group_size: int) -> tuple[Variant, ...]:
    return tuple(
        Variant.mixed(f"group-{index:02d}", layers)
        for index, layers in enumerate(group_layers(num_layers, group_size))
    )


def layer_probe_variants(layers: Sequence[int]) -> tuple[Variant, ...]:
    normalized = tuple(sorted(set(layers)))
    if len(normalized) != len(layers):
        raise ValueError("candidate layers must be unique")
    return tuple(Variant.mixed(f"layer-{layer:02d}", (layer,)) for layer in normalized)


def select_top_layers(scores: Mapping[int, float], k: int) -> tuple[int, ...]:
    if k <= 0 or k > len(scores):
        raise ValueError("k must select at least one available layer")
    ranked = sorted(scores, key=lambda layer: (-scores[layer], layer))
    return tuple(sorted(ranked[:k]))


def select_bottom_layers(scores: Mapping[int, float], k: int) -> tuple[int, ...]:
    if k <= 0 or k > len(scores):
        raise ValueError("k must select at least one available layer")
    ranked = sorted(scores, key=lambda layer: (scores[layer], layer))
    return tuple(sorted(ranked[:k]))


def select_top_groups(
    scores: Mapping[tuple[int, ...], float], count: int
) -> tuple[tuple[int, ...], ...]:
    if count <= 0 or count > len(scores):
        raise ValueError("count must select at least one available group")
    ranked = sorted(scores, key=lambda group: (-scores[group], group))
    return tuple(ranked[:count])


def random_controls(
    num_layers: int,
    k: int,
    seeds: Sequence[int],
    forbidden: set[tuple[int, ...]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    if num_layers <= 0 or not 0 < k <= num_layers:
        raise ValueError("k must be between one and num_layers")
    if not seeds:
        raise ValueError("at least one random seed is required")

    blocked = {tuple(sorted(item)) for item in (forbidden or set())}
    if math.comb(num_layers, k) < len(blocked) + len(seeds):
        raise ValueError("not enough unique layer combinations for requested controls")

    chosen: list[tuple[int, ...]] = []
    used = set(blocked)
    for base_seed in seeds:
        candidate_seed = int(base_seed)
        while True:
            layers = tuple(sorted(random.Random(candidate_seed).sample(range(num_layers), k)))
            if layers not in used:
                chosen.append(layers)
                used.add(layers)
                break
            candidate_seed += 1
    return tuple(chosen)
