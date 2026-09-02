"""Approved experiment profile schema and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPECTED_DRIVER = "580.173.02"
EXPECTED_IMAGES = (
    "vllm/vllm-openai:v0.26.0",
    "vllm/vllm-openai:v0.19.1",
)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _int_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{field} must contain integers")
    return tuple(value)


@dataclass(frozen=True)
class Hardware:
    driver: str
    gpu_name: str
    compute_capability: str
    vram_mib: int


@dataclass(frozen=True)
class Model:
    model_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    attention_backend: str
    max_model_len: int


@dataclass(frozen=True)
class Selection:
    mode: str
    k: int
    group_size: int
    top_groups: int
    random_seeds: tuple[int, ...]


@dataclass(frozen=True)
class Quality:
    probe_lengths: tuple[int, ...]
    final_lengths: tuple[int, ...]
    probe_depths: tuple[float, ...]
    final_depths: tuple[float, ...]
    probe_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    max_tokens: int


@dataclass(frozen=True)
class Benchmark:
    input_lengths: tuple[int, ...]
    output_lengths: tuple[int, ...]
    num_prompts: int
    repeats: int


@dataclass(frozen=True)
class Profile:
    schema_version: int
    name: str
    hardware: Hardware
    images: tuple[str, ...]
    model: Model
    selection: Selection
    quality: Quality
    benchmark: Benchmark
    kv_cache_dtype: str
    kv_cache_memory: str
    seed: int
    calculate_kv_scales: bool

    @classmethod
    def default_dict(cls, name: str) -> dict[str, Any]:
        if name not in {"quick", "full"}:
            raise ValueError("profile name must be quick or full")
        full = name == "full"
        return {
            "schema_version": 1,
            "name": name,
            "hardware": {
                "driver": EXPECTED_DRIVER,
                "gpu_name": "NVIDIA RTX A6000",
                "compute_capability": "8.6",
                "vram_mib": 49140,
            },
            "images": list(EXPECTED_IMAGES),
            "model": {
                "id": "mistralai/Mistral-7B-Instruct-v0.3",
                "num_layers": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "dtype": "bfloat16",
                "attention_backend": "FLASHINFER",
                "max_model_len": 32768,
            },
            "selection": {
                "mode": "full" if full else "coarse_to_fine",
                "k": 4,
                "group_size": 4,
                "top_groups": 8 if full else 2,
                "random_seeds": [11, 23, 37, 53, 71],
            },
            "quality": {
                "probe_lengths": [8192, 16384],
                "final_lengths": (
                    [4096, 8192, 16384, 24576, 30000]
                    if full
                    else [8192, 16384, 30000]
                ),
                "probe_depths": [0.2, 0.5, 0.8],
                "final_depths": [0.1, 0.5, 0.9],
                "probe_seeds": [42] if not full else [41, 42, 43],
                "final_seeds": [101, 202] if not full else [101, 202, 303],
                "max_tokens": 96,
            },
            "benchmark": {
                "input_lengths": [1024, 8192, 16384],
                "output_lengths": [32, 256],
                "num_prompts": 20 if not full else 50,
                "repeats": 1 if not full else 3,
            },
            "kv_cache_dtype": "fp8_e4m3",
            "kv_cache_memory": "16G",
            "seed": 42,
            "calculate_kv_scales": False,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Profile":
        root = _mapping(data, "profile")
        hardware_data = _mapping(root.get("hardware"), "hardware")
        model_data = _mapping(root.get("model"), "model")
        selection_data = _mapping(root.get("selection"), "selection")
        quality_data = _mapping(root.get("quality"), "quality")
        benchmark_data = _mapping(root.get("benchmark"), "benchmark")

        driver = str(hardware_data.get("driver", ""))
        if driver != EXPECTED_DRIVER:
            raise ValueError(
                f"driver must remain exactly {EXPECTED_DRIVER}; observed {driver or 'missing'}"
            )

        images_value = root.get("images")
        if not isinstance(images_value, list) or tuple(images_value) != EXPECTED_IMAGES:
            raise ValueError(f"images must be fixed in this order: {EXPECTED_IMAGES}")

        name = str(root.get("name", ""))
        if name not in {"quick", "full"}:
            raise ValueError("profile name must be quick or full")
        approved = cls.default_dict(name)
        if root != approved:
            raise ValueError(
                f"profile must match the approved {name} profile exactly; "
                "do not tune parameters between configurations"
            )

        hardware = Hardware(
            driver=driver,
            gpu_name=str(hardware_data.get("gpu_name", "")),
            compute_capability=str(hardware_data.get("compute_capability", "")),
            vram_mib=int(hardware_data.get("vram_mib", 0)),
        )
        model = Model(
            model_id=str(model_data.get("id", "")),
            num_layers=int(model_data.get("num_layers", 0)),
            num_kv_heads=int(model_data.get("num_kv_heads", 0)),
            head_dim=int(model_data.get("head_dim", 0)),
            dtype=str(model_data.get("dtype", "")),
            attention_backend=str(model_data.get("attention_backend", "")),
            max_model_len=int(model_data.get("max_model_len", 0)),
        )
        selection = Selection(
            mode=str(selection_data.get("mode", "")),
            k=int(selection_data.get("k", 0)),
            group_size=int(selection_data.get("group_size", 0)),
            top_groups=int(selection_data.get("top_groups", 0)),
            random_seeds=_int_tuple(
                selection_data.get("random_seeds"), "selection.random_seeds"
            ),
        )
        quality = Quality(
            probe_lengths=_int_tuple(
                quality_data.get("probe_lengths"), "quality.probe_lengths"
            ),
            final_lengths=_int_tuple(
                quality_data.get("final_lengths"), "quality.final_lengths"
            ),
            probe_depths=tuple(float(item) for item in quality_data.get("probe_depths", [])),
            final_depths=tuple(float(item) for item in quality_data.get("final_depths", [])),
            probe_seeds=_int_tuple(
                quality_data.get("probe_seeds"), "quality.probe_seeds"
            ),
            final_seeds=_int_tuple(
                quality_data.get("final_seeds"), "quality.final_seeds"
            ),
            max_tokens=int(quality_data.get("max_tokens", 0)),
        )
        benchmark = Benchmark(
            input_lengths=_int_tuple(
                benchmark_data.get("input_lengths"), "benchmark.input_lengths"
            ),
            output_lengths=_int_tuple(
                benchmark_data.get("output_lengths"), "benchmark.output_lengths"
            ),
            num_prompts=int(benchmark_data.get("num_prompts", 0)),
            repeats=int(benchmark_data.get("repeats", 0)),
        )

        approved_model = (
            model.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
            and model.num_layers == 32
            and model.num_kv_heads == 8
            and model.head_dim == 128
            and model.dtype == "bfloat16"
            and model.attention_backend == "FLASHINFER"
            and model.max_model_len == 32768
        )
        if not approved_model:
            raise ValueError("model settings differ from the approved Mistral/FlashInfer profile")
        if selection.k != 4 or selection.group_size != 4:
            raise ValueError("selection must keep exactly four BF16 layers in groups of four")
        if str(root.get("kv_cache_memory")) != "16G":
            raise ValueError("kv_cache_memory must remain 16G")
        if str(root.get("kv_cache_dtype")) != "fp8_e4m3":
            raise ValueError("kv_cache_dtype must remain fp8_e4m3")

        return cls(
            schema_version=int(root.get("schema_version", 0)),
            name=name,
            hardware=hardware,
            images=tuple(images_value),
            model=model,
            selection=selection,
            quality=quality,
            benchmark=benchmark,
            kv_cache_dtype="fp8_e4m3",
            kv_cache_memory="16G",
            seed=int(root.get("seed", 0)),
            calculate_kv_scales=bool(root.get("calculate_kv_scales", False)),
        )


def load_profile(path: Path) -> Profile:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load profile {path}: {exc}") from exc
    return Profile.from_dict(_mapping(data, "profile"))
