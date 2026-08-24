"""Command-line workflow for the driver-locked AutoKV-Skip experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import statistics
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokv.benchmark import BenchmarkRunner, Capacity, build_benchmark_matrix
from autokv.client import VllmHttpError
from autokv.commands import format_command, run_command, server_command
from autokv.config import EXPECTED_DRIVER, Profile, load_profile
from autokv.doctor import (
    DoctorError,
    lock_first_compatible_image,
    parse_gpu_csv,
    validate_host,
)
from autokv.experiment import ExperimentRunner, canonical_run_id
from autokv.io import (
    atomic_write_json,
    atomic_write_text,
    ensure_within,
    read_json,
    read_jsonl,
    redact,
    sha256_file,
)
from autokv.niah import NiahCase, expected_answer, make_cases
from autokv.report import render_report
from autokv.selection import (
    Variant,
    group_probe_variants,
    random_controls,
    select_bottom_layers,
    select_top_groups,
    select_top_layers,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_STEPS = (
    "doctor",
    "lock-image",
    "make-data",
    "smoke",
    "probe",
    "select",
    "evaluate",
    "benchmark",
    "report",
)
EXIT_INVALID = 2
EXIT_EXTERNAL = 3
EXIT_HTTP = 4
EXIT_INCOMPLETE = 5


class IncompleteDataError(RuntimeError):
    """Raised when a prerequisite or artifact set is not complete."""


@dataclass(frozen=True)
class RuntimeContext:
    root: Path
    profile: Profile
    profile_path: Path
    profile_hash: str
    manifest: Mapping[str, Any]
    dataset_hash: str
    lock: Mapping[str, Any]
    run_id: str


def _resolved_root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _profile_path(root: Path, name: str) -> Path:
    project_candidate = root / "configs" / f"{name}.json"
    if project_candidate.is_file():
        return project_candidate
    packaged_candidate = REPOSITORY_ROOT / "configs" / f"{name}.json"
    if packaged_candidate.is_file():
        return packaged_candidate
    raise ValueError(f"missing approved profile: configs/{name}.json")


def _load_named_profile(root: Path, name: str) -> tuple[Profile, Path]:
    path = _profile_path(root, name)
    profile = load_profile(path)
    if profile.name != name:
        raise ValueError(f"profile file name does not match profile.name: {path}")
    return profile, path


def _data_paths(root: Path, profile_name: str) -> tuple[Path, Path, Path]:
    directory = ensure_within(root, root / "data" / "niah")
    return (
        directory / f"{profile_name}-probe.jsonl",
        directory / f"{profile_name}-final.jsonl",
        directory / f"{profile_name}-manifest.json",
    )


def _case_row(case: NiahCase) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": case.sample_id,
        "target_tokens": case.target_tokens,
        "depth": case.depth,
        "seed": case.seed,
        "code": case.code,
        "expected": expected_answer(case.code),
    }


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def make_data(root: Path, profile_name: str) -> dict[str, Any]:
    profile, profile_path = _load_named_profile(root, profile_name)
    probe_path, final_path, manifest_path = _data_paths(root, profile.name)
    probe_rows = [_case_row(case) for case in make_cases(profile, "probe")]
    final_rows = [_case_row(case) for case in make_cases(profile, "final")]
    atomic_write_text(probe_path, _jsonl_text(probe_rows))
    atomic_write_text(final_path, _jsonl_text(final_rows))
    profile_hash = sha256_file(profile_path)
    probe_hash = sha256_file(probe_path)
    final_hash = sha256_file(final_path)
    encoded = "\n".join((profile_hash, probe_hash, final_hash)).encode("utf-8")
    dataset_hash = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "schema_version": 1,
        "profile": profile.name,
        "profile_sha256": profile_hash,
        "model_id": profile.model.model_id,
        "probe": {
            "path": probe_path.relative_to(root).as_posix(),
            "sha256": probe_hash,
            "samples": len(probe_rows),
        },
        "final": {
            "path": final_path.relative_to(root).as_posix(),
            "sha256": final_hash,
            "samples": len(final_rows),
        },
        "dataset_hash": dataset_hash,
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "profile": profile.name,
        "dataset_hash": dataset_hash,
        "probe_samples": len(probe_rows),
        "final_samples": len(final_rows),
        "manifest": str(manifest_path),
        "executed": False,
    }


def _load_manifest(root: Path, profile: Profile, profile_path: Path) -> Mapping[str, Any]:
    probe_path, final_path, manifest_path = _data_paths(root, profile.name)
    if not manifest_path.is_file():
        raise IncompleteDataError(
            f"dataset manifest is missing; run: python3 -m autokv make-data --profile {profile.name}"
        )
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("profile") != profile.name:
        raise ValueError("dataset manifest does not match the selected profile")
    profile_hash = sha256_file(profile_path)
    if manifest.get("profile_sha256") != profile_hash:
        raise ValueError("profile changed after dataset generation; create a new dataset/run")
    artifact_hashes: dict[str, str] = {}
    for phase, path in (("probe", probe_path), ("final", final_path)):
        metadata = manifest.get(phase)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"dataset manifest is missing {phase} metadata")
        if metadata.get("path") != path.relative_to(root).as_posix():
            raise ValueError(f"dataset manifest {phase} path is not canonical")
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"dataset {phase} artifact hash mismatch")
        artifact_hashes[phase] = str(metadata["sha256"])
        rows = read_jsonl(path)
        if metadata.get("samples") != len(rows):
            raise ValueError(f"dataset {phase} sample count mismatch")
    dataset_hash = manifest.get("dataset_hash")
    if not isinstance(dataset_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_hash):
        raise ValueError("dataset manifest has no valid dataset_hash")
    expected_hash = hashlib.sha256(
        "\n".join(
            (profile_hash, artifact_hashes["probe"], artifact_hashes["final"])
        ).encode("utf-8")
    ).hexdigest()
    if dataset_hash != expected_hash:
        raise ValueError("dataset_hash does not match profile/probe/final artifacts")
    return manifest


def _load_cases(root: Path, profile: Profile, phase: str) -> tuple[NiahCase, ...]:
    probe_path, final_path, _ = _data_paths(root, profile.name)
    path = probe_path if phase == "probe" else final_path
    cases = []
    for row in read_jsonl(path):
        if not isinstance(row, Mapping):
            raise ValueError(f"dataset row is not an object: {path}")
        case = NiahCase(
            sample_id=str(row.get("sample_id", "")),
            target_tokens=int(row.get("target_tokens", 0)),
            depth=float(row.get("depth", -1)),
            seed=int(row.get("seed", 0)),
            code=str(row.get("code", "")),
        )
        if row.get("expected") != expected_answer(case.code):
            raise ValueError(f"dataset expected answer mismatch: {case.sample_id}")
        cases.append(case)
    return tuple(cases)


def _lock_path(root: Path) -> Path:
    return root / "runs" / "_environment" / "lock.json"


def _doctor_path(root: Path) -> Path:
    return root / "runs" / "_environment" / "doctor.json"


def _validate_lock(lock: Any, profile: Profile) -> Mapping[str, Any]:
    if not isinstance(lock, Mapping):
        raise ValueError("environment lock is not a JSON object")
    image_ref = lock.get("image_ref")
    image_digest = lock.get("image_digest")
    revision = lock.get("model_revision")
    if not isinstance(image_ref, str) or not re.search(
        r"@sha256:[0-9a-f]{64}$", image_ref
    ):
        raise ValueError("environment lock image_ref is not an immutable digest")
    if not isinstance(image_digest, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_digest
    ):
        raise ValueError("environment lock image_digest is invalid")
    if not image_ref.endswith("@" + image_digest):
        raise ValueError("environment lock image_ref and image_digest disagree")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("environment lock model_revision is not immutable")
    if lock.get("model_id") not in (None, profile.model.model_id):
        raise ValueError("environment lock model does not match profile")
    host = lock.get("host")
    if not isinstance(host, Mapping) or host.get("driver") != EXPECTED_DRIVER:
        raise ValueError("environment lock does not prove the fixed 535.230.02 driver")
    return lock


def _load_lock(root: Path, profile: Profile) -> Mapping[str, Any]:
    path = _lock_path(root)
    if not path.is_file():
        raise IncompleteDataError(
            f"environment lock is missing; run: python3 -m autokv doctor --profile {profile.name}"
        )
    return _validate_lock(read_json(path), profile)


def _runtime_context(root: Path, profile_name: str) -> RuntimeContext:
    profile, profile_path = _load_named_profile(root, profile_name)
    manifest = _load_manifest(root, profile, profile_path)
    lock = _load_lock(root, profile)
    dataset_hash = str(manifest["dataset_hash"])
    profile_hash = sha256_file(profile_path)
    run_id = canonical_run_id(
        profile_hash,
        str(lock["image_digest"]),
        str(lock["model_revision"]),
        dataset_hash,
    )
    return RuntimeContext(
        root,
        profile,
        profile_path,
        profile_hash,
        manifest,
        dataset_hash,
        lock,
        run_id,
    )


def _linux_gate() -> None:
    if not sys.platform.startswith("linux"):
        raise ValueError(
            "真实 GPU 命令只能在目标 Linux 服务器执行；当前主机不是 Linux。不要修改驱动。"
        )


def _assert_current_host(profile: Profile) -> None:
    result = run_command(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ),
        timeout=30,
    )
    if not result.ok:
        raise DoctorError("nvidia-smi read-only query failed; do not modify the driver")
    facts = parse_gpu_csv(result.stdout)
    failed = [gate for gate in validate_host(facts, profile.hardware.driver) if not gate.ok]
    if failed:
        details = "; ".join(
            f"{gate.name}={gate.observed}, expected={gate.expected}" for gate in failed
        )
        raise DoctorError(f"target-host gate failed ({details}); do not modify the driver")


def _gpu_context(root: Path, profile_name: str) -> RuntimeContext:
    _linux_gate()
    context = _runtime_context(root, profile_name)
    _assert_current_host(context.profile)
    return context


def dry_run(root: Path, profile_name: str) -> dict[str, Any]:
    profile, _ = _load_named_profile(root, profile_name)
    probe_samples = len(make_cases(profile, "probe"))
    final_samples = len(make_cases(profile, "final"))
    if profile.selection.mode == "coarse_to_fine":
        core = 1 + 8 + 8 + 1
    else:
        core = 1 + profile.model.num_layers + 1
    quality_configurations = 11
    benchmark_scenarios = len(build_benchmark_matrix(profile))
    placeholder_image = "vllm/vllm-openai@sha256:" + "f" * 64
    placeholder_revision = "a" * 40
    example = server_command(
        profile,
        placeholder_image,
        Variant.mixed("auto-4", (0, 1, 2, 3)),
        root,
        8000,
        f"dry-{profile.name}",
        placeholder_revision,
    )
    return {
        "profile": profile.name,
        "executed": False,
        "driver_policy": f"must remain exactly {EXPECTED_DRIVER}",
        "core_probe_configurations": core,
        "probe_samples_per_configuration": probe_samples,
        "probe_requests": core * probe_samples,
        "quality_configurations": quality_configurations,
        "quality_samples_per_configuration": final_samples,
        "quality_requests": quality_configurations * final_samples,
        "benchmark_configurations": 3,
        "benchmark_scenarios": benchmark_scenarios,
        "benchmark_requests": (
            3 * benchmark_scenarios * profile.benchmark.num_prompts
        ),
        "smoke_engine_starts": 2,
        "estimated_total_engine_starts": 2 + core + quality_configurations + 3,
        "run_steps": list(RUN_STEPS),
        "server_command_example": format_command(example),
        "notes": [
            "dry-run never invokes Docker, nvidia-smi, HTTP, or the GPU",
            "A6000 FP8 is a KV storage-compression path, not native FP8 Tensor Core compute",
        ],
    }


def _relative(root: Path, path: Path) -> str:
    return ensure_within(root, path).relative_to(root).as_posix()


def _verified_path(root: Path, relative: Any, expected_hash: Any) -> Path:
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError("artifact index is missing path/hash")
    path = ensure_within(root, root / relative)
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def _artifact_record(root: Path, path: Path) -> dict[str, str]:
    return {"path": _relative(root, path), "sha256": sha256_file(path)}


def _mean_quality(path: Path, expected_mode: str | None = None) -> float:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"quality artifact is empty: {path}")
    modes = {row.get("quality_mode") for row in rows}
    if modes - {"nll", "edit_distance"} or len(modes) != 1:
        raise ValueError(f"quality artifact mixes or omits scoring mode: {path}")
    if expected_mode is not None and modes != {expected_mode}:
        raise ValueError(
            f"quality artifact does not use locked {expected_mode} mode: {path}"
        )
    try:
        return statistics.fmean(float(row["quality_score"]) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"quality artifact has invalid quality_score: {path}") from exc


def _smoke_path(context: RuntimeContext) -> Path:
    return context.root / "runs" / context.run_id / "smoke" / "smoke.json"


def _smoke_is_complete(context: RuntimeContext) -> bool:
    path = _smoke_path(context)
    try:
        payload = read_json(path)
        if not isinstance(payload, Mapping) or payload.get("complete") is not True:
            return False
        for key in ("first", "second"):
            artifact = payload.get(key)
            if not isinstance(artifact, Mapping):
                return False
            _verified_path(context.root, artifact.get("path"), artifact.get("sha256"))
        return (
            payload.get("deterministic") is True
            and payload.get("quality_mode") in {"nll", "edit_distance"}
        )
    except (OSError, TypeError, ValueError):
        return False


def run_smoke(context: RuntimeContext, port: int) -> dict[str, Any]:
    if _smoke_is_complete(context):
        return dict(read_json(_smoke_path(context)))
    case = NiahCase("smoke-1024-50-42", 1024, 0.5, 42, "KV-SMOKE-4242")
    runner = ExperimentRunner(context.profile, context.root, context.lock)
    first_path = runner.run_variant(
        Variant.fp8(),
        (case,),
        phase="smoke-a",
        run_id=context.run_id,
        port=port,
    )
    second_path = runner.run_variant(
        Variant.fp8(),
        (case,),
        phase="smoke-b",
        run_id=context.run_id,
        port=port,
    )
    first = read_jsonl(first_path)[0]
    second = read_jsonl(second_path)[0]
    from autokv.benchmark import parse_capacity_tokens

    first_capacity = parse_capacity_tokens(
        first_path.with_suffix(".server.log").read_text(encoding="utf-8")
    ).tokens
    second_capacity = parse_capacity_tokens(
        second_path.with_suffix(".server.log").read_text(encoding="utf-8")
    ).tokens
    fields = {
        "output_text": first.get("output_text") == second.get("output_text"),
        "output_tokens": first.get("output_tokens") == second.get("output_tokens"),
        "capacity_tokens": first_capacity == second_capacity,
        "quality_mode": first.get("quality_mode") == second.get("quality_mode")
        and first.get("quality_mode") in {"nll", "edit_distance"},
    }
    deterministic = all(fields.values())
    quality_mode = first.get("quality_mode") if fields["quality_mode"] else None
    payload = {
        "schema_version": 1,
        "complete": deterministic,
        "deterministic": deterministic,
        "run_id": context.run_id,
        "comparison": fields,
        "quality_mode": quality_mode,
        "capacity_tokens": [first_capacity, second_capacity],
        "first": _artifact_record(context.root, first_path),
        "second": _artifact_record(context.root, second_path),
    }
    atomic_write_json(_smoke_path(context), payload)
    if not deterministic:
        command = format_command(
            (
                "python3",
                "-m",
                "autokv",
                "diagnose",
                "--project-root",
                str(context.root),
                "--profile",
                context.profile.name,
                "--json",
            )
        )
        raise IncompleteDataError(
            "two-run FP8 scale determinism gate failed; do not continue to probe. "
            f"Collect the extension input with: {command}; then follow the "
            "RUNBOOK.zh-CN.md dataset-calibrated KV-only scales extension."
        )
    return payload


def _require_smoke(context: RuntimeContext) -> None:
    if not _smoke_is_complete(context):
        raise IncompleteDataError(
            f"smoke gate is incomplete; run: python3 -m autokv smoke --profile {context.profile.name}"
        )


def _locked_quality_mode(context: RuntimeContext) -> str:
    _require_smoke(context)
    payload = read_json(_smoke_path(context))
    if not isinstance(payload, Mapping):
        raise ValueError("smoke manifest is malformed")
    mode = payload.get("quality_mode")
    if mode not in {"nll", "edit_distance"}:
        raise ValueError("smoke manifest has no locked quality mode")
    return str(mode)


def run_probe(context: RuntimeContext, port: int) -> dict[str, Any]:
    quality_mode = _locked_quality_mode(context)
    cases = _load_cases(context.root, context.profile, "probe")
    runner = ExperimentRunner(
        context.profile,
        context.root,
        context.lock,
        quality_mode=quality_mode,
    )
    artifacts: dict[str, dict[str, str]] = {}
    baseline_path = runner.run_variant(
        Variant.fp8(), cases, phase="probe", run_id=context.run_id, port=port
    )
    artifacts["fp8"] = _artifact_record(context.root, baseline_path)
    baseline = _mean_quality(baseline_path, quality_mode)
    group_scores: dict[str, float] = {}

    if context.profile.selection.mode == "coarse_to_fine":
        groups_by_name: dict[str, tuple[int, ...]] = {}
        for variant in group_probe_variants(
            context.profile.model.num_layers, context.profile.selection.group_size
        ):
            path = runner.run_variant(
                variant, cases, phase="probe", run_id=context.run_id, port=port
            )
            artifacts[variant.name] = _artifact_record(context.root, path)
            groups_by_name[variant.name] = variant.skip_layers
            group_scores[variant.name] = _mean_quality(path, quality_mode) - baseline
        tuple_scores = {
            groups_by_name[name]: score for name, score in group_scores.items()
        }
        selected_groups = select_top_groups(
            tuple_scores, context.profile.selection.top_groups
        )
        candidate_layers = tuple(sorted(layer for group in selected_groups for layer in group))
    else:
        selected_groups = ()
        candidate_layers = tuple(range(context.profile.model.num_layers))

    layer_scores: dict[int, float] = {}
    for layer in candidate_layers:
        variant = Variant.mixed(f"layer-{layer:02d}", (layer,))
        path = runner.run_variant(
            variant, cases, phase="probe", run_id=context.run_id, port=port
        )
        artifacts[variant.name] = _artifact_record(context.root, path)
        layer_scores[layer] = _mean_quality(path, quality_mode) - baseline
    auto_layers = select_top_layers(layer_scores, context.profile.selection.k)
    auto_path = runner.run_variant(
        Variant.mixed("auto-4", auto_layers),
        cases,
        phase="probe",
        run_id=context.run_id,
        port=port,
    )
    artifacts["auto-4"] = _artifact_record(context.root, auto_path)
    payload = {
        "schema_version": 1,
        "complete": True,
        "run_id": context.run_id,
        "profile": context.profile.name,
        "dataset_hash": context.dataset_hash,
        "quality_mode": quality_mode,
        "baseline_quality": baseline,
        "selected_groups": [list(group) for group in selected_groups],
        "candidate_layers": list(candidate_layers),
        "group_scores": group_scores,
        "layer_scores": {str(layer): score for layer, score in sorted(layer_scores.items())},
        "auto_layers": list(auto_layers),
        "artifacts": artifacts,
    }
    index_path = context.root / "runs" / context.run_id / "probe" / "index.json"
    atomic_write_json(index_path, payload)
    return {**payload, "index": str(index_path)}


def run_select(context: RuntimeContext) -> dict[str, Any]:
    index_path = context.root / "runs" / context.run_id / "probe" / "index.json"
    if not index_path.is_file():
        raise IncompleteDataError(
            f"probe index is missing; run: python3 -m autokv probe --profile {context.profile.name}"
        )
    index = read_json(index_path)
    if not isinstance(index, Mapping) or index.get("complete") is not True:
        raise IncompleteDataError("probe index is incomplete")
    quality_mode = _locked_quality_mode(context)
    if index.get("dataset_hash") != context.dataset_hash:
        raise ValueError("probe index dataset hash differs from this run")
    if index.get("quality_mode") != quality_mode:
        raise ValueError("probe index quality mode differs from the smoke lock")
    artifact_index = index.get("artifacts")
    if not isinstance(artifact_index, Mapping):
        raise ValueError("probe index has no artifact map")
    baseline_record = artifact_index.get("fp8")
    if not isinstance(baseline_record, Mapping):
        raise ValueError("probe index has no FP8 baseline")
    baseline_path = _verified_path(
        context.root, baseline_record.get("path"), baseline_record.get("sha256")
    )
    baseline = _mean_quality(baseline_path, quality_mode)
    candidate_layers_raw = index.get("candidate_layers")
    if not isinstance(candidate_layers_raw, list):
        raise ValueError("probe index has no candidate layer list")
    candidate_layers = tuple(int(layer) for layer in candidate_layers_raw)
    layer_scores: dict[int, float] = {}
    for layer in candidate_layers:
        record = artifact_index.get(f"layer-{layer:02d}")
        if not isinstance(record, Mapping):
            raise ValueError(f"probe artifact missing for layer {layer}")
        path = _verified_path(context.root, record.get("path"), record.get("sha256"))
        layer_scores[layer] = _mean_quality(path, quality_mode) - baseline
    auto_layers = select_top_layers(layer_scores, context.profile.selection.k)
    if list(auto_layers) != index.get("auto_layers"):
        raise ValueError("probe auto layer decision does not reproduce from artifacts")
    inverted_layers = select_bottom_layers(layer_scores, context.profile.selection.k)
    first_layers = tuple(range(context.profile.selection.k))
    last_layers = tuple(
        range(
            context.profile.model.num_layers - context.profile.selection.k,
            context.profile.model.num_layers,
        )
    )
    forbidden = {auto_layers, inverted_layers, first_layers, last_layers}
    random_layers = random_controls(
        context.profile.model.num_layers,
        context.profile.selection.k,
        context.profile.selection.random_seeds,
        forbidden,
    )
    payload = {
        "schema_version": 1,
        "complete": True,
        "run_id": context.run_id,
        "profile": context.profile.name,
        "dataset_hash": context.dataset_hash,
        "quality_mode": quality_mode,
        "selection_scope": (
            "two-best-groups/eight-layers"
            if context.profile.selection.mode == "coarse_to_fine"
            else "all-32-layers"
        ),
        "candidate_layers": list(candidate_layers),
        "auto_layers": list(auto_layers),
        "inverted_layers": list(inverted_layers),
        "first_layers": list(first_layers),
        "last_layers": list(last_layers),
        "random_layers": [list(layers) for layers in random_layers],
        "random_seeds": list(context.profile.selection.random_seeds),
        "layer_scores": {str(layer): score for layer, score in sorted(layer_scores.items())},
        "probe_index": _artifact_record(context.root, index_path),
    }
    selection_path = context.root / "runs" / context.run_id / "selection.json"
    atomic_write_json(selection_path, payload)
    return {**payload, "selection": str(selection_path)}


def _load_selection(context: RuntimeContext) -> Mapping[str, Any]:
    path = context.root / "runs" / context.run_id / "selection.json"
    if not path.is_file():
        raise IncompleteDataError(
            f"selection is missing; run: python3 -m autokv select --profile {context.profile.name}"
        )
    selection = read_json(path)
    if not isinstance(selection, Mapping) or selection.get("complete") is not True:
        raise IncompleteDataError("selection is incomplete")
    if selection.get("run_id") != context.run_id:
        raise ValueError("selection run ID does not match current immutable inputs")
    if selection.get("quality_mode") != _locked_quality_mode(context):
        raise ValueError("selection quality mode differs from the smoke lock")
    return selection


def _quality_variants(selection: Mapping[str, Any]) -> tuple[Variant, ...]:
    variants: list[Variant] = [
        Variant.bf16(),
        Variant.fp8(),
        Variant.mixed("auto-4", selection["auto_layers"]),
    ]
    random_layers = selection.get("random_layers")
    if not isinstance(random_layers, list) or len(random_layers) != 5:
        raise ValueError("selection must contain exactly five random controls")
    variants.extend(
        Variant.mixed(f"random-4-{index}", layers)
        for index, layers in enumerate(random_layers, start=1)
    )
    variants.extend(
        (
            Variant.mixed("first-4", selection["first_layers"]),
            Variant.mixed("last-4", selection["last_layers"]),
            Variant.mixed("inverted-4", selection["inverted_layers"]),
        )
    )
    return tuple(variants)


def run_evaluate(context: RuntimeContext, port: int) -> dict[str, Any]:
    quality_mode = _locked_quality_mode(context)
    selection = _load_selection(context)
    cases = _load_cases(context.root, context.profile, "final")
    runner = ExperimentRunner(
        context.profile,
        context.root,
        context.lock,
        quality_mode=quality_mode,
    )
    artifacts = {}
    for variant in _quality_variants(selection):
        path = runner.run_variant(
            variant,
            cases,
            phase="quality",
            run_id=context.run_id,
            port=port,
        )
        artifacts[variant.name] = _artifact_record(context.root, path)
    payload = {
        "schema_version": 1,
        "complete": True,
        "run_id": context.run_id,
        "dataset_hash": context.dataset_hash,
        "quality_mode": quality_mode,
        "samples_per_configuration": len(cases),
        "configurations": len(artifacts),
        "artifacts": artifacts,
    }
    path = context.root / "runs" / context.run_id / "quality" / "index.json"
    atomic_write_json(path, payload)
    return {**payload, "index": str(path)}


def run_benchmark(context: RuntimeContext, port: int) -> dict[str, Any]:
    _require_smoke(context)
    selection = _load_selection(context)
    variants = (
        Variant.bf16(),
        Variant.fp8(),
        Variant.mixed("auto-4", selection["auto_layers"]),
    )
    runner = BenchmarkRunner(context.profile, context.root, context.lock)
    summaries = {}
    for variant in variants:
        path = runner.run_variant(variant, run_id=context.run_id, port=port)
        summaries[variant.name] = _artifact_record(context.root, path)
    payload = {
        "schema_version": 1,
        "complete": True,
        "run_id": context.run_id,
        "scenario_count_per_configuration": len(build_benchmark_matrix(context.profile)),
        "summaries": summaries,
    }
    path = context.root / "runs" / context.run_id / "perf" / "index.json"
    atomic_write_json(path, payload)
    return {**payload, "index": str(path)}


def run_report(context: RuntimeContext) -> dict[str, Any]:
    selection = _load_selection(context)
    quality_mode = _locked_quality_mode(context)
    quality_index_path = context.root / "runs" / context.run_id / "quality" / "index.json"
    perf_index_path = context.root / "runs" / context.run_id / "perf" / "index.json"
    if not quality_index_path.is_file() or not perf_index_path.is_file():
        raise IncompleteDataError("quality/performance indexes are incomplete")
    quality_index = read_json(quality_index_path)
    perf_index = read_json(perf_index_path)
    if not isinstance(quality_index, Mapping) or quality_index.get("complete") is not True:
        raise IncompleteDataError("quality index is incomplete")
    if not isinstance(perf_index, Mapping) or perf_index.get("complete") is not True:
        raise IncompleteDataError("performance index is incomplete")
    if quality_index.get("quality_mode") != quality_mode:
        raise ValueError("quality index differs from the locked scoring mode")
    quality_records = quality_index.get("artifacts")
    summary_records = perf_index.get("summaries")
    if not isinstance(quality_records, Mapping) or not isinstance(summary_records, Mapping):
        raise ValueError("quality/performance index is malformed")
    quality: dict[str, Sequence[Mapping[str, Any]]] = {}
    for name, record in quality_records.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("quality artifact index is malformed")
        path = _verified_path(context.root, record.get("path"), record.get("sha256"))
        rows = read_jsonl(path)
        _mean_quality(path, quality_mode)
        quality[name] = rows
    capacities: dict[str, Capacity] = {}
    performance: dict[str, Mapping[str, Any]] = {}
    for name, record in summary_records.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("performance summary index is malformed")
        path = _verified_path(context.root, record.get("path"), record.get("sha256"))
        summary = read_json(path)
        if not isinstance(summary, Mapping) or summary.get("complete") is not True:
            raise IncompleteDataError(f"benchmark summary is incomplete: {path}")
        capacity = summary.get("capacity")
        aggregate = summary.get("aggregate")
        if not isinstance(capacity, Mapping) or not isinstance(aggregate, Mapping):
            raise ValueError(f"benchmark summary is malformed: {path}")
        capacities[name] = Capacity(
            tokens=int(capacity["tokens"]),
            model_length=(
                None if capacity.get("model_length") is None else int(capacity["model_length"])
            ),
            max_concurrency=(
                None
                if capacity.get("max_concurrency") is None
                else float(capacity["max_concurrency"])
            ),
        )
        performance[name] = aggregate
    output_dir = context.root / "runs" / context.run_id / "report"
    artifacts = render_report(
        output_dir,
        context.profile,
        context.lock,
        selection,
        quality,
        capacities,
        performance,
    )
    index = {
        "schema_version": 1,
        "complete": True,
        "run_id": context.run_id,
        "quality_mode": quality_mode,
        "artifacts": {
            name: _artifact_record(context.root, path) for name, path in artifacts.items()
        },
    }
    index_path = output_dir / "index.json"
    atomic_write_json(index_path, index)
    return {
        "run_id": context.run_id,
        "report": str(artifacts["markdown"]),
        "csv": str(artifacts["csv"]),
        "svg": str(artifacts["svg"]),
        "index": str(index_path),
    }


def read_status(root: Path, profile_name: str) -> dict[str, Any]:
    profile, profile_path = _load_named_profile(root, profile_name)
    doctor_ready = False
    if _doctor_path(root).is_file():
        try:
            doctor = read_json(_doctor_path(root))
            doctor_ready = isinstance(doctor, Mapping) and doctor.get("ok") is True
        except (OSError, ValueError):
            doctor_ready = False
    try:
        manifest = _load_manifest(root, profile, profile_path)
        data_ready = True
    except (IncompleteDataError, OSError, ValueError):
        manifest = None
        data_ready = False
    try:
        lock = _load_lock(root, profile)
        lock_ready = True
    except (IncompleteDataError, OSError, ValueError):
        lock = None
        lock_ready = False
    run_id = None
    if data_ready and lock_ready and manifest is not None and lock is not None:
        run_id = canonical_run_id(
            sha256_file(profile_path),
            str(lock["image_digest"]),
            str(lock["model_revision"]),
            str(manifest["dataset_hash"]),
        )
    steps = {
        "doctor": doctor_ready,
        "lock-image": lock_ready,
        "make-data": data_ready,
        "smoke": False,
        "probe": False,
        "select": False,
        "evaluate": False,
        "benchmark": False,
        "report": False,
    }
    if run_id is not None:
        run_root = root / "runs" / run_id
        steps.update(
            {
                "smoke": (run_root / "smoke" / "smoke.json").is_file(),
                "probe": (run_root / "probe" / "index.json").is_file(),
                "select": (run_root / "selection.json").is_file(),
                "evaluate": (run_root / "quality" / "index.json").is_file(),
                "benchmark": (run_root / "perf" / "index.json").is_file(),
                "report": (run_root / "report" / "index.json").is_file(),
            }
        )
    next_step = next((name for name in RUN_STEPS if not steps[name]), None)
    return {
        "profile": profile.name,
        "driver_required": EXPECTED_DRIVER,
        "doctor_ready": doctor_ready,
        "lock_ready": lock_ready,
        "data_ready": data_ready,
        "run_id": run_id,
        "steps": steps,
        "next_step": next_step,
        "read_only": True,
    }


_DIAGNOSTIC_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".csv",
    ".svg",
    ".txt",
    ".toml",
}


def _redact_diagnostic_text(text: str) -> str:
    text = redact(text, (os.environ.get("HF_TOKEN", ""),))
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+", r"\1***", text
    )
    text = re.sub(r"(?i)(HF_TOKEN\s*[:=]\s*)[^\s\"']+", r"\1***", text)
    return text


def create_diagnostic_archive(root: Path, profile_name: str) -> dict[str, Any]:
    _load_named_profile(root, profile_name)
    files = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            lowered_parts = {part.lower() for part in relative.parts}
            if lowered_parts & {".cache", ".git", "__pycache__", "_diagnostics"}:
                continue
            if path.name.endswith(".tar.gz") or path.suffix.lower() not in _DIAGNOSTIC_SUFFIXES:
                continue
            if path.stat().st_size > 20 * 1024 * 1024:
                continue
            files.append(path)
    archive_dir = ensure_within(root, root / "runs" / "_diagnostics")
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_dir / f"autokv-{profile_name}-{timestamp}-{os.getpid()}.tar.gz"
    entries = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            body = _redact_diagnostic_text(text).encode("utf-8")
            info = tarfile.TarInfo(f"autokv-skip/{relative}")
            info.size = len(body)
            info.mtime = 0
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(body))
            entries.append(
                {"path": relative, "sha256_redacted": hashlib.sha256(body).hexdigest()}
            )
        manifest_body = json.dumps(
            {
                "schema_version": 1,
                "profile": profile_name,
                "token_present": bool(os.environ.get("HF_TOKEN")),
                "files": entries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        info = tarfile.TarInfo("autokv-skip/diagnostics-manifest.json")
        info.size = len(manifest_body)
        info.mtime = 0
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(manifest_body))
    return {
        "profile": profile_name,
        "archive": str(archive_path),
        "files": len(entries),
        "cache_excluded": True,
        "token_redacted": True,
    }


def _doctor(root: Path, profile_name: str) -> dict[str, Any]:
    _linux_gate()
    profile, _ = _load_named_profile(root, profile_name)
    lock = lock_first_compatible_image(profile, root)
    return {
        "ok": True,
        "profile": profile.name,
        "driver": lock["host"]["driver"],
        "image_ref": lock["image_ref"],
        "model_revision": lock["model_revision"],
        "doctor": str(_doctor_path(root)),
        "lock": str(_lock_path(root)),
    }


def _lock_image(root: Path, profile_name: str) -> dict[str, Any]:
    _linux_gate()
    profile, _ = _load_named_profile(root, profile_name)
    _assert_current_host(profile)
    path = _lock_path(root)
    reused = False
    if path.is_file():
        lock = _validate_lock(read_json(path), profile)
        reused = True
    else:
        lock = lock_first_compatible_image(profile, root)
    return {
        "ok": True,
        "profile": profile.name,
        "reused": reused,
        "image_ref": lock["image_ref"],
        "model_revision": lock["model_revision"],
        "lock": str(path),
    }


def run_all(root: Path, profile_name: str, port: int) -> dict[str, Any]:
    _linux_gate()
    profile, _ = _load_named_profile(root, profile_name)
    _assert_current_host(profile)
    completed = []
    try:
        existing_lock = _validate_lock(read_json(_lock_path(root)), profile)
        doctor = read_json(_doctor_path(root))
        if not isinstance(doctor, Mapping) or doctor.get("ok") is not True:
            raise ValueError("doctor state is not complete")
        del existing_lock
    except (OSError, TypeError, ValueError):
        lock_first_compatible_image(profile, root)
    completed.extend(("doctor", "lock-image"))
    make_data(root, profile_name)
    completed.append("make-data")
    context = _runtime_context(root, profile_name)
    run_smoke(context, port)
    completed.append("smoke")
    run_probe(context, port)
    completed.append("probe")
    run_select(context)
    completed.append("select")
    run_evaluate(context, port)
    completed.append("evaluate")
    run_benchmark(context, port)
    completed.append("benchmark")
    report = run_report(context)
    completed.append("report")
    return {
        "complete": True,
        "profile": profile_name,
        "run_id": context.run_id,
        "completed_steps": completed,
        **report,
    }


def _emit(payload: Mapping[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _add_common(parser: argparse.ArgumentParser, *, port: bool = False) -> None:
    parser.add_argument(
        "--project-root",
        default=str(REPOSITORY_ROOT),
        help="AutoKV-Skip repository/output root",
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    if port:
        parser.add_argument("--port", type=int, default=8000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autokv",
        description=(
            "AutoKV-Skip: driver-locked mixed BF16/FP8 KV-cache selection for vLLM"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "doctor": "read-only host/container gates and immutable lock creation",
        "lock-image": "reuse or create the immutable image/model lock",
        "make-data": "materialize deterministic NIAH specifications",
        "dry-run": "show counts and commands without touching Docker/GPU",
        "smoke": "run the two-engine FP8 determinism gate",
        "probe": "run coarse/full sensitivity probes and joint Auto-4",
        "select": "recompute Auto-4 and controls from verified probe artifacts",
        "evaluate": "evaluate BF16, FP8, Auto-4 and eight controls",
        "benchmark": "measure KV capacity and vLLM service performance",
        "report": "render Markdown, CSV and SVG from verified artifacts",
        "status": "read artifact state without modifying it",
        "diagnose": "create a redacted archive excluding model/cache files",
        "run": "execute the complete resumable workflow",
    }
    gpu_port_commands = {"smoke", "probe", "evaluate", "benchmark", "run"}
    for name, description in descriptions.items():
        command_parser = subparsers.add_parser(name, help=description, description=description)
        _add_common(command_parser, port=name in gpu_port_commands)
    return parser


def _dispatch(args: argparse.Namespace) -> Mapping[str, Any]:
    root = _resolved_root(args.project_root)
    command = args.command
    if command == "doctor":
        return _doctor(root, args.profile)
    if command == "lock-image":
        return _lock_image(root, args.profile)
    if command == "make-data":
        return make_data(root, args.profile)
    if command == "dry-run":
        return dry_run(root, args.profile)
    if command == "status":
        return read_status(root, args.profile)
    if command == "diagnose":
        return create_diagnostic_archive(root, args.profile)
    if command in {"select", "report"}:
        context = _runtime_context(root, args.profile)
        return run_select(context) if command == "select" else run_report(context)
    context = _gpu_context(root, args.profile)
    if command == "smoke":
        return run_smoke(context, args.port)
    if command == "probe":
        return run_probe(context, args.port)
    if command == "evaluate":
        return run_evaluate(context, args.port)
    if command == "benchmark":
        return run_benchmark(context, args.port)
    if command == "run":
        return run_all(root, args.profile, args.port)
    raise ValueError(f"unknown command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = _dispatch(args)
        _emit(payload, args.json)
        return 0
    except IncompleteDataError as exc:
        print(f"incomplete: {exc}", file=sys.stderr)
        return EXIT_INCOMPLETE
    except (VllmHttpError, TimeoutError) as exc:
        print(f"server/http error: {exc}", file=sys.stderr)
        return EXIT_HTTP
    except (DoctorError, ValueError, TypeError) as exc:
        print(f"invalid/gate error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"external command/error: {exc}", file=sys.stderr)
        return EXIT_EXTERNAL
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
