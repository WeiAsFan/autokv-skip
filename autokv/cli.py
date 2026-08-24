"""Command-line workflow for the driver-locked AutoKV-Skip experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokv.benchmark import (
    BenchmarkRunner,
    Capacity,
    aggregate_scenario_groups,
    build_benchmark_matrix,
    capacity_validation,
    extract_performance_metrics,
    parse_capacity_tokens,
    telemetry_is_usable,
)
from autokv.client import VllmHttpError
from autokv.commands import format_command, run_command, server_command
from autokv.config import EXPECTED_DRIVER, Profile, load_profile
from autokv.doctor import (
    DoctorError,
    lock_first_compatible_image,
    parse_gpu_csv,
    validate_host,
)
from autokv.experiment import (
    ExperimentRunner,
    canonical_run_id,
    is_complete,
    validate_container_command,
    validate_server_log,
)
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
    canonical_config_id,
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
    source: Mapping[str, Any]
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


def _source_identity(source_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    candidates = [source_root / "pyproject.toml"]
    for directory, pattern in (("autokv", "*.py"), ("scripts", "*.py"), ("configs", "*.json")):
        candidates.extend((source_root / directory).rglob(pattern))
    files = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not files:
        raise ValueError("cannot identify AutoKV-Skip source files")
    entries = []
    tree = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()
        entries.append({"path": relative, "sha256": digest})
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")

    commit_result = run_command(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"), timeout=10
    )
    commit = commit_result.stdout.strip().lower()
    if not commit_result.ok or not re.fullmatch(r"[0-9a-f]{40}", commit):
        commit = None
    dirty_result = run_command(
        (
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "autokv",
            "scripts",
            "configs",
            "pyproject.toml",
        ),
        timeout=10,
    )
    dirty = bool(dirty_result.stdout.strip()) if dirty_result.ok else None
    return {
        "tree_sha256": tree.hexdigest(),
        "git_commit": commit,
        "git_dirty": dirty,
        "files": entries,
    }


def _ensure_run_manifest(context: RuntimeContext) -> Path:
    path = context.root / "runs" / context.run_id / "run-manifest.json"
    try:
        profile_display_path = _relative(context.root, context.profile_path)
    except ValueError:
        profile_display_path = "packaged:" + context.profile_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix()
    expected = {
        "schema_version": 1,
        "run_id": context.run_id,
        "source": context.source,
        "profile": context.profile.name,
        "profile_path": profile_display_path,
        "profile_sha256": context.profile_hash,
        "dataset_hash": context.dataset_hash,
        "image_ref": context.lock["image_ref"],
        "image_digest": context.lock["image_digest"],
        "model_id": context.profile.model.model_id,
        "model_revision": context.lock["model_revision"],
        "storage_timezone": "UTC",
        "display_timezone": "Asia/Shanghai",
    }
    if path.is_file():
        observed = read_json(path)
        if not isinstance(observed, Mapping):
            raise ValueError("run manifest is not a JSON object")
        immutable_keys = tuple(key for key in expected if key != "source")
        if any(observed.get(key) != expected[key] for key in immutable_keys):
            raise ValueError("run manifest differs from current immutable inputs")
        if _canonical_source_identity(observed.get("source")) != (
            _canonical_source_identity(context.source)
        ):
            raise ValueError("run manifest source identity differs from current code")
        return path
    atomic_write_json(
        path,
        {
            **expected,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def _canonical_source_identity(source: Any) -> dict[str, Any]:
    """Return only source fields that participate in the immutable run ID.

    Git metadata is intentionally run-start provenance: a documentation-only
    commit does not change the runtime tree hash and must not invalidate an
    already-created run manifest.
    """
    if not isinstance(source, Mapping):
        raise ValueError("source identity is not an object")
    tree_sha256 = source.get("tree_sha256")
    files = source.get("files")
    if not isinstance(tree_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", tree_sha256
    ):
        raise ValueError("source identity has an invalid tree hash")
    if not isinstance(files, list) or any(
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is None
        for record in files
    ):
        raise ValueError("source identity has an invalid runtime file list")
    return {
        "tree_sha256": tree_sha256,
        "files": [dict(record) for record in files],
    }


def _write_completion_manifest(
    root: Path, run_id: str, source_tree_sha256: str
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id contains unsafe characters")
    if not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256):
        raise ValueError("source tree hash must be a 64-character SHA-256")
    run_root = ensure_within(root, root / "runs" / run_id)
    if not run_root.is_dir():
        raise ValueError(f"run directory does not exist: {run_root}")
    manifest_path = run_root / "completed-manifest.json"
    entries = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative_to_run = path.relative_to(run_root)
        if "_superseded" in relative_to_run.parts:
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ValueError("cannot complete a run with no active artifacts")
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "complete": True,
            "run_id": run_id,
            "source_tree_sha256": source_tree_sha256,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": entries,
        },
    )
    return manifest_path


def _completion_manifest_is_valid(root: Path, run_id: str) -> bool:
    try:
        run_root = ensure_within(root, root / "runs" / run_id)
        path = run_root / "completed-manifest.json"
        payload = read_json(path)
        if (
            not isinstance(payload, Mapping)
            or payload.get("complete") is not True
            or payload.get("run_id") != run_id
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(payload.get("source_tree_sha256", ""))
            )
        ):
            return False
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return False
        observed_paths: set[str] = set()
        for record in artifacts:
            if not isinstance(record, Mapping):
                return False
            relative = record.get("path")
            expected_hash = record.get("sha256")
            if (
                not isinstance(relative, str)
                or relative in observed_paths
                or not isinstance(expected_hash, str)
            ):
                return False
            artifact = ensure_within(run_root, root / relative)
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                return False
            observed_paths.add(relative)
        active_paths = {
            path.relative_to(root).as_posix()
            for path in run_root.rglob("*")
            if path.is_file()
            and path != run_root / "completed-manifest.json"
            and "_superseded" not in path.relative_to(run_root).parts
        }
        return observed_paths == active_paths
    except (OSError, TypeError, ValueError):
        return False


def _runtime_context(root: Path, profile_name: str) -> RuntimeContext:
    profile, profile_path = _load_named_profile(root, profile_name)
    manifest = _load_manifest(root, profile, profile_path)
    lock = _load_lock(root, profile)
    dataset_hash = str(manifest["dataset_hash"])
    profile_hash = sha256_file(profile_path)
    source = _source_identity()
    run_id = canonical_run_id(
        profile_hash,
        str(lock["image_digest"]),
        str(lock["model_revision"]),
        dataset_hash,
        str(source["tree_sha256"]),
    )
    context = RuntimeContext(
        root,
        profile,
        profile_path,
        profile_hash,
        manifest,
        dataset_hash,
        lock,
        source,
        run_id,
    )
    _ensure_run_manifest(context)
    return context


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
        groups = group_probe_variants(
            profile.model.num_layers, profile.selection.group_size
        )
        core = (
            1
            + len(groups)
            + profile.selection.top_groups * profile.selection.group_size
            + 1
        )
    else:
        core = 1 + profile.model.num_layers + 1
    quality_configurations = 3 + len(profile.selection.random_seeds) + 3
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
            if not _experiment_record_is_complete(
                context.root,
                artifact,
                expected_rows=1,
                expected_variant=Variant.fp8(),
            ):
                return False
        return (
            payload.get("deterministic") is True
            and payload.get("run_id") == context.run_id
            and payload.get("quality_mode") in {"nll", "edit_distance"}
        )
    except (OSError, TypeError, ValueError):
        return False


def _validate_force_target(
    force_config_id: str | None, variants: Sequence[Variant]
) -> None:
    if force_config_id is None:
        return
    if not re.fullmatch(r"[0-9a-f]{12}", force_config_id):
        raise ValueError("--force must be an exact 12-character config ID")
    available = {canonical_config_id(variant) for variant in variants}
    if force_config_id not in available:
        raise ValueError(
            f"--force config ID is not part of this phase: {force_config_id}"
        )


def run_smoke(
    context: RuntimeContext, port: int, force_config_id: str | None = None
) -> dict[str, Any]:
    _validate_force_target(force_config_id, (Variant.fp8(),))
    if _smoke_is_complete(context) and force_config_id is None:
        return dict(read_json(_smoke_path(context)))
    case = NiahCase("smoke-1024-50-42", 1024, 0.5, 42, "KV-SMOKE-4242")
    runner = ExperimentRunner(
        context.profile,
        context.root,
        context.lock,
        force_config_id=force_config_id,
    )
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


def run_probe(
    context: RuntimeContext, port: int, force_config_id: str | None = None
) -> dict[str, Any]:
    quality_mode = _locked_quality_mode(context)
    cases = _load_cases(context.root, context.profile, "probe")
    runner = ExperimentRunner(
        context.profile,
        context.root,
        context.lock,
        quality_mode=quality_mode,
        force_config_id=force_config_id,
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
    if force_config_id is not None and not runner.force_applied:
        raise ValueError(
            f"--force config ID is not part of the realized probe: {force_config_id}"
        )
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


def _derive_selection(
    root: Path,
    run_id: str,
    dataset_hash: str,
    profile: Profile,
    quality_mode: str,
    index_path: Path,
) -> dict[str, Any]:
    """Reproduce every selected/control layer from hashed probe artifacts."""
    if not index_path.is_file():
        raise IncompleteDataError(
            f"probe index is missing; run: python3 -m autokv probe --profile {profile.name}"
        )
    index = read_json(index_path)
    if not isinstance(index, Mapping) or index.get("complete") is not True:
        raise IncompleteDataError("probe index is incomplete")
    if index.get("run_id") != run_id:
        raise ValueError("probe index run ID differs from this run")
    if index.get("dataset_hash") != dataset_hash:
        raise ValueError("probe index dataset hash differs from this run")
    if index.get("quality_mode") != quality_mode:
        raise ValueError("probe index quality mode differs from the smoke lock")
    if not _probe_status_is_complete(
        root,
        run_id,
        dataset_hash,
        profile,
    ):
        raise IncompleteDataError(
            "probe artifact state/server/command evidence is incomplete"
        )
    artifact_index = index.get("artifacts")
    if not isinstance(artifact_index, Mapping):
        raise ValueError("probe index has no artifact map")
    baseline_record = artifact_index.get("fp8")
    if not isinstance(baseline_record, Mapping):
        raise ValueError("probe index has no FP8 baseline")
    baseline_path = _verified_path(
        root, baseline_record.get("path"), baseline_record.get("sha256")
    )
    baseline = _mean_quality(baseline_path, quality_mode)
    group_scores: dict[str, float] = {}
    selected_groups: tuple[tuple[int, ...], ...] = ()
    if profile.selection.mode == "coarse_to_fine":
        indexed_group_scores = index.get("group_scores")
        if not isinstance(indexed_group_scores, Mapping):
            raise ValueError("probe index has no group score map")
        scores_by_group: dict[tuple[int, ...], float] = {}
        for variant in group_probe_variants(
            profile.model.num_layers,
            profile.selection.group_size,
        ):
            record = artifact_index.get(variant.name)
            if not isinstance(record, Mapping):
                raise ValueError(f"probe artifact missing for group {variant.name}")
            path = _verified_path(root, record.get("path"), record.get("sha256"))
            score = _mean_quality(path, quality_mode) - baseline
            indexed_score = indexed_group_scores.get(variant.name)
            if not isinstance(indexed_score, (int, float)) or not math.isclose(
                score, float(indexed_score), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"probe group score does not reproduce: {variant.name}"
                )
            group_scores[variant.name] = score
            scores_by_group[variant.skip_layers] = score
        selected_groups = select_top_groups(
            scores_by_group, profile.selection.top_groups
        )
        expected_candidates = tuple(
            sorted(layer for group in selected_groups for layer in group)
        )
    else:
        expected_candidates = tuple(range(profile.model.num_layers))
    candidate_layers_raw = index.get("candidate_layers")
    if not isinstance(candidate_layers_raw, list):
        raise ValueError("probe index has no candidate layer list")
    candidate_layers = tuple(int(layer) for layer in candidate_layers_raw)
    if candidate_layers != expected_candidates:
        raise ValueError("probe candidate layers do not reproduce from group artifacts")
    if index.get("selected_groups", []) != [list(group) for group in selected_groups]:
        raise ValueError("probe selected groups do not reproduce from artifacts")
    layer_scores: dict[int, float] = {}
    for layer in candidate_layers:
        record = artifact_index.get(f"layer-{layer:02d}")
        if not isinstance(record, Mapping):
            raise ValueError(f"probe artifact missing for layer {layer}")
        path = _verified_path(root, record.get("path"), record.get("sha256"))
        layer_scores[layer] = _mean_quality(path, quality_mode) - baseline
    auto_layers = select_top_layers(layer_scores, profile.selection.k)
    if list(auto_layers) != index.get("auto_layers"):
        raise ValueError("probe auto layer decision does not reproduce from artifacts")
    inverted_layers = select_bottom_layers(layer_scores, profile.selection.k)
    first_layers = tuple(range(profile.selection.k))
    last_layers = tuple(
        range(
            profile.model.num_layers - profile.selection.k,
            profile.model.num_layers,
        )
    )
    forbidden = {auto_layers, inverted_layers, first_layers, last_layers}
    random_layers = random_controls(
        profile.model.num_layers,
        profile.selection.k,
        profile.selection.random_seeds,
        forbidden,
    )
    return {
        "schema_version": 1,
        "complete": True,
        "run_id": run_id,
        "profile": profile.name,
        "dataset_hash": dataset_hash,
        "quality_mode": quality_mode,
        "selection_scope": (
            "two-best-groups/eight-layers"
            if profile.selection.mode == "coarse_to_fine"
            else "all-32-layers"
        ),
        "candidate_layers": list(candidate_layers),
        "auto_layers": list(auto_layers),
        "inverted_layers": list(inverted_layers),
        "first_layers": list(first_layers),
        "last_layers": list(last_layers),
        "random_layers": [list(layers) for layers in random_layers],
        "random_seeds": list(profile.selection.random_seeds),
        "group_scores": group_scores,
        "layer_scores": {str(layer): score for layer, score in sorted(layer_scores.items())},
        "probe_index": _artifact_record(root, index_path),
    }


def run_select(context: RuntimeContext) -> dict[str, Any]:
    index_path = context.root / "runs" / context.run_id / "probe" / "index.json"
    if not index_path.is_file():
        raise IncompleteDataError(
            f"probe index is missing; run: python3 -m autokv probe --profile {context.profile.name}"
        )
    payload = _derive_selection(
        context.root,
        context.run_id,
        context.dataset_hash,
        context.profile,
        _locked_quality_mode(context),
        index_path,
    )
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
    expected = _derive_selection(
        context.root,
        context.run_id,
        context.dataset_hash,
        context.profile,
        _locked_quality_mode(context),
        context.root / "runs" / context.run_id / "probe" / "index.json",
    )
    if dict(selection) != expected:
        raise ValueError("selection does not reproduce exactly from probe artifacts")
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


def run_evaluate(
    context: RuntimeContext, port: int, force_config_id: str | None = None
) -> dict[str, Any]:
    quality_mode = _locked_quality_mode(context)
    selection = _load_selection(context)
    cases = _load_cases(context.root, context.profile, "final")
    variants = _quality_variants(selection)
    _validate_force_target(force_config_id, variants)
    runner = ExperimentRunner(
        context.profile,
        context.root,
        context.lock,
        quality_mode=quality_mode,
        force_config_id=force_config_id,
    )
    artifacts = {}
    for variant in variants:
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


def run_benchmark(
    context: RuntimeContext, port: int, force_config_id: str | None = None
) -> dict[str, Any]:
    _require_smoke(context)
    selection = _load_selection(context)
    variants = (
        Variant.bf16(),
        Variant.fp8(),
        Variant.mixed("auto-4", selection["auto_layers"]),
    )
    _validate_force_target(force_config_id, variants)
    runner = BenchmarkRunner(
        context.profile,
        context.root,
        context.lock,
        force_config_id=force_config_id,
    )
    summaries = {}
    for variant in variants:
        path = runner.run_variant(variant, run_id=context.run_id, port=port)
        summaries[variant.name] = _artifact_record(context.root, path)
    payload = {
        "schema_version": 1,
        "complete": True,
        "run_id": context.run_id,
        "dataset_hash": context.dataset_hash,
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
    if not _quality_status_is_complete(
        context.root, context.run_id, context.dataset_hash, context.profile
    ) or not _perf_status_is_complete(
        context.root, context.run_id, context.dataset_hash, context.profile
    ):
        raise IncompleteDataError(
            "quality/performance artifact hashes or schemas are incomplete"
        )
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
        scenario_groups = summary.get("scenario_groups")
        validation = summary.get("capacity_validation")
        telemetry = summary.get("telemetry")
        server_log = summary.get("server_log")
        matrix_state = summary.get("matrix_state")
        if (
            not isinstance(capacity, Mapping)
            or not isinstance(aggregate, Mapping)
            or not isinstance(scenario_groups, list)
            or not isinstance(validation, Mapping)
            or not isinstance(telemetry, Mapping)
            or not isinstance(server_log, Mapping)
            or not isinstance(matrix_state, Mapping)
        ):
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
        performance[name] = {
            "overall_descriptive_mean": aggregate,
            "scenario_groups": scenario_groups,
            "capacity_validation": validation,
            "telemetry": telemetry,
            "server_log": server_log,
            "matrix_state": matrix_state,
        }
    output_dir = context.root / "runs" / context.run_id / "report"
    run_manifest = read_json(
        context.root / "runs" / context.run_id / "run-manifest.json"
    )
    if not isinstance(run_manifest, Mapping) or not isinstance(
        run_manifest.get("source"), Mapping
    ):
        raise ValueError("run manifest has no source provenance")
    report_selection = {**dict(selection), "source": run_manifest["source"]}
    artifacts = render_report(
        output_dir,
        context.profile,
        context.lock,
        report_selection,
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
    completed_manifest = _write_completion_manifest(
        context.root,
        context.run_id,
        str(context.source["tree_sha256"]),
    )
    return {
        "run_id": context.run_id,
        "report": str(artifacts["markdown"]),
        "csv": str(artifacts["csv"]),
        "performance_csv": str(artifacts["performance_csv"]),
        "svg": str(artifacts["svg"]),
        "index": str(index_path),
        "completed_manifest": str(completed_manifest),
    }


def _record_is_valid(root: Path, record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    try:
        _verified_path(root, record.get("path"), record.get("sha256"))
        return True
    except (OSError, TypeError, ValueError):
        return False


def _experiment_record_is_complete(
    root: Path,
    record: Any,
    *,
    expected_rows: int,
    expected_variant: Variant,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    try:
        artifact = _verified_path(root, record.get("path"), record.get("sha256"))
        evidence = {
            "server_log": artifact.with_suffix(".server.log"),
            "command_record": artifact.with_suffix(".command.json"),
        }
        if not is_complete(
            artifact.with_suffix(".state.json"),
            artifact,
            expected_rows,
            evidence=evidence,
        ):
            return False
        command = read_json(evidence["command_record"])
        if not isinstance(command, Mapping):
            return False
        if command.get("variant") != {
            "name": expected_variant.name,
            "kv_dtype": expected_variant.kv_dtype,
            "skip_layers": list(expected_variant.skip_layers),
        }:
            return False
        inspected = command.get("inspected_argv")
        if not isinstance(inspected, list):
            return False
        validate_container_command(json.dumps(inspected), expected_variant)
        validate_server_log(
            evidence["server_log"].read_text(encoding="utf-8"),
            expected_variant,
        )
        return True
    except (OSError, TypeError, ValueError):
        return False


def _smoke_status_is_complete(root: Path, run_id: str) -> bool:
    path = root / "runs" / run_id / "smoke" / "smoke.json"
    try:
        payload = read_json(path)
        return (
            isinstance(payload, Mapping)
            and payload.get("complete") is True
            and payload.get("deterministic") is True
            and payload.get("run_id") == run_id
            and payload.get("quality_mode") in {"nll", "edit_distance"}
            and _experiment_record_is_complete(
                root,
                payload.get("first"),
                expected_rows=1,
                expected_variant=Variant.fp8(),
            )
            and _experiment_record_is_complete(
                root,
                payload.get("second"),
                expected_rows=1,
                expected_variant=Variant.fp8(),
            )
        )
    except (OSError, TypeError, ValueError):
        return False


def _probe_status_is_complete(
    root: Path,
    run_id: str,
    dataset_hash: str,
    profile: Profile,
) -> bool:
    path = root / "runs" / run_id / "probe" / "index.json"
    try:
        payload = read_json(path)
        if (
            not isinstance(payload, Mapping)
            or payload.get("complete") is not True
            or payload.get("run_id") != run_id
            or payload.get("dataset_hash") != dataset_hash
            or payload.get("quality_mode") not in {"nll", "edit_distance"}
        ):
            return False
        smoke = read_json(root / "runs" / run_id / "smoke" / "smoke.json")
        if (
            not isinstance(smoke, Mapping)
            or smoke.get("quality_mode") != payload.get("quality_mode")
        ):
            return False
        artifacts = payload.get("artifacts")
        candidates = payload.get("candidate_layers")
        auto_layers = payload.get("auto_layers")
        if (
            not isinstance(artifacts, Mapping)
            or not isinstance(candidates, list)
            or not isinstance(auto_layers, list)
            or len(auto_layers) != profile.selection.k
        ):
            return False
        expected_variants = {
            "fp8": Variant.fp8(),
            "auto-4": Variant.mixed("auto-4", tuple(int(layer) for layer in auto_layers)),
            **{
                f"layer-{int(layer):02d}": Variant.mixed(
                    f"layer-{int(layer):02d}", (int(layer),)
                )
                for layer in candidates
            },
        }
        if profile.selection.mode == "coarse_to_fine":
            expected_variants.update(
                {
                    variant.name: variant
                    for variant in group_probe_variants(
                        profile.model.num_layers, profile.selection.group_size
                    )
                }
            )
        if set(artifacts) != set(expected_variants):
            return False
        mode = str(payload["quality_mode"])
        expected_rows = len(make_cases(profile, "probe"))
        for name, record in artifacts.items():
            if not _experiment_record_is_complete(
                root,
                record,
                expected_rows=expected_rows,
                expected_variant=expected_variants[name],
            ):
                return False
            assert isinstance(record, Mapping)
            artifact_path = _verified_path(
                root, record.get("path"), record.get("sha256")
            )
            _mean_quality(artifact_path, mode)
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _selection_status_is_complete(
    root: Path,
    run_id: str,
    dataset_hash: str,
    profile: Profile,
) -> bool:
    path = root / "runs" / run_id / "selection.json"
    try:
        payload = read_json(path)
        if not isinstance(payload, Mapping) or payload.get("quality_mode") not in {
            "nll",
            "edit_distance",
        }:
            return False
        smoke = read_json(root / "runs" / run_id / "smoke" / "smoke.json")
        if (
            not isinstance(smoke, Mapping)
            or smoke.get("quality_mode") != payload.get("quality_mode")
        ):
            return False
        expected = _derive_selection(
            root,
            run_id,
            dataset_hash,
            profile,
            str(payload["quality_mode"]),
            root / "runs" / run_id / "probe" / "index.json",
        )
        return dict(payload) == expected
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _quality_status_is_complete(
    root: Path,
    run_id: str,
    dataset_hash: str,
    profile: Profile,
) -> bool:
    path = root / "runs" / run_id / "quality" / "index.json"
    required = {
        "bf16",
        "fp8",
        "auto-4",
        *(f"random-4-{index}" for index in range(1, len(profile.selection.random_seeds) + 1)),
        "first-4",
        "last-4",
        "inverted-4",
    }
    try:
        payload = read_json(path)
        if (
            not isinstance(payload, Mapping)
            or payload.get("complete") is not True
            or payload.get("run_id") != run_id
            or payload.get("dataset_hash") != dataset_hash
            or payload.get("quality_mode") not in {"nll", "edit_distance"}
        ):
            return False
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != required:
            return False
        if not _selection_status_is_complete(root, run_id, dataset_hash, profile):
            return False
        selection = read_json(root / "runs" / run_id / "selection.json")
        if (
            not isinstance(selection, Mapping)
            or selection.get("quality_mode") != payload.get("quality_mode")
        ):
            return False
        expected_variants = {
            variant.name: variant for variant in _quality_variants(selection)
        }
        if set(expected_variants) != required:
            return False
        expected_ids: set[str] | None = None
        for name, record in artifacts.items():
            if not _experiment_record_is_complete(
                root,
                record,
                expected_rows=len(make_cases(profile, "final")),
                expected_variant=expected_variants[name],
            ):
                return False
            assert isinstance(record, Mapping)
            artifact_path = _verified_path(
                root, record.get("path"), record.get("sha256")
            )
            rows = read_jsonl(artifact_path)
            if len(rows) != len(make_cases(profile, "final")):
                return False
            _mean_quality(artifact_path, str(payload["quality_mode"]))
            ids = {str(row.get("sample_id")) for row in rows}
            if len(ids) != len(rows):
                return False
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _perf_status_is_complete(
    root: Path, run_id: str, dataset_hash: str, profile: Profile
) -> bool:
    path = root / "runs" / run_id / "perf" / "index.json"
    try:
        payload = read_json(path)
        if (
            not isinstance(payload, Mapping)
            or payload.get("complete") is not True
            or payload.get("run_id") != run_id
            or payload.get("dataset_hash") != dataset_hash
        ):
            return False
        summaries = payload.get("summaries")
        if not isinstance(summaries, Mapping) or set(summaries) != {
            "bf16",
            "fp8",
            "auto-4",
        }:
            return False
        if not _selection_status_is_complete(root, run_id, dataset_hash, profile):
            return False
        selection = read_json(root / "runs" / run_id / "selection.json")
        if not isinstance(selection, Mapping):
            return False
        expected_variants = {
            variant.name: variant
            for variant in (
                Variant.bf16(),
                Variant.fp8(),
                Variant.mixed("auto-4", selection["auto_layers"]),
            )
        }
        expected_scenarios = len(build_benchmark_matrix(profile))
        expected_cases = [
            {
                "input_length": case.input_length,
                "output_length": case.output_length,
                "repeat": case.repeat,
            }
            for case in build_benchmark_matrix(profile)
        ]
        for name, record in summaries.items():
            if not _record_is_valid(root, record):
                return False
            assert isinstance(record, Mapping)
            summary_path = _verified_path(
                root, record.get("path"), record.get("sha256")
            )
            summary = read_json(summary_path)
            if (
                not isinstance(summary, Mapping)
                or summary.get("schema_version") != 3
                or summary.get("complete") is not True
                or summary.get("run_id") != run_id
                or not isinstance(summary.get("variant"), Mapping)
                or summary["variant"].get("name") != name
                or not isinstance(summary.get("scenarios"), list)
                or len(summary["scenarios"]) != expected_scenarios
                or not isinstance(summary.get("scenario_groups"), list)
                or not isinstance(summary.get("capacity_validation"), Mapping)
                or not _record_is_valid(root, summary.get("telemetry"))
                or not _record_is_valid(root, summary.get("server_log"))
                or not _record_is_valid(root, summary.get("command_record"))
                or not _record_is_valid(root, summary.get("matrix_state"))
            ):
                return False
            variant_raw = summary["variant"]
            variant = Variant(
                str(variant_raw.get("name", "")),
                str(variant_raw.get("kv_dtype", "")),
                tuple(int(layer) for layer in variant_raw.get("skip_layers", [])),
            )
            variant.validate_for_model(profile.model.num_layers)
            if variant != expected_variants[name]:
                return False
            actual_cases = []
            for scenario in summary["scenarios"]:
                if not isinstance(scenario, Mapping) or not _record_is_valid(
                    root,
                    {
                        "path": scenario.get("raw_result"),
                        "sha256": scenario.get("sha256"),
                    },
                ):
                    return False
                raw_path = _verified_path(
                    root, scenario.get("raw_result"), scenario.get("sha256")
                )
                raw_payload = read_json(raw_path)
                if (
                    not isinstance(raw_payload, Mapping)
                    or scenario.get("metrics")
                    != extract_performance_metrics(raw_payload)
                ):
                    return False
                actual_cases.append(
                    {
                        "input_length": scenario.get("input_length"),
                        "output_length": scenario.get("output_length"),
                        "repeat": scenario.get("repeat"),
                    }
                )
            if actual_cases != expected_cases or summary.get(
                "scenario_groups"
            ) != aggregate_scenario_groups(summary["scenarios"]):
                return False
            telemetry_record = summary["telemetry"]
            assert isinstance(telemetry_record, Mapping)
            telemetry_path = _verified_path(
                root,
                telemetry_record.get("path"),
                telemetry_record.get("sha256"),
            )
            if not telemetry_is_usable(telemetry_path):
                return False
            capacity_raw = summary.get("capacity")
            server_record = summary.get("server_log")
            if not isinstance(capacity_raw, Mapping) or not isinstance(
                server_record, Mapping
            ):
                return False
            measured_capacity = Capacity(
                int(capacity_raw["tokens"]),
                (
                    None
                    if capacity_raw.get("model_length") is None
                    else int(capacity_raw["model_length"])
                ),
                (
                    None
                    if capacity_raw.get("max_concurrency") is None
                    else float(capacity_raw["max_concurrency"])
                ),
            )
            server_path = _verified_path(
                root, server_record.get("path"), server_record.get("sha256")
            )
            server_log = server_path.read_text(encoding="utf-8")
            validate_server_log(
                server_log, variant, num_layers=profile.model.num_layers
            )
            command_record = summary.get("command_record")
            matrix_record = summary.get("matrix_state")
            if not isinstance(command_record, Mapping) or not isinstance(
                matrix_record, Mapping
            ):
                return False
            command_path = _verified_path(
                root,
                command_record.get("path"),
                command_record.get("sha256"),
            )
            command = read_json(command_path)
            if (
                not isinstance(command, Mapping)
                or command.get("variant") != dict(variant_raw)
                or not isinstance(command.get("inspected_server_argv"), list)
            ):
                return False
            validate_container_command(
                json.dumps(command["inspected_server_argv"]), variant
            )
            matrix_path = _verified_path(
                root, matrix_record.get("path"), matrix_record.get("sha256")
            )
            matrix_state = read_json(matrix_path)
            if (
                not isinstance(matrix_state, Mapping)
                or matrix_state.get("schema_version") != 1
                or matrix_state.get("complete") is not True
                or matrix_state.get("run_id") != run_id
                or matrix_state.get("variant") != dict(variant_raw)
                or matrix_state.get("image_digest") != summary.get("image_digest")
                or matrix_state.get("model_revision")
                != summary.get("model_revision")
                or matrix_state.get("scenarios") != summary.get("scenarios")
            ):
                return False
            matrix_telemetry = matrix_state.get("telemetry")
            if not isinstance(matrix_telemetry, Mapping) or (
                matrix_telemetry.get("path") != telemetry_record.get("path")
                or matrix_telemetry.get("sha256")
                != telemetry_record.get("sha256")
            ):
                return False
            if (
                parse_capacity_tokens(server_log) != measured_capacity
                or summary.get("capacity_validation")
                != capacity_validation(profile, variant, measured_capacity, server_log)
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


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
    source = _source_identity()
    if data_ready and lock_ready and manifest is not None and lock is not None:
        run_id = canonical_run_id(
            sha256_file(profile_path),
            str(lock["image_digest"]),
            str(lock["model_revision"]),
            str(manifest["dataset_hash"]),
            str(source["tree_sha256"]),
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
        dataset_hash = str(manifest["dataset_hash"])
        smoke_ready = _smoke_status_is_complete(root, run_id)
        probe_ready = smoke_ready and _probe_status_is_complete(
            root, run_id, dataset_hash, profile
        )
        selection_ready = probe_ready and _selection_status_is_complete(
            root, run_id, dataset_hash, profile
        )
        quality_ready = selection_ready and _quality_status_is_complete(
            root, run_id, dataset_hash, profile
        )
        perf_ready = selection_ready and _perf_status_is_complete(
            root, run_id, dataset_hash, profile
        )
        steps.update(
            {
                "smoke": smoke_ready,
                "probe": probe_ready,
                "select": selection_ready,
                "evaluate": quality_ready,
                "benchmark": perf_ready,
                "report": quality_ready
                and perf_ready
                and _completion_manifest_is_valid(root, run_id),
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
        "source_tree_sha256": source["tree_sha256"],
        "git_commit": source["git_commit"],
        "git_dirty": source["git_dirty"],
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
    text = re.sub(
        r"(?i)([?&](?:access_token|api_key|auth|authorization|key|token|"
        r"x-amz-signature|x-goog-signature)=)[^&\s\"']+",
        r"\1***",
        text,
    )
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


def _add_common(
    parser: argparse.ArgumentParser, *, port: bool = False, force: bool = False
) -> None:
    parser.add_argument(
        "--project-root",
        default=str(REPOSITORY_ROOT),
        help="AutoKV-Skip repository/output root",
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    if port:
        parser.add_argument("--port", type=int, default=8000)
    if force:
        parser.add_argument(
            "--force",
            metavar="CONFIG_ID",
            help=(
                "archive and rerun only this exact 12-character configuration ID"
            ),
        )


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
    force_commands = {"smoke", "probe", "evaluate", "benchmark"}
    for name, description in descriptions.items():
        command_parser = subparsers.add_parser(name, help=description, description=description)
        _add_common(
            command_parser,
            port=name in gpu_port_commands,
            force=name in force_commands,
        )
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
        return run_smoke(context, args.port, args.force)
    if command == "probe":
        return run_probe(context, args.port, args.force)
    if command == "evaluate":
        return run_evaluate(context, args.port, args.force)
    if command == "benchmark":
        return run_benchmark(context, args.port, args.force)
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
    except (DoctorError, IndexError, KeyError, ValueError, TypeError) as exc:
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
