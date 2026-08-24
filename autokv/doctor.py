"""Read-only target-host gates and immutable image/model lock creation."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from autokv.commands import CommandResult, image_probe_commands, run_command
from autokv.config import Profile
from autokv.io import atomic_write_json, redact


REQUIRED_FLAGS = (
    "--attention-backend",
    "--kv-cache-dtype",
    "--kv-cache-dtype-skip-layers",
    "--kv-cache-memory-bytes",
    "--calculate-kv-scales",
)


class DoctorError(RuntimeError):
    """Raised when an immutable environment gate fails."""


@dataclass(frozen=True)
class HostFacts:
    gpu_name: str
    driver: str
    vram_mib: int
    compute_capability: str


@dataclass(frozen=True)
class Gate:
    name: str
    ok: bool
    observed: str
    expected: str
    remediation: str


@dataclass(frozen=True)
class ImageIdentity:
    image_id: str
    image_ref: str


Runner = Callable[..., CommandResult]
MetadataLoader = Callable[[str], Mapping[str, Any]]


def parse_gpu_csv(output: str) -> HostFacts:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one GPU row, observed {len(lines)}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4:
        raise ValueError("nvidia-smi GPU row must contain name, driver, VRAM and capability")
    try:
        vram_mib = int(fields[2])
    except ValueError as exc:
        raise ValueError(f"invalid GPU memory value: {fields[2]}") from exc
    return HostFacts(fields[0], fields[1], vram_mib, fields[3])


def validate_host(facts: HostFacts, expected_driver: str) -> tuple[Gate, ...]:
    return (
        Gate(
            "gpu_name",
            facts.gpu_name == "NVIDIA RTX A6000",
            facts.gpu_name,
            "NVIDIA RTX A6000",
            "确认连接的是目标 A6000 服务器。",
        ),
        Gate(
            "driver",
            facts.driver == expected_driver,
            facts.driver,
            expected_driver,
            "确认连接的是目标服务器；不要修改驱动。",
        ),
        Gate(
            "compute_capability",
            facts.compute_capability == "8.6",
            facts.compute_capability,
            "8.6",
            "确认 GPU 型号；不要安装宿主 CUDA 或修改驱动。",
        ),
        Gate(
            "vram_mib",
            facts.vram_mib >= 48000,
            str(facts.vram_mib),
            ">=48000 MiB",
            "检查是否为 48 GiB A6000，以及是否连接了正确节点。",
        ),
    )


def classify_pull_failure(message: str) -> str:
    lowered = message.lower()
    if any(token in lowered for token in ("timeout", "tls", "connection", "network")):
        return "network"
    if any(token in lowered for token in ("unauthorized", "denied", "authentication")):
        return "auth"
    if any(token in lowered for token in ("no space", "disk quota", "not enough space")):
        return "disk"
    return "other"


def fallback_allowed(stage: str, pull_failure_class: str | None) -> bool:
    return pull_failure_class is None and stage in {
        "cuda_probe",
        "feature_probe",
        "flashinfer_probe",
    }


def missing_required_flags(help_text: str) -> tuple[str, ...]:
    return tuple(flag for flag in REQUIRED_FLAGS if flag not in help_text)


def parse_cuda_probe(output: str) -> Mapping[str, Any]:
    payload: Any = None
    for line in reversed([line.strip() for line in output.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, Mapping):
        raise ValueError("CUDA probe did not emit a JSON object")
    if payload.get("cuda_available") is not True:
        raise ValueError("CUDA probe reports cuda_available=false")
    if int(payload.get("device_count", 0)) != 1:
        raise ValueError("CUDA probe must observe exactly one GPU")
    if payload.get("compute_capability") != [8, 6]:
        raise ValueError("CUDA probe did not observe compute capability 8.6")
    if payload.get("gpu_name") != "NVIDIA RTX A6000":
        raise ValueError("CUDA probe did not observe NVIDIA RTX A6000")
    return payload


def parse_hf_revision(metadata: Mapping[str, Any]) -> str:
    revision = metadata.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Hugging Face metadata must contain a 40-character commit sha")
    return revision.lower()


def parse_image_inspect(output: str) -> ImageIdentity:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("docker image inspect did not emit valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("docker image inspect must emit one JSON object")
    image_id = payload.get("Id")
    repo_digests = payload.get("RepoDigests")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("docker image inspect is missing Id")
    if not isinstance(repo_digests, list) or not repo_digests:
        raise ValueError("docker image inspect is missing RepoDigests")
    image_ref = repo_digests[0]
    if not isinstance(image_ref, str) or not re.search(
        r"@sha256:[0-9a-fA-F]{64}$", image_ref
    ):
        raise ValueError("docker image inspect has no immutable sha256 RepoDigest")
    return ImageIdentity(image_id, image_ref)


def fetch_hf_metadata(model_id: str) -> Mapping[str, Any]:
    encoded_model = urllib.parse.quote(model_id, safe="/")
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{encoded_model}",
        headers={"Accept": "application/json", "User-Agent": "autokv-skip/0.1"},
    )
    token = os.environ.get("HF_TOKEN", "")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, Mapping):
        raise ValueError("Hugging Face model endpoint returned a non-object")
    return payload


def _event(stage: str, image: str | None, result: CommandResult) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN", "")
    secrets = [token]
    return {
        "stage": stage,
        "image": image,
        "returncode": result.returncode,
        "stdout": redact(result.stdout[-4000:], secrets),
        "stderr": redact(result.stderr[-4000:], secrets),
        "duration_seconds": result.duration_seconds,
    }


def _run(runner: Runner, argv: Sequence[str], timeout: float) -> CommandResult:
    return runner(tuple(argv), timeout=timeout)


def lock_first_compatible_image(
    profile: Profile,
    project_root: Path,
    *,
    runner: Runner = run_command,
    hf_metadata_loader: MetadataLoader = fetch_hf_metadata,
) -> dict[str, Any]:
    environment_dir = project_root.resolve() / "runs" / "_environment"
    doctor_path = environment_dir / "doctor.json"
    lock_path = environment_dir / "lock.json"
    events: list[dict[str, Any]] = []

    gpu_result = _run(
        runner,
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ),
        30,
    )
    events.append(_event("host_gpu", None, gpu_result))
    if not gpu_result.ok:
        atomic_write_json(doctor_path, {"ok": False, "events": events})
        raise DoctorError("nvidia-smi read-only query failed; do not modify the driver")
    facts = parse_gpu_csv(gpu_result.stdout)
    gates = validate_host(facts, profile.hardware.driver)
    if not all(gate.ok for gate in gates):
        atomic_write_json(
            doctor_path,
            {"ok": False, "host": asdict(facts), "gates": [asdict(g) for g in gates], "events": events},
        )
        raise DoctorError("host does not match the locked A6000/535.230.02 profile")

    docker_result = _run(
        runner, ("docker", "version", "--format", "{{.Server.Version}}"), 30
    )
    events.append(_event("docker_daemon", None, docker_result))
    if not docker_result.ok:
        atomic_write_json(
            doctor_path,
            {"ok": False, "host": asdict(facts), "gates": [asdict(g) for g in gates], "events": events},
        )
        raise DoctorError("Docker daemon is unavailable; the doctor never installs it")

    selected_tag: str | None = None
    selected_payload: Mapping[str, Any] | None = None
    selected_identity: ImageIdentity | None = None

    for image in profile.images:
        pull_argv, cuda_argv, help_argv = image_probe_commands(image)
        pull_result = _run(runner, pull_argv, 1800)
        events.append(_event("pull", image, pull_result))
        if not pull_result.ok:
            failure_class = classify_pull_failure(
                f"{pull_result.stdout}\n{pull_result.stderr}"
            )
            atomic_write_json(
                doctor_path,
                {
                    "ok": False,
                    "host": asdict(facts),
                    "gates": [asdict(g) for g in gates],
                    "pull_failure_class": failure_class,
                    "events": events,
                },
            )
            raise DoctorError(
                f"image pull failed ({failure_class}); version fallback is not allowed"
            )

        cuda_result = _run(runner, cuda_argv, 300)
        events.append(_event("cuda_probe", image, cuda_result))
        if not cuda_result.ok:
            if fallback_allowed("cuda_probe", None):
                continue
            raise DoctorError("CUDA probe failed")
        try:
            cuda_payload = parse_cuda_probe(cuda_result.stdout)
        except ValueError as exc:
            events.append(
                {
                    "stage": "cuda_probe_parse",
                    "image": image,
                    "returncode": 1,
                    "stderr": str(exc),
                }
            )
            continue

        help_result = _run(runner, help_argv, 120)
        events.append(_event("feature_probe", image, help_result))
        missing = () if not help_result.ok else missing_required_flags(help_result.stdout)
        if not help_result.ok or missing:
            events.append(
                {
                    "stage": "feature_probe_result",
                    "image": image,
                    "returncode": 1,
                    "missing_flags": list(missing),
                }
            )
            continue

        inspect_result = _run(
            runner,
            (
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{json .}}",
            ),
            30,
        )
        events.append(_event("image_inspect", image, inspect_result))
        if not inspect_result.ok:
            raise DoctorError("compatible image could not be inspected for an immutable digest")
        selected_identity = parse_image_inspect(inspect_result.stdout)
        selected_tag = image
        selected_payload = cuda_payload
        break

    if selected_tag is None or selected_payload is None or selected_identity is None:
        atomic_write_json(
            doctor_path,
            {"ok": False, "host": asdict(facts), "gates": [asdict(g) for g in gates], "events": events},
        )
        raise DoctorError(
            "neither pinned vLLM image passed CUDA/FlashInfer/feature gates; do not change the driver"
        )

    model_metadata = hf_metadata_loader(profile.model.model_id)
    model_revision = parse_hf_revision(model_metadata)
    lock = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": asdict(facts),
        "image_tag": selected_tag,
        "image_id": selected_identity.image_id,
        "image_ref": selected_identity.image_ref,
        "image_digest": selected_identity.image_ref.split("@", 1)[1],
        "model_id": profile.model.model_id,
        "model_revision": model_revision,
        "versions": {
            key: selected_payload.get(key)
            for key in ("torch", "cuda", "vllm", "flashinfer")
        },
        "compatibility_env": "VLLM_ENABLE_CUDA_COMPATIBILITY=1",
    }
    atomic_write_json(lock_path, lock)
    atomic_write_json(
        doctor_path,
        {
            "ok": True,
            "host": asdict(facts),
            "gates": [asdict(gate) for gate in gates],
            "selected_image": selected_identity.image_ref,
            "events": events,
        },
    )
    return lock
