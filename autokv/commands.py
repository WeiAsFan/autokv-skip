"""Shell-free subprocess execution and Docker/local vLLM command construction."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from autokv.config import Profile
from autokv.io import redact
from autokv.selection import Variant, canonical_config_id


PROJECT_LABEL = "io.autokv.project=autokv-skip"
COMPATIBILITY_ENV = "VLLM_ENABLE_CUDA_COMPATIBILITY=1"
VLLM_DEBUG_ENV = "VLLM_LOGGING_LEVEL=DEBUG"
HF_CACHE_IN_CONTAINER = "/root/.cache/huggingface"


LOCAL_CUDA_HOME = "/usr/local/cuda"
LOCAL_NVCC = "/usr/local/cuda/bin/nvcc"

def runtime_identity(lock: Mapping[str, str]) -> str:
    """Return the immutable identity for either backend."""
    backend = lock.get("backend", "docker")
    key = "runtime_id" if backend == "local_vllm" else "image_digest"
    value = lock.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"environment lock is missing {key}")
    return value


def local_cuda_toolchain_env(
    environment: Mapping[str, str],
    *,
    cuda_home: str = LOCAL_CUDA_HOME,
    nvcc: str = LOCAL_NVCC,
) -> dict[str, str]:
    """Return an environment with a deterministic host CUDA toolchain."""
    configured = dict(environment)
    configured["CUDA_HOME"] = cuda_home
    configured["CUDA_PATH"] = cuda_home
    configured["FLASHINFER_NVCC"] = nvcc
    path = configured.get("PATH", "")
    configured["PATH"] = f"{cuda_home}/bin" + (os.pathsep + path if path else "")
    return configured


def local_vllm_env(lock: Mapping[str, str]) -> dict[str, str]:
    if lock.get("backend") != "local_vllm":
        raise ValueError("local_vllm_env requires a local_vllm lock")
    environment = os.environ.copy()
    library_path = lock.get("ld_library_path")
    if not isinstance(library_path, str) or not library_path:
        raise ValueError("local vLLM lock has no LD_LIBRARY_PATH")
    cuda_home = lock.get("cuda_home", LOCAL_CUDA_HOME)
    nvcc = lock.get("nvcc", f"{cuda_home}/bin/nvcc")
    if not isinstance(cuda_home, str) or not cuda_home:
        raise ValueError("local vLLM lock has no CUDA_HOME")
    if not isinstance(nvcc, str) or not nvcc:
        raise ValueError("local vLLM lock has no nvcc path")
    environment = local_cuda_toolchain_env(
        environment,
        cuda_home=cuda_home,
        nvcc=nvcc,
    )
    for environment_name, lock_key in (
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer_workspace_base"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions_dir"),
    ):
        cache_path = lock.get(lock_key)
        if isinstance(cache_path, str) and cache_path:
            environment[environment_name] = cache_path
    environment["LD_LIBRARY_PATH"] = library_path
    environment["VLLM_ENABLE_CUDA_COMPATIBILITY"] = "1"
    environment["VLLM_LOGGING_LEVEL"] = "DEBUG"
    return environment


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    argv: Sequence[str],
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> CommandResult:
    if not argv:
        raise ValueError("command argv cannot be empty")
    merged_env = None
    if env is not None:
        merged_env = os.environ.copy()
        merged_env.update(env)
    start = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            shell=False,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=merged_env,
            cwd=None if cwd is None else str(cwd),
        )
    except subprocess.TimeoutExpired as exc:
        def as_text(value: str | bytes | None) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value

        elapsed = time.monotonic() - start
        stderr = as_text(exc.stderr)
        timeout_text = f"command timed out after {exc.timeout:g} seconds"
        stderr = f"{stderr.rstrip()}\n{timeout_text}" if stderr else timeout_text
        return CommandResult(
            argv=tuple(str(item) for item in argv),
            returncode=124,
            stdout=as_text(exc.stdout),
            stderr=stderr,
            duration_seconds=elapsed,
        )
    return CommandResult(
        argv=tuple(str(item) for item in argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=time.monotonic() - start,
    )


def format_command(argv: Sequence[str], secrets: Iterable[str] = ()) -> str:
    return shlex.join([redact(str(item), secrets) for item in argv])


def _safe_fragment(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-.")
    return normalized or "run"


def container_name(run_id: str, variant: Variant) -> str:
    prefix = f"autokv-{_safe_fragment(run_id)}-{canonical_config_id(variant)}"
    return prefix[:63].rstrip("-.")


def image_probe_commands(image: str) -> tuple[tuple[str, ...], ...]:
    if not image.strip():
        raise ValueError("image cannot be empty")
    python_probe = (
        "import json, torch, flashinfer, vllm; "
        "print(json.dumps({"
        "'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count(), "
        "'compute_capability': list(torch.cuda.get_device_capability(0)), "
        "'gpu_name': torch.cuda.get_device_name(0), "
        "'torch': torch.__version__, 'cuda': torch.version.cuda, "
        "'vllm': vllm.__version__, "
        "'flashinfer': getattr(flashinfer, '__version__', 'unknown')}))"
    )
    return (
        ("docker", "pull", image),
        (
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-e",
            COMPATIBILITY_ENV,
            "--entrypoint",
            "python",
            image,
            "-c",
            python_probe,
        ),
        (
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "vllm",
            image,
            "serve",
            "--help",
        ),
    )


def server_command(
    profile: Profile,
    image_ref: str,
    variant: Variant,
    project_root: Path,
    port: int,
    run_id: str,
    model_revision: str | None = None,
) -> tuple[str, ...]:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not image_ref.strip():
        raise ValueError("image_ref cannot be empty")
    if model_revision is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", model_revision):
        raise ValueError("model_revision must be a 40-character commit hash")
    variant.validate_for_model(profile.model.num_layers)

    cache = project_root / ".cache" / "huggingface"
    argv: list[str] = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name(run_id, variant),
        "--label",
        PROJECT_LABEL,
        "--label",
        f"io.autokv.run={_safe_fragment(run_id)}",
        "--gpus",
        "all",
        "--ipc=host",
        "--network=host",
        "-e",
        COMPATIBILITY_ENV,
        "-e",
        VLLM_DEBUG_ENV,
        "-e",
        f"HF_HOME={HF_CACHE_IN_CONTAINER}",
        "-e",
        "HF_TOKEN",
        "-v",
        f"{cache}:{HF_CACHE_IN_CONTAINER}",
        image_ref,
        "--model",
        profile.model.model_id,
    ]
    if model_revision is not None:
        argv.extend(("--revision", model_revision))
    argv.extend(
        (
            "--dtype",
            profile.model.dtype,
            "--attention-backend",
            profile.model.attention_backend,
            "--max-model-len",
            str(profile.model.max_model_len),
            "--kv-cache-memory-bytes",
            profile.kv_cache_memory,
            "--seed",
            str(profile.seed),
            "--tensor-parallel-size",
            "1",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--kv-cache-dtype",
            variant.kv_dtype,
        )
    )
    if variant.kv_dtype == "fp8_e4m3" and profile.calculate_kv_scales:
        argv.append("--calculate-kv-scales")
    if variant.skip_layers:
        argv.append("--kv-cache-dtype-skip-layers")
        argv.extend(str(layer) for layer in variant.skip_layers)
    return tuple(argv)


def _model_snapshot(project_root: Path, model_id: str, revision: str) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("model_revision must be a 40-character commit hash")
    model_cache_name = "models--" + model_id.replace("/", "--")
    snapshot = (
        project_root
        / ".cache"
        / "huggingface"
        / model_cache_name
        / "snapshots"
        / revision.lower()
    ).resolve()
    if not snapshot.is_dir():
        raise ValueError(f"local Hugging Face snapshot is missing: {snapshot}")
    return snapshot


def local_server_command(
    profile: Profile,
    vllm_python: str,
    variant: Variant,
    project_root: Path,
    port: int,
    model_revision: str,
    model_path: Path | None = None,
) -> tuple[str, ...]:
    """Build the direct local-vLLM equivalent of ``server_command``."""
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not vllm_python.strip():
        raise ValueError("vllm_python cannot be empty")
    variant.validate_for_model(profile.model.num_layers)
    model_path = model_path or _model_snapshot(
        project_root, profile.model.model_id, model_revision
    )
    argv: list[str] = [
        vllm_python,
        "serve",
        "--model",
        str(model_path),
        "--served-model-name",
        profile.model.model_id,
        "--dtype",
        profile.model.dtype,
        "--attention-backend",
        profile.model.attention_backend,
        "--max-model-len",
        str(profile.model.max_model_len),
        "--kv-cache-memory-bytes",
        profile.kv_cache_memory,
        "--seed",
        str(profile.seed),
        "--tensor-parallel-size",
        "1",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--kv-cache-dtype",
        variant.kv_dtype,
    ]
    if variant.kv_dtype == "fp8_e4m3" and profile.calculate_kv_scales:
        argv.append("--calculate-kv-scales")
    if variant.skip_layers:
        argv.append("--kv-cache-dtype-skip-layers")
        argv.extend(str(layer) for layer in variant.skip_layers)
    return tuple(argv)


def _container_result_path(project_root: Path, relative_path: Path) -> str:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("benchmark result path must be project-relative")
    return "/workspace/autokv-skip/" + relative_path.as_posix()


def benchmark_container_name(result_relative_path: Path) -> str:
    if result_relative_path.is_absolute() or ".." in result_relative_path.parts:
        raise ValueError("benchmark result path must be project-relative")
    digest = hashlib.sha256(
        result_relative_path.as_posix().encode("utf-8")
    ).hexdigest()[:12]
    return f"autokv-bench-{digest}"


def bench_command(
    profile: Profile,
    image_ref: str,
    project_root: Path,
    port: int,
    input_length: int,
    output_length: int,
    result_relative_path: Path,
    model_revision: str,
) -> tuple[str, ...]:
    if input_length <= 0 or output_length <= 0:
        raise ValueError("benchmark lengths must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", model_revision):
        raise ValueError("benchmark model_revision must be a 40-character commit hash")
    result_path = _container_result_path(project_root, result_relative_path)
    result_directory, result_filename = result_path.rsplit("/", 1)
    model_cache_name = "models--" + profile.model.model_id.replace("/", "--")
    tokenizer_path = (
        f"/workspace/autokv-skip/.cache/huggingface/hub/{model_cache_name}/"
        f"snapshots/{model_revision.lower()}"
    )
    return (
        "docker",
        "run",
        "--rm",
        "--name",
        benchmark_container_name(result_relative_path),
        "--network",
        "host",
        "--gpus",
        "all",
        "-e",
        COMPATIBILITY_ENV,
        "-e",
        "HF_HOME=/workspace/autokv-skip/.cache/huggingface",
        "-e",
        "HF_TOKEN",
        "--label",
        PROJECT_LABEL,
        "-v",
        f"{project_root}:/workspace/autokv-skip",
        "--entrypoint",
        "vllm",
        image_ref,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--endpoint",
        "/v1/completions",
        "--model",
        profile.model.model_id,
        "--tokenizer",
        tokenizer_path,
        "--seed",
        str(profile.seed),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_length),
        "--random-output-len",
        str(output_length),
        "--num-prompts",
        str(profile.benchmark.num_prompts),
        "--request-rate",
        "inf",
        "--temperature",
        "0",
        "--num-warmups",
        "1",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "90,99",
        "--save-result",
        "--result-dir",
        result_directory,
        "--result-filename",
        result_filename,
    )


def local_bench_command(
    profile: Profile,
    vllm_python: str,
    project_root: Path,
    port: int,
    input_length: int,
    output_length: int,
    result_path: Path,
    model_revision: str,
    model_path: Path | None = None,
) -> tuple[str, ...]:
    """Build a local ``vllm bench serve`` command from the project venv."""
    if input_length <= 0 or output_length <= 0:
        raise ValueError("benchmark lengths must be positive")
    if result_path.is_absolute() is False:
        result_path = (project_root / result_path).resolve()
    result_path = result_path.resolve()
    try:
        result_path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("benchmark result path must be project-relative") from exc
    model_path = model_path or _model_snapshot(
        project_root, profile.model.model_id, model_revision
    )
    return (
        vllm_python,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--endpoint",
        "/v1/completions",
        "--model",
        profile.model.model_id,
        "--tokenizer",
        str(model_path),
        "--seed",
        str(profile.seed),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_length),
        "--random-output-len",
        str(output_length),
        "--num-prompts",
        str(profile.benchmark.num_prompts),
        "--request-rate",
        "inf",
        "--temperature",
        "0",
        "--num-warmups",
        "1",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "90,99",
        "--save-result",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    )
