"""vLLM capacity parsing and resumable service benchmarking."""

from __future__ import annotations

import itertools
import json
import re
import statistics
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from autokv.client import VllmClient, wait_until_ready
from autokv.commands import (
    CommandResult,
    bench_command,
    benchmark_container_name,
    container_name,
    local_bench_command,
    local_server_command,
    local_vllm_env,
    run_command,
    runtime_identity,
    server_command,
)
from autokv.config import Profile
from autokv.experiment import (
    MAX_TIMEOUT_RESTARTS,
    archive_exact_artifacts,
    inspect_container_command,
    safe_cleanup_owned_container,
    safe_remove_stale_container,
    validate_container_command,
    validate_local_vllm_command,
    validate_server_log,
)
from autokv.local_runtime import LocalVllmProcess
from autokv.io import (
    atomic_write_json,
    atomic_write_text,
    ensure_within,
    read_json,
    sha256_file,
)
from autokv.selection import Variant, canonical_config_id


Runner = Callable[..., CommandResult]
ClientFactory = Callable[[str, str], Any]


class Telemetry(Protocol):
    command: tuple[str, ...]

    def start(self) -> None: ...

    def stop(self) -> None: ...


TelemetryFactory = Callable[[Path], Telemetry]


@dataclass(frozen=True)
class Capacity:
    tokens: int
    model_length: int | None
    max_concurrency: float | None


@dataclass(frozen=True)
class BenchmarkCase:
    input_length: int
    output_length: int
    repeat: int


class NvidiaDmonSampler:
    """Capture read-only GPU telemetry while the benchmark matrix is running."""

    command = ("nvidia-smi", "dmon", "-s", "pucm", "-d", "1", "-o", "DT")

    def __init__(self, path: Path) -> None:
        self.path = path
        self._process: subprocess.Popen[str] | None = None
        self._stream: Any = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("nvidia-smi dmon telemetry is already running")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8", newline="\n")
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=self._stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except BaseException:
            self._stream.close()
            self._stream = None
            raise

    def stop(self) -> None:
        process = self._process
        stream = self._stream
        self._process = None
        self._stream = None
        early_returncode: int | None = None
        try:
            if process is not None:
                if process.poll() is not None:
                    early_returncode = process.returncode
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
        finally:
            if stream is not None:
                stream.flush()
                stream.close()
        if early_returncode is not None:
            raise RuntimeError(
                "nvidia-smi dmon exited before benchmark completion "
                f"with code {early_returncode}; inspect {self.path}"
            )


def parse_capacity_tokens(log: str) -> Capacity:
    capacity_match = re.search(
        r"GPU KV cache size:\s*([\d,]+)\s*tokens", log, flags=re.IGNORECASE
    )
    if capacity_match is None:
        raise ValueError("server log does not contain GPU KV cache size")
    tokens = int(capacity_match.group(1).replace(",", ""))
    concurrency_match = re.search(
        r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x",
        log,
        flags=re.IGNORECASE,
    )
    if concurrency_match is None:
        return Capacity(tokens=tokens, model_length=None, max_concurrency=None)
    return Capacity(
        tokens=tokens,
        model_length=int(concurrency_match.group(1).replace(",", "")),
        max_concurrency=float(concurrency_match.group(2)),
    )


def build_benchmark_matrix(profile: Profile) -> tuple[BenchmarkCase, ...]:
    return tuple(
        BenchmarkCase(input_length, output_length, repeat)
        for repeat, input_length, output_length in itertools.product(
            range(profile.benchmark.repeats),
            profile.benchmark.input_lengths,
            profile.benchmark.output_lengths,
        )
    )


_PERFORMANCE_KEYS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p90_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p90_e2el_ms",
    "p99_e2el_ms",
)


def extract_performance_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in _PERFORMANCE_KEYS:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = float(value)
    completed = payload.get("completed")
    if not isinstance(completed, int) or isinstance(completed, bool):
        raise ValueError("vLLM benchmark result has no valid completed count")
    failed = payload.get("failed")
    if not isinstance(failed, int) or isinstance(failed, bool):
        raise ValueError("vLLM benchmark result has no valid failed count")
    if completed <= 0:
        raise ValueError(
            f"vLLM benchmark completed no requests (completed={completed})"
        )
    if failed != 0:
        raise ValueError(
            f"vLLM benchmark result contains failed requests (failed={failed})"
        )
    if "request_throughput" not in metrics:
        raise ValueError("vLLM benchmark result has no request_throughput")
    return metrics


def aggregate_scenario_groups(
    scenarios: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        key = (int(scenario["input_length"]), int(scenario["output_length"]))
        metrics = scenario.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("benchmark scenario is missing metrics")
        grouped.setdefault(key, []).append(metrics)
    result: list[dict[str, Any]] = []
    for (input_length, output_length), repetitions in sorted(grouped.items()):
        values_by_metric: dict[str, list[float]] = {}
        for metrics in repetitions:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values_by_metric.setdefault(str(key), []).append(float(value))
        result.append(
            {
                "input_length": input_length,
                "output_length": output_length,
                "repeats": len(repetitions),
                "metrics": {
                    key: statistics.fmean(values)
                    for key, values in sorted(values_by_metric.items())
                },
            }
        )
    return result


def _kv_budget_bytes(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([KMG])", value)
    if match is None:
        raise ValueError(f"unsupported KV cache memory budget: {value}")
    powers = {"K": 1, "M": 2, "G": 3}
    return int(match.group(1)) * 1024 ** powers[match.group(2)]


def _bytes_per_token(profile: Profile, variant: Variant) -> int:
    dtype_bytes_by_layer = (
        profile.model.num_layers * 2
        if variant.kv_dtype == "bfloat16"
        else profile.model.num_layers + len(variant.skip_layers)
    )
    return (
        2
        * profile.model.num_kv_heads
        * profile.model.head_dim
        * dtype_bytes_by_layer
    )


def capacity_validation(
    profile: Profile, variant: Variant, capacity: Capacity, server_log: str
) -> dict[str, Any]:
    budget_bytes = _kv_budget_bytes(profile.kv_cache_memory)
    bytes_per_token = _bytes_per_token(profile, variant)
    theoretical_tokens = budget_bytes // bytes_per_token
    relative_deviation = abs(capacity.tokens - theoretical_tokens) / theoretical_tokens
    runtime_evidence = [
        line.strip()
        for line in server_log.splitlines()
        if re.search(
            r"(?:GPU KV cache|cache memory|kv_cache_memory|padding layers|\bblocks?\b|\bpages?\b)",
            line,
            flags=re.IGNORECASE,
        )
    ]
    alignment_evidence: list[dict[str, Any]] = []
    quantitative_explanations: list[str] = []
    explanation_margin = 0.02
    signed_shortfall = (theoretical_tokens - capacity.tokens) / theoretical_tokens

    padding_pattern = re.compile(
        r"Add\s+(\d+)\s+padding layers?.*?waste at most\s+([\d.]+)%\s+KV cache memory",
        flags=re.IGNORECASE,
    )
    available_pattern = re.compile(
        r"Available KV cache memory:\s*([\d.]+)\s*(KiB|MiB|GiB)",
        flags=re.IGNORECASE,
    )
    unit_bytes = {"kib": 1024, "mib": 1024**2, "gib": 1024**3}
    for line in runtime_evidence:
        padding = padding_pattern.search(line)
        if padding is not None:
            max_waste_fraction = float(padding.group(2)) / 100.0
            record = {
                "kind": "padding_waste_bound",
                "padding_layers": int(padding.group(1)),
                "max_waste_fraction": max_waste_fraction,
                "line": line,
            }
            alignment_evidence.append(record)
            if (
                signed_shortfall > 0.10
                and signed_shortfall <= max_waste_fraction + explanation_margin
            ):
                quantitative_explanations.append(
                    "measured shortfall is bounded by the logged maximum padding waste"
                )
        available = available_pattern.search(line)
        if available is not None:
            available_bytes = int(
                float(available.group(1)) * unit_bytes[available.group(2).lower()]
            )
            predicted_tokens = available_bytes // bytes_per_token
            prediction_error = (
                abs(capacity.tokens - predicted_tokens) / theoretical_tokens
            )
            record = {
                "kind": "available_kv_memory",
                "available_bytes": available_bytes,
                "predicted_tokens": predicted_tokens,
                "prediction_error_vs_budget_theory": prediction_error,
                "line": line,
            }
            alignment_evidence.append(record)
            if relative_deviation > 0.10 and prediction_error <= explanation_margin:
                quantitative_explanations.append(
                    "measured tokens agree with the logged available KV memory"
                )
    within_tolerance = relative_deviation <= 0.10
    return {
        "kv_budget_bytes": budget_bytes,
        "bytes_per_token": bytes_per_token,
        "theoretical_tokens": theoretical_tokens,
        "measured_tokens": capacity.tokens,
        "relative_deviation": relative_deviation,
        "within_10_percent": within_tolerance,
        "runtime_evidence": runtime_evidence,
        "alignment_evidence": alignment_evidence,
        "quantitative_explanations": quantitative_explanations,
        "explanation_margin": explanation_margin,
        "evidence_complete": within_tolerance or bool(quantitative_explanations),
    }


def telemetry_is_usable(path: Path) -> bool:
    try:
        return any(
            re.match(r"^\d{8}\s+\d{2}:\d{2}:\d{2}\s+", line.strip())
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        )
    except OSError:
        return False


class BenchmarkRunner:
    """Run the fixed benchmark matrix for one isolated server configuration."""

    def __init__(
        self,
        profile: Profile,
        project_root: Path,
        lock: Mapping[str, Any],
        *,
        command_runner: Runner = run_command,
        client_factory: ClientFactory | None = None,
        telemetry_factory: TelemetryFactory | None = None,
        force_config_id: str | None = None,
    ) -> None:
        self.profile = profile
        self.project_root = project_root.resolve()
        self.lock = lock
        self.command_runner = command_runner
        self.client_factory = client_factory or (
            lambda base_url, model_id: VllmClient(base_url, model_id)
        )
        self.telemetry_factory = telemetry_factory or NvidiaDmonSampler
        if force_config_id is not None and not re.fullmatch(
            r"[0-9a-f]{12}", force_config_id
        ):
            raise ValueError("force config ID must be 12 lowercase hex characters")
        self.force_config_id = force_config_id
        self._forced_keys: set[tuple[str, str]] = set()
        required = (
            ("runtime_id", "python", "vllm", "model_revision")
            if lock.get("backend") == "local_vllm"
            else ("image_ref", "image_digest", "model_revision")
        )
        for key in required:
            if not isinstance(lock.get(key), str) or not lock[key]:
                raise ValueError(f"environment lock is missing {key}")

    @property
    def local_backend(self) -> bool:
        return self.lock.get("backend") == "local_vllm"

    @property
    def force_applied(self) -> bool:
        return bool(self._forced_keys)

    def _paths(
        self, run_id: str, variant: Variant
    ) -> tuple[Path, Path, Path, Path]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("run_id contains unsafe characters")
        stem = f"{variant.name}-{canonical_config_id(variant)}"
        perf_root = ensure_within(
            self.project_root, self.project_root / "runs" / run_id / "perf"
        )
        directory = ensure_within(self.project_root, perf_root / stem)
        return (
            perf_root / f"{stem}.summary.json",
            perf_root / f"{stem}.server.log",
            perf_root / f"{stem}.command.json",
            directory,
        )

    @staticmethod
    def _matrix_state_path(summary_path: Path) -> Path:
        return summary_path.with_name(
            summary_path.name.removesuffix(".summary.json")
            + ".matrix.state.json"
        )

    def _matrix_state_is_complete(
        self,
        state_path: Path,
        *,
        run_id: str,
        variant: Variant,
        telemetry_path: Path,
        result_directory: Path,
    ) -> bool:
        try:
            state = read_json(state_path)
            matrix = build_benchmark_matrix(self.profile)
            expected_variant = {
                "name": variant.name,
                "kv_dtype": variant.kv_dtype,
                "skip_layers": list(variant.skip_layers),
            }
            if (
                not isinstance(state, Mapping)
                or state.get("schema_version") != 1
                or state.get("complete") is not True
                or state.get("run_id") != run_id
                or state.get("variant") != expected_variant
                or state.get("runtime_id") != runtime_identity(self.lock)
                or state.get("model_revision") != self.lock["model_revision"]
            ):
                return False
            telemetry = state.get("telemetry")
            if not isinstance(telemetry, Mapping):
                return False
            if telemetry.get("path") != telemetry_path.relative_to(
                self.project_root
            ).as_posix() or telemetry.get("sha256") != sha256_file(telemetry_path):
                return False
            if not telemetry_is_usable(telemetry_path):
                return False
            scenarios = state.get("scenarios")
            if not isinstance(scenarios, list) or len(scenarios) != len(matrix):
                return False
            for case, scenario in zip(matrix, scenarios):
                if not isinstance(scenario, Mapping):
                    return False
                raw_path = result_directory / (
                    f"in{case.input_length}-out{case.output_length}-rep{case.repeat}.json"
                )
                if (
                    scenario.get("input_length") != case.input_length
                    or scenario.get("output_length") != case.output_length
                    or scenario.get("repeat") != case.repeat
                    or scenario.get("raw_result")
                    != raw_path.relative_to(self.project_root).as_posix()
                    or scenario.get("sha256") != sha256_file(raw_path)
                ):
                    return False
                payload = read_json(raw_path)
                if not isinstance(payload, Mapping) or scenario.get(
                    "metrics"
                ) != extract_performance_metrics(payload):
                    return False
            return True
        except (KeyError, OSError, TypeError, ValueError):
            return False

    def _is_complete(
        self,
        summary_path: Path,
        run_id: str,
        expected_scenarios: int,
        variant: Variant,
    ) -> bool:
        try:
            summary = read_json(summary_path)
            if (
                not isinstance(summary, Mapping)
                or summary.get("schema_version") != 3
                or summary.get("complete") is not True
                or summary.get("run_id") != run_id
            ):
                return False
            expected_variant = {
                "name": variant.name,
                "kv_dtype": variant.kv_dtype,
                "skip_layers": list(variant.skip_layers),
            }
            if summary.get("variant") != expected_variant:
                return False
            scenarios = summary.get("scenarios")
            if not isinstance(scenarios, list) or len(scenarios) != expected_scenarios:
                return False
            expected_cases = [asdict(case) for case in build_benchmark_matrix(self.profile)]
            for scenario in scenarios:
                if not isinstance(scenario, Mapping):
                    return False
                relative = scenario.get("raw_result")
                expected_hash = scenario.get("sha256")
                if not isinstance(relative, str) or not isinstance(expected_hash, str):
                    return False
                raw_path = ensure_within(self.project_root, self.project_root / relative)
                if not raw_path.is_file() or sha256_file(raw_path) != expected_hash:
                    return False
                payload = read_json(raw_path)
                if not isinstance(payload, Mapping):
                    return False
                if scenario.get("metrics") != extract_performance_metrics(payload):
                    return False
            actual_cases = [
                {
                    "input_length": scenario.get("input_length"),
                    "output_length": scenario.get("output_length"),
                    "repeat": scenario.get("repeat"),
                }
                for scenario in scenarios
            ]
            if actual_cases != expected_cases:
                return False
            if summary.get("scenario_groups") != aggregate_scenario_groups(scenarios):
                return False
            for artifact_key in (
                "telemetry",
                "server_log",
                "command_record",
                "matrix_state",
            ):
                record = summary.get(artifact_key)
                if not isinstance(record, Mapping):
                    return False
                path = ensure_within(
                    self.project_root, self.project_root / str(record.get("path", ""))
                )
                if not path.is_file() or sha256_file(path) != record.get("sha256"):
                    return False
                if artifact_key == "telemetry" and not telemetry_is_usable(path):
                    return False
            capacity_raw = summary.get("capacity")
            if not isinstance(capacity_raw, Mapping):
                return False
            observed_capacity = Capacity(
                tokens=int(capacity_raw["tokens"]),
                model_length=(
                    None
                    if capacity_raw.get("model_length") is None
                    else int(capacity_raw["model_length"])
                ),
                max_concurrency=(
                    None
                    if capacity_raw.get("max_concurrency") is None
                    else float(capacity_raw["max_concurrency"])
                ),
            )
            server_record = summary["server_log"]
            server_path = ensure_within(
                self.project_root, self.project_root / str(server_record["path"])
            )
            server_log = server_path.read_text(encoding="utf-8")
            validate_server_log(
                server_log, variant, num_layers=self.profile.model.num_layers
            )
            command_record = summary["command_record"]
            assert isinstance(command_record, Mapping)
            command_path = ensure_within(
                self.project_root,
                self.project_root / str(command_record["path"]),
            )
            command = read_json(command_path)
            if not isinstance(command, Mapping) or not isinstance(
                command.get("inspected_server_argv"), list
            ):
                return False
            if self.local_backend:
                validate_local_vllm_command(
                    json.dumps(command["inspected_server_argv"]),
                    variant,
                    self.profile.calculate_kv_scales,
                )
            else:
                validate_container_command(
                    json.dumps(command["inspected_server_argv"]),
                    variant,
                    self.profile.calculate_kv_scales,
                )
            if parse_capacity_tokens(server_log) != observed_capacity:
                return False
            if summary.get("capacity_validation") != capacity_validation(
                self.profile, variant, observed_capacity, server_log
            ):
                return False
            matrix_record = summary["matrix_state"]
            telemetry_record = summary["telemetry"]
            assert isinstance(matrix_record, Mapping)
            assert isinstance(telemetry_record, Mapping)
            matrix_path = ensure_within(
                self.project_root,
                self.project_root / str(matrix_record["path"]),
            )
            telemetry_path = ensure_within(
                self.project_root,
                self.project_root / str(telemetry_record["path"]),
            )
            if not self._matrix_state_is_complete(
                matrix_path,
                run_id=run_id,
                variant=variant,
                telemetry_path=telemetry_path,
                result_directory=summary_path.with_name(
                    summary_path.name.removesuffix(".summary.json")
                ),
            ):
                return False
            matrix_state = read_json(matrix_path)
            if not isinstance(matrix_state, Mapping) or matrix_state.get(
                "scenarios"
            ) != scenarios:
                return False
            return True
        except (KeyError, OSError, TypeError, ValueError):
            return False

    def _logs(self, name: str) -> str:
        result = self.command_runner(("docker", "logs", name), timeout=60)
        return result.stdout + result.stderr

    def run_variant(
        self,
        variant: Variant,
        *,
        run_id: str,
        port: int,
    ) -> Path:
        variant.validate_for_model(self.profile.model.num_layers)
        matrix = build_benchmark_matrix(self.profile)
        summary_path, log_path, command_path, result_directory = self._paths(
            run_id, variant
        )
        telemetry_path = summary_path.with_name(
            summary_path.name.removesuffix(".summary.json") + ".dmon.log"
        )
        matrix_state_path = self._matrix_state_path(summary_path)
        config_id = canonical_config_id(variant)
        force_key = (run_id, config_id)
        if self.force_config_id == config_id and force_key not in self._forced_keys:
            archive_exact_artifacts(
                self.project_root,
                run_id,
                "perf",
                config_id,
                (
                    summary_path,
                    log_path,
                    command_path,
                    telemetry_path,
                    matrix_state_path,
                    result_directory,
                ),
            )
            self._forced_keys.add(force_key)
        if self._is_complete(summary_path, run_id, len(matrix), variant):
            return summary_path
        if summary_path.exists():
            raise ValueError(
                f"benchmark summary integrity mismatch; preserve evidence and use "
                f"--force {config_id}: {summary_path}"
            )
        result_directory.mkdir(parents=True, exist_ok=True)

        model_revision = str(self.lock["model_revision"])
        if self.local_backend:
            server_argv = local_server_command(
                self.profile,
                str(self.lock["vllm"]),
                variant,
                self.project_root,
                port,
                model_revision,
            )
        else:
            image_ref = str(self.lock["image_ref"])
            server_argv = server_command(
                self.profile,
                image_ref,
                variant,
                self.project_root,
                port,
                run_id,
                model_revision,
            )
        bench_commands: list[list[str]] = []
        for case in matrix:
            raw_path = result_directory / (
                f"in{case.input_length}-out{case.output_length}-rep{case.repeat}.json"
            )
            if self.local_backend:
                command = local_bench_command(
                    self.profile,
                    str(self.lock["vllm"]),
                    self.project_root,
                    port,
                    case.input_length,
                    case.output_length,
                    raw_path,
                    model_revision,
                )
            else:
                command = bench_command(
                    self.profile,
                    image_ref,
                    self.project_root,
                    port,
                    case.input_length,
                    case.output_length,
                    raw_path.relative_to(self.project_root),
                    model_revision,
                )
            bench_commands.append(list(command))
        telemetry = self.telemetry_factory(telemetry_path)
        cached_matrix_complete = self._matrix_state_is_complete(
            matrix_state_path,
            run_id=run_id,
            variant=variant,
            telemetry_path=telemetry_path,
            result_directory=result_directory,
        )
        if matrix_state_path.exists() and not cached_matrix_complete:
            raise ValueError(
                "benchmark matrix state integrity mismatch; preserve evidence and "
                f"use --force {config_id}: {matrix_state_path}"
            )
        command_record = {
            "server_argv": list(server_argv),
            "benchmark_argv": bench_commands,
            "backend": str(self.lock.get("backend", "docker")),
            "runtime_id": runtime_identity(self.lock),
            "model_revision": model_revision,
            "variant": {
                "name": variant.name,
                "kv_dtype": variant.kv_dtype,
                "skip_layers": list(variant.skip_layers),
            },
            "telemetry_argv": list(telemetry.command),
        }
        atomic_write_json(command_path, command_record)

        name = container_name(run_id, variant)
        server_process: LocalVllmProcess | None = None
        started = False
        cleanup_required = False
        captured_log = ""
        scenarios: list[dict[str, Any]] = []
        capacity: Capacity | None = None
        telemetry_started = False
        try:
            if self.local_backend:
                server_process = LocalVllmProcess.start(
                    server_argv,
                    log_path,
                    cwd=self.project_root,
                    env=local_vllm_env(self.lock),
                )
                cleanup_required = True
            else:
                safe_remove_stale_container(name, self.command_runner)
                cleanup_required = True
                start_result = self.command_runner(server_argv, timeout=120)
                if not start_result.ok:
                    raise RuntimeError(
                        f"docker run failed for benchmark {variant.name}: "
                        f"returncode={start_result.returncode}; "
                        f"stderr={start_result.stderr[-1000:] or '<empty>'}"
                    )
            started = True
            client = self.client_factory(
                f"http://127.0.0.1:{port}", self.profile.model.model_id
            )
            wait_until_ready(client, timeout_seconds=900, interval_seconds=2)
            if self.local_backend:
                command_record["inspected_server_argv"] = list(server_argv)
            else:
                command_record["inspected_server_argv"] = list(
                    inspect_container_command(
                        name,
                        variant,
                        self.command_runner,
                        calculate_kv_scales=self.profile.calculate_kv_scales,
                    )
                )
            atomic_write_json(command_path, command_record)
            client.complete("Reply with OK.", 1)
            captured_log = (
                server_process.log_text()
                if self.local_backend and server_process
                else self._logs(name)
            )
            validate_server_log(
                captured_log, variant, num_layers=self.profile.model.num_layers
            )
            capacity = parse_capacity_tokens(captured_log)
            if not cached_matrix_complete:
                telemetry.start()
                telemetry_started = True

            for case, argv in zip(matrix, bench_commands):
                raw_path = result_directory / (
                    f"in{case.input_length}-out{case.output_length}-rep{case.repeat}.json"
                )
                reusable = cached_matrix_complete
                if reusable and raw_path.is_file():
                    try:
                        cached_payload = read_json(raw_path)
                        reusable = isinstance(cached_payload, Mapping)
                        if reusable:
                            extract_performance_metrics(cached_payload)
                    except (OSError, TypeError, ValueError):
                        reusable = False
                if not reusable:
                    bench_name = benchmark_container_name(
                        raw_path.relative_to(self.project_root)
                    )
                    if not self.local_backend:
                        safe_remove_stale_container(bench_name, self.command_runner)
                    for attempt in range(MAX_TIMEOUT_RESTARTS + 1):
                        result = self.command_runner(tuple(argv), timeout=3600)
                        if result.ok:
                            break
                        if result.returncode == 124 and not self.local_backend:
                            safe_cleanup_owned_container(
                                bench_name, self.command_runner
                            )
                        if (
                            result.returncode != 124
                            or attempt >= MAX_TIMEOUT_RESTARTS
                        ):
                            raise RuntimeError(
                                "vllm bench serve failed for "
                                f"input={case.input_length}, "
                                f"output={case.output_length}, "
                                f"repeat={case.repeat}: {result.stderr[-1000:]}"
                            )
                if not raw_path.is_file():
                    raise ValueError(f"benchmark result was not created: {raw_path}")
                payload = read_json(raw_path)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"benchmark result is not a JSON object: {raw_path}")
                scenarios.append(
                    {
                        **asdict(case),
                        "raw_result": raw_path.relative_to(self.project_root).as_posix(),
                        "sha256": sha256_file(raw_path),
                        "metrics": extract_performance_metrics(payload),
                    }
                )
        finally:
            primary_error = sys.exc_info()[1]
            finalization_errors: list[tuple[str, BaseException]] = []
            if telemetry_started:
                try:
                    telemetry.stop()
                except BaseException as error:
                    finalization_errors.append(("telemetry", error))
            if started:
                try:
                    captured_log = (
                        server_process.log_text()
                        if self.local_backend and server_process
                        else self._logs(name)
                    ) or captured_log
                    atomic_write_text(log_path, captured_log)
                except BaseException as error:
                    finalization_errors.append(("server-log", error))
            if cleanup_required:
                try:
                    if self.local_backend:
                        assert server_process is not None
                        server_process.stop()
                    else:
                        safe_cleanup_owned_container(name, self.command_runner)
                except BaseException as error:
                    finalization_errors.append(("server-cleanup", error))
            cleanup_errors = [
                error
                for stage, error in finalization_errors
                if stage in {"container-cleanup", "server-cleanup"}
            ]
            if primary_error is not None and cleanup_errors:
                raise RuntimeError(
                    f"benchmark failed: {primary_error}; exact owned server cleanup "
                    f"also failed: {cleanup_errors[0]}"
                ) from primary_error
            if primary_error is None and cleanup_errors:
                earlier_errors = [
                    error
                    for stage, error in finalization_errors
                    if stage not in {"container-cleanup", "server-cleanup"}
                ]
                if earlier_errors:
                    raise RuntimeError(
                        f"benchmark finalization failed: {earlier_errors[0]}; "
                        "exact owned server cleanup also failed: "
                        f"{cleanup_errors[0]}"
                    ) from earlier_errors[0]
                raise cleanup_errors[0]
            if primary_error is None and finalization_errors:
                raise finalization_errors[0][1]

        if capacity is None:
            raise ValueError("benchmark server did not expose KV capacity")
        if not telemetry_is_usable(telemetry_path):
            raise ValueError("nvidia-smi dmon telemetry was not captured")
        if len(scenarios) != len(matrix):
            raise ValueError("benchmark matrix did not produce every registered scenario")
        if not cached_matrix_complete:
            atomic_write_json(
                matrix_state_path,
                {
                    "schema_version": 1,
                    "complete": True,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "variant": {
                        "name": variant.name,
                        "kv_dtype": variant.kv_dtype,
                        "skip_layers": list(variant.skip_layers),
                    },
                    "runtime_id": runtime_identity(self.lock),
                    "backend": str(self.lock.get("backend", "docker")),
                    "model_revision": model_revision,
                    "telemetry": {
                        "path": telemetry_path.relative_to(
                            self.project_root
                        ).as_posix(),
                        "sha256": sha256_file(telemetry_path),
                    },
                    "scenarios": scenarios,
                },
            )
        if not self._matrix_state_is_complete(
            matrix_state_path,
            run_id=run_id,
            variant=variant,
            telemetry_path=telemetry_path,
            result_directory=result_directory,
        ):
            raise ValueError("benchmark matrix state failed its integrity check")
        values_by_metric: dict[str, list[float]] = {}
        for scenario in scenarios:
            for key, value in scenario["metrics"].items():
                values_by_metric.setdefault(key, []).append(float(value))
        aggregate = {
            key: statistics.fmean(values)
            for key, values in sorted(values_by_metric.items())
        }
        scenario_groups = aggregate_scenario_groups(scenarios)
        validation = capacity_validation(
            self.profile, variant, capacity, captured_log
        )
        atomic_write_json(
            summary_path,
            {
                "schema_version": 3,
                "complete": True,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "variant": {
                    "name": variant.name,
                    "kv_dtype": variant.kv_dtype,
                    "skip_layers": list(variant.skip_layers),
                },
                "runtime_id": runtime_identity(self.lock),
                "backend": str(self.lock.get("backend", "docker")),
                "model_revision": model_revision,
                "capacity": asdict(capacity),
                "aggregate": aggregate,
                "aggregate_semantics": (
                    "unweighted descriptive mean across heterogeneous workloads; "
                    "do not use for latency comparison"
                ),
                "scenario_groups": scenario_groups,
                "capacity_validation": validation,
                "telemetry": {
                    "path": telemetry_path.relative_to(self.project_root).as_posix(),
                    "sha256": sha256_file(telemetry_path),
                    "argv": list(telemetry.command),
                },
                "server_log": {
                    "path": log_path.relative_to(self.project_root).as_posix(),
                    "sha256": sha256_file(log_path),
                },
                "command_record": {
                    "path": command_path.relative_to(self.project_root).as_posix(),
                    "sha256": sha256_file(command_path),
                },
                "matrix_state": {
                    "path": matrix_state_path.relative_to(
                        self.project_root
                    ).as_posix(),
                    "sha256": sha256_file(matrix_state_path),
                },
                "scenarios": scenarios,
            },
        )
        return summary_path
