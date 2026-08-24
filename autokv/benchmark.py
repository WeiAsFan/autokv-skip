"""vLLM capacity parsing and resumable service benchmarking."""

from __future__ import annotations

import itertools
import re
import statistics
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from autokv.client import VllmClient, wait_until_ready
from autokv.commands import (
    CommandResult,
    bench_command,
    container_name,
    run_command,
    server_command,
)
from autokv.config import Profile
from autokv.experiment import (
    MAX_TIMEOUT_RESTARTS,
    safe_remove_stale_container,
    safe_stop_container,
    validate_server_log,
)
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
    if "request_throughput" not in metrics:
        raise ValueError("vLLM benchmark result has no request_throughput")
    return metrics


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
    ) -> None:
        self.profile = profile
        self.project_root = project_root.resolve()
        self.lock = lock
        self.command_runner = command_runner
        self.client_factory = client_factory or (
            lambda base_url, model_id: VllmClient(base_url, model_id)
        )
        for key in ("image_ref", "image_digest", "model_revision"):
            if not isinstance(lock.get(key), str) or not lock[key]:
                raise ValueError(f"environment lock is missing {key}")

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

    def _is_complete(self, summary_path: Path, expected_scenarios: int) -> bool:
        try:
            summary = read_json(summary_path)
            if not isinstance(summary, Mapping) or summary.get("complete") is not True:
                return False
            scenarios = summary.get("scenarios")
            if not isinstance(scenarios, list) or len(scenarios) != expected_scenarios:
                return False
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
            return True
        except (OSError, TypeError, ValueError):
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
        if self._is_complete(summary_path, len(matrix)):
            return summary_path
        result_directory.mkdir(parents=True, exist_ok=True)

        image_ref = str(self.lock["image_ref"])
        model_revision = str(self.lock["model_revision"])
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
            bench_commands.append(
                list(
                    bench_command(
                        self.profile,
                        image_ref,
                        self.project_root,
                        port,
                        case.input_length,
                        case.output_length,
                        raw_path.relative_to(self.project_root),
                        model_revision,
                    )
                )
            )
        atomic_write_json(
            command_path,
            {
                "server_argv": list(server_argv),
                "benchmark_argv": bench_commands,
                "image_ref": image_ref,
                "model_revision": model_revision,
                "variant": {
                    "name": variant.name,
                    "kv_dtype": variant.kv_dtype,
                    "skip_layers": list(variant.skip_layers),
                },
            },
        )

        name = container_name(run_id, variant)
        started = False
        captured_log = ""
        scenarios: list[dict[str, Any]] = []
        capacity: Capacity | None = None
        try:
            safe_remove_stale_container(name, self.command_runner)
            start_result = self.command_runner(server_argv, timeout=120)
            if not start_result.ok:
                raise RuntimeError(
                    f"docker run failed for benchmark {variant.name}: "
                    f"{start_result.stderr[-1000:]}"
                )
            started = True
            client = self.client_factory(
                f"http://127.0.0.1:{port}", self.profile.model.model_id
            )
            wait_until_ready(client, timeout_seconds=900, interval_seconds=2)
            client.complete("Reply with OK.", 1)
            captured_log = self._logs(name)
            validate_server_log(captured_log, variant)
            capacity = parse_capacity_tokens(captured_log)

            for case, argv in zip(matrix, bench_commands):
                raw_path = result_directory / (
                    f"in{case.input_length}-out{case.output_length}-rep{case.repeat}.json"
                )
                reusable = False
                if raw_path.is_file():
                    try:
                        cached_payload = read_json(raw_path)
                        reusable = isinstance(cached_payload, Mapping)
                        if reusable:
                            extract_performance_metrics(cached_payload)
                    except (OSError, TypeError, ValueError):
                        reusable = False
                if not reusable:
                    for attempt in range(MAX_TIMEOUT_RESTARTS + 1):
                        result = self.command_runner(tuple(argv), timeout=3600)
                        if result.ok:
                            break
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
            if started:
                try:
                    captured_log = self._logs(name) or captured_log
                    atomic_write_text(log_path, captured_log)
                finally:
                    safe_stop_container(name, self.command_runner)

        if capacity is None:
            raise ValueError("benchmark server did not expose KV capacity")
        values_by_metric: dict[str, list[float]] = {}
        for scenario in scenarios:
            for key, value in scenario["metrics"].items():
                values_by_metric.setdefault(key, []).append(float(value))
        aggregate = {
            key: statistics.fmean(values)
            for key, values in sorted(values_by_metric.items())
        }
        atomic_write_json(
            summary_path,
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
                "image_digest": self.lock["image_digest"],
                "model_revision": model_revision,
                "capacity": asdict(capacity),
                "aggregate": aggregate,
                "scenarios": scenarios,
            },
        )
        return summary_path
