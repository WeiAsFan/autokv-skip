"""Resumable execution of one vLLM KV-cache variant at a time."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from autokv.client import VllmClient, VllmHttpError, wait_until_ready
from autokv.commands import (
    CommandResult,
    container_name,
    run_command,
    server_command,
)
from autokv.config import Profile
from autokv.io import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    ensure_within,
    read_json,
    read_jsonl,
    sha256_file,
)
from autokv.niah import NiahCase, expected_answer, fit_prompt
from autokv.scoring import answer_nll_from_echo, quality_score, score_generation
from autokv.selection import Variant, canonical_config_id


Runner = Callable[..., CommandResult]
ClientFactory = Callable[[str, str], Any]
QUALITY_MODES = frozenset(("auto", "nll", "edit_distance"))
MAX_TIMEOUT_RESTARTS = 2


def archive_exact_artifacts(
    project_root: Path,
    run_id: str,
    phase: str,
    config_id: str,
    paths: Sequence[Path],
) -> Path | None:
    """Move one explicitly forced configuration to a recoverable evidence folder."""
    for label, value in (("run_id", run_id), ("phase", phase)):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"{label} contains unsafe characters")
    if not re.fullmatch(r"[0-9a-f]{12}", config_id):
        raise ValueError("force config ID must be 12 lowercase hex characters")
    existing = [ensure_within(project_root, path) for path in paths if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = ensure_within(
        project_root,
        project_root
        / "runs"
        / run_id
        / "_superseded"
        / f"{stamp}-{phase}-{config_id}-{time.time_ns()}",
    )
    destination.mkdir(parents=True, exist_ok=False)
    for source in existing:
        target = ensure_within(project_root, destination / source.name)
        source.replace(target)
    return destination


def canonical_run_id(
    profile_hash: str,
    image_digest: str,
    model_revision: str,
    dataset_hash: str,
    source_tree_hash: str,
) -> str:
    raw = "\n".join(
        (
            profile_hash,
            image_digest,
            model_revision,
            dataset_hash,
            source_tree_hash,
        )
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def mark_complete(
    state_path: Path,
    artifact: Path,
    expected_rows: int,
    *,
    evidence: Mapping[str, Path] | None = None,
) -> None:
    rows = read_jsonl(artifact)
    if len(rows) != expected_rows:
        raise ValueError(
            f"cannot mark complete: expected {expected_rows} rows, observed {len(rows)}"
        )
    evidence_records: dict[str, dict[str, str]] = {}
    for key, path in sorted((evidence or {}).items()):
        if not re.fullmatch(r"[a-z0-9_]+", key):
            raise ValueError(f"invalid completion evidence key: {key}")
        if path.parent.resolve() != state_path.parent.resolve() or not path.is_file():
            raise ValueError(f"completion evidence is missing or misplaced: {path}")
        evidence_records[key] = {
            "path": path.name,
            "sha256": sha256_file(path),
        }
    atomic_write_json(
        state_path,
        {
            "schema_version": 2,
            "complete": True,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "rows": len(rows),
            "evidence": evidence_records,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def is_complete(
    state_path: Path,
    artifact: Path,
    expected_rows: int,
    *,
    evidence: Mapping[str, Path] | None = None,
) -> bool:
    try:
        state = read_json(state_path)
        if not isinstance(state, Mapping) or state.get("complete") is not True:
            return False
        if (
            state.get("artifact") != artifact.name
            or state.get("rows") != expected_rows
            or not artifact.is_file()
        ):
            return False
        if state.get("artifact_sha256") != sha256_file(artifact):
            return False
        if len(read_jsonl(artifact)) != expected_rows:
            return False
        observed_evidence = state.get("evidence", {})
        if not isinstance(observed_evidence, Mapping):
            return False
        if evidence is not None and (
            state.get("schema_version") != 2
            or set(observed_evidence) != set(evidence)
        ):
            return False
        expected_evidence = evidence or {
            str(key): state_path.parent / str(record.get("path", ""))
            for key, record in observed_evidence.items()
            if isinstance(record, Mapping)
        }
        if set(expected_evidence) != set(observed_evidence):
            return False
        for key, path in expected_evidence.items():
            record = observed_evidence.get(key)
            if (
                not isinstance(record, Mapping)
                or path.parent.resolve() != state_path.parent.resolve()
                or record.get("path") != path.name
                or not path.is_file()
                or record.get("sha256") != sha256_file(path)
            ):
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def validate_container_command(output: str, variant: Variant) -> None:
    try:
        command = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("container inspect command is not valid JSON") from exc
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError("container inspect command must be a string array")

    def exact_value(flag: str) -> str:
        indices = [index for index, item in enumerate(command) if item == flag]
        if len(indices) != 1 or indices[0] + 1 >= len(command):
            raise ValueError(f"container command must contain exactly one {flag}")
        return command[indices[0] + 1]

    if exact_value("--attention-backend") != "FLASHINFER":
        raise ValueError("container command does not use FLASHINFER")
    if exact_value("--kv-cache-dtype") != variant.kv_dtype:
        raise ValueError("container command KV dtype differs from the variant")
    scale_count = command.count("--calculate-kv-scales")
    expected_scale_count = int(variant.kv_dtype == "fp8_e4m3")
    if scale_count != expected_scale_count:
        raise ValueError("container command has the wrong KV scale mode")

    flag = "--kv-cache-dtype-skip-layers"
    if flag in command:
        if command.count(flag) != 1:
            raise ValueError("container command repeats skip layers")
        start = command.index(flag) + 1
        end = start
        while end < len(command) and not command[end].startswith("--"):
            end += 1
        try:
            observed_layers = tuple(int(value) for value in command[start:end])
        except ValueError as exc:
            raise ValueError("container command has non-integer skip layers") from exc
    else:
        observed_layers = ()
    if observed_layers != variant.skip_layers:
        raise ValueError(
            "container command skip layers differ from the requested variant"
        )


def inspect_container_command(
    name: str, variant: Variant, runner: Runner = run_command
) -> tuple[str, ...]:
    result = runner(
        ("docker", "inspect", "--format", "{{json .Config.Cmd}}", name),
        timeout=30,
    )
    if not result.ok:
        raise RuntimeError(f"cannot inspect actual container command: {name}")
    validate_container_command(result.stdout, variant)
    parsed = json.loads(result.stdout)
    return tuple(parsed)


def validate_server_log(log: str, variant: Variant, num_layers: int = 32) -> None:
    upper = log.upper()
    if not re.search(
        r"\bUSING\s+(?:ATTENTIONBACKENDENUM\.)?FLASHINFER\s+BACKEND\b", upper
    ):
        raise ValueError("server log does not confirm the active FLASHINFER backend")
    if not re.search(r"GPU KV CACHE SIZE:\s*[\d,]+\s*TOKENS", upper):
        raise ValueError("server log does not contain GPU KV cache token capacity")
    fp8_message = "USING FP8_E4M3 DATA TYPE TO STORE KV CACHE"
    if variant.kv_dtype == "fp8_e4m3" and fp8_message not in upper:
        raise ValueError("server log does not confirm exact FP8 E4M3 KV cache")
    if variant.kv_dtype == "bfloat16" and fp8_message in upper:
        raise ValueError("BF16 server log unexpectedly confirms FP8 E4M3 KV cache")
    if variant.kv_dtype == "bfloat16" and not re.search(
        r"\bKV_CACHE_DTYPE\s*=\s*['\"]?BFLOAT16\b['\"]?", upper
    ):
        raise ValueError("server log does not positively confirm BF16 KV cache")
    if not variant.skip_layers:
        return
    observed: dict[int, str] = {}
    pattern = re.compile(
        r"Layer\s+\S*layers\.(\d+)\S*:\s*kv_cache_dtype=([^,\s]+)",
        flags=re.IGNORECASE,
    )
    for layer_text, dtype in pattern.findall(log):
        layer = int(layer_text)
        normalized_dtype = dtype.lower()
        previous = observed.get(layer)
        if previous is not None and previous != normalized_dtype:
            raise ValueError(f"conflicting effective KV dtype logs for layer {layer}")
        observed[layer] = normalized_dtype
    expected_layers = set(range(num_layers))
    if set(observed) != expected_layers:
        raise ValueError(
            "server log does not expose the effective KV dtype for every layer"
        )
    skipped = set(variant.skip_layers)
    for layer in range(num_layers):
        expected = "auto" if layer in skipped else "fp8_e4m3"
        if observed[layer] != expected:
            raise ValueError(
                f"layer {layer} effective KV dtype is {observed[layer]}, expected {expected}"
            )


def safe_stop_container(name: str, runner: Runner = run_command) -> None:
    inspect_argv = (
        "docker",
        "inspect",
        "--format",
        '{{ index .Config.Labels "io.autokv.project" }}',
        name,
    )
    inspected = runner(inspect_argv, timeout=30)
    if not inspected.ok:
        raise RuntimeError(f"cannot verify container ownership before stop: {name}")
    if inspected.stdout.strip() != "autokv-skip":
        raise RuntimeError(f"refusing to stop foreign container: {name}")
    stopped = runner(("docker", "stop", "--time", "30", name), timeout=60)
    if not stopped.ok:
        raise RuntimeError(f"failed to stop AutoKV-Skip container: {name}")
    removed = runner(("docker", "rm", name), timeout=30)
    if not removed.ok:
        raise RuntimeError(f"failed to remove stopped AutoKV-Skip container: {name}")


def _owned_container_running(name: str, runner: Runner) -> bool | None:
    inspect_argv = (
        "docker",
        "inspect",
        "--format",
        '{{ index .Config.Labels "io.autokv.project" }}|{{ .State.Running }}',
        name,
    )
    inspected = runner(inspect_argv, timeout=30)
    if not inspected.ok:
        combined = f"{inspected.stdout}\n{inspected.stderr}"
        if re.search(r"no such (?:object|container)", combined, re.IGNORECASE):
            return None
        raise RuntimeError(f"cannot inspect existing container state: {name}")
    fields = inspected.stdout.strip().split("|")
    if len(fields) != 2:
        raise RuntimeError(f"cannot parse existing container ownership/state: {name}")
    project, running = fields
    if project != "autokv-skip":
        raise RuntimeError(f"refusing to remove foreign container: {name}")
    if running not in {"true", "false"}:
        raise RuntimeError(f"cannot determine existing container state: {name}")
    return running == "true"


def safe_remove_stale_container(name: str, runner: Runner = run_command) -> None:
    """Remove only an exited container that is provably owned by this project."""
    running = _owned_container_running(name, runner)
    if running is None:
        return
    if running:
        raise RuntimeError(
            f"AutoKV-Skip container is already running: {name}; "
            "stop the concurrent/stale run explicitly before retrying"
        )
    removed = runner(("docker", "rm", name), timeout=30)
    if not removed.ok:
        raise RuntimeError(f"failed to remove stale AutoKV-Skip container: {name}")


def safe_cleanup_owned_container(name: str, runner: Runner = run_command) -> None:
    """Stop/remove an exact owned container after this process timed it out."""
    running = _owned_container_running(name, runner)
    if running is None:
        return
    if running:
        stopped = runner(("docker", "stop", "--time", "30", name), timeout=60)
        if not stopped.ok:
            raise RuntimeError(f"failed to stop timed-out AutoKV container: {name}")
        running = _owned_container_running(name, runner)
        if running is None:
            return
        if running:
            raise RuntimeError(f"timed-out AutoKV container is still running: {name}")
    removed = runner(("docker", "rm", name), timeout=30)
    if not removed.ok:
        raise RuntimeError(f"failed to remove timed-out AutoKV container: {name}")


def _completion_text(response: Mapping[str, Any]) -> str:
    try:
        text = response["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("completion response has no choices[0].text") from exc
    if not isinstance(text, str):
        raise ValueError("completion text is not a string")
    return text


def validate_experiment_rows(
    rows: Sequence[Mapping[str, Any]],
    artifact: Path,
    variant: Variant,
    cases: Sequence[NiahCase],
    *,
    run_id: str,
    image_digest: str,
    model_revision: str,
    backend: str,
    quality_mode: str,
) -> None:
    """Validate row schema plus every immutable experiment context field."""
    if quality_mode not in QUALITY_MODES:
        raise ValueError(f"quality_mode must be one of {sorted(QUALITY_MODES)}")
    if rows:
        observed_modes = {row.get("quality_mode") for row in rows}
        if not observed_modes.issubset({"nll", "edit_distance"}):
            raise ValueError(f"artifact has missing/invalid quality mode: {artifact}")
        if len(observed_modes) > 1:
            raise ValueError(f"artifact mixes quality modes: {artifact}")
        if quality_mode != "auto" and observed_modes != {quality_mode}:
            raise ValueError(
                f"artifact quality mode differs from locked {quality_mode}: {artifact}"
            )

    case_by_id = {case.sample_id: case for case in cases}
    expected_context = {
        "schema_version": 1,
        "run_id": run_id,
        "config_id": canonical_config_id(variant),
        "image_digest": image_digest,
        "model_revision": model_revision,
        "backend": backend,
        "kv_dtype": variant.kv_dtype,
        "skip_layers": list(variant.skip_layers),
    }
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in case_by_id:
            raise ValueError(f"artifact contains unexpected sample ID: {artifact}")
        if sample_id in seen:
            raise ValueError(f"partial artifact has duplicate sample IDs: {artifact}")
        seen.add(sample_id)
        for key, expected in expected_context.items():
            if row.get(key) != expected:
                raise ValueError(f"artifact context mismatch for {key}: {artifact}")
        case = case_by_id[sample_id]
        if row.get("seed") != case.seed or row.get("expected") != expected_answer(
            case.code
        ):
            raise ValueError(f"artifact context mismatch for sample: {artifact}")
        if not isinstance(row.get("output_text"), str):
            raise ValueError(f"artifact row has invalid output_text: {artifact}")
        for key in ("prompt_tokens", "edit_distance"):
            value = row.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"artifact row has invalid {key}: {artifact}")
        output_tokens = row.get("output_tokens")
        if output_tokens is not None and (
            not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
        ):
            raise ValueError(f"artifact row has invalid output_tokens: {artifact}")
        for key in ("needle_depth", "exact_match", "quality_score", "e2e_ms"):
            value = row.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"artifact row has invalid {key}: {artifact}")
        if not 0.0 <= float(row["needle_depth"]) <= 1.0:
            raise ValueError(f"artifact row has invalid needle_depth: {artifact}")
        if float(row["exact_match"]) not in {0.0, 1.0}:
            raise ValueError(f"artifact row has invalid exact_match: {artifact}")
        answer_nll = row.get("answer_nll")
        if answer_nll is not None and (
            not isinstance(answer_nll, (int, float))
            or isinstance(answer_nll, bool)
            or not math.isfinite(float(answer_nll))
        ):
            raise ValueError(f"artifact row has invalid answer_nll: {artifact}")
        if "ttft_ms" not in row or row["ttft_ms"] is not None:
            raise ValueError(
                f"artifact ttft_ms must be explicit null for non-streaming quality: {artifact}"
            )
        if not isinstance(row.get("timestamp_utc"), str):
            raise ValueError(f"artifact row has invalid timestamp_utc: {artifact}")


class ExperimentRunner:
    def __init__(
        self,
        profile: Profile,
        project_root: Path,
        lock: Mapping[str, Any],
        *,
        command_runner: Runner = run_command,
        client_factory: ClientFactory | None = None,
        quality_mode: str = "auto",
        force_config_id: str | None = None,
    ) -> None:
        self.profile = profile
        self.project_root = project_root.resolve()
        self.lock = lock
        self.command_runner = command_runner
        self.client_factory = client_factory or (
            lambda base_url, model_id: VllmClient(base_url, model_id)
        )
        if quality_mode not in QUALITY_MODES:
            raise ValueError(f"quality_mode must be one of {sorted(QUALITY_MODES)}")
        self.quality_mode = quality_mode
        if force_config_id is not None and not re.fullmatch(
            r"[0-9a-f]{12}", force_config_id
        ):
            raise ValueError("force config ID must be 12 lowercase hex characters")
        self.force_config_id = force_config_id
        self._forced_keys: set[tuple[str, str, str]] = set()
        for key in ("image_ref", "image_digest", "model_revision"):
            if not isinstance(lock.get(key), str) or not lock[key]:
                raise ValueError(f"environment lock is missing {key}")

    @property
    def force_applied(self) -> bool:
        return bool(self._forced_keys)

    def _paths(
        self, run_id: str, phase: str, variant: Variant
    ) -> tuple[Path, Path, Path, Path]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("run_id contains unsafe characters")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", phase):
            raise ValueError("phase contains unsafe characters")
        stem = f"{variant.name}-{canonical_config_id(variant)}"
        directory = ensure_within(
            self.project_root,
            self.project_root / "runs" / run_id / phase,
        )
        artifact = directory / f"{stem}.jsonl"
        return (
            artifact,
            artifact.with_suffix(".state.json"),
            artifact.with_suffix(".server.log"),
            artifact.with_suffix(".command.json"),
        )

    def _logs(self, name: str) -> str:
        result = self.command_runner(("docker", "logs", name), timeout=60)
        return result.stdout + result.stderr

    def _validate_existing_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        artifact: Path,
        variant: Variant,
        cases: Sequence[NiahCase],
        run_id: str,
    ) -> None:
        validate_experiment_rows(
            rows,
            artifact,
            variant,
            cases,
            run_id=run_id,
            image_digest=str(self.lock["image_digest"]),
            model_revision=str(self.lock["model_revision"]),
            backend=self.profile.model.attention_backend,
            quality_mode=self.quality_mode,
        )

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        return isinstance(exc, VllmHttpError) and exc.status is None and any(
            token in exc.body.lower() for token in ("timeout", "timed out")
        )

    def run_variant(
        self,
        variant: Variant,
        cases: Sequence[NiahCase],
        *,
        phase: str,
        run_id: str,
        port: int,
    ) -> Path:
        for restart in range(MAX_TIMEOUT_RESTARTS + 1):
            try:
                return self._run_variant_once(
                    variant,
                    cases,
                    phase=phase,
                    run_id=run_id,
                    port=port,
                )
            except (TimeoutError, VllmHttpError) as exc:
                if not self._is_timeout(exc) or restart >= MAX_TIMEOUT_RESTARTS:
                    raise
        raise AssertionError("timeout restart loop exhausted without returning")

    def _run_variant_once(
        self,
        variant: Variant,
        cases: Sequence[NiahCase],
        *,
        phase: str,
        run_id: str,
        port: int,
    ) -> Path:
        if not cases:
            raise ValueError("at least one NIAH case is required")
        variant.validate_for_model(self.profile.model.num_layers)
        artifact, state_path, log_path, command_path = self._paths(
            run_id, phase, variant
        )
        completion_evidence = {
            "server_log": log_path,
            "command_record": command_path,
        }
        config_id = canonical_config_id(variant)
        force_key = (run_id, phase, config_id)
        if self.force_config_id == config_id and force_key not in self._forced_keys:
            archive_exact_artifacts(
                self.project_root,
                run_id,
                phase,
                config_id,
                (artifact, state_path, log_path, command_path),
            )
            self._forced_keys.add(force_key)
        expected_rows = len(cases)
        complete = is_complete(
            state_path,
            artifact,
            expected_rows,
            evidence=completion_evidence,
        )
        if state_path.exists() and not complete:
            raise ValueError(
                f"completed state integrity mismatch; preserve evidence and use "
                f"--force {canonical_config_id(variant)}: {state_path}"
            )
        if complete:
            self._validate_existing_rows(
                read_jsonl(artifact), artifact, variant, cases, run_id
            )
            return artifact

        existing_rows = read_jsonl(artifact) if artifact.exists() else []
        self._validate_existing_rows(
            existing_rows, artifact, variant, cases, run_id
        )
        active_quality_mode = self.quality_mode
        if active_quality_mode == "auto" and existing_rows:
            active_quality_mode = str(existing_rows[0]["quality_mode"])
        completed_ids = {row.get("sample_id") for row in existing_rows}
        expected_ids = {case.sample_id for case in cases}
        if not completed_ids.issubset(expected_ids):
            raise ValueError(f"partial artifact contains unexpected sample IDs: {artifact}")

        image_ref = str(self.lock["image_ref"])
        model_revision = str(self.lock["model_revision"])
        argv = server_command(
            self.profile,
            image_ref,
            variant,
            self.project_root,
            port,
            run_id,
            model_revision,
        )
        command_record = {
            "argv": list(argv),
            "variant": {
                "name": variant.name,
                "kv_dtype": variant.kv_dtype,
                "skip_layers": list(variant.skip_layers),
            },
            "image_ref": image_ref,
            "model_revision": model_revision,
            "quality_mode": self.quality_mode,
        }
        atomic_write_json(command_path, command_record)

        name = container_name(run_id, variant)
        started = False
        cleanup_required = False
        completed = False
        captured_log = ""
        try:
            safe_remove_stale_container(name, self.command_runner)
            cleanup_required = True
            start_result = self.command_runner(argv, timeout=120)
            if not start_result.ok:
                raise RuntimeError(
                    f"docker run failed for {variant.name}: "
                    f"returncode={start_result.returncode}; "
                    f"stderr={start_result.stderr[-1000:] or '<empty>'}"
                )
            started = True
            client = self.client_factory(
                f"http://127.0.0.1:{port}", self.profile.model.model_id
            )
            wait_until_ready(client, timeout_seconds=900, interval_seconds=2)
            command_record["inspected_argv"] = list(
                inspect_container_command(name, variant, self.command_runner)
            )
            atomic_write_json(command_path, command_record)
            captured_log = self._logs(name)
            validate_server_log(
                captured_log, variant, num_layers=self.profile.model.num_layers
            )

            for case in cases:
                if case.sample_id in completed_ids:
                    continue
                materialized = fit_prompt(case, client.tokenize)
                started_at = time.monotonic()
                response = client.complete(
                    materialized.prompt, self.profile.quality.max_tokens
                )
                elapsed_ms = (time.monotonic() - started_at) * 1000.0
                output = _completion_text(response)
                expected = expected_answer(case.code)
                generation = score_generation(output, expected)
                answer_prefix = materialized.prompt + "\n\nAnswer: "
                if active_quality_mode == "edit_distance":
                    answer_nll = None
                    quality_mode = "edit_distance"
                else:
                    try:
                        echo = client.echo_logprobs(answer_prefix + expected)
                        answer_nll = answer_nll_from_echo(echo, len(answer_prefix))
                    except VllmHttpError as exc:
                        if active_quality_mode == "nll" or exc.status not in {
                            400,
                            404,
                            405,
                            422,
                        }:
                            raise
                        answer_nll = None
                    except ValueError:
                        if active_quality_mode == "nll":
                            raise
                        answer_nll = None
                    if active_quality_mode == "nll" and answer_nll is None:
                        raise ValueError(
                            "NLL quality mode was locked by smoke, but echo logprobs "
                            "were missing; stop instead of mixing scoring functions"
                        )
                    quality_mode = "nll" if answer_nll is not None else "edit_distance"
                    if active_quality_mode == "auto":
                        active_quality_mode = quality_mode
                q_score = quality_score(
                    generation.exact_match,
                    answer_nll,
                    generation.edit_distance,
                    len(generation.normalized_expected),
                )
                usage = response.get("usage", {})
                output_tokens = (
                    usage.get("completion_tokens")
                    if isinstance(usage, Mapping)
                    else None
                )
                append_jsonl(
                    artifact,
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "config_id": canonical_config_id(variant),
                        "image_digest": self.lock["image_digest"],
                        "model_revision": model_revision,
                        "backend": self.profile.model.attention_backend,
                        "kv_dtype": variant.kv_dtype,
                        "skip_layers": list(variant.skip_layers),
                        "seed": case.seed,
                        "sample_id": case.sample_id,
                        "prompt_tokens": materialized.token_count,
                        "output_tokens": output_tokens,
                        "needle_depth": materialized.needle_depth,
                        "expected": expected,
                        "output_text": output,
                        "exact_match": generation.exact_match,
                        "edit_distance": generation.edit_distance,
                        "answer_nll": answer_nll,
                        "quality_mode": quality_mode,
                        "quality_score": q_score,
                        "ttft_ms": None,
                        "e2e_ms": elapsed_ms,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            completed = True
        finally:
            primary_error = sys.exc_info()[1]
            finalization_errors: list[tuple[str, BaseException]] = []
            if started:
                try:
                    captured_log = self._logs(name) or captured_log
                    atomic_write_text(log_path, captured_log)
                except BaseException as error:
                    finalization_errors.append(("server-log", error))
            if cleanup_required:
                try:
                    safe_cleanup_owned_container(name, self.command_runner)
                except BaseException as error:
                    finalization_errors.append(("container-cleanup", error))
            cleanup_errors = [
                error
                for stage, error in finalization_errors
                if stage == "container-cleanup"
            ]
            if primary_error is not None and cleanup_errors:
                raise RuntimeError(
                    f"experiment failed: {primary_error}; exact owned server cleanup "
                    f"also failed: {cleanup_errors[0]}"
                ) from primary_error
            if primary_error is None and cleanup_errors:
                earlier_errors = [
                    error
                    for stage, error in finalization_errors
                    if stage != "container-cleanup"
                ]
                if earlier_errors:
                    raise RuntimeError(
                        f"experiment finalization failed: {earlier_errors[0]}; "
                        "exact owned server cleanup also failed: "
                        f"{cleanup_errors[0]}"
                    ) from earlier_errors[0]
                raise cleanup_errors[0]
            if primary_error is None and finalization_errors:
                raise finalization_errors[0][1]

        if completed:
            mark_complete(
                state_path,
                artifact,
                expected_rows,
                evidence=completion_evidence,
            )
        return artifact
