"""Resumable execution of one vLLM KV-cache variant at a time."""

from __future__ import annotations

import hashlib
import re
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


def canonical_run_id(
    profile_hash: str,
    image_digest: str,
    model_revision: str,
    dataset_hash: str,
) -> str:
    raw = "\n".join(
        (profile_hash, image_digest, model_revision, dataset_hash)
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def mark_complete(state_path: Path, artifact: Path, expected_rows: int) -> None:
    rows = read_jsonl(artifact)
    if len(rows) != expected_rows:
        raise ValueError(
            f"cannot mark complete: expected {expected_rows} rows, observed {len(rows)}"
        )
    atomic_write_json(
        state_path,
        {
            "complete": True,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "rows": len(rows),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def is_complete(state_path: Path, artifact: Path, expected_rows: int) -> bool:
    try:
        state = read_json(state_path)
        if not isinstance(state, Mapping) or state.get("complete") is not True:
            return False
        if state.get("rows") != expected_rows or not artifact.is_file():
            return False
        if state.get("artifact_sha256") != sha256_file(artifact):
            return False
        return len(read_jsonl(artifact)) == expected_rows
    except (OSError, ValueError, TypeError):
        return False


def validate_server_log(log: str, variant: Variant) -> None:
    upper = log.upper()
    if "FLASHINFER" not in upper:
        raise ValueError("server log does not confirm FLASHINFER")
    if not re.search(r"GPU KV CACHE SIZE:\s*[\d,]+\s*TOKENS", upper):
        raise ValueError("server log does not contain GPU KV cache token capacity")
    if variant.kv_dtype == "fp8_e4m3":
        if "FP8" not in upper or "KV CACHE" not in upper:
            raise ValueError("server log does not confirm FP8 KV cache")


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


def _completion_text(response: Mapping[str, Any]) -> str:
    try:
        text = response["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("completion response has no choices[0].text") from exc
    if not isinstance(text, str):
        raise ValueError("completion text is not a string")
    return text


class ExperimentRunner:
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

    def run_variant(
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
        expected_rows = len(cases)
        if is_complete(state_path, artifact, expected_rows):
            return artifact

        existing_rows = read_jsonl(artifact) if artifact.exists() else []
        completed_ids = {row.get("sample_id") for row in existing_rows}
        if len(completed_ids) != len(existing_rows):
            raise ValueError(f"partial artifact has duplicate sample IDs: {artifact}")
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
        atomic_write_json(
            command_path,
            {
                "argv": list(argv),
                "variant": {
                    "name": variant.name,
                    "kv_dtype": variant.kv_dtype,
                    "skip_layers": list(variant.skip_layers),
                },
                "image_ref": image_ref,
                "model_revision": model_revision,
            },
        )

        name = container_name(run_id, variant)
        started = False
        completed = False
        captured_log = ""
        try:
            start_result = self.command_runner(argv, timeout=120)
            if not start_result.ok:
                raise RuntimeError(
                    f"docker run failed for {variant.name}: {start_result.stderr[-1000:]}"
                )
            started = True
            client = self.client_factory(
                f"http://127.0.0.1:{port}", self.profile.model.model_id
            )
            wait_until_ready(client, timeout_seconds=900, interval_seconds=2)
            captured_log = self._logs(name)
            validate_server_log(captured_log, variant)

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
                try:
                    echo = client.echo_logprobs(answer_prefix + expected)
                    answer_nll = answer_nll_from_echo(echo, len(answer_prefix))
                except (VllmHttpError, ValueError):
                    answer_nll = None
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
                        "quality_score": q_score,
                        "e2e_ms": elapsed_ms,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            completed = True
        finally:
            if started:
                try:
                    captured_log = self._logs(name) or captured_log
                    atomic_write_text(log_path, captured_log)
                finally:
                    safe_stop_container(name, self.command_runner)

        if completed:
            mark_complete(state_path, artifact, expected_rows)
        return artifact
