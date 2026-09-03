"""v2.0 单策略运行器：一次启动、策略级恢复、精简证据。"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from autokv.benchmark import parse_capacity_tokens
from autokv.client import VllmClient, VllmHttpError, wait_until_ready
from autokv.commands import (
    CommandResult,
    container_name,
    local_server_command,
    local_vllm_env,
    run_command,
    runtime_identity,
    server_command,
)
from autokv.config import Profile
from autokv.experiment import (
    inspect_container_command,
    safe_cleanup_owned_container,
    safe_remove_stale_container,
    validate_local_vllm_command,
    validate_server_log,
)
from autokv.io import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    ensure_within,
    read_json,
    read_jsonl,
    sha256_file,
)
from autokv.local_runtime import LocalVllmProcess
from autokv.v2_config import V2QualityConfig
from autokv.v2_metrics import score_v2_output
from autokv.v2_policy import Policy


Runner = Callable[..., CommandResult]
ClientFactory = Callable[[str, str], Any]


def _chat_completion_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("chat completion 响应缺少 choices[0].message.content") from exc
    if not isinstance(content, str):
        raise ValueError("chat completion 内容不是字符串")
    return content


def validate_first_output(output: str) -> None:
    if "\ufffd" in output:
        raise ValueError("首条响应含 Unicode 替换字符，疑似乱码")
    compact = re.sub(r"\s+", " ", output).strip()
    if len(compact) >= 64 and re.search(r"(.{8,64})\1{3,}", compact):
        raise ValueError("首条响应出现重复片段循环，疑似乱码")


def validate_prefix_caching_disabled(log: str, argv: Sequence[str]) -> None:
    if argv.count("--no-enable-prefix-caching") != 1:
        raise ValueError("server 命令未精确包含一次 --no-enable-prefix-caching")
    if "--enable-prefix-caching" in argv:
        raise ValueError("server 命令同时启用了 prefix caching")
    if re.search(r"enable_prefix_caching['\"]?\s*[=:]\s*(?:false|False)", log) is None:
        raise ValueError("server 日志未证明 enable_prefix_caching=False")


def _validate_result_rows(
    rows: Sequence[Mapping[str, Any]],
    policy: Policy,
    samples: Sequence[Mapping[str, Any]],
    split: str,
    run_id: str,
) -> None:
    expected = {str(sample["sample_id"]): sample for sample in samples}
    if len(rows) != len(expected):
        raise ValueError("策略结果行数不完整")
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or sample_id not in expected
            or sample_id in seen
        ):
            raise ValueError("策略结果含未知或重复 sample_id")
        seen.add(sample_id)
        if (
            row.get("schema_version") != 2
            or row.get("run_id") != run_id
            or row.get("split") != split
            or row.get("policy_config_id") != policy.config_id
            or row.get("policy_k") != policy.k
            or row.get("bf16_layers") != list(policy.bf16_layers)
            or row.get("tier") != expected[sample_id].get("tier")
            or row.get("task") != expected[sample_id].get("task")
            or row.get("error") is not None
        ):
            raise ValueError(f"{sample_id} 的策略结果上下文不一致")
        score = row.get("task_score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(f"{sample_id} 的 task_score 无效")


def policy_manifest_is_valid(
    manifest_path: Path,
    result_path: Path,
    log_path: Path,
    policy: Policy,
    samples: Sequence[Mapping[str, Any]],
    *,
    split: str,
    split_sha256: str,
    run_id: str,
) -> bool:
    try:
        manifest = read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            return False
        if (
            manifest.get("schema_version") != 2
            or manifest.get("complete") is not True
            or manifest.get("run_id") != run_id
            or manifest.get("split") != split
            or manifest.get("dataset_split_sha256") != split_sha256
            or manifest.get("policy") != policy.record()
            or manifest.get("enable_prefix_caching") is not False
            or manifest.get("rows") != len(samples)
            or manifest.get("failures") != 0
            or not result_path.is_file()
            or not log_path.is_file()
            or manifest.get("result_sha256") != sha256_file(result_path)
            or manifest.get("server_log_sha256") != sha256_file(log_path)
        ):
            return False
        rows = read_jsonl(result_path)
        _validate_result_rows(rows, policy, samples, split, run_id)
        return True
    except (OSError, TypeError, ValueError):
        return False


class V2PolicyRunner:
    def __init__(
        self,
        config: V2QualityConfig,
        profile: Profile,
        project_root: Path,
        lock: Mapping[str, Any],
        run_id: str,
        *,
        port: int,
        command_runner: Runner = run_command,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.project_root = project_root.resolve()
        self.lock = lock
        self.run_id = run_id
        self.port = port
        self.command_runner = command_runner
        self.client_factory = client_factory or (
            lambda base_url, model_id: VllmClient(base_url, model_id)
        )
        self.server_starts = 0
        self.requests = 0
        if not 1 <= port <= 65535:
            raise ValueError("port 必须在 1..65535")
        if lock.get("model_revision") != config.model_revision:
            raise ValueError("环境锁模型 revision 与 v2 配置不一致")
        if profile.calculate_kv_scales != config.calculate_kv_scales:
            raise ValueError("full profile 与 v2 配置的 KV scale 方案不一致")

    @property
    def local_backend(self) -> bool:
        return self.lock.get("backend") == "local_vllm"

    def _paths(
        self, relative_directory: Path, policy: Policy
    ) -> tuple[Path, Path, Path, Path]:
        directory = ensure_within(
            self.project_root,
            self.project_root / "runs" / self.run_id / relative_directory,
        )
        if not re.fullmatch(r"[a-z0-9-]+", policy.name):
            raise ValueError("策略名含不安全字符")
        result = directory / f"{policy.name}.jsonl"
        return (
            result,
            directory / f"{policy.name}.policy-manifest.json",
            directory / f"{policy.name}.server.log",
            directory / f".{policy.name}.working.jsonl",
        )

    def _archive_invalid(
        self, paths: Sequence[Path], relative_directory: Path, policy: Policy
    ) -> None:
        existing = [path for path in paths if path.exists()]
        if not existing:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive = ensure_within(
            self.project_root,
            self.project_root
            / "runs"
            / self.run_id
            / "_incomplete"
            / relative_directory
            / f"{policy.name}-{stamp}",
        )
        archive.mkdir(parents=True, exist_ok=False)
        for source in existing:
            source.replace(archive / source.name)

    def _docker_logs(self, name: str) -> str:
        result = self.command_runner(("docker", "logs", name), timeout=60)
        return result.stdout + result.stderr

    def _request_once(
        self, client: Any, sample: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return client.chat_complete(str(sample["prompt"]), int(sample["max_tokens"]))

    def _request_with_one_retry(
        self, client: Any, sample: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], int]:
        try:
            return self._request_once(client, sample), 0
        except VllmHttpError as exc:
            transient = exc.status is None or (
                isinstance(exc.status, int) and exc.status >= 500
            )
            if not transient:
                raise
            return self._request_once(client, sample), 1

    def run_policy(
        self,
        policy: Policy,
        samples: Sequence[Mapping[str, Any]],
        *,
        split: str,
        split_sha256: str,
        relative_directory: Path,
    ) -> Path:
        if not samples:
            raise ValueError("策略至少需要一个样本")
        if any(sample.get("split") != split for sample in samples):
            raise ValueError("策略样本与目标 split 不一致")
        policy.variant.validate_for_model(self.config.num_layers)
        result_path, manifest_path, log_path, working_path = self._paths(
            relative_directory, policy
        )
        if policy_manifest_is_valid(
            manifest_path,
            result_path,
            log_path,
            policy,
            samples,
            split=split,
            split_sha256=split_sha256,
            run_id=self.run_id,
        ):
            return result_path

        self._archive_invalid(
            (result_path, manifest_path, log_path, working_path),
            relative_directory,
            policy,
        )
        atomic_write_text(working_path, "")
        model_revision = str(self.lock["model_revision"])
        if self.local_backend:
            argv = local_server_command(
                self.profile,
                str(self.lock["vllm"]),
                policy.variant,
                self.project_root,
                self.port,
                model_revision,
            )
        else:
            argv = server_command(
                self.profile,
                str(self.lock["image_ref"]),
                policy.variant,
                self.project_root,
                self.port,
                self.run_id,
                model_revision,
            )

        name = container_name(self.run_id, policy.variant)
        process: LocalVllmProcess | None = None
        started = False
        inspected_argv: tuple[str, ...] = ()
        captured_log = ""
        capacity: dict[str, Any] | None = None
        completed = False
        try:
            if self.local_backend:
                process = LocalVllmProcess.start(
                    argv,
                    log_path,
                    cwd=self.project_root,
                    env=local_vllm_env(self.lock),
                )
            else:
                safe_remove_stale_container(name, self.command_runner)
                start = self.command_runner(argv, timeout=120)
                if not start.ok:
                    raise RuntimeError(
                        f"启动 {policy.name} 失败：returncode={start.returncode}; "
                        f"stderr={start.stderr[-1000:] or '<empty>'}"
                    )
            started = True
            self.server_starts += 1
            client = self.client_factory(
                f"http://127.0.0.1:{self.port}", self.config.model_id
            )
            wait_until_ready(client, timeout_seconds=900, interval_seconds=2)
            if self.local_backend:
                inspected_argv = tuple(argv)
                validate_local_vllm_command(
                    json.dumps(argv), policy.variant, self.config.calculate_kv_scales
                )
            else:
                inspected_argv = inspect_container_command(
                    name,
                    policy.variant,
                    self.command_runner,
                    calculate_kv_scales=self.config.calculate_kv_scales,
                )
            captured_log = (
                process.log_text()
                if self.local_backend and process
                else self._docker_logs(name)
            )
            validate_server_log(captured_log, policy.variant, self.config.num_layers)
            validate_prefix_caching_disabled(captured_log, inspected_argv)

            for index, sample in enumerate(samples):
                started_at = time.monotonic()
                try:
                    response, retry_count = self._request_with_one_retry(client, sample)
                    self.requests += 1 + retry_count
                    output = _chat_completion_text(response)
                    if index == 0:
                        validate_first_output(output)
                    task_score = score_v2_output(output, sample)
                    usage = response.get("usage")
                    if not isinstance(usage, Mapping):
                        raise ValueError("chat completion 响应缺少 usage")
                    observed_prompt_tokens = usage.get("prompt_tokens")
                    if observed_prompt_tokens != sample["prompt_tokens"]:
                        raise ValueError(
                            "服务端 chat template/tokenizer 计数与冻结数据不一致："
                            f"sample={sample['sample_id']}, "
                            f"frozen={sample['prompt_tokens']}, observed={observed_prompt_tokens}"
                        )
                except BaseException as exc:
                    append_jsonl(
                        working_path,
                        {
                            "schema_version": 2,
                            "run_id": self.run_id,
                            "sample_id": sample.get("sample_id"),
                            "split": split,
                            "policy_config_id": policy.config_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    raise
                output_tokens = usage.get("completion_tokens")
                append_jsonl(
                    working_path,
                    {
                        "schema_version": 2,
                        "run_id": self.run_id,
                        "runtime_id": runtime_identity(self.lock),
                        "model_revision": model_revision,
                        "policy_config_id": policy.config_id,
                        "policy_name": policy.name,
                        "policy_k": policy.k,
                        "bf16_layers": list(policy.bf16_layers),
                        "split": split,
                        "sample_id": sample["sample_id"],
                        "tier": sample["tier"],
                        "task": sample["task"],
                        "prompt_tokens": sample["prompt_tokens"],
                        "output_tokens": output_tokens,
                        "output_text": output,
                        "task_score": task_score,
                        "answer_nll": None,
                        "e2e_ms": (time.monotonic() - started_at) * 1000.0,
                        "retry_count": retry_count,
                        "error": None,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            completed = True
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error: BaseException | None = None
            if started:
                try:
                    captured_log = (
                        process.log_text()
                        if self.local_backend and process
                        else self._docker_logs(name)
                    ) or captured_log
                    atomic_write_text(log_path, captured_log)
                except BaseException as exc:
                    if primary_error is None:
                        primary_error = exc
            try:
                if started:
                    if self.local_backend:
                        assert process is not None
                        process.stop()
                    else:
                        safe_cleanup_owned_container(name, self.command_runner)
            except BaseException as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                if primary_error is not None:
                    raise RuntimeError(
                        f"策略运行失败：{primary_error}；清理本项目 server 又失败：{cleanup_error}"
                    ) from primary_error
                raise cleanup_error
            if sys.exc_info()[1] is None and primary_error is not None:
                raise primary_error

        if not completed:
            raise RuntimeError(f"策略未完成：{policy.name}")
        rows = read_jsonl(working_path)
        _validate_result_rows(rows, policy, samples, split, self.run_id)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(working_path, result_path)
        capacity_value = parse_capacity_tokens(captured_log)
        capacity = {
            "tokens": capacity_value.tokens,
            "model_length": capacity_value.model_length,
            "max_concurrency": capacity_value.max_concurrency,
        }
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 2,
                "complete": True,
                "run_id": self.run_id,
                "split": split,
                "dataset_split_sha256": split_sha256,
                "policy": policy.record(),
                "runtime_backend": str(self.lock.get("backend", "docker")),
                "runtime_id": runtime_identity(self.lock),
                "model_revision": model_revision,
                "effective_argv": list(inspected_argv),
                "enable_prefix_caching": False,
                "capacity": capacity,
                "rows": len(rows),
                "failures": 0,
                "result_path": result_path.name,
                "result_sha256": sha256_file(result_path),
                "server_log_path": log_path.name,
                "server_log_sha256": sha256_file(log_path),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return result_path
