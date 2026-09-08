"""本地 vLLM 增量评估；成功回答逐题保存，配置切换重建引擎。"""
from __future__ import annotations

import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from autokv.benchmark import parse_capacity_tokens
from autokv.client import VllmClient, VllmHttpError, wait_until_ready
from autokv.commands import local_server_command
from autokv.experiment import validate_server_log
from autokv.io import append_jsonl, atomic_write_json, read_json, sha256_file
from autokv.local_runtime import LocalVllmProcess
from autokv.v2_runtime import _chat_completion_text, v2_local_env, validate_prefix_caching_disabled
from autokv.v3_config import identity
from autokv.v3_metrics import score_output


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_environment(root, config, *, observe_runtime=False):
    path = root / "runs/_environment/lock.json"
    env = dict(read_json(path)) if path.exists() else {}
    for key, variable in (("vllm", "AUTOKV_VLLM_BIN"), ("model_path", "AUTOKV_MODEL_PATH")):
        if os.environ.get(variable):
            env[key] = os.environ[variable]
        if not env.get(key):
            raise ValueError(f"缺少 {key}；设置 {variable} 或复用已有 runs/_environment/lock.json")
    if env.get("backend", "local_vllm") != "local_vllm":
        raise ValueError("v3 首版使用已有本地 vLLM 环境；请设置本地模型与 vLLM 路径")
    if not Path(env["model_path"]).is_dir():
        raise ValueError("已有模型目录不存在：" + env["model_path"])
    if env.get("model_revision", config.model_revision) != config.model_revision:
        raise ValueError("环境模型 revision 与协议不一致")
    if env.get("model_id", config.model_id) != config.model_id:
        raise ValueError("已有环境记录的模型与协议不一致")
    env.update(backend="local_vllm", model_revision=config.model_revision, model_id=config.model_id,
               python=sys.executable, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    versions = {}
    for name in ("vllm", "torch", "flashinfer-python", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    env["observed_versions"] = versions
    if observe_runtime:
        try:
            observed = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,compute_cap,memory.total",
                                       "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            env["gpu_observation"] = observed.stdout.strip() if observed.returncode == 0 else observed.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            env["gpu_observation"] = f"无法读取 GPU 信息：{exc}"
        model_config = Path(env["model_path"]) / "config.json"
        if model_config.exists():
            model = read_json(model_config)
            dimensions = {"num_layers": model.get("num_hidden_layers"), "num_kv_heads": model.get("num_key_value_heads"),
                          "head_dim": model.get("head_dim") or model["hidden_size"]//model["num_attention_heads"]}
            if any(value != config.raw["model"][key] for key, value in dimensions.items()):
                raise ValueError("实际模型的 KV 结构与协议不一致")
            env["model_config"] = model
        env["tokenizer_files"] = {name: sha256_file(Path(env["model_path"]) / name)
                                  for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
                                  if (Path(env["model_path"]) / name).is_file()}
    return {key: env[key] for key in ("backend", "vllm", "python", "model_path", "model_revision", "model_id",
            "ld_library_path", "cuda_home", "nvcc", "flashinfer_workspace_base", "torch_extensions_dir",
            "observed_versions", "cuda_visible_devices", "gpu_observation", "model_config", "tokenizer_files") if key in env}


class SearchBudgetExceeded(RuntimeError):
    pass


def recover_rows(path):
    """仅恢复损坏尾行，完整中间行损坏必须保留并报告。"""
    if not path.exists():
        return []
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    rows, offset = [], 0
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines)-1:
                raise ValueError(f"结果中间行损坏：{path}:{index+1}")
            tail = path.with_name(path.name + f".tail-{time.time_ns()}.bin")
            tail.write_bytes(content[offset:])
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        if not isinstance(value, dict):
            raise ValueError(f"结果不是 JSON 对象：{path}:{index+1}")
        rows.append(value)
        offset += len(line)
    if rows and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


class V3PolicyRunner:
    def __init__(self, config, root, run_dir, environment, samples, *, port=8010,
                 client_factory=VllmClient, process_factory=LocalVllmProcess.start):
        self.config, self.root, self.run_dir, self.environment = config, root, run_dir, environment
        self.samples = {split: {r["sample_id"]: r for r in rows} for split, rows in samples.items()}
        self.port, self.client_factory, self.process_factory = port, client_factory, process_factory
        self.phase = "endpoints"
        self.context_id = identity({"config": config.raw, "environment": environment})
        self._cached = {}
        self._attempts = [read_json(p) for p in sorted(run_dir.glob("policies/*/attempts/*.json"))]
        self._active = None

    def cached(self, policy, split):
        key = (policy.config_id, split)
        if key not in self._cached:
            rows = recover_rows(self.run_dir / "policies" / policy.config_id / f"{split}.jsonl")
            indexed = {}
            for row in rows:
                sample = self.samples[split].get(row["sample_id"])
                if sample is None or row.get("context_id") != self.context_id or row.get("sample_signature") != identity(sample):
                    raise ValueError("缓存的运行设置或输入已改变；请使用新运行，不能混用旧回答")
                if row.get("policy_config_id") != policy.config_id or row.get("error") or row["sample_id"] in indexed:
                    raise ValueError("结果包含错误、重复样本或不同策略")
                if any(row.get(k) != sample[k] for k in ("split", "task", "length_bucket", "source_group_id")):
                    raise ValueError("缓存行的分层元数据与输入不一致")
                row["task_score"] = score_output(row["output_text"], sample)
                indexed[row["sample_id"]] = row
            self._cached[key] = indexed
        return self._cached[key]

    def _check_budget(self):
        if self.phase != "search":
            return
        attempts = [a for a in self._attempts if a["phase"] == "search"]
        requests = sum(a["requests"] for a in attempts)
        seconds = sum(a.get("seconds", 0) for a in attempts)
        if self._active:
            requests += self._active["requests"]
            seconds += time.monotonic()-self._active["monotonic_start"]
        limit = self.config.raw["search"]
        if limit.get("max_requests") is not None and requests >= limit["max_requests"]:
            raise SearchBudgetExceeded("选层请求预算已用尽")
        if limit.get("max_seconds") is not None and seconds >= limit["max_seconds"]:
            raise SearchBudgetExceeded("选层墙钟预算已用尽")

    def evaluate(self, policy, split, sample_ids):
        if len(set(sample_ids)) != len(sample_ids) or not sample_ids:
            raise ValueError("评估需要不重复的样本 ID")
        samples = [self.samples[split][sid] for sid in sample_ids]
        cached = self.cached(policy, split)
        for sample in samples:
            if sample["sample_id"] in cached and cached[sample["sample_id"]]["sample_signature"] != identity(sample):
                raise ValueError("同一运行中的样本被修改")
        missing = [s for s in samples if s["sample_id"] not in cached]
        if not missing:
            return [cached[sid] for sid in sample_ids]
        self._check_budget()
        print(f"{self.phase} / P{policy.k} {list(policy.bf16_layers)}：补 {len(missing)} 题，目标 {len(samples)} 题", file=sys.stderr, flush=True)
        directory = self.run_dir / "policies" / policy.config_id
        attempt_dir = directory / "attempts"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{split}-{time.time_ns()}"
        log_path = attempt_dir / f"{stem}.log"
        attempt_path = attempt_dir / f"{stem}.json"
        argv = local_server_command(self.config.profile(), self.environment["vllm"], policy.variant,
                                    self.root, self.port, self.config.model_revision,
                                    model_path=Path(self.environment["model_path"]))
        attempt = {"phase": self.phase, "split": split, "policy": policy.record(), "sample_ids": sample_ids,
                   "missing_ids": [s["sample_id"] for s in missing], "started_at": utc_now(), "argv": list(argv),
                   "requests": 0, "retries": 0, "server_starts": 0, "complete": False, "capacity_tokens": None}
        atomic_write_json(attempt_path, attempt)
        process = None
        started = time.monotonic()
        self._active = {**attempt, "monotonic_start": started}
        try:
            with socket.socket() as probe:
                if sys.platform.startswith("linux"):
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("0.0.0.0", self.port))
            process = self.process_factory(argv, log_path, cwd=self.root, env=v2_local_env(self.environment))
            attempt["server_starts"] = 1
            atomic_write_json(attempt_path, attempt)
            client = self.client_factory(f"http://127.0.0.1:{self.port}", self.config.model_id)
            wait_until_ready(client, timeout_seconds=900, interval_seconds=2)
            log = process.log_text()
            validate_server_log(log, policy.variant, self.config.num_layers, require_capacity=False)
            validate_prefix_caching_disabled(log, argv)
            attempt["dtype_verified"] = True
            for sample in missing:
                request_started = time.monotonic()
                retries = 0
                for retry in range(2):
                    self._check_budget()
                    attempt["requests"] += 1
                    attempt["seconds"] = time.monotonic()-started
                    self._active["requests"] = attempt["requests"]
                    atomic_write_json(attempt_path, attempt)
                    try:
                        response = client.chat_complete(sample["user_prompt"], sample["max_tokens"])
                        break
                    except (VllmHttpError, TimeoutError) as exc:
                        append_jsonl(directory / "request-errors.jsonl", {"sample_id": sample["sample_id"],
                                     "attempt": stem, "error": str(exc), "retry": retry, "timestamp": utc_now()})
                        if retry or isinstance(exc, VllmHttpError) and exc.status is not None and exc.status < 500:
                            raise
                        retries += 1
                        attempt["retries"] += 1
                output = _chat_completion_text(response)
                usage = response.get("usage", {})
                if usage.get("prompt_tokens") != sample["prompt_tokens"]:
                    raise ValueError(f"服务端 prompt token 与数据不一致：{sample['sample_id']}")
                if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 0:
                    raise ValueError("响应缺少有效 completion_tokens")
                row = {"schema_version": 3, "sample_id": sample["sample_id"], "split": split,
                       "task": sample["task"], "length_bucket": sample["length_bucket"],
                       "source_group_id": sample["source_group_id"], "policy_config_id": policy.config_id,
                       "sample_signature": identity(sample), "context_id": self.context_id,
                       "prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                       "output_text": output, "task_score": score_output(output, sample),
                       "finish_reason": response["choices"][0].get("finish_reason"),
                       "e2e_ms": (time.monotonic()-request_started)*1000, "retry_count": retries,
                       "attempt": stem, "error": None, "timestamp": utc_now()}
                append_jsonl(directory / f"{split}.jsonl", row)
                cached[row["sample_id"]] = row
                if len(cached) % 16 == 0:
                    print(f"  已保存 {len(cached)} 条 {split} 回答", file=sys.stderr, flush=True)
            attempt["complete"] = True
        except BaseException as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                if process is not None:
                    process.stop()
            except BaseException as exc:
                attempt["cleanup_error"] = str(exc)
                attempt["complete"] = False
                raise
            finally:
                if process is not None:
                    try:
                        attempt["capacity_tokens"] = parse_capacity_tokens(process.log_text()).tokens
                    except ValueError:
                        pass
                attempt.update(seconds=time.monotonic()-started, finished_at=utc_now())
                atomic_write_json(attempt_path, attempt)
                self._attempts.append(attempt)
                self._active = None
        return [cached[sid] for sid in sample_ids]

    def statistics(self):
        phases = {}
        for phase in ("development", "endpoints", "search", "test"):
            attempts = [a for a in self._attempts if a["phase"] == phase]
            phases[phase] = {key: sum(a.get(key, 0) for a in attempts)
                             for key in ("requests", "retries", "server_starts", "seconds")}
        return {"phases": phases, "timing_incomplete": any(not a.get("finished_at") for a in self._attempts),
                "total": {k: sum(p[k] for p in phases.values())
                                            for k in ("requests", "retries", "server_starts", "seconds")}}

    def capacities(self, policy, split="test"):
        return [a.get("capacity_tokens") for a in self._attempts
                if a["policy"]["config_id"] == policy.config_id and a["split"] == split and a.get("dtype_verified")]
