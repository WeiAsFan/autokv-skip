import copy
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from autokv.client import VllmHttpError
from autokv.v2_policy import endpoint_policies
from autokv.v3_runtime import SearchBudgetExceeded, V3PolicyRunner, recover_rows
from tests.v3_helpers import config, samples


class RuntimeHarness:
    def __init__(self, cfg, root, data):
        self.cfg, self.root, self.data = cfg, root, data
        self.count = 0
        self.fail_at = None
        self.empty = False
        self.wrong_tokens = False
        self.wrong_dtype = False
        self.starts, self.processes = [], []
        self.lookup = {r["user_prompt"]: r for rows in data.values() for r in rows}
        self.client = SimpleNamespace(health=lambda: True, chat_complete=self.complete)
        self.environment = {"vllm": "/existing/vllm", "model_path": "/existing/model", "model_revision": cfg.model_revision}
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]

    def start(self, argv, log_path, **kwargs):
        self.starts.append((argv, kwargs))
        dtype = argv[argv.index("--kv-cache-dtype")+1]
        log = "Using AttentionBackendEnum.FLASHINFER backend.\n"
        log += "Using fp8_e4m3 data type to store kv cache.\n" if dtype == "fp8_e4m3" else "kv_cache_dtype=bfloat16\n"
        if "--kv-cache-dtype-skip-layers" in argv:
            skip = set(map(int, argv[argv.index("--kv-cache-dtype-skip-layers")+1:]))
            log += "\n".join(f"Layer model.layers.{i}.self_attn: kv_cache_dtype={'auto' if i in skip else 'fp8_e4m3'}"
                             for i in range(self.cfg.num_layers))
        log += "\nGPU KV cache size: 100,000 tokens\n"
        if self.wrong_dtype:
            log = "wrong backend"
        log_path.write_text(log, encoding="utf-8")
        process = SimpleNamespace(log_text=lambda: log, stop=Mock())
        self.processes.append(process)
        return process

    def complete(self, prompt, max_tokens):
        self.count += 1
        if self.fail_at == self.count:
            raise VllmHttpError(400, "模拟请求中断")
        row = self.lookup[prompt]
        task = row["task"]
        output = "K00000001=V00000001" if task == "multi_key" else "V00000001" if task == "multi_value" else "Paris"
        return {"choices": [{"message": {"content": "" if self.empty else output}, "finish_reason": "length" if self.empty else "stop"}],
                "usage": {"prompt_tokens": row["prompt_tokens"]+int(self.wrong_tokens), "completion_tokens": 3}}

    def runner(self, environment=None):
        return V3PolicyRunner(self.cfg, self.root, self.root / "runs/v3-unit", environment or self.environment,
                              self.data, port=self.port, client_factory=lambda *_: self.client, process_factory=self.start)


class V3RuntimeTests(unittest.TestCase):
    def test_incremental_32_96_256_and_restart_reuse(self):
        cfg = config(False)
        data = samples(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            h = RuntimeHarness(cfg, Path(tmp), data)
            runner = h.runner()
            _, policy = endpoint_policies(cfg.num_layers)
            ids = [r["sample_id"] for r in data["experiment"]]
            for n in (32, 96, 256):
                runner.evaluate(policy, "experiment", ids[:n])
            self.assertEqual(h.count, 256)
            self.assertEqual(len(h.starts), 3)
            rows = h.runner().evaluate(policy, "experiment", ids)
            self.assertEqual(h.count, 256)
            self.assertEqual(len(rows), 256)
            self.assertTrue(all(r["task_score"] == 1 for r in rows))
            self.assertIn("--no-enable-prefix-caching", h.starts[0][0])
            self.assertEqual(h.starts[0][1]["env"]["HF_HUB_OFFLINE"], "1")

    def test_interruption_keeps_first_31_and_corrupt_tail(self):
        cfg, data = config(), samples(config())
        with tempfile.TemporaryDirectory() as tmp:
            h = RuntimeHarness(cfg, Path(tmp), data)
            _, p = endpoint_policies(cfg.num_layers)
            ids = [r["sample_id"] for r in data["experiment"]]
            h.fail_at = 32
            with self.assertRaises(VllmHttpError):
                h.runner().evaluate(p, "experiment", ids)
            path = Path(tmp) / "runs/v3-unit/policies" / p.config_id / "experiment.jsonl"
            self.assertEqual(len(recover_rows(path)), 31)
            with path.open("ab") as stream:
                stream.write(b'{"unfinished":')
            h.runner().evaluate(p, "experiment", ids)
            self.assertEqual(h.count, len(ids)+1)
            self.assertEqual(len(recover_rows(path)), len(ids))
            self.assertTrue(list(path.parent.glob("*.tail-*.bin")))

    def test_input_and_environment_changes_cannot_reuse(self):
        cfg, data = config(), samples(config())
        with tempfile.TemporaryDirectory() as tmp:
            h = RuntimeHarness(cfg, Path(tmp), data)
            p, _ = endpoint_policies(cfg.num_layers)
            sid = data["experiment"][0]["sample_id"]
            h.runner().evaluate(p, "experiment", [sid])
            with self.assertRaisesRegex(ValueError, "缓存"):
                h.runner({**h.environment, "model_path": "/different/model"}).evaluate(p, "experiment", [sid])
            data["experiment"][0]["user_prompt"] += "changed"
            with self.assertRaisesRegex(ValueError, "缓存"):
                h.runner().evaluate(p, "experiment", [sid])

    def test_wrong_answers_not_retried_and_length_recorded(self):
        cfg, data = config(), samples(config())
        with tempfile.TemporaryDirectory() as tmp:
            h = RuntimeHarness(cfg, Path(tmp), data)
            h.empty = True
            p, _ = endpoint_policies(cfg.num_layers)
            rows = h.runner().evaluate(p, "experiment", [data["experiment"][0]["sample_id"]])
            self.assertEqual(h.count, 1)
            self.assertEqual(rows[0]["task_score"], 0)
            self.assertEqual(rows[0]["finish_reason"], "length")

    def test_actual_dtype_and_prompt_checks_stop_invalid_results(self):
        for setting in ("wrong_tokens", "wrong_dtype"):
            cfg, data = config(), samples(config())
            with tempfile.TemporaryDirectory() as tmp:
                h = RuntimeHarness(cfg, Path(tmp), data)
                setattr(h, setting, True)
                p, _ = endpoint_policies(cfg.num_layers)
                with self.assertRaises(ValueError):
                    h.runner().evaluate(p, "experiment", [data["experiment"][0]["sample_id"]])
                self.assertEqual(h.processes[0].stop.call_count, 1)

    def test_search_budget_persists_after_restart(self):
        cfg = config(search={"max_requests": 1})
        data = samples(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            h = RuntimeHarness(cfg, Path(tmp), data)
            p, _ = endpoint_policies(cfg.num_layers)
            for _ in range(2):
                runner = h.runner()
                runner.phase = "search"
                with self.assertRaises(SearchBudgetExceeded):
                    runner.evaluate(p, "experiment", [r["sample_id"] for r in data["experiment"]])
            self.assertEqual(h.count, 1)

    def test_middle_corruption_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"rows.jsonl"
            path.write_bytes(b'{"a":1}\nnot json\n{"b":2}\n')
            with self.assertRaises(ValueError):
                recover_rows(path)
