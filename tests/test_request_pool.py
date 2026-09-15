import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from autokv.client import VllmHttpError
from autokv.request_pool import evaluate_requests
from autokv.v2_policy import endpoint_policies
from autokv.v3_runtime import SearchBudgetExceeded, recover_rows
from tests.test_v3_runtime import RuntimeHarness
from tests.v3_helpers import config, samples


def inputs(count):
    return [{"user_prompt": str(i), "max_tokens": 2} for i in range(count)]


class RequestPoolTests(unittest.TestCase):
    def test_four_in_flight_and_single_writer(self):
        barrier = threading.Barrier(4, timeout=5)
        lock = threading.Lock()
        active = maximum = 0
        main = threading.get_ident()
        saved = []

        def complete(prompt, tokens):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait()
            with lock:
                active -= 1
            return prompt

        def save(sample, response, started, retry):
            self.assertEqual(threading.get_ident(), main)
            saved.append(response)

        evaluate_requests(inputs(8), SimpleNamespace(chat_complete=complete), 4,
                          lambda retry: None, lambda *args: None, save)
        self.assertEqual(maximum, 4)
        self.assertCountEqual(saved, list(map(str, range(8))))

    def test_failure_stops_dispatch_and_saves_in_flight_success(self):
        released = threading.Event()
        saved, sent = [], []

        def complete(prompt, tokens):
            if prompt == "0":
                raise VllmHttpError(400, "不能重试")
            self.assertTrue(released.wait(5))
            return prompt

        with self.assertRaises(VllmHttpError):
            evaluate_requests(inputs(6), SimpleNamespace(chat_complete=complete), 2,
                              lambda retry: sent.append(retry), lambda *args: released.set(),
                              lambda sample, response, *args: saved.append(response))
        self.assertEqual(len(sent), 2)
        self.assertEqual(saved, ["1"])

    def test_budget_counts_dispatched_requests_before_completion(self):
        barrier = threading.Barrier(3, timeout=5)
        sent, saved = [], []

        def send(retry):
            if len(sent) == 3:
                raise SearchBudgetExceeded("预算耗尽")
            sent.append(retry)

        def complete(prompt, tokens):
            barrier.wait()
            return prompt

        with self.assertRaises(SearchBudgetExceeded):
            evaluate_requests(inputs(8), SimpleNamespace(chat_complete=complete), 4, send,
                              lambda *args: None, lambda sample, response, *args: saved.append(response))
        self.assertEqual(len(sent), 3)
        self.assertCountEqual(saved, ["0", "1", "2"])

    def test_retry_is_counted_and_keeps_sample_identity(self):
        calls, sent, errors, saved = [], [], [], []

        def complete(prompt, tokens):
            calls.append(prompt)
            if len(calls) == 1:
                raise VllmHttpError(503, "暂时不可用")
            return prompt

        evaluate_requests(inputs(1), SimpleNamespace(chat_complete=complete), 4,
                          sent.append, lambda sample, retry, exc: errors.append(retry),
                          lambda sample, response, start, retry: saved.append((response, retry)))
        self.assertEqual(sent, [0, 1])
        self.assertEqual(errors, [0])
        self.assertEqual(saved, [("0", 1)])

    def test_runner_writes_rows_and_resumes_without_duplicate_requests(self):
        cfg = config()
        cfg.raw["runtime"]["request_concurrency"] = 4
        data = samples(cfg)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            harness = RuntimeHarness(cfg, root, data)
            original = harness.client.chat_complete
            barrier = threading.Barrier(4, timeout=5)

            def complete(prompt, tokens):
                barrier.wait()
                return original(prompt, tokens)

            harness.client.chat_complete = complete
            policy = endpoint_policies(cfg.num_layers)[1]
            ids = [r["sample_id"] for r in data["experiment"][:4]]
            runner = harness.runner()
            rows = runner.evaluate(policy, "experiment", ids)
            self.assertEqual([r["sample_id"] for r in rows], ids)
            self.assertEqual(len(recover_rows(root / "runs/v3-unit/policies" / policy.config_id / "experiment.jsonl")), 4)
            harness.runner().evaluate(policy, "experiment", ids)
            self.assertEqual(harness.count, 4)
            self.assertEqual(runner.statistics()["total"]["requests"], 4)


if __name__ == "__main__":
    unittest.main()
