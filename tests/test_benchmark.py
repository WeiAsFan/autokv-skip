import json
import tempfile
import unittest
from pathlib import Path

from autokv.benchmark import BenchmarkRunner, build_benchmark_matrix, parse_capacity_tokens
from autokv.commands import CommandResult
from autokv.config import Profile
from autokv.io import read_json
from autokv.selection import Variant


def command_result(argv, returncode=0, stdout="", stderr=""):
    return CommandResult(tuple(argv), returncode, stdout, stderr, 0.01)


class FakeClient:
    def health(self):
        return True

    def complete(self, prompt, max_tokens):
        return {"choices": [{"text": "warm"}]}


class BenchmarkTests(unittest.TestCase):
    def test_parses_vllm_capacity_log(self):
        log = (
            "GPU KV cache size: 233,104 tokens\n"
            "Maximum concurrency for 32,768 tokens per request: 7.11x\n"
        )
        parsed = parse_capacity_tokens(log)
        self.assertEqual(parsed.tokens, 233104)
        self.assertEqual(parsed.model_length, 32768)
        self.assertAlmostEqual(parsed.max_concurrency, 7.11)

    def test_missing_capacity_is_a_hard_error(self):
        with self.assertRaisesRegex(ValueError, "KV cache size"):
            parse_capacity_tokens("server ready")

    def test_quick_matrix_has_six_scenarios(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        cases = build_benchmark_matrix(profile)
        self.assertEqual(len(cases), 6)
        self.assertEqual(cases[0].input_length, 1024)
        self.assertEqual(cases[-1].output_length, 256)

    def test_benchmark_runner_writes_resumable_summary(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        lock = {
            "image_ref": "vllm/vllm-openai@sha256:" + "f" * 64,
            "image_digest": "sha256:" + "f" * 64,
            "model_revision": "a" * 40,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            timed_out_once = False

            def fake_runner(argv, timeout=None, **kwargs):
                nonlocal timed_out_once
                calls.append(tuple(argv))
                if tuple(argv[:2]) == ("docker", "logs"):
                    return command_result(
                        argv,
                        stdout=(
                            "Using FLASHINFER backend\n"
                            "FP8 KV cache enabled\n"
                            "GPU KV cache size: 233,104 tokens\n"
                            "Maximum concurrency for 32,768 tokens per request: 7.11x\n"
                        ),
                    )
                if tuple(argv[:2]) == ("docker", "inspect"):
                    if "State.Running" in " ".join(argv):
                        return command_result(
                            argv, returncode=1, stderr="No such object"
                        )
                    return command_result(argv, stdout="autokv-skip\n")
                if "bench" in argv:
                    if not timed_out_once:
                        timed_out_once = True
                        return command_result(argv, returncode=124, stderr="timed out")
                    container_dir = argv[argv.index("--result-dir") + 1]
                    filename = argv[argv.index("--result-filename") + 1]
                    relative = Path(
                        container_dir.removeprefix("/workspace/autokv-skip/")
                    )
                    result_path = root / relative / filename
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(
                        json.dumps(
                            {
                                "request_throughput": 2.5,
                                "output_throughput": 30.0,
                                "median_ttft_ms": 123.0,
                                "median_tpot_ms": 8.0,
                                "median_itl_ms": 7.0,
                            }
                        ),
                        encoding="utf-8",
                    )
                return command_result(argv)

            runner = BenchmarkRunner(
                profile,
                root,
                lock,
                command_runner=fake_runner,
                client_factory=lambda base_url, model_id: FakeClient(),
            )
            variant = Variant.mixed("auto-4", (2, 7, 18, 29))
            _, _, _, result_directory = runner._paths("abc123", variant)
            result_directory.mkdir(parents=True, exist_ok=True)
            (result_directory / "in1024-out32-rep0.json").write_text(
                "{}", encoding="utf-8"
            )
            summary_path = runner.run_variant(
                variant,
                run_id="abc123",
                port=8000,
            )
            summary = read_json(summary_path)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["capacity"]["tokens"], 233104)
            self.assertEqual(len(summary["scenarios"]), 6)
            self.assertEqual(summary["aggregate"]["request_throughput"], 2.5)
            bench_calls = [call for call in calls if "bench" in call]
            self.assertEqual(len(bench_calls), 7)
            call_count = len(calls)
            self.assertEqual(
                runner.run_variant(
                    variant,
                    run_id="abc123",
                    port=8000,
                ),
                summary_path,
            )
            self.assertEqual(len(calls), call_count)


if __name__ == "__main__":
    unittest.main()
