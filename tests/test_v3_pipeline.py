import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autokv.io import atomic_write_json, atomic_write_text, read_json
from autokv.v3_config import CONFIG_PATH
from autokv.v3_data import load_dataset, make_data
from autokv.v3_pipeline import execute, prepare_run, run_pipeline
from autokv.v3_runtime import V3PolicyRunner
from scripts.export_v3_results import export_results
from tests.test_v3_runtime import RuntimeHarness
from tests.v3_helpers import CharCodec, FakeRunner, config, samples, source_fixture


class FrozenTestRunner(FakeRunner):
    def evaluate(self, policy, split, ids):
        if split == "test":
            assert (self.directory / "selection.json").exists(), "测试前未冻结"
        return super().evaluate(policy, split, ids)


class V3PipelineTests(unittest.TestCase):
    def test_generated_data_to_native_runner_report_and_export(self):
        """只替换外部服务和 tokenizer，联调实际运行器、计分、容量与归档。"""
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg, codec = Path(tmp), config(), CharCodec()
            atomic_write_json(root / CONFIG_PATH, cfg.raw)
            with patch("autokv.v3_data.load_sources", return_value=(*source_fixture(), {})):
                make_data(root, root, codec=codec)
            _, data = load_dataset(root, cfg)
            harness = RuntimeHarness(cfg, root, data)

            def complete(prompt, max_tokens):
                harness.count += 1
                row = harness.lookup[prompt]
                answers = row["expected_answers"]
                output = "\n".join("=".join(pair) for pair in answers) if row["task"] == "multi_key" else ", ".join(answers)
                return {"choices": [{"message": {"content": output}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": row["prompt_tokens"], "completion_tokens": 8}}

            harness.client.chat_complete = complete

            def runner(cfg, root, directory, env, splits, **kwargs):
                return V3PolicyRunner(cfg, root, directory, env, splits, port=harness.port,
                                      client_factory=lambda *_: harness.client, process_factory=harness.start)

            with patch("autokv.v3_pipeline.sys.platform", "linux"), \
                 patch("autokv.v3_pipeline.load_environment", return_value=harness.environment), \
                 patch("autokv.v3_pipeline.TransformersPromptCodec", return_value=codec), \
                 patch("autokv.v3_pipeline.V3PolicyRunner", side_effect=runner):
                result = run_pipeline(root)
                self.assertEqual(result["status"], "capacity_below_target")
                self.assertTrue(result["quality_passed"])
                self.assertFalse(result["technical_goal_passed"])
                # 模拟日志两端容量相等，理论 2× 不能替代实测判断。
                self.assertEqual(result["capacity"]["ratio"], 1.0)
                self.assertEqual(result["cost"]["total"]["server_starts"], 4)
                self.assertEqual(harness.count, 2*(len(data["experiment"])+len(data["test"])))
                run_pipeline(root)
                self.assertEqual(harness.count, result["cost"]["total"]["requests"])
            directory = root / "runs" / result["run_id"]
            self.assertTrue((directory / "inputs/autokv/v3_pipeline.py").exists())
            self.assertTrue((directory / "selection.json").exists())
            self.assertIn("capacity_below_target", Path(result["report"]).read_text(encoding="utf-8"))
            exported = export_results(root, result["run_id"])
            self.assertTrue((exported / "autokv-v3.tar.gz").exists())

    def make_case(self, score=None, cfg=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name) / "runs/v3-pipeline"
        cfg = cfg or config()
        data = samples(cfg)
        runner = FrozenTestRunner(cfg, data, directory, score)
        return cfg, data, runner, directory

    def test_p0_passed_deduplicates_test_and_rerun(self):
        cfg, data, runner, directory = self.make_case()
        result = execute(cfg, data, runner, directory)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["technical_goal_passed"])
        self.assertEqual(len([call for call in runner.calls if call[1] == "test"]), 2)
        self.assertEqual(result["cost"]["total"]["requests"], 2*(len(data["experiment"])+len(data["test"])))
        self.assertTrue(Path(result["report"]).exists())
        self.assertNotIn("report_error", result)
        count = len(runner.calls)
        execute(cfg, data, runner, directory)
        self.assertEqual(len(runner.calls), count)

    def test_selected_mixed_passes_quality_and_capacity(self):
        cfg, data, runner, directory = self.make_case(lambda layers, row: 1.0 if len(layers) == 8 or layers == {2} else .8)
        result = execute(cfg, data, runner, directory)
        self.assertEqual(result["candidate"]["bf16_layers"], [2])
        self.assertTrue(result["technical_goal_passed"])
        self.assertEqual(len([call for call in runner.calls if call[1] == "test"]), 3)

    def test_test_failure_never_reselects(self):
        score = lambda layers, row: 1.0 if row["split"] == "experiment" or len(layers) == 8 else .8
        cfg, data, runner, directory = self.make_case(score)
        result = execute(cfg, data, runner, directory)
        self.assertEqual(result["status"], "test_quality_failed")
        self.assertTrue(result["complete"])
        self.assertFalse(result["technical_goal_passed"])
        frozen = (directory / "selection.json").read_bytes()
        with patch("autokv.v3_pipeline.search", side_effect=AssertionError("不能重新选择")):
            execute(cfg, data, runner, directory)
        self.assertEqual((directory / "selection.json").read_bytes(), frozen)

    def test_new_run_id_cannot_make_used_test_independent(self):
        cfg, data, runner, directory = self.make_case()
        execute(cfg, data, runner, directory)
        another = directory.with_name("v3-another")
        runner2 = FrozenTestRunner(cfg, data, another)
        result = execute(cfg, data, runner2, another)
        self.assertEqual(result["status"], "test_not_independent")
        self.assertTrue(result["complete"])
        self.assertFalse(result["technical_goal_passed"])
        self.assertEqual(result["test_reused_from"], [directory.name])

    def test_test_interruption_resumes_frozen_policy(self):
        cfg, data, runner, directory = self.make_case()
        runner.fail_test_once = True
        with self.assertRaises(RuntimeError):
            execute(cfg, data, runner, directory)
        partial = read_json(directory / "completed-manifest.json")
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["status"], "runtime_failed")
        self.assertTrue((directory / "selection.json").exists())
        with patch("autokv.v3_pipeline.search", side_effect=AssertionError("测试失败不能换层")):
            result = execute(cfg, data, runner, directory)
        self.assertTrue(result["technical_goal_passed"])

    def test_missing_conflicting_and_low_capacity(self):
        for observation, expected in ((lambda p: [None], "capacity_unverified"),
                                      (lambda p: [10000, 20000], "capacity_unverified"),
                                      (lambda p: [10000 if p.k == 8 else 11000], "capacity_below_target")):
            cfg, data, runner, directory = self.make_case()
            runner.capacity_override = observation
            result = execute(cfg, data, runner, directory)
            self.assertEqual(result["status"], expected)
            self.assertTrue(result["quality_passed"])
            self.assertFalse(result["technical_goal_passed"])

    def test_invalid_reference_and_no_candidate_do_not_use_test(self):
        cases = [(config(), lambda layers, row: 0.0 if row["task"] == "qa_multi" else 1.0, "reference_degenerate"),
                 (config(search={"max_bf16_layers": 0}), lambda layers, row: 1.0 if len(layers) == 8 else .8,
                  "no_feasible_within_budget")]
        for cfg, score, expected in cases:
            cfg, data, runner, directory = self.make_case(score, cfg)
            result = execute(cfg, data, runner, directory)
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["technical_goal_passed"])
            self.assertFalse(any(split == "test" for _, split, _ in runner.calls))

    def test_development_does_not_search_or_read_formal_data(self):
        cfg, data, runner, directory = self.make_case(lambda *_: 0)
        result = execute(cfg, {"development": data["development"]}, runner, directory, development=True)
        self.assertEqual(result["status"], "development_complete")
        self.assertFalse(result["technical_goal_passed"])
        self.assertEqual(len(runner.calls), 1)
        self.assertFalse((directory / "selection.json").exists())

    def test_source_zip_without_git_and_changed_code_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = Path(tmp), config()
            atomic_write_json(root / CONFIG_PATH, cfg.raw)
            atomic_write_text(root / "autokv/example.py", "value = 1\n")
            atomic_write_json(root / "data/v3.0/quality/dataset-manifest.json", {"dataset_id": "fixture"})
            with patch("autokv.v3_pipeline.__file__", str(root / "autokv/v3_pipeline.py")):
                directory = prepare_run(root, cfg, {"dataset_id": "fixture"}, {}, False)
                self.assertTrue((directory / "inputs/autokv/example.py").exists())
                self.assertFalse((root / ".git").exists())
                self.assertEqual(prepare_run(root, cfg, {"dataset_id": "fixture"}, {}, False), directory)
                atomic_write_text(root / "autokv/example.py", "value = 2\n")
                self.assertNotEqual(prepare_run(root, cfg, {"dataset_id": "fixture"}, {}, False), directory)
