import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from autokv.io import atomic_write_json, read_json, read_jsonl
from autokv.v4_config import V4Config
from autokv.v4_pipeline import execute_v4
from autokv.v42_pipeline import describe_source, execute_reused, import_dataset, load_reused, run_reused
from autokv.v42_search import search
from scripts.export_v42_results import export_results
from tests.test_v41_fp4 import FP4Harness
from tests.v3_helpers import FakeRunner
from tests.v4_helpers import CharCodec, config, sources

ROOT = Path(__file__).resolve().parents[1]


class BoundedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = read_json(ROOT / "configs/v4.2/quality.json")
        self.cfg = V4Config.from_dict(raw)
        self.cfg = self.cfg.for_condition(self.cfg.conditions()[0])
        self.data = {"experiment": [{"sample_id": str(i), "index": i, "split": "experiment",
            "task": self.cfg.tasks[0], "length_bucket": self.cfg.lengths[0], "source_group_id": str(i)}
            for i in range(512)]}

    def run_search(self, score, runner=None):
        runner = runner or FakeRunner(self.cfg, self.data, self.root, score)
        result = search(self.cfg, self.data["experiment"], runner, self.root / "trace.jsonl")
        return result, runner, read_jsonl(self.root / "trace.jsonl")

    def test_all_ties_reach_nine_layers_with_six_full_evaluations(self):
        result, runner, trace = self.run_search(lambda layers, _: 1.0 if len(layers) == 32 else .5)
        self.assertEqual(self.cfg.max_layers, 9)
        self.assertEqual(result["depth_reached"], 9)
        self.assertEqual(result["full_candidates_used"], 6)
        self.assertEqual(result["early_full_candidates_used"], 0)
        self.assertIsNone(result["candidate"])
        self.assertLessEqual(runner.counts["search"]["requests"], 24320)
        for depth in range(1, 10):
            medium = [r for r in trace if r["event"] == "evaluation" and r["depth"] == depth and r["samples"] == 128]
            self.assertEqual(len(medium), 8)
            beam = next(r for r in trace if r["event"] == "beam" and r["depth"] == depth)
            self.assertEqual(len(beam["policies"]), 2)
        self.assertEqual(len([r for r in trace if r["event"] == "full_confirmation"]), 6)
        self.assertFalse(any(split == "test" for _, split, _ in runner.calls))

    def test_early_false_positives_leave_confirmation_for_deep_solution(self):
        def score(layers, sample):
            if len(layers) in (9, 32):
                return 1.0
            if layers and sample["index"] < 128:
                return .99+len(layers)*.001
            return .5
        result, _, trace = self.run_search(score)
        self.assertEqual(result["candidate"]["k"], 9)
        self.assertEqual(result["early_full_candidates_used"], 3)
        self.assertEqual(result["full_candidates_used"], 4)
        full = [r for r in trace if r["event"] == "full_confirmation"]
        self.assertEqual([r["stage"] for r in full], ["early"]*3+["final"])

    def test_second_parent_conditional_gain_and_full_confirmation(self):
        def score(layers, _):
            if len(layers) == 32 or layers == {3, 7}:
                return 1.0
            if len(layers) == 1:
                return .94 if 0 in layers else .93 if 3 in layers else .5
            return .6
        result, _, trace = self.run_search(score)
        self.assertEqual(result["candidate"]["bf16_layers"], [3, 7])
        self.assertEqual(result["full_candidates_used"], 1)
        self.assertTrue(any(r["event"] == "evaluation" and r["samples"] == 512 and
                            r["policy"]["bf16_layers"] == [3, 7] for r in trace))

    def test_resume_during_full_evaluation_preserves_unique_slots_and_request_budget(self):
        score = lambda layers, _: 1.0 if len(layers) == 32 else .5
        runner = FakeRunner(self.cfg, self.data, self.root, score)
        original = runner.evaluate
        failed = False
        def interrupt(policy, split, ids):
            nonlocal failed
            if policy.k not in (0, 32) and len(ids) == 512 and not failed:
                failed = True
                original(policy, split, ids[:200])
                raise RuntimeError("模拟完整评估中断")
            return original(policy, split, ids)
        runner.evaluate = interrupt
        with self.assertRaisesRegex(RuntimeError, "中断"):
            self.run_search(score, runner)
        runner.evaluate = original
        result, runner, _ = self.run_search(score, runner)
        completed = [rows for (pid, split), rows in runner.rows.items() if len(rows) == 512]
        self.assertEqual(len(completed), 8)  # 两个端点与六个混合候选。
        self.assertEqual(result["full_candidates_used"], 6)
        self.assertLessEqual(runner.counts["search"]["requests"], 24320)

    def test_budget_exhaustion_cannot_accept_partial_candidate(self):
        self.cfg.raw["search"]["max_requests"] = 4
        result, runner, _ = self.run_search(lambda layers, _: 1.0 if len(layers) == 32 else .5)
        self.assertEqual(result["status"], "no_feasible_within_budget")
        self.assertIsNone(result["candidate"])
        self.assertEqual(runner.counts["search"]["requests"], 4)

    def test_p0_pass_and_bf16_degenerate_use_no_search_budget(self):
        for value, expected in ((1.0, "selected"), (0.0, "reference_degenerate")):
            result, runner, _ = self.run_search(lambda *_: value)
            self.assertEqual(result["status"], expected)
            self.assertEqual(runner.counts["search"]["requests"], 0)


class ReusedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "runs/v41-source"
        raw = config(search={"max_requests": 0}).raw
        raw["experiment_version"] = "4.1"
        raw["runtime"].update(kv_cache_dtype="nvfp4", enforce_eager=True)
        raw["construction"]["min_fp4_gap"] = raw["construction"].pop("min_fp8_gap")
        old = V4Config.from_dict(raw)
        atomic_write_json(self.source / "run-manifest.json", {"config": raw})
        harness = FP4Harness(old, self.root, self.source)
        result = execute_v4(old, self.root, self.source, CharCodec(), sources, harness.runner(), {})
        self.assertNotIn("test", result.get("scores", {}))
        new = json.loads(json.dumps(raw))
        new["experiment_version"] = "4.2"
        new["thresholds"]["min_capacity_ratio"] = 2.0
        new["search"].update(schedule="bounded", medium_candidates=8, full_candidates=6,
                             early_full_candidates=3, max_requests=2000)
        self.cfg = V4Config.from_dict(new)
        self.directory = self.root / "runs/v42-test"
        info, self.folder, construction = describe_source(self.source, self.cfg, CharCodec())
        import_dataset(self.directory, self.cfg, self.source, self.folder, construction, info)
        atomic_write_json(self.directory / "run-manifest.json", {"config": new})

    def test_data_are_byte_identical_and_complete_chain_exports_rescorable_evidence(self):
        for path in self.folder.glob("*.jsonl"):
            target = self.directory / "inputs/data/v4.2/v42-test/quality" / path.name
            self.assertEqual(path.read_bytes(), target.read_bytes())
        harness = FP4Harness(self.cfg, self.root, self.directory)
        result = execute_reused(self.cfg, self.directory, harness.runner())
        self.assertTrue(result["technical_goal_passed"])
        self.assertTrue(result["recovery_demonstrated"])
        self.assertGreaterEqual(result["capacity"]["ratio"], 2.0)
        self.assertEqual(result["candidate"]["bf16_layers"], [2])
        self.assertFalse(any(split in {"discovery", "confirmation"} for _, split, _ in harness.service_calls))
        requests = harness.count
        self.assertEqual(execute_reused(self.cfg, self.directory, harness.runner())["status"], "passed")
        self.assertEqual(harness.count, requests)
        output = export_results(self.root, self.directory.name)
        self.assertFalse(list(output.glob("*.tar*")))
        with zipfile.ZipFile(next(output.glob("*.zip"))) as archive:
            from autokv.v4_metrics import score_output
            samples = {r["sample_id"]: r for r in map(json.loads, archive.read("samples.jsonl").splitlines())}
            answers = list(map(json.loads, archive.read("answers.jsonl").splitlines()))
            self.assertEqual(len(samples), 32)
            self.assertFalse(any("user_prompt" in r or "world" in r for r in samples.values()))
            for row in answers:
                self.assertEqual(score_output(row["output_text"], samples[row["sample_id"]]), row["task_score"])
            meta = json.loads(archive.read("export-summary.json"))
            self.assertFalse(meta["warnings"])
            self.assertEqual(sum(meta["counts"]["answers"].values()), len(answers))

    def test_changed_data_are_not_regenerated_or_silently_reused(self):
        path = self.directory / "inputs/data/v4.2/v42-test/quality/experiment.jsonl"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "数据已改变"):
            load_reused(self.directory, self.cfg)
        self.cfg.raw["data"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "data"):
            describe_source(self.source, self.cfg, CharCodec())

    def test_runtime_resume_completes_partial_full_evaluation_without_duplicate_answers(self):
        harness = FP4Harness(self.cfg, self.root, self.directory)
        harness.fail_at = 90
        with self.assertRaisesRegex(Exception, "模拟 HTTP 中断"):
            execute_reused(self.cfg, self.directory, harness.runner())
        self.assertFalse((self.directory / "selection.json").exists())
        result = execute_reused(self.cfg, self.directory, harness.runner())
        self.assertTrue(result["technical_goal_passed"])
        self.assertEqual(result["selection"]["full_candidates_used"], 1)
        self.assertEqual(len(harness.service_calls), len(set(harness.service_calls)))

    def test_public_entry_snapshots_data_and_resumes_without_source_directory(self):
        harnesses = []
        environment = {"vllm": "/existing/vllm", "model_path": "/existing/model",
                       "model_revision": self.cfg.model_revision, "pythonpath_prefix": "/compat",
                       "flashinfer_cubin_dir": "/cubin", "flashinfer_workspace_base": "/fi-cache",
                       "torch_extensions_dir": "/torch-cache"}
        def factory(cfg, root, directory, env, samples, **kwargs):
            harness = FP4Harness(cfg, root, directory)
            harnesses.append(harness)
            return harness.runner(environment=env)
        with patch("autokv.v42_pipeline.V3PolicyRunner", side_effect=factory):
            result = run_reused(self.root, self.cfg, environment, CharCodec(), reuse_run=str(self.source))
            self.assertTrue(result["technical_goal_passed"])
            run = self.root / "runs" / result["run_id"]
            self.assertTrue(result["run_id"].startswith("v42-"))
            self.assertTrue((run / "inputs/autokv/v42_search.py").exists())
            process_env = harnesses[0].starts[0][1]["env"]
            self.assertTrue(process_env["PYTHONPATH"].startswith("/compat"))
            self.assertEqual(process_env["FLASHINFER_CUBIN_DIR"], "/cubin")
            self.assertEqual(process_env["FLASHINFER_WORKSPACE_BASE"], "/fi-cache")
            self.assertEqual(process_env["TORCH_EXTENSIONS_DIR"], "/torch-cache")
            self.source.rename(self.source.with_name("moved-source"))
            again = run_reused(self.root, self.cfg, environment, CharCodec(), run_id=result["run_id"])
            self.assertEqual(again, json.loads(json.dumps(result)))
            self.assertEqual(harnesses[-1].count, 0)

    def test_test_failure_keeps_one_frozen_candidate(self):
        def score(layers, row):
            if len(layers) == 8:
                return True
            if row["split"] == "test":
                return row["index"] % 2 == 0
            return layers == {2} or row["index"] % 2 == 0
        harness = FP4Harness(self.cfg, self.root, self.directory, score)
        result = execute_reused(self.cfg, self.directory, harness.runner())
        self.assertEqual(result["status"], "test_quality_failed")
        self.assertFalse(result["technical_goal_passed"])
        mixed = {layers for layers, split, _ in harness.service_calls if split == "test" and 0 < len(layers) < 8}
        self.assertEqual(mixed, {frozenset({2})})

    def test_previous_test_use_and_test_failure_never_select_another_candidate(self):
        info_path = self.directory / "inputs/reused-data.json"
        info = read_json(info_path)
        info["test_previously_used"] = True
        atomic_write_json(info_path, info)
        harness = FP4Harness(self.cfg, self.root, self.directory)
        result = execute_reused(self.cfg, self.directory, harness.runner())
        self.assertEqual(result["status"], "test_not_independent")
        self.assertFalse(result["technical_goal_passed"])
        self.assertIn("v41-source", result["test_reused_from"])
        mixed_test = {layers for layers, split, _ in harness.service_calls if split == "test" and 0 < len(layers) < 8}
        self.assertEqual(mixed_test, {frozenset({2})})

    def test_partial_export_preserves_source_and_failure_metadata(self):
        path = self.directory / "policies/partial/experiment.jsonl"
        path.parent.mkdir(parents=True)
        content = b'{"sample_id":"partial", "output_text":"x"}\n{"broken":'
        path.write_bytes(content)
        log = self.directory / "policies/partial/attempts/failure.log"
        log.parent.mkdir()
        log.write_text("start\n"+"padding\n"*3000+"RuntimeError: failed", encoding="utf-8")
        output = export_results(self.root, self.directory.name)
        self.assertEqual(path.read_bytes(), content)
        with zipfile.ZipFile(next(output.glob("*.zip"))) as archive:
            self.assertEqual(len(json.loads(archive.read("export-summary.json"))["warnings"]), 1)
            self.assertIn(b"RuntimeError: failed", archive.read("diagnostics.txt"))
            self.assertLess(len(archive.read("diagnostics.txt")), 9000)


class ExportStartupFailureTests(unittest.TestCase):
    def test_only_cli_failure_without_run_or_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runs").mkdir()
            (root / "runs/v42-cli.log").write_text("ImportError: missing dependency", encoding="utf-8")
            output = export_results(root)
            self.assertIn("ImportError", (output / "CLI-DIAGNOSTICS.txt").read_text(encoding="utf-8"))
            self.assertFalse(list(output.glob("*.zip")))
