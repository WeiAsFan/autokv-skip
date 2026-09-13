import copy
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autokv.cli import main
from autokv.client import VllmHttpError
from autokv.io import atomic_write_json, atomic_write_text, read_json, read_jsonl
from autokv.v2_policy import endpoint_policies
from autokv.v4_config import CONFIG_PATH
from autokv.v4_construction import choose_confirmation, choose_rule
from autokv.v4_pipeline import execute_v4, run_pipeline
from scripts.export_v4_results import export_results
from tests.v4_helpers import CharCodec, Harness, config, sources


class V4PipelineTests(unittest.TestCase):
    def make_case(self, score=None, cfg=None, name="v4-test"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root, cfg = Path(temporary.name), cfg or config()
        directory = root / "runs" / name
        h = Harness(cfg, root, directory, score)
        return root, cfg, directory, h

    def run_case(self, case, **kwargs):
        root, cfg, directory, h = case
        return execute_v4(cfg, root, directory, CharCodec(), sources, h.runner(), {}, **kwargs)

    def test_full_chain_uses_native_runner_scoring_capacity_and_export(self):
        case = self.make_case()
        root, cfg, directory, h = case
        result = self.run_case(case)
        self.assertTrue(result["complete"])
        self.assertTrue(result["construction_passed"])
        self.assertTrue(result["technical_goal_passed"])
        self.assertTrue(result["data_conditions_reproduced"])
        self.assertTrue(result["recovery_demonstrated"])
        self.assertEqual(result["candidate"]["bf16_layers"], [2])
        self.assertEqual(result["cost"]["phases"]["discovery"]["requests"], 12)
        self.assertEqual(result["cost"]["phases"]["confirmation"]["requests"], 40)
        for phase in ("discovery", "confirmation"):
            self.assertEqual(result["cost"]["phases"][phase]["server_starts"], 2)
        all_rows = [read_jsonl(directory/"construction"/f"{phase}.jsonl") for phase in ("discovery", "confirmation")]
        data_dir = root/f"data/v4.0/{directory.name}/quality"
        all_rows += [read_jsonl(data_dir/f"{phase}.jsonl") for phase in ("experiment", "test")]
        self.assertEqual(len({r["base_instance_id"] for group in all_rows for r in group}), 58)
        report = Path(result["report"]).read_text(encoding="utf-8")
        self.assertIn("v4.0", report)
        self.assertIn("single_lookup", report)
        self.assertNotIn("multi_value", report)
        self.assertNotIn("1280", report)
        exported = export_results(root, directory.name)
        with tarfile.open(exported/"autokv-v4.tar.gz") as archive:
            names = archive.getnames()
            self.assertTrue(any(n.endswith("construction/confirmation.jsonl") for n in names))
            self.assertTrue(any(n.endswith("selected-rule.json") for n in names))
            self.assertTrue(any(n.endswith("quality/test.jsonl") for n in names))
        calls = h.count
        self.assertEqual(self.run_case(case)["status"], result["status"])
        self.assertEqual(h.count, calls)

    def test_confirmation_failure_is_complete_negative_without_search_or_test(self):
        def score(layers, row):
            if row["split"] == "confirmation" and row["batch_id"] == "batch-2":
                return row["index"] < (7 if len(layers) == 8 else 1)
            return len(layers) == 8 or row["index"] % 2 == 0
        case = self.make_case(score)
        _, _, directory, h = case
        result = self.run_case(case)
        self.assertEqual(result["status"], "construction_not_found")
        self.assertTrue(result["complete"])
        self.assertFalse(result["construction_passed"])
        self.assertIsNone(result["data_conditions_reproduced"])
        self.assertFalse((directory/"selection.json").exists())
        self.assertTrue(all(split in ("discovery", "confirmation") for _, split, _ in h.service_calls))
        count = h.count
        self.run_case(case)
        self.assertEqual(h.count, count)

    def test_all_structurally_impossible_never_starts_service(self):
        case = self.make_case(cfg=config(construction={"record_counts": [256]}))
        result = self.run_case(case)
        self.assertEqual(result["status"], "construction_not_found")
        self.assertEqual(case[-1].count, 0)
        self.assertEqual(case[-1].starts, [])

    def test_discovery_grid_and_confirmation_batches_are_batched_by_precision(self):
        cfg = config(construction={"tasks": ["single_lookup", "two_hop_lookup"], "target_lengths": [1600, 2200]})
        case = self.make_case(cfg=cfg)
        result = self.run_case(case, construction_only=True)
        self.assertTrue(result["construction_passed"])
        self.assertEqual(result["cost"]["total"]["server_starts"], 4)
        self.assertEqual(result["cost"]["total"]["requests"], 4*6*2+2*2*10*2)
        confirm = read_jsonl(case[2]/"construction/confirmation.jsonl")
        self.assertEqual(len({r["condition_id"] for r in confirm}), 2)
        self.assertEqual(len({r["base_instance_id"] for r in confirm}), 40)

    def test_only_construction_resumes_without_resampling(self):
        case = self.make_case()
        result = self.run_case(case, construction_only=True)
        self.assertEqual(result["stage"], "construction")
        self.assertTrue(result["complete"])
        directory, h = case[2:]
        old = (directory/"construction/confirmation.jsonl").read_bytes()
        old_rule = (directory/"construction/selected-rule.json").read_bytes()
        before = h.count
        result = self.run_case(case)
        self.assertTrue(result["recovery_demonstrated"])
        self.assertEqual((directory/"construction/confirmation.jsonl").read_bytes(), old)
        self.assertEqual((directory/"construction/selected-rule.json").read_bytes(), old_rule)
        self.assertTrue(all(s in ("experiment", "test") for _, s, _ in h.service_calls[before:]))

    def test_interrupted_confirmation_recovers_only_missing_rows(self):
        case = self.make_case()
        _, _, directory, h = case
        h.fail_at = 17
        with self.assertRaises(VllmHttpError):
            self.run_case(case)
        partial = read_json(directory/"completed-manifest.json")
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["status"], "runtime_failed")
        original = (directory/"construction/confirmation.jsonl").read_bytes()
        result = self.run_case(case, construction_only=True)
        self.assertTrue(result["construction_passed"])
        self.assertEqual(h.count, 53)
        self.assertEqual(len(h.service_calls), 52)
        self.assertEqual(len(set(h.service_calls)), 52)
        self.assertEqual((directory/"construction/confirmation.jsonl").read_bytes(), original)

    def test_formal_p0_early_stop_is_preserved(self):
        def score(layers, row):
            return row["split"] in ("experiment", "test") or len(layers) == 8 or row["index"] % 2 == 0
        case = self.make_case(score)
        result = self.run_case(case)
        self.assertTrue(result["construction_passed"])
        self.assertTrue(result["technical_goal_passed"])
        self.assertEqual(result["candidate"]["bf16_layers"], [])
        self.assertFalse(result["data_conditions_reproduced"])
        self.assertFalse(result["recovery_demonstrated"])
        self.assertEqual(result["cost"]["phases"]["search"]["requests"], 0)

    def test_quality_success_does_not_imply_strong_gap_reproduced(self):
        def score(layers, row):
            return len(layers) == 8 or layers == {2} or (row["index"] != 0 if row["split"] == "test" else row["index"] % 2 == 0)
        result = self.run_case(self.make_case(score))
        self.assertTrue(result["technical_goal_passed"])
        self.assertFalse(result["data_conditions_reproduced"])
        self.assertFalse(result["recovery_demonstrated"])
        self.assertAlmostEqual(result["test_data_conditions"]["gap"], .05)
        self.assertIsNone(result["intervals"]["selected"]["intervals"])

    def test_search_budget_exit_has_no_fake_test(self):
        case = self.make_case(cfg=config(search={"max_requests": 1}))
        result = self.run_case(case)
        self.assertEqual(result["status"], "no_feasible_within_budget")
        self.assertTrue(result["complete"])
        self.assertIsNone(result["data_conditions_reproduced"])
        self.assertEqual(result["cost"]["phases"]["search"]["requests"], 1)
        self.assertFalse(any(split == "test" for _, split, _ in case[-1].service_calls))

    def test_test_interruption_and_failure_never_reselect(self):
        case = self.make_case()
        self.run_case(case, construction_only=True)
        h, directory = case[-1], case[2]
        old_complete = h.complete
        def fail_test(prompt, max_tokens):
            if h.lookup[prompt]["split"] == "test":
                raise VllmHttpError(400, "测试中断")
            return old_complete(prompt, max_tokens)
        h.client.chat_complete = fail_test
        with self.assertRaises(VllmHttpError):
            self.run_case(case)
        frozen = (directory/"selection.json").read_bytes()
        h.client.chat_complete = old_complete
        h.score = lambda layers, row: len(layers) == 8
        with patch("autokv.v3_pipeline.search", side_effect=AssertionError("测试不能回流选层")):
            result = self.run_case(case)
        self.assertEqual(result["status"], "test_quality_failed")
        self.assertEqual((directory/"selection.json").read_bytes(), frozen)
        self.assertFalse(result["recovery_demonstrated"])

    def test_missing_capacity_keeps_quality_but_not_goal(self):
        case = self.make_case()
        case[-1].capacity = lambda _: None
        result = self.run_case(case)
        self.assertEqual(result["status"], "capacity_unverified")
        self.assertTrue(result["quality_passed"])
        self.assertFalse(result["technical_goal_passed"])

    def test_report_failure_cannot_leave_recovery_success_flag(self):
        case = self.make_case()
        with patch("autokv.v3_pipeline.coverage", side_effect=ValueError("统计写入失败")):
            result = self.run_case(case)
        self.assertFalse(result["complete"])
        self.assertFalse(result["technical_goal_passed"])
        self.assertFalse(result["recovery_demonstrated"])
        self.assertEqual(result["status"], "runtime_failed")
        self.assertTrue((case[2]/"selection.json").exists())

    def test_two_hop_formal_protocol_can_recover_noncontiguous_layers(self):
        def score(layers, row):
            if len(layers) == 8 or {1, 5} <= layers:
                return True
            if layers == {1}:
                return row["index"] % 6 < 5
            if layers == {5}:
                return row["index"] % 6 < 4
            return row["index"] % 2 == 0
        cfg = config(construction={"tasks": ["two_hop_lookup"], "target_lengths": [2200]})
        case = self.make_case(score, cfg)
        result = self.run_case(case)
        self.assertEqual(result["candidate"]["bf16_layers"], [1, 5])
        self.assertTrue(result["recovery_demonstrated"])
        self.assertEqual(set(result["scores"]["test"]["P32"]["cells"]), {"two_hop_lookup:2200"})

    def test_runtime_recomputes_cached_v4_score_and_rejects_changed_sample(self):
        case = self.make_case()
        self.run_case(case, construction_only=True)
        root, cfg, directory, h = case
        _, p0 = endpoint_policies(cfg.num_layers)
        path = directory/"policies"/p0.config_id/"discovery.jsonl"
        rows = read_jsonl(path)
        for r in rows:
            r["task_score"] = .123
        import json
        atomic_write_text(path, "".join(json.dumps(r)+"\n" for r in rows))
        runner = h.runner()
        inputs = read_jsonl(directory/"construction/discovery.jsonl")
        runner.samples["discovery"] = {r["sample_id"]: r for r in inputs}
        before = h.count
        cached = runner.evaluate(p0, "discovery", [r["sample_id"] for r in inputs])
        self.assertEqual(h.count, before)
        self.assertEqual([r["task_score"] for r in cached], [1, 0, 1, 0, 1, 0])
        changed = h.runner()
        inputs[0]["user_prompt"] += " changed"
        changed.samples["discovery"] = {r["sample_id"]: r for r in inputs}
        with self.assertRaisesRegex(ValueError, "缓存"):
            changed.evaluate(p0, "discovery", [r["sample_id"] for r in inputs])

    def test_cross_version_used_test_is_marked_and_same_run_can_resume(self):
        case = self.make_case()
        root, _, directory, _ = case
        first = self.run_case(case)
        prior = root/"runs/v3-prior"
        atomic_write_json(prior/"test-use.json", read_json(directory/"test-use.json"))
        atomic_write_text(prior/"policies/fp8/test.jsonl", '{"sample_id":"prior"}\n')
        self.assertEqual(self.run_case(case)["status"], first["status"])
        newdir = root/"runs/v4-new"
        h = Harness(case[1], root, newdir)
        result = self.run_case((root, case[1], newdir, h))
        self.assertEqual(result["status"], "test_not_independent")
        self.assertIn("v3-prior", result["test_reused_from"])
        self.assertFalse(result["technical_goal_passed"])

    def test_explicit_run_uses_saved_config_without_git_or_current_config_override(self):
        cfg, codec = config(), CharCodec()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_write_json(root/CONFIG_PATH, cfg.raw)
            h = Harness(cfg, root, root/"runs/unused")
            def runner(c, r, directory, environment, samples, **kwargs):
                h.directory = directory
                return h.runner(c, environment)
            with patch("autokv.v4_pipeline.sys.platform", "linux"), \
                 patch("autokv.v4_pipeline.load_environment", return_value=h.environment), \
                 patch("autokv.v4_pipeline.TransformersPromptCodec", return_value=codec), \
                 patch("autokv.v4_pipeline.source_inputs", return_value={}), \
                 patch("autokv.v4_pipeline.source_pool", side_effect=lambda *_: sources()), \
                 patch("autokv.v4_pipeline.V3PolicyRunner", side_effect=runner):
                first = run_pipeline(root, construction_only=True)
                original = h.count
                changed = copy.deepcopy(cfg.raw)
                changed["construction"]["min_fp8_gap"] = .20
                atomic_write_json(root/CONFIG_PATH, changed)
                result = run_pipeline(root, run_id=first["run_id"])
                self.assertEqual(first["run_id"], result["run_id"])
                self.assertTrue(result["recovery_demonstrated"])
                self.assertTrue(all(split in ("experiment", "test") for _, split, _ in h.service_calls[original:]))
                self.assertEqual(h.active.config.raw["construction"]["min_fp8_gap"], .10)
                with patch("autokv.v4_pipeline.actual_code", return_value={"changed.py": "changed"}), self.assertRaises(ValueError):
                    run_pipeline(root, run_id=first["run_id"])
            self.assertFalse((root/".git").exists())
            self.assertTrue((h.directory/"inputs/autokv/v4_pipeline.py").exists())

    def test_cli_dispatch_does_not_call_historical_prerequisites(self):
        with patch("autokv.v4_pipeline.run_pipeline", return_value={"complete": True, "status": "construction_not_found"}) as run, \
             patch("autokv.cli._gpu_context", side_effect=AssertionError("不能调用历史门禁")), \
             patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(main(["v4-run", "--construction-only", "--run-id", "v4-existing", "--json"]), 0)
            self.assertTrue(run.call_args.kwargs["construction_only"])
            self.assertEqual(run.call_args.kwargs["run_id"], "v4-existing")


class V4SelectionTests(unittest.TestCase):
    def row(self, name, bf16, gap, length=1600, records=4):
        return {"condition": {"condition_id": name, "task": "single_lookup", "length_bucket": length, "record_count": records},
                "discovery": {"bf16_solvable": bf16 >= .8, "bf16_score": bf16, "gap": gap}, "confirmation": []}

    def test_confirmation_selection_and_bf16_fallback(self):
        a, b, c = self.row("a", .9, .02), self.row("b", .8, .15), self.row("c", .7, .6)
        self.assertEqual(choose_confirmation([a, b, c], 2), [b["condition"], a["condition"]])
        self.assertEqual(choose_confirmation([a, c], 2), [a["condition"]])
        d = self.row("d", .6, .7)
        self.assertEqual(choose_confirmation([c, d], 2), [c["condition"], d["condition"]])

    def test_final_rule_prefers_bf16_then_cost_after_both_batches_pass(self):
        a, b, c = self.row("a", 1, .8), self.row("b", 1, .2, length=2200), self.row("c", 1, .7)
        a["confirmation"] = [{"passed": True, "bf16_score": .85}]*2
        b["confirmation"] = [{"passed": True, "bf16_score": .95}]*2
        c["confirmation"] = [{"passed": True, "bf16_score": 1}, {"passed": False, "bf16_score": 1}]
        self.assertEqual(choose_rule([a, b, c], config()), b)


class V4ExportTests(unittest.TestCase):
    def test_negative_partial_cli_only_and_chunked_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_write_text(root/"runs/v4-cli-failed.log", "模型路径无效")
            out = export_results(root)
            self.assertTrue((out/"autokv-v4.tar.gz").exists())
            run = root/"runs/v4-partial"
            atomic_write_text(run/"construction/discovery.jsonl", '{"sample_id":"partial"}\n')
            atomic_write_text(run/"inputs/source-manifest.json", '{}\n')
            atomic_write_text(run/"policies/p0/attempts/failure.log", "实际失败日志")
            atomic_write_text(root/"data/v4.0/source/unused.json", "整个未使用来源不导出")
            atomic_write_text(root/".cache/model/weights.bin", "权重不导出")
            atomic_write_json(run/"completed-manifest.json", {"status": "construction_not_found", "technical_goal_passed": False})
            out = export_results(root, run.name, part_bytes=100)
            parts = sorted(out.glob("*.part-*"))
            self.assertGreater(len(parts), 1)
            with tarfile.open(fileobj=io.BytesIO(b"".join(p.read_bytes() for p in parts))) as archive:
                names = archive.getnames()
                self.assertTrue(any(n.endswith("construction/discovery.jsonl") for n in names))
                self.assertTrue(any(n.endswith("failure.log") for n in names))
                self.assertFalse(any("unused.json" in n or "weights.bin" in n for n in names))
            self.assertIn("v4.0", (out/"README.zh-CN.md").read_text(encoding="utf-8"))
