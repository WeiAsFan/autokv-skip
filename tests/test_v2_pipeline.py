import json
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

from autokv.config import Profile
from autokv.v2_config import load_v2_config
from autokv.v2_pipeline import (
    V2RunContext,
    _load_v2_lock,
    load_v2_run_context,
    recommend_pilot_difficulty,
    run_v2_pipeline,
    run_v2_random_controls,
)


ROOT = Path(__file__).resolve().parents[1]


def samples(split, counts):
    rows = []
    for tier, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "sample_id": f"{split}-{tier}-{index}",
                    "split": split,
                    "tier": tier,
                    "task": tier,
                }
            )
    return tuple(rows)


class FakePolicyRunner:
    mode = "no_gap"

    def __init__(self, config, profile, project_root, lock, run_id, *, port):
        self.project_root = project_root
        self.run_id = run_id
        self.server_starts = 0
        self.requests = 0

    def _tier_score(self, policy, tier):
        if self.mode == "no_gap":
            return 1.0
        if policy.name == "p32":
            return 1.0
        if policy.name == "p0":
            return 1.0 if tier == "easy" else 0.8
        if policy.name.startswith("group-"):
            group = int(policy.name.split("-")[1])
            return 0.82 + group * 0.01
        if policy.name.startswith("layer-"):
            layer = int(policy.name.split("-")[1])
            return 0.90 + layer / 1000
        if policy.name == "selected-p2":
            return 1.0 if tier == "easy" else 0.995
        if policy.name.startswith("random-"):
            return 0.90
        raise AssertionError(policy.name)

    def run_policy(
        self,
        policy,
        policy_samples,
        *,
        split,
        split_sha256,
        relative_directory,
    ):
        self.server_starts += 1
        self.requests += len(policy_samples)
        directory = self.project_root / "runs" / self.run_id / relative_directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{policy.name}.jsonl"
        rows = [
            {
                "sample_id": sample["sample_id"],
                "tier": sample["tier"],
                "task_score": self._tier_score(policy, sample["tier"]),
                "error": None,
            }
            for sample in policy_samples
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        path.with_name(path.stem + ".policy-manifest.json").write_text(
            json.dumps({"capacity": {"tokens": int(131072 * 64 / (32 + policy.k))}}),
            encoding="utf-8",
        )
        path.with_name(path.stem + ".server.log").write_text(
            "enable_prefix_caching=False\n", encoding="utf-8"
        )
        return path


class V2PipelineTests(unittest.TestCase):
    def test_cli_cold_start_prepares_offline_data_runs_and_exports(self):
        from autokv.cli import main
        from scripts.export_v2_results import export_results
        from types import SimpleNamespace
        import tarfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            shutil.copytree(ROOT / "configs", root / "configs")
            shutil.copytree(ROOT / "data/v2/quality", root / "data/v2/quality")
            shutil.copytree(ROOT / "autokv", root / "autokv", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
            model = root / "existing-model"
            model.mkdir()
            codec = SimpleNamespace(template_sha256="a" * 64, render_and_count=lambda prompt: (prompt, len(prompt.split())))
            FakePolicyRunner.mode = "no_gap"
            stdout = io.StringIO()
            with patch.dict(os.environ, {"AUTOKV_VLLM_BIN": "/existing/bin/vllm", "AUTOKV_MODEL_PATH": str(model)}), patch("autokv.v2_data.TransformersPromptCodec", return_value=codec), patch("autokv.v2_pipeline._require_linux"), patch("autokv.v2_pipeline.V2PolicyRunner", FakePolicyRunner), redirect_stdout(stdout):
                exit_code = main(["v2-run", "--project-root", str(root), "--json"])
            self.assertEqual(exit_code, 0)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["technical_goal_passed"])
            self.assertEqual(result["requests_this_invocation"], 90)
            self.assertTrue((root / "data/v2.1/quality/dataset-manifest.json").is_file())
            exported = export_results(root)
            with tarfile.open(exported / "autokv-v2.tar.gz") as archive:
                self.assertIn(f"autokv-skip/runs/{result['run_id']}/inputs/data/v2.1/quality/heldout.jsonl", archive.getnames())
            self.assertIn("主实验达标", (exported / "README.zh-CN.md").read_text(encoding="utf-8"))

    def test_zero_baseline_stops_with_saved_negative_result(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(FakePolicyRunner, "_tier_score", return_value=0.0):
            root = Path(directory).resolve()
            result = self.run_mode(root, "no_gap")
            self.assertTrue(result["complete"])
            self.assertEqual(result["status"], "invalid_reference")
            self.assertFalse(result["technical_goal_passed"])
            self.assertIsNone(result["final"])
            self.assertEqual(result["server_starts_this_invocation"], 1)
            self.assertTrue((root / result["report_path"]).is_file())

    def test_qualified_group_is_kept_instead_of_selecting_eight_layers(self):
        def score(runner, policy, tier):
            layers = set(policy.bf16_layers)
            if tier == "easy" or policy.k == 32 or layers in (set(range(4)), set(range(8))):
                return 1.0
            if policy.k == 1:
                layer = policy.bf16_layers[0]
                return (0.9 if layer >= 4 else 0.8) + layer / 1000
            return 0.95 if layers == set(range(4, 8)) else 0.8

        with tempfile.TemporaryDirectory() as directory, patch.object(FakePolicyRunner, "_tier_score", score):
            root = Path(directory).resolve()
            result = self.run_mode(root, "gap")
            self.assertEqual(result["candidate"]["bf16_layers"], [0, 1, 2, 3])
            self.assertTrue(result["technical_goal_passed"])
            budgets = root / "runs/v2-test-run/quality/calibration/budgets"
            self.assertEqual([p.stem for p in budgets.glob("*.jsonl")], ["selected-p2"])

    def test_p1_can_win_and_optional_controls_leave_main_result_unchanged(self):
        def score(runner, policy, tier):
            return 1.0 if tier == "easy" or policy.k == 32 or policy.bf16_layers == (0,) else 0.8

        with tempfile.TemporaryDirectory() as directory, patch.object(FakePolicyRunner, "_tier_score", score):
            root = Path(directory).resolve()
            result = self.run_mode(root, "gap")
            self.assertEqual(result["candidate"]["k"], 1)
            self.assertEqual(result["server_starts_this_invocation"], 21)
            run = root / "runs/v2-test-run"
            original = [(run / name).read_bytes() for name in ("selection.json", "completed-manifest.json")]
            with patch("autokv.v2_pipeline.load_v2_run_context", return_value=self.context(root)), patch("autokv.v2_pipeline._require_linux"), patch("autokv.v2_pipeline.V2PolicyRunner", FakePolicyRunner):
                controls = run_v2_random_controls(root)
            self.assertEqual(controls["server_starts_this_invocation"], 3)
            self.assertEqual(controls["requests_this_invocation"], 54)
            self.assertNotIn("layer_selection_supported", controls)
            self.assertEqual(original, [(run / name).read_bytes() for name in ("selection.json", "completed-manifest.json")])
            with patch("autokv.v2_pipeline.load_v2_run_context", return_value=self.context(root)), patch("autokv.v2_pipeline._require_linux"), patch("autokv.v2_pipeline.V2PolicyRunner", FakePolicyRunner), patch.object(FakePolicyRunner, "run_policy", side_effect=RuntimeError("optional failure")):
                with self.assertRaises(RuntimeError):
                    run_v2_random_controls(root)
            self.assertEqual(original, [(run / name).read_bytes() for name in ("selection.json", "completed-manifest.json")])
            self.assertFalse(json.loads((run / "random-controls.json").read_text())["complete"])

    def test_duplicate_layer_set_is_not_evaluated_twice_and_fallback_is_not_success(self):
        def score(runner, policy, tier):
            return 1.0 if tier == "easy" or policy.k == 32 else 0.8

        calls = []
        original = FakePolicyRunner.run_policy
        def capture(runner, policy, policy_samples, **kwargs):
            calls.append((kwargs["split"], policy.config_id))
            return original(runner, policy, policy_samples, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(FakePolicyRunner, "_tier_score", score), patch.object(FakePolicyRunner, "run_policy", capture):
            result = self.run_mode(Path(directory).resolve(), "gap")
            self.assertEqual(result["final"]["k"], 32)
            self.assertEqual(result["status"], "no_qualifying_mixed_policy")
            self.assertFalse(result["technical_goal_passed"])
            self.assertEqual(len(calls), len(set(calls)))

    def test_capacity_is_measured_and_missing_capacity_keeps_quality_result(self):
        original = FakePolicyRunner.run_policy
        for measured, status in ((None, "capacity_unverified"), (190000, "capacity_below_target"), (196608, "passed")):
            with self.subTest(measured=measured), tempfile.TemporaryDirectory() as directory:
                def run(runner, policy, policy_samples, **kwargs):
                    path = original(runner, policy, policy_samples, **kwargs)
                    if policy.k == 0:
                        path.with_name(path.stem + ".policy-manifest.json").write_text(json.dumps({"capacity": {"tokens": measured}}))
                    return path
                with patch.object(FakePolicyRunner, "run_policy", run):
                    result = self.run_mode(Path(directory).resolve(), "no_gap")
                self.assertEqual(result["status"], status)
                self.assertTrue(result["quality_passed"])
                self.assertEqual(result["technical_goal_passed"], status == "passed")

    def test_heldout_failure_and_invalid_reference_do_not_reselect(self):
        original = FakePolicyRunner.run_policy
        for failed_k, status in ((0, "heldout_failed"), (32, "invalid_reference")):
            with self.subTest(failed_k=failed_k), tempfile.TemporaryDirectory() as directory:
                def run(runner, policy, policy_samples, **kwargs):
                    path = original(runner, policy, policy_samples, **kwargs)
                    if kwargs["split"] == "heldout" and policy.k == failed_k:
                        rows = [json.loads(line) for line in path.read_text().splitlines()]
                        for row in rows:
                            row["task_score"] = 0.0
                        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    return path
                root = Path(directory).resolve()
                with patch.object(FakePolicyRunner, "run_policy", run):
                    result = self.run_mode(root, "no_gap")
                self.assertEqual(result["status"], status)
                self.assertFalse(result["technical_goal_passed"])
                self.assertEqual(result["candidate"]["k"], 0)
                self.assertFalse((root / "runs/v2-test-run/quality/calibration/groups").exists())
                self.assertEqual(result["final"]["k"] if result["final"] else None, 32 if failed_k == 0 else None)

    def context(self, root):
        config_path = ROOT / "configs/v2.1/quality.json"
        config = load_v2_config(config_path)
        return V2RunContext(
            root=root,
            config_path=config_path,
            config=config,
            profile=Profile.from_dict(Profile.default_dict("full")),
            lock={
                "backend": "docker",
                "image_digest": "sha256:" + "f" * 64,
                "model_revision": config.model_revision,
            },
            dataset_manifest={
                "dataset_sha256": "d" * 64,
                "splits": {
                    "calibration": {"sha256": "a" * 64},
                    "heldout": {"sha256": "b" * 64},
                },
            },
            calibration=samples("calibration", {"easy": 3, "hard": 18, "natural": 6}),
            heldout=samples("heldout", {"easy": 3, "hard": 9, "natural": 6}),
            source={
                "git_commit": "c" * 40,
                "tree_sha256": "e" * 64,
                "files": [],
            },
            run_id="v2-test-run",
        )

    def run_mode(self, root, mode):
        FakePolicyRunner.mode = mode
        context = self.context(root)
        with (
            patch("autokv.v2_pipeline.load_v2_run_context", return_value=context),
            patch("autokv.v2_pipeline._require_linux"),
            patch("autokv.v2_pipeline.V2PolicyRunner", FakePolicyRunner),
        ):
            return run_v2_pipeline(root, port=8000)

    def test_pilot_recommendation_is_preregistered(self):
        config = load_v2_config(ROOT / "configs/v2/quality.json")
        names = config.hard_families

        self.assertEqual(
            recommend_pilot_difficulty(dict.fromkeys(names, 0.99), config)[0],
            "hard",
        )
        self.assertEqual(
            recommend_pilot_difficulty(
                {names[0]: 0.59, names[1]: 0.9, names[2]: 0.9}, config
            )[0],
            "easy",
        )
        self.assertEqual(
            recommend_pilot_difficulty(dict.fromkeys(names, 0.8), config)[0],
            "standard",
        )

    def test_offline_source_copy_runs_and_resumes_across_git_metadata_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            shutil.copytree(ROOT / "configs", root / "configs")
            shutil.copytree(ROOT / "data/v2/quality", root / "data/v2/quality")
            from types import SimpleNamespace
            from autokv.v2_data import upgrade_v2_dataset
            upgrade_v2_dataset(
                load_v2_config(root / "configs/v2.1/quality.json"),
                SimpleNamespace(template_sha256="a" * 64, render_and_count=lambda prompt: (prompt, len(prompt.split()))),
                root / "data/v2/quality", root / "data/v2.1/quality",
                config_path=root / "configs/v2.1/quality.json", legacy_config_path=root / "configs/v2/quality.json",
            )
            shutil.copytree(ROOT / "autokv", root / "autokv", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
            model = root / "existing-model"
            model.mkdir()
            with (
                patch.dict(os.environ, {"AUTOKV_VLLM_BIN": "/old-project/venv/bin/vllm", "AUTOKV_MODEL_PATH": str(model)}),
                patch("autokv.cli.run_command", side_effect=FileNotFoundError("git")),
            ):
                first = load_v2_run_context(root)
                self.assertIsNone(first.source["git_commit"])
                run_root = root / "runs" / first.run_id
                self.assertTrue((run_root / "inputs/autokv/v2_pipeline.py").is_file())
                self.assertTrue((run_root / "inputs/data/v2.1/quality/heldout.jsonl").is_file())
                with patch("autokv.v2_pipeline._source_identity", return_value={
                    **first.source, "git_commit": "a" * 40, "git_dirty": True,
                }):
                    resumed = load_v2_run_context(root)
                self.assertEqual(resumed.run_id, first.run_id)
                provenance = json.loads((run_root / "run-manifest.json").read_text(encoding="utf-8"))
                self.assertIsNone(provenance["git_commit"])
                with (root / "autokv/v2_pipeline.py").open("a", encoding="utf-8") as stream:
                    stream.write("\n# 修改运行代码后必须使用新运行目录\n")
                changed = load_v2_run_context(root)
                self.assertNotEqual(changed.run_id, first.run_id)

    def test_legacy_lock_needs_only_paths_and_path_changes_separate_runs(self):
        config = load_v2_config(ROOT / "configs/v2/quality.json")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory).resolve()
            model = root / "model"
            model.mkdir()
            path = root / "runs/_environment/lock.json"
            path.parent.mkdir(parents=True)
            lock = {
                "backend": "local_vllm", "model_path": str(model),
                "vllm": "/old/bin/vllm", "runtime_id": "legacy-id",
                "host": {"driver": "different-driver"},
            }
            path.write_text(json.dumps(lock))
            first = _load_v2_lock(root, config)
            lock["vllm"] = "/other/bin/vllm"
            path.write_text(json.dumps(lock))
            second = _load_v2_lock(root, config)
            self.assertNotEqual(first["runtime_id"], second["runtime_id"])

    def test_failed_resume_removes_stale_completed_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.run_mode(root, "no_gap")
            marker = root / "runs/v2-test-run/completed-manifest.json"
            self.assertTrue(marker.is_file())
            with patch.object(FakePolicyRunner, "run_policy", side_effect=RuntimeError("failed")):
                with self.assertRaises(RuntimeError):
                    self.run_mode(root, "no_gap")
            self.assertFalse(marker.exists())

    def test_no_gap_path_stops_without_layer_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_mode(root, "no_gap")

            self.assertEqual(result["decision"], "no_quality_gap")
            self.assertEqual(result["final"]["k"], 0)
            self.assertEqual(result["server_starts_this_invocation"], 4)
            self.assertEqual(result["requests_this_invocation"], 90)
            self.assertFalse(
                (root / "runs/v2-test-run/quality/calibration/groups").exists()
            )

    def test_gap_path_selects_p2_without_mandatory_random_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_mode(root, "gap")
            selection = json.loads(
                (root / "runs/v2-test-run/selection.json").read_text(encoding="utf-8")
            )

            self.assertEqual(result["decision"], "search_required")
            self.assertEqual(result["candidate"]["k"], 2)
            self.assertEqual(result["final"]["k"], 2)
            self.assertEqual(result["server_starts_this_invocation"], 22)
            self.assertEqual(result["requests_this_invocation"], 567)
            self.assertEqual(len(selection["group_ranking"]), 8)
            self.assertEqual(len(selection["layer_ranking"]), 8)
            self.assertEqual(len(selection["budget_trace"]), 1)
            self.assertTrue(result["technical_goal_passed"])
            self.assertNotIn("layer_selection_supported", selection)
            self.assertIn("heldout_P32-candidate", selection["paired_differences"])


if __name__ == "__main__":
    unittest.main()
