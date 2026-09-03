import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autokv.config import Profile
from autokv.v2_config import load_v2_config
from autokv.v2_pipeline import (
    V2RunContext,
    recommend_pilot_difficulty,
    run_v2_pipeline,
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
            json.dumps({"capacity": {"tokens": 100000 + policy.k * 1000}}),
            encoding="utf-8",
        )
        path.with_name(path.stem + ".server.log").write_text(
            "enable_prefix_caching=False\n", encoding="utf-8"
        )
        return path


class V2PipelineTests(unittest.TestCase):
    def context(self, root):
        config_path = ROOT / "configs/v2/quality.json"
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
            patch("autokv.v2_pipeline._assert_linux_a6000"),
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

    def test_gap_path_selects_p2_and_runs_three_random_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_mode(root, "gap")
            selection = json.loads(
                (root / "runs/v2-test-run/selection.json").read_text(encoding="utf-8")
            )

            self.assertEqual(result["decision"], "search_required")
            self.assertEqual(result["candidate"]["k"], 2)
            self.assertEqual(result["final"]["k"], 2)
            self.assertEqual(result["server_starts_this_invocation"], 25)
            self.assertEqual(result["requests_this_invocation"], 621)
            self.assertEqual(len(selection["group_ranking"]), 8)
            self.assertEqual(len(selection["layer_ranking"]), 8)
            self.assertEqual(len(selection["budget_trace"]), 1)
            self.assertTrue(selection["layer_selection_supported"])


if __name__ == "__main__":
    unittest.main()
