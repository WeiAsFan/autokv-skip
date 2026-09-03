import json
import tempfile
import unittest
from pathlib import Path

from autokv.config import Profile, load_profile


class ProfileTests(unittest.TestCase):
    def test_rejects_non_locked_driver(self):
        data = Profile.default_dict("quick")
        data["hardware"]["driver"] = "550.0"
        with self.assertRaisesRegex(ValueError, "580.173.02"):
            Profile.from_dict(data)

    def test_accepts_approved_quick_profile(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        self.assertEqual(profile.model.num_layers, 32)
        self.assertEqual(profile.selection.k, 4)
        self.assertEqual(profile.images[0], "vllm/vllm-openai:v0.26.0")

    def test_rejects_any_unapproved_profile_change(self):
        changes = (
            ("seed", lambda data: data.__setitem__("seed", 7)),
            (
                "benchmark.num_prompts",
                lambda data: data["benchmark"].__setitem__("num_prompts", 99),
            ),
            (
                "quality.final_seeds",
                lambda data: data["quality"].__setitem__("final_seeds", [999]),
            ),
            (
                "calculate_kv_scales",
                lambda data: data.__setitem__("calculate_kv_scales", True),
            ),
        )
        for label, mutate in changes:
            with self.subTest(label=label):
                data = Profile.default_dict("quick")
                mutate(data)
                with self.assertRaisesRegex(ValueError, "approved quick profile"):
                    Profile.from_dict(data)

    def test_loads_profile_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quick.json"
            path.write_text(
                json.dumps(Profile.default_dict("quick")), encoding="utf-8"
            )
            self.assertEqual(load_profile(path).name, "quick")


if __name__ == "__main__":
    unittest.main()
