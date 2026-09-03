import copy
import unittest
from pathlib import Path

from autokv.v2_config import load_v2_config, V2QualityConfig


ROOT = Path(__file__).resolve().parents[1]


class V2ConfigTests(unittest.TestCase):
    def test_loads_the_only_v2_quality_configuration(self):
        config = load_v2_config(ROOT / "configs/v2/quality.json")

        self.assertEqual(config.target_lengths, (8192, 16384, 24576))
        self.assertEqual(config.candidate_budgets, (0, 2, 4, 8, 32))
        self.assertFalse(config.enable_prefix_caching)
        self.assertEqual(config.hard_calibration_seeds, (41, 42))
        self.assertEqual(config.hard_heldout_seeds, (43,))

    def test_rejects_changes_to_frozen_identity_or_search_space(self):
        config = load_v2_config(ROOT / "configs/v2/quality.json")
        for mutate in (
            lambda value: value["runtime"].__setitem__("enable_prefix_caching", True),
            lambda value: value["selection"].__setitem__(
                "candidate_budgets", [0, 2, 4, 8, 16, 32]
            ),
            lambda value: value["model"].__setitem__("revision", "a" * 40),
            lambda value: value["data"]["natural"].__setitem__(
                "source_length_buckets", [4096, 8192, None]
            ),
            lambda value: value["runtime"].__setitem__("calculate_kv_scales", True),
        ):
            value = copy.deepcopy(config.raw)
            mutate(value)
            with self.assertRaises(ValueError):
                V2QualityConfig.from_dict(value)


if __name__ == "__main__":
    unittest.main()
