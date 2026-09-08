import copy
import unittest

from autokv.cli import build_parser
from autokv.v3_config import V3Config
from tests.v3_helpers import config


class V3ConfigTests(unittest.TestCase):
    def test_capacity_bound_and_explicit_zero(self):
        self.assertEqual(config(False).max_layers, 10)
        self.assertEqual(config(False, thresholds={"min_capacity_ratio": 1.9}).max_layers, 1)
        self.assertEqual(config(search={"max_bf16_layers": 0}).max_layers, 0)

    def test_invalid_lengths_prefixes_and_scale(self):
        cfg = config()
        for section, key, value in (("search", "fidelity_samples", [16, 24, 48]),
                                    ("data", "max_tokens", 4000),
                                    ("runtime", "enable_prefix_caching", True),
                                    ("runtime", "calculate_kv_scales", True)):
            raw = copy.deepcopy(cfg.raw)
            raw[section][key] = value
            with self.assertRaises(ValueError):
                V3Config.from_dict(raw)

    def test_cli_independent_from_old_gates(self):
        parser = build_parser()
        args = parser.parse_args(["v3-run", "--development"])
        self.assertEqual(args.port, 8010)
        self.assertTrue(args.development)
        args = parser.parse_args(["v3-make-data", "--mode", "formal"])
        self.assertEqual(args.mode, "formal")
