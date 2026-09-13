import unittest

from autokv.v3_metrics import aggregate, comparison, reference_valid
from autokv.v4_config import SCORE_VERSION
from autokv.v4_metrics import confirmation_passed, paired_summary, score_output
from tests.v4_helpers import config


def rows(n, successes, *, output="123456"):
    return [{"sample_id": str(i), "source_group_id": str(i), "split": "confirmation", "task": "single_lookup",
             "length_bucket": 1600, "task_score": int(i < successes), "output_text": output, "finish_reason": "stop"} for i in range(n)]


class V4MetricsTests(unittest.TestCase):
    def test_short_answer_semantics(self):
        sample = {"task": "single_lookup", "expected_answers": ["123456"], "scorer": SCORE_VERSION}
        cases = {"123456": 1, 'The code is 123456.': 1, '{"code":"123456"}': 1,
                 "123456, 123456": 1, "123456 or 654321": 0, "654321": 0, "": 0,
                 "0123456": 0, "1234567": 0, "ID123456": 0, "123456abc": 0, "１２３４５６": 0}
        for output, expected in cases.items():
            with self.subTest(output=output):
                self.assertEqual(score_output(output, sample), expected)

    def test_strict_gap_boundary_and_two_batches(self):
        cfg = config()
        at = paired_summary(rows(100, 80), rows(100, 70), cfg)
        above = paired_summary(rows(100, 80), rows(100, 69), cfg)
        self.assertTrue(at["bf16_solvable"])
        self.assertFalse(at["passed"])
        self.assertTrue(above["passed"])
        self.assertFalse(confirmation_passed([at, above], cfg))
        self.assertFalse(confirmation_passed([above], cfg))
        self.assertTrue(confirmation_passed([above, above], cfg))
        self.assertFalse(paired_summary(rows(100, 79), rows(100, 0), cfg)["passed"])
        strong = config(construction={"min_fp8_gap": .20})
        self.assertFalse(paired_summary(rows(100, 90), rows(100, 70), strong)["passed"])
        self.assertTrue(paired_summary(rows(100, 90), rows(100, 69), strong)["passed"])
        self.assertEqual(cfg.epsilons, strong.epsilons)
        self.assertEqual(cfg.max_layers, strong.max_layers)

    def test_degenerate_interval_does_not_claim_zero_uncertainty(self):
        result = paired_summary(rows(10, 10), rows(10, 10), config())
        self.assertIsNone(result["gap_interval95"])
        self.assertEqual(result["gap_interval_status"], "degenerate")
        self.assertLess(result["bf16_interval95"][0], 1)

    def test_pairing_and_http_failure(self):
        ref, cand = rows(10, 9), rows(10, 5)
        self.assertTrue(paired_summary(ref, list(reversed(cand)), config())["passed"])
        with self.assertRaises(ValueError):
            paired_summary(ref, cand[:-1], config())
        cand[0]["error"] = "HTTP failed"
        with self.assertRaises(ValueError):
            paired_summary(ref, cand, config())

    def test_single_cell_protocol_keeps_formal_tolerances(self):
        cfg = config().for_condition(config().conditions()[0])
        self.assertEqual(aggregate(rows(100, 90), cfg)["all"], .9)
        self.assertTrue(reference_valid(rows(100, 90), cfg))
        self.assertFalse(reference_valid(rows(100, 0), cfg))
        self.assertFalse(comparison(rows(100, 90), rows(100, 88), cfg)["passed"])

    def test_invalid_configuration_and_selected_rule(self):
        for overrides in ({"construction": {"min_fp8_gap": .01}}, {"construction": {"confirmation_batches": 1}},
                          {"data": {"max_tokens": 4096}}, {"search": {"fidelity_samples": [4, 8, 11]}}):
            with self.assertRaises(ValueError):
                config(**overrides)
        with self.assertRaises(ValueError):
            config().for_condition({"task": "unknown"})
