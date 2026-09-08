import copy
import unittest

from autokv.v3_metrics import aggregate, bootstrap_indices, comparison, paired_interval, score_output
from tests.v3_helpers import config, samples


class V3MetricsTests(unittest.TestCase):
    def test_pairing_and_extra_values(self):
        sample = {"task": "multi_key", "expected_answers": [["K00000001", "V00000001"], ["K00000002", "V00000002"]]}
        self.assertEqual(score_output('K00000001=V00000002\nK00000002=V00000001', sample), 0)
        self.assertEqual(score_output('{"K00000001":"V00000001","K00000002":"V00000002"}', sample), 1)
        sample = {"task": "multi_value", "expected_answers": ["V00000001", "V00000002"]}
        self.assertEqual(score_output("V00000001,V00000001,V00000002", sample), 1)
        self.assertEqual(score_output("V00000001 V00000002 V00000003", sample), .8)
        self.assertEqual(score_output("XV00000001X", sample), 0)

    def test_qa_alias_and_empty(self):
        sample = {"task": "qa_single", "expected_answers": ["the city of Paris", "Paris"]}
        self.assertEqual(score_output("PARIS.", sample), 1)
        self.assertEqual(score_output("", sample), 0)
        self.assertLess(score_output("Paris followed by an unrelated explanation", sample), 1)

    def test_task_regression_not_offset_by_improvement(self):
        cfg = config()
        ref = [{**r, "task_score": .8} for r in samples(cfg)["experiment"]]
        cand = [{**r, "task_score": .77 if r["task"] == "qa_multi" else .9} for r in ref]
        result = comparison(ref, cand, cfg)
        self.assertGreater(result["scores"]["all"], .8)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["gaps"]["qa_multi"], .03)
        with self.assertRaises(ValueError):
            comparison(ref, cand[:-1], cfg)

    def test_uniform_loss_boundary_and_length_weighting(self):
        cfg = config()
        ref = [{**r, "task_score": 1.0} for r in samples(cfg)["experiment"]]
        cand = [{**r, "task_score": .99} for r in ref]
        self.assertTrue(comparison(ref, cand, cfg)["passed"])
        self.assertAlmostEqual(aggregate(cand)["all"], .99)
        interval = paired_interval(ref, cand, cfg)
        self.assertAlmostEqual(interval["intervals"]["all"][0], .01)

    def test_group_sampling_and_unknown_intervals(self):
        cfg = config()
        rows = [{**r, "task_score": 1.0} for r in samples(cfg)["experiment"]]
        for row in rows:
            row["source_group_id"] = f"{row['task']}:{row['length_bucket']}"
        self.assertIsNone(bootstrap_indices(rows, 10, 42))
        self.assertIsNone(paired_interval(rows, rows, cfg)["intervals"])
        rows[1]["source_group_id"] = rows[0]["source_group_id"]
        with self.assertRaises(ValueError):
            bootstrap_indices(rows, 10, 42)
