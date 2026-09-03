import unittest
from pathlib import Path

from autokv.v2_config import load_v2_config
from autokv.v2_metrics import (
    aggregate_v2,
    paired_gap_summary,
    qa_f1_score,
    quality_constraints,
    score_v2_output,
    set_f1_score,
)


ROOT = Path(__file__).resolve().parents[1]


def result_rows(scores):
    rows = []
    for tier in ("easy", "hard", "natural"):
        for index, score in enumerate(scores[tier]):
            rows.append(
                {
                    "sample_id": f"{tier}-{index}",
                    "tier": tier,
                    "task_score": score,
                    "error": None,
                }
            )
    return rows


class V2MetricsTests(unittest.TestCase):
    def test_three_scoring_modes_have_known_outputs(self):
        self.assertEqual(
            score_v2_output(
                "prefix KV-1234|KV-1234 suffix",
                {"answer_mode": "contains", "expected_answers": ["KV-1234|KV-1234"]},
            ),
            1.0,
        )
        self.assertEqual(qa_f1_score("The Eiffel Tower", "eiffel tower"), 1.0)
        self.assertAlmostEqual(qa_f1_score("red blue", "red green"), 0.5)
        self.assertAlmostEqual(
            set_f1_score(
                "VAL-AAAA1111|VAL-WRONG999",
                ["VAL-AAAA1111", "VAL-BBBB2222"],
                r"VAL-[A-Z0-9]{8}",
            ),
            0.5,
        )

    def test_tier_equal_aggregation_and_threshold_boundaries(self):
        config = load_v2_config(ROOT / "configs/v2/quality.json")
        reference_rows = result_rows(
            {"easy": [1.0, 1.0], "hard": [1.0, 1.0], "natural": [1.0, 1.0]}
        )
        candidate_rows = result_rows(
            {"easy": [1.0, 1.0], "hard": [0.98, 0.98], "natural": [0.99, 0.99]}
        )
        reference = aggregate_v2(reference_rows)
        candidate = aggregate_v2(candidate_rows)

        self.assertAlmostEqual(candidate["s_v2"], 0.99)
        self.assertTrue(
            quality_constraints(reference, candidate, config, endpoint=True)["passed"]
        )
        gaps = paired_gap_summary(
            reference_rows,
            candidate_rows,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
        self.assertAlmostEqual(gaps["global"]["gap"], 0.01)
        self.assertAlmostEqual(gaps["hard"]["gap"], 0.02)


if __name__ == "__main__":
    unittest.main()
