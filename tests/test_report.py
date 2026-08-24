import tempfile
import unittest
from pathlib import Path

from autokv.benchmark import Capacity
from autokv.config import Profile
from autokv.report import paired_bootstrap_ci, render_report


class ReportTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self):
        first = paired_bootstrap_ci(
            [1.0, 0.8, 0.9], [0.7, 0.6, 0.8], seed=42, samples=1000
        )
        second = paired_bootstrap_ci(
            [1.0, 0.8, 0.9], [0.7, 0.6, 0.8], seed=42, samples=1000
        )
        self.assertEqual(first, second)
        self.assertGreater(first.mean, 0)

    def test_report_writes_markdown_csv_and_svg(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        quality = {
            "bf16": [
                {"sample_id": "a", "exact_match": 1.0, "quality_score": 0.95},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.93},
            ],
            "fp8": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.60},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.70},
            ],
            "auto-4": [
                {"sample_id": "a", "exact_match": 1.0, "quality_score": 0.85},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.88},
            ],
            "random-4-1": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.65},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.72},
            ],
            "random-4-2": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.66},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.73},
            ],
            "random-4-3": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.64},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.71},
            ],
            "random-4-4": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.63},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.70},
            ],
            "random-4-5": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.62},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.68},
            ],
            "first-4": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.62},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.70},
            ],
            "last-4": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.63},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.71},
            ],
            "inverted-4": [
                {"sample_id": "a", "exact_match": 0.0, "quality_score": 0.61},
                {"sample_id": "b", "exact_match": 1.0, "quality_score": 0.69},
            ],
        }
        capacities = {
            "bf16": Capacity(131072, 32768, 4.0),
            "fp8": Capacity(260000, 32768, 7.93),
            "auto-4": Capacity(230000, 32768, 7.02),
        }
        performance = {
            "bf16": {
                "overall_descriptive_mean": {
                    "request_throughput": 1.0,
                    "output_throughput": 32.0,
                },
                "scenario_groups": [
                    {
                        "input_length": 1024,
                        "output_length": 32,
                        "repeats": 1,
                        "metrics": {
                            "request_throughput": 1.0,
                            "median_ttft_ms": 100.0,
                            "p99_ttft_ms": 150.0,
                            "median_tpot_ms": 8.0,
                            "p99_tpot_ms": 12.0,
                            "median_itl_ms": 7.0,
                            "p99_itl_ms": 11.0,
                            "median_e2el_ms": 400.0,
                            "p99_e2el_ms": 600.0,
                        },
                    }
                ],
                "capacity_validation": {
                    "theoretical_tokens": 262144,
                    "measured_tokens": 131072,
                    "relative_deviation": 0.5,
                    "within_10_percent": False,
                    "page_or_block_evidence": ["# GPU blocks: 8192"],
                    "evidence_complete": True,
                },
            },
            "fp8": {
                "overall_descriptive_mean": {"request_throughput": 0.95},
                "scenario_groups": [{"input_length": 1024, "output_length": 32, "repeats": 1, "metrics": {"request_throughput": 0.95, "median_ttft_ms": 110.0}}],
                "capacity_validation": {"theoretical_tokens": 524288, "measured_tokens": 260000, "relative_deviation": 0.504, "within_10_percent": False, "page_or_block_evidence": ["# GPU blocks: 16250"], "evidence_complete": True},
            },
            "auto-4": {
                "overall_descriptive_mean": {"request_throughput": 0.97},
                "scenario_groups": [{"input_length": 1024, "output_length": 32, "repeats": 1, "metrics": {"request_throughput": 0.97, "median_ttft_ms": 106.0}}],
                "capacity_validation": {"theoretical_tokens": 466033, "measured_tokens": 230000, "relative_deviation": 0.506, "within_10_percent": False, "page_or_block_evidence": ["# GPU blocks: 14375"], "evidence_complete": True},
            },
        }
        selection = {
            "auto_layers": [2, 7, 18, 29],
            "layer_scores": {"2": 0.2, "7": 0.3, "18": 0.1, "29": 0.4},
            "group_scores": {"group-00-03": 0.1, "group-04-07": 0.2},
            "source": {"tree_sha256": "b" * 64, "git_commit": "c" * 40, "git_dirty": False},
        }
        lock = {
            "image_ref": "vllm/vllm-openai@sha256:" + "f" * 64,
            "model_revision": "a" * 40,
            "versions": {"vllm": "0.26.0", "cuda": "13.0"},
            "host": {"driver": "535.230.02", "gpu_name": "NVIDIA RTX A6000"},
        }
        with tempfile.TemporaryDirectory() as directory:
            artifacts = render_report(
                Path(directory),
                profile,
                lock,
                selection,
                quality,
                capacities,
                performance,
            )
            for path in artifacts.values():
                self.assertTrue(path.exists(), path)
            markdown = artifacts["markdown"].read_text(encoding="utf-8")
            self.assertIn("AutoKV-Skip 实验报告", markdown)
            self.assertIn("535.230.02", markdown)
            self.assertIn("2, 7, 18, 29", markdown)
            self.assertIn("A6000 不具备原生 FP8 Tensor Core", markdown)
            self.assertIn("Gap recovery", markdown)
            self.assertIn("Median TPOT", markdown)
            self.assertIn("P99 ITL", markdown)
            self.assertIn("逐场景性能", markdown)
            self.assertIn("分组与逐层敏感性", markdown)
            self.assertIn("group-04-07", markdown)
            self.assertIn("理论 tokens", markdown)
            self.assertIn("bbbbbbbbbbbb", markdown)
            summary_csv = artifacts["csv"].read_text(encoding="utf-8")
            self.assertIn("output_throughput", summary_csv.splitlines()[0])
            self.assertIn("p99_e2el_ms", summary_csv.splitlines()[0])

    def test_report_rejects_missing_preregistered_control(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        row = [{"sample_id": "a", "exact_match": 1.0, "quality_score": 1.0}]
        quality = {name: row for name in ("bf16", "fp8", "auto-4")}
        capacities = {name: Capacity(1, None, None) for name in quality}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "pre-registered quality configurations"):
                render_report(
                    Path(directory),
                    profile,
                    {},
                    {"auto_layers": [0, 1, 2, 3]},
                    quality,
                    capacities,
                    {},
                )


if __name__ == "__main__":
    unittest.main()
