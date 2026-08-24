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
            "bf16": {"request_throughput": 1.0, "median_ttft_ms": 100.0},
            "fp8": {"request_throughput": 0.95, "median_ttft_ms": 110.0},
            "auto-4": {"request_throughput": 0.97, "median_ttft_ms": 106.0},
        }
        selection = {
            "auto_layers": [2, 7, 18, 29],
            "layer_scores": {"2": 0.2, "7": 0.3, "18": 0.1, "29": 0.4},
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


if __name__ == "__main__":
    unittest.main()
