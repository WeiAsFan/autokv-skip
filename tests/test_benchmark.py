import unittest

from autokv.benchmark import build_benchmark_matrix, parse_capacity_tokens
from autokv.config import Profile


class BenchmarkTests(unittest.TestCase):
    def test_parses_vllm_capacity_log(self):
        log = (
            "GPU KV cache size: 233,104 tokens\n"
            "Maximum concurrency for 32,768 tokens per request: 7.11x\n"
        )
        parsed = parse_capacity_tokens(log)
        self.assertEqual(parsed.tokens, 233104)
        self.assertEqual(parsed.model_length, 32768)
        self.assertAlmostEqual(parsed.max_concurrency, 7.11)

    def test_missing_capacity_is_a_hard_error(self):
        with self.assertRaisesRegex(ValueError, "KV cache size"):
            parse_capacity_tokens("server ready")

    def test_quick_matrix_has_six_scenarios(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        cases = build_benchmark_matrix(profile)
        self.assertEqual(len(cases), 6)
        self.assertEqual(cases[0].input_length, 1024)
        self.assertEqual(cases[-1].output_length, 256)


if __name__ == "__main__":
    unittest.main()
