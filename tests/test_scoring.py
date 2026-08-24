import unittest

from autokv.scoring import (
    answer_nll_from_echo,
    levenshtein_distance,
    quality_score,
    score_generation,
)


class ScoringTests(unittest.TestCase):
    def test_generation_score_normalizes_whitespace_and_case(self):
        expected = "A-1|A-1|A-1|A-1|A-1"
        score = score_generation(
            "Answer:  a-1 | a-1 | a-1 | a-1 | a-1.", expected
        )
        self.assertEqual(score.exact_match, 1.0)

    def test_extracts_answer_nll_from_openai_echo_offsets(self):
        response = {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [None, -0.2, -0.4],
                        "text_offset": [0, 10, 12],
                    }
                }
            ]
        }
        self.assertAlmostEqual(answer_nll_from_echo(response, 10), 0.3)

    def test_missing_echo_logprobs_returns_none(self):
        self.assertIsNone(answer_nll_from_echo({"choices": [{}]}, 10))

    def test_quality_uses_nll_or_edit_distance_fallback(self):
        with_nll = quality_score(1.0, 0.0, 5, 10)
        fallback = quality_score(1.0, None, 5, 10)
        self.assertAlmostEqual(with_nll, 1.0)
        self.assertAlmostEqual(fallback, 0.9995)

    def test_levenshtein_distance_handles_empty_and_substitution(self):
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("kitten", "sitten"), 1)


if __name__ == "__main__":
    unittest.main()
