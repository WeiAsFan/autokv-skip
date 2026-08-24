import unittest

from autokv.config import Profile
from autokv.niah import NiahCase, expected_answer, fit_prompt, make_cases


class NiahTests(unittest.TestCase):
    def test_expected_answer_repeats_code_five_times(self):
        self.assertEqual(
            expected_answer("ZEBRA-4821"),
            "|".join(["ZEBRA-4821"] * 5),
        )

    def test_fit_prompt_is_deterministic_and_within_half_percent(self):
        case = NiahCase("case-1", 2000, 0.5, 42, "ZEBRA-4821")
        count_tokens = lambda text: len(text.split())
        first = fit_prompt(case, count_tokens)
        second = fit_prompt(case, count_tokens)
        self.assertEqual(first, second)
        self.assertLessEqual(abs(first.token_count - 2000), 10)
        self.assertIn("VERIFICATION-CODE: ZEBRA-4821", first.prompt)
        self.assertLess(abs(first.needle_depth - 0.5), 0.02)

    def test_quick_profile_has_six_probe_and_eighteen_final_cases(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        probe = make_cases(profile, "probe")
        final = make_cases(profile, "final")
        self.assertEqual(len(probe), 6)
        self.assertEqual(len(final), 18)
        self.assertEqual(make_cases(profile, "probe"), probe)
        self.assertEqual(len({case.sample_id for case in final}), 18)


if __name__ == "__main__":
    unittest.main()
