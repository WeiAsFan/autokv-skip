import unittest

from autokv.v2_policy import (
    endpoint_policies,
    group_policies,
    layer_policies,
    nested_budget_policy,
    random_control_policies,
    theoretical_capacity,
)


class V2PolicyTests(unittest.TestCase):
    def test_coarse_fine_and_nested_budget_are_bounded(self):
        groups = group_policies()
        self.assertEqual(len(groups), 8)
        self.assertEqual(groups[0].bf16_layers, (0, 1, 2, 3))
        self.assertEqual(groups[-1].bf16_layers, (28, 29, 30, 31))
        layers = layer_policies((groups[2], groups[5]))
        self.assertEqual(len(layers), 8)
        ranking = [policy.bf16_layers[0] for policy in reversed(layers)]
        self.assertLess(
            set(nested_budget_policy(ranking, 2).bf16_layers),
            set(nested_budget_policy(ranking, 4).bf16_layers),
        )

    def test_endpoints_random_controls_and_capacity(self):
        p32, p0 = endpoint_policies()
        selected = nested_budget_policy((3, 7, 11, 15, 19, 23, 27, 31), 4)
        controls = random_control_policies(selected, (11, 23, 37))

        self.assertEqual((p32.k, p0.k), (32, 0))
        self.assertEqual(len({policy.bf16_layers for policy in controls}), 3)
        self.assertNotIn(
            selected.bf16_layers, {policy.bf16_layers for policy in controls}
        )
        self.assertAlmostEqual(theoretical_capacity(p0)["capacity_ratio_vs_p32"], 2.0)
        self.assertAlmostEqual(
            theoretical_capacity(selected)["capacity_ratio_vs_p32"], 1.7777777778
        )


if __name__ == "__main__":
    unittest.main()
