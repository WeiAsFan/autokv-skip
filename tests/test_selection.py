import unittest

from autokv.selection import (
    Variant,
    canonical_config_id,
    group_layers,
    group_probe_variants,
    random_controls,
    select_bottom_layers,
    select_top_groups,
    select_top_layers,
)


class SelectionTests(unittest.TestCase):
    def test_groups_32_layers_into_eight_contiguous_groups(self):
        groups = group_layers(32, 4)
        self.assertEqual(groups[0], (0, 1, 2, 3))
        self.assertEqual(groups[-1], (28, 29, 30, 31))
        self.assertEqual(len(groups), 8)

    def test_rejects_uneven_grouping(self):
        with self.assertRaisesRegex(ValueError, "evenly divisible"):
            group_layers(30, 4)

    def test_selects_highest_scored_layers_with_stable_tie_break(self):
        scores = {7: 0.3, 2: 0.3, 9: -0.1, 4: 0.2}
        self.assertEqual(select_top_layers(scores, 2), (2, 7))
        self.assertEqual(select_bottom_layers(scores, 2), (4, 9))

    def test_selects_best_groups_with_lexicographic_tie_break(self):
        first = (0, 1, 2, 3)
        second = (4, 5, 6, 7)
        third = (8, 9, 10, 11)
        self.assertEqual(
            select_top_groups({third: 0.1, second: 0.5, first: 0.5}, 2),
            (first, second),
        )

    def test_random_controls_are_unique_and_do_not_duplicate_named_sets(self):
        forbidden = {(0, 1, 2, 3), (28, 29, 30, 31)}
        controls = random_controls(32, 4, [11, 23, 37, 53, 71], forbidden)
        self.assertEqual(len(controls), 5)
        self.assertEqual(len(set(controls)), 5)
        self.assertTrue(all(len(item) == 4 for item in controls))
        self.assertTrue(all(item not in forbidden for item in controls))

    def test_variant_and_identifier_are_order_independent(self):
        first = Variant.mixed("auto-4", (7, 2, 29, 18))
        second = Variant.mixed("auto-4", (2, 7, 18, 29))
        self.assertEqual(first.skip_layers, (2, 7, 18, 29))
        self.assertEqual(canonical_config_id(first), canonical_config_id(second))

    def test_group_probe_variants_cover_every_layer_once(self):
        variants = group_probe_variants(32, 4)
        flattened = [layer for variant in variants for layer in variant.skip_layers]
        self.assertEqual(flattened, list(range(32)))
        self.assertEqual(len(variants), 8)


if __name__ == "__main__":
    unittest.main()
