import unittest

from autokv.memory import (
    ideal_capacity_gain,
    kv_bytes_per_token,
    mixed_kv_bytes_per_token,
)


class MemoryTests(unittest.TestCase):
    def test_mistral_memory_numbers(self):
        self.assertEqual(kv_bytes_per_token(32, 8, 128, 2), 131072)
        self.assertEqual(kv_bytes_per_token(32, 8, 128, 1), 65536)
        self.assertEqual(mixed_kv_bytes_per_token(32, 4, 8, 128), 73728)
        self.assertAlmostEqual(ideal_capacity_gain(32, 4), 1.7777777777777777)

    def test_rejects_out_of_range_bf16_layer_count(self):
        with self.assertRaisesRegex(ValueError, "between zero and num_layers"):
            mixed_kv_bytes_per_token(32, 33, 8, 128)


if __name__ == "__main__":
    unittest.main()
