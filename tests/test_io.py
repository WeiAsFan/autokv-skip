import tempfile
import unittest
from pathlib import Path

from autokv.io import (
    append_jsonl,
    atomic_write_json,
    ensure_within,
    read_json,
    read_jsonl,
    redact,
    sha256_file,
)


class IoTests(unittest.TestCase):
    def test_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, "outside project root"):
                ensure_within(root, root.parent / "escaped.json")

    def test_accepts_path_below_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            child = root / "runs" / "state.json"
            self.assertEqual(ensure_within(root, child), child.resolve())

    def test_atomic_json_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(read_json(path), {"a": 1, "b": 2})
            self.assertEqual(len(sha256_file(path)), 64)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_jsonl_append_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            append_jsonl(path, {"row": 1})
            append_jsonl(path, {"row": 2})
            self.assertEqual(read_jsonl(path), [{"row": 1}, {"row": 2}])

    def test_redacts_each_nonempty_secret(self):
        value = "Authorization: abc123; token=xyz"
        self.assertEqual(
            redact(value, ["", "abc123", "xyz"]),
            "Authorization: ***; token=***",
        )


if __name__ == "__main__":
    unittest.main()
