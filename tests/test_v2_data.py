import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from autokv.io import sha256_file
from autokv.v2_config import load_v2_config
from autokv.v2_data import (
    freeze_v2_dataset,
    load_frozen_v2_dataset,
    make_easy_rows,
    make_hard_rows,
    make_natural_rows,
    validate_v2_rows,
)


ROOT = Path(__file__).resolve().parents[1]


class WordCodec:
    template_sha256 = "a" * 64

    def render_and_count(self, user_prompt):
        rendered = "<s> [INST] " + user_prompt + " [/INST]"
        return rendered, len(rendered.split())


def natural_sources():
    result = {}
    for dataset in ("qasper_e", "hotpotqa_e"):
        rows = []
        for bucket, length in enumerate((1000, 5000, 9000)):
            for offset in range(2):
                identifier = f"{dataset}-{bucket}-{offset}"
                rows.append(
                    {
                        "_id": identifier,
                        "input": f"Question {identifier}?",
                        "context": f"Context {identifier} with the answer.",
                        "answers": [f"answer {identifier}"],
                        "length": length,
                        "dataset": dataset,
                        "language": "en",
                        "all_classes": [],
                    }
                )
        result[dataset] = list(reversed(rows))
    return result


class V2DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v2_config(ROOT / "configs/v2/quality.json")
        cls.codec = WordCodec()
        cls.rows = [
            *make_easy_rows(cls.config, cls.codec),
            *make_hard_rows(cls.config, cls.codec),
            *make_natural_rows(cls.config, cls.codec, natural_sources()),
        ]

    def test_exact_scale_seed_and_length_contract(self):
        validate_v2_rows(self.config, self.rows)
        counts = Counter((row["split"], row["tier"]) for row in self.rows)

        self.assertEqual(len(self.rows), 45)
        self.assertEqual(counts[("calibration", "easy")], 3)
        self.assertEqual(counts[("calibration", "hard")], 18)
        self.assertEqual(counts[("calibration", "natural")], 6)
        self.assertEqual(counts[("heldout", "easy")], 3)
        self.assertEqual(counts[("heldout", "hard")], 9)
        self.assertEqual(counts[("heldout", "natural")], 6)
        for row in self.rows:
            if row["tier"] in {"easy", "hard"}:
                self.assertLessEqual(
                    abs(row["prompt_tokens"] - row["target_tokens"]), 32
                )

    def test_generation_is_deterministic_and_splits_do_not_overlap(self):
        repeated = [
            *make_easy_rows(self.config, self.codec),
            *make_hard_rows(self.config, self.codec),
            *make_natural_rows(self.config, self.codec, natural_sources()),
        ]

        self.assertEqual(self.rows, repeated)
        calibration = {
            row["sample_id"] for row in self.rows if row["split"] == "calibration"
        }
        heldout = {row["sample_id"] for row in self.rows if row["split"] == "heldout"}
        self.assertFalse(calibration & heldout)

    def test_freeze_round_trip_has_stable_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            files = {}
            counts = {}
            for dataset, rows in natural_sources().items():
                content = "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                )
                filename = f"{dataset}.jsonl"
                source_path = source_root / filename
                source_path.write_text(content, encoding="utf-8")
                files[filename] = sha256_file(source_path)
                counts[dataset] = len(rows)
            source_manifest = {
                "schema_version": 1,
                "repository": "THUDM/LongBench",
                "revision": self.config.natural_source_revision,
                "split": "test",
                "datasets": list(self.config.natural_datasets),
                "rows": counts,
                "files": files,
            }
            (source_root / "source-manifest.json").write_text(
                json.dumps(source_manifest), encoding="utf-8"
            )
            config_path = ROOT / "configs/v2/quality.json"
            first = freeze_v2_dataset(
                self.config,
                self.codec,
                source_root,
                root / "first",
                config_path=config_path,
            )
            second = freeze_v2_dataset(
                self.config,
                self.codec,
                source_root,
                root / "second",
                config_path=config_path,
            )

            self.assertEqual(first, second)
            loaded, calibration, heldout = load_frozen_v2_dataset(
                self.config, root / "first", config_path=config_path
            )
            self.assertEqual(loaded["dataset_sha256"], first["dataset_sha256"])
            self.assertEqual((len(calibration), len(heldout)), (27, 18))


if __name__ == "__main__":
    unittest.main()
