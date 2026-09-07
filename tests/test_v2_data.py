import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

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
    def test_v21_variable_answers_match_independent_step_execution(self):
        config = load_v2_config(ROOT / "configs/v2.1/quality.json")
        for row in make_hard_rows(config, WordCodec()):
            if row["task"] != "variable_tracking":
                continue
            state = {}
            for line in row["prompt"].splitlines():
                if line.startswith("STEP "):
                    name, value = line.split(": SET ", 1)[1].split(" = ")
                    state[name] = state[value.split()[1]] if value.startswith("VALUE-OF ") else value
            names = row["metadata"]["query_variables"]
            self.assertEqual(row["expected_answers"], [f"{name}={state[name]}" for name in names])
            self.assertGreaterEqual(len({state[name] for name in names}), 2)
            self.assertEqual(row["answer_mode"], "variable_f1")

    def test_first_v21_freeze_reuses_natural_data_offline_and_then_resumes(self):
        from autokv.cli import _freeze_v2_data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            shutil.copytree(ROOT / "configs", root / "configs")
            shutil.copytree(ROOT / "data/v2/quality", root / "data/v2/quality")
            before = {p.name: p.read_bytes() for p in (root / "data/v2/quality").iterdir()}
            with patch("autokv.v2_pipeline._load_v2_lock", return_value={"backend": "local_vllm", "model_path": str(root / "model")}), patch("autokv.v2_data.TransformersPromptCodec", return_value=WordCodec()):
                first = _freeze_v2_data(root, "missing-source")
            second = _freeze_v2_data(root, "missing-source")
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "data/v2/quality").iterdir()})
            config_path = root / "configs/v2.1/quality.json"
            config = load_v2_config(config_path)
            manifest, calibration, heldout = load_frozen_v2_dataset(config, root / "data/v2.1/quality", config_path=config_path)
            self.assertEqual(manifest["scoring_version"], "autokv-v2.1-score-v1")
            self.assertEqual((len(calibration), len(heldout)), (27, 18))
            old = [json.loads(line) for name in ("calibration.jsonl", "heldout.jsonl") for line in before[name].decode("utf-8").splitlines()]
            natural = lambda rows: {(r["split"], r["metadata"]["source_id"]): (r["prompt"], r["expected_answers"]) for r in rows if r["tier"] == "natural"}
            self.assertEqual(natural(old), natural([*calibration, *heldout]))

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

    def test_published_dataset_accepts_crlf_but_rejects_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            shutil.copytree(ROOT / "configs", root / "configs")
            shutil.copytree(ROOT / "data/v2/quality", root / "data/v2/quality")
            config_path = root / "configs/v2/quality.json"
            calibration_path = root / "data/v2/quality/calibration.jsonl"
            for path in (config_path, calibration_path):
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
            config = load_v2_config(config_path)
            _, calibration, heldout = load_frozen_v2_dataset(config, root / "data/v2/quality", config_path=config_path)
            self.assertEqual((len(calibration), len(heldout)), (27, 18))
            calibration_path.write_bytes(calibration_path.read_bytes().replace(b'"prompt":', b'"changed_prompt":', 1))
            with self.assertRaisesRegex(ValueError, "calibration hash"):
                load_frozen_v2_dataset(config, root / "data/v2/quality", config_path=config_path)


if __name__ == "__main__":
    unittest.main()
