import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autokv.io import atomic_write_json
from autokv.v3_config import CONFIG_PATH
from autokv.v3_data import (EvidenceTooLong, fit_prompt, generate_dataset, load_dataset,
                            load_sources, make_data, validate_splits)
from tests.v3_helpers import CharCodec, config, source_fixture


class V3DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        cls.docs, cls.questions, cls.assignment = source_fixture()
        cls.splits, _ = generate_dataset(cls.cfg, CharCodec(), cls.docs, cls.questions, cls.assignment)

    def test_counts_disjointness_prefixes_and_protected_evidence(self):
        validate_splits(self.cfg, self.splits)
        self.assertEqual(len(self.splits["experiment"]), 48)
        self.assertEqual(len(self.splits["test"]), 64)
        self.assertEqual(len({r['source_question_id'] for rows in self.splits.values() for r in rows}), 144)
        for rows in self.splits.values():
            for row in rows:
                self.assertTrue(abs(len(row["user_prompt"])-row["length_bucket"]) <= 16)
                if row["task"].startswith("qa_"):
                    self.assertIn("Paris", row["user_prompt"])

    def test_separate_generation_matches_combined(self):
        separate, _ = generate_dataset(self.cfg, CharCodec(), self.docs, self.questions, self.assignment, ("experiment", "test"))
        self.assertEqual(separate["experiment"], self.splits["experiment"])
        self.assertEqual(separate["test"], self.splits["test"])

    def test_leakage_and_duplicate_length_variant_rejected(self):
        bad = copy.deepcopy(self.splits)
        bad["test"][0]["source_document_ids"].append(bad["experiment"][0]["source_document_ids"][0])
        with self.assertRaisesRegex(ValueError, "跨 split"):
            validate_splits(self.cfg, bad)
        bad = copy.deepcopy(self.splits)
        bad["test"][0]["source_question_id"] = bad["experiment"][0]["source_question_id"]
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_splits(self.cfg, bad)

    def test_evidence_cannot_be_trimmed(self):
        with self.assertRaises(EvidenceTooLong):
            fit_prompt(CharCodec(), 100, 2, "Read", "Answer", ["Evidence"*100], "background"*100)

    def test_frozen_files_reuse_and_input_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_write_json(root / CONFIG_PATH, self.cfg.raw)
            with patch("autokv.v3_data.load_sources", return_value=(self.docs, self.questions, self.assignment, {})):
                result = make_data(root, root, codec=CharCodec())
                self.assertFalse(result["reused"])
                self.assertTrue(make_data(root, root, codec=CharCodec())["reused"])
            manifest, loaded = load_dataset(root, self.cfg)
            self.assertEqual(loaded["test"], self.splits["test"])
            path = root / "data/v3.0/quality/test.jsonl"
            rows = path.read_text(encoding="utf-8").splitlines()
            row = json.loads(rows[0])
            row["user_prompt"] += "changed"
            rows[0] = json.dumps(row)
            path.write_text("\n".join(rows)+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "实际数据"):
                load_dataset(root, self.cfg)

    def test_raw_sources_keep_answers_and_all_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            squad = {"data": [{"title": "Article_A", "paragraphs": [{"context": "Paris is here.", "qas": [
                {"id": "q1", "question": "Where?", "answers": [{"text": "Paris", "answer_start": 0}]},
                {"id": "q2", "question": "Missing?", "is_impossible": True, "answers": []}]}]}]}
            hotpot = [{"_id": "h1", "question": "Which city?", "answer": "Paris",
                       "context": [["Article A", ["Paris is here."]], ["Article B", ["The city is in France."]]],
                       "supporting_facts": [["Article A", 0], ["Article B", 0]]}]
            sources = []
            for i, name in enumerate(self.cfg.raw["data"]["sources"]):
                filename = str(i)+".json"
                atomic_write_json(root / filename, squad if name.startswith("squad") else hotpot)
                sources.append({"name": name, "file": filename})
            atomic_write_json(root / "source-manifest.json", {"sources": sources})
            docs, qs, split, _ = load_sources(root, self.cfg)
            self.assertEqual(len(qs), 2)
            self.assertEqual(len(next(q for q in qs if q["task"] == "qa_multi")["evidence"]), 2)
            self.assertEqual(len(docs), 2)
