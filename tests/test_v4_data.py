import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autokv.io import atomic_write_json
from autokv.v4_data import (PHASES, ensure_samples, generate_sample, make_world, source_pool,
                            base_identity, validate_samples)
from tests.v4_helpers import CharCodec, config, sources


class V4DataTests(unittest.TestCase):
    def test_mapping_truth_unique_codes_and_partial_key_distractors(self):
        for seed in range(10):
            world, blocks, targets, answer, question = make_world("single_lookup", 256, random.Random(seed))
            pairs = dict(world["pairs"])
            self.assertEqual(len(pairs), 256)
            self.assertEqual(len(set(pairs.values())), 256)
            self.assertEqual(pairs[world["query_key"]], answer)
            a, b = world["query_key"].split()
            shared = sum((x == a) != (y == b) for x, y in (k.split() for k in pairs))
            self.assertGreaterEqual(shared, 128)
            self.assertIn(answer, blocks[targets[0]])

    def test_two_hop_truth_requires_both_facts(self):
        world, blocks, targets, answer, _ = make_world("two_hop_lookup", 128, random.Random(6))
        cases, lockers = dict(world["cases"]), dict(world["lockers"])
        self.assertEqual(len(cases), 128)
        self.assertEqual(len(lockers), 128)
        self.assertEqual(len(set(lockers.values())), 128)
        self.assertEqual(lockers[cases[world["query_case"]]], answer)
        self.assertNotIn(answer, blocks[targets[0]])
        self.assertNotIn(world["query_case"], blocks[targets[1]])

    def test_same_mapping_cannot_become_new_instance_by_changing_query(self):
        world, _, _, _, _ = make_world("single_lookup", 4, random.Random(5))
        base = base_identity("single_lookup", world)
        world["query_key"] = next(k for k, _ in world["pairs"] if k != world["query_key"])
        world["pairs"].reverse()
        self.assertEqual(base_identity("single_lookup", world), base)

    def test_codes_exclude_background_and_truth_changes_are_rejected(self):
        original, *_ = make_world("single_lookup", 256, random.Random(0))
        excluded = {code for _, code in original["pairs"]}
        world, *_ = make_world("single_lookup", 256, random.Random(0), excluded)
        self.assertFalse({code for _, code in world["pairs"]} & excluded)
        cfg = config()
        row = generate_sample(cfg, CharCodec(), sources(), "test", cfg.conditions()[0], "test", 0)
        row["expected_answers"] = ["000000"]
        with self.assertRaisesRegex(ValueError, "真值"):
            validate_samples(cfg, {"test": [row]})

    def test_evidence_and_length_survive_background_fitting(self):
        cfg = config(construction={"tasks": ["single_lookup", "two_hop_lookup"], "target_lengths": [1600, 2200]})
        groups, orientations = {}, set()
        for phase in PHASES:
            groups[phase] = []
            for chosen in cfg.conditions():
                for i in range(3):
                    row = generate_sample(cfg, CharCodec(), sources(), phase, chosen, "batch", i)
                    self.assertLessEqual(abs(len(row["user_prompt"])-chosen["length_bucket"]), cfg.tolerance)
                    self.assertTrue(all(block in row["user_prompt"] for block in row["record_blocks"]))
                    if len(row["evidence_positions"]) == 2:
                        a, b = row["evidence_positions"]
                        self.assertGreaterEqual(abs(a-b), row["prompt_tokens"]/3)
                        orientations.add(a < b)
                    groups[phase].append(row)
        validate_samples(cfg, groups)
        self.assertEqual(orientations, {True, False})
        self.assertEqual(len({r["base_instance_id"] for rs in groups.values() for r in rs}), 48)

    def test_token_distance_uses_codec_and_rendered_chat(self):
        class UnevenCodec(CharCodec):
            def encode_text(self, text):
                return [ch for ch in text for _ in range(3 if ch == "a" else 1)]
            def render_and_count(self, prompt):
                rendered = "CHAT PREFIX "*8+prompt+" END"
                return rendered, len(self.encode_text(rendered))
        cfg = config(construction={"tasks": ["two_hop_lookup"], "target_lengths": [2200]})
        codec = UnevenCodec()
        row = generate_sample(cfg, codec, sources(), "test", cfg.conditions()[0], "test", 3)
        rendered, n = codec.render_and_count(row["user_prompt"])
        positions = [len(codec.encode_text(rendered[:rendered.index(row["record_blocks"][i])])) for i in row["target_record_indices"]]
        self.assertEqual(row["evidence_positions"], positions)
        self.assertGreaterEqual(abs(positions[0]-positions[1]), n/3)

    def test_resume_generated_prefix_and_no_source_dependency_for_complete_input(self):
        cfg, chosen = config(), config().conditions()[0]
        spec = [{"split": "confirmation", "condition": chosen, "batch_id": "batch-1", "count": 5}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"confirmation.jsonl"
            real = generate_sample
            def interrupted(*args):
                if args[-1] == 2:
                    raise KeyboardInterrupt()
                return real(*args)
            with patch("autokv.v4_data.generate_sample", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
                ensure_samples(path, spec, cfg, CharCodec(), sources)
            prefix = path.read_bytes()
            rows, _ = ensure_samples(path, spec, cfg, CharCodec(), sources)
            self.assertTrue(path.read_bytes().startswith(prefix))
            self.assertEqual(len(rows), 5)
            def unavailable(): raise AssertionError("已有数据无需来源")
            reused, _ = ensure_samples(path, spec, cfg, CharCodec(), unavailable)
            self.assertEqual(rows, reused)

    def test_unconstructible_condition_preserves_reason(self):
        cfg = config(construction={"record_counts": [256]})
        spec = [{"split": "discovery", "condition": cfg.conditions()[0], "batch_id": "discovery", "count": 4}]
        with tempfile.TemporaryDirectory() as tmp:
            rows, meta = ensure_samples(Path(tmp)/"discovery.jsonl", spec, cfg, CharCodec(), sources, allow_unconstructible=True)
            self.assertEqual(rows, [])
            state = next(iter(meta["conditions"].values()))
            self.assertEqual(state["status"], "unconstructible")
            self.assertIn("tokens", state["reason"])

    def test_duplicate_instance_and_cross_phase_background_rejected(self):
        cfg, chosen = config(), config().conditions()[0]
        a = generate_sample(cfg, CharCodec(), sources(), "experiment", chosen, "experiment", 0)
        b = generate_sample(cfg, CharCodec(), sources(), "test", chosen, "test", 0)
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_samples(cfg, {"experiment": [a, a]})
        b["source_document_ids"].append(a["background_document_ids"][0])
        with self.assertRaisesRegex(ValueError, "跨阶段"):
            validate_samples(cfg, {"experiment": [a], "test": [b]})

    def test_four_way_canonical_source_allocation(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            squad = {"data": [{"title": f"Article_{i}", "paragraphs": [{"context": f"Unique source text {i}.", "qas": []}]} for i in range(100)]}
            hotpot = [{"_id": str(i), "context": [[f"Other_{i}", [f"Other unique source text {i}."]]], "supporting_facts": []} for i in range(100)]
            entries = []
            for i, name in enumerate(cfg.raw["data"]["sources"]):
                file = f"{i}.json"
                atomic_write_json(folder/file, squad if name.startswith("squad") else hotpot)
                entries.append({"name": name, "file": file})
            atomic_write_json(folder/"source-manifest.json", {"sources": entries})
            pool = source_pool(folder, cfg)
            self.assertEqual(pool["allocation"], dict(zip(PHASES, [10, 10, 20, 160])))
            self.assertEqual(sum(map(len, pool["pools"].values())), 200)
