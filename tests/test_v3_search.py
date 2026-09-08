import tempfile
import unittest
from pathlib import Path

from autokv.io import read_jsonl
from autokv.v3_search import search
from tests.v3_helpers import FakeRunner, config, samples


class V3SearchTests(unittest.TestCase):
    def run_search(self, cfg, score):
        directory = Path(self.addCleanupDirectory())
        data = samples(cfg)
        runner = FakeRunner(cfg, data, directory, score)
        result = search(cfg, data["experiment"], runner, directory / "trace.jsonl")
        return result, runner, directory

    def addCleanupDirectory(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return temp.name

    def test_p0_endpoint_skips_search(self):
        result, runner, _ = self.run_search(config(), lambda *_: 1.0)
        self.assertEqual(result["candidate"]["k"], 0)
        self.assertEqual(runner.statistics()["phases"]["search"]["requests"], 0)

    def test_beam_second_path_and_conditionally_useful_layer(self):
        def score(layers, sample):
            if len(layers) == 8 or layers == {3, 7}:
                return 1.0
            if not layers:
                return .5
            if len(layers) == 1:
                return .94 if 0 in layers else .93 if 3 in layers else .55
            return .96
        result, runner, directory = self.run_search(config(), score)
        self.assertEqual(result["candidate"]["bf16_layers"], [3, 7])
        self.assertEqual({layers[0] for layers, split, n in runner.calls if len(layers) == 1}, set(range(8)))
        trace = read_jsonl(directory/"trace.jsonl")
        pair = next(r for r in trace if r["event"] == "evaluation" and r["policy"]["bf16_layers"] == [3, 7])
        self.assertTrue(pair["marginal_gains"])
        greedy, _, _ = self.run_search(config(search={"beam_width": 1}), score)
        self.assertIsNone(greedy["candidate"])

    def test_all_early_ties_expand_instead_of_index_pruning(self):
        cfg = config()
        result, runner, directory = self.run_search(cfg, lambda layers, row: 1.0 if len(layers) == 8 else .5)
        full_events = [r for r in read_jsonl(directory/"trace.jsonl") if r["event"] == "evaluation" and
                       r["depth"] == 1 and r["samples"] == cfg.experiment_size]
        self.assertEqual(len(full_events), 8)
        self.assertEqual(result["status"], "no_feasible_within_budget")

    def test_clear_first_round_uses_1856_answers(self):
        cfg = config(False)
        def score(layers, sample):
            if len(layers) == 32 or layers == {31}:
                return 1.0
            return .1+.02*next(iter(layers)) if layers else .05
        result, runner, _ = self.run_search(cfg, score)
        self.assertEqual(result["candidate"]["bf16_layers"], [31])
        self.assertEqual(runner.statistics()["phases"]["search"]["requests"], 1856)
        self.assertEqual(runner.statistics()["phases"]["search"]["server_starts"], 42)

    def test_backward_deletion_reaches_previously_missed_pair(self):
        cfg = config(model={"num_layers": 12})
        def score(layers, sample):
            if len(layers) == 12 or {1, 2}.issubset(layers):
                return 1.0
            if len(layers) == 1:
                return .95 if 0 in layers else .94 if 3 in layers else .6
            if layers == {0, 1}:
                return .98
            if layers == {0, 2}:
                return .979
            return .5 if not layers else .96
        result, _, directory = self.run_search(cfg, score)
        self.assertEqual(result["candidate"]["bf16_layers"], [1, 2])
        self.assertTrue(any(r["event"] == "delete_accepted" for r in read_jsonl(directory/"trace.jsonl")))

    def test_budget_exhaustion_and_reference_degeneracy(self):
        cfg = config(search={"max_requests": 0})
        result, _, _ = self.run_search(cfg, lambda layers, _: 1.0 if len(layers) == 8 else .5)
        self.assertEqual(result["status"], "no_feasible_within_budget")
        result, _, _ = self.run_search(config(), lambda *_: 0.0)
        self.assertEqual(result["status"], "reference_degenerate")
