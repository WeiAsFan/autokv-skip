import copy
from pathlib import Path

from autokv.io import append_jsonl, read_json
from autokv.v3_config import TASKS, V3Config
from autokv.v3_runtime import SearchBudgetExceeded

ROOT = Path(__file__).resolve().parents[1]


def config(small=True, **overrides):
    raw = read_json(ROOT / "configs/v3.0/quality.json")
    raw["scoring"]["bootstrap_samples"] = 20
    if small:
        raw["model"].update(num_layers=8, max_model_len=4096)
        raw["data"].update(target_lengths=[1200, 1600, 2000, 2400], input_tolerance_tokens=16,
                           per_cell={"development": 2, "experiment": 3, "test": 4},
                           retrieval={"key_records": 4, "query_keys": 2, "values": 2, "distractors": 4})
        raw["search"]["fidelity_samples"] = [16, 32, 48]
    for section, values in overrides.items():
        raw[section].update(values)
    return V3Config.from_dict(raw)


def samples(cfg):
    return {split: [{"sample_id": f"{split}:{i}:{t}:{l}", "source_question_id": f"{split}:{i}:{t}:{l}",
                    "split": split, "task": t, "length_bucket": l, "source_group_id": f"{split}:{i}:{t}:{l}",
                    "user_prompt": f"{split}:{i}:{t}:{l}", "expected_answers": [["K00000001", "V00000001"]] if t == "multi_key"
                    else ["V00000001"] if t == "multi_value" else ["Paris"],
                    "prompt_tokens": l, "max_tokens": cfg.max_tokens, "scorer": "autokv-v3-score-v1",
                    "source_document_ids": [f"{split}:{i}:{t}:{l}"], "evidence_document_ids": [f"{split}:{i}:{t}:{l}"],
                    "background_document_ids": []}
                   for i in range(cfg.per_cell(split)) for t in TASKS for l in cfg.lengths]
            for split in ("development", "experiment", "test")}


class FakeRunner:
    """协议模拟：合成分数不能当作模型质量证据。"""
    def __init__(self, cfg, data, directory, score=None):
        self.config, self.data, self.directory = cfg, data, directory
        self.score = score or (lambda layers, sample: 1.0)
        self.phase, self.rows, self.calls = "endpoints", {}, []
        self.counts = {phase: {"requests": 0, "retries": 0, "server_starts": 0, "seconds": 0.0}
                       for phase in ("development", "endpoints", "search", "test")}
        self.fail_test_once = False
        self.capacity_override = None

    def evaluate(self, policy, split, ids):
        lookup = {s["sample_id"]: s for s in self.data[split]}
        cache = self.rows.setdefault((policy.config_id, split), {})
        missing = [sid for sid in ids if sid not in cache]
        if missing:
            self.calls.append((policy.bf16_layers, split, len(missing)))
            self.counts[self.phase]["server_starts"] += 1
        for sid in missing:
            if self.phase == "test" and self.fail_test_once:
                self.fail_test_once = False
                raise RuntimeError("模拟测试连接中断")
            limit = self.config.raw["search"].get("max_requests")
            if self.phase == "search" and limit is not None and self.counts["search"]["requests"] >= limit:
                raise SearchBudgetExceeded("模拟预算耗尽")
            self.counts[self.phase]["requests"] += 1
            row = {**lookup[sid], "task_score": self.score(frozenset(policy.bf16_layers), lookup[sid]),
                   "output_text": "模拟回答", "output_tokens": 2, "finish_reason": "stop", "error": None}
            cache[sid] = row
            append_jsonl(self.directory / "policies" / policy.config_id / f"{split}.jsonl", row)
        return [cache[sid] for sid in ids]

    def statistics(self):
        return {"phases": copy.deepcopy(self.counts),
                "total": {key: sum(v[key] for v in self.counts.values()) for key in next(iter(self.counts.values()))}}

    def capacities(self, policy, split="test"):
        if self.capacity_override is not None:
            return self.capacity_override(policy)
        return [int(10000*2*self.config.num_layers/(self.config.num_layers+policy.k))]


class CharCodec:
    template_sha256 = "测试模板"
    template_text = "测试模板"

    def render_and_count(self, prompt):
        return prompt, len(prompt)


def source_fixture():
    docs, questions, assignment = {}, [], {}
    for split in ("development", "experiment", "test"):
        for i in range(250):
            did = f"wiki:{split}-{i}"
            docs[did] = {"title": f"{split} document {i}", "text": f"Article {i}. " + "Natural background text. "*70}
            assignment[did] = split
        for i in range(60):
            did = f"wiki:{split}-{i}"
            questions.append({"id": f"single-{split}-{i}", "task": "qa_single", "question": f"Where is city {i}?",
                              "answers": ["Paris"], "evidence": [(did, docs[did]["title"], f"City {i} is Paris.")]})
        for i in range(40):
            ds = [f"wiki:{split}-{60+2*i+j}" for j in (0, 1)]
            questions.append({"id": f"multi-{split}-{i}", "task": "qa_multi", "question": f"Where was person {i} born?",
                              "answers": ["Paris"], "evidence": [(d, docs[d]["title"], f"Evidence {d}: Paris.") for d in ds]})
    return docs, questions, assignment
