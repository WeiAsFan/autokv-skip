from pathlib import Path

from autokv.client import VllmHttpError
from autokv.io import read_json
from autokv.v3_runtime import V3PolicyRunner
from autokv.v4_config import V4Config
from autokv.v4_data import PHASES
from autokv.v4_metrics import score_output
from tests.test_v3_runtime import RuntimeHarness
from tests.v3_helpers import CharCodec as V3CharCodec


def config(**overrides):
    raw = read_json(Path(__file__).resolve().parents[1] / "configs/v4.0/quality.json")
    raw["model"].update(num_layers=8, max_model_len=4096)
    raw["construction"].update(tasks=["single_lookup"], target_lengths=[1600], record_counts=[4],
                               discovery_per_condition=6, confirmation_per_batch=10)
    raw["data"].update(experiment_samples=12, test_samples=20, input_tolerance_tokens=8)
    raw["search"].update(fidelity_samples=[4, 8, 12], max_requests=2000)
    raw["scoring"]["bootstrap_samples"] = 20
    for section, values in overrides.items():
        raw[section].update(values)
    return V4Config.from_dict(raw)


class CharCodec(V3CharCodec):
    def encode_text(self, text): return list(text)
    def decode_tokens(self, tokens): return "".join(tokens)


def sources():
    docs = {f"{phase}:{i}": {"title": f"Article {phase} {i}",
                            "text": (f"Natural background about article {i}, trees and birds. "*100)}
            for phase in PHASES for i in range(5)}
    return {"docs": docs, "pools": {p: sorted(d for d in docs if d.startswith(p+":")) for p in PHASES},
            "manifest": {}, "allocation": {p: 5 for p in PHASES}}


class Harness(RuntimeHarness):
    """仅替换外部服务和 tokenizer；使用生产运行器和逐题缓存。"""
    def __init__(self, cfg, root, directory, score=None):
        super().__init__(cfg, root, {})
        self.directory = directory
        self.score = score or (lambda layers, r: len(layers) == cfg.num_layers or layers == {2} or r["index"] % 2 == 0)
        self.layers = set()
        self.service_calls = []
        self.capacity = None

    def start(self, argv, log_path, **kwargs):
        process = super().start(argv, log_path, **kwargs)
        dtype = argv[argv.index("--kv-cache-dtype")+1]
        self.layers = set(range(self.cfg.num_layers)) if dtype != "fp8_e4m3" else set()
        if "--kv-cache-dtype-skip-layers" in argv:
            self.layers = set(map(int, argv[argv.index("--kv-cache-dtype-skip-layers")+1:]))
        log = process.log_text()
        capacity = self.capacity(self.layers) if self.capacity else int(100000*2*self.cfg.num_layers/(self.cfg.num_layers+len(self.layers)))
        log = log.replace("GPU KV cache size: 100,000 tokens", f"GPU KV cache size: {capacity:,} tokens" if capacity else "capacity unavailable")
        log_path.write_text(log, encoding="utf-8")
        process.log_text = lambda: log
        self.lookup = {r["user_prompt"]: r for rows in self.active.samples.values() for r in rows.values()}
        return process

    def complete(self, prompt, max_tokens):
        self.count += 1
        if self.count == self.fail_at:
            raise VllmHttpError(400, "模拟 HTTP 中断")
        row = self.lookup[prompt]
        if row["split"] == "test":
            assert (self.directory / "selection.json").exists(), "测试前必须冻结"
        self.service_calls.append((frozenset(self.layers), row["split"], row["sample_id"]))
        answer = row["expected_answers"][0] if self.score(self.layers, row) else "000000"
        return {"choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": row["prompt_tokens"], "completion_tokens": 3}}

    def runner(self, cfg=None, environment=None):
        self.active = V3PolicyRunner(cfg or self.cfg, self.root, self.directory, environment or self.environment, {},
                                     port=self.port, client_factory=lambda *_: self.client, process_factory=self.start,
                                     scorer=score_output)
        return self.active
