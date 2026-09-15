import math
import tempfile
import unittest
from pathlib import Path

from autokv.fp4_runtime import patch_backend
from autokv.io import read_json
from autokv.v2_policy import Policy, endpoint_policies, theoretical_capacity
from autokv.v4_config import V4Config, load_config
from autokv.v4_pipeline import execute_v4
from scripts.export_v41_results import export_results
from tests.v4_helpers import CharCodec, Harness, config, sources

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None, "数值测试需要 PyTorch；不要求 GPU")
class FP4NumericalTests(unittest.TestCase):
    def test_codebook_and_ties_round_even(self):
        from autokv.fp4_cache import quantize, dequantize
        values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -1, -1.5, -2, -3, -4, -6, 0.])
        packed, scales = quantize(values)
        self.assertEqual(packed.numel()+scales.numel(), 9)
        torch.testing.assert_close(dequantize(packed, scales), values)
        values = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6]*2)
        packed, scales = quantize(values)
        torch.testing.assert_close(dequantize(packed, scales), torch.tensor([0, 1, 1, 2, 2, 4, 4, 6.]*2))

    def test_zero_tiny_large_and_nonunit_global_scale(self):
        from autokv.fp4_cache import quantize, dequantize
        for multiplier in (0, 1e-5, .03, 1, 23):
            values = torch.linspace(-6, 6, 128).reshape(2, 64)*multiplier
            packed, scales = quantize(values, .5)
            result = dequantize(packed, scales, .5)
            self.assertTrue(torch.isfinite(result).all())
            self.assertLessEqual(float((result-values).abs().max()), multiplier*1.2+1e-4)

    def test_page_offsets_padding_and_exact_storage(self):
        from autokv.fp4_cache import cache_views, quantize, write_cache
        cache = torch.zeros((4, 4, 3, 36), dtype=torch.uint8)
        torch.manual_seed(9)
        key, value = torch.randn(6, 2, 64), torch.randn(6, 2, 64)*3
        slots = torch.tensor([7, 0, -1, 11, 4])
        write_cache(key, value, cache, slots, 2, 64, .5, 2.)
        self.assertEqual(cache.numel(), 4*3*2*2*64*9//16)
        for tensor, (data, sf), scale in zip((key, value), cache_views(cache, 2, 64), (.5, 2.)):
            for index, slot in enumerate(slots.tolist()):
                if slot < 0:
                    continue
                packed, scales = quantize(tensor[index], scale)
                torch.testing.assert_close(data[slot//3, :, slot%3], packed)
                torch.testing.assert_close(sf.view(torch.uint8)[slot//3, :, slot%3], scales.view(torch.uint8))
                # 独立核对物理字节，避免写入器和视图函数共享错误却互相通过。
                side = 0 if scale == .5 else 1
                base = slot//3 * (4*3*36) + side*(2*3*36)
                flat = cache.flatten()
                for head in range(2):
                    offset = base + (head*3+slot%3)*32
                    torch.testing.assert_close(flat[offset:offset+32], packed[head])
                    offset = base + 2*3*32 + (head*3+slot%3)*4
                    torch.testing.assert_close(flat[offset:offset+4], scales.view(torch.uint8)[head])
            self.assertEqual(int(data[0, :, 1].sum()), 0)


class FP4Harness(Harness):
    def start(self, argv, log_path, **kwargs):
        # 仅替换外部服务；生产运行器实际 argv 留在 attempt.json。
        fake = tuple("fp8_e4m3" if x == "nvfp4" else x for x in argv if x != "--enforce-eager")
        self.capacity = lambda layers: int(100000*2*self.cfg.num_layers/(2*len(layers)+(self.cfg.num_layers-len(layers))*9/16))
        process = super().start(fake, log_path, **kwargs)
        log = process.log_text().replace("FP8_E4M3", "NVFP4").replace("fp8_e4m3", "nvfp4")
        process.log_text = lambda: log
        log_path.write_text(log, encoding="utf-8")
        return process


class FP4ProtocolTests(unittest.TestCase):
    def test_capacity_and_policy_identity(self):
        cfg = load_config(ROOT, "configs/v4.1/quality.json")
        self.assertEqual(cfg.max_layers, 17)
        p32, p0 = endpoint_policies(32, cfg.low_kv_dtype)
        self.assertAlmostEqual(theoretical_capacity(p0)["capacity_ratio_vs_p32"], 32/9)
        self.assertEqual(theoretical_capacity(p32)["bytes_per_token"], 131072)
        self.assertNotEqual(p0.config_id, endpoint_policies()[1].config_id)
        self.assertIn("fp4_layers", Policy("mix", (1, 7), 32, "nvfp4").record())

    def test_full_fp4_chain_and_export(self):
        raw = config().raw
        raw["experiment_version"] = "4.1"
        raw["runtime"].update(kv_cache_dtype="nvfp4", enforce_eager=True)
        raw["construction"]["min_fp4_gap"] = raw["construction"].pop("min_fp8_gap")
        cfg = V4Config.from_dict(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root / "runs/v41-test"
            h = FP4Harness(cfg, root, directory)
            result = execute_v4(cfg, root, directory, CharCodec(), sources, h.runner(), {})
            self.assertTrue(result["recovery_demonstrated"])
            self.assertEqual(result["candidate"]["bf16_layers"], [2])
            self.assertIn("fp4_score", result["test_data_conditions"])
            for attempt in directory.glob("policies/*/attempts/*.json"):
                argv = read_json(attempt)["argv"]
                self.assertIn("--enforce-eager", argv)
                self.assertIn(argv[argv.index("--kv-cache-dtype")+1], ("nvfp4", "bfloat16"))
            report = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("v4.1", report)
            self.assertNotIn("FP8 scale", report)
            output = export_results(root, directory.name)
            self.assertIn("v4.1 分支", (output/"README.zh-CN.md").read_text(encoding="utf-8"))

    def test_attention_scale_regression(self):
        # BF16 Q 未除以 q_scale；再乘 q_scale 会错误改变 softmax 温度。
        def probability(scale):
            return math.exp(4*scale)/(math.exp(4*scale)+1)
        self.assertGreater(probability(1), .98)
        self.assertLess(probability(.01), .52)
        with self.assertRaises(ValueError):
            patch_backend("unknown backend")
