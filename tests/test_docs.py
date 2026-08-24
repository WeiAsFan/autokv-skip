import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_runbook_has_every_numbered_phase_and_no_unfinished_marker(self):
        text = (ROOT / "RUNBOOK.zh-CN.md").read_text(encoding="utf-8")
        for phase in range(14):
            self.assertIn(f"阶段 {phase}", text)
        for marker in ("T" + "BD", "T" + "ODO", "CHANGE" + "ME"):
            self.assertNotIn(marker, text)
        self.assertIn("535.230.02", text)
        self.assertIn("VLLM_ENABLE_CUDA_COMPATIBILITY=1", text)
        self.assertIn("不要升级驱动", text)

    def test_runbook_contains_every_cli_step_and_operator_contract(self):
        text = (ROOT / "RUNBOOK.zh-CN.md").read_text(encoding="utf-8")
        for command in (
            "doctor",
            "lock-image",
            "make-data",
            "dry-run",
            "smoke",
            "probe",
            "select",
            "evaluate",
            "benchmark",
            "report",
            "run",
            "status",
            "diagnose",
        ):
            self.assertIn(f"python3 -m autokv {command}", text)
        for phrase in (
            "预期时间",
            "成功判据",
            "失败分支",
            "退出码 2",
            "退出码 3",
            "退出码 4",
            "退出码 5",
            "tmux",
            "SSH",
            "至少 80 GiB",
            "最多尝试 3 次",
            "最多重启 2 次",
            "quality_mode",
            "--force CONFIG_ID",
            "_superseded",
            "nvidia-smi dmon",
            "run-manifest.json",
            "completed-manifest.json",
            "performance-by-scenario.csv",
            "源码树 SHA-256",
        ):
            self.assertIn(phrase, text)

    def test_research_is_cited_and_covers_the_approved_landscape(self):
        text = (
            ROOT / "docs" / "research" / "inference-optimization-landscape.zh-CN.md"
        ).read_text(encoding="utf-8")
        for phrase in (
            "FlashAttention",
            "PagedAttention",
            "FlashInfer",
            "量化",
            "KV Cache 压缩",
            "算子融合",
            "vLLM",
            "SGLang",
            "KVTuner",
            "AutoKV-Skip",
            "不是算法首创",
        ):
            self.assertIn(phrase, text)
        links = set(re.findall(r"https://[^)\s>]+", text))
        self.assertGreaterEqual(len(links), 20)

    def test_interview_guide_has_three_depths_and_null_result_language(self):
        text = (
            ROOT / "docs" / "interview" / "AutoKV-Skip-interview-guide.zh-CN.md"
        ).read_text(encoding="utf-8")
        for phrase in (
            "30 秒",
            "5 分钟",
            "15 分钟",
            "负结果",
            "1.7778",
            "A6000 不具备原生 FP8 Tensor Core",
            "为什么不用 SGLang",
            "为什么不是 KVTuner",
        ):
            self.assertIn(phrase, text)

    def test_readme_distinguishes_local_verification_from_server_validation(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("本地代码验证", text)
        self.assertIn("目标服务器尚未验证", text)
        self.assertIn("python scripts/verify.py", text)


if __name__ == "__main__":
    unittest.main()
