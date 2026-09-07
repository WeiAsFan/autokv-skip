import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.export_v2_results import export_results


class V2ExportTests(unittest.TestCase):
    def test_v21_data_and_both_reports_are_exported_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / "runs/test-run"
            (run / "report").mkdir(parents=True)
            (run / "run-manifest.json").write_text(json.dumps({"schema_version": 2}))
            (run / "completed-manifest.json").write_text(json.dumps({"complete": True, "technical_goal_passed": False, "status": "capacity_unverified"}))
            for name, content in (("QUALITY-v2.zh-CN.md", "主报告"), ("RANDOM-v2.zh-CN.md", "随机报告")):
                (run / "report" / name).write_text(content, encoding="utf-8")
            data = root / "data/v2.1/quality"
            data.mkdir(parents=True)
            (data / "calibration.jsonl").write_text("{}\n")
            output = export_results(root)
            self.assertEqual((output / "test-run-QUALITY-v2.zh-CN.md").read_text(encoding="utf-8"), "主报告")
            self.assertEqual((output / "test-run-RANDOM-v2.zh-CN.md").read_text(encoding="utf-8"), "随机报告")
            self.assertIn("主实验未达标", (output / "README.zh-CN.md").read_text(encoding="utf-8"))
            with tarfile.open(output / "autokv-v2.tar.gz") as archive:
                self.assertIn("autokv-skip/data/v2.1/quality/calibration.jsonl", archive.getnames())

    def test_incomplete_runs_logs_and_input_snapshot_survive_split_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / "runs/test-run"
            (run / "inputs/autokv").mkdir(parents=True)
            (run / "run-manifest.json").write_text(json.dumps({"schema_version": 2}))
            (run / "inputs/autokv/example.py").write_text("# 运行时的源码\n", encoding="utf-8")
            (run / "quality/calibration/endpoints").mkdir(parents=True)
            incomplete = run / "quality/calibration/endpoints/.p0.working.jsonl"
            incomplete.write_text('{"error":"test failure"}\n')
            server_log = run / "quality/calibration/endpoints/p0.server.log"
            server_log.write_bytes(os.urandom(8000))
            (root / "runs/v2-run-test.log").write_text("运行失败\n", encoding="utf-8")
            (root / ".cache").mkdir()
            (root / ".cache/model.safetensors").write_bytes(b"weights")
            (root / "data/v2/source").mkdir(parents=True)
            (root / "data/v2/source/private.jsonl").write_text("source")
            with patch("scripts.export_v2_results.PART_BYTES", 1024):
                output = export_results(root)
            parts = sorted(output.glob("*.part-*"))
            self.assertGreater(len(parts), 1)
            self.assertTrue(all(part.stat().st_size <= 1024 for part in parts))
            with tarfile.open(fileobj=io.BytesIO(b"".join(part.read_bytes() for part in parts)), mode="r:gz") as archive:
                names = archive.getnames()
                self.assertIn("autokv-skip/runs/test-run/quality/calibration/endpoints/.p0.working.jsonl", names)
                self.assertIn("autokv-skip/runs/test-run/inputs/autokv/example.py", names)
                self.assertIn("autokv-skip/runs/v2-run-test.log", names)
                self.assertFalse(any(".cache" in name or "data/v2/source" in name for name in names))
                self.assertEqual(archive.extractfile("autokv-skip/runs/test-run/quality/calibration/endpoints/p0.server.log").read(), server_log.read_bytes())
            self.assertIn("未完成", (output / "README.zh-CN.md").read_text(encoding="utf-8"))

    def test_startup_failure_can_be_exported_without_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "runs").mkdir()
            (root / "runs/v2-run-test.log").write_text("model path missing\n")
            output = export_results(root)
            with tarfile.open(output / "autokv-v2.tar.gz") as archive:
                self.assertIn("autokv-skip/runs/v2-run-test.log", archive.getnames())


if __name__ == "__main__":
    unittest.main()
