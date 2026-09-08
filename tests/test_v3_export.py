import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from autokv.io import atomic_write_json, atomic_write_text
from scripts.export_v3_results import export_results
from scripts.prepare_v3_sources import SOURCES, download_hotpot_mirror, prepare


class V3ExportTests(unittest.TestCase):
    def test_partial_run_split_archive_restores_inputs_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs/v3-failed"
            atomic_write_text(run / "policies/abc/attempts/test.log", "服务失败日志\n")
            atomic_write_text(run / "policies/abc/experiment.jsonl", '{"sample_id":"partial"}\n')
            atomic_write_text(run / "inputs/actual.py", "print('实际运行源码')\n")
            atomic_write_json(run / "completed-manifest.json", {"status": "runtime_failed", "technical_goal_passed": False})
            atomic_write_text(root / ".cache/model.bin", "权重不应被导出")
            output = export_results(root, "v3-failed", part_bytes=100)
            parts = sorted(output.glob("*.part-*"))
            self.assertGreater(len(parts), 1)
            with tarfile.open(fileobj=io.BytesIO(b"".join(p.read_bytes() for p in parts)), mode="r:gz") as archive:
                names = archive.getnames()
                self.assertIn("autokv-skip/runs/v3-failed/inputs/actual.py", names)
                self.assertTrue(any(name.endswith("test.log") for name in names))
                self.assertFalse(any(".cache" in name for name in names))
            self.assertIn("runtime_failed", (output / "README.zh-CN.md").read_text(encoding="utf-8"))

    def test_cli_only_failure_without_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_write_text(root / "runs/v3-run-failed.log", "模型路径错误")
            output = export_results(root)
            self.assertIn("仅 CLI 诊断", (output / "README.zh-CN.md").read_text(encoding="utf-8"))
            self.assertTrue((output / "autokv-v3.tar.gz").exists())

    def test_existing_downloads_need_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incoming, output = root / "incoming", root / "source"
            for filename, _ in SOURCES.values():
                atomic_write_json(incoming / filename, [])
            with patch("urllib.request.urlopen", side_effect=AssertionError("本地导入不能访问网络")):
                prepare(output, incoming)
                prepare(output)
            self.assertTrue((output / "source-manifest.json").exists())

    def test_mirror_pagination_restores_complete_support(self):
        calls = []
        def page(url):
            query = parse_qs(urlparse(url).query)
            offset, length = int(query["offset"][0]), int(query["length"][0])
            calls.append((offset, length))
            rows = [{"row_idx": i, "truncated_cells": ["context"] if i == 5 and length > 1 else [],
                     "row": {"id": str(i), "question": f"Question {i}", "answer": "Paris",
                             "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 1]},
                             "context": {"title": ["A", "B"], "sentences": [["First"], ["Second", "Paris"]]}}}
                    for i in range(offset, min(102, offset+length))]
            return {"num_rows_total": 102, "rows": rows}
        with tempfile.TemporaryDirectory() as tmp, patch("scripts.prepare_v3_sources._json_url", side_effect=page):
            path = Path(tmp)/"mirror.json"
            origin = download_hotpot_mirror(path, "train")
            rows = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(origin["rows"], 102)
            self.assertEqual(len(rows), 102)
            self.assertIn((5, 1), calls)
            self.assertEqual(rows[5]["supporting_facts"], [["A", 0], ["B", 1]])
            self.assertEqual(rows[5]["context"][1][1], ["Second", "Paris"])
