import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from autokv.cli import _completion_manifest_is_valid, _write_completion_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "autokv", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class CliTests(unittest.TestCase):
    def test_completion_manifest_hashes_every_active_run_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs" / "run-one"
            run_root.mkdir(parents=True)
            artifact = run_root / "artifact.json"
            artifact.write_text('{"ok":true}\n', encoding="utf-8")

            manifest_path = _write_completion_manifest(
                root, "run-one", "a" * 64
            )

            self.assertTrue(manifest_path.is_file())
            self.assertTrue(_completion_manifest_is_valid(root, "run-one"))
            artifact.write_text('{"ok":false}\n', encoding="utf-8")
            self.assertFalse(_completion_manifest_is_valid(root, "run-one"))
            artifact.write_text('{"ok":true}\n', encoding="utf-8")
            self.assertTrue(_completion_manifest_is_valid(root, "run-one"))
            (run_root / "unlisted.json").write_text("{}\n", encoding="utf-8")
            self.assertFalse(_completion_manifest_is_valid(root, "run-one"))

    def test_help_lists_complete_server_workflow(self):
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
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
            self.assertIn(command, result.stdout)
        probe_help = run_cli("probe", "--help")
        self.assertEqual(probe_help.returncode, 0, probe_help.stderr)
        self.assertIn("--force", probe_help.stdout)

    def test_quick_dry_run_is_json_and_never_invokes_docker(self):
        result = run_cli("dry-run", "--profile", "quick", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["profile"], "quick")
        self.assertEqual(data["core_probe_configurations"], 18)
        self.assertEqual(data["quality_configurations"], 11)
        self.assertEqual(data["benchmark_scenarios"], 6)
        self.assertFalse(data["executed"])
        self.assertIn("VLLM_ENABLE_CUDA_COMPATIBILITY=1", data["server_command_example"])
        self.assertNotIn("docker was executed", result.stderr.lower())

    def test_full_dry_run_has_thirty_four_core_configurations(self):
        result = run_cli("dry-run", "--profile", "full", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["profile"], "full")
        self.assertEqual(data["core_probe_configurations"], 34)
        self.assertEqual(data["benchmark_scenarios"], 18)
        self.assertFalse(data["executed"])

    def test_make_data_is_deterministic_and_status_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_cli(
                "make-data",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            second = run_cli(
                "make-data",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            first_data = json.loads(first.stdout)
            second_data = json.loads(second.stdout)
            self.assertEqual(first_data["dataset_hash"], second_data["dataset_hash"])
            self.assertEqual(first_data["probe_samples"], 6)
            self.assertEqual(first_data["final_samples"], 18)
            self.assertTrue(Path(first_data["manifest"]).is_file())

            before = sorted(path.relative_to(root) for path in root.rglob("*"))
            status = run_cli(
                "status",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            status_data = json.loads(status.stdout)
            self.assertTrue(status_data["data_ready"])
            after = sorted(path.relative_to(root) for path in root.rglob("*"))
            self.assertEqual(before, after)

    def test_status_rejects_a_forged_dataset_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            made = run_cli(
                "make-data",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(made.returncode, 0, made.stderr)
            manifest_path = Path(json.loads(made.stdout)["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_hash"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            status = run_cli(
                "status",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertFalse(json.loads(status.stdout)["data_ready"])

    def test_diagnose_excludes_cache_and_redacts_hf_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "runs" / "_environment"
            environment.mkdir(parents=True)
            (environment / "doctor.json").write_text(
                '{"message":"token super-secret-token '
                'https://example.test/file?X-Amz-Signature=signed-secret&part=1"}\n',
                encoding="utf-8",
            )
            cache = root / ".cache" / "huggingface"
            cache.mkdir(parents=True)
            (cache / "model.bin").write_bytes(b"model")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "autokv",
                    "diagnose",
                    "--project-root",
                    str(root),
                    "--profile",
                    "quick",
                    "--json",
                ],
                cwd=REPOSITORY_ROOT,
                env={**__import__("os").environ, "HF_TOKEN": "super-secret-token"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            archive = Path(json.loads(result.stdout)["archive"])
            self.assertTrue(archive.is_file())
            with tarfile.open(archive, "r:gz") as handle:
                names = handle.getnames()
                self.assertFalse(any(".cache" in name for name in names))
                doctor_member = next(name for name in names if name.endswith("doctor.json"))
                body = handle.extractfile(doctor_member).read().decode("utf-8")
                self.assertNotIn("super-secret-token", body)
                self.assertNotIn("signed-secret", body)
                self.assertIn("***", body)

    def test_real_gpu_command_refuses_non_linux_host(self):
        if sys.platform.startswith("linux"):
            self.skipTest("this guard is specifically exercised on the local non-Linux host")
        result = run_cli("doctor", "--profile", "quick", "--json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Linux", result.stderr)
        self.assertIn("不要修改驱动", result.stderr)

    def test_select_and_report_remain_offline_cross_platform_commands(self):
        if sys.platform.startswith("linux"):
            self.skipTest("this regression specifically distinguishes the non-Linux gate")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            made = run_cli(
                "make-data",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(made.returncode, 0, made.stderr)
            environment = root / "runs" / "_environment"
            environment.mkdir(parents=True)
            (environment / "lock.json").write_text(
                json.dumps(
                    {
                        "image_ref": "vllm/vllm-openai@sha256:" + "f" * 64,
                        "image_digest": "sha256:" + "f" * 64,
                        "model_revision": "a" * 40,
                        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                        "host": {"driver": "535.230.02"},
                    }
                ),
                encoding="utf-8",
            )
            for command, expected in (("select", "probe index"), ("report", "selection")):
                result = run_cli(
                    command,
                    "--project-root",
                    str(root),
                    "--profile",
                    "quick",
                    "--json",
                )
                self.assertEqual(result.returncode, 5, result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("Linux", result.stderr)

    def test_select_recomputes_top_and_control_layers_from_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            made = run_cli(
                "make-data",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(made.returncode, 0, made.stderr)
            made_payload = json.loads(made.stdout)
            environment = root / "runs" / "_environment"
            environment.mkdir(parents=True)
            (environment / "lock.json").write_text(
                json.dumps(
                    {
                        "image_ref": "vllm/vllm-openai@sha256:" + "f" * 64,
                        "image_digest": "sha256:" + "f" * 64,
                        "model_revision": "a" * 40,
                        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                        "host": {"driver": "535.230.02"},
                    }
                ),
                encoding="utf-8",
            )
            status = run_cli(
                "status",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            run_id = json.loads(status.stdout)["run_id"]
            smoke_dir = root / "runs" / run_id / "smoke"
            smoke_dir.mkdir(parents=True)
            smoke_records = []
            for name in ("first", "second"):
                path = smoke_dir / f"{name}.jsonl"
                path.write_text(
                    json.dumps({"quality_mode": "edit_distance"}) + "\n",
                    encoding="utf-8",
                )
                smoke_records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            (smoke_dir / "smoke.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "deterministic": True,
                        "run_id": run_id,
                        "quality_mode": "edit_distance",
                        "first": smoke_records[0],
                        "second": smoke_records[1],
                    }
                ),
                encoding="utf-8",
            )
            probe_dir = root / "runs" / run_id / "probe"
            probe_dir.mkdir(parents=True)
            artifacts = {}

            def add_artifact(name: str, quality: float):
                path = probe_dir / f"{name}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "sample_id": "one",
                            "quality_mode": "edit_distance",
                            "quality_score": quality,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                artifacts[name] = {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            add_artifact("fp8", 0.2)
            group_scores = {}
            for group in range(8):
                quality = 0.21 + group / 100
                add_artifact(f"group-{group:02d}", quality)
                group_scores[f"group-{group:02d}"] = quality - 0.2
            for layer in range(8):
                add_artifact(f"layer-{layer:02d}", 0.2 + layer / 100)
            add_artifact("auto-4", 0.5)
            (probe_dir / "index.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "run_id": run_id,
                        "dataset_hash": made_payload["dataset_hash"],
                        "quality_mode": "edit_distance",
                        "group_scores": group_scores,
                        "candidate_layers": list(range(8)),
                        "auto_layers": [4, 5, 6, 7],
                        "artifacts": artifacts,
                    }
                ),
                encoding="utf-8",
            )
            selected = run_cli(
                "select",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(selected.returncode, 0, selected.stderr)
            payload = json.loads(selected.stdout)
            self.assertEqual(payload["auto_layers"], [4, 5, 6, 7])
            self.assertEqual(payload["inverted_layers"], [0, 1, 2, 3])
            self.assertEqual(payload["quality_mode"], "edit_distance")
            controls = [tuple(layers) for layers in payload["random_layers"]]
            self.assertEqual(len(set(controls)), 5)
            self.assertNotIn((4, 5, 6, 7), controls)
            run_manifest = json.loads(
                (root / "runs" / run_id / "run-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(run_manifest["run_id"], run_id)
            self.assertRegex(run_manifest["source"]["tree_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("git_commit", run_manifest["source"])
            self.assertEqual(
                run_manifest["profile_sha256"],
                json.loads(
                    (root / "data/niah/quick-manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["profile_sha256"],
            )
            self.assertEqual(run_manifest["storage_timezone"], "UTC")
            self.assertEqual(run_manifest["display_timezone"], "Asia/Shanghai")

            valid_status = run_cli(
                "status",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(valid_status.returncode, 0, valid_status.stderr)
            valid_steps = json.loads(valid_status.stdout)["steps"]
            self.assertTrue(valid_steps["smoke"])
            self.assertTrue(valid_steps["probe"])
            self.assertTrue(valid_steps["select"])

            (probe_dir / "layer-00.jsonl").write_text(
                "corrupted\n", encoding="utf-8"
            )
            invalid_status = run_cli(
                "status",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(invalid_status.returncode, 0, invalid_status.stderr)
            invalid_steps = json.loads(invalid_status.stdout)["steps"]
            self.assertTrue(invalid_steps["smoke"])
            self.assertFalse(invalid_steps["probe"])
            self.assertFalse(invalid_steps["select"])


if __name__ == "__main__":
    unittest.main()
