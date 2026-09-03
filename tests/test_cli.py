import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from autokv.cli import (
    RuntimeContext,
    _canonical_source_identity,
    _completion_manifest_is_valid,
    _ensure_run_manifest,
    _write_completion_manifest,
)
from autokv.config import Profile
from autokv.niah import NiahCase, expected_answer, make_cases
from autokv.selection import Variant, canonical_config_id, group_probe_variants


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
    def test_runtime_source_identity_ignores_git_only_provenance_changes(self):
        runtime_files = [{"path": "autokv/cli.py", "sha256": "b" * 64}]
        first = {
            "tree_sha256": "a" * 64,
            "files": runtime_files,
            "git_commit": "c" * 40,
            "git_dirty": False,
        }
        second = {
            **first,
            "git_commit": "d" * 40,
            "git_dirty": True,
        }
        self.assertEqual(
            _canonical_source_identity(first),
            _canonical_source_identity(second),
        )

    def test_run_manifest_keeps_run_start_git_provenance_for_same_runtime_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = Profile.from_dict(Profile.default_dict("quick"))
            profile_path = root / "configs" / "quick.json"
            profile_path.parent.mkdir(parents=True)
            profile_path.write_text("{}\n", encoding="utf-8")
            source = {
                "tree_sha256": "a" * 64,
                "files": [{"path": "autokv/cli.py", "sha256": "b" * 64}],
                "git_commit": "c" * 40,
                "git_dirty": False,
            }
            common = {
                "root": root,
                "profile": profile,
                "profile_path": profile_path,
                "profile_hash": "d" * 64,
                "manifest": {},
                "dataset_hash": "e" * 64,
                "lock": {
                    "image_ref": "image@sha256:" + "f" * 64,
                    "image_digest": "sha256:" + "f" * 64,
                    "model_revision": "1" * 40,
                },
                "run_id": "same-runtime",
            }
            first = RuntimeContext(source=source, **common)
            path = _ensure_run_manifest(first)
            second = RuntimeContext(
                source={
                    **source,
                    "git_commit": "2" * 40,
                    "git_dirty": True,
                },
                **common,
            )
            self.assertEqual(_ensure_run_manifest(second), path)
            observed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(observed["source"]["git_commit"], "c" * 40)

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
            "v2-freeze-data",
            "v2-run",
            "v2-pilot",
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
        self.assertEqual(
            set(data["server_commands"]),
            {"bf16", "fp8", "auto-4-placeholder"},
        )
        self.assertIn("--kv-cache-dtype bfloat16", data["server_commands"]["bf16"])
        self.assertIn("--kv-cache-dtype fp8_e4m3", data["server_commands"]["fp8"])
        self.assertIn(
            "--kv-cache-dtype-skip-layers 0 1 2 3",
            data["server_commands"]["auto-4-placeholder"],
        )
        self.assertIn("bench serve", data["benchmark_command_example"])
        self.assertIn(
            "runs/<immutable-run-id>/perf/*.matrix.state.json",
            data["planned_artifacts"],
        )
        self.assertIn(
            "runs/<immutable-run-id>/report/REPORT.zh-CN.md",
            data["planned_artifacts"],
        )
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
                        "host": {"driver": "580.173.02"},
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
            image_ref = "vllm/vllm-openai@sha256:" + "f" * 64
            image_digest = "sha256:" + "f" * 64
            model_revision = "a" * 40
            (environment / "lock.json").write_text(
                json.dumps(
                    {
                        "image_ref": image_ref,
                        "image_digest": image_digest,
                        "model_revision": model_revision,
                        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                        "host": {"driver": "580.173.02"},
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
            profile = Profile.from_dict(Profile.default_dict("quick"))
            smoke_dir = root / "runs" / run_id / "smoke"
            smoke_dir.mkdir(parents=True)

            def experiment_row(
                case: NiahCase, variant: Variant, quality: float
            ) -> dict[str, object]:
                answer = expected_answer(case.code)
                return {
                    "schema_version": 1,
                    "run_id": run_id,
                    "config_id": canonical_config_id(variant),
                    "image_digest": image_digest,
                    "model_revision": model_revision,
                    "backend": "FLASHINFER",
                    "kv_dtype": variant.kv_dtype,
                    "skip_layers": list(variant.skip_layers),
                    "seed": case.seed,
                    "sample_id": case.sample_id,
                    "prompt_tokens": case.target_tokens,
                    "output_tokens": 1,
                    "needle_depth": case.depth,
                    "expected": answer,
                    "output_text": answer,
                    "exact_match": 1.0,
                    "edit_distance": 0,
                    "answer_nll": None,
                    "quality_mode": "edit_distance",
                    "quality_score": quality,
                    "ttft_ms": None,
                    "e2e_ms": 1.0,
                    "timestamp_utc": "2026-08-25T00:00:00+00:00",
                }

            def add_completion_evidence(
                path: Path,
                rows: int,
                variant: Variant,
                command_quality_mode: str,
            ):
                server_log = path.with_suffix(".server.log")
                command_record = path.with_suffix(".command.json")
                state = path.with_suffix(".state.json")
                argv = [
                    "--attention-backend",
                    "FLASHINFER",
                    "--kv-cache-dtype",
                    variant.kv_dtype,
                ]
                log_lines = [
                    "Using FLASHINFER backend",
                    "GPU KV cache size: 233,104 tokens",
                ]
                if variant.kv_dtype == "fp8_e4m3":
                    log_lines.append(
                        "Using fp8_e4m3 data type to store kv cache."
                    )
                else:
                    log_lines.append("Engine config: kv_cache_dtype=bfloat16")
                if variant.skip_layers:
                    argv.append("--kv-cache-dtype-skip-layers")
                    argv.extend(str(layer) for layer in variant.skip_layers)
                    skipped = set(variant.skip_layers)
                    log_lines.extend(
                        "Layer model.layers."
                        f"{layer}.self_attn: kv_cache_dtype="
                        f"{'auto' if layer in skipped else 'fp8_e4m3'}, "
                        "sliding_window=None"
                        for layer in range(32)
                    )
                server_log.write_text(
                    "\n".join(log_lines) + "\n", encoding="utf-8"
                )
                command_record.write_text(
                    json.dumps(
                        {
                            "variant": {
                                "name": variant.name,
                                "kv_dtype": variant.kv_dtype,
                                "skip_layers": list(variant.skip_layers),
                            },
                            "image_ref": image_ref,
                            "model_revision": model_revision,
                            "quality_mode": command_quality_mode,
                            "inspected_argv": argv,
                        }
                    ),
                    encoding="utf-8",
                )
                state.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "complete": True,
                            "artifact": path.name,
                            "artifact_sha256": hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest(),
                            "rows": rows,
                            "evidence": {
                                "server_log": {
                                    "path": server_log.name,
                                    "sha256": hashlib.sha256(
                                        server_log.read_bytes()
                                    ).hexdigest(),
                                },
                                "command_record": {
                                    "path": command_record.name,
                                    "sha256": hashlib.sha256(
                                        command_record.read_bytes()
                                    ).hexdigest(),
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            smoke_records = []
            smoke_case = NiahCase(
                "smoke-1024-50-42", 1024, 0.5, 42, "KV-SMOKE-4242"
            )
            smoke_variant = Variant.fp8()
            smoke_stem = (
                f"{smoke_variant.name}-{canonical_config_id(smoke_variant)}.jsonl"
            )
            for phase in ("smoke-a", "smoke-b"):
                path = root / "runs" / run_id / phase / smoke_stem
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(experiment_row(smoke_case, smoke_variant, 1.0))
                    + "\n",
                    encoding="utf-8",
                )
                add_completion_evidence(path, 1, smoke_variant, "auto")
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
            probe_cases = make_cases(profile, "probe")

            def add_artifact(variant: Variant, quality: float):
                path = probe_dir / (
                    f"{variant.name}-{canonical_config_id(variant)}.jsonl"
                )
                path.write_text(
                    "".join(
                        json.dumps(experiment_row(case, variant, quality))
                        + "\n"
                        for case in probe_cases
                    ),
                    encoding="utf-8",
                )
                add_completion_evidence(
                    path, len(probe_cases), variant, "edit_distance"
                )
                artifacts[variant.name] = {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            add_artifact(Variant.fp8(), 0.2)
            group_scores = {}
            for group, variant in enumerate(group_probe_variants(32, 4)):
                quality = 0.28 - group / 100
                add_artifact(variant, quality)
                group_scores[variant.name] = quality - 0.2
            for layer in range(8):
                add_artifact(
                    Variant.mixed(f"layer-{layer:02d}", (layer,)),
                    0.2 + layer / 100,
                )
            add_artifact(Variant.mixed("auto-4", (4, 5, 6, 7)), 0.5)
            (probe_dir / "index.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "run_id": run_id,
                        "dataset_hash": made_payload["dataset_hash"],
                        "quality_mode": "edit_distance",
                        "group_scores": group_scores,
                        "selected_groups": [list(range(4)), list(range(4, 8))],
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

            probe_index_path = probe_dir / "index.json"
            original_probe_index = probe_index_path.read_text(encoding="utf-8")
            layer_variant = Variant.mixed("layer-00", (0,))
            layer_path = probe_dir / (
                f"{layer_variant.name}-{canonical_config_id(layer_variant)}.jsonl"
            )
            foreign_dir = root / "runs" / "foreign-run" / "probe"
            foreign_dir.mkdir(parents=True)
            foreign_layer_path = foreign_dir / layer_path.name
            for source in (
                layer_path,
                layer_path.with_suffix(".state.json"),
                layer_path.with_suffix(".server.log"),
                layer_path.with_suffix(".command.json"),
            ):
                (foreign_dir / source.name).write_bytes(source.read_bytes())
            redirected_index = json.loads(original_probe_index)
            redirected_index["artifacts"]["layer-00"]["path"] = (
                foreign_layer_path.relative_to(root).as_posix()
            )
            probe_index_path.write_text(
                json.dumps(redirected_index), encoding="utf-8"
            )
            redirected_status = run_cli(
                "status",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(
                redirected_status.returncode, 0, redirected_status.stderr
            )
            redirected_steps = json.loads(redirected_status.stdout)["steps"]
            self.assertTrue(redirected_steps["smoke"])
            self.assertFalse(redirected_steps["probe"])
            self.assertFalse(redirected_steps["select"])
            probe_index_path.write_text(original_probe_index, encoding="utf-8")

            original_layer = layer_path.read_text(encoding="utf-8")
            layer_state_path = layer_path.with_suffix(".state.json")
            original_layer_state = layer_state_path.read_text(encoding="utf-8")
            forged_rows = [json.loads(line) for line in original_layer.splitlines()]
            for row in forged_rows:
                row["run_id"] = "foreign-run"
            layer_path.write_text(
                "".join(json.dumps(row) + "\n" for row in forged_rows),
                encoding="utf-8",
            )
            forged_hash = hashlib.sha256(layer_path.read_bytes()).hexdigest()
            forged_state = json.loads(original_layer_state)
            forged_state["artifact_sha256"] = forged_hash
            layer_state_path.write_text(json.dumps(forged_state), encoding="utf-8")
            forged_index = json.loads(original_probe_index)
            forged_index["artifacts"]["layer-00"]["sha256"] = forged_hash
            probe_index_path.write_text(json.dumps(forged_index), encoding="utf-8")
            forged_status = run_cli(
                "status",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(forged_status.returncode, 0, forged_status.stderr)
            forged_steps = json.loads(forged_status.stdout)["steps"]
            self.assertTrue(forged_steps["smoke"])
            self.assertFalse(forged_steps["probe"])
            self.assertFalse(forged_steps["select"])
            layer_path.write_text(original_layer, encoding="utf-8")
            layer_state_path.write_text(original_layer_state, encoding="utf-8")
            probe_index_path.write_text(original_probe_index, encoding="utf-8")

            selection_path = root / "runs" / run_id / "selection.json"
            original_selection = selection_path.read_text(encoding="utf-8")
            tampered_selection = json.loads(original_selection)
            tampered_selection["auto_layers"] = [0, 1, 2, 3]
            selection_path.write_text(
                json.dumps(tampered_selection), encoding="utf-8"
            )
            tampered_status = run_cli(
                "status",
                "--project-root",
                str(root),
                "--profile",
                "quick",
                "--json",
            )
            self.assertEqual(tampered_status.returncode, 0, tampered_status.stderr)
            self.assertFalse(json.loads(tampered_status.stdout)["steps"]["select"])
            selection_path.write_text(original_selection, encoding="utf-8")

            layer_path.write_text("corrupted\n", encoding="utf-8")
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
