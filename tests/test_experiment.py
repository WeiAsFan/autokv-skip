import re
import json
import tempfile
import unittest
from pathlib import Path

from autokv.commands import CommandResult
from autokv.config import Profile
from autokv.experiment import (
    ExperimentRunner,
    canonical_run_id,
    is_complete,
    mark_complete,
    safe_remove_stale_container,
    safe_stop_container,
    validate_server_log,
)
from autokv.niah import NiahCase, expected_answer
from autokv.selection import Variant


def _result(argv, returncode=0, stdout="", stderr=""):
    return CommandResult(tuple(argv), returncode, stdout, stderr, 0.01)


class ExperimentTests(unittest.TestCase):
    def test_complete_state_requires_matching_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "rows.jsonl"
            artifact.write_text('{"ok":true}\n', encoding="utf-8")
            mark_complete(root / "state.json", artifact, expected_rows=1)
            self.assertTrue(is_complete(root / "state.json", artifact, expected_rows=1))
            artifact.write_text('{"ok":false}\n', encoding="utf-8")
            self.assertFalse(is_complete(root / "state.json", artifact, expected_rows=1))

    def test_run_id_is_deterministic_and_sensitive_to_image(self):
        first = canonical_run_id("profile", "sha256:a", "model", "data")
        self.assertEqual(first, canonical_run_id("profile", "sha256:a", "model", "data"))
        self.assertNotEqual(first, canonical_run_id("profile", "sha256:b", "model", "data"))

    def test_server_log_requires_flashinfer_dtype_and_capacity(self):
        log = (
            "Using AttentionBackendEnum.FLASHINFER backend.\n"
            "Using fp8_e4m3 data type to store kv cache.\n"
            "GPU KV cache size: 233,104 tokens\n"
        )
        validate_server_log(log, Variant.fp8())
        with self.assertRaisesRegex(ValueError, "FLASHINFER"):
            validate_server_log(log.replace("FLASHINFER", "FLASH_ATTN"), Variant.fp8())

    def test_safe_stop_refuses_foreign_container(self):
        calls = []

        def runner(argv, timeout=None):
            calls.append(tuple(argv))
            return _result(argv, stdout="foreign-project\n")

        with self.assertRaisesRegex(RuntimeError, "refusing to stop"):
            safe_stop_container("autokv-test", runner)
        self.assertFalse(any(call[:2] == ("docker", "stop") for call in calls))

    def test_safe_remove_stale_container_only_removes_stopped_owned_container(self):
        calls = []

        def runner(argv, timeout=None):
            calls.append(tuple(argv))
            if tuple(argv[:2]) == ("docker", "inspect"):
                return _result(argv, stdout="autokv-skip|false\n")
            if tuple(argv[:2]) == ("docker", "rm"):
                return _result(argv, stdout="removed\n")
            raise AssertionError(f"unexpected command: {' '.join(argv)}")

        safe_remove_stale_container("autokv-test", runner)

        self.assertIn(("docker", "rm", "autokv-test"), calls)

    def test_safe_remove_stale_container_refuses_running_container(self):
        def runner(argv, timeout=None):
            return _result(argv, stdout="autokv-skip|true\n")

        with self.assertRaisesRegex(RuntimeError, "already running"):
            safe_remove_stale_container("autokv-test", runner)

    def test_fake_variant_run_writes_scored_resumable_result(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        log = (
            "Using AttentionBackendEnum.FLASHINFER backend.\n"
            "Using fp8_e4m3 data type to store kv cache.\n"
            "GPU KV cache size: 233,104 tokens\n"
        )

        def runner(argv, timeout=None):
            if tuple(argv[:2]) == ("docker", "run"):
                return _result(argv, stdout="container-id\n")
            if tuple(argv[:2]) == ("docker", "logs"):
                return _result(argv, stdout=log)
            if tuple(argv[:2]) == ("docker", "inspect"):
                if "State.Running" in " ".join(argv):
                    return _result(argv, returncode=1, stderr="No such object")
                return _result(argv, stdout="autokv-skip\n")
            if tuple(argv[:2]) == ("docker", "stop"):
                return _result(argv, stdout="stopped\n")
            if tuple(argv[:2]) == ("docker", "rm"):
                return _result(argv, stdout="removed\n")
            raise AssertionError(f"unexpected command: {' '.join(argv)}")

        class FakeClient:
            def health(self):
                return True

            def tokenize(self, text):
                return len(text.split())

            def complete(self, prompt, max_tokens):
                code = re.search(r"VERIFICATION-CODE: ([A-Z0-9-]+)", prompt).group(1)
                return {"choices": [{"text": expected_answer(code)}]}

            def echo_logprobs(self, prompt):
                return {"choices": [{}]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = {
                "image_ref": "vllm/vllm-openai@sha256:" + "f" * 64,
                "image_digest": "sha256:" + "f" * 64,
                "model_revision": "a" * 40,
            }
            experiment = ExperimentRunner(
                profile,
                root,
                lock,
                command_runner=runner,
                client_factory=lambda base_url, model_id: FakeClient(),
            )
            case = NiahCase("fake-1", 2000, 0.5, 42, "ZEBRA-4821")
            artifact = experiment.run_variant(
                Variant.fp8(), (case,), phase="probe", run_id="run1", port=8000
            )
            self.assertTrue(artifact.exists())
            self.assertTrue(
                is_complete(artifact.with_suffix(".state.json"), artifact, 1)
            )
            self.assertIn('"exact_match":1.0', artifact.read_text(encoding="utf-8"))

    def test_request_timeout_restarts_the_exact_configuration_twice_at_most(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        log = (
            "Using AttentionBackendEnum.FLASHINFER backend.\n"
            "Using fp8_e4m3 data type to store kv cache.\n"
            "GPU KV cache size: 233,104 tokens\n"
        )
        starts = []

        def runner(argv, timeout=None):
            if tuple(argv[:2]) == ("docker", "run"):
                starts.append(tuple(argv))
                return _result(argv, stdout="container-id\n")
            if tuple(argv[:2]) == ("docker", "logs"):
                return _result(argv, stdout=log)
            if tuple(argv[:2]) == ("docker", "inspect"):
                if "State.Running" in " ".join(argv):
                    return _result(argv, returncode=1, stderr="No such object")
                return _result(argv, stdout="autokv-skip\n")
            if tuple(argv[:2]) in {("docker", "stop"), ("docker", "rm")}:
                return _result(argv)
            raise AssertionError(f"unexpected command: {' '.join(argv)}")

        clients = []

        class FakeClient:
            def __init__(self, fail):
                self.fail = fail

            def health(self):
                return True

            def tokenize(self, text):
                return len(text.split())

            def complete(self, prompt, max_tokens):
                if self.fail:
                    raise TimeoutError("request timed out")
                code = re.search(r"VERIFICATION-CODE: ([A-Z0-9-]+)", prompt).group(1)
                return {"choices": [{"text": expected_answer(code)}]}

            def echo_logprobs(self, prompt):
                return {"choices": [{}]}

        def client_factory(base_url, model_id):
            client = FakeClient(fail=not clients)
            clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as directory:
            lock = {
                "image_ref": "vllm/vllm-openai@sha256:" + "f" * 64,
                "image_digest": "sha256:" + "f" * 64,
                "model_revision": "a" * 40,
            }
            experiment = ExperimentRunner(
                profile,
                Path(directory),
                lock,
                command_runner=runner,
                client_factory=client_factory,
            )
            artifact = experiment.run_variant(
                Variant.fp8(),
                (NiahCase("retry-1", 2000, 0.5, 42, "RETRY-4821"),),
                phase="probe",
                run_id="retry-run",
                port=8000,
            )

            self.assertEqual(len(starts), 2)
            self.assertTrue(is_complete(artifact.with_suffix(".state.json"), artifact, 1))

    def test_forced_edit_distance_mode_never_calls_echo_logprobs(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        log = (
            "Using AttentionBackendEnum.FLASHINFER backend.\n"
            "Using fp8_e4m3 data type to store kv cache.\n"
            "GPU KV cache size: 233,104 tokens\n"
        )

        def runner(argv, timeout=None):
            if tuple(argv[:2]) == ("docker", "run"):
                return _result(argv)
            if tuple(argv[:2]) == ("docker", "logs"):
                return _result(argv, stdout=log)
            if tuple(argv[:2]) == ("docker", "inspect"):
                if "State.Running" in " ".join(argv):
                    return _result(argv, returncode=1, stderr="No such object")
                return _result(argv, stdout="autokv-skip\n")
            if tuple(argv[:2]) in {("docker", "stop"), ("docker", "rm")}:
                return _result(argv)
            raise AssertionError(f"unexpected command: {' '.join(argv)}")

        class FakeClient:
            def health(self):
                return True

            def tokenize(self, text):
                return len(text.split())

            def complete(self, prompt, max_tokens):
                code = re.search(r"VERIFICATION-CODE: ([A-Z0-9-]+)", prompt).group(1)
                return {"choices": [{"text": expected_answer(code)}]}

            def echo_logprobs(self, prompt):
                raise AssertionError("echo must not be called in edit-distance mode")

        with tempfile.TemporaryDirectory() as directory:
            lock = {
                "image_ref": "vllm/vllm-openai@sha256:" + "f" * 64,
                "image_digest": "sha256:" + "f" * 64,
                "model_revision": "a" * 40,
            }
            experiment = ExperimentRunner(
                profile,
                Path(directory),
                lock,
                command_runner=runner,
                client_factory=lambda base_url, model_id: FakeClient(),
                quality_mode="edit_distance",
            )
            artifact = experiment.run_variant(
                Variant.fp8(),
                (NiahCase("mode-1", 2000, 0.5, 42, "MODE-4821"),),
                phase="probe",
                run_id="mode-run",
                port=8000,
            )
            row = json.loads(artifact.read_text(encoding="utf-8"))

            self.assertEqual(row["quality_mode"], "edit_distance")
            self.assertIsNone(row["answer_nll"])


if __name__ == "__main__":
    unittest.main()
