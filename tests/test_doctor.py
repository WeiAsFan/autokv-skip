import tempfile
import unittest
from pathlib import Path

from autokv.commands import CommandResult
from autokv.config import Profile
from autokv.doctor import (
    REQUIRED_FLAGS,
    classify_pull_failure,
    fallback_allowed,
    missing_required_flags,
    lock_first_compatible_image,
    parse_cuda_probe,
    parse_gpu_csv,
    parse_hf_revision,
    parse_image_inspect,
    pull_with_retry,
    validate_host,
)


class DoctorTests(unittest.TestCase):
    def test_parses_expected_a6000(self):
        facts = parse_gpu_csv("NVIDIA RTX A6000, 580.173.02, 49140, 8.6\n")
        self.assertEqual(facts.driver, "580.173.02")
        self.assertEqual(facts.compute_capability, "8.6")
        self.assertEqual(facts.vram_mib, 49140)

    def test_rejects_driver_change(self):
        facts = parse_gpu_csv("NVIDIA RTX A6000, 550.54.15, 49140, 8.6\n")
        gates = validate_host(facts, expected_driver="580.173.02")
        self.assertFalse(all(gate.ok for gate in gates))
        driver_gate = next(gate for gate in gates if gate.name == "driver")
        self.assertIn("不要修改驱动", driver_gate.remediation)

    def test_network_pull_error_does_not_allow_version_fallback(self):
        self.assertEqual(classify_pull_failure("TLS handshake timeout"), "network")
        self.assertFalse(fallback_allowed("pull", "network"))
        self.assertTrue(fallback_allowed("cuda_probe", None))

    def test_classifies_auth_and_disk_pull_failures(self):
        self.assertEqual(classify_pull_failure("unauthorized: access denied"), "auth")
        self.assertEqual(classify_pull_failure("no space left on device"), "disk")

    def test_image_pull_retries_three_times_with_exponential_backoff(self):
        calls = []
        delays = []

        def runner(argv, timeout=None):
            calls.append(tuple(argv))
            if len(calls) < 3:
                return CommandResult(tuple(argv), 1, "", "TLS timeout", 0.01)
            return CommandResult(tuple(argv), 0, "pulled", "", 0.01)

        results = pull_with_retry(
            runner,
            ("docker", "pull", "example:v1"),
            timeout=10,
            sleep=delays.append,
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual([result.returncode for result in results], [1, 1, 0])
        self.assertEqual(delays, [1.0, 2.0])

    def test_reads_immutable_hugging_face_revision(self):
        self.assertEqual(parse_hf_revision({"sha": "a" * 40}), "a" * 40)
        with self.assertRaisesRegex(ValueError, "40-character"):
            parse_hf_revision({"sha": "main"})

    def test_finds_missing_vllm_help_flags(self):
        help_text = " ".join(flag for flag in REQUIRED_FLAGS if flag != "--calculate-kv-scales")
        self.assertEqual(missing_required_flags(help_text), ("--calculate-kv-scales",))

    def test_parses_cuda_probe_last_json_line(self):
        payload = parse_cuda_probe(
            "warning line\n"
            '{"cuda_available": true, "device_count": 1, '
            '"compute_capability": [8, 6], "gpu_name": "NVIDIA RTX A6000", '
            '"torch": "2.9", "cuda": "13.0", "vllm": "0.26.0", '
            '"flashinfer": "0.5"}\n'
        )
        self.assertTrue(payload["cuda_available"])
        self.assertEqual(payload["compute_capability"], [8, 6])

    def test_parses_image_digest_from_inspect(self):
        digest = "d" * 64
        image = parse_image_inspect(
            '{"Id":"sha256:id","RepoDigests":'
            f'["vllm/vllm-openai@sha256:{digest}"]}}'
        )
        self.assertEqual(image.image_id, "sha256:id")
        self.assertEqual(
            image.image_ref, f"vllm/vllm-openai@sha256:{digest}"
        )

    def test_rejects_non_sha256_image_digest(self):
        with self.assertRaisesRegex(ValueError, "immutable sha256"):
            parse_image_inspect(
                '{"Id":"sha256:id","RepoDigests":'
                '["vllm/vllm-openai@sha256:short"]}'
            )

    def test_lock_falls_back_only_after_primary_cuda_probe_failure(self):
        calls = []

        def result(argv, returncode=0, stdout="", stderr=""):
            return CommandResult(tuple(argv), returncode, stdout, stderr, 0.01)

        def runner(argv, timeout=None):
            calls.append(tuple(argv))
            joined = " ".join(argv)
            if argv[0] == "nvidia-smi":
                return result(
                    argv, stdout="NVIDIA RTX A6000, 580.173.02, 49140, 8.6\n"
                )
            if tuple(argv[:2]) == ("docker", "version"):
                return result(argv, stdout="27.0.0\n")
            if tuple(argv[:2]) == ("docker", "pull"):
                return result(argv, stdout="pulled\n")
            if "--entrypoint python" in joined:
                if "v0.26.0" in joined:
                    return result(argv, returncode=1, stderr="unsupported PTX")
                return result(
                    argv,
                    stdout=(
                        '{"cuda_available": true, "device_count": 1, '
                        '"compute_capability": [8, 6], "gpu_name": '
                        '"NVIDIA RTX A6000", "torch": "2.8", '
                        '"cuda": "12.9", "vllm": "0.19.1", '
                        '"flashinfer": "0.4"}\n'
                    ),
                )
            if "serve --help" in joined:
                return result(argv, stdout=" ".join(REQUIRED_FLAGS))
            if tuple(argv[:3]) == ("docker", "image", "inspect"):
                digest = "f" * 64
                return result(
                    argv,
                    stdout=(
                        '{"Id":"sha256:fallback","RepoDigests":'
                        f'["vllm/vllm-openai@sha256:{digest}"]}}'
                    ),
                )
            raise AssertionError(f"unexpected command: {joined}")

        with tempfile.TemporaryDirectory() as directory:
            profile = Profile.from_dict(Profile.default_dict("quick"))
            lock = lock_first_compatible_image(
                profile,
                Path(directory),
                runner=runner,
                hf_metadata_loader=lambda model_id: {"sha": "b" * 40},
            )
            self.assertEqual(lock["image_tag"], "vllm/vllm-openai:v0.19.1")
            self.assertEqual(lock["model_revision"], "b" * 40)
            self.assertTrue(
                (Path(directory) / "runs" / "_environment" / "lock.json").exists()
            )
            self.assertTrue(any("v0.26.0" in " ".join(call) for call in calls))
            self.assertTrue(any("v0.19.1" in " ".join(call) for call in calls))


if __name__ == "__main__":
    unittest.main()
