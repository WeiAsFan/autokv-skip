import json
import tempfile
import unittest
from pathlib import Path

from autokv.commands import CommandResult, local_server_command
from autokv.config import Profile
from autokv.v2_config import load_v2_config
from autokv.v2_policy import endpoint_policies
from autokv.v2_runtime import (
    V2PolicyRunner,
    policy_manifest_is_valid,
    validate_first_output,
    validate_prefix_caching_disabled,
)


ROOT = Path(__file__).resolve().parents[1]


class V2RuntimeTests(unittest.TestCase):
    def test_server_command_and_log_both_prove_prefix_cache_is_off(self):
        profile = Profile.from_dict(Profile.default_dict("full"))
        _, p0 = endpoint_policies()
        argv = local_server_command(
            profile,
            "/workspace/.venv/bin/vllm",
            p0.variant,
            __import__("pathlib").Path("/workspace"),
            8000,
            "a" * 40,
            model_path=__import__("pathlib").Path("/model"),
        )
        validate_prefix_caching_disabled(
            "EngineArgs(enable_prefix_caching=False)", argv
        )
        self.assertEqual(argv.count("--no-enable-prefix-caching"), 1)

    def test_first_output_rejects_replacement_and_repetition(self):
        with self.assertRaises(ValueError):
            validate_first_output("bad \ufffd output")
        with self.assertRaises(ValueError):
            validate_first_output("abcdefgh" * 8)

    def test_one_policy_writes_minimal_manifest_and_resumes_without_restart(self):
        config = load_v2_config(ROOT / "configs/v2/quality.json")
        profile = Profile.from_dict(Profile.default_dict("full"))
        lock = {
            "backend": "docker",
            "image_ref": "vllm/vllm-openai@sha256:" + "f" * 64,
            "image_digest": "sha256:" + "f" * 64,
            "model_revision": config.model_revision,
        }
        _, policy = endpoint_policies()
        sample = {
            "sample_id": "easy-1",
            "split": "calibration",
            "tier": "easy",
            "task": "niah",
            "prompt": "Return CODE-1.",
            "prompt_tokens": 7,
            "max_tokens": 8,
            "expected_answers": ["CODE-1"],
            "answer_mode": "contains",
        }
        containers = {}

        def result(argv, returncode=0, stdout="", stderr=""):
            return CommandResult(tuple(argv), returncode, stdout, stderr, 0.01)

        def command_runner(argv, timeout=None, **kwargs):
            if tuple(argv[:3]) == ("docker", "run", "-d"):
                containers[argv[argv.index("--name") + 1]] = True
                return result(argv)
            if tuple(argv[:2]) == ("docker", "logs"):
                return result(
                    argv,
                    stdout=(
                        "Using AttentionBackendEnum.FLASHINFER backend.\n"
                        "Using fp8_e4m3 data type to store kv cache.\n"
                        "GPU KV cache size: 262,144 tokens\n"
                        "EngineArgs(enable_prefix_caching=False)\n"
                    ),
                )
            if tuple(argv[:2]) == ("docker", "inspect"):
                if "{{json .Config.Cmd}}" in argv:
                    return result(
                        argv,
                        stdout=json.dumps(
                            [
                                "--attention-backend",
                                "FLASHINFER",
                                "--kv-cache-dtype",
                                "fp8_e4m3",
                                "--no-enable-prefix-caching",
                            ]
                        ),
                    )
                name = argv[-1]
                if name not in containers:
                    return result(argv, returncode=1, stderr="No such object")
                return result(
                    argv,
                    stdout=f"autokv-skip|{'true' if containers[name] else 'false'}\n",
                )
            if tuple(argv[:2]) == ("docker", "stop"):
                containers[argv[-1]] = False
                return result(argv)
            if tuple(argv[:2]) == ("docker", "rm"):
                containers.pop(argv[-1], None)
                return result(argv)
            raise AssertionError(f"unexpected command: {argv}")

        class Client:
            def health(self):
                return True

            def chat_complete(self, prompt, max_tokens):
                return {
                    "choices": [{"message": {"content": "CODE-1"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = V2PolicyRunner(
                config,
                profile,
                root,
                lock,
                "run-v2",
                port=8000,
                command_runner=command_runner,
                client_factory=lambda base_url, model_id: Client(),
            )
            path = runner.run_policy(
                policy,
                (sample,),
                split="calibration",
                split_sha256="a" * 64,
                relative_directory=Path("quality/calibration/endpoints"),
            )
            manifest = path.with_name(path.stem + ".policy-manifest.json")
            log = path.with_name(path.stem + ".server.log")

            self.assertTrue(
                policy_manifest_is_valid(
                    manifest,
                    path,
                    log,
                    policy,
                    (sample,),
                    split="calibration",
                    split_sha256="a" * 64,
                    run_id="run-v2",
                )
            )
            self.assertEqual(runner.server_starts, 1)
            self.assertEqual(
                runner.run_policy(
                    policy,
                    (sample,),
                    split="calibration",
                    split_sha256="a" * 64,
                    relative_directory=Path("quality/calibration/endpoints"),
                ),
                path,
            )
            self.assertEqual(runner.server_starts, 1)


if __name__ == "__main__":
    unittest.main()
