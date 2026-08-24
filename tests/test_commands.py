import sys
import unittest
from pathlib import Path

from autokv.commands import (
    bench_command,
    container_name,
    format_command,
    image_probe_commands,
    run_command,
    server_command,
)
from autokv.config import Profile
from autokv.selection import Variant


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.profile = Profile.from_dict(Profile.default_dict("quick"))
        self.root = Path("/srv/autokv-skip")

    def test_mixed_server_command_contains_compatibility_and_skip_layers(self):
        argv = server_command(
            self.profile,
            "sha256:locked",
            Variant.mixed("auto-4", (2, 7, 18, 29)),
            self.root,
            8000,
            "run1",
            model_revision="a" * 40,
        )
        joined = " ".join(argv)
        self.assertIn("VLLM_ENABLE_CUDA_COMPATIBILITY=1", joined)
        self.assertIn("--attention-backend FLASHINFER", joined)
        self.assertIn("--kv-cache-dtype fp8_e4m3", joined)
        self.assertIn("--kv-cache-dtype-skip-layers 2 7 18 29", joined)
        self.assertIn(f"--revision {'a' * 40}", joined)
        self.assertNotIn("--privileged", argv)
        self.assertNotIn("--rm", argv)
        self.assertNotIn("/usr/local/cuda", joined)
        self.assertNotIn("nvidia-driver", joined.lower())

    def test_bf16_command_does_not_calculate_fp8_scales(self):
        argv = server_command(
            self.profile,
            "sha256:locked",
            Variant.bf16(),
            self.root,
            8000,
            "run1",
        )
        self.assertNotIn("--calculate-kv-scales", argv)
        self.assertEqual(argv[argv.index("--kv-cache-dtype") + 1], "bfloat16")

    def test_probe_commands_use_compatibility_for_cuda_probe(self):
        commands = image_probe_commands("vllm/vllm-openai:v0.26.0")
        self.assertEqual(commands[0], ("docker", "pull", "vllm/vllm-openai:v0.26.0"))
        self.assertIn("VLLM_ENABLE_CUDA_COMPATIBILITY=1", " ".join(commands[1]))
        self.assertIn("serve --help", " ".join(commands[2]))

    def test_benchmark_uses_same_image_and_host_network(self):
        argv = bench_command(
            self.profile,
            "sha256:locked",
            self.root,
            port=8000,
            input_length=8192,
            output_length=32,
            result_relative_path=Path("runs/run1/perf/bf16-8k.json"),
        )
        joined = " ".join(argv)
        self.assertIn("--network host", joined)
        self.assertIn("sha256:locked bench serve", joined)
        self.assertIn("--random-input-len 8192", joined)
        self.assertIn("--random-output-len 32", joined)

    def test_container_name_is_deterministic_and_docker_safe(self):
        name = container_name("Run With Spaces", Variant.fp8())
        self.assertRegex(name, r"^autokv-[a-z0-9_.-]+$")
        self.assertEqual(name, container_name("Run With Spaces", Variant.fp8()))

    def test_run_command_does_not_interpret_shell_metacharacters(self):
        marker = "$(echo should-not-run)"
        result = run_command(
            [sys.executable, "-c", "import sys; print(sys.argv[1])", marker]
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), marker)

    def test_format_command_redacts_token(self):
        rendered = format_command(["tool", "--token", "secret"], secrets=["secret"])
        self.assertNotIn("secret", rendered)
        self.assertIn("***", rendered)


if __name__ == "__main__":
    unittest.main()
