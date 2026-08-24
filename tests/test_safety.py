import unittest
from pathlib import Path

from scripts.verify import verification_commands


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_RUNTIME_PATTERNS = (
    "apt install nvidia-driver",
    "dnf install nvidia-driver",
    "docker system prune",
    "--privileged",
    "/usr/local/cuda:/usr/local/cuda",
)


class SafetyTests(unittest.TestCase):
    def test_runtime_python_contains_no_forbidden_mutation(self):
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "autokv").glob("*.py")
        ).lower()
        for pattern in FORBIDDEN_RUNTIME_PATTERNS:
            self.assertNotIn(pattern, text)

    def test_release_verifier_runs_tests_compile_and_both_dry_runs(self):
        commands = verification_commands()
        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("unittest discover" in command for command in flattened))
        self.assertTrue(any("compileall" in command for command in flattened))
        self.assertTrue(
            any("dry-run --profile quick" in command for command in flattened)
        )
        self.assertTrue(any("dry-run --profile full" in command for command in flattened))


if __name__ == "__main__":
    unittest.main()
