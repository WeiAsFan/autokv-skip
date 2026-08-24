"""Cross-platform release verification for AutoKV-Skip."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_RUNTIME_PATTERNS = (
    "apt install nvidia-driver",
    "dnf install nvidia-driver",
    "docker system prune",
    "--privileged",
    "/usr/local/cuda:/usr/local/cuda",
)


def verification_commands() -> tuple[tuple[str, ...], ...]:
    """Return the exact commands used by the release gate."""

    return (
        (
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
        ),
        (sys.executable, "-m", "compileall", "-q", "autokv", "scripts"),
        (
            sys.executable,
            "-m",
            "autokv",
            "dry-run",
            "--profile",
            "quick",
            "--json",
        ),
        (
            sys.executable,
            "-m",
            "autokv",
            "dry-run",
            "--profile",
            "full",
            "--json",
        ),
    )


def _format_argv(argv: Sequence[str]) -> str:
    return shlex.join(argv)


def _parse_dry_run(stdout: str, profile: str, expected_core: int) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{profile} dry-run did not emit one JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{profile} dry-run JSON must be an object")
    if payload.get("profile") != profile:
        raise ValueError(f"{profile} dry-run reported the wrong profile")
    if payload.get("executed") is not False:
        raise ValueError(f"{profile} dry-run unexpectedly executed work")
    if payload.get("core_probe_configurations") != expected_core:
        raise ValueError(
            f"{profile} dry-run expected {expected_core} core configurations, "
            f"got {payload.get('core_probe_configurations')!r}"
        )
    return payload


def _assert_generated_commands_are_safe(payloads: Sequence[dict[str, object]]) -> None:
    generated = json.dumps(payloads, ensure_ascii=False).lower()
    for pattern in FORBIDDEN_RUNTIME_PATTERNS:
        if pattern in generated:
            raise ValueError(
                f"generated dry-run contains forbidden runtime pattern: {pattern}"
            )


def main() -> int:
    dry_run_outputs: list[str] = []
    commands = verification_commands()
    for index, argv in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {_format_argv(argv)}", flush=True)
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            shell=False,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(
                completed.stderr,
                end="" if completed.stderr.endswith("\n") else "\n",
                file=sys.stderr,
            )
        if completed.returncode != 0:
            print(
                f"VERIFICATION_FAILED command={index} exit={completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
        if "dry-run" in argv:
            dry_run_outputs.append(completed.stdout)

    try:
        quick = _parse_dry_run(dry_run_outputs[0], "quick", 18)
        full = _parse_dry_run(dry_run_outputs[1], "full", 34)
        _assert_generated_commands_are_safe((quick, full))
    except (IndexError, ValueError) as exc:
        print(f"VERIFICATION_FAILED validation: {exc}", file=sys.stderr)
        return 1

    print("VERIFICATION_OK tests compileall quick=18 full=34 safety=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
