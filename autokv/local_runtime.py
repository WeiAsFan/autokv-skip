"""User-owned local vLLM process lifecycle management."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


class LocalVllmProcess:
    """A process handle that can only terminate the process group it created."""

    def __init__(self, process: subprocess.Popen[str], log_path: Path) -> None:
        self.process = process
        self.log_path = log_path

    @classmethod
    def start(
        cls,
        argv: Sequence[str],
        log_path: Path,
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> "LocalVllmProcess":
        if not argv:
            raise ValueError("local vLLM command cannot be empty")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8", newline="\n")
        try:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(cwd),
                env=dict(env),
                start_new_session=True,
            )
        except BaseException:
            stream.close()
            raise
        stream.close()
        return cls(process, log_path)

    @property
    def pid(self) -> int:
        return self.process.pid

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def stop(self, timeout: float = 30.0) -> None:
        process = self.process
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=timeout)
        else:
            process.wait()
