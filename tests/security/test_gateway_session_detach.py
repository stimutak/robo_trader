"""Behavioral tests for terminal-independent supervised child launch."""

from __future__ import annotations

import os
import pty
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DETACH_HELPER = ROOT / "scripts" / "launch_detached.py"


def test_detached_helper_survives_controlling_terminal_close(tmp_path):
    """Closing the launcher's PTY must not kill the detached child."""

    marker = tmp_path / "child-finished"
    launch_log = tmp_path / "launch.log"
    child_command = (
        "import os, pathlib, time; "
        "time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text(f'{{os.getsid(0)}}:{{os.getpid()}}')"
    )
    launch_command = " ".join(
        shlex.quote(part)
        for part in (
            "/usr/bin/nohup",
            sys.executable,
            str(DETACH_HELPER),
            str(tmp_path),
            sys.executable,
            "-c",
            child_command,
        )
    )
    launch_command += f" </dev/null >{shlex.quote(str(launch_log))} 2>&1 & sleep 0.2"

    shell_pid, terminal_fd = pty.fork()
    if shell_pid == 0:
        os.execv("/bin/bash", ["/bin/bash", "-c", launch_command])

    _, wait_status = os.waitpid(shell_pid, 0)
    os.close(terminal_fd)
    assert os.waitstatus_to_exitcode(wait_status) == 0

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)

    assert marker.exists(), launch_log.read_text() if launch_log.exists() else "no launch log"
    session_id, process_id = (int(value) for value in marker.read_text().split(":"))
    assert session_id == process_id
    assert session_id != shell_pid
