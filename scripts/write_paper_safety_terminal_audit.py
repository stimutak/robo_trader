#!/usr/bin/env python3
"""Durably record the fixed paper-safety startup terminal exit.

This helper is intentionally sealed: it accepts no reason, exception, account,
or path arguments from the launcher.  Its only possible record is the exact
terminal pair understood by ``watchdog_restart_policy.py``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TERMINAL_REASON = "paper_safety_journal_replay_blocked"
TERMINAL_EXIT_CODE = 7


def write_terminal_audit(audit_path: Path | None = None) -> None:
    """Atomically replace and fsync the terminal runner-exit audit."""

    final_path = audit_path or (PROJECT_ROOT / "data" / "runner_exit.json")
    final_path = Path(final_path)
    final_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "iso_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reason": TERMINAL_REASON,
        "exit_code": TERMINAL_EXIT_CODE,
        "pid": os.getpid(),
        "source": "supervised_launcher",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(temporary_path, flags, 0o600)
        with os.fdopen(file_descriptor, "wb", closefd=True) as stream:
            file_descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, final_path)
        directory_descriptor = os.open(
            final_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    try:
        write_terminal_audit()
    except Exception as exc:
        print(
            f"ERROR: terminal audit write failed ({type(exc).__name__})",
            file=os.sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
