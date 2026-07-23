#!/usr/bin/env python3
"""Decide whether watchdog may restart after a runner exit.

Exit codes:
  0  Restart is allowed after an explicit non-terminal exit.
  20 Restart is blocked by an explicit terminal safety exit.
  21 Restart is blocked because exit evidence is missing or cannot be trusted.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Final

RESTART_ALLOWED: Final = 0
TERMINAL_SAFETY_BLOCK: Final = 20
POLICY_EVIDENCE_INVALID: Final = 21
MAX_AUDIT_BYTES: Final = 64 * 1024
MAX_NONTERMINAL_AGE_SECONDS: Final = 5 * 60
MAX_FUTURE_SKEW_SECONDS: Final = 5
TERMINAL_EXITS: Final = {
    ("unprotected_existing_positions", 6),
}
RESTARTABLE_EXITS: Final = {
    ("clean_shutdown", 0),
    ("keyboard_interrupt", 0),
    ("pre_flight_gateway_unreachable", 1),
    ("sigint", 0),
    ("sigterm", 0),
    ("unhandled_exception", 2),
}


def evaluate_restart_policy(
    audit_path: Path,
    expected_pid: int | None,
    *,
    now: float | None = None,
) -> tuple[int, str]:
    """Return the supervisor decision and a non-sensitive reason code.

    ``expected_pid`` is the exact runner PID previously observed alive by this
    watchdog process. It binds a restartable audit to the process that just
    disappeared. PID reuse is additionally bounded by the short freshness
    window below. The policy is deliberately read-only.
    """

    if not audit_path.exists():
        # Absence is ambiguous, not permission. _write_exit_audit() must remain
        # best-effort on fatal paths, so an unwritable/full filesystem or a
        # hard process death can produce no file. Auto-restarting in that state
        # would erase the supervisor boundary for terminal safety exits.
        return POLICY_EVIDENCE_INVALID, "exit_audit_missing"
    try:
        if audit_path.is_symlink():
            return POLICY_EVIDENCE_INVALID, "exit_audit_symlink"
        stat = audit_path.stat()
        if not audit_path.is_file() or stat.st_size > MAX_AUDIT_BYTES:
            return POLICY_EVIDENCE_INVALID, "exit_audit_invalid_file"
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        return POLICY_EVIDENCE_INVALID, "exit_audit_unreadable"

    if not isinstance(payload, dict):
        return POLICY_EVIDENCE_INVALID, "exit_audit_invalid_schema"
    reason = payload.get("reason")
    exit_code = payload.get("exit_code")
    if not isinstance(reason, str) or isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return POLICY_EVIDENCE_INVALID, "exit_audit_invalid_schema"

    # An exact terminal pair is sticky. It must remain blocked indefinitely
    # even after its timestamp becomes old or the watchdog is restarted.
    if (reason, exit_code) in TERMINAL_EXITS:
        return TERMINAL_SAFETY_BLOCK, reason

    # A terminal reason with a different code, or any arbitrary new pair, must
    # never be silently treated as restartable.
    if (reason, exit_code) not in RESTARTABLE_EXITS:
        return POLICY_EVIDENCE_INVALID, "exit_audit_unknown_pair"

    timestamp = payload.get("timestamp")
    audit_pid = payload.get("pid")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
        or isinstance(audit_pid, bool)
        or not isinstance(audit_pid, int)
        or audit_pid <= 0
    ):
        return POLICY_EVIDENCE_INVALID, "exit_audit_invalid_schema"
    if expected_pid is None:
        return POLICY_EVIDENCE_INVALID, "runner_pid_unobserved"
    if audit_pid != expected_pid:
        return POLICY_EVIDENCE_INVALID, "runner_pid_mismatch"

    current_time = time.time() if now is None else now
    age = float(current_time) - float(timestamp)
    if age < -MAX_FUTURE_SKEW_SECONDS:
        return POLICY_EVIDENCE_INVALID, "exit_audit_from_future"
    if age > MAX_NONTERMINAL_AGE_SECONDS:
        return POLICY_EVIDENCE_INVALID, "exit_audit_stale"

    return RESTART_ALLOWED, "nonterminal_exit"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(
            "usage: watchdog_restart_policy.py RUNNER_EXIT_JSON EXPECTED_RUNNER_PID",
            file=sys.stderr,
        )
        return POLICY_EVIDENCE_INVALID
    try:
        expected_pid = None if args[1] == "unavailable" else int(args[1])
        if expected_pid is not None and expected_pid <= 0:
            raise ValueError
    except ValueError:
        print("runner_pid_invalid")
        return POLICY_EVIDENCE_INVALID
    decision, reason = evaluate_restart_policy(Path(args[0]), expected_pid)
    print(reason)
    return decision


if __name__ == "__main__":
    raise SystemExit(main())
