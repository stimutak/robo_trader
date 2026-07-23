#!/usr/bin/env python3
"""Decide whether watchdog may restart after a runner exit.

Exit codes:
  0  Restart is allowed (no audit or a non-terminal exit).
  20 Restart is blocked by an explicit terminal safety exit.
  21 Restart is blocked because an existing audit cannot be trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

RESTART_ALLOWED: Final = 0
TERMINAL_SAFETY_BLOCK: Final = 20
POLICY_EVIDENCE_INVALID: Final = 21
MAX_AUDIT_BYTES: Final = 64 * 1024
TERMINAL_EXITS: Final = {
    ("unprotected_existing_positions", 6),
}


def evaluate_restart_policy(audit_path: Path) -> tuple[int, str]:
    """Return the supervisor decision and a non-sensitive reason code."""

    if not audit_path.exists():
        return RESTART_ALLOWED, "no_exit_audit"
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
    if (reason, exit_code) in TERMINAL_EXITS:
        return TERMINAL_SAFETY_BLOCK, reason
    return RESTART_ALLOWED, "nonterminal_exit"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: watchdog_restart_policy.py RUNNER_EXIT_JSON", file=sys.stderr)
        return POLICY_EVIDENCE_INVALID
    decision, reason = evaluate_restart_policy(Path(args[0]))
    print(reason)
    return decision


if __name__ == "__main__":
    raise SystemExit(main())
