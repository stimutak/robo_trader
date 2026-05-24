#!/usr/bin/env python3
"""Preflight safety gate — runs before the trading runner starts.

Usage::

    python3 scripts/preflight_check.py            # standard run, plaintext
    python3 scripts/preflight_check.py --json     # machine-readable
    python3 scripts/preflight_check.py --verbose  # add details block
    python3 scripts/preflight_check.py --force "<reason>"  # bypass BLOCKs

Exit codes (spec §5.2):

    0  All checks PASS or WARN — safe to proceed
    1  At least one check BLOCKED — abort startup
    2  Operator passed --force on a BLOCK — proceeded with audit log
    3  Preflight itself failed (uncaught exception, broken environment)

START_TRADER.sh wraps this in step 4.5. The bash side adds the
meta-instruction ("re-run after fixing") so this script focuses on
diagnosis.

Coordination with later layers
------------------------------
On a clean run (exit code 0), writes ``data/.preflight_last_ok`` with
the current ISO timestamp. This is the "preflight is happy" signal that
the watchdog content-liveness work (MVP-2) and ``runner_async``'s
optional server-side enforcement (Q11.6) will both read. Operator-side
this is invisible.

Spec: docs/superpowers/specs/2026-05-23-startup-safety-gate-design.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

# When invoked via START_TRADER.sh, the cwd is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure the repo's source is importable when the script is run from any cwd.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.preflight import PreflightContext  # noqa: E402
from robo_trader.preflight.checks import ALL_CHECKS  # noqa: E402
from robo_trader.preflight.result import CheckStatus  # noqa: E402
from robo_trader.preflight.runner import (  # noqa: E402
    RunReport,
    format_json,
    format_plaintext,
    run_all_checks,
)

# Reason denylist (spec §6.3 step 4): reject obviously-low-effort
# justifications. Forces a real-sentence rationale at the moment of
# decision so 24h-later "what did I bypass and why" can be answered.
_REASON_MIN_LENGTH = 10
_REASON_DENYLIST = frozenset(
    {
        "force",
        "bypass",
        "whatever",
        "idk",
        "test",
        "skip",
        "ignore",
        "go",
        "yes",
        "ok",
    }
)

_BYPASS_LOG_PATH = "data/preflight_bypass.log"
_LAST_OK_FLAG_PATH = "data/.preflight_last_ok"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight_check",
        description="Pre-startup safety gate for the robo_trader runner.",
        epilog=(
            "On BLOCKs, fix the underlying condition and re-run, or use "
            "--force with a real-sentence reason to override (audited)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of plaintext.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Append a per-check details block (plaintext only).",
    )
    parser.add_argument(
        "--force",
        metavar="REASON",
        help=(
            "Bypass any BLOCKs. Requires a real-sentence reason (>=10 chars, "
            "not a placeholder). Exit code becomes 2 (not 0) and the bypass "
            f"is appended to {_BYPASS_LOG_PATH}."
        ),
    )
    return parser


def _resolve_target_port() -> int:
    """4001 for live, 4002 for paper (default)."""
    mode = os.environ.get("EXECUTION_MODE", "paper").strip().lower()
    return 4001 if mode == "live" else 4002


def _build_context(project_root: Path) -> PreflightContext:
    """Build the shared PreflightContext from the resolved env.

    Env is snapshotted into a plain dict here so the checks see the same
    state regardless of subprocess timing. We do NOT call ``load_dotenv``
    in this script — START_TRADER.sh has already exported `.env` into
    the shell environment by the time this runs.
    """
    return PreflightContext(
        project_root=project_root,
        env=dict(os.environ),
        target_port=_resolve_target_port(),
        dry_run=False,
    )


def _validate_force_reason(reason: str) -> Optional[str]:
    """Return None if the reason is acceptable, else a complaint string."""
    stripped = reason.strip()
    if len(stripped) < _REASON_MIN_LENGTH:
        return (
            f"--force reason must be at least {_REASON_MIN_LENGTH} characters "
            f"(got {len(stripped)})"
        )
    low = stripped.lower()
    if low in _REASON_DENYLIST:
        return f"--force reason cannot be a placeholder like {low!r} — explain what and why"
    return None


def _write_bypass_log_entry(
    project_root: Path,
    reason: str,
    blocked_check_names: List[str],
) -> None:
    """Append one newline-delimited JSON entry to data/preflight_bypass.log."""
    path = project_root / _BYPASS_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": _dt.datetime.now().astimezone().isoformat(),
        "reason": reason,
        "bypassed_checks": blocked_check_names,
        "operator": os.environ.get("USER", "unknown"),
    }
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")


def _write_last_ok_flag(project_root: Path) -> None:
    """Write ``data/.preflight_last_ok`` with the current ISO timestamp.

    Q11.4 decision: a tiny signal future tooling can poll. Cost is one
    file write; benefit is the runner can refuse to start if preflight
    hasn't passed recently (Q11.6 server-side enforcement).
    """
    path = project_root / _LAST_OK_FLAG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dt.datetime.now().astimezone().isoformat() + "\n", encoding="utf-8")


def _log_bypass_event(reason: str, blocked_names: List[str]) -> None:
    """Emit a WARNING-level structured log line for the bypass.

    We use Python's stdlib logger here (no robo_trader logger setup) so
    this script stays standalone — if it's invoked outside START_TRADER.sh
    (e.g. by an operator probing state), it still works. The line lands
    on stderr by default.
    """
    logger = logging.getLogger("preflight")
    if not logger.handlers:
        # Standalone setup — single stderr handler.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.warning(
        "event=preflight_bypass reason=%r bypassed=%s operator=%s",
        reason,
        ",".join(blocked_names) or "<none>",
        os.environ.get("USER", "unknown"),
    )


def _decide_exit_code(report: RunReport, force_reason: Optional[str]) -> int:
    """Translate the run outcome into the §5.2 exit code."""
    blocks = report.blocked
    if force_reason is not None:
        if not blocks:
            # --force was passed but nothing actually blocked. Don't silently
            # exit 2 — the operator may have copy-pasted a stale invocation.
            # Exit 0 with a warning printed by main().
            return 0
        return 2
    return 1 if blocks else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # Validate --force early so we don't run checks just to reject the bypass.
    force_reason: Optional[str] = None
    if args.force is not None:
        complaint = _validate_force_reason(args.force)
        if complaint:
            print(f"error: {complaint}", file=sys.stderr)
            return 1
        force_reason = args.force.strip()

    project_root = PROJECT_ROOT

    try:
        context = _build_context(project_root)
        report = run_all_checks(context, checks=ALL_CHECKS)
    except BaseException:
        # Code 3 territory: preflight itself failed before/around the
        # check loop. Per spec §5.2 this should be impossible in normal
        # operation since per-check exceptions are caught by the runner.
        print("error: preflight itself failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 3

    # Print results
    if args.json:
        sys.stdout.write(format_json(report))
    else:
        sys.stdout.write(format_plaintext(report, verbose=args.verbose))

    blocked_names = [r.name for r in report.blocked]

    # --force bookkeeping
    if force_reason is not None and blocked_names:
        try:
            _write_bypass_log_entry(project_root, force_reason, blocked_names)
        except OSError as exc:
            # Audit log write failed. We refuse to bypass without an
            # audit trail — that's the whole point of the mechanism.
            print(
                f"\nerror: could not write bypass audit log ({exc}); refusing to bypass",
                file=sys.stderr,
            )
            return 1
        _log_bypass_event(force_reason, blocked_names)
        print(
            f"\n⚠ PREFLIGHT BYPASS — operator override: {force_reason!r}",
            file=sys.stderr,
        )
        print(
            f"   Bypassed checks: {', '.join(blocked_names)}",
            file=sys.stderr,
        )
        print(
            f"   Audit log entry appended to {_BYPASS_LOG_PATH}",
            file=sys.stderr,
        )
    elif force_reason is not None and not blocked_names:
        # --force given but nothing blocked. Per §6.3 step 3, exit 0 with
        # a warning so operators don't leave --force in shell history as
        # a copy-paste default.
        print(
            "\nwarning: --force given but no checks BLOCKED. Force was not needed.",
            file=sys.stderr,
        )

    exit_code = _decide_exit_code(report, force_reason)

    # Q11.4: write the last-ok flag on any clean exit (0 or 2 — bypass
    # still counts as "preflight has run and the operator made a
    # decision"). The runner's Q11.6 check reads this file's mtime; the
    # presence of the flag (regardless of how it was earned) means
    # preflight has been run recently.
    if exit_code in (0, 2):
        try:
            _write_last_ok_flag(project_root)
        except OSError:
            # Non-fatal — the gate did its job, the flag is just a hint
            # for downstream tooling.
            pass

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
