"""Kill-switch persisted state check (spec §7.1, G1).

This is the check that would have prevented the 2026-05-22 4-hour silent
livelock. The ``KillSwitch`` class persists ``triggered=True`` to
``data/kill_switch_state.json`` once a per-position loss limit fires;
every subsequent runner load reads that state and refuses to trade. The
runtime side of that is correct behavior (we don't want auto-reset on
loss-based trips). What was missing was operator visibility: the
watchdog kept restarting, the runner kept dying within seconds, and
nothing told the human "you have a persisted kill switch — clear it or
raise the threshold."

This check surfaces that condition before the runner even calls
``connect()``: file missing or ``triggered=False`` → PASS; everything
else (triggered, corrupt JSON, unreadable) → BLOCK with remediation that
names the file, the trigger reason, and the exact commands to back it up
and clear it.

The check is **read-only by design** (spec §5.7, N1). It never modifies
the state file. The H1 in-flight auto-reset heuristic in
``runner_async.py`` is deliberately NOT mirrored here — see spec §8 for
the full justification, but in short: H1 has direct evidence that
``recover_connection`` succeeded; preflight runs before any connection
attempt and has no such evidence. The bypass is ``--force "<reason>"``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import PreflightContext
from .result import CheckResult, CheckStatus


class KillSwitchStateCheck:
    """BLOCK if ``data/kill_switch_state.json`` shows ``triggered=True``.

    Fail-closed on parse / read errors — matches the KillSwitch class's
    own behavior in ``_load_persisted_state``. A corrupt state file is
    treated as "we cannot prove the switch is safe to ignore," so we
    refuse to proceed.
    """

    name = "kill_switch_state"
    description = "Kill switch persisted state"
    timeout_seconds = 1.0

    def run(self, context: PreflightContext) -> CheckResult:
        path = context.project_root / "data" / "kill_switch_state.json"

        if not path.exists():
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message="no triggered state",
                details={"state_path": str(path), "exists": False},
            )

        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"state file unreadable ({exc.__class__.__name__})",
                remediation=(
                    "Kill switch state file is corrupt. Fail-closed policy "
                    "blocks startup.\n"
                    f"  Inspect:   cat {path}\n"
                    f"  Backup:    cp {path} {path}.bak.$(date +%Y-%m-%d-%H%M)\n"
                    f"  Clear:     rm {path}\n"
                    "  Then re-run ./START_TRADER.sh"
                ),
                details={"state_path": str(path), "error": str(exc)},
            )

        if not payload.get("triggered"):
            details = {
                "state_path": str(path),
                "exists": True,
                "triggered": False,
            }
            previous = payload.get("previous_trigger_reason")
            if previous is not None:
                details["previous_trigger_reason"] = previous
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message="no triggered state",
                details=details,
            )

        reason = payload.get("trigger_reason", "unknown")
        trigger_time = payload.get("trigger_time", "unknown")
        return CheckResult(
            name=self.name,
            status=CheckStatus.BLOCK,
            message=f"triggered=True since {trigger_time}",
            remediation=self._build_remediation(path, reason, trigger_time),
            details={
                "state_path": str(path),
                "trigger_reason": reason,
                "trigger_time": trigger_time,
            },
        )

    def _build_remediation(self, path: Path, reason: str, trigger_time: str) -> str:
        # Escape embedded double-quotes so the remediation text (which
        # wraps the reason in quotes for display) doesn't render with a
        # broken quote and confuse the operator at 3am.
        safe_reason = str(reason).replace('"', '\\"')
        return (
            f"Kill switch is triggered.\n\n"
            f'Trigger reason: "{safe_reason}"\n'
            f"Triggered at:   {trigger_time}\n"
            f"State file:     {path}\n\n"
            "What to do:\n"
            "  1. Decide if the loss-trigger is still relevant.\n"
            "     - If a real loss event: review positions before clearing.\n"
            "     - If a stale trip from a transient issue: clear it.\n"
            "  2. To clear (DESTRUCTIVE — review first):\n"
            f"       cp {path} {path}.bak.$(date +%Y-%m-%d-%H%M)\n"
            f"       rm {path} data/kill_switch.lock\n"
            "  3. Re-run: ./START_TRADER.sh\n\n"
            "To proceed anyway (NOT recommended without step 1):\n"
            '  python3 scripts/preflight_check.py --force "<your reason here>"\n'
            "  then re-run ./START_TRADER.sh"
        )
