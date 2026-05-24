"""Block startup when ``data/kill_switch.lock`` is present (spec §7.2).

Why this check exists
---------------------
``data/kill_switch.lock`` is the deny-by-default fail-closed signal called
out in the plan. :class:`robo_trader.risk.KillSwitch` already treats the
lock's presence as triggered (``R2-M2``), but the lock can exist
independently of the JSON state file — for example, an operator may have
created it manually to force a stop. Surfacing it during preflight gives
the operator an actionable message ("remove the lock") rather than
letting the runner spin up and immediately die.

The lock check and the state check (§7.1) are intentionally separate:
both BLOCKs may fire on the same startup, and that's fine — the
remediation actions differ, so the operator wants both messages.

Symlink semantics
-----------------
We use :meth:`pathlib.Path.is_file`, which *follows* symlinks. That
matches :class:`KillSwitch._load_persisted_state` and gives sensible
behavior for operators who symlink the lock into a shared location:

- symlink pointing at an existing file → BLOCK (lock is effectively present)
- symlink whose target is missing (broken symlink) → PASS
  (``is_file`` returns ``False`` for broken symlinks; the runner would
  see the same "no lock" state, so preflight must agree)
"""

from __future__ import annotations

from .protocol import PreflightContext
from .result import CheckResult, CheckStatus


class KillSwitchLockCheck:
    """Preflight check for ``data/kill_switch.lock``.

    Satisfies :class:`robo_trader.preflight.protocol.Check` structurally.
    """

    name = "kill_switch_lock"
    description = "Kill switch lock file"
    timeout_seconds = 1.0

    def run(self, context: PreflightContext) -> CheckResult:
        lock_path = context.project_root / "data" / "kill_switch.lock"

        # Path.is_file() follows symlinks: a symlink to a real file is
        # "present"; a broken symlink reports False. Matches the runtime
        # KillSwitch behavior so preflight and runner agree.
        if not lock_path.is_file():
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message="no lock file",
                details={"lock_path": str(lock_path)},
            )

        remediation = (
            "Kill switch lock file present. This is a deny-by-default safety "
            "signal — the runner will refuse to trade while it exists.\n"
            "Either:\n"
            f"  - Remove the lock: rm {lock_path} "
            "(after confirming nothing intentionally placed it)\n"
            '  - Or proceed with: scripts/preflight_check.py --force "reason"'
        )

        return CheckResult(
            name=self.name,
            status=CheckStatus.BLOCK,
            message=f"lock file present at {lock_path}",
            remediation=remediation,
            details={"lock_path": str(lock_path)},
        )
