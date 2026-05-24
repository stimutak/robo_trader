"""Zombie (CLOSE_WAIT) connection check on the Gateway port (spec §7.6, G5).

Why this check exists
---------------------
A ``CLOSE_WAIT`` connection sitting on the Gateway API port will block
the runner's next ``ib.connect()`` handshake. The handshake times out
with a cryptic error several seconds in, the runner dies, the watchdog
restarts it, the zombie is still there, and the loop repeats — exactly
the "looks healthy enough to fool monitoring" failure pattern the
preflight gate exists to surface.

START_TRADER.sh already runs a zombie sweep as part of its boot
sequence, but preflight verifies it *took*. The watchdog can also
launch ``runner_async.py`` directly without going through START_TRADER,
so this check is the only line of defense in that path.

Critical implementation constraint
----------------------------------
**Use ``subprocess.run(["lsof", ...])``. NEVER use ``socket.connect_ex``.**

Per CLAUDE.md row 2025-12-06: ``socket.connect_ex`` opens a real socket
which, even when immediately closed, leaves a fresh ``CLOSE_WAIT``
zombie on the port — i.e. a naive Python probe would *create* the
exact failure class this check exists to detect. ``lsof`` reads
kernel socket tables without opening any socket.

Why both Python-zombie AND Gateway-zombie remediation paths
-----------------------------------------------------------
Spec §7.6: zombies come from two sources.

- A prior Python process (runner, sweep script) that opened a socket
  and exited without ``close()`` — these clear with
  ``python3 scripts/gateway_manager.py clear-zombies``.
- The Gateway itself holding stale half-open sockets — these only
  clear with a full Gateway restart
  (``python3 scripts/gateway_manager.py restart``).

We can't tell the two apart from ``lsof`` output alone (the owning PID
in column 2 distinguishes them, but the operator-actionable answer is
"try the cheap option first, then the expensive one"), so the
remediation lists both in order of escalating cost.
"""

from __future__ import annotations

import subprocess
from typing import List

from .protocol import PreflightContext
from .result import CheckResult, CheckStatus


class ZombieConnectionsCheck:
    """BLOCK when any ``CLOSE_WAIT`` connection is present on the Gateway port.

    Satisfies :class:`robo_trader.preflight.protocol.Check` structurally.

    The runner enforces ``timeout_seconds`` as a hard wall budget; the
    inner subprocess timeout (3s, passed to ``subprocess.run``) is the
    primary defense against an ``lsof`` hang under system load.
    """

    name = "zombie_connections"
    description = "Zombie connections"
    timeout_seconds = 3.0

    def run(self, context: PreflightContext) -> CheckResult:
        port = context.target_port
        # NEVER replace this with socket.connect_ex (CLAUDE.md 2025-12-06):
        # that call leaves a CLOSE_WAIT zombie on the Gateway port — i.e.
        # it would MANUFACTURE the failure class this check is meant to
        # detect. lsof reads kernel socket tables without opening a socket.
        argv = ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:CLOSE_WAIT"]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message="zombie check timed out — system likely overloaded",
                remediation=self._build_remediation(port, count=None),
                details={"port": port, "error": "lsof timeout after 3s"},
            )
        except FileNotFoundError:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message="lsof not installed — required for safety checks",
                remediation=(
                    "The preflight zombie-connections check shells out to "
                    "`lsof`, which is missing on this system. On macOS lsof "
                    "is part of the base install; on Linux install via your "
                    "package manager (e.g. `apt install lsof` or "
                    "`yum install lsof`).\n"
                    "Without lsof there is no safe way to enumerate stale "
                    "CLOSE_WAIT sockets without creating new zombies (see "
                    "CLAUDE.md 2025-12-06 row), so this check fails closed."
                ),
                details={"port": port, "error": "lsof binary not found"},
            )

        # lsof exit-code semantics on macOS (verified 2026-05-24):
        #   no matches    → returncode 1, stdout empty   → PASS (normal case)
        #   matches found → returncode 0, stdout has hdr + rows → BLOCK
        # Any other (returncode, stdout) combination is ambiguous; we
        # treat it as PASS but stash the exit code in details so future
        # debugging has the breadcrumb (per task brief).
        stdout = result.stdout or ""
        zombie_pids = _parse_zombie_pids(stdout)
        if zombie_pids:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=(
                    f"{len(zombie_pids)} zombie connection"
                    f"{'s' if len(zombie_pids) != 1 else ''} on port {port}"
                ),
                remediation=self._build_remediation(port, count=len(zombie_pids)),
                details={
                    "port": port,
                    "zombie_count": len(zombie_pids),
                    "zombie_pids": zombie_pids,
                    "lsof_exit_code": result.returncode,
                },
            )

        # Empty parse: PASS. Stash exit code for the "ambiguous returncode
        # with no parseable rows" diagnostic case mentioned above.
        details = {"port": port, "zombie_count": 0}
        if result.returncode not in (0, 1):
            details["lsof_exit_code"] = result.returncode
        return CheckResult(
            name=self.name,
            status=CheckStatus.PASS,
            message=f"no CLOSE_WAIT connections on port {port}",
            details=details,
        )

    def _build_remediation(self, port: int, count: int | None) -> str:
        count_phrase = (
            f"Found {count} zombie connection{'s' if count != 1 else ''}"
            if count is not None
            else "Zombie connections may be present"
        )
        return (
            f"{count_phrase} on port {port}. These will block the runner's "
            "next ib.connect() handshake and cause a silent restart loop.\n\n"
            "Try the cheap fix first (Python-owned zombies):\n"
            "  python3 scripts/gateway_manager.py clear-zombies\n\n"
            "If zombies persist, the Gateway itself is holding stale sockets "
            "— restart it:\n"
            "  python3 scripts/gateway_manager.py restart\n"
            "  (this requires 2FA approval on your IBKR Mobile app)\n\n"
            "Then re-run ./START_TRADER.sh."
        )


def _parse_zombie_pids(stdout: str) -> List[str]:
    """Extract PIDs from lsof CLOSE_WAIT output, skipping any header line.

    lsof on macOS prints a ``COMMAND PID USER ...`` header row when it
    has results, followed by one data row per matching socket. With the
    ``-nP`` flags (no name/port resolution) the header is still present.
    We detect and skip it by spotting the literal ``PID`` token in
    column 2 of the first line, which is unambiguous because real PIDs
    are integers.
    """
    pids: List[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Skip the header row (column 2 is the literal "PID", not a number).
        if parts[1] == "PID":
            continue
        # Defensive: only accept numeric PIDs. Filters any noise lines.
        if not parts[1].isdigit():
            continue
        pids.append(parts[1])
    return pids
