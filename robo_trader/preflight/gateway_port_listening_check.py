"""Gateway API port LISTEN check (spec §7.5, G4).

Why this check exists
---------------------
START_TRADER.sh already waits for the Gateway API port to enter LISTEN
state before launching the runner. But the watchdog can — and does —
restart ``runner_async.py`` directly without going through START_TRADER.
In that path, preflight is the first (and only) line of defense against
"runner started, IBKR Gateway never came up, runner immediately dies in
a loop." That cycle is one of the silent failure modes the gate exists
to surface.

The check is intentionally narrow: it reports only whether something is
listening on the configured port (4002 for paper, 4001 for live). It
does NOT try to verify the listener is actually IB Gateway, nor does it
attempt an API handshake. The next layer (the runner's own connect
call) is responsible for that. Preflight just rules out the trivially
hopeless case.

Critical implementation constraint
----------------------------------
**Use ``subprocess.run(["lsof", ...])``. NEVER use ``socket.connect_ex``.**

Per CLAUDE.md row dated 2025-12-06 (and as called out explicitly in spec
§7.5): ``socket.connect_ex`` opens a TCP socket which, even when closed
immediately, leaves a ``CLOSE_WAIT`` zombie connection on the Gateway
port. That zombie then blocks the *real* ib_insync handshake the runner
attempts microseconds later. The very failure class this check exists
to guard against — port-not-actually-available — would be CREATED by a
naive Python port check.

``lsof`` reads kernel socket tables without opening any new socket and
is the only safe option. There is no portable Python-API equivalent.
"""

from __future__ import annotations

import subprocess

from .protocol import PreflightContext
from .result import CheckResult, CheckStatus


class GatewayPortListeningCheck:
    """BLOCK when no process is LISTENing on the configured Gateway port.

    Satisfies :class:`robo_trader.preflight.protocol.Check` structurally.

    The runner enforces ``timeout_seconds`` as a hard wall budget; the
    inner subprocess timeout (3s, passed to ``subprocess.run``) is the
    primary defense against an ``lsof`` hang under system load.
    """

    name = "gateway_port_listening"
    description = "Gateway port listening"
    timeout_seconds = 3.0

    def run(self, context: PreflightContext) -> CheckResult:
        port = context.target_port
        # NEVER replace this with socket.connect_ex (CLAUDE.md 2025-12-06):
        # that call leaves a CLOSE_WAIT zombie on the Gateway port which
        # blocks the runner's subsequent ib.connect() handshake — i.e. it
        # MANUFACTURES the failure class this check is meant to detect.
        # lsof reads kernel socket tables without opening a socket.
        argv = ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"]
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
                message="port check timed out — system likely overloaded",
                remediation=self._build_remediation(port),
                details={"port": port, "error": "lsof timeout after 3s"},
            )
        except FileNotFoundError:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message="lsof not installed — required for safety checks",
                remediation=(
                    "The preflight gateway-port check shells out to `lsof`, "
                    "which is missing on this system. On macOS lsof is part of "
                    "the base install; on Linux install via your package "
                    "manager (e.g. `apt install lsof` or `yum install lsof`).\n"
                    "Without lsof there is no safe way to probe a TCP port "
                    "without creating a CLOSE_WAIT zombie (see CLAUDE.md "
                    "2025-12-06 row), so this check fails closed."
                ),
                details={"port": port, "error": "lsof binary not found"},
            )

        # Treat empty stdout the same as non-zero returncode: nothing
        # is actually LISTENing. lsof returns 1 when no matching socket
        # exists, but a defensive check against stdout handles the edge
        # case of a returncode-0 + empty-output combination some lsof
        # builds produce.
        if result.returncode == 0 and result.stdout.strip():
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message=f"port {port} listening",
                details={"port": port},
            )

        return CheckResult(
            name=self.name,
            status=CheckStatus.BLOCK,
            message=f"port {port} not listening",
            remediation=self._build_remediation(port),
            details={
                "port": port,
                "lsof_returncode": result.returncode,
                "lsof_stdout": result.stdout.strip(),
            },
        )

    def _build_remediation(self, port: int) -> str:
        return (
            f"Gateway API port {port} is not listening. Without it the runner "
            "cannot connect to IBKR and will spin in a restart loop.\n\n"
            "To bring it up:\n"
            "  ./scripts/start_gateway.sh\n"
            "    (or)\n"
            "  python3 scripts/gateway_manager.py start --paper\n\n"
            "If 2FA is pending, check your IBKR Mobile app on your phone.\n"
            "Once Gateway is up, re-run ./START_TRADER.sh."
        )
