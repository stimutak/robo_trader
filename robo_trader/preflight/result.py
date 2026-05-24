"""Immutable result type for preflight checks.

Each :class:`~robo_trader.preflight.protocol.Check` returns a
:class:`CheckResult`. The runner aggregates results and decides exit code
based on the worst severity seen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict


class CheckStatus(Enum):
    """Severity tiers (spec §5.6).

    - ``PASS``  — the check found no issue.
    - ``WARN``  — found something the operator should know but not a blocker.
    - ``BLOCK`` — found something that should prevent startup until resolved.

    ``REQUIRE_CONFIRM`` was considered and rejected (the runner can be
    invoked from the watchdog at 3am with no human at the keyboard; an
    interactive prompt becomes a hang). The bypass mechanism in
    ``scripts/preflight_check.py --force "reason"`` is the right knob for
    "I know about this block and I'm proceeding anyway."
    """

    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class CheckResult:
    """The result of running a single :class:`Check`.

    Frozen so a result can be passed around the runner without anyone
    accidentally mutating its severity mid-flight. Tests rely on
    frozen-ness to assert immutability invariants.

    Attributes
    ----------
    name
        Stable identifier (kebab-case), e.g. ``"kill_switch_state"``.
        Used as the JSON-output dict key and for grep-friendly logging.
    status
        One of :class:`CheckStatus`.
    message
        One-line summary for human display next to the status badge.
    remediation
        Operator instructions, multiline OK. Shown indented below the
        message in plaintext output; exposed as a separate key in JSON
        output for downstream tooling.
    details
        Machine-readable extras (file paths, threshold values, port
        numbers). Stays out of the human display unless ``--verbose`` is
        set, always present in JSON output.
    duration_ms
        Populated by the runner; left as 0 by the check itself.
    """

    name: str
    status: CheckStatus
    message: str
    remediation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output.

        Returns a plain dict with the enum collapsed to its string value
        so the result round-trips through ``json.dumps`` without a custom
        encoder.
        """
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
