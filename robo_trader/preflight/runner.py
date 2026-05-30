"""Parallel preflight check runner with plaintext + JSON output.

Coordinates execution of the registered :data:`~robo_trader.preflight.checks.ALL_CHECKS`,
enforces per-check timeouts via ``future.result(timeout=...)``, and
synthesizes BLOCK results for any check that times out or raises.

Threading model
---------------
``ThreadPoolExecutor(max_workers=6)``. Checks are pure functions over a
frozen :class:`PreflightContext` — no shared mutable state — so thread
safety is structural, not requiring locks.

Output
------
Two formatters: :func:`format_plaintext` (the 3am-page operator view)
and :func:`format_json` (for downstream tooling and Phase 2 health
endpoint). Both consume a :class:`RunReport`.

Exit code
---------
Pulled off ``report.exit_code``:

- ``0`` if all checks PASS or WARN
- ``1`` if any check BLOCKs
- ``2`` reserved for the CLI script's ``--force`` bypass path
- ``3`` reserved for "preflight itself failed" (handled in CLI)

The runner returns the report; the script translates to exit codes per §5.2.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .protocol import Check, PreflightContext
from .result import CheckResult, CheckStatus


@dataclass(frozen=True)
class RunReport:
    """Aggregated result of running every check in :data:`ALL_CHECKS`.

    Frozen for the same reason as :class:`CheckResult` — the CLI script
    consumes this without worrying about late mutation by formatters.
    """

    started_at: _dt.datetime
    completed_at: _dt.datetime
    results: List[CheckResult]
    context: PreflightContext = field(repr=False)

    @property
    def duration_ms(self) -> int:
        return int((self.completed_at - self.started_at).total_seconds() * 1000)

    @property
    def passed(self) -> List[CheckResult]:
        return [r for r in self.results if r.status is CheckStatus.PASS]

    @property
    def warned(self) -> List[CheckResult]:
        return [r for r in self.results if r.status is CheckStatus.WARN]

    @property
    def blocked(self) -> List[CheckResult]:
        return [r for r in self.results if r.status is CheckStatus.BLOCK]

    @property
    def exit_code(self) -> int:
        """``0`` if all PASS/WARN, ``1`` if any BLOCKs.

        The ``--force`` bypass path is handled by the CLI script, not
        here — this property reflects the raw check outcome.
        """
        return 1 if self.blocked else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "summary": {
                "total": len(self.results),
                "passed": len(self.passed),
                "warned": len(self.warned),
                "blocked": len(self.blocked),
            },
            "checks": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_all_checks(
    context: PreflightContext,
    checks: Optional[Sequence[Check]] = None,
    max_workers: int = 6,
) -> RunReport:
    """Run every check in parallel and aggregate results.

    Parameters
    ----------
    context
        Shared context passed to every check.
    checks
        Sequence of checks to run. Defaults to :data:`ALL_CHECKS` (resolved
        lazily so tests can pass an empty list or a subset without
        importing the full registry).
    max_workers
        Thread pool size. Defaults to 6 (matches MVP-1's six checks).

    Returns
    -------
    A :class:`RunReport` with one :class:`CheckResult` per check, in the
    same order as ``checks``.
    """
    if checks is None:
        # Lazy import so tests can construct a RunReport without importing
        # the heavy concrete check classes.
        from .checks import ALL_CHECKS

        checks = ALL_CHECKS

    started_at = _dt.datetime.now().astimezone()

    # Submit every check, then collect by index so display order matches
    # registry order regardless of completion order.
    results_by_index: Dict[int, CheckResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_index = {
            pool.submit(_run_one, check, context): idx for idx, check in enumerate(checks)
        }
        for future, idx in future_to_index.items():
            check = checks[idx]
            try:
                results_by_index[idx] = future.result(timeout=check.timeout_seconds)
            except _FutureTimeout:
                results_by_index[idx] = CheckResult(
                    name=getattr(check, "name", f"check_{idx}"),
                    status=CheckStatus.BLOCK,
                    message=f"check exceeded {check.timeout_seconds}s timeout",
                    remediation=(
                        f"The {getattr(check, 'name', 'check')} check did not return "
                        f"within its {check.timeout_seconds}s budget. This usually "
                        "indicates a degraded subprocess (lsof hang) or a stuck I/O "
                        "operation. Investigate before re-running."
                    ),
                    details={"timeout_seconds": check.timeout_seconds},
                    duration_ms=int(check.timeout_seconds * 1000),
                )
            except BaseException as exc:  # noqa: BLE001 — boundary handler
                # Per spec §5.4: checks "should raise rather than return a
                # malformed result"; the runner wraps unexpected exceptions
                # in a BLOCK with the exception info. Code 3 (preflight
                # itself failed) is reserved for things like ImportError at
                # module-load — not per-check failures.
                results_by_index[idx] = CheckResult(
                    name=getattr(check, "name", f"check_{idx}"),
                    status=CheckStatus.BLOCK,
                    message=f"check raised {exc.__class__.__name__}: {exc}",
                    remediation=(
                        "An unexpected exception escaped the check. Likely a bug — "
                        "report with the JSON output from --json mode."
                    ),
                    details={
                        "exception_type": exc.__class__.__name__,
                        "exception_message": str(exc),
                    },
                )

    completed_at = _dt.datetime.now().astimezone()
    ordered = [results_by_index[i] for i in range(len(checks))]
    return RunReport(
        started_at=started_at,
        completed_at=completed_at,
        results=ordered,
        context=context,
    )


def _run_one(check: Check, context: PreflightContext) -> CheckResult:
    """Invoke one check and stamp its ``duration_ms``.

    The check itself leaves ``duration_ms=0`` (the dataclass default);
    the runner replaces it with the measured wall-clock elapsed so
    timings are authoritative.
    """
    start = time.monotonic()
    result = check.run(context)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    # CheckResult is frozen — build a copy with duration filled in.
    return CheckResult(
        name=result.name,
        status=result.status,
        message=result.message,
        remediation=result.remediation,
        details=result.details,
        duration_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

_STATUS_BADGE = {
    CheckStatus.PASS: "[PASS]",
    CheckStatus.WARN: "[WARN]",
    CheckStatus.BLOCK: "[BLOCK]",
}


def format_plaintext(report: RunReport, *, verbose: bool = False) -> str:
    """Render a 3am-page-friendly summary + boxed remediation blocks.

    Format mirrors spec §6.4.1–§6.4.3 exactly.
    """
    n = len(report.results)
    name_width = max((len(r.name) for r in report.results), default=20)
    name_width = max(name_width, 20)
    msg_width = max((len(r.message) for r in report.results), default=40)

    lines: List[str] = []
    lines.append(f"Preflight Safety Gate — checking {n} conditions")
    lines.append("─" * 60)
    for r in report.results:
        badge = _STATUS_BADGE[r.status]
        # uniform-width badge column so columns align
        lines.append(
            f"{badge:<7} {r.name:<{name_width}}  {r.message:<{msg_width}}  ({r.duration_ms}ms)"
        )
    lines.append("─" * 60)

    n_blocked = len(report.blocked)
    n_warned = len(report.warned)
    if n_blocked:
        lines.append(f"{n_blocked}/{n} checks BLOCKED. Cannot proceed.")
    elif n_warned:
        lines.append(f"{n}/{n} checks passed ({n_warned} with warnings). Safe to proceed.")
    else:
        lines.append(f"{n}/{n} checks passed. Safe to proceed.")

    # Boxed remediation per BLOCK
    for idx, r in enumerate(report.blocked, start=1):
        lines.append("")
        lines.extend(_box(f"BLOCK #{idx} — {r.name}", r.remediation or r.message))

    # Indented WARN footer
    for r in report.warned:
        lines.append("")
        lines.append(f"⚠ WARN — {r.name}")
        for ln in (r.remediation or r.message).splitlines():
            lines.append(f"  {ln}")
        lines.append("  Not blocking — proceeding.")

    if verbose:
        lines.append("")
        lines.append("--- details ---")
        for r in report.results:
            if r.details:
                lines.append(f"{r.name}: {json.dumps(r.details, default=str)}")

    return "\n".join(lines) + "\n"


def format_json(report: RunReport) -> str:
    """Render the run as a single JSON object suitable for piping."""
    return json.dumps(report.to_dict(), indent=2, default=str) + "\n"


def _box(title: str, body: str, width: int = 72) -> List[str]:
    """Render a unicode-bordered box for a BLOCK remediation block."""
    inner = width - 2
    top = "╔" + "═" * inner + "╗"
    mid = "╠" + "═" * inner + "╣"
    bot = "╚" + "═" * inner + "╝"

    def _row(s: str) -> str:
        # Pad/truncate so the right border lines up. Truncate long lines
        # with an ellipsis rather than wrap, to keep the box readable in
        # narrow terminals.
        if len(s) > inner - 2:
            s = s[: inner - 3] + "…"
        return "║ " + s.ljust(inner - 2) + " ║"

    out: List[str] = [top, _row(title), mid]
    for ln in body.splitlines():
        out.append(_row(ln))
    out.append(bot)
    return out
