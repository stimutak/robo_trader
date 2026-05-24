"""Tests for the parallel preflight runner + output formatters.

Covers spec §6.1 (parallelism + timeouts), §6.4 (output format), §10
(performance budget assumption).
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path

import pytest

from robo_trader.preflight.protocol import Check, PreflightContext
from robo_trader.preflight.result import CheckResult, CheckStatus
from robo_trader.preflight.runner import (
    RunReport,
    format_json,
    format_plaintext,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Test doubles — three-line classes since Check is a Protocol
# ---------------------------------------------------------------------------


class _FakeCheck:
    """Minimal Check implementation. Tests instantiate with the result it
    should return (or an exception it should raise / a sleep duration)."""

    def __init__(
        self,
        name: str,
        status: CheckStatus = CheckStatus.PASS,
        message: str = "fake ok",
        sleep_seconds: float = 0.0,
        raises: BaseException | None = None,
        timeout_seconds: float = 1.0,
        details: dict | None = None,
    ):
        self.name = name
        self.description = name.replace("_", " ").title()
        self.timeout_seconds = timeout_seconds
        self._status = status
        self._message = message
        self._sleep = sleep_seconds
        self._raises = raises
        self._details = details or {}

    def run(self, context: PreflightContext) -> CheckResult:
        if self._sleep:
            time.sleep(self._sleep)
        if self._raises is not None:
            raise self._raises
        return CheckResult(
            name=self.name,
            status=self._status,
            message=self._message,
            details=dict(self._details),
        )


# ---------------------------------------------------------------------------
# RunReport
# ---------------------------------------------------------------------------


class TestRunReport:
    def _make(self, results: list[CheckResult]) -> RunReport:
        now = _dt.datetime.now().astimezone()
        return RunReport(
            started_at=now,
            completed_at=now + _dt.timedelta(milliseconds=42),
            results=results,
            context=PreflightContext.for_test(Path("/tmp/x")),
        )

    def test_duration_ms_computed_from_timestamps(self) -> None:
        report = self._make([CheckResult(name="x", status=CheckStatus.PASS, message="ok")])
        assert report.duration_ms == 42

    def test_exit_code_zero_when_all_pass(self) -> None:
        report = self._make(
            [
                CheckResult(name="a", status=CheckStatus.PASS, message="ok"),
                CheckResult(name="b", status=CheckStatus.PASS, message="ok"),
            ]
        )
        assert report.exit_code == 0

    def test_exit_code_zero_when_warn_but_no_block(self) -> None:
        # WARN does not block. The plan accepted this — Q11.1 et al.
        report = self._make(
            [
                CheckResult(name="a", status=CheckStatus.WARN, message="meh"),
                CheckResult(name="b", status=CheckStatus.PASS, message="ok"),
            ]
        )
        assert report.exit_code == 0

    def test_exit_code_one_when_any_block(self) -> None:
        report = self._make(
            [
                CheckResult(name="a", status=CheckStatus.PASS, message="ok"),
                CheckResult(name="b", status=CheckStatus.BLOCK, message="bad"),
            ]
        )
        assert report.exit_code == 1

    def test_summary_buckets(self) -> None:
        report = self._make(
            [
                CheckResult(name="p1", status=CheckStatus.PASS, message=""),
                CheckResult(name="p2", status=CheckStatus.PASS, message=""),
                CheckResult(name="w1", status=CheckStatus.WARN, message=""),
                CheckResult(name="b1", status=CheckStatus.BLOCK, message=""),
            ]
        )
        assert len(report.passed) == 2
        assert len(report.warned) == 1
        assert len(report.blocked) == 1

    def test_to_dict_round_trips_through_json(self) -> None:
        report = self._make(
            [
                CheckResult(
                    name="x",
                    status=CheckStatus.BLOCK,
                    message="bad",
                    remediation="fix it",
                    details={"k": 1},
                    duration_ms=5,
                )
            ]
        )
        encoded = json.dumps(report.to_dict())
        decoded = json.loads(encoded)
        assert decoded["summary"]["blocked"] == 1
        assert decoded["checks"][0]["status"] == "BLOCK"
        assert decoded["exit_code"] == 1
        assert decoded["version"] == 1


# ---------------------------------------------------------------------------
# run_all_checks — execution behavior
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    @pytest.fixture
    def context(self, tmp_path: Path) -> PreflightContext:
        return PreflightContext.for_test(tmp_path)

    def test_runs_every_check_in_registry_order(self, context: PreflightContext) -> None:
        checks = [
            _FakeCheck("alpha"),
            _FakeCheck("bravo"),
            _FakeCheck("charlie"),
        ]
        report = run_all_checks(context, checks=checks)
        assert [r.name for r in report.results] == ["alpha", "bravo", "charlie"]

    def test_records_per_check_duration(self, context: PreflightContext) -> None:
        # 50ms sleep — should show up as duration_ms >= ~40 (allowing scheduler slack)
        checks = [_FakeCheck("slow", sleep_seconds=0.05)]
        report = run_all_checks(context, checks=checks)
        assert report.results[0].duration_ms >= 40

    def test_runs_checks_in_parallel(self, context: PreflightContext) -> None:
        # Three checks each sleeping 100ms. Sequential would take ~300ms;
        # parallel should be ~100-150ms. Use a generous upper bound to
        # avoid CI flakiness, but tight enough to prove parallelism.
        checks = [
            _FakeCheck("a", sleep_seconds=0.1),
            _FakeCheck("b", sleep_seconds=0.1),
            _FakeCheck("c", sleep_seconds=0.1),
        ]
        start = time.monotonic()
        run_all_checks(context, checks=checks, max_workers=3)
        elapsed = time.monotonic() - start
        # Sequential would be 300ms; parallel should be well under 250ms.
        assert elapsed < 0.25, f"expected parallel <0.25s, got {elapsed:.3f}s"

    def test_timeout_becomes_block_not_crash(self, context: PreflightContext) -> None:
        # Check sleeps longer than its own timeout — runner must
        # synthesize a BLOCK, not propagate the TimeoutError.
        checks = [_FakeCheck("slow", sleep_seconds=0.5, timeout_seconds=0.05)]
        report = run_all_checks(context, checks=checks)
        assert report.results[0].status is CheckStatus.BLOCK
        assert "timeout" in report.results[0].message.lower()
        assert report.results[0].details["timeout_seconds"] == 0.05

    def test_unexpected_exception_becomes_block_with_traceback(
        self, context: PreflightContext
    ) -> None:
        # Check raises ValueError — runner wraps in BLOCK rather than
        # propagating. Other checks must still run (exception isolation).
        checks = [
            _FakeCheck("explodes", raises=ValueError("kaboom")),
            _FakeCheck("survivor"),
        ]
        report = run_all_checks(context, checks=checks)
        assert report.results[0].status is CheckStatus.BLOCK
        assert "ValueError" in report.results[0].message
        assert "kaboom" in report.results[0].message
        assert report.results[0].details["exception_type"] == "ValueError"
        # Sibling check still ran
        assert report.results[1].status is CheckStatus.PASS

    def test_empty_check_list_is_pass(self, context: PreflightContext) -> None:
        report = run_all_checks(context, checks=[])
        assert report.exit_code == 0
        assert report.results == []

    def test_one_block_one_pass_exits_one(self, context: PreflightContext) -> None:
        checks = [
            _FakeCheck("ok"),
            _FakeCheck("bad", status=CheckStatus.BLOCK, message="bad thing"),
        ]
        report = run_all_checks(context, checks=checks)
        assert report.exit_code == 1
        assert len(report.blocked) == 1


# ---------------------------------------------------------------------------
# format_plaintext
# ---------------------------------------------------------------------------


class TestFormatPlaintext:
    @pytest.fixture
    def context(self, tmp_path: Path) -> PreflightContext:
        return PreflightContext.for_test(tmp_path)

    def test_happy_path_format(self, context: PreflightContext) -> None:
        checks = [
            _FakeCheck("kill_switch_state", message="no triggered state"),
            _FakeCheck("kill_switch_lock", message="no lock file"),
        ]
        report = run_all_checks(context, checks=checks)
        out = format_plaintext(report)
        assert "Preflight Safety Gate" in out
        # Badge is in a fixed-width column, so allow >=1 spaces between
        # badge and name. The brittle exact-spacing assertion ate me once.
        assert "[PASS]" in out and "kill_switch_state" in out
        assert "kill_switch_lock" in out
        assert "2/2 checks passed. Safe to proceed." in out

    def test_block_renders_boxed_remediation(self, context: PreflightContext) -> None:
        checks = [
            _FakeCheck(
                "kill_switch_state",
                status=CheckStatus.BLOCK,
                message="triggered=True",
                details={"trigger_reason": "NVDA loss"},
            ),
        ]
        # Build report manually to force a remediation string
        report = RunReport(
            started_at=_dt.datetime.now().astimezone(),
            completed_at=_dt.datetime.now().astimezone(),
            results=[
                CheckResult(
                    name="kill_switch_state",
                    status=CheckStatus.BLOCK,
                    message="triggered=True",
                    remediation="Kill switch is triggered.\nRun rm data/kill_switch_state.json",
                )
            ],
            context=context,
        )
        out = format_plaintext(report)
        assert "[BLOCK]" in out and "kill_switch_state" in out
        assert "1/1 checks BLOCKED. Cannot proceed." in out
        assert "BLOCK #1 — kill_switch_state" in out
        assert "Kill switch is triggered." in out
        assert "╔" in out and "╚" in out  # box drawing chars

    def test_warn_renders_after_summary(self, context: PreflightContext) -> None:
        report = RunReport(
            started_at=_dt.datetime.now().astimezone(),
            completed_at=_dt.datetime.now().astimezone(),
            results=[
                CheckResult(
                    name="risk_threshold",
                    status=CheckStatus.WARN,
                    message="max_position_loss_pct=0.018 (<0.02)",
                    remediation="Consider RISK_MAX_POSITION_LOSS_PCT=0.05",
                )
            ],
            context=context,
        )
        out = format_plaintext(report)
        # Allow fixed-width column spacing between badge and name.
        assert "[WARN]" in out and "risk_threshold" in out
        assert "1/1 checks passed (1 with warnings). Safe to proceed." in out
        assert "⚠ WARN — risk_threshold" in out
        # WARN footer must come AFTER the summary line
        warn_pos = out.find("⚠ WARN")
        summary_pos = out.find("Safe to proceed.")
        assert warn_pos > summary_pos

    def test_verbose_appends_details_block(self, context: PreflightContext) -> None:
        report = RunReport(
            started_at=_dt.datetime.now().astimezone(),
            completed_at=_dt.datetime.now().astimezone(),
            results=[
                CheckResult(
                    name="x",
                    status=CheckStatus.PASS,
                    message="ok",
                    details={"foo": "bar"},
                )
            ],
            context=context,
        )
        terse = format_plaintext(report, verbose=False)
        verbose = format_plaintext(report, verbose=True)
        assert "--- details ---" not in terse
        assert "--- details ---" in verbose
        assert '"foo": "bar"' in verbose


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------


class TestFormatJson:
    def test_is_valid_json_with_expected_shape(self, tmp_path: Path) -> None:
        context = PreflightContext.for_test(tmp_path)
        checks = [_FakeCheck("a"), _FakeCheck("b", status=CheckStatus.WARN, message="meh")]
        report = run_all_checks(context, checks=checks)
        encoded = format_json(report)
        decoded = json.loads(encoded)
        assert decoded["version"] == 1
        assert decoded["summary"] == {
            "total": 2,
            "passed": 1,
            "warned": 1,
            "blocked": 0,
        }
        assert decoded["exit_code"] == 0
        assert [c["name"] for c in decoded["checks"]] == ["a", "b"]
