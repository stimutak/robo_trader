"""Fail-closed startup coverage for existing positions."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.portfolio import Portfolio
from robo_trader.risk_manager import Position
from robo_trader.runner_async import (
    AsyncRunner,
    UnprotectedExistingPositionsError,
    _setup_continuous_runner,
    run_continuous,
)
from robo_trader.stop_loss_monitor import StopStatus

ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_POLICY = ROOT / "scripts" / "watchdog_restart_policy.py"
WATCHDOG_GUARD = ROOT / "scripts" / "watchdog_restart_guard.sh"


class _AliveTask:
    def done(self) -> bool:
        return False


def _protected_runner(
    *,
    quantity: int = 10,
    status: StopStatus = StopStatus.PENDING,
    stop_quantity: int | None = None,
    source: str = "live_protective",
    available: bool = True,
    live_grade: bool = True,
    event_offset_seconds: float = -1.0,
    receipt_offset_seconds: float = -1.0,
) -> AsyncRunner:
    now = datetime.now(timezone.utc)
    monotonic_now = 1000.0
    stop_qty = quantity if stop_quantity is None else stop_quantity
    monitor = SimpleNamespace(
        monitoring_active=True,
        monitor_task=_AliveTask(),
        active_stops={
            "default:AAPL": SimpleNamespace(
                status=status,
                position_qty=stop_qty,
            )
        },
        last_prices={"AAPL": 101.0},
        price_event_times={"AAPL": now + timedelta(seconds=event_offset_seconds)},
        price_receipt_monotonic={"AAPL": monotonic_now + receipt_offset_seconds},
        max_price_age_seconds=10,
        _utcnow=lambda: now,
        _monotonic=lambda: monotonic_now,
        _stop_key=lambda symbol: f"default:{symbol}",
    )
    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {"AAPL": Position("AAPL", quantity, 100.0)}
    runner.stop_loss_monitor = monitor
    runner._protective_feed_status = {
        "AAPL": {
            "available": available,
            "live_grade": live_grade,
            "source": source,
        }
    }
    return runner


def _assert_protection(runner: AsyncRunner) -> None:
    AsyncRunner._assert_existing_position_protection(runner)


def test_exact_fresh_monitor_owned_live_price_satisfies_startup_invariant() -> None:
    _assert_protection(_protected_runner())


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda runner: setattr(runner, "stop_loss_monitor", None),
            "stop_monitor_missing",
        ),
        (
            lambda runner: setattr(runner.stop_loss_monitor, "monitoring_active", False),
            "stop_monitor_not_running",
        ),
        (
            lambda runner: runner.stop_loss_monitor.active_stops.clear(),
            "active_stop_missing",
        ),
        (
            lambda runner: setattr(
                runner.stop_loss_monitor.active_stops["default:AAPL"],
                "status",
                StopStatus.CANCELLED,
            ),
            "active_stop_not_pending",
        ),
        (
            lambda runner: setattr(
                runner.stop_loss_monitor.active_stops["default:AAPL"],
                "position_qty",
                9,
            ),
            "active_stop_quantity_mismatch",
        ),
        (
            lambda runner: runner.stop_loss_monitor.last_prices.clear(),
            "protective_price_state_invalid",
        ),
    ],
)
def test_missing_or_invalid_stop_coverage_aborts_startup(mutate, reason: str) -> None:
    runner = _protected_runner()
    mutate(runner)

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        _assert_protection(runner)

    assert caught.value.reason_code == reason
    assert caught.value.position_count == 1


def test_recent_historical_bar_never_satisfies_protective_contract() -> None:
    runner = _protected_runner(source="historical_bar")
    runner.latest_prices = {"AAPL": 101.0}
    runner.latest_price_times = {"AAPL": datetime.now(timezone.utc)}

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        _assert_protection(runner)

    assert caught.value.reason_code == "live_protective_feed_unavailable"


@pytest.mark.parametrize(
    ("event_offset", "receipt_offset"),
    [
        (-11.0, -1.0),
        (1.0, -1.0),
        (-1.0, -11.0),
        (-1.0, 1.0),
    ],
)
def test_stale_or_future_protective_times_abort(event_offset: float, receipt_offset: float) -> None:
    runner = _protected_runner(
        event_offset_seconds=event_offset,
        receipt_offset_seconds=receipt_offset,
    )

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        _assert_protection(runner)

    assert caught.value.reason_code == "protective_price_stale"


def test_zero_quantity_rows_do_not_require_protective_feed() -> None:
    runner = _protected_runner(quantity=0)
    runner.stop_loss_monitor = None
    runner._protective_feed_status = {}

    _assert_protection(runner)


@pytest.mark.asyncio
async def test_database_position_load_failure_is_fatal_not_empty_fallback() -> None:
    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {}
    runner.db = SimpleNamespace(
        get_account_info=AsyncMock(side_effect=RuntimeError("database unavailable"))
    )
    runner.cfg = SimpleNamespace(default_cash=100_000)

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.load_existing_positions()

    assert caught.value.reason_code == "position_load_failed"
    assert runner.positions == {}


@pytest.mark.asyncio
async def test_stop_registration_failure_is_fatal() -> None:
    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {}
    runner.db = SimpleNamespace(
        get_account_info=AsyncMock(return_value=None),
        get_positions=AsyncMock(
            return_value=[{"symbol": "AAPL", "quantity": 10, "avg_cost": 100.0}]
        ),
    )
    runner.portfolio = Portfolio(100_000)
    runner.stop_loss_monitor = SimpleNamespace(
        add_stop_loss=AsyncMock(side_effect=RuntimeError("registration failed"))
    )
    runner.use_trailing_stop = False
    runner.stop_loss_percent = 0.02
    runner.trailing_stop_pct = 0.05

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.load_existing_positions()

    assert caught.value.reason_code == "stop_registration_failed"
    assert caught.value.position_count == 1


@pytest.mark.asyncio
async def test_run_cleans_partial_setup_on_protection_abort() -> None:
    error = UnprotectedExistingPositionsError("default", 1, "protective_price_stale")
    runner = object.__new__(AsyncRunner)
    runner.setup = AsyncMock(side_effect=error)
    runner.cleanup = AsyncMock()

    with pytest.raises(UnprotectedExistingPositionsError):
        await AsyncRunner.run(runner, [])

    runner.cleanup.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_continuous_runner_setup_cleans_before_propagating_abort() -> None:
    error = UnprotectedExistingPositionsError("default", 1, "active_stop_missing")
    runner = SimpleNamespace(
        setup=AsyncMock(side_effect=error),
        _attach_health_monitor=AsyncMock(),
        cleanup=AsyncMock(),
    )

    with pytest.raises(UnprotectedExistingPositionsError):
        await _setup_continuous_runner(runner)

    runner.cleanup.assert_awaited_once_with()
    runner._attach_health_monitor.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_reaches_ibkr_and_database_after_stop_monitor_failure() -> None:
    runner = object.__new__(AsyncRunner)
    runner.health = None
    runner.subprocess_monitor_task = None
    runner.risk_monitor_task = None
    runner.cleanup_task = None
    runner.stop_loss_monitor = SimpleNamespace(
        stop_monitoring=AsyncMock(side_effect=RuntimeError("monitor failed")),
        get_metrics=lambda: (_ for _ in ()).throw(RuntimeError("metrics failed")),
    )
    runner.use_advanced_risk = False
    runner.advanced_risk = None
    runner.ib = SimpleNamespace(
        disconnect=AsyncMock(side_effect=RuntimeError("disconnect failed")),
        stop=AsyncMock(),
    )
    runner.db = SimpleNamespace(close=AsyncMock())
    runner.ws_client = None

    await runner.cleanup()

    runner.stop_loss_monitor.stop_monitoring.assert_awaited_once_with()
    runner.ib.disconnect.assert_awaited_once_with()
    runner.ib.stop.assert_awaited_once_with()
    runner.db.close.assert_awaited_once_with()


def test_continuous_loop_treats_unprotected_positions_as_nonretryable() -> None:
    source = inspect.getsource(run_continuous)
    handler = source.index("except UnprotectedExistingPositionsError as e:")
    generic = source.index("except Exception as e:", handler)

    assert handler < generic
    assert "fatal_safety_exit_written = True" in source[handler:generic]
    assert "raise SystemExit(6) from e" in source[handler:generic]
    assert "if not fatal_safety_exit_written:" in source


def _run_watchdog_policy(audit: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(WATCHDOG_POLICY), str(audit)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_watchdog_policy_suppresses_terminal_protection_restart(tmp_path: Path) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "reason": "unprotected_existing_positions",
                "exit_code": 6,
            }
        )
    )

    result = _run_watchdog_policy(audit)

    assert result.returncode == 20
    assert result.stdout.strip() == "unprotected_existing_positions"


@pytest.mark.parametrize(
    "payload",
    [
        {"reason": "clean_shutdown", "exit_code": 0},
        {"reason": "unprotected_existing_positions", "exit_code": 1},
        {"reason": "recovery_exhausted", "exit_code": 6},
    ],
)
def test_watchdog_policy_allows_nonterminal_exit(tmp_path: Path, payload: dict) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(json.dumps(payload))

    result = _run_watchdog_policy(audit)

    assert result.returncode == 0
    assert result.stdout.strip() == "nonterminal_exit"


def test_watchdog_policy_blocks_untrusted_existing_audit(tmp_path: Path) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text("{not-json")

    result = _run_watchdog_policy(audit)

    assert result.returncode == 21
    assert result.stdout.strip() == "exit_audit_unreadable"


def test_watchdog_policy_blocks_parser_recursion_failure(tmp_path: Path) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text("[" * 10_000 + "0" + "]" * 10_000)

    result = _run_watchdog_policy(audit)

    assert result.returncode == 21
    assert result.stdout.strip() == "exit_audit_unreadable"


@pytest.mark.parametrize(
    ("policy_rc", "restart_allowed"),
    [
        ("0", True),
        ("1", False),
        ("20", False),
        ("21", False),
        ("99", False),
        ("invalid", False),
        ("", False),
    ],
)
def test_watchdog_guard_requires_explicit_policy_success(
    policy_rc: str, restart_allowed: bool
) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; watchdog_restart_allowed_for_policy_rc "$2"',
            "watchdog-guard-test",
            str(WATCHDOG_GUARD),
            policy_rc,
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert (result.returncode == 0) is restart_allowed


def test_watchdog_calls_policy_before_authoritative_launcher() -> None:
    source = (ROOT / "scripts" / "watchdog.sh").read_text()
    restart_body = source.split("restart_trader() {", 1)[1].split("# Main loop", 1)[0]

    assert restart_body.index('"$RESTART_POLICY"') < restart_body.index(
        '"$PROJECT_DIR/START_TRADER.sh"'
    )
    assert "return 2" in restart_body
