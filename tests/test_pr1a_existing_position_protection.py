"""Fail-closed startup coverage for existing positions."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from robo_trader.config import RuntimeContract
from robo_trader.database_validator import ValidationError
from robo_trader.execution import ExecutionResult, LocalPaperExecutionEvidence
from robo_trader.paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperOutcomeProvenance,
    LocalPaperTerminalOutcome,
)
from robo_trader.portfolio import Portfolio
from robo_trader.protective_quote_evidence import (
    ProtectiveQuoteSource,
    _produce_protective_quote,
)
from robo_trader.risk_manager import Position
from robo_trader.runner_async import (
    AsyncRunner,
    UnprotectedExistingPositionsError,
    _cleanup_runner_owned,
    _clear_exit_audit,
    _setup_continuous_runner,
    _write_exit_audit,
    run_continuous,
)
from robo_trader.stop_loss_monitor import (
    StopExecutionPhase,
    StopExecutionPhaseRecord,
    StopLossMonitor,
    StopLossOrder,
    StopStatus,
    StopType,
    _PendingStopTrigger,
)

ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_POLICY = ROOT / "scripts" / "watchdog_restart_policy.py"
WATCHDOG_GUARD = ROOT / "scripts" / "watchdog_restart_guard.sh"
TEST_RUNTIME_CONTRACT = RuntimeContract(
    environment="test",
    execution_mode="paper",
    execution_source="paper_simulator",
    ibkr_host="127.0.0.1",
    ibkr_port=4002,
    ibkr_readonly=True,
    database_path="/tmp/robotrader-existing-position-test.db",
    account_alias="***PER",
    account_type="paper",
    model_artifact_set="test",
    build_id="test-build",
    state_namespace="paper",
    safety_account_scope="acct_v1_" + ("0123456789abcdef" * 4),
    safety_execution_domain_scope="paper-simulator-v1",
)


class _AliveTask:
    def done(self) -> bool:
        return False


async def _unused_reduction(_stop, _order):
    raise AssertionError("stop reduction callback was not expected")


def _reduction_callback(executor):
    async def execute(stop, order):
        del stop
        return await executor.place_order_async(order)

    return execute


def _filled_terminal_outcome(
    order,
    *,
    fill_price: Decimal = Decimal("97.5"),
) -> LocalPaperTerminalOutcome:
    quantity = Decimal(order.quantity)
    observed_at = datetime.now(timezone.utc)
    return LocalPaperTerminalOutcome(
        order_ref=order.order_ref,
        status=LocalPaperOrderStatus.FILLED,
        requested_quantity=quantity,
        filled_quantity=quantity,
        remaining_quantity=Decimal("0"),
        exact_fill_price=fill_price,
        observed_at=observed_at,
        provenance=LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR,
        terminal=True,
        message="exact local paper fill",
        fill_evidence=LocalPaperExecutionEvidence(
            execution_id="lpfill-"
            + hashlib.sha256(order.order_ref.encode("utf-8")).hexdigest()[:32],
            filled_quantity=quantity,
            exact_fill_price=fill_price,
            commission_minor=0,
            commission_currency="USD",
            commission_source="LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1",
            occurred_at=observed_at,
        ),
    )


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
    stop = StopLossOrder(
        symbol="AAPL",
        position_qty=stop_qty,
        stop_price=98.0,
        entry_price=100.0,
        stop_type=StopType.FIXED,
        created_at=now - timedelta(minutes=1),
        status=status,
        portfolio_id="default",
    )
    if status is StopStatus.TRIGGERED:
        stop.trigger_price = 97.0
        stop.triggered_at = now + timedelta(seconds=event_offset_seconds)
    monitor = StopLossMonitor(
        execute_reduction=_unused_reduction,
        risk_manager=None,
        portfolio_id="default",
    )
    monitor.monitoring_active = True
    monitor.monitor_task = _AliveTask()
    monitor.pending_drain_timeout_seconds = 30.0
    monitor.queue_timeout_seconds = 30.0
    monitor.broker_attempt_timeout_seconds = 30.0
    monitor.settlement_timeout_seconds = 30.0
    monitor.active_stops = {"default:AAPL": stop}
    monitor.last_prices = {"AAPL": 101.0}
    event_time = now + timedelta(seconds=event_offset_seconds)
    receipt_time = monotonic_now + receipt_offset_seconds
    monitor.price_event_times = {"AAPL": event_time}
    monitor.price_receipt_monotonic = {"AAPL": receipt_time}
    monitor.price_receipt_orders = {"AAPL": 1}
    monitor._price_receipt_order = 1
    monitor._utcnow = lambda: now
    monitor._monotonic = lambda: monotonic_now
    if available and live_grade and source == "live_protective":
        monitor._protective_quote_evidence["AAPL"] = _produce_protective_quote(
            monitor,
            portfolio_id="default",
            symbol="AAPL",
            price=Decimal("101.0"),
            source_timestamp=event_time,
            receipt_monotonic=float(receipt_time),
            receipt_order=1,
            source=ProtectiveQuoteSource.LIVE_BROKER,
            con_id=265598,
            transport_generation="test-generation-1",
            source_event_id="protected-runner-event",
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
            "con_id": 265598,
            "transport_generation": "test-generation-1",
        }
    }
    return runner


def _assert_protection(runner: AsyncRunner) -> None:
    AsyncRunner._assert_existing_position_protection(runner)


def _executed_stop(
    quantity: int,
    *,
    symbol: str = "AAPL",
    portfolio_id: str = "default",
) -> StopLossOrder:
    stop = StopLossOrder(
        symbol=symbol,
        position_qty=quantity,
        stop_price=98.0,
        entry_price=100.0,
        stop_type=StopType.FIXED,
        created_at=datetime.now(timezone.utc),
        portfolio_id=portfolio_id,
    )
    stop.status = StopStatus.EXECUTED
    stop.trigger_price = 97.0
    stop.triggered_at = datetime.now(timezone.utc)
    return stop


def _phase_record(
    stop: StopLossOrder,
    phase: StopExecutionPhase,
    *,
    started: float = 999.0,
    timeout: float = 30.0,
) -> StopExecutionPhaseRecord:
    return StopExecutionPhaseRecord(
        stop=stop,
        phase=phase,
        started_monotonic=started,
        timeout_seconds=timeout,
        deadline_monotonic=started + timeout,
    )


def test_exact_fresh_monitor_owned_live_price_satisfies_startup_invariant() -> None:
    _assert_protection(_protected_runner())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop_price", float("nan")),
        ("stop_price", 0.0),
        ("entry_price", float("nan")),
        ("entry_price", 0.0),
        ("position_qty", True),
        ("position_qty", 1.5),
        ("stop_type", "fixed"),
        ("created_at", datetime.now()),
        ("created_at", datetime.now(timezone.utc) + timedelta(hours=1)),
    ],
)
def test_stop_structure_matrix_fails_closed(field, value) -> None:
    runner = _protected_runner()
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    setattr(stop, field, value)

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        _assert_protection(runner)

    assert caught.value.reason_code == "stop_structure_invalid"


@pytest.mark.parametrize("tamper", ["naive", "before_created", "future"])
def test_trigger_timestamp_structure_fails_closed(tamper) -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    if tamper == "naive":
        stop.triggered_at = stop.triggered_at.replace(tzinfo=None)
    elif tamper == "before_created":
        stop.triggered_at = stop.created_at - timedelta(seconds=1)
    else:
        stop.triggered_at = runner.stop_loss_monitor._utcnow() + timedelta(hours=1)
    runner.stop_loss_monitor._queued_stop_orders["default:AAPL"] = _phase_record(
        stop,
        StopExecutionPhase.QUEUED,
    )

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        runner._assert_existing_position_protection(
            allow_runtime_tracked_states=True,
        )

    assert caught.value.reason_code == "stop_structure_invalid"


async def _cached_quote_trigger_runner():
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    current_time = [event_time]
    monotonic_now = [100.0]
    monitor = StopLossMonitor(
        execute_reduction=_unused_reduction,
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    monitor._utcnow = lambda: current_time[0]
    monitor._monotonic = lambda: monotonic_now[0]
    assert await monitor.update_price("AAPL", 97.0, source_timestamp=event_time)

    current_time[0] = event_time + timedelta(seconds=1)
    monotonic_now[0] = 101.0
    position = Position("AAPL", 10, 100.0)
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    assert stop.created_at > event_time
    assert await monitor.check_stops() == [stop]
    assert stop.triggered_at == event_time
    stop_key = monitor._stop_key("AAPL")
    monitor._queued_stop_orders[stop_key] = monitor._new_phase_record(
        stop,
        StopExecutionPhase.QUEUED,
        monitor.queue_timeout_seconds,
    )
    monitor.monitoring_active = True
    monitor.monitor_task = _AliveTask()

    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {"AAPL": position}
    runner.stop_loss_monitor = monitor
    runner._protective_feed_status = {}
    return runner, stop, current_time


@pytest.mark.asyncio
async def test_cached_accepted_quote_can_trigger_newer_stop_with_exact_lineage() -> None:
    runner, _stop, _current_time = await _cached_quote_trigger_runner()

    runner._assert_existing_position_protection(
        allow_runtime_tracked_states=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["receipt_order", "future_event", "orphan"])
async def test_cached_trigger_exception_rejects_tampered_or_orphan_lineage(
    tamper,
) -> None:
    runner, stop, current_time = await _cached_quote_trigger_runner()
    monitor = runner.stop_loss_monitor
    evidence = monitor._latched_stop_crossings[id(stop)]
    if tamper == "receipt_order":
        replacement = _PendingStopTrigger(
            stop=stop,
            trigger_price=evidence.trigger_price,
            event_time=evidence.event_time,
            receipt_monotonic=evidence.receipt_monotonic,
            receipt_order=monitor._price_receipt_order + 1,
            drain_timeout_seconds=evidence.drain_timeout_seconds,
            drain_deadline_monotonic=evidence.drain_deadline_monotonic,
        )
        monitor._latched_stop_crossings[id(stop)] = replacement
    elif tamper == "future_event":
        future_time = current_time[0] + timedelta(hours=1)
        stop.triggered_at = future_time
        replacement = _PendingStopTrigger(
            stop=stop,
            trigger_price=evidence.trigger_price,
            event_time=future_time,
            receipt_monotonic=evidence.receipt_monotonic,
            receipt_order=evidence.receipt_order,
            drain_timeout_seconds=evidence.drain_timeout_seconds,
            drain_deadline_monotonic=evidence.drain_deadline_monotonic,
        )
        monitor._latched_stop_crossings[id(stop)] = replacement
    else:
        monitor._queued_stop_orders.clear()
        monitor.active_stops.clear()

    with pytest.raises(UnprotectedExistingPositionsError):
        runner._assert_existing_position_protection(
            allow_runtime_tracked_states=True,
        )


@pytest.mark.asyncio
async def test_replacement_cleans_untracked_old_latched_evidence() -> None:
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor = StopLossMonitor(
        execute_reduction=_unused_reduction,
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    monitor._utcnow = lambda: now
    monitor._monotonic = lambda: 100.0
    position = Position("AAPL", 10, 100.0)
    old_stop = await monitor.add_stop_loss("AAPL", position)
    assert await monitor.update_price("AAPL", 97.0, source_timestamp=now)
    assert id(old_stop) in monitor._latched_stop_crossings

    replacement = await monitor.add_stop_loss("AAPL", position)

    assert replacement is monitor.active_stops["default:AAPL"]
    assert id(old_stop) not in monitor._latched_stop_crossings


@pytest.mark.asyncio
async def test_cancel_stop_cleans_latched_evidence() -> None:
    runner, stop, _current_time = await _cached_quote_trigger_runner()
    monitor = runner.stop_loss_monitor
    assert id(stop) in monitor._latched_stop_crossings

    assert monitor.cancel_stop("AAPL") is True

    assert id(stop) not in monitor._latched_stop_crossings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("success", True),
        ("failure", False),
        ("exception", False),
    ],
)
async def test_direct_stop_execution_always_cleans_exact_phase_state(
    outcome,
    expected,
) -> None:
    class _Executor:
        async def place_order_async(self, order):
            if outcome == "success":
                return _filled_terminal_outcome(order)
            if outcome == "failure":
                return ExecutionResult(False, "rejected")
            raise RuntimeError("transport failure")

    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(_Executor()),
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    monitor._utcnow = lambda: now
    monitor._monotonic = lambda: 100.0
    stop = await monitor.add_stop_loss("AAPL", Position("AAPL", 10, 100.0))
    stop.status = StopStatus.TRIGGERED
    stop.triggered_at = now
    stop.trigger_price = 97.0

    with patch(
        "robo_trader.stop_loss_monitor.asyncio.sleep",
        AsyncMock(),
    ):
        assert await monitor.execute_stop_loss(stop) is expected

    assert "default:AAPL" not in monitor._inflight_stop_orders
    assert "default:AAPL" not in monitor._queued_stop_orders
    assert id(stop) not in monitor._latched_stop_crossings
    if outcome != "success":
        assert stop.status is StopStatus.FAILED


@pytest.mark.asyncio
async def test_direct_obsolete_execution_preserves_replacement_phase_records() -> None:
    monitor = StopLossMonitor(
        execute_reduction=_unused_reduction,
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    position = Position("AAPL", 10, 100.0)
    old_stop = await monitor.add_stop_loss("AAPL", position)
    replacement = await monitor.add_stop_loss("AAPL", position)
    stop_key = "default:AAPL"
    replacement_queued = _phase_record(
        replacement,
        StopExecutionPhase.QUEUED,
    )
    replacement_inflight = _phase_record(
        replacement,
        StopExecutionPhase.BROKER_WAIT,
    )
    monitor._queued_stop_orders[stop_key] = replacement_queued
    monitor._inflight_stop_orders[stop_key] = replacement_inflight

    assert await monitor.execute_stop_loss(old_stop) is False

    assert monitor._queued_stop_orders[stop_key] is replacement_queued
    assert monitor._inflight_stop_orders[stop_key] is replacement_inflight


@pytest.mark.asyncio
async def test_direct_exact_queued_stop_has_single_broker_owner_during_await() -> None:
    broker_started = asyncio.Event()
    release_broker = asyncio.Event()

    class _GatedExecutor:
        async def place_order_async(self, order):
            broker_started.set()
            await release_broker.wait()
            return _filled_terminal_outcome(order)

    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(_GatedExecutor()),
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    monitor._utcnow = lambda: now
    stop = await monitor.add_stop_loss("AAPL", Position("AAPL", 10, 100.0))
    stop.status = StopStatus.TRIGGERED
    stop.triggered_at = now
    stop.trigger_price = 97.0
    stop_key = "default:AAPL"
    monitor._queued_stop_orders[stop_key] = _phase_record(
        stop,
        StopExecutionPhase.QUEUED,
    )

    execution = asyncio.create_task(monitor.execute_stop_loss(stop))
    await broker_started.wait()

    assert stop_key not in monitor._queued_stop_orders
    broker_record = monitor._inflight_stop_orders[stop_key]
    assert broker_record.stop is stop
    assert broker_record.phase is StopExecutionPhase.BROKER_WAIT

    release_broker.set()
    assert await execution is True
    assert stop_key not in monitor._queued_stop_orders
    assert stop_key not in monitor._inflight_stop_orders


@pytest.mark.asyncio
async def test_direct_success_keeps_post_fill_visible_through_callback() -> None:
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()

    class _Executor:
        async def place_order_async(self, order):
            return _filled_terminal_outcome(order)

    async def gated_callback(_stop, _result) -> None:
        callback_started.set()
        await release_callback.wait()

    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(_Executor()),
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
        position_closed_callback=gated_callback,
    )
    monitor._utcnow = lambda: now
    stop = await monitor.add_stop_loss("AAPL", Position("AAPL", 10, 100.0))
    stop.status = StopStatus.TRIGGERED
    stop.triggered_at = now
    stop.trigger_price = 97.0

    execution = asyncio.create_task(monitor.execute_stop_loss(stop))
    await callback_started.wait()
    record = monitor._inflight_stop_orders["default:AAPL"]
    assert record.stop is stop
    assert record.phase is StopExecutionPhase.POST_FILL_SETTLEMENT

    release_callback.set()
    assert await execution is True
    assert "default:AAPL" not in monitor._inflight_stop_orders


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantity", True),
        ("quantity", 1.5),
        ("avg_price", Decimal("NaN")),
        ("avg_price", Decimal("0")),
        ("symbol", "MSFT"),
    ],
)
def test_position_structure_matrix_fails_closed(field, value) -> None:
    runner = _protected_runner()
    position = runner.positions["AAPL"]
    setattr(position, field, value)

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        _assert_protection(runner)

    assert caught.value.reason_code == "position_structure_invalid"


def test_zero_quantity_position_does_not_hide_orphan_active_stop() -> None:
    runner = _protected_runner()
    runner.positions["AAPL"].quantity = 0

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        runner._assert_existing_position_protection(
            allow_runtime_tracked_states=True,
        )

    assert caught.value.reason_code == "orphan_active_stop"


@pytest.mark.parametrize(
    ("stop_type", "trailing_amount", "trailing_percent", "high_water_mark", "stop_price"),
    [
        (StopType.FIXED, 1.0, None, None, 98.0),
        (StopType.TRAILING, 0.0, None, 100.0, 98.0),
        (StopType.TRAILING, float("nan"), None, 100.0, 98.0),
        (StopType.TRAILING_PERCENT, None, 0.0, 100.0, 98.0),
        (StopType.TRAILING_PERCENT, None, 1.0, 100.0, 98.0),
        (StopType.TRAILING_PERCENT, None, float("nan"), 100.0, 98.0),
        (StopType.TRAILING_PERCENT, None, 0.05, 0.0, 98.0),
        (StopType.TRAILING_PERCENT, None, 0.05, 98.0, 98.0),
    ],
)
def test_trailing_structure_matrix_fails_closed(
    stop_type,
    trailing_amount,
    trailing_percent,
    high_water_mark,
    stop_price,
) -> None:
    runner = _protected_runner()
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    stop.stop_type = stop_type
    stop.trailing_amount = trailing_amount
    stop.trailing_percent = trailing_percent
    stop.high_water_mark = high_water_mark
    stop.stop_price = stop_price

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        _assert_protection(runner)

    assert caught.value.reason_code == "stop_structure_invalid"


@pytest.mark.parametrize(
    ("quantity", "entry_price", "stop_price", "current_price"),
    [
        (10, 100.0, 105.0, 108.0),
        (-10, 100.0, 95.0, 92.0),
    ],
)
def test_profit_protecting_fixed_stop_is_valid_relative_to_current_quote(
    quantity,
    entry_price,
    stop_price,
    current_price,
) -> None:
    runner = _protected_runner(quantity=quantity)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    stop.entry_price = entry_price
    stop.stop_price = stop_price
    runner.stop_loss_monitor.last_prices["AAPL"] = current_price
    previous_quote = runner.stop_loss_monitor.get_protective_quote_evidence("AAPL")
    runner.stop_loss_monitor._protective_quote_evidence["AAPL"] = _produce_protective_quote(
        runner.stop_loss_monitor,
        portfolio_id="default",
        symbol="AAPL",
        price=Decimal(str(current_price)),
        source_timestamp=previous_quote.source_timestamp,
        receipt_monotonic=previous_quote.receipt_monotonic,
        receipt_order=previous_quote.receipt_order,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=previous_quote.con_id,
        transport_generation=previous_quote.transport_generation,
        source_event_id="profit-protecting-event",
    )

    _assert_protection(runner)


@pytest.mark.parametrize(
    ("quantity", "stop_price", "current_price"),
    [
        (10, 98.0, 97.0),
        (-10, 102.0, 103.0),
    ],
)
def test_fresh_pending_stop_that_already_crosses_quote_fails_closed(
    quantity,
    stop_price,
    current_price,
) -> None:
    runner = _protected_runner(quantity=quantity)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    stop.stop_price = stop_price
    runner.stop_loss_monitor.last_prices["AAPL"] = current_price

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        _assert_protection(runner)

    assert caught.value.reason_code == "pending_stop_already_crossed"


@pytest.mark.asyncio
async def test_add_stop_loss_rejects_invalid_replacement_before_publish() -> None:
    monitor = StopLossMonitor(
        execute_reduction=_unused_reduction,
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    position = Position("AAPL", 10, 100.0)
    existing = await monitor.add_stop_loss("AAPL", position)

    with pytest.raises(ValidationError):
        await monitor.add_stop_loss(
            "AAPL",
            position,
            stop_type=StopType.TRAILING_PERCENT,
            trailing_percent=1.0,
        )

    assert monitor.active_stops["default:AAPL"] is existing
    assert existing.status is StopStatus.PENDING


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
async def test_persistent_ping_fast_path_requires_and_accepts_protection() -> None:
    runner = _protected_runner()
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        await runner.setup()

    runner.ib.ping.assert_awaited_once_with()
    cold_setup.assert_not_called()


@pytest.mark.asyncio
async def test_persistent_legacy_connection_fast_path_accepts_protection() -> None:
    runner = _protected_runner()
    runner._setup_complete = True
    runner.ib = SimpleNamespace(isConnected=lambda: True)

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        await runner.setup()

    cold_setup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("tracking", ["pending", "queued", "inflight"])
async def test_persistent_fast_path_accepts_exact_tracked_trigger(
    tracking,
) -> None:
    event_offset_seconds = -1.0 if tracking == "pending" else -60.0
    runner = _protected_runner(
        status=StopStatus.TRIGGERED,
        event_offset_seconds=event_offset_seconds,
        receipt_offset_seconds=-60.0,
    )
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    if tracking == "pending":
        runner.stop_loss_monitor._pending_stop_triggers["default:AAPL"] = _PendingStopTrigger(
            stop=stop,
            trigger_price=stop.trigger_price,
            event_time=stop.triggered_at,
            receipt_monotonic=999.0,
            receipt_order=1,
            drain_timeout_seconds=30.0,
            drain_deadline_monotonic=1029.0,
        )
    elif tracking == "queued":
        runner.stop_loss_monitor._queued_stop_orders["default:AAPL"] = _phase_record(
            stop,
            StopExecutionPhase.QUEUED,
        )
    else:
        runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
            stop,
            StopExecutionPhase.BROKER_WAIT,
        )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        await runner.setup()

    cold_setup.assert_not_called()


@pytest.mark.asyncio
async def test_persistent_fast_path_preserves_exact_broker_inflight_execution() -> None:
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    broker_started = asyncio.Event()
    release_broker = asyncio.Event()
    callback_complete = asyncio.Event()

    class _GatedExecutor:
        async def place_order_async(self, order):
            broker_started.set()
            await release_broker.wait()
            return _filled_terminal_outcome(order)

    async def filled_callback(_stop, _result) -> None:
        callback_complete.set()

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(_GatedExecutor()),
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
        position_closed_callback=filled_callback,
    )
    monitor._utcnow = lambda: now
    monitor._monotonic = lambda: 1000.0
    position = Position("AAPL", 10, 100.0)
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    assert await monitor.update_price(
        "AAPL",
        97.0,
        source_timestamp=now,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="test-generation-1",
        source_event_id="inflight-stop-event",
    )

    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {"AAPL": position}
    runner.stop_loss_monitor = monitor
    runner._protective_feed_status = {
        "AAPL": {
            "available": True,
            "live_grade": True,
            "source": "live_protective",
            "con_id": 265598,
            "transport_generation": "test-generation-1",
        }
    }
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))
    runner.cleanup = AsyncMock()
    runner.cancel_all_orders = AsyncMock()

    await monitor.start_monitoring()
    await broker_started.wait()
    assert monitor._inflight_stop_orders["default:AAPL"].stop is stop
    assert monitor._inflight_stop_orders["default:AAPL"].phase is StopExecutionPhase.BROKER_WAIT

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        await runner.setup()

    cold_setup.assert_not_called()
    runner.cleanup.assert_not_awaited()
    runner.cancel_all_orders.assert_not_awaited()

    release_broker.set()
    await callback_complete.wait()
    await asyncio.sleep(0)
    assert stop.status is StopStatus.EXECUTED
    assert "default:AAPL" not in monitor._inflight_stop_orders
    await monitor.stop_monitoring()


@pytest.mark.asyncio
async def test_pending_replacement_can_coexist_with_unresolved_old_broker_stop() -> None:
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    broker_started = asyncio.Event()
    release_broker = asyncio.Event()

    class _GatedExecutor:
        async def place_order_async(self, order):
            broker_started.set()
            await release_broker.wait()
            return _filled_terminal_outcome(order)

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(_GatedExecutor()),
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    monitor._utcnow = lambda: now
    monitor._monotonic = lambda: 1000.0
    position = Position("AAPL", 10, 100.0)
    old_stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    assert await monitor.update_price(
        "AAPL",
        97.0,
        source_timestamp=now,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="test-generation-1",
        source_event_id="old-stop-event",
    )

    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {"AAPL": position}
    runner.stop_loss_monitor = monitor
    runner._protective_feed_status = {
        "AAPL": {
            "available": True,
            "live_grade": True,
            "source": "live_protective",
            "con_id": 265598,
            "transport_generation": "test-generation-1",
        }
    }
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    await monitor.start_monitoring()
    await broker_started.wait()
    replacement = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    assert old_stop.status is StopStatus.CANCELLED
    assert replacement.status is StopStatus.PENDING
    assert monitor.active_stops["default:AAPL"] is replacement
    assert monitor._inflight_stop_orders["default:AAPL"].stop is old_stop
    assert await monitor.update_price(
        "AAPL",
        101.0,
        source_timestamp=now,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="test-generation-1",
        source_event_id="replacement-stop-event",
    )

    await runner.setup()

    release_broker.set()
    for _ in range(10):
        if "default:AAPL" not in monitor._inflight_stop_orders:
            break
        await asyncio.sleep(0)
    await monitor.stop_monitoring()


@pytest.mark.asyncio
async def test_cancelled_broker_stop_without_replacement_fails_closed() -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops.pop("default:AAPL")
    stop.status = StopStatus.CANCELLED
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        stop,
        StopExecutionPhase.BROKER_WAIT,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "broker_stop_state_invalid"


@pytest.mark.asyncio
async def test_persistent_fast_path_accepts_exact_terminal_cleanup_in_progress() -> None:
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    callback_complete = asyncio.Event()

    class _ImmediateExecutor:
        async def place_order_async(self, order):
            return _filled_terminal_outcome(order)

    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {"AAPL": Position("AAPL", 10, 100.0)}
    runner.portfolio = SimpleNamespace(update_fill=AsyncMock(return_value=None))
    runner.db = SimpleNamespace(
        record_trade=AsyncMock(return_value=None),
        update_position=AsyncMock(return_value=None),
    )
    runner.use_advanced_risk = False
    runner.advanced_risk = None
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))
    runner.cleanup = AsyncMock()
    runner.cancel_all_orders = AsyncMock()

    async def gated_terminal_cleanup(_stop, _result) -> None:
        callback_started.set()
        await release_callback.wait()
        callback_complete.set()

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(_ImmediateExecutor()),
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
        position_closed_callback=gated_terminal_cleanup,
    )
    monitor._utcnow = lambda: now
    monitor._monotonic = lambda: 1000.0
    runner.stop_loss_monitor = monitor
    runner._protective_feed_status = {
        "AAPL": {
            "available": True,
            "live_grade": True,
            "source": "live_protective",
            "con_id": 265598,
            "transport_generation": "test-generation-1",
        }
    }
    stop = await monitor.add_stop_loss(
        "AAPL",
        runner.positions["AAPL"],
        stop_percent=0.02,
    )
    assert await monitor.update_price(
        "AAPL",
        97.0,
        source_timestamp=now,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="test-generation-1",
        source_event_id="settlement-stop-event",
    )

    await monitor.start_monitoring()
    await callback_started.wait()
    assert stop.status is StopStatus.EXECUTED
    assert "default:AAPL" not in monitor.active_stops
    assert monitor._inflight_stop_orders["default:AAPL"].stop is stop
    assert (
        monitor._inflight_stop_orders["default:AAPL"].phase
        is StopExecutionPhase.POST_FILL_SETTLEMENT
    )

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        await runner.setup()

    cold_setup.assert_not_called()
    runner.cleanup.assert_not_awaited()
    runner.cancel_all_orders.assert_not_awaited()

    release_callback.set()
    await callback_complete.wait()
    for _ in range(10):
        if "default:AAPL" not in monitor._inflight_stop_orders:
            break
        await asyncio.sleep(0)
    # Runtime/DB mutation is owned by the gateway's producer receipt.  The
    # monitor compatibility callback may not perform a second mutation.
    assert runner.positions["AAPL"].quantity == 10
    runner.portfolio.update_fill.assert_not_awaited()
    runner.db.record_trade.assert_not_awaited()
    runner.db.update_position.assert_not_awaited()
    assert "default:AAPL" not in monitor._inflight_stop_orders
    await monitor.stop_monitoring()


@pytest.mark.asyncio
async def test_persistent_fast_path_accepts_complete_two_stop_execution_batch() -> None:
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    first_broker_call = asyncio.Event()
    release_first_call = asyncio.Event()
    both_callbacks_complete = asyncio.Event()
    callback_count = 0

    class _BatchExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def place_order_async(self, order):
            self.calls.append(order.symbol)
            if len(self.calls) == 1:
                first_broker_call.set()
                await release_first_call.wait()
            return _filled_terminal_outcome(order)

    executor = _BatchExecutor()

    async def filled_callback(_stop, _result) -> None:
        nonlocal callback_count
        callback_count += 1
        if callback_count == 2:
            both_callbacks_complete.set()

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(executor),
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
        position_closed_callback=filled_callback,
    )
    monitor._utcnow = lambda: now
    monitor._monotonic = lambda: 1000.0
    positions = {
        "AAPL": Position("AAPL", 10, 100.0),
        "MSFT": Position("MSFT", 7, 200.0),
    }
    stops = {}
    for symbol, position in positions.items():
        stops[symbol] = await monitor.add_stop_loss(
            symbol,
            position,
            stop_percent=0.02,
        )
        assert await monitor.update_price(
            symbol,
            0.97 * float(position.avg_price),
            source_timestamp=now,
            source=ProtectiveQuoteSource.LIVE_BROKER,
            con_id=265598 if symbol == "AAPL" else 272093,
            transport_generation="test-generation-1",
            source_event_id=f"parallel-stop-{symbol}",
        )

    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = positions
    runner.stop_loss_monitor = monitor
    runner._protective_feed_status = {
        symbol: {
            "available": True,
            "live_grade": True,
            "source": "live_protective",
            "con_id": 265598 if symbol == "AAPL" else 272093,
            "transport_generation": "test-generation-1",
        }
        for symbol in positions
    }
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))
    runner.cleanup = AsyncMock()
    runner.cancel_all_orders = AsyncMock()

    await monitor.start_monitoring()
    await first_broker_call.wait()
    assert monitor._inflight_stop_orders["default:AAPL"].stop is stops["AAPL"]
    assert monitor._inflight_stop_orders["default:AAPL"].phase is StopExecutionPhase.BROKER_WAIT
    assert monitor._queued_stop_orders["default:MSFT"].stop is stops["MSFT"]
    assert monitor._queued_stop_orders["default:MSFT"].phase is StopExecutionPhase.QUEUED

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        await runner.setup()

    cold_setup.assert_not_called()
    runner.cleanup.assert_not_awaited()
    runner.cancel_all_orders.assert_not_awaited()

    release_first_call.set()
    await both_callbacks_complete.wait()
    for _ in range(10):
        if not monitor._queued_stop_orders and not monitor._inflight_stop_orders:
            break
        await asyncio.sleep(0)
    assert executor.calls == ["AAPL", "MSFT"]
    assert all(stop.status is StopStatus.EXECUTED for stop in stops.values())
    assert monitor._queued_stop_orders == {}
    assert monitor._inflight_stop_orders == {}
    await monitor.stop_monitoring()


@pytest.mark.asyncio
async def test_inflight_tracking_clears_on_execution_exception() -> None:
    monitor = StopLossMonitor(
        execute_reduction=_unused_reduction,
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    position = Position("AAPL", 10, 100.0)
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    monitor.execute_stop_loss = AsyncMock(side_effect=RuntimeError("execution crashed"))

    with pytest.raises(RuntimeError, match="execution crashed"):
        await monitor._execute_tracked_stop(stop)

    assert monitor._inflight_stop_orders == {}


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_untracked_trigger() -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    assert caught.value.reason_code == "active_stop_trigger_untracked"
    cold_setup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [StopStatus.PENDING, StopStatus.TRIGGERED])
async def test_persistent_fast_path_rejects_miskeyed_active_stop(status) -> None:
    runner = _protected_runner(status=status)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    stop.symbol = "MSFT"
    if status is StopStatus.TRIGGERED:
        runner.stop_loss_monitor._queued_stop_orders["default:AAPL"] = _phase_record(
            stop,
            StopExecutionPhase.QUEUED,
        )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "active_stop_identity_mismatch"


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_cross_portfolio_active_stop() -> None:
    runner = _protected_runner()
    runner.stop_loss_monitor.active_stops["default:AAPL"].portfolio_id = "other"
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "active_stop_identity_mismatch"


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_miskeyed_post_fill_settlement() -> None:
    runner = _protected_runner()
    runner.stop_loss_monitor.active_stops.clear()
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        _executed_stop(10, symbol="MSFT"),
        StopExecutionPhase.POST_FILL_SETTLEMENT,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "post_fill_identity_mismatch"


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_duck_typed_pending_evidence() -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    runner.stop_loss_monitor._pending_stop_triggers["default:AAPL"] = SimpleNamespace(
        stop=stop,
        trigger_price=stop.trigger_price,
        event_time=stop.triggered_at,
        receipt_monotonic=999.0,
        receipt_order=1,
        drain_timeout_seconds=30.0,
        drain_deadline_monotonic=1029.0,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "pending_trigger_identity_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["non_crossing", "timestamp", "receipt_order"])
async def test_persistent_fast_path_rejects_tampered_pending_lineage(tamper) -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    trigger_price = stop.trigger_price
    event_time = stop.triggered_at
    receipt_order = 1
    if tamper == "non_crossing":
        trigger_price = 99.0
        stop.trigger_price = trigger_price
    elif tamper == "timestamp":
        event_time = event_time + timedelta(seconds=1)
    else:
        receipt_order = 2
    runner.stop_loss_monitor._pending_stop_triggers["default:AAPL"] = _PendingStopTrigger(
        stop=stop,
        trigger_price=trigger_price,
        event_time=event_time,
        receipt_monotonic=999.0,
        receipt_order=receipt_order,
        drain_timeout_seconds=30.0,
        drain_deadline_monotonic=1029.0,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    expected_reason = (
        "pending_trigger_crossing_invalid"
        if tamper == "non_crossing"
        else "pending_trigger_lineage_invalid"
    )
    assert caught.value.reason_code == expected_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_offset_seconds", "reason"),
    [
        (1.0, "pending_trigger_event_time_stale"),
        (-11.0, "pending_trigger_event_time_stale"),
    ],
)
async def test_persistent_fast_path_rejects_future_or_stale_pending_event(
    event_offset_seconds,
    reason,
) -> None:
    runner = _protected_runner(
        status=StopStatus.TRIGGERED,
        event_offset_seconds=event_offset_seconds,
    )
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    runner.stop_loss_monitor._pending_stop_triggers["default:AAPL"] = _PendingStopTrigger(
        stop=stop,
        trigger_price=stop.trigger_price,
        event_time=stop.triggered_at,
        receipt_monotonic=999.0,
        receipt_order=1,
        drain_timeout_seconds=30.0,
        drain_deadline_monotonic=1029.0,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "field", "value", "reason"),
    [
        (
            StopExecutionPhase.QUEUED,
            "trigger_price",
            float("nan"),
            "stop_structure_invalid",
        ),
        (
            StopExecutionPhase.BROKER_WAIT,
            "triggered_at",
            None,
            "stop_structure_invalid",
        ),
    ],
)
async def test_persistent_fast_path_rejects_invalid_latched_crossing(
    phase,
    field,
    value,
    reason,
) -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    setattr(stop, field, value)
    phase_records = (
        runner.stop_loss_monitor._queued_stop_orders
        if phase is StopExecutionPhase.QUEUED
        else runner.stop_loss_monitor._inflight_stop_orders
    )
    phase_records["default:AAPL"] = _phase_record(stop, phase)
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == reason


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["fresh_queued_expired_broker", "stale_pending_valid_broker"])
async def test_persistent_fast_path_rejects_conflicting_trigger_owners(
    conflict,
) -> None:
    event_offset_seconds = -11.0 if conflict == "stale_pending_valid_broker" else -1.0
    runner = _protected_runner(
        status=StopStatus.TRIGGERED,
        event_offset_seconds=event_offset_seconds,
    )
    monitor = runner.stop_loss_monitor
    stop = monitor.active_stops["default:AAPL"]
    if conflict == "fresh_queued_expired_broker":
        monitor._queued_stop_orders["default:AAPL"] = _phase_record(
            stop,
            StopExecutionPhase.QUEUED,
        )
        monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
            stop,
            StopExecutionPhase.BROKER_WAIT,
            started=900.0,
            timeout=30.0,
        )
    else:
        monitor._pending_stop_triggers["default:AAPL"] = _PendingStopTrigger(
            stop=stop,
            trigger_price=stop.trigger_price,
            event_time=stop.triggered_at,
            receipt_monotonic=999.0,
            receipt_order=1,
            drain_timeout_seconds=30.0,
            drain_deadline_monotonic=1029.0,
        )
        monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
            stop,
            StopExecutionPhase.BROKER_WAIT,
        )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "active_stop_trigger_ownership_conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expired", "expected_reason"),
    [
        (False, None),
        (True, "broker_stop_progress_expired"),
    ],
)
async def test_pending_replacement_validates_older_broker_owner(
    expired,
    expected_reason,
) -> None:
    runner = _protected_runner()
    old_runner = _protected_runner(status=StopStatus.TRIGGERED)
    old_stop = old_runner.stop_loss_monitor.active_stops["default:AAPL"]
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        old_stop,
        StopExecutionPhase.BROKER_WAIT,
        started=900.0 if expired else 999.0,
        timeout=30.0,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    if expected_reason is None:
        await runner.setup()
    else:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()
        assert caught.value.reason_code == expected_reason


@pytest.mark.asyncio
async def test_zero_positions_rejects_orphan_active_stop() -> None:
    runner = _protected_runner()
    runner.positions.clear()
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "orphan_active_stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "reason"),
    [
        (StopExecutionPhase.QUEUED, "orphan_queued_stop"),
        (StopExecutionPhase.BROKER_WAIT, "orphan_broker_stop"),
    ],
)
async def test_zero_positions_rejects_orphan_execution_phase(
    phase,
    reason,
) -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops.pop("default:AAPL")
    runner.positions.clear()
    phase_records = (
        runner.stop_loss_monitor._queued_stop_orders
        if phase is StopExecutionPhase.QUEUED
        else runner.stop_loss_monitor._inflight_stop_orders
    )
    phase_records["default:AAPL"] = _phase_record(stop, phase)
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == reason


@pytest.mark.asyncio
async def test_zero_positions_rejects_expired_post_fill_settlement() -> None:
    runner = _protected_runner()
    runner.positions.clear()
    runner.stop_loss_monitor.active_stops.clear()
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        _executed_stop(10),
        StopExecutionPhase.POST_FILL_SETTLEMENT,
        started=900.0,
        timeout=30.0,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "post_fill_progress_expired"


@pytest.mark.asyncio
async def test_zero_positions_accepts_unexpired_exact_post_fill_settlement() -> None:
    runner = _protected_runner()
    runner.positions.clear()
    runner.stop_loss_monitor.active_stops.clear()
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        _executed_stop(10),
        StopExecutionPhase.POST_FILL_SETTLEMENT,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    await runner.setup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "reason"),
    [
        ("pending", "pending_trigger_progress_expired"),
        ("queued", "queued_stop_progress_expired"),
        ("broker", "broker_stop_progress_expired"),
    ],
)
async def test_persistent_fast_path_rejects_expired_trigger_progress(
    phase,
    reason,
) -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    if phase == "pending":
        runner.stop_loss_monitor._pending_stop_triggers["default:AAPL"] = _PendingStopTrigger(
            stop=stop,
            trigger_price=stop.trigger_price,
            event_time=stop.triggered_at,
            receipt_monotonic=900.0,
            receipt_order=1,
            drain_timeout_seconds=30.0,
            drain_deadline_monotonic=930.0,
        )
    elif phase == "queued":
        runner.stop_loss_monitor._queued_stop_orders["default:AAPL"] = _phase_record(
            stop,
            StopExecutionPhase.QUEUED,
            started=900.0,
            timeout=30.0,
        )
    else:
        runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
            stop,
            StopExecutionPhase.BROKER_WAIT,
            started=900.0,
            timeout=30.0,
        )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == reason


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_expired_post_fill_settlement() -> None:
    runner = _protected_runner()
    runner.stop_loss_monitor.active_stops.clear()
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        _executed_stop(10),
        StopExecutionPhase.POST_FILL_SETTLEMENT,
        started=900.0,
        timeout=30.0,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.setup()

    assert caught.value.reason_code == "post_fill_progress_expired"


@pytest.mark.asyncio
async def test_run_cleans_up_after_expired_runtime_progress() -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    runner.stop_loss_monitor._queued_stop_orders["default:AAPL"] = _phase_record(
        stop,
        StopExecutionPhase.QUEUED,
        started=900.0,
        timeout=30.0,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))
    runner.cleanup = AsyncMock()

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        await runner.run([])

    assert caught.value.reason_code == "queued_stop_progress_expired"
    runner.cleanup.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_nonactive_inflight_identity() -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops.pop("default:AAPL")
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        stop,
        StopExecutionPhase.BROKER_WAIT,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    assert caught.value.reason_code == "active_stop_missing"
    cold_setup.assert_not_called()


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_untracked_executed_settlement() -> None:
    runner = _protected_runner()
    runner.stop_loss_monitor.active_stops.clear()
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    assert caught.value.reason_code == "active_stop_missing"
    cold_setup.assert_not_called()


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_settlement_quantity_mismatch() -> None:
    runner = _protected_runner()
    runner.stop_loss_monitor.active_stops.clear()
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        _executed_stop(9),
        StopExecutionPhase.POST_FILL_SETTLEMENT,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    assert caught.value.reason_code == "post_fill_settlement_quantity_mismatch"
    cold_setup.assert_not_called()


@pytest.mark.asyncio
async def test_executed_inflight_stop_does_not_mask_unsafe_active_replacement() -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        _executed_stop(10),
        StopExecutionPhase.POST_FILL_SETTLEMENT,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    assert caught.value.reason_code == "active_stop_trigger_untracked"
    cold_setup.assert_not_called()


@pytest.mark.asyncio
async def test_persistent_fast_path_rejects_replaced_queued_identity() -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    old_stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    runner.stop_loss_monitor._queued_stop_orders["default:AAPL"] = _phase_record(
        old_stop,
        StopExecutionPhase.QUEUED,
    )
    replacement = StopLossOrder(
        symbol="AAPL",
        position_qty=10,
        stop_price=98.0,
        entry_price=100.0,
        stop_type=StopType.FIXED,
        created_at=datetime.now(timezone.utc),
        status=StopStatus.TRIGGERED,
        portfolio_id="default",
    )
    replacement.trigger_price = 97.0
    replacement.triggered_at = datetime.now(timezone.utc)
    runner.stop_loss_monitor.active_stops["default:AAPL"] = replacement
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    assert caught.value.reason_code == "active_stop_trigger_untracked"
    cold_setup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        StopStatus.CANCELLED,
        StopStatus.FAILED,
        StopStatus.EXECUTED,
    ],
)
async def test_persistent_fast_path_rejects_nontrigger_terminal_status(
    status,
) -> None:
    runner = _protected_runner(status=status)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        stop,
        StopExecutionPhase.BROKER_WAIT,
    )
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    expected_reason = (
        "stop_structure_invalid" if status is StopStatus.EXECUTED else "broker_stop_state_invalid"
    )
    assert caught.value.reason_code == expected_reason
    cold_setup.assert_not_called()


def test_cold_protection_assertion_rejects_even_exact_tracked_trigger() -> None:
    runner = _protected_runner(status=StopStatus.TRIGGERED)
    stop = runner.stop_loss_monitor.active_stops["default:AAPL"]
    runner.stop_loss_monitor._inflight_stop_orders["default:AAPL"] = _phase_record(
        stop,
        StopExecutionPhase.BROKER_WAIT,
    )

    with pytest.raises(UnprotectedExistingPositionsError) as caught:
        runner._assert_existing_position_protection()

    assert caught.value.reason_code == "active_stop_not_pending"


@pytest.mark.asyncio
async def test_persistent_ping_missing_stop_coverage_raises_before_cold_setup(
    caplog,
) -> None:
    caplog.set_level("INFO")
    runner = _protected_runner()
    runner.stop_loss_monitor.active_stops.clear()
    runner._setup_complete = True
    runner.ib = SimpleNamespace(ping=AsyncMock(return_value=True))

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    assert caught.value.reason_code == "active_stop_missing"
    cold_setup.assert_not_called()
    assert "Persistent IBKR connection still active" not in caplog.text


@pytest.mark.asyncio
async def test_persistent_legacy_connection_stale_coverage_raises_before_cold_setup(
    caplog,
) -> None:
    caplog.set_level("INFO")
    runner = _protected_runner(event_offset_seconds=-11.0)
    runner._setup_complete = True
    runner.ib = SimpleNamespace(isConnected=lambda: True)

    with patch("robo_trader.runner_async.load_config") as cold_setup:
        with pytest.raises(UnprotectedExistingPositionsError) as caught:
            await runner.setup()

    assert caught.value.reason_code == "protective_price_stale"
    cold_setup.assert_not_called()
    assert "Persistent IBKR connection still active" not in caplog.text


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
    runner.cfg = SimpleNamespace(runtime_contract=TEST_RUNTIME_CONTRACT)
    runner.db = SimpleNamespace(
        get_account_info=AsyncMock(
            return_value={
                "cash_exact": Decimal("100000"),
                "realized_pnl_exact": Decimal("0"),
                "daily_pnl_exact": Decimal("0"),
                "daily_pnl_baseline_exact": Decimal("0"),
                "daily_pnl_date_exact": datetime.utcnow().date(),
                "source_settlement_id": None,
                "bootstrap_lineage_valid": True,
            }
        ),
        get_positions=AsyncMock(
            return_value=[
                {
                    "symbol": "AAPL",
                    "quantity": 10,
                    "avg_cost": 100.0,
                    "market_price_exact": Decimal("100"),
                    "bootstrap_lineage_valid": True,
                }
            ]
        ),
    )
    runner.portfolio = Portfolio(100_000)
    runner.latest_prices = {}
    runner.latest_price_sources = {}
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

    with pytest.raises(RuntimeError, match="monitor failed"):
        await runner.cleanup()

    runner.stop_loss_monitor.stop_monitoring.assert_awaited_once_with()
    runner.ib.disconnect.assert_awaited_once_with()
    runner.ib.stop.assert_awaited_once_with()
    runner.db.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cleanup_internal_cancellation_still_reaches_ibkr_and_database() -> None:
    runner = object.__new__(AsyncRunner)
    runner._setup_complete = True
    runner.health = SimpleNamespace(stop_monitoring=AsyncMock(side_effect=asyncio.CancelledError()))
    runner.subprocess_monitor_task = None
    runner.risk_monitor_task = None
    runner.cleanup_task = None
    runner.stop_loss_monitor = None
    runner.use_advanced_risk = False
    runner.advanced_risk = None
    runner.ib = SimpleNamespace(disconnect=AsyncMock(), stop=AsyncMock())
    runner.db = SimpleNamespace(close=AsyncMock())
    runner._owns_database = True
    runner.ws_client = None

    with pytest.raises(asyncio.CancelledError):
        await _cleanup_runner_owned(runner)

    runner.ib.disconnect.assert_awaited_once_with()
    runner.ib.stop.assert_awaited_once_with()
    runner.db.close.assert_awaited_once_with()
    assert runner.health is None
    assert runner._setup_complete is False


def test_continuous_loop_treats_unprotected_positions_as_nonretryable() -> None:
    source = inspect.getsource(run_continuous)
    handler = source.index("except UnprotectedExistingPositionsError as e:")
    generic = source.index("except Exception as e:", handler)

    assert handler < generic
    assert "fatal_safety_exit_written = True" in source[handler:generic]
    assert "raise SystemExit(6) from e" in source[handler:generic]
    assert "if not fatal_safety_exit_written:" in source


def _run_watchdog_policy(
    audit: Path, expected_pid: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(WATCHDOG_POLICY),
            str(audit),
            "unavailable" if expected_pid is None else str(expected_pid),
        ],
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


def test_watchdog_policy_blocks_missing_exit_audit(tmp_path: Path) -> None:
    """An audit-write failure must never become automatic restart permission."""

    result = _run_watchdog_policy(tmp_path / "runner_exit.json")

    assert result.returncode == 21
    assert result.stdout.strip() == "exit_audit_missing"


def test_watchdog_policy_keeps_stale_terminal_safety_exit_blocked(tmp_path: Path) -> None:
    """Only a verified manual startup may clear a terminal safety stop."""

    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": 1,
                "reason": "unprotected_existing_positions",
                "exit_code": 6,
            }
        )
    )

    result = _run_watchdog_policy(audit)

    assert result.returncode == 20
    assert result.stdout.strip() == "unprotected_existing_positions"


@pytest.mark.parametrize(
    ("reason", "exit_code"),
    [
        ("clean_shutdown", 0),
        ("keyboard_interrupt", 0),
        ("pre_flight_gateway_unreachable", 1),
        ("sigint", 0),
        ("sigterm", 0),
        ("unhandled_exception", 2),
    ],
)
def test_watchdog_policy_allows_known_fresh_exit_from_observed_runner(
    tmp_path: Path, reason: str, exit_code: int
) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "pid": 4242,
                "reason": reason,
                "exit_code": exit_code,
            }
        )
    )

    result = _run_watchdog_policy(audit, expected_pid=4242)

    assert result.returncode == 0
    assert result.stdout.strip() == "nonterminal_exit"


@pytest.mark.parametrize(
    ("reason", "exit_code"),
    [
        ("made_up_exit", 0),
        ("unprotected_existing_positions", 1),
        ("recovery_exhausted", 6),
        ("unhandled_exception", 0),
    ],
)
def test_watchdog_policy_blocks_unknown_or_wrong_exit_pair(
    tmp_path: Path, reason: str, exit_code: int
) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "pid": 4242,
                "reason": reason,
                "exit_code": exit_code,
            }
        )
    )

    result = _run_watchdog_policy(audit, expected_pid=4242)

    assert result.returncode == 21
    assert result.stdout.strip() == "exit_audit_unknown_pair"


def test_watchdog_policy_blocks_stale_nonterminal_audit(tmp_path: Path) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": time.time() - 301,
                "pid": 4242,
                "reason": "unhandled_exception",
                "exit_code": 2,
            }
        )
    )

    result = _run_watchdog_policy(audit, expected_pid=4242)

    assert result.returncode == 21
    assert result.stdout.strip() == "exit_audit_stale"


def test_watchdog_policy_blocks_nonterminal_audit_from_unobserved_runner(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "pid": 4242,
                "reason": "unhandled_exception",
                "exit_code": 2,
            }
        )
    )

    result = _run_watchdog_policy(audit)

    assert result.returncode == 21
    assert result.stdout.strip() == "runner_pid_unobserved"


def test_watchdog_policy_blocks_nonterminal_audit_from_different_runner(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "pid": 1111,
                "reason": "unhandled_exception",
                "exit_code": 2,
            }
        )
    )

    result = _run_watchdog_policy(audit, expected_pid=2222)

    assert result.returncode == 21
    assert result.stdout.strip() == "runner_pid_mismatch"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "timestamp": True,
            "pid": 4242,
            "reason": "unhandled_exception",
            "exit_code": 2,
        },
        {
            "timestamp": float("inf"),
            "pid": 4242,
            "reason": "unhandled_exception",
            "exit_code": 2,
        },
        {
            "timestamp": 1,
            "pid": True,
            "reason": "unhandled_exception",
            "exit_code": 2,
        },
        {
            "timestamp": 1,
            "reason": "unhandled_exception",
            "exit_code": 2,
        },
    ],
)
def test_watchdog_policy_requires_complete_finite_nonterminal_evidence(
    tmp_path: Path, payload: dict
) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(json.dumps(payload))

    result = _run_watchdog_policy(audit, expected_pid=4242)

    assert result.returncode == 21
    assert result.stdout.strip() == "exit_audit_invalid_schema"


def test_watchdog_policy_blocks_future_nonterminal_audit(tmp_path: Path) -> None:
    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": time.time() + 60,
                "pid": 4242,
                "reason": "unhandled_exception",
                "exit_code": 2,
            }
        )
    )

    result = _run_watchdog_policy(audit, expected_pid=4242)

    assert result.returncode == 21
    assert result.stdout.strip() == "exit_audit_from_future"


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


def test_watchdog_missing_audit_blocks_launcher_after_audit_write_failure(
    tmp_path: Path,
) -> None:
    """Exercise the real missing-audit policy through the shell boundary."""

    source = (ROOT / "scripts" / "watchdog.sh").read_text()
    restart_function = (
        "restart_trader() {" + source.split("restart_trader() {", 1)[1].split("# Main loop", 1)[0]
    )
    audit = tmp_path / "runner_exit.json"
    capture = tmp_path / "watchdog.log"
    launcher = tmp_path / "START_TRADER.sh"
    launcher.write_text(f"#!/bin/bash\necho LAUNCHER_RAN >> {capture!s}\n")
    launcher.chmod(0o700)

    harness = f"""
RUNNER_EXIT_AUDIT={audit!s}
PYTHON3_BIN={sys.executable!s}
RESTART_POLICY={WATCHDOG_POLICY!s}
PROJECT_DIR={tmp_path!s}
WATCHDOG_LOG={capture!s}
CAPTURE={capture!s}
LAST_TERMINAL_SAFETY_REASON=""
LAST_OBSERVED_RUNNER_PID=4242
RESTART_VERIFY_WAIT=0
is_runner_alive() {{ return 1; }}
watchdog_restart_allowed_for_policy_rc() {{ [ "$1" -eq 0 ]; }}
log() {{ printf '%s\\n' "$1" >> "$CAPTURE"; }}
notify_user() {{ :; }}
reset_failures() {{ :; }}
get_failure_count() {{ echo 0; }}
set_failure_count() {{ :; }}
{restart_function}
restart_trader
rc=$?
printf 'RETURN_CODE=%s\\n' "$rc" >> "$CAPTURE"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_output = capture.read_text()
    assert (
        "AUTOMATIC RESTART REQUEST DENIED: launcher not invoked "
        "(restart_rc=2, policy_rc=21, reason=exit_audit_missing)" in log_output
    )
    assert "RETURN_CODE=2" in log_output
    assert "LAUNCHER_RAN" not in log_output


def test_watchdog_audited_ordinary_crash_still_invokes_launcher(tmp_path: Path) -> None:
    """A trusted non-terminal exit remains automatically recoverable."""

    source = (ROOT / "scripts" / "watchdog.sh").read_text()
    restart_function = (
        "restart_trader() {" + source.split("restart_trader() {", 1)[1].split("# Main loop", 1)[0]
    )
    audit = tmp_path / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "pid": 4242,
                "reason": "unhandled_exception",
                "exit_code": 2,
            }
        )
    )
    capture = tmp_path / "watchdog.log"
    launcher = tmp_path / "START_TRADER.sh"
    launcher.write_text(f"#!/bin/bash\necho LAUNCHER_RAN >> {capture!s}\n")
    launcher.chmod(0o700)

    harness = f"""
RUNNER_EXIT_AUDIT={audit!s}
PYTHON3_BIN={sys.executable!s}
RESTART_POLICY={WATCHDOG_POLICY!s}
PROJECT_DIR={tmp_path!s}
WATCHDOG_LOG={capture!s}
FAILURE_STATE_FILE={tmp_path / ".watchdog_failures"!s}
CAPTURE={capture!s}
LAST_TERMINAL_SAFETY_REASON=""
LAST_OBSERVED_RUNNER_PID=4242
RESTART_VERIFY_WAIT=0
runner_checks=0
is_runner_alive() {{
    runner_checks=$((runner_checks + 1))
    [ "$runner_checks" -ge 2 ]
}}
pgrep() {{ printf '5555\\n'; }}
watchdog_restart_allowed_for_policy_rc() {{ [ "$1" -eq 0 ]; }}
log() {{ printf '%s\\n' "$1" >> "$CAPTURE"; }}
notify_user() {{ :; }}
reset_failures() {{ :; }}
get_failure_count() {{ echo 0; }}
set_failure_count() {{ :; }}
{restart_function}
restart_trader
rc=$?
printf 'RETURN_CODE=%s\\n' "$rc" >> "$CAPTURE"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_output = capture.read_text()
    assert "LAUNCHER_RAN" in log_output
    assert "Restart verified: new runner_async PID 5555 is alive" in log_output
    assert "RETURN_CODE=0" in log_output


def test_watchdog_clear_failure_then_terminal_write_failure_blocks_stale_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior session's audit cannot authorize restart for a later runner."""

    source = (ROOT / "scripts" / "watchdog.sh").read_text()
    restart_function = (
        "restart_trader() {" + source.split("restart_trader() {", 1)[1].split("# Main loop", 1)[0]
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    audit = data_dir / "runner_exit.json"
    audit.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "pid": 1111,
                "reason": "unhandled_exception",
                "exit_code": 2,
            }
        )
    )
    stale_payload = audit.read_text()
    monkeypatch.chdir(tmp_path)

    # Exercise both real best-effort helpers: setup cannot clear PID 1111's
    # record, then PID 2222's terminal path cannot replace it.
    with patch.object(Path, "unlink", side_effect=PermissionError("read-only")):
        _clear_exit_audit()
    with patch.object(Path, "write_text", side_effect=OSError("disk full")):
        _write_exit_audit(
            "unprotected_existing_positions",
            exit_code=6,
            extra={"portfolio_id": "default", "position_count": 1},
        )
    assert audit.read_text() == stale_payload

    capture = tmp_path / "watchdog.log"
    launcher = tmp_path / "START_TRADER.sh"
    launcher.write_text(f"#!/bin/bash\necho LAUNCHER_RAN >> {capture!s}\n")
    launcher.chmod(0o700)

    harness = f"""
RUNNER_EXIT_AUDIT={audit!s}
PYTHON3_BIN={sys.executable!s}
RESTART_POLICY={WATCHDOG_POLICY!s}
PROJECT_DIR={tmp_path!s}
WATCHDOG_LOG={capture!s}
CAPTURE={capture!s}
LAST_TERMINAL_SAFETY_REASON=""
LAST_OBSERVED_RUNNER_PID=2222
RESTART_VERIFY_WAIT=0
is_runner_alive() {{ return 1; }}
watchdog_restart_allowed_for_policy_rc() {{ [ "$1" -eq 0 ]; }}
log() {{ printf '%s\\n' "$1" >> "$CAPTURE"; }}
notify_user() {{ :; }}
reset_failures() {{ :; }}
get_failure_count() {{ echo 0; }}
set_failure_count() {{ :; }}
{restart_function}
restart_trader
rc=$?
printf 'RETURN_CODE=%s\\n' "$rc" >> "$CAPTURE"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_output = capture.read_text()
    assert (
        "AUTOMATIC RESTART REQUEST DENIED: launcher not invoked "
        "(restart_rc=2, policy_rc=21, reason=runner_pid_mismatch)" in log_output
    )
    assert "RETURN_CODE=2" in log_output
    assert "LAUNCHER_RAN" not in log_output


def test_watchdog_logs_terminal_restart_request_without_claiming_launcher_ran(
    tmp_path: Path,
) -> None:
    source = (ROOT / "scripts" / "watchdog.sh").read_text()
    restart_function = (
        "restart_trader() {" + source.split("restart_trader() {", 1)[1].split("# Main loop", 1)[0]
    )
    audit = tmp_path / "runner_exit.json"
    audit.write_text('{"reason":"unprotected_existing_positions","exit_code":6}')
    policy = tmp_path / "policy"
    policy.write_text("#!/bin/bash\necho unprotected_existing_positions\nexit 20\n")
    policy.chmod(0o700)
    capture = tmp_path / "watchdog.log"
    launcher = tmp_path / "START_TRADER.sh"
    launcher.write_text(f"#!/bin/bash\necho LAUNCHER_RAN >> {capture!s}\n")
    launcher.chmod(0o700)

    harness = f"""
RUNNER_EXIT_AUDIT={audit!s}
PYTHON3_BIN={policy!s}
RESTART_POLICY={policy!s}
PROJECT_DIR={tmp_path!s}
WATCHDOG_LOG={capture!s}
CAPTURE={capture!s}
LAST_TERMINAL_SAFETY_REASON=""
RESTART_VERIFY_WAIT=0
is_runner_alive() {{ return 1; }}
watchdog_restart_allowed_for_policy_rc() {{ return 1; }}
log() {{ printf '%s\\n' "$1" >> "$CAPTURE"; }}
notify_user() {{ :; }}
reset_failures() {{ :; }}
get_failure_count() {{ echo 0; }}
set_failure_count() {{ :; }}
{restart_function}
restart_trader
rc=$?
printf 'RETURN_CODE=%s\\n' "$rc" >> "$CAPTURE"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_output = capture.read_text()
    assert (
        "AUTOMATIC RESTART REQUEST DENIED: launcher not invoked "
        "(restart_rc=2, policy_rc=20, reason=unprotected_existing_positions)" in log_output
    )
    assert "RETURN_CODE=2" in log_output
    assert "LAUNCHER_RAN" not in log_output
