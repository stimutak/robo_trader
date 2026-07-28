"""Broker-event and monotonic receipt freshness for protective stops."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperOutcomeProvenance,
    LocalPaperTerminalOutcome,
)
from robo_trader.risk_manager import Position
from robo_trader.runner_async import AsyncRunner
from robo_trader.stop_loss_monitor import StopLossMonitor, StopStatus, StopType


def _monitor() -> StopLossMonitor:
    return StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=MagicMock(),
        portfolio_id="default",
    )


def _filled_terminal_outcome(order, fill_price: float) -> LocalPaperTerminalOutcome:
    quantity = Decimal(order.quantity)
    return LocalPaperTerminalOutcome(
        order_ref=order.order_ref,
        status=LocalPaperOrderStatus.FILLED,
        requested_quantity=quantity,
        filled_quantity=quantity,
        remaining_quantity=Decimal("0"),
        exact_fill_price=Decimal(str(fill_price)),
        observed_at=datetime.now(timezone.utc),
        provenance=LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR,
        terminal=True,
        message="exact local paper fill",
    )


def test_progress_deadline_defaults_cover_one_attempt_and_monitor_cadence() -> None:
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=MagicMock(),
        portfolio_id="default",
        order_timeout_seconds=5,
    )

    assert monitor.broker_attempt_timeout_seconds == 5
    assert monitor.pending_drain_timeout_seconds == 3
    assert monitor.max_execution_retries == 1
    assert monitor.queue_timeout_seconds >= monitor.broker_attempt_timeout_seconds
    assert monitor.settlement_timeout_seconds == 10


async def _trailing_monitor() -> tuple[StopLossMonitor, object]:
    monitor = _monitor()
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss(
        "AAPL",
        position,
        stop_type=StopType.TRAILING_PERCENT,
        trailing_percent=0.05,
    )
    return monitor, stop


@pytest.mark.asyncio
async def test_missing_stale_future_and_naive_events_do_not_mutate_or_trail() -> None:
    monitor, stop = await _trailing_monitor()
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=now)
    monitor._monotonic = MagicMock(return_value=100.0)
    initial_stop = stop.stop_price

    rejected = [
        None,
        now - timedelta(seconds=monitor.max_price_age_seconds + 0.001),
        now + timedelta(microseconds=1),
        now.replace(tzinfo=None),
    ]
    for event_time in rejected:
        accepted = await monitor.update_price(
            "AAPL",
            90.0,
            source_timestamp=event_time,
        )
        assert accepted is False

    assert monitor.last_prices == {}
    assert monitor.price_event_times == {}
    assert monitor._pending_stop_triggers == {}
    assert stop.status == StopStatus.PENDING
    assert stop.stop_price == initial_stop
    assert monitor.metrics.trailing_adjustments_today == 0


@pytest.mark.asyncio
async def test_exact_max_age_is_accepted_then_stale_by_monotonic_receipt() -> None:
    monitor, stop = await _trailing_monitor()
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=now)
    monitor._monotonic = MagicMock(return_value=100.0)
    event_time = now - timedelta(seconds=monitor.max_price_age_seconds)

    accepted = await monitor.update_price(
        "AAPL",
        90.0,
        source_timestamp=event_time,
    )
    assert accepted is True

    # Exactly at the boundary is fresh.
    monitor._monotonic = MagicMock(return_value=100.0 + monitor.max_price_age_seconds)
    triggered = await monitor.check_stops()
    assert triggered == [stop]


@pytest.mark.asyncio
async def test_latched_crossing_survives_wall_clock_rollback() -> None:
    monitor = _monitor()
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)
    assert await monitor.update_price(
        "AAPL",
        90.0,
        source_timestamp=event_time,
    )

    # Once validated at ingestion, a wall-clock rollback cannot erase the
    # crossing before the monitor loop drains it.
    monitor._utcnow = MagicMock(return_value=event_time - timedelta(seconds=5))
    monitor._monotonic = MagicMock(return_value=101.0)
    assert await monitor.check_stops() == [stop]
    assert stop.triggered_at == event_time
    assert stop.trigger_price == 90.0
    monitor._execute_reduction.assert_not_awaited()


@pytest.mark.asyncio
async def test_out_of_order_event_does_not_replace_fresh_price_or_trail() -> None:
    monitor, stop = await _trailing_monitor()
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=now)
    monitor._monotonic = MagicMock(side_effect=[100.0, 101.0])
    first = now - timedelta(seconds=1)
    assert await monitor.update_price("AAPL", 110.0, source_timestamp=first)
    adjusted_stop = stop.stop_price

    accepted = await monitor.update_price(
        "AAPL",
        100.0,
        source_timestamp=first - timedelta(microseconds=1),
    )
    assert accepted is False
    assert monitor.last_prices["AAPL"] == 110.0
    assert stop.stop_price == adjusted_stop
    assert stop.status == StopStatus.PENDING
    assert monitor._pending_stop_triggers == {}


@pytest.mark.asyncio
async def test_same_timestamp_later_quote_uses_receipt_order_and_triggers_stop() -> None:
    monitor = _monitor()
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    # Adjacent callbacks can share a monotonic-clock tick. Receipt sequence
    # must still provide a strict, deterministic order.
    monitor._monotonic = MagicMock(return_value=100.0)

    assert await monitor.update_price("AAPL", 101.0, source_timestamp=event_time)
    first_order = monitor.price_receipt_orders["AAPL"]
    assert await monitor.update_price("AAPL", 97.0, source_timestamp=event_time)
    assert await monitor.update_price("AAPL", 101.0, source_timestamp=event_time)

    assert monitor.last_prices["AAPL"] == 101.0
    assert monitor.price_receipt_orders["AAPL"] > first_order
    assert stop.status == StopStatus.TRIGGERED
    assert stop.trigger_price == 97.0
    monitor._execute_reduction.assert_not_awaited()
    assert await monitor.check_stops() == [stop]
    assert await monitor.check_stops() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "quantity",
        "stop_type",
        "trailing_percent",
        "prices",
        "trigger_price",
        "newer_recovery",
    ),
    [
        (10, StopType.FIXED, None, (101.0, 97.0, 101.0), 97.0, False),
        (-10, StopType.FIXED, None, (99.0, 103.0, 99.0), 103.0, True),
        (
            10,
            StopType.TRAILING_PERCENT,
            0.05,
            (110.0, 104.0, 110.0),
            104.0,
            True,
        ),
        (
            -10,
            StopType.TRAILING_PERCENT,
            0.05,
            (90.0, 95.0, 90.0),
            95.0,
            False,
        ),
    ],
)
async def test_burst_crossing_is_latched_for_fixed_and_trailing_long_and_short(
    quantity,
    stop_type,
    trailing_percent,
    prices,
    trigger_price,
    newer_recovery,
) -> None:
    monitor = _monitor()
    monitor._execute_reduction.side_effect = lambda _stop, order: _filled_terminal_outcome(
        order, trigger_price
    )
    position = Position(symbol="AAPL", quantity=quantity, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss(
        "AAPL",
        position,
        stop_percent=0.02,
        stop_type=stop_type,
        trailing_percent=trailing_percent,
    )
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time + timedelta(seconds=1))
    monitor._monotonic = MagicMock(return_value=100.0)

    assert await monitor.update_price("AAPL", prices[0], source_timestamp=event_time)
    assert await monitor.update_price("AAPL", prices[1], source_timestamp=event_time)
    evidence = monitor._pending_stop_triggers["default:AAPL"]
    recovery_time = event_time + (timedelta(microseconds=1) if newer_recovery else timedelta(0))
    assert await monitor.update_price(
        "AAPL",
        prices[2],
        source_timestamp=recovery_time,
    )

    assert monitor.last_prices["AAPL"] == prices[2]
    assert monitor._pending_stop_triggers["default:AAPL"] is evidence
    assert evidence.trigger_price == trigger_price
    assert stop.status == StopStatus.TRIGGERED
    assert stop.triggered_at == event_time
    assert stop.trigger_price == trigger_price
    assert monitor.metrics.triggered_today == 1
    monitor._execute_reduction.assert_not_awaited()

    assert await monitor.check_stops() == [stop]
    assert await monitor.check_stops() == []
    assert await monitor.execute_stop_loss(stop)
    monitor._execute_reduction.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_quote_callbacks_preserve_crossing_order() -> None:
    monitor, stop = await _trailing_monitor()
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)

    # Hold the lock so all three callback tasks become waiters. asyncio.Lock
    # admits waiters FIFO, making the intended callback arrival order explicit.
    await monitor._price_update_lock.acquire()
    first = asyncio.create_task(monitor.update_price("AAPL", 110.0, source_timestamp=event_time))
    await asyncio.sleep(0)
    crossing = asyncio.create_task(monitor.update_price("AAPL", 104.0, source_timestamp=event_time))
    await asyncio.sleep(0)
    recovery = asyncio.create_task(monitor.update_price("AAPL", 110.0, source_timestamp=event_time))
    await asyncio.sleep(0)
    monitor._price_update_lock.release()

    assert await asyncio.gather(first, crossing, recovery) == [True, True, True]
    assert monitor.price_receipt_orders["AAPL"] == 3
    assert monitor.last_prices["AAPL"] == 110.0
    assert stop.status == StopStatus.TRIGGERED
    assert stop.trigger_price == 104.0
    assert await monitor.check_stops() == [stop]
    assert await monitor.check_stops() == []
    monitor._execute_reduction.assert_not_awaited()


@pytest.mark.asyncio
async def test_replacing_or_cancelling_stop_discards_latched_object() -> None:
    monitor = _monitor()
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))

    old_stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    assert await monitor.update_price("AAPL", 97.0, source_timestamp=event_time)
    assert monitor._pending_stop_triggers["default:AAPL"].stop is old_stop

    new_stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.10)
    assert old_stop.status == StopStatus.CANCELLED
    assert monitor._pending_stop_triggers == {}
    assert await monitor.check_stops() == []

    assert await monitor.update_price("AAPL", 89.0, source_timestamp=event_time)
    assert monitor._pending_stop_triggers["default:AAPL"].stop is new_stop
    assert monitor.cancel_stop("AAPL")
    assert new_stop.status == StopStatus.CANCELLED
    assert monitor._pending_stop_triggers == {}
    assert await monitor.check_stops() == []
    monitor._execute_reduction.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_added_after_latest_quote_uses_fresh_fallback_once() -> None:
    monitor = _monitor()
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)
    assert await monitor.update_price("AAPL", 97.0, source_timestamp=event_time)

    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)

    assert await monitor.check_stops() == [stop]
    assert stop.triggered_at == event_time
    assert stop.trigger_price == 97.0
    assert await monitor.check_stops() == []


@pytest.mark.asyncio
async def test_same_timestamp_duplicate_gets_strictly_later_receipt_order() -> None:
    monitor = _monitor()
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)

    assert await monitor.update_price("AAPL", 101.0, source_timestamp=event_time)
    first_order = monitor.price_receipt_orders["AAPL"]
    assert await monitor.update_price("AAPL", 101.0, source_timestamp=event_time)
    assert monitor.last_prices["AAPL"] == 101.0
    assert monitor.price_receipt_orders["AAPL"] > first_order


@pytest.mark.asyncio
async def test_same_timestamp_revisited_value_is_a_legitimate_later_quote() -> None:
    monitor = _monitor()
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)

    orders = []
    for price in (101.0, 97.0, 101.0):
        assert await monitor.update_price("AAPL", price, source_timestamp=event_time)
        orders.append(monitor.price_receipt_orders["AAPL"])

    assert orders[0] < orders[1] < orders[2]
    assert monitor.last_prices["AAPL"] == 101.0


@pytest.mark.asyncio
async def test_same_timestamp_duplicate_cannot_refresh_stale_broker_event() -> None:
    monitor = _monitor()
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)

    assert await monitor.update_price("AAPL", 101.0, source_timestamp=event_time)
    receipt = monitor.price_receipt_monotonic["AAPL"]
    order = monitor.price_receipt_orders["AAPL"]

    monitor._utcnow = MagicMock(
        return_value=event_time + timedelta(seconds=monitor.max_price_age_seconds + 0.001)
    )
    monitor._monotonic = MagicMock(return_value=105.0)
    assert not await monitor.update_price("AAPL", 101.0, source_timestamp=event_time)
    assert monitor.price_receipt_monotonic["AAPL"] == receipt
    assert monitor.price_receipt_orders["AAPL"] == order


@pytest.mark.asyncio
async def test_rejected_older_stale_and_future_quotes_preserve_equal_time_winner() -> None:
    monitor = _monitor()
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    event_time = now - timedelta(seconds=1)
    monitor._utcnow = MagicMock(return_value=now)
    monitor._monotonic = MagicMock(return_value=100.0)

    assert await monitor.update_price("AAPL", 101.0, source_timestamp=event_time)
    assert await monitor.update_price("AAPL", 97.0, source_timestamp=event_time)
    receipt = monitor.price_receipt_monotonic["AAPL"]
    order = monitor.price_receipt_orders["AAPL"]

    rejected = (
        event_time - timedelta(microseconds=1),
        now - timedelta(seconds=monitor.max_price_age_seconds + 0.001),
        now + timedelta(microseconds=1),
    )
    for rejected_time in rejected:
        assert not await monitor.update_price(
            "AAPL",
            120.0,
            source_timestamp=rejected_time,
        )

    assert monitor.last_prices["AAPL"] == 97.0
    assert monitor.price_event_times["AAPL"] == event_time
    assert monitor.price_receipt_monotonic["AAPL"] == receipt
    assert monitor.price_receipt_orders["AAPL"] == order


@pytest.mark.asyncio
async def test_latched_crossing_survives_delayed_check() -> None:
    monitor = _monitor()
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)
    assert await monitor.update_price(
        "AAPL",
        90.0,
        source_timestamp=event_time,
    )

    monitor._utcnow = MagicMock(return_value=event_time + timedelta(seconds=11))
    monitor._monotonic = MagicMock(return_value=111.0)
    assert await monitor.check_stops() == [stop]
    assert stop.triggered_at == event_time


@pytest.mark.asyncio
async def test_stale_historical_close_is_not_rewarmed_as_protection(caplog) -> None:
    monitor = _monitor()
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=now)

    runner = AsyncRunner.__new__(AsyncRunner)
    runner.stop_loss_monitor = monitor
    runner.latest_prices = {"AAPL": 90.0}
    runner.latest_price_times = {"AAPL": now - timedelta(seconds=11)}
    runner.latest_price_sources = {"AAPL": "historical_bar"}

    await runner._rewarm_stop_loss_prices_after_recovery()

    assert monitor.last_prices == {}
    assert "event=stop_loss_prices_rewarmed count=0 skipped=1" in caplog.text


@pytest.mark.asyncio
async def test_fresh_historical_close_is_never_rewarmed_as_live_protection(caplog) -> None:
    monitor = _monitor()
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=now)

    runner = AsyncRunner.__new__(AsyncRunner)
    runner.stop_loss_monitor = monitor
    runner.latest_prices = {"AAPL": 90.0}
    runner.latest_price_times = {"AAPL": now - timedelta(seconds=1)}
    runner.latest_price_sources = {"AAPL": "historical_bar"}

    await runner._rewarm_stop_loss_prices_after_recovery()

    assert monitor.last_prices == {}
    assert "event=stop_loss_prices_rewarmed count=0 skipped=1" in caplog.text


@pytest.mark.asyncio
async def test_cached_live_label_cannot_manufacture_post_reconnect_quote_lineage() -> None:
    monitor = _monitor()
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    monitor._utcnow = MagicMock(return_value=now)

    runner = AsyncRunner.__new__(AsyncRunner)
    runner.stop_loss_monitor = monitor
    runner.latest_prices = {"AAPL": 99.0}
    runner.latest_price_times = {"AAPL": now - timedelta(seconds=1)}
    runner.latest_price_sources = {"AAPL": "live_protective"}

    assert await runner._rewarm_stop_loss_prices_after_recovery() is False

    assert monitor.last_prices == {}
