"""Broker-event and monotonic receipt freshness for protective stops."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.risk_manager import Position
from robo_trader.runner_async import AsyncRunner
from robo_trader.stop_loss_monitor import StopLossMonitor, StopType


def _monitor() -> StopLossMonitor:
    return StopLossMonitor(
        executor=AsyncMock(),
        risk_manager=MagicMock(),
        portfolio_id="default",
    )


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
            120.0,
            source_timestamp=event_time,
        )
        assert accepted is False

    assert monitor.last_prices == {}
    assert monitor.price_event_times == {}
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
async def test_wall_clock_rollback_cannot_make_price_trigger_stop() -> None:
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

    # A wall-clock rollback produces a negative event age. Even though receipt
    # age is short, fail closed.
    monitor._utcnow = MagicMock(return_value=event_time - timedelta(seconds=5))
    monitor._monotonic = MagicMock(return_value=101.0)
    assert await monitor.check_stops() == []
    assert stop.triggered_at is None
    monitor.executor.place_order_async.assert_not_awaited()


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
        120.0,
        source_timestamp=first - timedelta(microseconds=1),
    )
    assert accepted is False
    assert monitor.last_prices["AAPL"] == 110.0
    assert stop.stop_price == adjusted_stop


@pytest.mark.asyncio
async def test_stale_after_receipt_does_not_trigger() -> None:
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
    assert await monitor.check_stops() == []
    assert stop.triggered_at is None


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

    await runner._rewarm_stop_loss_prices_after_recovery()

    assert monitor.last_prices == {}
    assert "event=stop_loss_prices_rewarmed count=0 skipped=1" in caplog.text
