"""PR2B3 regression tests for emergency entry freezes.

Generic emergency handling must preserve protective stops and monitoring while
preventing any new risk-increasing order from crossing final admission.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from robo_trader.execution import Order
from robo_trader.runner_async import AsyncRunner


def _minimal_runner() -> AsyncRunner:
    """Build only the runner state exercised by emergency admission tests."""

    runner = object.__new__(AsyncRunner)
    runner._pending_orders = set()
    runner._pending_orders_lock = asyncio.Lock()
    runner._order_admission_lock = asyncio.Lock()
    runner._emergency_entry_freeze_reason = None
    runner._kill_switch_log_last = {}
    runner._kill_switch_log_throttle_seconds = 60.0
    runner.risk = None
    runner.advanced_risk = None
    runner.running = True
    runner.stop_loss_monitor = None
    return runner


@pytest.mark.asyncio
async def test_emergency_freeze_preserves_protective_stops_and_monitoring():
    runner = _minimal_runner()
    active_stops = {"AAPL": object(), "TSLA": object()}
    cancel_all_stops = Mock(side_effect=AssertionError("protective stops must survive"))
    runner.stop_loss_monitor = SimpleNamespace(
        active_stops=active_stops,
        monitoring_active=True,
        cancel_all_stops=cancel_all_stops,
    )
    runner._pending_orders.update({"MSFT", "NVDA"})

    cleared = await runner.cancel_all_orders("stop settlement is uncertain")

    assert cleared == 2
    assert runner._pending_orders == set()
    assert runner._emergency_entry_freeze_reason == "stop settlement is uncertain"
    assert runner.stop_loss_monitor.active_stops is active_stops
    assert runner.stop_loss_monitor.monitoring_active is True
    cancel_all_stops.assert_not_called()
    assert runner._trading_blocked() == (
        True,
        "Emergency entry freeze active: stop settlement is uncertain",
    )


@pytest.mark.asyncio
async def test_emergency_callback_keeps_runner_alive_and_triggers_entry_kill_switch():
    runner = _minimal_runner()
    trigger = Mock()
    runner.advanced_risk = SimpleNamespace(kill_switch=SimpleNamespace(trigger=trigger))
    runner.stop_loss_monitor = SimpleNamespace(
        active_stops={"AAPL": object()},
        monitoring_active=True,
    )

    await runner._enter_emergency_entry_freeze("protective exit failed")

    assert runner.running is True
    assert runner.stop_loss_monitor.monitoring_active is True
    assert len(runner.stop_loss_monitor.active_stops) == 1
    trigger.assert_called_once_with("protective exit failed")
    assert runner._emergency_entry_freeze_reason == "protective exit failed"


@pytest.mark.asyncio
async def test_entry_waiting_at_final_admission_is_rejected_after_freeze():
    runner = _minimal_runner()
    runner.circuit_breaker = SimpleNamespace(can_proceed=AsyncMock(return_value=True))
    runner.rate_limiter = SimpleNamespace(acquire=AsyncMock())
    runner._symbol_cycle_abort_event = asyncio.Event()

    # Hold final admission so the order passes the first gate and waits. The
    # emergency is then latched before the task can cross the point of no return.
    await runner._order_admission_lock.acquire()
    order_task = asyncio.create_task(
        runner._place_order_with_circuit_breaker(
            Order(
                symbol="NVDA",
                quantity=1,
                side="BUY",
                price=100.0,
                intent_source="baseline_sma",
            )
        )
    )
    await asyncio.sleep(0)
    runner.rate_limiter.acquire.assert_awaited_once()
    runner._emergency_entry_freeze_reason = "late settlement failure"
    runner._order_admission_lock.release()

    result = await order_task

    assert result.ok is False
    assert result.message == (
        "Trading blocked: Emergency entry freeze active: late settlement failure"
    )
