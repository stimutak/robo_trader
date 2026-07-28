"""Tests for the centralized kill-switch log throttle.

Per M1 (2026-05-22): four per-path kill-switch checks each emitted ERROR-level
logs per cycle per symbol per side. Production showed 5000+ "KILL SWITCH ACTIVE"
lines/hour when the switch was triggered and entry signals kept resolving.

The fix:
- Per-path checks are removed; the central _place_order_with_circuit_breaker
  guard via _trading_blocked() is authoritative.
- The central log is throttled per (symbol, action) to avoid spam.
- First occurrence always logs so operators see the block; subsequent identical
  messages within the throttle window are suppressed.
- Order is still BLOCKED on every call regardless of whether the log fires.
"""

from unittest.mock import MagicMock, patch

import pytest

from robo_trader.execution import Order
from robo_trader.runner_async import AsyncRunner


def _make_runner_with_triggered_kill_switch():
    """Build an AsyncRunner with a triggered kill switch and stub dependencies.
    Skips heavy __init__ side effects."""
    runner = AsyncRunner.__new__(AsyncRunner)

    # Triggered kill switch on advanced_risk
    kill_switch = MagicMock()
    kill_switch.triggered = True
    kill_switch.trigger_reason = "Daily loss limit exceeded"
    runner.advanced_risk = MagicMock()
    runner.advanced_risk.kill_switch = kill_switch

    # Emergency-shutdown branch not active
    runner.risk = MagicMock()
    runner.risk.emergency_shutdown_triggered = False

    # Throttle state (normally initialized in __init__)
    runner._kill_switch_log_last = {}
    runner._kill_switch_log_throttle_seconds = 60.0

    # Stub circuit breaker + rate limiter so they're not reached (blocked first).
    runner.circuit_breaker = MagicMock()
    runner.rate_limiter = MagicMock()
    runner.executor = MagicMock()
    runner.monitor = MagicMock()
    runner._baseline_entry_handle = object()
    runner._baseline_entry_intent = object()
    return runner


async def _place(runner: AsyncRunner, order: Order):
    return await runner._place_order_with_circuit_breaker(
        order,
        _entry_intent=(runner._baseline_entry_intent if order.side == "BUY" else None),
    )


@pytest.mark.asyncio
async def test_kill_switch_log_fires_once_then_throttles():
    """10 rapid blocked calls for the same (symbol, action) → log fires exactly once."""
    runner = _make_runner_with_triggered_kill_switch()
    order = Order(symbol="AAPL", quantity=10, side="BUY", price=150.0)

    # Freeze monotonic time so all 10 calls fall in the same throttle window.
    with patch("robo_trader.runner_async.time.monotonic", return_value=1000.0):
        with patch("robo_trader.runner_async.logger") as mock_logger:
            for _ in range(10):
                result = await _place(runner, order)
                # Order MUST be blocked on every call (functionality preserved)
                assert result.ok is False
                assert "Trading blocked" in result.message

    # The gate log must have fired exactly ONCE
    gate_log_calls = [
        c
        for c in mock_logger.error.call_args_list
        if "Order blocked by trading_blocked gate" in c.args[0]
    ]
    assert len(gate_log_calls) == 1, (
        f"Expected exactly 1 gate log within throttle window, got {len(gate_log_calls)}: "
        f"{gate_log_calls}"
    )


@pytest.mark.asyncio
async def test_kill_switch_log_fires_again_after_throttle_window():
    """After 61s elapses, the log should fire again for the same (symbol, action)."""
    runner = _make_runner_with_triggered_kill_switch()
    order = Order(symbol="AAPL", quantity=10, side="BUY", price=150.0)

    # First call at t=1000 fires the log
    with patch("robo_trader.runner_async.time.monotonic", return_value=1000.0):
        with patch("robo_trader.runner_async.logger") as mock_logger_1:
            result1 = await _place(runner, order)
    assert result1.ok is False
    first_log_count = sum(
        1
        for c in mock_logger_1.error.call_args_list
        if "Order blocked by trading_blocked gate" in c.args[0]
    )
    assert first_log_count == 1

    # 10 more calls within the throttle window do NOT fire
    with patch("robo_trader.runner_async.time.monotonic", return_value=1030.0):
        with patch("robo_trader.runner_async.logger") as mock_logger_2:
            for _ in range(10):
                result = await _place(runner, order)
                assert result.ok is False
    suppressed_log_count = sum(
        1
        for c in mock_logger_2.error.call_args_list
        if "Order blocked by trading_blocked gate" in c.args[0]
    )
    assert suppressed_log_count == 0

    # 61s after first log → throttle window elapsed → must log again
    with patch("robo_trader.runner_async.time.monotonic", return_value=1061.0):
        with patch("robo_trader.runner_async.logger") as mock_logger_3:
            result3 = await _place(runner, order)
    assert result3.ok is False
    third_log_count = sum(
        1
        for c in mock_logger_3.error.call_args_list
        if "Order blocked by trading_blocked gate" in c.args[0]
    )
    assert (
        third_log_count == 1
    ), f"Expected log to fire again after 61s, got {third_log_count} calls"


@pytest.mark.asyncio
async def test_short_containment_precedes_kill_switch_log_throttle():
    """A disabled short never enters the kill-switch logging path."""
    runner = _make_runner_with_triggered_kill_switch()
    sell_order = Order(symbol="AAPL", quantity=10, side="SELL_SHORT", price=150.0)
    buy_order = Order(symbol="AAPL", quantity=10, side="BUY", price=150.0)

    with patch("robo_trader.runner_async.time.monotonic", return_value=1000.0):
        with patch("robo_trader.runner_async.logger") as mock_logger:
            sell_res = await _place(runner, sell_order)
            buy_res = await _place(runner, buy_order)

    # Both blocked
    assert sell_res.ok is False
    assert "blocks new short exposure" in sell_res.message
    assert buy_res.ok is False

    gate_log_calls = [
        c
        for c in mock_logger.error.call_args_list
        if "Order blocked by trading_blocked gate" in c.args[0]
    ]
    assert len(gate_log_calls) == 1


@pytest.mark.asyncio
async def test_kill_switch_log_keyed_by_symbol_distinct_symbols_both_log():
    """Distinct entry symbols receive independent throttled logs."""
    runner = _make_runner_with_triggered_kill_switch()
    aapl_order = Order(symbol="AAPL", quantity=10, side="BUY", price=150.0)
    tsla_order = Order(symbol="TSLA", quantity=10, side="BUY", price=200.0)

    with patch("robo_trader.runner_async.time.monotonic", return_value=1000.0):
        with patch("robo_trader.runner_async.logger") as mock_logger:
            await _place(runner, aapl_order)
            await _place(runner, tsla_order)

    gate_log_calls = [
        c
        for c in mock_logger.error.call_args_list
        if "Order blocked by trading_blocked gate" in c.args[0]
    ]
    assert len(gate_log_calls) == 2


@pytest.mark.asyncio
async def test_order_blocked_on_every_call_even_when_log_suppressed():
    """Functionality preserved: every call returns ok=False even if log suppressed."""
    runner = _make_runner_with_triggered_kill_switch()
    order = Order(symbol="AAPL", quantity=10, side="BUY", price=150.0)

    with patch("robo_trader.runner_async.time.monotonic", return_value=1000.0):
        for i in range(50):
            result = await _place(runner, order)
            assert result.ok is False, f"Call {i} should be blocked, got {result}"
            assert "Trading blocked" in result.message
            assert result.fill_price is None

    # Executor must never have been reached
    runner.executor.place_order.assert_not_called()
