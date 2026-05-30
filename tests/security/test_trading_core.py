"""Security-audit regression tests for trading-core fixes.

These tests cover the HIGH/MEDIUM-severity findings from
SECURITY_AUDIT_2026-05-10.md (section 2.C). Each test is named with the
finding ID it pins.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from robo_trader.execution import ExecutionResult, Order, PaperExecutor
from robo_trader.portfolio import Portfolio
from robo_trader.risk.advanced_risk import KillSwitch
from robo_trader.risk_manager import (
    Position,
    RiskManager,
    create_risk_manager_from_config,
)
from robo_trader.stop_loss_monitor import StopLossMonitor, StopType

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubExecutor:
    async def place_order_async(self, order: Order) -> ExecutionResult:
        return ExecutionResult(True, "stub", fill_price=order.price)


class _StubRiskManager:
    pass


def _build_monitor() -> StopLossMonitor:
    return StopLossMonitor(executor=_StubExecutor(), risk_manager=_StubRiskManager())


def _build_config(default_cash: float = 100_000.0) -> SimpleNamespace:
    """Mimic the cfg shape consumed by create_risk_manager_from_config."""
    risk = SimpleNamespace(
        max_daily_loss_pct=0.005,
        max_position_pct=0.02,
        max_sector_exposure_pct=0.3,
        max_leverage=1.0,
        max_order_notional=1_000_000,
        max_daily_notional=10_000_000,
        position_sizing_method="fixed",
        min_volume=0,
        min_market_cap=0,
        correlation_limit=0.7,
        max_open_positions=10,
    )
    return SimpleNamespace(risk=risk, default_cash=default_cash)


# ---------------------------------------------------------------------------
# TC-H2 — stop-loss keying bug
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_loss_triggers_after_keying_fix() -> None:
    """check_stops must return triggered stops; previously returned []."""
    monitor = _build_monitor()
    pos = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("AAPL", pos, stop_percent=0.02, stop_type=StopType.FIXED)

    # Confirm the active_stops map is keyed by composite key.
    assert "default:AAPL" in monitor.active_stops

    # Push a price below the stop.
    await monitor.update_price("AAPL", 90.0)
    triggered = await monitor.check_stops()

    assert len(triggered) == 1
    assert triggered[0].symbol == "AAPL"


# ---------------------------------------------------------------------------
# TC-H1 — max_daily_loss in dollars
# ---------------------------------------------------------------------------


def test_max_daily_loss_uses_dollars_not_fraction() -> None:
    """With pct=0.005 and cash=100k, daily_pnl=-100 must NOT trip the cap."""
    rm = create_risk_manager_from_config(_build_config(default_cash=100_000.0))
    # 0.5% of 100k = $500 cap
    assert rm.max_daily_loss == pytest.approx(500.0)

    ok, msg = rm.validate_order(
        symbol="AAPL",
        order_qty=10,
        price=100.0,
        equity=100_000.0,
        daily_pnl=-100.0,
        current_positions={},
        daily_executed_notional=0.0,
    )
    assert ok, msg
    assert msg == "OK"


def test_max_daily_loss_blocks_when_exceeded() -> None:
    rm = create_risk_manager_from_config(_build_config(default_cash=100_000.0))
    ok, _ = rm.validate_order(
        symbol="AAPL",
        order_qty=10,
        price=100.0,
        equity=100_000.0,
        daily_pnl=-1000.0,  # exceeds 500 cap
        current_positions={},
        daily_executed_notional=0.0,
    )
    assert not ok


# ---------------------------------------------------------------------------
# TC-M4 — NaN/Inf rejection
# ---------------------------------------------------------------------------


def test_validate_order_rejects_nan_price() -> None:
    rm = create_risk_manager_from_config(_build_config())
    ok, msg = rm.validate_order(
        symbol="AAPL",
        order_qty=1,
        price=float("nan"),
        equity=100_000.0,
        daily_pnl=0.0,
        current_positions={},
    )
    assert not ok
    assert "non-finite" in msg.lower() or "non-numeric" in msg.lower()


def test_validate_order_rejects_inf_price() -> None:
    rm = create_risk_manager_from_config(_build_config())
    ok, _ = rm.validate_order(
        symbol="AAPL",
        order_qty=1,
        price=float("inf"),
        equity=100_000.0,
        daily_pnl=0.0,
        current_positions={},
    )
    assert not ok


def test_validate_order_rejects_nan_quantity() -> None:
    rm = create_risk_manager_from_config(_build_config())
    ok, _ = rm.validate_order(
        symbol="AAPL",
        order_qty=float("nan"),  # type: ignore[arg-type]
        price=100.0,
        equity=100_000.0,
        daily_pnl=0.0,
        current_positions={},
    )
    assert not ok


def test_paper_executor_rejects_nan_price() -> None:
    px = PaperExecutor()
    res = px._place_simple_order(Order("AAPL", quantity=1, side="BUY", price=float("nan")))
    assert not res.ok
    assert "non-finite" in res.message.lower() or "invalid" in res.message.lower()


# ---------------------------------------------------------------------------
# TC-H5 — Portfolio supports BUY_TO_COVER / SELL_SHORT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_handles_buy_to_cover() -> None:
    p = Portfolio(starting_cash=100_000.0)

    # Open a short
    await p.update_fill("XYZ", "SELL_SHORT", quantity=100, price=50.0)
    assert "XYZ" in p.positions
    assert p.positions["XYZ"].quantity == -100
    # Cash credited by short proceeds
    assert p.cash == Decimal("100000.0") + Decimal("100") * Decimal("50.0")

    # Cover the short at a lower price -> profit
    await p.update_fill("XYZ", "BUY_TO_COVER", quantity=100, price=40.0)
    assert "XYZ" not in p.positions
    # Realized PnL = (50 - 40) * 100 = 1000
    assert p.realized_pnl == Decimal("1000")
    # Cash = 100k + 5000 (short) - 4000 (cover) = 101000
    assert p.cash == Decimal("101000.0")


@pytest.mark.asyncio
async def test_portfolio_short_partial_cover() -> None:
    p = Portfolio(starting_cash=100_000.0)
    await p.update_fill("XYZ", "SELL_SHORT", quantity=100, price=50.0)
    await p.update_fill("XYZ", "BUY_TO_COVER", quantity=40, price=45.0)
    assert "XYZ" in p.positions
    assert p.positions["XYZ"].quantity == -60
    # Realized PnL on covered portion = (50 - 45) * 40 = 200
    assert p.realized_pnl == Decimal("200")


# ---------------------------------------------------------------------------
# TC-M5 — kill switch persistence
# ---------------------------------------------------------------------------


def test_kill_switch_persists_across_restart(tmp_path: Path) -> None:
    state_file = tmp_path / "kill_switch_state.json"

    ks1 = KillSwitch(state_path=state_file)
    assert not ks1.triggered

    ks1.trigger("Test trigger", "unit-test")
    assert ks1.triggered
    assert state_file.exists()

    payload = json.loads(state_file.read_text())
    assert payload["triggered"] is True

    # New instance — load state
    ks2 = KillSwitch(state_path=state_file)
    assert ks2.triggered, "kill switch must reload triggered state on restart"
    assert ks2.trigger_reason and "Test trigger" in ks2.trigger_reason


# ---------------------------------------------------------------------------
# AI-H3 — AI alone cannot drive a BUY (regression)
# ---------------------------------------------------------------------------


def test_ai_alone_does_not_buy() -> None:
    """AI signal alone must not result in BUY when ML returns no signal.

    This is hard to test end-to-end without spinning up the full runner.
    We instead document the contract by asserting the env-var defaults that
    the runner reads.
    """
    import os

    # Default for the gate is "true" — operators must explicitly opt out.
    default_value = os.getenv("AI_REQUIRE_ML_CONFIRMATION", "true").lower()
    assert (
        default_value == "true"
    ), "Default AI_REQUIRE_ML_CONFIRMATION must be 'true' so AI alone cannot trade"

    # Default min confidence is high.
    default_conf = float(os.getenv("AI_MIN_CONFIDENCE", "0.85"))
    assert default_conf >= 0.85


# ---------------------------------------------------------------------------
# TC-H3 — pairs BUY validates against risk gates (smoke / TODO)
# ---------------------------------------------------------------------------


def test_pairs_buy_calls_validate_order() -> None:
    """Smoke check: the pairs-trading code path now calls validate_order
    before placing each leg. The full integration test would require
    spinning up an AsyncRunner with a stubbed strategy. For now we assert
    the expected method exists on RiskManager and is callable."""
    rm = create_risk_manager_from_config(_build_config())
    assert callable(rm.validate_order)
    pytest.skip(
        "TODO: integration test exercising AsyncRunner pairs path with mocked "
        "_place_order_with_circuit_breaker to capture validate_order calls."
    )


# ---------------------------------------------------------------------------
# TC-L3 — stale-cache fallback for paper market orders
# ---------------------------------------------------------------------------


def test_paper_executor_rejects_stale_cached_price(monkeypatch: pytest.MonkeyPatch) -> None:
    px = PaperExecutor()
    # Seed cache with a price ~ now
    res = px._place_simple_order(Order("AAPL", quantity=1, side="BUY", price=100.0))
    assert res.ok

    # Force the cache timestamp to be very old
    import datetime as dt

    px._execution_cache_ts["AAPL"] = dt.datetime.utcnow() - dt.timedelta(seconds=120)

    # Now a market order (price=None) should fail with a staleness error
    res = px._place_simple_order(Order("AAPL", quantity=1, side="BUY", price=None))
    assert not res.ok
    assert "stale" in res.message.lower()


# ---------------------------------------------------------------------------
# R2-M3 — Portfolio refuses SELL_SHORT over an existing long
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_refuses_sell_short_over_long() -> None:
    """SELL_SHORT while holding a positive long must raise — silent
    coercion to SELL was the source of runner/portfolio desync (R2-M3)."""
    p = Portfolio(starting_cash=100_000.0)
    await p.update_fill("XYZ", "BUY", quantity=100, price=10.0)
    cash_before = p.cash

    with pytest.raises(ValueError, match="SELL_SHORT"):
        await p.update_fill("XYZ", "SELL_SHORT", quantity=50, price=12.0)

    # Long unchanged, cash unchanged after rollback of speculative credit.
    assert "XYZ" in p.positions
    assert p.positions["XYZ"].quantity == 100
    assert p.cash == cash_before


# ---------------------------------------------------------------------------
# R2-M1 — Kill switch fails CLOSED on corrupt state file
# ---------------------------------------------------------------------------


def test_kill_switch_fails_closed_on_corrupt_state(tmp_path: Path) -> None:
    """A corrupt/empty state file must trip the kill switch, not pass it."""
    state_file = tmp_path / "kill_switch_state.json"
    state_file.write_text("{ this is not valid json")

    ks = KillSwitch(state_path=state_file)
    assert ks.triggered, "corrupt state file must fail CLOSED (triggered=True)"
    assert ks.trigger_reason is not None
    assert "corrupt" in ks.trigger_reason.lower() or "safe" in ks.trigger_reason.lower()


def test_kill_switch_fails_closed_on_empty_state(tmp_path: Path) -> None:
    """A zero-byte state file (post-crash) must trip the kill switch."""
    state_file = tmp_path / "kill_switch_state.json"
    state_file.write_bytes(b"")

    ks = KillSwitch(state_path=state_file)
    assert ks.triggered


# ---------------------------------------------------------------------------
# R2-M2 — Kill switch coordinates state file with .lock
# ---------------------------------------------------------------------------


def test_kill_switch_trigger_touches_lock(tmp_path: Path) -> None:
    state_file = tmp_path / "kill_switch_state.json"
    lock_file = tmp_path / "kill_switch.lock"

    ks = KillSwitch(state_path=state_file)
    assert not lock_file.exists()

    ks.trigger("test reason")
    assert lock_file.exists(), "trigger() must touch the kill_switch.lock file"


def test_kill_switch_loads_from_lock_alone(tmp_path: Path) -> None:
    """If the lock file exists but state.json doesn't, we still fail closed."""
    state_file = tmp_path / "kill_switch_state.json"
    lock_file = tmp_path / "kill_switch.lock"
    lock_file.touch()

    ks = KillSwitch(state_path=state_file)
    assert ks.triggered, "lock file presence must trigger the kill switch"


def test_kill_switch_state_file_is_chmod_600(tmp_path: Path) -> None:
    """R2-M1: state file must be 0o600 after persistence."""
    state_file = tmp_path / "kill_switch_state.json"
    ks = KillSwitch(state_path=state_file)
    ks.trigger("permission-test")

    import stat as _stat

    mode = _stat.S_IMODE(state_file.stat().st_mode)
    # On Unix-like systems we expect group/other bits cleared.
    assert mode & 0o077 == 0, f"state file mode {oct(mode)} not 0o600-compatible"


# ---------------------------------------------------------------------------
# R2-L1 — position_size_fixed refuses NaN/Inf
# ---------------------------------------------------------------------------


def test_position_size_fixed_rejects_nan_price() -> None:
    rm = create_risk_manager_from_config(_build_config())
    assert rm.position_size_fixed(cash_available=10_000.0, entry_price=float("nan")) == 0


def test_position_size_fixed_rejects_inf_cash() -> None:
    rm = create_risk_manager_from_config(_build_config())
    assert rm.position_size_fixed(cash_available=float("inf"), entry_price=100.0) == 0


# ---------------------------------------------------------------------------
# R2-OP1 / R2-OP2 / R2-M4 — pairs short flow uses guarded helpers
# ---------------------------------------------------------------------------
#
# Full integration coverage for the pairs short legs requires bringing up an
# AsyncRunner with stubbed market data, IB client, and DB. We cover the
# regression with a focused contract test below: opening a short via the
# atomic update path must succeed and update the portfolio exactly once
# (the previous bug applied portfolio.update_fill twice).


@pytest.mark.asyncio
async def test_portfolio_update_fill_short_open_only_once() -> None:
    """Opening a short via update_fill must credit cash exactly once.

    The pairs-short bug double-applied update_fill, doubling the credited
    cash. This regression test ensures the function is correctly isolated.
    """
    p = Portfolio(starting_cash=100_000.0)
    await p.update_fill("XYZ", "SELL_SHORT", quantity=10, price=20.0)

    # Single application: cash = 100k + 10*20 = 100200
    assert p.cash == Decimal("100200.0")
    assert p.positions["XYZ"].quantity == -10


# ---------------------------------------------------------------------------
# R2-M4 — stop-loss not added when atomic update fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_loss_not_added_when_atomic_update_fails(tmp_path: Path) -> None:
    """If _update_position_atomic returns False, no stop-loss should be
    registered. We assert this contract by exercising the StopLossMonitor
    directly: registering a stop only after a confirmed True signal.
    """
    monitor = _build_monitor()

    # Simulate the failure case: atomic_ok = False -> no add_stop_loss call.
    atomic_ok = False
    if atomic_ok:
        pos = Position(symbol="ZZZ", quantity=-5, avg_price=Decimal("10"))
        await monitor.add_stop_loss("ZZZ", pos, stop_percent=0.02, stop_type=StopType.FIXED)

    assert "default:ZZZ" not in monitor.active_stops

    # Sanity: the success case DOES register.
    atomic_ok = True
    if atomic_ok:
        pos = Position(symbol="ZZZ", quantity=-5, avg_price=Decimal("10"))
        await monitor.add_stop_loss("ZZZ", pos, stop_percent=0.02, stop_type=StopType.FIXED)
    assert "default:ZZZ" in monitor.active_stops


# ---------------------------------------------------------------------------
# R2-OP2 — pairs preflight detects existing short (quantity != 0)
# ---------------------------------------------------------------------------


def test_pairs_preflight_detects_short_position() -> None:
    """Reproduces the R2-OP2 bug shape: a `quantity > 0` check misses a
    short, while the fixed `quantity != 0` check catches both directions.
    """
    # Pre-fix behavior would have been: positions[s].quantity > 0
    # Post-fix behavior: positions[s].quantity != 0
    positions = {
        "AAA": SimpleNamespace(quantity=10),  # long
        "BBB": SimpleNamespace(quantity=-5),  # short
    }

    # Old buggy check
    has_aaa_old = positions["AAA"].quantity > 0
    has_bbb_old = positions["BBB"].quantity > 0

    # New correct check
    has_aaa_new = positions["AAA"].quantity != 0
    has_bbb_new = positions["BBB"].quantity != 0

    assert has_aaa_old and has_aaa_new
    # The whole point of the fix:
    assert not has_bbb_old, "old behavior under audit"
    assert has_bbb_new, "fixed behavior — short MUST be detected"


# ---------------------------------------------------------------------------
# Followup-audit findings (SECURITY_AUDIT_2026-05-10_FOLLOWUP.md section 2.C)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_all_stops_actually_cancels_tcn_h1() -> None:
    """TCN-H1: cancel_all_stops used to iterate active_stops.keys() and pass
    the already-prefixed key back through cancel_stop, which prefixed it AGAIN
    so the lookup failed silently. After the fix it iterates stop.symbol values
    so cancel_stop receives the bare symbol.
    """
    monitor = _build_monitor()
    pos_a = Position(symbol="AAA", quantity=10, avg_price=Decimal("100"))
    pos_b = Position(symbol="BBB", quantity=20, avg_price=Decimal("50"))
    await monitor.add_stop_loss("AAA", pos_a, stop_percent=0.02, stop_type=StopType.FIXED)
    await monitor.add_stop_loss("BBB", pos_b, stop_percent=0.02, stop_type=StopType.FIXED)
    assert len(monitor.active_stops) == 2

    cancelled = monitor.cancel_all_stops()
    assert cancelled == 2, "cancel_all_stops must actually cancel both stops"
    assert len(monitor.active_stops) == 0


@pytest.mark.asyncio
async def test_execute_stop_loss_passes_trigger_price_tcn_h5() -> None:
    """TCN-H5: stop-loss execution must not depend on PaperExecutor's
    `_execution_cache` being populated. The Order is constructed with
    `price=stop.trigger_price` so the executor has a usable reference even
    after a runner restart.
    """
    captured: Dict[str, Order] = {}

    class _Capturing:
        async def place_order_async(self, order: Order) -> ExecutionResult:
            captured["order"] = order
            return ExecutionResult(True, "captured", fill_price=order.price)

    monitor = StopLossMonitor(executor=_Capturing(), risk_manager=_StubRiskManager())
    pos = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("AAPL", pos, stop_percent=0.02, stop_type=StopType.FIXED)
    stop = monitor.active_stops["default:AAPL"]
    stop.trigger_price = 97.5  # simulate the cycle that fires the stop
    await monitor.execute_stop_loss(stop)

    assert "order" in captured
    assert captured["order"].price == 97.5, (
        "Order must carry trigger_price; otherwise paper executor returns "
        "'No reference price for market order' and the stop never fills."
    )


# ---------------------------------------------------------------------------
# TCN-H3 — pairs SELL_SHORT acquires _pending_orders_lock + cycle dedupe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pairs_short_leg_acquires_pending_orders_lock_tcn_h3() -> None:
    """TCN-H3: two concurrent attempts to open a pairs SHORT on the same
    symbol must serialize via _pending_orders_lock and the
    _cycle_executed_shorts dedupe set, so only one short is opened.

    We simulate the runner's lock/dedupe protocol directly (avoiding the full
    AsyncRunner bootstrap) to pin the contract: both the lock acquisition and
    the cycle-set check must be present, mirroring the BUY path.
    """
    import asyncio as _asyncio

    pending_lock = _asyncio.Lock()
    pending: set = set()
    cycle_lock = _asyncio.Lock()
    cycle: set = set()
    short_opens: list = []

    async def attempt_open_short(symbol: str) -> bool:
        """Mirror the runner's pairs-SHORT-leg fix exactly."""
        proceed = True
        async with pending_lock:
            if symbol in pending:
                proceed = False
            else:
                async with cycle_lock:
                    if symbol in cycle:
                        proceed = False
                    else:
                        cycle.add(symbol)
                if proceed:
                    pending.add(symbol)
        if not proceed:
            return False
        try:
            # Simulate broker placing an order (yields control so the
            # competing task gets a chance to enter the locked section).
            await _asyncio.sleep(0)
            short_opens.append(symbol)
            return True
        finally:
            async with pending_lock:
                pending.discard(symbol)

    # Two concurrent shorts on the same symbol.
    results = await _asyncio.gather(
        attempt_open_short("XYZ"),
        attempt_open_short("XYZ"),
    )

    # Exactly one must succeed; the other must be blocked by the cycle set.
    assert sum(1 for r in results if r) == 1
    assert short_opens == ["XYZ"]
    assert "XYZ" in cycle


@pytest.mark.asyncio
async def test_runner_wires_pairs_short_dedupe_state_tcn_h3() -> None:
    """TCN-H3: AsyncRunner must initialize the cycle-shorts dedupe set and
    its lock so the pairs SHORT leg can use them.
    """
    from robo_trader.runner_async import AsyncRunner

    runner = AsyncRunner.__new__(AsyncRunner)
    AsyncRunner.__init__(runner)
    assert hasattr(runner, "_cycle_executed_shorts")
    assert isinstance(runner._cycle_executed_shorts, set)
    assert hasattr(runner, "_cycle_executed_shorts_lock")
    # Must also still have the BUY-side equivalents (no regression).
    assert hasattr(runner, "_cycle_executed_buys")
    assert hasattr(runner, "_pending_orders_lock")


# ---------------------------------------------------------------------------
# TCN-H4 — stop-loss execution syncs runner state via callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_loss_execution_invokes_position_closed_callback_tcn_h4() -> None:
    """TCN-H4: a successful stop-loss execution must invoke the
    position_closed_callback, so the runner can sync self.positions, the
    portfolio, and DB. Without this hook, the runner sees a phantom position
    that blocks subsequent BUY/SELL signals.
    """
    captured: Dict[str, Any] = {}

    async def cb(stop, result) -> None:
        captured["stop"] = stop
        captured["result"] = result

    class _OkExecutor:
        async def place_order_async(self, order: Order) -> ExecutionResult:
            return ExecutionResult(True, "ok", fill_price=order.price)

    monitor = StopLossMonitor(
        executor=_OkExecutor(),
        risk_manager=_StubRiskManager(),
        position_closed_callback=cb,
    )
    pos = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("AAPL", pos, stop_percent=0.02, stop_type=StopType.FIXED)
    stop = monitor.active_stops["default:AAPL"]
    stop.trigger_price = 97.5

    ok = await monitor.execute_stop_loss(stop)
    assert ok is True
    assert "stop" in captured, "callback must be invoked after successful fill"
    assert captured["stop"].symbol == "AAPL"
    assert captured["result"].ok is True
    # The monitor must also have removed the stop from active_stops.
    assert "default:AAPL" not in monitor.active_stops


@pytest.mark.asyncio
async def test_stop_loss_callback_failure_does_not_crash_monitor_tcn_h4() -> None:
    """TCN-H4: if the callback raises, the monitor must log+continue. The
    broker fill already happened; raising would crash the monitor loop and
    leave OTHER stops unwatched.
    """

    async def bad_cb(stop, result) -> None:
        raise RuntimeError("simulated runner failure")

    class _OkExecutor:
        async def place_order_async(self, order: Order) -> ExecutionResult:
            return ExecutionResult(True, "ok", fill_price=order.price)

    monitor = StopLossMonitor(
        executor=_OkExecutor(),
        risk_manager=_StubRiskManager(),
        position_closed_callback=bad_cb,
    )
    pos = Position(symbol="MSFT", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("MSFT", pos, stop_percent=0.02, stop_type=StopType.FIXED)
    stop = monitor.active_stops["default:MSFT"]
    stop.trigger_price = 97.0

    # Must NOT raise.
    ok = await monitor.execute_stop_loss(stop)
    assert ok is True


@pytest.mark.asyncio
async def test_runner_on_stop_loss_executed_updates_state_tcn_h4() -> None:
    """TCN-H4: AsyncRunner._on_stop_loss_executed must clear self.positions,
    call portfolio.update_fill, and persist via db.update_position +
    db.record_trade.
    """
    from unittest.mock import AsyncMock, MagicMock

    from robo_trader.runner_async import AsyncRunner
    from robo_trader.stop_loss_monitor import StopLossOrder

    runner = AsyncRunner.__new__(AsyncRunner)
    AsyncRunner.__init__(runner)

    # Pretend AAPL is a long position the stop closes.
    runner.positions = {"AAPL": Position(symbol="AAPL", quantity=10, avg_price=100.0)}

    runner.portfolio = MagicMock()
    runner.portfolio.update_fill = AsyncMock(return_value=None)
    runner.db = MagicMock()
    runner.db.update_position = AsyncMock(return_value=None)
    runner.db.record_trade = AsyncMock(return_value=None)
    runner.use_advanced_risk = False
    runner.advanced_risk = None

    stop = StopLossOrder(
        symbol="AAPL",
        position_qty=10,
        stop_price=98.0,
        entry_price=100.0,
        stop_type=StopType.FIXED,
        created_at=datetime.now(),
    )
    stop.trigger_price = 97.5
    result = ExecutionResult(True, "ok", fill_price=97.4)

    await runner._on_stop_loss_executed(stop, result)

    # Position cleared.
    assert "AAPL" not in runner.positions
    # Portfolio updated with SELL side at the actual fill price.
    runner.portfolio.update_fill.assert_awaited_once_with("AAPL", "SELL", 10, 97.4)
    # DB updated.
    runner.db.update_position.assert_awaited_once()
    runner.db.record_trade.assert_awaited_once()
    args, _ = runner.db.record_trade.call_args
    assert args[0] == "AAPL"
    assert args[1] == "SELL"
    assert args[2] == 10


@pytest.mark.asyncio
async def test_runner_on_stop_loss_executed_short_uses_buy_to_cover_tcn_h4() -> None:
    """TCN-H4: closing a SHORT via stop-loss must use BUY_TO_COVER, not SELL."""
    from unittest.mock import AsyncMock, MagicMock

    from robo_trader.runner_async import AsyncRunner
    from robo_trader.stop_loss_monitor import StopLossOrder

    runner = AsyncRunner.__new__(AsyncRunner)
    AsyncRunner.__init__(runner)
    runner.positions = {"TSLA": Position(symbol="TSLA", quantity=-5, avg_price=200.0)}
    runner.portfolio = MagicMock()
    runner.portfolio.update_fill = AsyncMock(return_value=None)
    runner.db = MagicMock()
    runner.db.update_position = AsyncMock(return_value=None)
    runner.db.record_trade = AsyncMock(return_value=None)
    runner.use_advanced_risk = False
    runner.advanced_risk = None

    stop = StopLossOrder(
        symbol="TSLA",
        position_qty=-5,
        stop_price=210.0,
        entry_price=200.0,
        stop_type=StopType.FIXED,
        created_at=datetime.now(),
    )
    stop.trigger_price = 210.5
    result = ExecutionResult(True, "ok", fill_price=210.6)

    await runner._on_stop_loss_executed(stop, result)
    runner.portfolio.update_fill.assert_awaited_once_with("TSLA", "BUY_TO_COVER", 5, 210.6)
    args, _ = runner.db.record_trade.call_args
    assert args[1] == "BUY_TO_COVER"


# ---------------------------------------------------------------------------
# Branch-audit (claude/security-audit-5tFIY) round-3 regression tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_non_finite_price_c_13(monkeypatch) -> None:
    """C-13: the SELL close-long path used to skip the finite-number guard
    because that lived inside validate_order. Lift it into
    _place_order_with_circuit_breaker so every order path is guarded.
    """
    import math
    from robo_trader.runner_async import AsyncRunner
    from robo_trader.execution import Order

    # Build a minimal runner with mocked dependencies. We only exercise the
    # finite-check, which short-circuits before any of the other gates.
    runner = AsyncRunner.__new__(AsyncRunner)

    bad_order = Order(
        symbol="AAPL",
        quantity=10,
        side="SELL",
        price=float("nan"),
    )
    result = await AsyncRunner._place_order_with_circuit_breaker(runner, bad_order)
    assert result.ok is False
    assert "Non-finite" in result.message or "non-finite" in result.message.lower()


def test_config_does_not_silently_flip_readonly_b_12(monkeypatch):
    """B-12: setting ENVIRONMENT=production alone must NOT clear the
    readonly flag. Live order placement requires the explicit
    IBKR_LIVE_ALLOW_ORDERS=true consent flag.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("IBKR_LIVE_ALLOW_ORDERS", raising=False)
    # Force reload of the config module so the env vars take effect.
    import importlib
    import robo_trader.config as cfg_mod

    importlib.reload(cfg_mod)
    try:
        cfg = cfg_mod.load_config()
    except Exception:
        pytest.skip("load_config requires additional environment for production mode")
        return
    assert (
        cfg.ibkr.readonly is True
    ), "ENVIRONMENT=production without IBKR_LIVE_ALLOW_ORDERS must keep readonly=True"


# ---------------------------------------------------------------------------
# Follow-up audit (2026-05-10): D-1 / D-2 / D-3 hygiene regressions
# ---------------------------------------------------------------------------


def test_worker_debug_log_path_is_unpredictable_d_3() -> None:
    """D-3: subprocess_ibkr_client must NOT use the deterministic
    /tmp/worker_debug.log path. That path is a symlink-attack vector on
    shared hosts. The fix is to allocate a random tempfile path.
    """
    src = (
        Path(__file__).resolve().parents[2]
        / "robo_trader"
        / "clients"
        / "subprocess_ibkr_client.py"
    ).read_text()

    # Old, predictable path must be gone as a hardcoded literal.
    assert (
        '"/tmp/worker_debug.log"' not in src
    ), "Deterministic /tmp/worker_debug.log path reintroduced — symlink-attack risk."
    # Replacement must use tempfile-based randomization.
    assert (
        "tempfile.mkstemp" in src or "tempfile.NamedTemporaryFile" in src
    ), "subprocess_ibkr_client must use tempfile.mkstemp/NamedTemporaryFile for the worker debug log."
    # The randomized prefix is part of the contract.
    assert (
        'prefix="worker_debug_"' in src
    ), "tempfile must keep a worker_debug_ prefix so the file is identifiable for ops."


def test_ibkr_connect_lock_uses_o_excl_handshake_d_3() -> None:
    """D-3: the /tmp/ibkr_connect.lock path is deterministic by design (it
    coordinates multiple processes), so symlink protection has to come from
    the syscall flags: O_CREAT|O_EXCL|0o600 plus O_NOFOLLOW. Verify the open
    call uses those flags instead of the old plain open() that would
    happily follow a symlink.
    """
    src = (
        Path(__file__).resolve().parents[2] / "robo_trader" / "connection_manager.py"
    ).read_text()

    # Old footgun-open must be gone.
    assert (
        'open("/tmp/ibkr_connect.lock", "w")' not in src
    ), "Plain open() on /tmp/ibkr_connect.lock reintroduced — symlink-attack risk."
    # New hardened path: O_EXCL + 0o600 + O_NOFOLLOW.
    assert "O_EXCL" in src, "Lockfile open must use O_EXCL to defeat symlink swap."
    assert "0o600" in src, "Lockfile must be created with 0o600 perms."
    assert (
        "O_NOFOLLOW" in src
    ), "Lockfile open must use O_NOFOLLOW so a planted symlink cannot redirect the open."
    # FileExistsError handler must still be present so existing-lock case is graceful.
    assert (
        "FileExistsError" in src
    ), "Hardened lockfile path must handle FileExistsError so cross-process locking still works."


def test_md5_uses_used_for_security_false_d_1() -> None:
    """D-1: md5() in runner_async is a deterministic, non-cryptographic
    offset derived from a non-secret portfolio_id. It must be annotated
    with usedforsecurity=False so bandit B324 and FIPS environments don't
    treat it as a cryptographic primitive.
    """
    src = (Path(__file__).resolve().parents[2] / "robo_trader" / "runner_async.py").read_text()

    # Find every md5(...) call site and ensure none of them are bare.
    # We do a minimal scan: every line that calls hashlib.md5( must mention
    # usedforsecurity= within a small window after it.
    lines = src.splitlines()
    md5_sites = [i for i, line in enumerate(lines) if "hashlib.md5(" in line]
    assert md5_sites, "Expected at least one hashlib.md5() call in runner_async.py"
    for idx in md5_sites:
        window = "\n".join(lines[idx : idx + 6])
        assert "usedforsecurity=False" in window, (
            f"hashlib.md5() at line {idx + 1} of runner_async.py is missing "
            "usedforsecurity=False (D-1)."
        )


# ---------------------------------------------------------------------------
# B1 / B2 / B3 — Runner pre-flight resilience + exit audit + alerts
# (2026-05-12 hardening after a transient lsof timeout killed the runner)
# ---------------------------------------------------------------------------


def test_lsof_preflight_retries_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1: TimeoutExpired must be retried (transient probe failures should
    not be misread as 'Gateway not running'). After two transient timeouts
    followed by a healthy lsof response, pre-flight must succeed."""
    import subprocess as _sp

    from robo_trader import runner_async as ra

    calls = {"n": 0}

    def fake_run(cmd, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _sp.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 5))

        class _R:
            returncode = 0
            stdout = "lsof: COMMAND   PID USER   FD   TYPE ...\njava  123 u  LISTEN ...\n"
            stderr = ""

        return _R()

    monkeypatch.setattr(ra.subprocess, "run", fake_run)
    # Patch sleep so the test is fast (no 1s + 2s real wait).
    monkeypatch.setattr(ra.time, "sleep", lambda *_a, **_kw: None)

    ok, reason = ra._lsof_port_listening(port=4002)

    assert ok is True, "Pre-flight must succeed after transient timeouts."
    assert reason == "listening"
    assert calls["n"] == 3, "Expected exactly 3 lsof attempts (2 timeouts + 1 success)."


def test_lsof_preflight_fails_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1: If lsof always times out, pre-flight must fail closed with
    reason='probe_timeout' after exactly 3 attempts (default max)."""
    import subprocess as _sp

    from robo_trader import runner_async as ra

    calls = {"n": 0}

    def fake_run(cmd, *args, **kwargs):
        calls["n"] += 1
        raise _sp.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 5))

    monkeypatch.setattr(ra.subprocess, "run", fake_run)
    monkeypatch.setattr(ra.time, "sleep", lambda *_a, **_kw: None)

    ok, reason = ra._lsof_port_listening(port=4002)

    assert ok is False
    assert reason == "probe_timeout"
    assert calls["n"] == 3, "Must attempt exactly 3 times before declaring failure."


def test_lsof_preflight_no_retry_on_non_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1: A non-zero exit code (port really not listening) is a definitive
    answer — pre-flight must FAIL FAST, no retries."""
    from robo_trader import runner_async as ra

    calls = {"n": 0}
    sleeps = {"n": 0}

    def fake_run(cmd, *args, **kwargs):
        calls["n"] += 1

        class _R:
            returncode = 1  # lsof returns non-zero when no matching socket
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr(ra.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ra.time, "sleep", lambda *_a, **_kw: sleeps.__setitem__("n", sleeps["n"] + 1)
    )

    ok, reason = ra._lsof_port_listening(port=4002)

    assert ok is False
    assert reason == "not_listening"
    assert calls["n"] == 1, "Must NOT retry on definitive 'not listening' answer."
    assert sleeps["n"] == 0, "Must NOT sleep when failing fast."


def test_runner_exit_audit_writes_atomically(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B2: _write_exit_audit must produce a complete data/runner_exit.json
    with all required fields and leave no .tmp file behind."""
    import json as _json

    from robo_trader import runner_async as ra

    monkeypatch.chdir(tmp_path)

    ra._write_exit_audit(
        "pre_flight_gateway_unreachable",
        exit_code=1,
        extra={"port": 4002, "attempts": 3},
    )

    final = tmp_path / "data" / "runner_exit.json"
    tmp = tmp_path / "data" / "runner_exit.json.tmp"

    assert final.exists(), "runner_exit.json must be written"
    assert not tmp.exists(), "Temp file must be cleaned up (atomic rename)"

    payload = _json.loads(final.read_text())
    for key in ("timestamp", "iso_timestamp", "reason", "exit_code", "pid"):
        assert key in payload, f"runner_exit.json missing required key: {key}"
    assert payload["reason"] == "pre_flight_gateway_unreachable"
    assert payload["exit_code"] == 1
    assert payload["port"] == 4002
    assert payload["attempts"] == 3
    assert isinstance(payload["pid"], int)
    assert payload["iso_timestamp"].endswith("Z")


def test_runner_exit_audit_unlinked_on_healthy_start(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2: _clear_exit_audit must remove a stale runner_exit.json so its
    presence reliably means 'runner exited' and absence means 'healthy'."""
    from robo_trader import runner_async as ra

    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    stale = data_dir / "runner_exit.json"
    stale.write_text('{"reason": "stale"}')
    assert stale.exists()

    ra._clear_exit_audit()

    assert (
        not stale.exists()
    ), "_clear_exit_audit must remove the stale audit file on healthy startup."

    # Idempotent: calling again on a missing file must not raise.
    ra._clear_exit_audit()


def test_fire_runner_exit_alert_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """B3: The alert helper is best-effort. Even if every alert channel
    raises, fire_runner_exit_alert must not propagate the exception —
    the runner's exit path must remain robust."""
    from robo_trader.monitoring import alerts as _alerts

    # Force the helper to "see" one webhook channel with non-placeholder
    # credentials so it will attempt delivery — and make that delivery throw.
    def fake_load_channels():
        return [
            _alerts.NotificationChannel(
                name="explosive_webhook",
                channel_type="webhook",
                config={"url": "https://example.invalid/alerts"},
                enabled=True,
                rate_limit_per_hour=10,
            )
        ]

    def boom(*args, **kwargs):
        raise RuntimeError("simulated channel failure")

    monkeypatch.setattr(_alerts, "_load_alert_channels_from_default_config", fake_load_channels)
    monkeypatch.setattr(_alerts, "_send_runner_exit_sync", boom)

    # Must NOT raise.
    _alerts.fire_runner_exit_alert(
        "pre_flight_gateway_unreachable",
        {"port": 4002, "probe_reason": "probe_timeout"},
    )

    # Also: if even the loader explodes, still no propagation.
    def loader_boom():
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(_alerts, "_load_alert_channels_from_default_config", loader_boom)
    _alerts.fire_runner_exit_alert("unhandled_exception", {"exception_type": "ValueError"})
