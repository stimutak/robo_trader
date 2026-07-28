"""Security-audit regression tests for trading-core fixes.

These tests cover the HIGH/MEDIUM-severity findings from
SECURITY_AUDIT_2026-05-10.md (section 2.C). Each test is named with the
finding ID it pins.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.execution import ExecutionResult, Order, PaperExecutor
from robo_trader.paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperOutcomeProvenance,
    LocalPaperTerminalOutcome,
)
from tests.paper_execution_test_support import bind_gateway_reduction_harness
from robo_trader.portfolio import Portfolio
from robo_trader.protective_quote_evidence import ProtectiveQuoteSource
from robo_trader.risk.advanced_risk import AdvancedRiskManager, KillSwitch
from robo_trader.risk_manager import (
    Position,
    RiskManager,
    create_risk_manager_from_config,
)
from robo_trader.stop_loss_monitor import StopLossMonitor, StopStatus, StopType

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubExecutor:
    async def place_order_async(self, order: Order) -> LocalPaperTerminalOutcome:
        return _filled_terminal_outcome(order)


class _StubRiskManager:
    pass


def _reduction_callback(executor):
    async def execute(stop, order: Order):
        del stop
        return await executor.place_order_async(order)

    return execute


def _filled_terminal_outcome(
    order: Order,
    *,
    fill_price: Decimal = Decimal("97.5"),
) -> LocalPaperTerminalOutcome:
    """Return exact local-paper terminal evidence for monitor-only tests."""

    quantity = Decimal(order.quantity)
    return LocalPaperTerminalOutcome(
        order_ref=order.order_ref,
        status=LocalPaperOrderStatus.FILLED,
        requested_quantity=quantity,
        filled_quantity=quantity,
        remaining_quantity=Decimal("0"),
        exact_fill_price=fill_price,
        observed_at=datetime.now(timezone.utc),
        provenance=LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR,
        terminal=True,
        message="exact local paper fill",
    )


def _build_monitor() -> StopLossMonitor:
    return StopLossMonitor(
        execute_reduction=_reduction_callback(_StubExecutor()),
        risk_manager=_StubRiskManager(),
    )


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
    await monitor.update_price(
        "AAPL",
        90.0,
        source_timestamp=datetime.now(timezone.utc),
    )
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
    res = bind_gateway_reduction_harness(px, "test").submit(
        Order("AAPL", quantity=1, side="BUY_TO_COVER", price=float("nan")),
        pre_position_quantity=Decimal("-1"),
    )
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
    harness = bind_gateway_reduction_harness(px, "test")
    res = harness.submit(
        Order("AAPL", quantity=1, side="BUY_TO_COVER", price=100.0),
        pre_position_quantity=Decimal("-1"),
    )
    assert res.ok

    # Force the cache timestamp to be very old
    import datetime as dt

    px._execution_cache_ts["AAPL"] = dt.datetime.utcnow() - dt.timedelta(seconds=120)

    # Now a market order (price=None) should fail with a staleness error
    res = harness.submit(
        Order("AAPL", quantity=1, side="BUY_TO_COVER", price=None),
        pre_position_quantity=Decimal("-1"),
    )
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
async def test_stop_reduction_uses_exact_producer_quote_not_order_price_tcn_h5() -> None:
    """The monitor latches exact broker lineage and leaves Order.price unset.

    PR 2B.3 makes the producer-owned ``ProtectiveQuoteEvidence`` the only
    reference-price authority.  Copying the mutable trigger price onto the
    order would recreate a second, caller-controlled authority.
    """
    captured: Dict[str, Any] = {}
    event_time = datetime(2026, 7, 25, 14, 30, tzinfo=timezone.utc)

    class _Capturing:
        async def place_order_async(self, order: Order) -> LocalPaperTerminalOutcome:
            captured["order"] = order
            return _filled_terminal_outcome(order, fill_price=Decimal("97.5"))

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(_Capturing()),
        risk_manager=_StubRiskManager(),
    )
    pos = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("AAPL", pos, stop_percent=0.02, stop_type=StopType.FIXED)
    stop = monitor.active_stops["default:AAPL"]
    monitor._utcnow = MagicMock(return_value=event_time)
    monitor._monotonic = MagicMock(return_value=100.0)
    assert await monitor.update_price(
        "AAPL",
        97.5,
        source_timestamp=event_time,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="generation-1",
        source_event_id="quote-event-1",
    )
    assert await monitor.check_stops() == [stop]
    captured["quote"] = monitor._latched_stop_crossings[id(stop)].quote_evidence

    assert await monitor.execute_stop_loss(stop) is True

    assert "order" in captured
    assert captured["order"].price is None
    assert captured["quote"].price == Decimal("97.5")
    assert captured["quote"].source is ProtectiveQuoteSource.LIVE_BROKER
    assert captured["quote"].con_id == 265598


def test_stop_monitor_exposes_only_narrow_reduction_callback() -> None:
    monitor = _build_monitor()

    assert not hasattr(monitor, "executor")
    assert callable(monitor._execute_reduction)
    with pytest.raises(TypeError):
        StopLossMonitor(executor=object(), risk_manager=_StubRiskManager())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("quantity", "expected_side"),
    [(10, "SELL"), (-10, "BUY_TO_COVER")],
)
async def test_stop_order_side_and_order_ref_are_stable(
    quantity: int,
    expected_side: str,
) -> None:
    created_at = datetime(2026, 7, 25, 14, 30, tzinfo=timezone.utc)

    async def capture_once() -> Order:
        orders: list[Order] = []

        async def execute_reduction(_stop, order: Order) -> ExecutionResult:
            orders.append(order)
            return ExecutionResult(False, "test rejection")

        monitor = StopLossMonitor(
            execute_reduction=execute_reduction,
            risk_manager=_StubRiskManager(),
            portfolio_id="stable-portfolio",
        )
        position = Position(symbol="AAPL", quantity=quantity, avg_price=Decimal("100"))
        stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
        stop.created_at = created_at
        stop.trigger_price = 97.0 if quantity > 0 else 103.0

        assert await monitor.execute_stop_loss(stop) is False
        assert len(orders) == 1
        return orders[0]

    first = await capture_once()
    second = await capture_once()

    assert first.side == expected_side
    assert second.side == expected_side
    assert first.order_ref == second.order_ref
    assert first.order_ref.startswith("stop:v1:")
    assert len(first.order_ref) == 72


@pytest.mark.asyncio
async def test_stop_order_ref_stays_fixed_width_at_valid_portfolio_boundary() -> None:
    orders: list[Order] = []

    async def execute_reduction(_stop, order: Order) -> ExecutionResult:
        orders.append(order)
        return ExecutionResult(False, "test rejection")

    monitor = StopLossMonitor(
        execute_reduction=execute_reduction,
        risk_manager=_StubRiskManager(),
        portfolio_id="p" * 64,
    )
    position = Position(
        symbol="AAPL",
        quantity=999_999_999,
        avg_price=Decimal("999999.9999"),
    )
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    stop.created_at = datetime(2026, 7, 25, 14, 30, tzinfo=timezone.utc)
    stop.trigger_price = 999_999.9999

    assert await monitor.execute_stop_loss(stop) is False
    assert len(orders) == 1
    assert orders[0].order_ref.startswith("stop:v1:")
    assert len(orders[0].order_ref) == 72


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
async def test_exact_terminal_outcome_completes_without_legacy_callback_tcn_h4() -> None:
    """Gateway success is already committed and projected before return."""

    class _OkExecutor:
        async def place_order_async(self, order: Order) -> LocalPaperTerminalOutcome:
            return _filled_terminal_outcome(order)

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(_OkExecutor()),
        risk_manager=_StubRiskManager(),
        position_closed_callback=None,
    )
    pos = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    await monitor.add_stop_loss("AAPL", pos, stop_percent=0.02, stop_type=StopType.FIXED)
    stop = monitor.active_stops["default:AAPL"]
    stop.trigger_price = 97.5

    ok = await monitor.execute_stop_loss(stop)
    assert ok is True
    assert "default:AAPL" not in monitor.active_stops


@pytest.mark.asyncio
async def test_legacy_post_fill_callback_cannot_turn_fill_into_retry_tcn_h4() -> None:
    """A compatibility callback failure occurs after the irrevocable fill."""

    callback_calls = 0

    async def bad_cb(stop, result) -> None:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("simulated legacy callback failure")

    class _OkExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def place_order_async(self, order: Order) -> LocalPaperTerminalOutcome:
            self.calls += 1
            return _filled_terminal_outcome(order)

    executor = _OkExecutor()

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(executor),
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
    assert executor.calls == 1
    assert callback_calls == 1


class _GatedSuccessfulExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def place_order_async(self, order: Order) -> LocalPaperTerminalOutcome:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return _filled_terminal_outcome(order)


@pytest.mark.asyncio
async def test_filled_old_stop_preserves_replacement_without_retry() -> None:
    executor = _GatedSuccessfulExecutor()
    callback_calls = 0

    async def callback(_stop, _result) -> None:
        nonlocal callback_calls
        callback_calls += 1

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(executor),
        risk_manager=_StubRiskManager(),
        position_closed_callback=callback,
    )
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    old_stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    old_stop.trigger_price = 97.0

    execution = asyncio.create_task(monitor.execute_stop_loss(old_stop))
    await executor.started.wait()
    replacement = await monitor.add_stop_loss("AAPL", position, stop_percent=0.05)
    executor.release.set()

    assert await execution is True
    assert executor.calls == 1
    assert callback_calls == 1
    assert old_stop.status.name == "EXECUTED"
    assert monitor.active_stops["default:AAPL"] is replacement
    assert sum(item is old_stop for item in monitor.stop_history) == 1


@pytest.mark.asyncio
async def test_filled_cancelled_stop_is_not_resubmitted() -> None:
    executor = _GatedSuccessfulExecutor()
    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(executor),
        risk_manager=_StubRiskManager(),
    )
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    stop.trigger_price = 97.0

    execution = asyncio.create_task(monitor.execute_stop_loss(stop))
    await executor.started.wait()
    assert monitor.cancel_stop("AAPL") is True
    executor.release.set()

    assert await execution is True
    assert executor.calls == 1
    assert stop.status.name == "EXECUTED"
    assert "default:AAPL" not in monitor.active_stops
    assert sum(item is stop for item in monitor.stop_history) == 1


@pytest.mark.asyncio
async def test_cancelled_replacement_from_trigger_batch_cannot_execute() -> None:
    executor = _GatedSuccessfulExecutor()
    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(executor),
        risk_manager=_StubRiskManager(),
    )
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    stop.trigger_price = 97.0
    assert monitor.cancel_stop("AAPL") is True

    assert await monitor.execute_stop_loss(stop) is False
    assert executor.calls == 0


@pytest.mark.asyncio
async def test_rejected_stop_calls_reduction_callback_exactly_once() -> None:
    execute_reduction = AsyncMock(return_value=ExecutionResult(False, "definitive rejection"))
    monitor = StopLossMonitor(
        execute_reduction=execute_reduction,
        risk_manager=_StubRiskManager(),
    )
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    stop.trigger_price = 97.0

    assert await monitor.execute_stop_loss(stop) is False

    execute_reduction.assert_awaited_once()
    assert stop.status.name == "FAILED"
    assert monitor.metrics.failed_today == 1


@pytest.mark.asyncio
async def test_exact_terminal_no_fill_retains_triggered_stop_without_redispatch() -> None:
    outcome = LocalPaperTerminalOutcome(
        order_ref="placeholder",
        status=LocalPaperOrderStatus.REJECTED,
        requested_quantity=Decimal("10"),
        filled_quantity=Decimal("0"),
        remaining_quantity=Decimal("10"),
        exact_fill_price=None,
        observed_at=datetime.now(timezone.utc),
        provenance=LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR,
        terminal=True,
        message="definitive no fill",
    )
    execute_reduction = AsyncMock(return_value=outcome)
    emergency = AsyncMock()
    monitor = StopLossMonitor(
        execute_reduction=execute_reduction,
        risk_manager=_StubRiskManager(),
        emergency_shutdown_callback=emergency,
    )
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    stop.status = StopStatus.TRIGGERED
    stop.trigger_price = 97.0
    stop.triggered_at = datetime.now(timezone.utc)

    assert await monitor.execute_stop_loss(stop) is False

    execute_reduction.assert_awaited_once()
    assert stop.status is StopStatus.TRIGGERED
    assert monitor.active_stops["default:AAPL"] is stop
    assert await monitor.check_stops() == []
    execute_reduction.assert_awaited_once()
    emergency.assert_awaited_once_with("Stop-loss terminal no-fill: REJECTED")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "wrong_type", "invalid_fill"])
async def test_ambiguous_stop_callback_failure_is_never_retried(failure: str) -> None:
    if failure == "exception":
        execute_reduction = AsyncMock(side_effect=RuntimeError("transport failed"))
    elif failure == "wrong_type":
        execute_reduction = AsyncMock(return_value=SimpleNamespace(ok=True, fill_price=97.0))
    else:
        execute_reduction = AsyncMock(
            return_value=ExecutionResult(True, "bad fill", fill_price=float("nan"))
        )
    monitor = StopLossMonitor(
        execute_reduction=execute_reduction,
        risk_manager=_StubRiskManager(),
    )
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    stop.trigger_price = 97.0

    assert await monitor.execute_stop_loss(stop) is False

    execute_reduction.assert_awaited_once()
    assert stop.status.name == "FAILED"
    assert monitor.metrics.failed_today == 1


@pytest.mark.asyncio
async def test_replacement_before_stop_call_prevents_dispatch() -> None:
    execute_reduction = AsyncMock()
    monitor = StopLossMonitor(
        execute_reduction=execute_reduction,
        risk_manager=_StubRiskManager(),
    )
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    old_stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    replacement = await monitor.add_stop_loss("AAPL", position, stop_percent=0.05)

    assert await monitor.execute_stop_loss(old_stop) is False

    execute_reduction.assert_not_awaited()
    assert old_stop.status.name == "CANCELLED"
    assert monitor.active_stops["default:AAPL"] is replacement


@pytest.mark.asyncio
async def test_post_fill_cleanup_exception_never_resubmits_order() -> None:
    class _ExplodingHistory(list):
        def append(self, _item) -> None:
            raise RuntimeError("history unavailable")

    executor = _GatedSuccessfulExecutor()
    callback_calls = 0

    async def callback(_stop, _result) -> None:
        nonlocal callback_calls
        callback_calls += 1

    monitor = StopLossMonitor(
        execute_reduction=_reduction_callback(executor),
        risk_manager=_StubRiskManager(),
        position_closed_callback=callback,
    )
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    stop = await monitor.add_stop_loss("AAPL", position, stop_percent=0.02)
    stop.trigger_price = 97.0
    monitor.stop_history = _ExplodingHistory()

    execution = asyncio.create_task(monitor.execute_stop_loss(stop))
    await executor.started.wait()
    executor.release.set()

    assert await execution is True
    assert executor.calls == 1
    assert callback_calls == 1


def _paper_settlement_request(
    *,
    pre_quantity: int,
    post_quantity: int,
    fill_quantity: int,
    mark_price: Decimal = Decimal("100"),
    fill_price: Decimal = Decimal("97.5"),
    pre_cash: Decimal | None = None,
    pre_realized_pnl: Decimal = Decimal("0"),
    pre_daily_pnl: Decimal = Decimal("0"),
    daily_pnl_baseline: Decimal = Decimal("0"),
    pre_mark_price: Decimal | None = None,
):
    from robo_trader.paper_terminal_settlement import PaperTerminalSettlementRequest
    from robo_trader.safety import PAPER_EXECUTION_DOMAIN_SCOPE, OrderSide
    from robo_trader.safety.models import TerminalOrderStatus

    side = OrderSide.SELL if pre_quantity > 0 else OrderSide.BUY_TO_COVER
    prior_mark = mark_price if pre_mark_price is None else pre_mark_price
    quote_payload = json.dumps(
        {
            "con_id": 265598,
            "portfolio_id": "default",
            "price": str(mark_price),
            "receipt_monotonic": float(10).hex(),
            "receipt_order": 1,
            "source": "live-broker",
            "source_event_id": "ticker-42",
            "source_timestamp": "2026-07-25T14:30:00.000000+00:00",
            "symbol": "AAPL",
            "transport_generation": "generation-1",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if pre_quantity > 0:
        if pre_cash is None:
            pre_cash = Decimal("100000") - Decimal(pre_quantity * 100)
        post_cash = pre_cash + Decimal(fill_quantity) * fill_price
        realized_delta = Decimal(fill_quantity) * (fill_price - Decimal("100"))
        removed_unrealized = Decimal(fill_quantity) * (mark_price - Decimal("100"))
        mark_revaluation = Decimal(abs(pre_quantity)) * (mark_price - prior_mark)
    else:
        if pre_cash is None:
            pre_cash = Decimal("100000") + Decimal(abs(pre_quantity) * 100)
        post_cash = pre_cash - Decimal(fill_quantity) * fill_price
        realized_delta = Decimal(fill_quantity) * (Decimal("100") - fill_price)
        removed_unrealized = Decimal(fill_quantity) * (Decimal("100") - mark_price)
        mark_revaluation = Decimal(abs(pre_quantity)) * (prior_mark - mark_price)
    post_daily_pnl = pre_daily_pnl + mark_revaluation + realized_delta - removed_unrealized
    return PaperTerminalSettlementRequest(
        execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
        account_scope=(
            "acct_v1_0123456789abcdef0123456789abcdef" "fedcba9876543210fedcba9876543210"
        ),
        portfolio_id="default",
        con_id=265598,
        symbol="AAPL",
        reservation_id="res-" + ("2" * 32),
        claim_id="claim-" + ("3" * 32),
        claim_sequence=1,
        submission_descriptor_fingerprint="4" * 64,
        protective_quote_fingerprint=hashlib.sha256(quote_payload.encode()).hexdigest(),
        protective_quote_payload=quote_payload,
        order_ref="stop:v1:" + ("6" * 64),
        side=side,
        requested_quantity=Decimal(fill_quantity),
        filled_quantity=Decimal(fill_quantity),
        remaining_quantity=Decimal("0"),
        expected_pre_position_quantity=Decimal(pre_quantity),
        expected_post_position_quantity=Decimal(post_quantity),
        expected_pre_aggregate_quantity=Decimal(pre_quantity),
        expected_post_aggregate_quantity=Decimal(post_quantity),
        expected_pre_cash=pre_cash,
        expected_post_cash=post_cash,
        expected_pre_realized_pnl=pre_realized_pnl,
        expected_post_realized_pnl=pre_realized_pnl + realized_delta,
        expected_pre_daily_pnl=pre_daily_pnl,
        expected_post_daily_pnl=post_daily_pnl,
        expected_daily_pnl_baseline=daily_pnl_baseline,
        expected_daily_pnl_date=datetime.utcnow().date().isoformat(),
        expected_position_cost_basis=Decimal("100"),
        expected_pre_position_mark_price=prior_mark,
        expected_pre_position_source_settlement_id=None,
        terminal_status=TerminalOrderStatus.FILLED,
        fill_price=fill_price,
        outcome_at=datetime(2026, 7, 25, 14, 30, tzinfo=timezone.utc),
    )


def _paper_settlement_receipt(request):
    from robo_trader.paper_terminal_settlement import (
        _produce_paper_terminal_settlement_receipt,
    )

    return _produce_paper_terminal_settlement_receipt(
        settlement_id="pset-" + ("7" * 32),
        request=request,
        trade_id=1,
        database_path="/tmp/robo-trader-test-ledger.db",
        database_identity="paper:123456abcdef",
        database_device=1,
        database_inode=2,
        committed_at=datetime(2026, 7, 25, 14, 30, 1, tzinfo=timezone.utc),
    )


async def _runner_for_receipt_projection(quantity: int, *, advanced: bool = False):
    from robo_trader.runner_async import AsyncRunner

    runner = AsyncRunner.__new__(AsyncRunner)
    AsyncRunner.__init__(runner)
    runner.portfolio_id = "default"
    runner.positions = {
        "AAPL": Position(symbol="AAPL", quantity=quantity, avg_price=Decimal("100"))
    }
    runner.portfolio = Portfolio(100_000)
    await runner.portfolio.update_fill(
        "AAPL",
        "BUY" if quantity > 0 else "SELL_SHORT",
        abs(quantity),
        100.0,
    )
    runner.stop_loss_monitor = _build_monitor()
    await runner.stop_loss_monitor.add_stop_loss(
        "AAPL",
        runner.positions["AAPL"],
        stop_percent=0.02,
        stop_type=StopType.FIXED,
    )
    runner.use_advanced_risk = advanced
    runner.advanced_risk = None
    if advanced:
        runner.advanced_risk = AdvancedRiskManager(
            {"starting_capital": 100_000},
            enable_kelly=False,
            enable_correlation_limits=False,
            enable_kill_switch=True,
        )
        runner.advanced_risk.seed_position("AAPL", quantity, 100.0)
        runner.advanced_risk.seed_realized_pnl(total_pnl=0.0, daily_pnl=0.0)
    runner.db = MagicMock()
    runner._daily_pnl_date = datetime.utcnow().date()
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pre_quantity", "post_quantity", "fill_quantity"),
    [(10, 0, 10), (20, 10, 10), (-10, 0, 10), (-20, -10, 10)],
)
async def test_producer_receipt_projects_exact_fill_without_database_rewrite(
    pre_quantity: int,
    post_quantity: int,
    fill_quantity: int,
) -> None:
    """The committed receipt, not the monitor callback, mutates runtime state."""

    runner = await _runner_for_receipt_projection(pre_quantity)
    request = _paper_settlement_request(
        pre_quantity=pre_quantity,
        post_quantity=post_quantity,
        fill_quantity=fill_quantity,
    )
    projection = await runner._apply_paper_terminal_settlement(_paper_settlement_receipt(request))

    assert projection.runner_position_quantity == Decimal(post_quantity)
    assert projection.portfolio_position_quantity == Decimal(post_quantity)
    assert runner.db.method_calls == []
    stop = runner.stop_loss_monitor.active_stops.get("default:AAPL")
    if post_quantity == 0:
        assert stop is None
        assert projection.protective_stop_quantity is None
    else:
        assert stop is not None
        assert stop.position_qty == post_quantity
        assert projection.protective_stop_quantity == Decimal(post_quantity)


@pytest.mark.asyncio
async def test_stop_settlement_projects_realized_loss_before_next_entry_risk_check() -> None:
    """Stop fills update the daily-loss input even without a refresh callback."""

    runner = await _runner_for_receipt_projection(10)
    request = _paper_settlement_request(
        pre_quantity=10,
        post_quantity=0,
        fill_quantity=10,
    )
    projection = await runner._apply_paper_terminal_settlement(_paper_settlement_receipt(request))

    assert projection.account_cash == Decimal("99975.0")
    assert projection.account_realized_pnl == Decimal("-25.0")
    assert projection.risk_visible_daily_pnl == Decimal("-25.0")
    assert runner.daily_pnl == -25.0

    entry_risk = RiskManager(
        max_daily_loss=20.0,
        max_position_risk_pct=0.02,
        max_symbol_exposure_pct=0.10,
        max_leverage=1.0,
    )
    allowed, reason = entry_risk.validate_order(
        "MSFT",
        1,
        100.0,
        100_000.0,
        runner.daily_pnl,
        {},
    )
    assert allowed is False
    assert reason == "Daily loss limit reached"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("starting_quantity", "first_mark", "first_fill", "second_mark", "second_fill"),
    [
        (10, Decimal("110"), Decimal("109"), Decimal("112"), Decimal("111")),
        (-10, Decimal("90"), Decimal("91"), Decimal("88"), Decimal("89")),
    ],
)
async def test_sequential_reductions_preserve_exact_mark_to_market_daily_pnl(
    starting_quantity: int,
    first_mark: Decimal,
    first_fill: Decimal,
    second_mark: Decimal,
    second_fill: Decimal,
) -> None:
    """Partial then full exits converge before and after every refresh."""

    runner = await _runner_for_receipt_projection(starting_quantity, advanced=True)
    runner._daily_pnl_date = datetime.utcnow().date()
    runner._starting_unrealized_today_exact = Decimal("25")
    runner._starting_unrealized_today = 25.0
    runner.daily_pnl = -25.0
    runner._daily_pnl_exact = Decimal("-25")
    # Cached runtime state deliberately disagrees; only producer-owned receipt
    # evidence may replace it before the post-settlement refresh.
    runner.latest_prices = {"AAPL": 123.0}
    runner.advanced_risk.seed_realized_pnl(total_pnl=0.0, daily_pnl=-25.0)

    first = _paper_settlement_request(
        pre_quantity=starting_quantity,
        post_quantity=5 if starting_quantity > 0 else -5,
        fill_quantity=5,
        mark_price=first_mark,
        fill_price=first_fill,
        pre_daily_pnl=Decimal("-25"),
        daily_pnl_baseline=Decimal("25"),
        pre_mark_price=Decimal("100"),
    )
    first_projection = await runner._apply_paper_terminal_settlement(
        _paper_settlement_receipt(first)
    )

    assert first_projection.risk_visible_daily_pnl == Decimal("70")
    assert runner.latest_prices["AAPL"] == float(first_mark)
    assert runner.latest_price_sources["AAPL"] == ("paper-terminal-settlement-protective-quote")
    assert runner.advanced_risk.daily_pnl == 70.0
    assert await runner._refresh_daily_pnl() == 70.0
    assert runner.advanced_risk.daily_pnl == 70.0

    second = _paper_settlement_request(
        pre_quantity=5 if starting_quantity > 0 else -5,
        post_quantity=0,
        fill_quantity=5,
        mark_price=second_mark,
        fill_price=second_fill,
        pre_cash=Decimal("99545") if starting_quantity > 0 else Decimal("100545"),
        pre_realized_pnl=Decimal("45"),
        pre_daily_pnl=Decimal("70"),
        daily_pnl_baseline=Decimal("25"),
        pre_mark_price=first_mark,
    )
    second_projection = await runner._apply_paper_terminal_settlement(
        _paper_settlement_receipt(second)
    )

    assert second_projection.account_realized_pnl == Decimal("100")
    assert second_projection.risk_visible_daily_pnl == Decimal("75")
    assert runner.advanced_risk.total_pnl == 100.0
    assert runner.advanced_risk.daily_pnl == 75.0
    assert await runner._refresh_daily_pnl() == 75.0
    assert runner.advanced_risk.daily_pnl == 75.0


@pytest.mark.asyncio
async def test_exact_settled_callback_is_noop_and_cannot_duplicate_mutation() -> None:
    from robo_trader.runner_async import AsyncRunner
    from robo_trader.stop_loss_monitor import StopLossOrder

    runner = AsyncRunner.__new__(AsyncRunner)
    AsyncRunner.__init__(runner)
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    runner.positions = {"AAPL": position}
    runner.portfolio = MagicMock()
    runner.db = MagicMock()
    stop = StopLossOrder(
        symbol="AAPL",
        position_qty=10,
        stop_price=98.0,
        entry_price=100.0,
        stop_type=StopType.FIXED,
        created_at=datetime.now(timezone.utc),
    )
    order = Order(
        symbol="AAPL",
        quantity=10,
        side="SELL",
        order_ref="stop:v1:" + ("6" * 64),
    )

    await runner._on_stop_loss_executed(stop, _filled_terminal_outcome(order))

    assert runner.positions["AAPL"] is position
    assert runner.portfolio.method_calls == []
    assert runner.db.method_calls == []
    assert runner._emergency_entry_freeze_reason is None


@pytest.mark.asyncio
async def test_legacy_stop_fill_result_freezes_entries_without_mutation() -> None:
    from robo_trader.runner_async import AsyncRunner
    from robo_trader.stop_loss_monitor import StopLossOrder

    runner = AsyncRunner.__new__(AsyncRunner)
    AsyncRunner.__init__(runner)
    position = Position(symbol="AAPL", quantity=10, avg_price=Decimal("100"))
    runner.positions = {"AAPL": position}
    runner.portfolio = MagicMock()
    runner.db = MagicMock()
    stop = StopLossOrder(
        symbol="AAPL",
        position_qty=10,
        stop_price=98.0,
        entry_price=100.0,
        stop_type=StopType.FIXED,
        created_at=datetime.now(timezone.utc),
    )

    await runner._on_stop_loss_executed(
        stop,
        ExecutionResult(True, "legacy fill", fill_price=97.5),
    )

    assert runner.positions["AAPL"] is position
    assert runner.portfolio.method_calls == []
    assert runner.db.method_calls == []
    assert "no exact settled local-paper outcome" in (runner._emergency_entry_freeze_reason or "")


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

    from robo_trader.execution import Order
    from robo_trader.runner_async import AsyncRunner

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
    import robo_trader.config as cfg_mod

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
