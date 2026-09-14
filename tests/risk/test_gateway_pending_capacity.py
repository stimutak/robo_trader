"""Gateway pending exposure is task-owned and tied to an unchanged journal."""

import asyncio
from decimal import Decimal

import pytest

from robo_trader.paper_reduction_gateway import PaperReductionGatewayError
from tests.risk.test_paper_entry_valuation import _gateway
from tests.risk.test_paper_ledger_snapshot import ledger  # noqa: F401
from tests.test_exact_state_bootstrap import _bootstrap_evidence_keys  # noqa: F401
from tests.risk.test_pending_entry_capacity import append


@pytest.mark.asyncio
async def test_gateway_pending_capacity_uses_real_journal_and_counts_held_union(
    ledger, monkeypatch
):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    append(gateway._coordinator._journal, portfolio="default")
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        result = await gateway.entry_pending_exposure(portfolio_id="default", sector="Technology")
        assert result.pending_account_notional_usd == Decimal("1998")
        assert result.account_occupied_position_slots == 3
        assert result.symbol_has_position_or_pending_entry is True


@pytest.mark.asyncio
async def test_gateway_pending_read_rejects_foreign_task(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):

        async def other():
            with pytest.raises(PaperReductionGatewayError, match="entry context"):
                await gateway.entry_pending_exposure(portfolio_id="default", sector="Technology")

        await asyncio.create_task(other())


@pytest.mark.asyncio
async def test_gateway_pending_read_rejects_journal_change_after_context_started(
    ledger, monkeypatch
):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        append(gateway._coordinator._journal, portfolio="default")
        with pytest.raises(PaperReductionGatewayError, match="journal.*changed"):
            await gateway.entry_pending_exposure(portfolio_id="default", sector="Technology")


@pytest.mark.asyncio
async def test_gateway_rejects_reservation_for_portfolio_missing_from_ledger(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    append(gateway._coordinator._journal, portfolio="unknown")
    with pytest.raises(PaperReductionGatewayError, match="portfolio.*coverage"):
        async with gateway.serialize_entry("AAPL", portfolio_id="default"):
            pytest.fail("uncovered pending portfolio admitted")


@pytest.mark.asyncio
async def test_journal_change_during_quotes_rejects_context_before_yield(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    original = gateway._client.get_protective_quotes

    async def fetch(*args, **kwargs):
        append(gateway._coordinator._journal, portfolio="default")
        return await original(*args, **kwargs)

    gateway._client.get_protective_quotes = fetch
    with pytest.raises(PaperReductionGatewayError, match="journal.*changed"):
        async with gateway.serialize_entry("AAPL", portfolio_id="default"):
            pytest.fail("changed head admitted")
    assert gateway._entry_market_context is None


@pytest.mark.asyncio
async def test_pending_and_held_same_symbol_share_one_position_slot(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    append(gateway._coordinator._journal, portfolio="default", symbol="NVDA", con_id=123)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        result = await gateway.entry_pending_exposure(portfolio_id="default", sector="Technology")
        assert result.account_occupied_position_slots == 2
        assert result.symbol_has_position_or_pending_entry is False


@pytest.mark.asyncio
async def test_cancellation_drains_journal_read_before_releasing_account_gate(ledger, monkeypatch):
    import threading
    from robo_trader.safety import SafetyRuntimeCoordinator

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    original = SafetyRuntimeCoordinator.replay_for_entry

    def blocked(coordinator):
        entered.set()
        assert release.wait(5)
        return original(coordinator)

    monkeypatch.setattr(SafetyRuntimeCoordinator, "replay_for_entry", blocked)

    async def enter():
        async with gateway.serialize_entry("AAPL", portfolio_id="default"):
            pytest.fail("cancelled context admitted")

    task = asyncio.create_task(enter())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert gateway._account_order_gate.locked()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not gateway._account_order_gate.locked()
    assert getattr(gateway, "_entry_market_context", None) is None


@pytest.mark.asyncio
async def test_entry_cost_uses_registered_executor_and_current_slippage(ledger, monkeypatch):
    gateway, quotes, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        initial = gateway.entry_execution_cost(portfolio_id="default")
        assert initial.reference_price_usd == quotes["AAPL"].price
        assert initial.commission_minor == 0
        executor = gateway._bindings["default"].executor
        executor.slippage_bps = 25.0
        updated = gateway.entry_execution_cost(portfolio_id="default")
        assert updated.price_ceiling_usd > initial.price_ceiling_usd
        assert updated.slippage_bps == Decimal("25.0")
        executor.slippage_bps = float("nan")
        with pytest.raises(PaperReductionGatewayError, match="slippage"):
            gateway.entry_execution_cost(portfolio_id="default")
    with pytest.raises(PaperReductionGatewayError, match="entry context"):
        gateway.entry_execution_cost(portfolio_id="default")
