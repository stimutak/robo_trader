"""Gateway entry valuation with real synthetic ledger and quote producers."""

import asyncio
from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.paper_reduction_gateway import (
    PaperReductionGateway,
    PaperReductionGatewayError,
    _PaperRuntimeBinding,
)
from robo_trader.risk.paper_ledger_snapshot import PaperRiskLedgerSnapshotError
from robo_trader.safety import readiness
from robo_trader.stop_loss_monitor import StopLossMonitor
from tests.risk.test_paper_ledger_snapshot import ledger  # noqa: F401
from tests.test_exact_state_bootstrap import _bootstrap_evidence_keys  # noqa: F401
from tests.test_pr3_protective_quote_channel import _broker_quote


def _gateway(ledger, monkeypatch):
    database, runtime, _ = ledger
    monkeypatch.setattr(readiness, "PAPER_TERMINAL_SETTLEMENT_READY", True)
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(), risk_manager=None, portfolio_id="default"
    )
    quotes = {
        "AAPL": _broker_quote(),
        "NVDA": _broker_quote(symbol="NVDA", con_id=123, price=Decimal("330")),
        "TSLA": _broker_quote(symbol="TSLA", con_id=456, price=Decimal("250")),
    }
    requests = []

    async def fetch(symbols, *, active_symbols):
        requests.append(tuple(symbols))
        return tuple(quotes[symbol] for symbol in symbols)

    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._database = database
    gateway._runtime_context = SimpleNamespace(runtime_contract=runtime)
    from robo_trader.safety import PaperExecutionIdentity, SafetyJournal, SafetyRuntimeCoordinator
    from tests.test_pr7_entry_risk_contract import NOW

    journal = SafetyJournal(runtime.safety_journal_path, clock=lambda: NOW)
    journal.initialize(
        execution_domain_scope=runtime.safety_execution_domain_scope,
        account_scope=runtime.safety_account_scope,
    )
    gateway._coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(runtime.safety_execution_domain_scope, runtime.safety_account_scope),
        journal,
    )
    gateway._coordinator.start()
    gateway._client = SimpleNamespace(
        protective_quote_generation="generation-1",
        ping=AsyncMock(return_value=True),
        get_protective_quotes=fetch,
        stop=AsyncMock(),
    )
    gateway._protective_quote_producers = {"default": monitor}
    from robo_trader.execution import PaperExecutor

    gateway._bindings = {
        "default": _PaperRuntimeBinding(None, None, monitor, None, PaperExecutor())
    }
    return gateway, quotes, requests, monitor


@pytest.mark.asyncio
async def test_entry_values_every_ledger_position_under_shared_lock(ledger, monkeypatch):
    gateway, _, requests, _ = _gateway(ledger, monkeypatch)
    cash = ledger[2].account.cash
    with localcontext() as context:
        context.prec = 3
        async with gateway.serialize_entry("AAPL", portfolio_id="default") as quote:
            assert quote.symbol == "AAPL"
            valuation = gateway.entry_valuation(portfolio_id="default")
            assert valuation.portfolio_cash_usd == cash
            assert valuation.portfolio_equity_usd == Decimal("100209.16")
            assert valuation.account_equity_usd == Decimal("100209.16")
            assert valuation.account_gross_notional_usd == Decimal("3470")
            assert valuation.portfolio_gross_notional_usd == Decimal("3470")
            assert valuation.account_occupied_position_slots == 2
            assert gateway._account_order_gate.locked()
    assert requests == [("AAPL", "NVDA", "TSLA")]
    with pytest.raises(PaperReductionGatewayError, match="entry context"):
        gateway.entry_valuation(portfolio_id="default")


@pytest.mark.asyncio
async def test_entry_valuation_is_task_owned(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):

        async def foreign():
            with pytest.raises(PaperReductionGatewayError, match="entry context"):
                gateway.entry_valuation(portfolio_id="default")

        await asyncio.create_task(foreign())


@pytest.mark.asyncio
async def test_entry_rejects_mismatched_held_contract_before_yield(ledger, monkeypatch):
    gateway, quotes, _, _ = _gateway(ledger, monkeypatch)
    quotes["NVDA"] = _broker_quote(symbol="NVDA", con_id=999)
    with pytest.raises(PaperReductionGatewayError, match="contract"):
        async with gateway.serialize_entry("AAPL", portfolio_id="default"):
            pytest.fail("mismatched ledger contract admitted")


@pytest.mark.asyncio
async def test_entry_rejects_ledger_rewrite_before_requesting_quotes(ledger, monkeypatch):
    gateway, _, requests, _ = _gateway(ledger, monkeypatch)
    async with ledger[0].get_connection() as connection:
        await connection.execute("UPDATE paper_account_settlement_state SET cash_text='1'")
        await connection.commit()
    with pytest.raises(PaperRiskLedgerSnapshotError, match="cash"):
        async with gateway.serialize_entry("AAPL", portfolio_id="default"):
            pytest.fail("rewritten ledger admitted")
    assert requests == []


@pytest.mark.asyncio
async def test_entry_valuation_rechecks_generation_and_quote_freshness(ledger, monkeypatch):
    from datetime import datetime, timedelta, timezone

    gateway, _, _, monitor = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        gateway._client.protective_quote_generation = "replacement"
        with pytest.raises(PaperReductionGatewayError, match="generation"):
            gateway.entry_valuation(portfolio_id="default")
        gateway._client.protective_quote_generation = "generation-1"
        monitor._utcnow = lambda: datetime.now(timezone.utc) + timedelta(minutes=10)
        with pytest.raises(PaperReductionGatewayError, match="authority"):
            gateway.entry_valuation(portfolio_id="default")


@pytest.mark.asyncio
async def test_entry_valuation_rejects_quote_mutation(ledger, monkeypatch):
    gateway, quotes, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        object.__setattr__(quotes["NVDA"], "price", Decimal("1"))
        with pytest.raises(PaperReductionGatewayError, match="differs"):
            gateway.entry_valuation(portfolio_id="default")


@pytest.mark.asyncio
async def test_entry_valuation_rejects_duplicate_quote_records(ledger, monkeypatch):
    gateway, quotes, _, _ = _gateway(ledger, monkeypatch)

    async def fetch(symbols, *, active_symbols):
        return tuple(quotes[symbol] for symbol in symbols) + (quotes["NVDA"],)

    gateway._client.get_protective_quotes = fetch
    with pytest.raises(PaperReductionGatewayError, match="ambiguous"):
        async with gateway.serialize_entry("AAPL", portfolio_id="default"):
            pytest.fail("duplicate quote admitted")
    assert gateway._entry_market_context is None


@pytest.mark.asyncio
async def test_entry_context_is_cleared_on_cancellation(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    entered = asyncio.Event()

    async def hold():
        async with gateway.serialize_entry("AAPL", portfolio_id="default"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gateway._entry_market_context is None
    assert not gateway._account_order_gate.locked()
    with pytest.raises(PaperReductionGatewayError, match="entry context"):
        gateway.entry_valuation(portfolio_id="default")


@pytest.mark.asyncio
async def test_entry_context_rejects_unregistered_account_portfolio(ledger, monkeypatch):
    gateway, _, requests, _ = _gateway(ledger, monkeypatch)
    gateway._bindings["foreign"] = gateway._bindings["default"]
    with pytest.raises(PaperReductionGatewayError, match="coverage"):
        async with gateway.serialize_entry("AAPL", portfolio_id="foreign"):
            pytest.fail("unbootstrapped portfolio admitted")
    assert requests == []


@pytest.mark.asyncio
async def test_entry_valuation_rejects_expired_ledger_read(ledger, monkeypatch):
    from datetime import datetime, timedelta, timezone

    import robo_trader.paper_reduction_gateway as module

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        future = datetime.now(timezone.utc) + timedelta(seconds=6)

        class Clock:
            @staticmethod
            def now(tz):
                return future

        monkeypatch.setattr(module, "datetime", Clock)
        with pytest.raises(PaperReductionGatewayError, match="ledger evidence is stale"):
            gateway.entry_valuation(portfolio_id="default")


@pytest.mark.asyncio
async def test_inactive_cash_and_short_positions_remain_in_account_valuation(ledger, monkeypatch):
    from dataclasses import fields

    import robo_trader.paper_reduction_gateway as module
    from robo_trader.risk.paper_ledger_snapshot import (
        PaperRiskPosition,
        _issue_snapshot,
        collect_paper_risk_ledger_snapshot,
    )

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    original = await collect_paper_risk_ledger_snapshot(ledger[0], ledger[1])
    # Test-only producer issuance isolates multi-portfolio valuation from the
    # independent bootstrap receipt construction tests. No runtime bypass.
    values = {field.name: getattr(original, field.name) for field in fields(original)}
    values["portfolio_cash"] += (("retired", Decimal("1000")),)
    values["bootstrap_effective_at"] += (("retired", original.bootstrap_effective_at[0][1]),)
    values["positions"] += (PaperRiskPosition("retired", "NVDA", 123, Decimal("-3")),)
    snapshot = _issue_snapshot(**values)
    monkeypatch.setattr(
        module, "collect_paper_risk_ledger_snapshot", AsyncMock(return_value=snapshot)
    )
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        result = gateway.entry_valuation(portfolio_id="default")
        assert result.portfolio_equity_usd == Decimal("100209.16")
        assert result.account_equity_usd == Decimal("100219.16")
        assert result.account_gross_notional_usd == Decimal("4460")
        assert result.account_occupied_position_slots == 2


@pytest.mark.asyncio
async def test_entry_ledger_age_cannot_be_hidden_by_wall_clock_shift(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    clock = [100.0]
    gateway._monotonic = lambda: clock[0]
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        clock[0] += 6.0
        with pytest.raises(PaperReductionGatewayError, match="ledger evidence is stale"):
            gateway.entry_valuation(portfolio_id="default")


@pytest.mark.asyncio
async def test_gateway_prepares_replay_before_entry_daily_notional(ledger, monkeypatch):
    from datetime import datetime, timezone

    from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
    from tests.risk.test_daily_filled_notional import MutableClock, _service
    from tests.risk.test_paper_ledger_snapshot import _settle

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    await _settle(ledger)
    runtime = ledger[1]
    risk = _service(
        ledger[0].db_path.parent / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        clock=MutableClock(datetime.now(timezone.utc)),
    )
    accounting = PaperFillAccounting(runtime, {"default": risk})
    replay = await gateway.prepare_entry_accounting(accounting)
    assert replay.receipts_seen == replay.fills_recorded == 1
    assert await accounting.current_total("default") == Decimal("660")
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        with pytest.raises(PaperReductionGatewayError, match="daily history"):
            await gateway.entry_daily_notional(portfolio_id="default")
    replay = await gateway.prepare_entry_accounting(accounting)
    assert replay.fills_recorded == 0


@pytest.mark.asyncio
async def test_entry_daily_notional_requires_completed_replay(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        with pytest.raises(PaperReductionGatewayError, match="accounting"):
            await gateway.entry_daily_notional(portfolio_id="default")


@pytest.mark.asyncio
async def test_empty_replay_still_requires_independent_authority(ledger, monkeypatch):
    from robo_trader.risk.filled_notional import FilledNotionalUnavailable
    from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
    from tests.risk.test_daily_filled_notional import _service

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    runtime = ledger[1]
    risk = _service(
        ledger[0].db_path.parent / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
    )
    accounting = PaperFillAccounting(runtime, {"default": risk})
    risk._monotonic_verifier = lambda _: False
    with pytest.raises(FilledNotionalUnavailable, match="monotonic"):
        await gateway.prepare_entry_accounting(accounting)
    assert gateway._entry_accounting_ready is False


@pytest.mark.asyncio
async def test_cancelled_replay_cannot_leave_accounting_ready(ledger, monkeypatch):
    from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
    from tests.risk.test_daily_filled_notional import _service

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    runtime = ledger[1]
    risk = _service(
        ledger[0].db_path.parent / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
    )
    accounting = PaperFillAccounting(runtime, {"default": risk})
    await gateway.prepare_entry_accounting(accounting)
    monkeypatch.setattr(accounting, "replay", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await gateway.prepare_entry_accounting(accounting)
    assert gateway._entry_accounting_ready is False
    assert not gateway._account_order_gate.locked()


@pytest.mark.asyncio
async def test_bootstrap_day_replay_does_not_prove_complete_daily_history(ledger, monkeypatch):
    from datetime import datetime, timezone

    from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
    from tests.risk.test_daily_filled_notional import MutableClock, _service

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    runtime = ledger[1]
    risk = _service(
        ledger[0].db_path.parent / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        clock=MutableClock(datetime.now(timezone.utc)),
    )
    await gateway.prepare_entry_accounting(PaperFillAccounting(runtime, {"default": risk}))
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        with pytest.raises(PaperReductionGatewayError, match="daily history"):
            await gateway.entry_daily_notional(portfolio_id="default")


@pytest.mark.asyncio
async def test_complete_day_total_is_bound_to_gateway_read_time(ledger, monkeypatch):
    from dataclasses import fields
    from datetime import datetime, timedelta, timezone

    import robo_trader.paper_reduction_gateway as module
    from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
    from robo_trader.risk.paper_ledger_snapshot import (
        _issue_snapshot,
        collect_paper_risk_ledger_snapshot,
    )
    from tests.risk.test_daily_filled_notional import MutableClock, _service
    from tests.risk.test_paper_ledger_snapshot import _settle

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    await _settle(ledger)
    original = await collect_paper_risk_ledger_snapshot(ledger[0], ledger[1])
    values = {field.name: getattr(original, field.name) for field in fields(original)}
    # Test-only issuer supplies a prior-day history boundary; actual collection
    # gets this field only from the authenticated bootstrap candidate.
    values["bootstrap_effective_at"] = (("default", original.observed_at - timedelta(days=2)),)
    snapshot = _issue_snapshot(**values)
    monkeypatch.setattr(
        module, "collect_paper_risk_ledger_snapshot", AsyncMock(return_value=snapshot)
    )
    risk = _service(
        ledger[0].db_path.parent / "risk.db",
        account_id="paper-simulator-v1:" + ledger[1].safety_account_scope,
        clock=MutableClock(datetime.now(timezone.utc)),
    )
    await gateway.prepare_entry_accounting(PaperFillAccounting(ledger[1], {"default": risk}))
    # A different ledger default date must not silently change the entry day's
    # total. Gateway selects one explicit timestamp, then rechecks its date.
    risk._clock = lambda: datetime.now(timezone.utc) - timedelta(days=1)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        assert await gateway.entry_daily_notional(portfolio_id="default") == Decimal("660")


@pytest.mark.asyncio
async def test_entry_admission_history_comes_from_verified_account_snapshot(ledger, monkeypatch):
    from datetime import timedelta

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        state = gateway.entry_valuation(portfolio_id="default")
        assert state.symbol == "AAPL"
        assert state.current_symbol_gross_notional_usd == Decimal("0")
        assert state.symbol_has_position is False
        assert state.symbol_entry_allowed_at == ledger[2].effective_at + timedelta(minutes=10)
    async with gateway.serialize_entry("NVDA", portfolio_id="default"):
        state = gateway.entry_valuation(portfolio_id="default")
        assert state.symbol == "NVDA"
        assert state.current_symbol_gross_notional_usd == Decimal("2970")
        assert state.symbol_has_position is True


@pytest.mark.asyncio
async def test_nonzero_terminal_fill_extends_symbol_cooldown(ledger, monkeypatch):
    from datetime import timedelta

    from tests.risk.test_paper_ledger_snapshot import _settle

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    receipt = await _settle(ledger)
    async with gateway.serialize_entry("NVDA", portfolio_id="default"):
        state = gateway.entry_valuation(portfolio_id="default")
        assert state.current_symbol_gross_notional_usd == Decimal("2310")
        assert state.symbol_entry_allowed_at == receipt.request.outcome_at + timedelta(minutes=10)


@pytest.mark.asyncio
async def test_rejected_zero_fill_does_not_extend_cooldown(ledger, monkeypatch):
    from datetime import timedelta

    from tests.risk.test_paper_ledger_snapshot import _settle

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    await _settle(ledger, filled=False)
    async with gateway.serialize_entry("NVDA", portfolio_id="default"):
        state = gateway.entry_valuation(portfolio_id="default")
        assert state.symbol_entry_allowed_at == ledger[2].effective_at + timedelta(minutes=10)
        assert state.current_symbol_gross_notional_usd == Decimal("2970")


@pytest.mark.asyncio
async def test_symbol_exposure_and_cooldown_include_other_portfolios(ledger, monkeypatch):
    from dataclasses import fields
    from datetime import timedelta

    import robo_trader.paper_reduction_gateway as module
    from robo_trader.risk.paper_ledger_snapshot import (
        PaperRiskFill,
        PaperRiskPosition,
        _issue_snapshot,
        collect_paper_risk_ledger_snapshot,
    )

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    original = await collect_paper_risk_ledger_snapshot(ledger[0], ledger[1])
    values = {field.name: getattr(original, field.name) for field in fields(original)}
    older = original.observed_at - timedelta(minutes=20)
    latest = original.observed_at - timedelta(minutes=1)
    # Test-only owned account evidence: the retired portfolio remains part of
    # duplicate/exposure/cooldown state even without an active runtime binding.
    values["portfolio_cash"] += (("retired", Decimal("1000")),)
    values["bootstrap_effective_at"] = (("default", older), ("retired", older))
    values["positions"] += (PaperRiskPosition("retired", "NVDA", 123, Decimal("-3")),)
    values["fills"] = (PaperRiskFill("retired", "NVDA", "lpfill-" + "1" * 32, latest),)
    snapshot = _issue_snapshot(**values)
    monkeypatch.setattr(
        module, "collect_paper_risk_ledger_snapshot", AsyncMock(return_value=snapshot)
    )
    async with gateway.serialize_entry("NVDA", portfolio_id="default"):
        state = gateway.entry_valuation(portfolio_id="default")
        assert state.symbol_has_position is True
        assert state.current_symbol_gross_notional_usd == Decimal("3960")
        assert state.symbol_entry_allowed_at == latest + timedelta(minutes=10)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        state = gateway.entry_valuation(portfolio_id="default")
        assert state.symbol_has_position is False
        assert state.symbol_entry_allowed_at == older + timedelta(minutes=10)
