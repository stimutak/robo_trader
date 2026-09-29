"""Gateway correlation covers verified held and pending account contracts."""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.paper_reduction_gateway import PaperReductionGatewayError
from robo_trader.clients.subprocess_ibkr_client import IBKRTransportPoisonedError
from tests.risk.test_paper_entry_valuation import _gateway
from tests.risk.test_paper_ledger_snapshot import ledger  # noqa: F401
from tests.test_exact_state_bootstrap import _bootstrap_evidence_keys  # noqa: F401
from tests.risk.test_pending_entry_capacity import append
from tests.canonical_batch_test_support import bind_test_canonical_batch
from tests.risk.test_canonical_correlation import batch as make_batch


def setup(ledger, monkeypatch):
    gateway, quotes, _, _ = _gateway(ledger, monkeypatch)
    # Explicit test transport ownership, with the real gateway quote fixture.
    batches = []
    for symbol, con_id in (("AAPL", 265598), ("NVDA", 123), ("TSLA", 456)):
        batch = make_batch(symbol, con_id)
        delta = quotes["AAPL"].retrieval_timestamp - batch.contract.retrieval_time
        contract = replace(
            batch.contract,
            retrieval_time=batch.contract.retrieval_time + delta,
            broker_time=batch.contract.broker_time + delta,
            primary_exchange=quotes[symbol].primary_exchange,
        )
        batch = replace(
            batch,
            contract=contract,
            bars=tuple(
                replace(bar, contract=contract, timestamp=bar.timestamp + delta)
                for bar in batch.bars
            ),
        )
        holder = SimpleNamespace()
        bind_test_canonical_batch(holder, batch)
        if not batches:
            client = holder.ib
        else:
            client._cache_historical_lineage(
                client._generation, holder.ib.get_cached_historical_lineage(symbol)
            )
            client._check_canonical_batch(batch, publishing_generation=client._generation)
        batches.append(batch)
    original = gateway._client
    client.ping = original.ping
    client.get_protective_quotes = original.get_protective_quotes
    client.stop = AsyncMock()
    gateway._client = client
    return gateway, tuple(batches)


@pytest.mark.asyncio
async def test_gateway_authenticates_complete_account_window(ledger, monkeypatch):
    gateway, batches = setup(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        result = await gateway.entry_correlation(
            portfolio_id="default", batches=batches, return_count=3
        )
    assert result.compared_symbols == ("NVDA", "TSLA")
    assert result.max_absolute_correlation == Decimal("1")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "copied", "pending", "head", "generation"])
async def test_gateway_rejects_incomplete_or_unowned_sources(ledger, monkeypatch, failure):
    gateway, batches = setup(ledger, monkeypatch)
    if failure == "pending":
        append(gateway._coordinator._journal, portfolio="default", symbol="MSFT", con_id=272093)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        if failure == "missing":
            batches = batches[:-1]
        elif failure == "copied":
            batches = (replace(batches[0]),) + batches[1:]
        elif failure == "head":
            append(gateway._coordinator._journal, portfolio="default")
        elif failure == "generation":
            gateway._client._generation.poisoned_reason = "test disconnect"
        with pytest.raises((PaperReductionGatewayError, IBKRTransportPoisonedError)):
            await gateway.entry_correlation(portfolio_id="default", batches=batches, return_count=3)


@pytest.mark.asyncio
async def test_correlation_rechecks_ownership_after_journal_await(ledger, monkeypatch):
    gateway, batches = setup(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        original = gateway._assert_entry_journal_current
        calls = 0

        async def replay(context):
            nonlocal calls
            state = await original(context)
            calls += 1
            if calls == 2:
                gateway._client._invalidate_historical_lineage()
            return state

        monkeypatch.setattr(gateway, "_assert_entry_journal_current", replay)
        with pytest.raises(PaperReductionGatewayError):
            await gateway.entry_correlation(portfolio_id="default", batches=batches, return_count=3)


@pytest.mark.asyncio
async def test_correlation_read_is_task_owned(ledger, monkeypatch):
    import asyncio

    gateway, batches = setup(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        with pytest.raises(PaperReductionGatewayError, match="owning entry"):
            await asyncio.create_task(
                gateway.entry_correlation(portfolio_id="default", batches=batches, return_count=3)
            )


@pytest.mark.asyncio
async def test_history_expiring_during_final_await_is_rejected(ledger, monkeypatch):
    from datetime import timedelta
    import robo_trader.paper_reduction_gateway as module

    gateway, batches = setup(ledger, monkeypatch)
    shifted = []
    for batch in batches:
        delta = timedelta(seconds=119)
        contract = replace(
            batch.contract,
            retrieval_time=batch.contract.retrieval_time - delta,
            broker_time=batch.contract.broker_time - delta,
        )
        older = replace(
            batch,
            contract=contract,
            bars=tuple(
                replace(bar, contract=contract, timestamp=bar.timestamp - delta)
                for bar in batch.bars
            ),
        )
        holder = SimpleNamespace()
        bind_test_canonical_batch(holder, older)
        gateway._client._cache_historical_lineage(
            gateway._client._generation, holder.ib.get_cached_historical_lineage(contract.symbol)
        )
        gateway._client._check_canonical_batch(
            older, publishing_generation=gateway._client._generation
        )
        shifted.append(older)
    original_clock = module.datetime
    offset = timedelta(0)

    class Clock(original_clock):
        @classmethod
        def now(cls, tz=None):
            return original_clock.now(tz) + offset

    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        original = gateway._assert_entry_journal_current
        calls = 0

        async def replay(context):
            nonlocal calls, offset
            state = await original(context)
            calls += 1
            if calls == 2:
                offset = timedelta(seconds=2)
            return state

        monkeypatch.setattr(module, "datetime", Clock)
        monkeypatch.setattr(gateway, "_assert_entry_journal_current", replay)
        with pytest.raises(PaperReductionGatewayError, match="correlation.*stale"):
            await gateway.entry_correlation(
                portfolio_id="default", batches=tuple(shifted), return_count=3
            )


@pytest.mark.asyncio
async def test_pending_contract_is_included_in_successful_comparison(ledger, monkeypatch):
    gateway, batches = setup(ledger, monkeypatch)
    append(gateway._coordinator._journal, portfolio="default", symbol="MSFT", con_id=272093)
    contract = replace(batches[0].contract, symbol="MSFT", con_id=272093, primary_exchange="NASDAQ")
    pending = replace(
        batches[0],
        contract=contract,
        bars=tuple(replace(bar, contract=contract) for bar in batches[0].bars),
    )
    holder = SimpleNamespace()
    bind_test_canonical_batch(holder, pending)
    gateway._client._cache_historical_lineage(
        gateway._client._generation, holder.ib.get_cached_historical_lineage("MSFT")
    )
    gateway._client._check_canonical_batch(
        pending, publishing_generation=gateway._client._generation
    )
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        result = await gateway.entry_correlation(
            portfolio_id="default", batches=batches + (pending,), return_count=3
        )
    assert result.compared_symbols == ("MSFT", "NVDA", "TSLA")
    assert result.max_absolute_correlation == Decimal("1")
