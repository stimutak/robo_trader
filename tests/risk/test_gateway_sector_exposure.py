"""Sector evidence requires a complete immutable classification policy."""

import asyncio
from decimal import Decimal, localcontext

import pytest

from robo_trader.paper_reduction_gateway import PaperReductionGatewayError
from tests.risk.test_paper_entry_valuation import _gateway
from tests.risk.test_paper_ledger_snapshot import ledger  # noqa: F401
from tests.test_exact_state_bootstrap import _bootstrap_evidence_keys  # noqa: F401
from tests.risk.test_pending_entry_capacity import append

POLICY = (("AAPL", "Technology"), ("NVDA", "Technology"), ("TSLA", "Consumer"))


@pytest.mark.asyncio
async def test_sector_marks_and_pending_are_exact(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    append(gateway._coordinator._journal, portfolio="default")
    with localcontext() as context:
        context.prec = 2
        async with gateway.serialize_entry("AAPL", portfolio_id="default"):
            result = await gateway.entry_sector_exposure(
                portfolio_id="default", classifications=POLICY
            )
    assert result.sector == "Technology"
    assert result.current_sector_gross_notional_usd == Decimal("2970")
    assert result.pending_sector_notional_usd == Decimal("1998")
    assert result.classifications == tuple(sorted(POLICY))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy",
    [POLICY[:-1], POLICY + (POLICY[0],), (("AAPL", "Unknown"),) + POLICY[1:], dict(POLICY)],
)
async def test_incomplete_or_ambiguous_classifications_rejected(ledger, monkeypatch, policy):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        with pytest.raises(PaperReductionGatewayError, match="classification"):
            await gateway.entry_sector_exposure(portfolio_id="default", classifications=policy)


@pytest.mark.asyncio
async def test_pending_sector_cannot_be_reclassified(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    append(gateway._coordinator._journal, portfolio="default", sector="Healthcare")
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        with pytest.raises(PaperReductionGatewayError, match="classification"):
            await gateway.entry_sector_exposure(portfolio_id="default", classifications=POLICY)


@pytest.mark.asyncio
async def test_sector_read_rechecks_journal_and_task_ownership(ledger, monkeypatch):
    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        with pytest.raises(PaperReductionGatewayError, match="owning entry"):
            await asyncio.create_task(
                gateway.entry_sector_exposure(portfolio_id="default", classifications=POLICY)
            )
        append(gateway._coordinator._journal, portfolio="default")
        with pytest.raises(PaperReductionGatewayError, match="journal changed"):
            await gateway.entry_sector_exposure(portfolio_id="default", classifications=POLICY)


@pytest.mark.asyncio
async def test_sector_includes_inactive_short_without_netting(ledger, monkeypatch):
    from dataclasses import fields
    from unittest.mock import AsyncMock
    import robo_trader.paper_reduction_gateway as module
    from robo_trader.risk.paper_ledger_snapshot import (
        PaperRiskPosition,
        _issue_snapshot,
        collect_paper_risk_ledger_snapshot,
    )

    gateway, _, _, _ = _gateway(ledger, monkeypatch)
    original = await collect_paper_risk_ledger_snapshot(ledger[0], ledger[1])
    values = {field.name: getattr(original, field.name) for field in fields(original)}
    values["portfolio_cash"] += (("retired", Decimal("1000")),)
    values["bootstrap_effective_at"] += (("retired", original.observed_at),)
    values["positions"] += (PaperRiskPosition("retired", "NVDA", 123, Decimal("-3")),)
    monkeypatch.setattr(
        module,
        "collect_paper_risk_ledger_snapshot",
        AsyncMock(return_value=_issue_snapshot(**values)),
    )
    append(gateway._coordinator._journal, portfolio="retired")
    async with gateway.serialize_entry("AAPL", portfolio_id="default"):
        result = await gateway.entry_sector_exposure(portfolio_id="default", classifications=POLICY)
    assert result.current_sector_gross_notional_usd == Decimal("3960")
    assert result.pending_sector_notional_usd == Decimal("1998")
