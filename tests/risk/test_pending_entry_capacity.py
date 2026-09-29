"""Pending capacity includes durable reservations across all account portfolios."""

import uuid
from decimal import Decimal, localcontext

import pytest

from robo_trader.risk import entry_reservations as module
from robo_trader.safety import SafetyJournal
from tests.safety.conftest import ACCOUNT_A
from tests.test_pr7_entry_risk_contract import (
    NOW,
    _contract,
    _correlation,
    _evaluate,
    _evidence,
    _intent,
    _limits,
    _liquidity,
    _quote,
)


def append(
    journal, *, portfolio="alpha", symbol="AAPL", con_id=265598, sector="Technology", cap="2000"
):
    contract = _contract(symbol=symbol, local_symbol=symbol, con_id=con_id)
    scope = dict(portfolio_id=portfolio, symbol=symbol, broker_contract=contract)
    intent = _intent(intent_id="pending-" + uuid.uuid4().hex, **scope)
    evidence = _evidence(
        portfolio_id=portfolio,
        symbol=symbol,
        sector=sector,
        quote=_quote(broker_contract=contract),
        correlation=_correlation(**scope),
        liquidity=_liquidity(**scope),
    )
    decision = _evaluate(
        intent=intent,
        evidence=evidence,
        limits=_limits(max_order_notional_usd=Decimal(cap)),
        expected_broker_contract=contract,
    )
    assert decision.risk_approved
    head = journal.replay()
    return module.reserve_entry_capacity(
        journal, decision, sector=sector, expected_head=(head.last_sequence, head.last_chain_hash)
    )


def journal_at(tmp_path):
    journal = SafetyJournal(tmp_path / "journal.db", clock=lambda: NOW)
    journal.initialize(execution_domain_scope="paper-domain", account_scope=ACCOUNT_A)
    return journal


def test_pending_aggregation_is_exact_and_account_wide(tmp_path):
    journal = journal_at(tmp_path)
    append(journal)
    append(journal, portfolio="inactive", symbol="MSFT", con_id=272093, cap="1000")
    append(journal, symbol="PFE", con_id=12345, sector="Healthcare", cap="1500")
    state = journal.replay()
    with localcontext() as context:
        context.prec = 2
        result = module.summarize_entry_capacity(
            state,
            portfolio_id="alpha",
            symbol="AAPL",
            sector="Technology",
            held_symbols=("NVDA", "MSFT"),
        )
    assert result.pending_symbol_notional_usd == Decimal("1998")
    assert result.pending_sector_notional_usd == Decimal("2997")
    assert result.pending_portfolio_notional_usd == Decimal("3330")
    assert result.pending_account_notional_usd == Decimal("4329")
    assert result.pending_cash_usd == Decimal("3330")
    assert result.pending_buying_power_usd == Decimal("4329")
    assert result.pending_daily_notional_usd == Decimal("3330")
    assert result.account_occupied_position_slots == 4
    assert result.symbol_has_position_or_pending_entry is True
    assert result.journal_head == (state.last_sequence, state.last_chain_hash)


def test_empty_verified_journal_retains_held_positions(tmp_path):
    result = module.summarize_entry_capacity(
        journal_at(tmp_path).replay(),
        portfolio_id="alpha",
        symbol="NVDA",
        sector="Technology",
        held_symbols=("NVDA", "NVDA"),
    )
    assert result.pending_account_notional_usd == Decimal("0")
    assert result.account_occupied_position_slots == 1
    assert result.symbol_has_position_or_pending_entry is True
