"""Owned decision adapter for dormant, durable entry capacity reservations."""

import json
from dataclasses import dataclass
from decimal import Decimal

from robo_trader.safety.entry_capacity import validate_entry_capacity_event
from robo_trader.safety.journal import ReservationConflict, SafetyJournal
from robo_trader.safety.models import ReplayState, parse_fixed_decimal, utc_to_text

from .entry_contract import (
    EntrySide,
    _exact_subtract,
    _identifier,
    _sector,
    _symbol,
    assert_and_consume_risk_decision,
)


def reserve_entry_capacity(journal, decision, *, sector, expected_head):
    """Consume a decision in the journal transaction; never grant a submit permit."""
    if type(journal) is not SafetyJournal:
        raise ValueError("entry capacity requires an exact SafetyJournal")
    _sector(sector)

    def produce_payload(at):
        consumed = assert_and_consume_risk_decision(decision, consumed_at=at)
        if not consumed.risk_approved or consumed.side is not EntrySide.BUY:
            raise ReservationConflict("entry capacity requires an approved BUY decision")
        return consumed.portfolio_id, {
            "version": 1,
            "intent_id": consumed.intent_id,
            "signal_id": consumed.signal_id,
            "symbol": consumed.symbol,
            "sector": sector,
            "quantity": consumed.approved_quantity,
            "notional_usd": format(consumed.approved_notional_usd, "f"),
            "quote_id": consumed.quote_id,
            "transport_generation": consumed.transport_generation,
            "evaluated_at": utc_to_text(consumed.evaluated_at),
            "expires_at": utc_to_text(consumed.expires_at),
            "broker_contract": list(consumed.broker_contract),
        }

    return journal._reserve_entry_capacity(produce_payload, expected_head=expected_head)


@dataclass(frozen=True, slots=True)
class PendingEntryCapacity:
    """Non-authorizing totals from a caller-verified journal replay."""

    journal_head: tuple[int, str]
    pending_symbol_notional_usd: Decimal
    pending_sector_notional_usd: Decimal
    pending_portfolio_notional_usd: Decimal
    pending_account_notional_usd: Decimal
    pending_cash_usd: Decimal
    pending_buying_power_usd: Decimal
    pending_daily_notional_usd: Decimal
    account_occupied_position_slots: int
    symbol_has_position_or_pending_entry: bool


def summarize_entry_capacity(state, *, portfolio_id, symbol, sector, held_symbols):
    """Aggregate every unresolved reservation; the caller owns replay provenance.

    Sector/symbol and buying-power totals cover the account. Cash/daily totals
    cover the requested portfolio. Expiry never releases an unresolved record.
    """
    if type(state) is not ReplayState:
        raise ValueError("pending capacity requires an exact journal replay")
    state.__post_init__()
    from robo_trader.safety.entry_release import pending_entries_from_events

    if pending_entries_from_events(state.events) != state.pending_entry_events:
        raise ValueError("pending capacity replay omits entry events")
    if state.active_reservations or state.quarantined_reservations:
        raise ValueError("unresolved reduction prevents entry capacity evaluation")
    _identifier(portfolio_id, "portfolio_id")
    _symbol(symbol)
    _sector(sector)
    if type(held_symbols) is not tuple:
        raise ValueError("held symbols must be an exact tuple")
    occupied = {_symbol(held) for held in held_symbols}
    account = portfolio = symbol_total = sector_total = Decimal("0")

    def add(left, right):
        return _exact_subtract(left, right.copy_negate(), "pending entry capacity")

    for event in state.pending_entry_events:
        payload = json.loads(event.payload_json)
        validate_entry_capacity_event(event, payload)
        notional = parse_fixed_decimal(payload["notional_usd"])
        account = add(account, notional)
        if event.portfolio_id == portfolio_id:
            portfolio = add(portfolio, notional)
        if payload["symbol"] == symbol:
            symbol_total = add(symbol_total, notional)
        if payload["sector"] == sector:
            sector_total = add(sector_total, notional)
        occupied.add(payload["symbol"])
    return PendingEntryCapacity(
        (state.last_sequence, state.last_chain_hash),
        symbol_total,
        sector_total,
        portfolio,
        account,
        portfolio,
        account,
        portfolio,
        len(occupied),
        symbol in occupied,
    )
