"""Owned decision adapter for dormant, durable entry capacity reservations."""

from robo_trader.safety.journal import ReservationConflict, SafetyJournal
from robo_trader.safety.models import utc_to_text

from .entry_contract import EntrySide, _sector, assert_and_consume_risk_decision


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
