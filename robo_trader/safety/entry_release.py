"""Standard-library validation of entry release events and pending subsets.

Runtime proof consumption is owned by paper_entry_release outside this package.
"""

import json
import re
from decimal import Context, Inexact, InvalidOperation, Overflow
from zoneinfo import ZoneInfo

from .entry_capacity import validate_entry_capacity_event, validate_entry_claim_event
from .models import (
    MAX_DECIMAL_DIGITS,
    JournalEventType,
    parse_fixed_decimal,
    parse_utc_text,
    sha256_text,
)

_FIELDS = frozenset(
    {
        "version",
        "reservation_sequence",
        "reservation_chain_hash",
        "claim_sequence",
        "claim_chain_hash",
        "settlement_id",
        "record_fingerprint",
        "receipt_fingerprint",
        "committed_at",
        "outcome_at",
        "terminal_status",
        "filled_quantity",
        "fill_price",
        "fill_execution_id",
        "fill_notional",
        "gross_filled_notional",
        "trading_date",
        "accounting_confirmed_at",
        "accounting_confirmation_fingerprint",
    }
)


def _release_key(reservation):
    return "erelease-" + sha256_text(reservation.idempotency_key).translate(
        str.maketrans("0123456789", "ghijklmnop")
    )


def validate_entry_release_event(event, payload, reservation, claim):
    """Validate historical event structure; runtime issuance additionally needs owned proofs."""
    from .journal import JournalIntegrityError

    try:
        if type(payload) is not dict or set(payload) != _FIELDS:
            raise ValueError("release fields differ")
        if type(payload["version"]) is not int or payload["version"] != 1:
            raise ValueError("release version differs")
        if reservation is None or claim is None:
            raise ValueError("release parent is missing")
        validate_entry_capacity_event(reservation, json.loads(reservation.payload_json))
        validate_entry_claim_event(claim, json.loads(claim.payload_json), reservation)
        if (
            event.event_type is not JournalEventType.ENTRY_SETTLEMENT_RELEASED
            or event.sequence <= claim.sequence
            or event.idempotency_key != _release_key(reservation)
            or event.intent_fingerprint != reservation.intent_fingerprint
            or event.claim_id != claim.claim_id
            or (event.execution_domain_scope, event.account_scope, event.portfolio_id, event.con_id)
            != (claim.execution_domain_scope, claim.account_scope, claim.portfolio_id, claim.con_id)
        ):
            raise ValueError("release event identity differs")
        for prefix, parent in (("reservation", reservation), ("claim", claim)):
            if (
                type(payload[prefix + "_sequence"]) is not int
                or payload[prefix + "_sequence"] != parent.sequence
                or payload[prefix + "_chain_hash"] != parent.chain_hash
            ):
                raise ValueError("release parent identity differs")
        for key in (
            "record_fingerprint",
            "receipt_fingerprint",
            "accounting_confirmation_fingerprint",
        ):
            if type(payload[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None:
                raise ValueError("release fingerprint is invalid")
        if (
            type(payload["settlement_id"]) is not str
            or re.fullmatch(r"pset-[0-9a-f]{32}", payload["settlement_id"]) is None
        ):
            raise ValueError("release settlement ID is invalid")
        at = parse_utc_text(payload["outcome_at"])
        committed = parse_utc_text(payload["committed_at"])
        confirmed = parse_utc_text(payload["accounting_confirmed_at"])
        if not claim.occurred_at <= at <= committed <= confirmed <= event.occurred_at:
            raise ValueError("release evidence time is invalid")
        if (event.occurred_at - confirmed).total_seconds() > 5:
            raise ValueError("release accounting evidence is stale")
        if (
            payload["trading_date"]
            != at.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        ):
            raise ValueError("release trading date differs")
        capacity = json.loads(reservation.payload_json)
        quantity = parse_fixed_decimal(payload["filled_quantity"])
        principal = parse_fixed_decimal(payload["fill_notional"])
        total = parse_fixed_decimal(payload["gross_filled_notional"])
        if not 0 <= principal <= parse_fixed_decimal(capacity["notional_usd"]) or total < principal:
            raise ValueError("release principal is not accounted")
        if payload["terminal_status"] == "FILLED":
            price = parse_fixed_decimal(payload["fill_price"])
            if (
                quantity != parse_fixed_decimal(str(capacity["quantity"]))
                or price <= 0
                or principal
                != Context(
                    prec=2 * MAX_DECIMAL_DIGITS, traps=[Inexact, InvalidOperation, Overflow]
                ).multiply(quantity, price)
                or type(payload["fill_execution_id"]) is not str
                or re.fullmatch(r"lpfill-[0-9a-f]{32}", payload["fill_execution_id"]) is None
            ):
                raise ValueError("release fill differs")
        elif payload["terminal_status"] in {"REJECTED", "CANCELLED", "EXPIRED"}:
            if (
                quantity != 0
                or principal != 0
                or payload["fill_execution_id"] is not None
                or payload["fill_price"] is not None
            ):
                raise ValueError("zero-fill release has execution evidence")
        else:
            raise ValueError("release outcome is unsupported")
    except (ValueError, TypeError, KeyError) as exc:
        raise JournalIntegrityError("invalid entry settlement release event") from exc


def pending_entries_from_events(events):
    """Recompute the pending subset without treating a release-shaped row as authority."""
    from .journal import JournalIntegrityError

    pending, claims = {}, {}
    for event in events:
        payload = json.loads(event.payload_json)
        if event.event_type is JournalEventType.ENTRY_CAPACITY_RESERVED:
            validate_entry_capacity_event(event, payload)
            pending[event.sequence] = event
        elif event.event_type is JournalEventType.ENTRY_SUBMISSION_CLAIMED:
            validate_entry_claim_event(
                event, payload, pending.get(payload.get("reservation_sequence"))
            )
            claims[event.sequence] = event
        elif event.event_type is JournalEventType.ENTRY_SETTLEMENT_RELEASED:
            reservation = pending.get(payload.get("reservation_sequence"))
            validate_entry_release_event(
                event, payload, reservation, claims.get(payload.get("claim_sequence"))
            )
            if reservation is None:
                raise JournalIntegrityError("entry release lacks pending capacity")
            del pending[reservation.sequence]
    return tuple(pending.values())
