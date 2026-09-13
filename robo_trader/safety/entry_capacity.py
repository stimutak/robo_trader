"""Reservation-only entry journal records; no order permits or release authority.

The final entry gateway must bind evaluated pending capacity to the journal head
and consume these records with atomic terminal settlement before enabling BUYs.
"""

from __future__ import annotations

import json
import re

from .models import (
    JournalEventType,
    canonical_json,
    parse_fixed_decimal,
    parse_utc_text,
    sha256_text,
)

_FIELDS = frozenset(
    {
        "version",
        "intent_id",
        "signal_id",
        "symbol",
        "sector",
        "quantity",
        "notional_usd",
        "quote_id",
        "transport_generation",
        "evaluated_at",
        "expires_at",
        "broker_contract",
    }
)


def _capacity_key(intent_id):
    # Alphabetic digest encoding keeps opaque IDs out of the raw-account scanner.
    return "ecap-" + sha256_text(intent_id).translate(str.maketrans("0123456789", "ghijklmnop"))


def validate_entry_capacity_event(event, payload):
    from .journal import JournalIntegrityError

    try:
        if type(payload) is not dict or set(payload) != _FIELDS:
            raise ValueError("invalid fields")
        if type(payload["version"]) is not int or payload["version"] != 1:
            raise ValueError("invalid version")
        for name in ("intent_id", "signal_id", "quote_id"):
            if type(payload[name]) is not str or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", payload[name]
            ):
                raise ValueError("invalid identifier")
        if type(payload["symbol"]) is not str or not re.fullmatch(
            r"[A-Z0-9][A-Z0-9._-]{0,31}", payload["symbol"]
        ):
            raise ValueError("invalid symbol")
        if (
            type(payload["sector"]) is not str
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 &./_-]{0,63}", payload["sector"])
            or payload["sector"] != payload["sector"].strip()
        ):
            raise ValueError("invalid sector")
        generation = payload["transport_generation"]
        if (
            type(generation) is not str
            or not generation
            or generation != generation.strip()
            or len(generation) > 256
            or any(ord(char) < 32 or ord(char) == 127 for char in generation)
        ):
            raise ValueError("invalid transport generation")
        if type(payload["quantity"]) is not int or payload["quantity"] <= 0:
            raise ValueError("invalid quantity")
        notional = parse_fixed_decimal(payload["notional_usd"])
        if notional <= 0 or format(notional, "f") != payload["notional_usd"]:
            raise ValueError("invalid notional")
        evaluated = parse_utc_text(payload["evaluated_at"])
        expires = parse_utc_text(payload["expires_at"])
        if not evaluated <= event.occurred_at < expires:
            raise ValueError("reservation is outside decision lifetime")
        contract = payload["broker_contract"]
        if type(contract) is not list or len(contract) != 8:
            raise ValueError("invalid contract")
        if type(contract[0]) is not int or contract[0] <= 0:
            raise ValueError("invalid contract identifier")
        if contract[3:6] != ["STK", "USD", "SMART"]:
            raise ValueError("entry contract must be SMART/USD stock")
        if contract[2] != payload["symbol"]:
            raise ValueError("local symbol mismatch")
        for value in (contract[1], contract[2], contract[6], contract[7]):
            if type(value) is not str or not re.fullmatch(r"[A-Z0-9][A-Z0-9._:/-]{0,63}", value):
                raise ValueError("invalid contract identity")
        if (
            contract[0] != event.con_id
            or contract[1] != payload["symbol"]
            or event.claim_id is not None
            or event.idempotency_key != _capacity_key(payload["intent_id"])
            or event.intent_fingerprint != sha256_text(canonical_json(payload))
        ):
            raise ValueError("entry event identity mismatch")
    except (ValueError, TypeError, KeyError) as exc:
        raise JournalIntegrityError("invalid entry capacity event") from exc


def reserve_entry_capacity(journal, produce_payload, *, expected_head):
    from .journal import JournalIntegrityError, ReservationConflict, StateTransitionError

    if (
        type(expected_head) is not tuple
        or len(expected_head) != 2
        or type(expected_head[0]) is not int
        or expected_head[0] < 0
        or type(expected_head[1]) is not str
        or len(expected_head[1]) != 64
    ):
        raise StateTransitionError("entry capacity requires an exact journal head")

    def operation(connection):
        state = journal._replay_connection(connection)
        if (state.last_sequence, state.last_chain_hash) != expected_head:
            raise StateTransitionError("entry capacity journal head changed")
        identity = journal._bound_runtime_identity(connection)
        if identity is None:
            raise JournalIntegrityError("entry capacity requires a bound journal")
        at = journal._event_time()
        portfolio_id, payload = produce_payload(at)
        key = _capacity_key(payload["intent_id"])
        if any(event.idempotency_key == key for event in state.events):
            raise ReservationConflict("entry capacity intent already exists")
        con_id = payload["broker_contract"][0]
        if any(item.con_id == con_id for item in state.active_reservations):
            raise ReservationConflict("entry capacity conflicts with an unresolved reduction")
        for event in state.pending_entry_events:
            prior = json.loads(event.payload_json)
            if event.con_id == con_id or prior["symbol"] == payload["symbol"]:
                raise ReservationConflict("entry capacity conflicts with an unresolved entry")
        event = journal._append(
            connection,
            JournalEventType.ENTRY_CAPACITY_RESERVED,
            at,
            key,
            identity[0],
            identity[1],
            portfolio_id,
            con_id,
            sha256_text(canonical_json(payload)),
            None,
            payload,
        )
        validate_entry_capacity_event(event, payload)
        return event

    return journal._write_transaction(operation)
