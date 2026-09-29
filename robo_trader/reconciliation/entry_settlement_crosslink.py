"""Crosslink entry releases against committed storage in a held bootstrap snapshot."""

from dataclasses import asdict
import json

from robo_trader.paper_entry_settlement import validate_stored_paper_entry_terminal_record
from robo_trader.paper_entry_storage_replay import _read_entry_storage_on_connection
from robo_trader.paper_terminal_settlement import _exact_decimal_multiply
from robo_trader.safety import SafetyJournal
from robo_trader.safety.entry_release import validate_entry_release_event
from robo_trader.safety.models import (
    JournalEventType,
    ValidationError,
    canonical_json,
    parse_fixed_decimal,
    utc_to_text,
)


def crosslink_entry_settlements(connection, replay_state, runtime, device, inode):
    """Return complete entry terminal/fill counts; any unresolved or orphan row fails."""
    if not connection.in_transaction or not runtime.safety_journal_path:
        raise ValidationError("entry crosslink requires held ledger and journal snapshots")
    if (
        runtime.execution_mode != "paper"
        or runtime.state_namespace != "paper"
        or runtime.ibkr_readonly is not True
    ):
        raise ValidationError("entry crosslink requires paper read-only runtime")
    events = {event.sequence: event for event in replay_state.events}
    reservations = [
        e for e in events.values() if e.event_type is JournalEventType.ENTRY_CAPACITY_RESERVED
    ]
    claims = [
        e for e in events.values() if e.event_type is JournalEventType.ENTRY_SUBMISSION_CLAIMED
    ]
    releases = [
        e for e in events.values() if e.event_type is JournalEventType.ENTRY_SETTLEMENT_RELEASED
    ]
    cursor = connection.execute(
        "SELECT settlement_id,request_payload_json,request_fingerprint FROM main.paper_reduction_settlements WHERE settlement_kind='ENTRY' ORDER BY rowid"
    )
    rows = cursor.fetchall()
    if not len(rows) == len(reservations) == len(claims) == len(releases):
        raise ValidationError("entry journal and terminal storage cardinality differ")
    by_reservation = {}
    for release in releases:
        payload = json.loads(release.payload_json)
        reservation = events.get(payload.get("reservation_sequence"))
        claim = events.get(payload.get("claim_sequence"))
        validate_entry_release_event(release, payload, reservation, claim)
        if reservation.sequence in by_reservation:
            raise ValidationError("entry release duplicates a reservation")
        by_reservation[reservation.sequence] = (release, payload)
    journal = SafetyJournal(runtime.safety_journal_path)
    used = set()
    filled = 0
    for settlement_id, payload_json, fingerprint in rows:
        record = validate_stored_paper_entry_terminal_record(
            payload_json, fingerprint=fingerprint, journal=journal
        )
        data = json.loads(record.payload_json)
        for key in ("reservation", "claim"):
            actual = events.get(data[key]["sequence"])
            if actual is None or canonical_json(asdict(actual)) != canonical_json(data[key]):
                raise ValidationError("entry record differs from held journal snapshot")
        claim, outcome = data["claim"], data["outcome"]
        if (
            claim["execution_domain_scope"] != runtime.safety_execution_domain_scope
            or claim["account_scope"] != runtime.safety_account_scope
        ):
            raise ValidationError("entry crosslink runtime scope differs")
        sequence = data["reservation"]["sequence"]
        if sequence in used or sequence not in by_reservation:
            raise ValidationError("entry storage lacks its unique released reservation")
        used.add(sequence)
        _, released = by_reservation[sequence]
        stored = _read_entry_storage_on_connection(
            connection, record, runtime.database_path, runtime.database_identity, device, inode
        )
        quantity = parse_fixed_decimal(outcome["filled_quantity"])
        principal = (
            quantity
            if not quantity
            else _exact_decimal_multiply(
                quantity,
                parse_fixed_decimal(outcome["exact_fill_price"]),
                "entry crosslink principal",
            )
        )
        expected = json.loads(
            canonical_json(
                dict(
                    settlement_id=settlement_id,
                    record_fingerprint=record.fingerprint,
                    receipt_fingerprint=stored.fingerprint,
                    committed_at=utc_to_text(stored.committed_at),
                    outcome_at=outcome["observed_at"],
                    terminal_status=outcome["status"],
                    filled_quantity=outcome["filled_quantity"],
                    fill_price=outcome["exact_fill_price"],
                    fill_notional=principal,
                    fill_execution_id=(
                        None
                        if outcome["fill_evidence"] is None
                        else outcome["fill_evidence"]["execution_id"]
                    ),
                )
            )
        )
        if stored.settlement_id != settlement_id or any(
            released[key] != value for key, value in expected.items()
        ):
            raise ValidationError("entry release differs from committed outcome")
        filled += int(quantity > 0)
    return len(rows), filled
