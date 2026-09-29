"""Bind owned entry settlement/accounting proofs to durable safety-journal release."""

import json
import re
from dataclasses import asdict
from pathlib import Path

from .paper_entry_receipt import assert_owned_entry_receipt
from .paper_terminal_settlement import _exact_decimal_multiply
from .risk.paper_entry_accounting_confirmation import consume_entry_accounting_confirmation
from .safety.entry_release import _release_key, validate_entry_release_event
from .safety.models import (
    JournalEventType,
    canonical_json,
    parse_fixed_decimal,
    sha256_text,
    utc_to_text,
)


def release_entry_capacity(journal, receipt, confirmation, *, database, accounting, expected_head):
    """Append one release only after consuming genuine fresh accounting evidence.

    Failure never makes an old confirmation reusable. Retrying obtains a fresh
    confirmation; an exact already-committed release returns its original event.
    The final gateway must hold its account lock over confirmation and release.
    """
    from robo_trader.risk.paper_fill_accounting import PaperFillAccounting

    from .safety.journal import SafetyJournal, StateTransitionError

    if type(journal) is not SafetyJournal or type(accounting) is not PaperFillAccounting:
        raise StateTransitionError("entry release requires exact journal and accounting producers")
    runtime = accounting._runtime
    if not runtime.safety_journal_path or journal.database_path != Path(
        runtime.safety_journal_path
    ):
        raise StateTransitionError("entry release journal differs from runtime")
    if (
        type(expected_head) is not tuple
        or len(expected_head) != 2
        or type(expected_head[0]) is not int
        or expected_head[0] < 0
        or type(expected_head[1]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_head[1]) is None
    ):
        raise StateTransitionError("entry release requires an exact journal head")
    assert_owned_entry_receipt(receipt, database=database, runtime_contract=runtime)
    data = json.loads(receipt.record.payload_json)

    def operation(connection):
        state = journal._replay_connection(connection)
        events = {e.sequence: e for e in state.events}
        reservation = events.get(data["reservation"]["sequence"])
        claim = events.get(data["claim"]["sequence"])
        if (
            reservation is None
            or claim is None
            or canonical_json(asdict(reservation)) != canonical_json(data["reservation"])
            or canonical_json(asdict(claim)) != canonical_json(data["claim"])
        ):
            raise StateTransitionError(
                "entry release journal parents differ from committed receipt"
            )
        prior = next(
            (e for e in state.events if e.idempotency_key == _release_key(reservation)), None
        )
        if prior is not None:
            prior_data = json.loads(prior.payload_json)
            outcome = data["outcome"]
            quantity = parse_fixed_decimal(outcome["filled_quantity"])
            principal = (
                quantity
                if not quantity
                else _exact_decimal_multiply(
                    quantity,
                    parse_fixed_decimal(outcome["exact_fill_price"]),
                    "entry release retry",
                )
            )
            expected = json.loads(
                canonical_json(
                    dict(
                        record_fingerprint=receipt.record.fingerprint,
                        receipt_fingerprint=receipt.fingerprint(),
                        settlement_id=receipt.settlement_id,
                        committed_at=utc_to_text(receipt.committed_at),
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
            if any(prior_data[key] != value for key, value in expected.items()):
                raise StateTransitionError("entry release identity is bound to another settlement")
            consume_entry_accounting_confirmation(
                confirmation, receipt=receipt, database=database, accounting=accounting
            )
            return prior
        if (
            expected_head != (state.last_sequence, state.last_chain_hash)
            or reservation not in state.pending_entry_events
        ):
            raise StateTransitionError("entry release head or pending capacity changed")
        proof = consume_entry_accounting_confirmation(
            confirmation, receipt=receipt, database=database, accounting=accounting
        )
        outcome = data["outcome"]
        payload = dict(
            version=1,
            reservation_sequence=reservation.sequence,
            reservation_chain_hash=reservation.chain_hash,
            claim_sequence=claim.sequence,
            claim_chain_hash=claim.chain_hash,
            settlement_id=receipt.settlement_id,
            record_fingerprint=receipt.record.fingerprint,
            receipt_fingerprint=receipt.fingerprint(),
            committed_at=utc_to_text(receipt.committed_at),
            outcome_at=outcome["observed_at"],
            terminal_status=outcome["status"],
            filled_quantity=outcome["filled_quantity"],
            fill_price=outcome["exact_fill_price"],
            fill_execution_id=(
                None
                if outcome["fill_evidence"] is None
                else outcome["fill_evidence"]["execution_id"]
            ),
            fill_notional=proof.fill_notional,
            gross_filled_notional=proof.gross_filled_notional,
            trading_date=proof.trading_date,
            accounting_confirmed_at=utc_to_text(proof.confirmed_at),
            accounting_confirmation_fingerprint=sha256_text(proof.canonical_payload()),
        )
        # Canonicalize decimal representations before validating the exact same
        # payload that replay will later read from SQLite.
        payload = json.loads(canonical_json(payload))
        event = journal._append(
            connection,
            JournalEventType.ENTRY_SETTLEMENT_RELEASED,
            journal._event_time(),
            _release_key(reservation),
            claim.execution_domain_scope,
            claim.account_scope,
            claim.portfolio_id,
            claim.con_id,
            claim.intent_fingerprint,
            claim.claim_id,
            payload,
        )
        validate_entry_release_event(event, payload, reservation, claim)
        return event

    return journal._write_transaction(operation)
