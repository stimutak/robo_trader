"""Entry attempts require one durable claim without releasing capacity."""

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from robo_trader.safety import SafetyJournal
from robo_trader.safety.entry_capacity import claim_entry_capacity, validate_entry_claim_event
from robo_trader.safety.journal import (
    IdempotencyConflict,
    JournalIntegrityError,
    StateTransitionError,
)
from robo_trader.safety.models import JournalEventType
from tests.risk.test_pending_entry_capacity import append, journal_at
from tests.test_pr7_entry_risk_contract import NOW


def claim(journal, reservation, **overrides):
    state = journal.replay()
    values = dict(
        reservation_sequence=reservation.sequence,
        reservation_chain_hash=reservation.chain_hash,
        expected_head=(state.last_sequence, state.last_chain_hash),
    )
    values.update(overrides)
    return claim_entry_capacity(journal, **values)


def test_claim_survives_reopen_and_keeps_capacity_pending(tmp_path):
    journal = journal_at(tmp_path)
    reservation = append(journal)
    claimed = claim(journal, reservation)
    assert claimed.event_type is JournalEventType.ENTRY_SUBMISSION_CLAIMED
    assert claimed.claim_id.startswith("claim-")
    assert claimed.intent_fingerprint == reservation.intent_fingerprint
    reopened = SafetyJournal(tmp_path / "journal.db", clock=lambda: NOW)
    state = reopened.replay()
    assert state.events[-1] == claimed
    assert state.pending_entry_events == (reservation,)
    with pytest.raises(StateTransitionError):
        claim(reopened, reservation)
    assert reopened.replay() == state


@pytest.mark.parametrize(
    "change",
    [
        {"reservation_sequence": True},
        {"reservation_sequence": 999},
        {"reservation_chain_hash": "a" * 64},
        {"expected_head": (0, "0" * 64)},
    ],
)
def test_wrong_parent_or_stale_head_cannot_claim(tmp_path, change):
    journal = journal_at(tmp_path)
    reservation = append(journal)
    before = journal.replay()
    with pytest.raises(StateTransitionError):
        claim(journal, reservation, **change)
    assert journal.replay() == before


def test_expired_capacity_is_not_released_or_claimed(tmp_path):
    journal = journal_at(tmp_path)
    reservation = append(journal)
    later = SafetyJournal(tmp_path / "journal.db", clock=lambda: NOW + timedelta(days=1))
    with pytest.raises(StateTransitionError):
        claim(later, reservation)
    assert later.replay().pending_entry_events == (reservation,)


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"reservation_sequence": True},
        {"reservation_chain_hash": "b" * 64},
        {"order_ref": "different-order"},
        {"extra": "value"},
    ],
)
def test_claim_payload_is_bound_to_exact_parent_and_attempt(tmp_path, change):
    journal = journal_at(tmp_path)
    reserved = append(journal)
    claimed = claim(journal, reserved)
    payload = dict(json.loads(claimed.payload_json), **change)
    with pytest.raises(JournalIntegrityError):
        validate_entry_claim_event(claimed, payload, reserved)


@pytest.mark.parametrize(
    "change",
    [
        {"portfolio_id": "other"},
        {"con_id": 12345},
        {"intent_fingerprint": "b" * 64},
        {"claim_id": "claim-" + "b" * 32},
        {"occurred_at": NOW + timedelta(days=1)},
    ],
)
def test_claim_envelope_cannot_change_scope_or_lifetime(tmp_path, change):
    journal = journal_at(tmp_path)
    reserved = append(journal)
    claimed = claim(journal, reserved)
    with pytest.raises(JournalIntegrityError):
        validate_entry_claim_event(
            replace(claimed, **change), json.loads(claimed.payload_json), reserved
        )


def test_two_connections_cannot_claim_same_capacity(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    journal = journal_at(tmp_path)
    reserved = append(journal)
    head = (reserved.sequence, reserved.chain_hash)

    def attempt(_):
        other = SafetyJournal(tmp_path / "journal.db", clock=lambda: NOW)
        try:
            return claim(other, reserved, expected_head=head)
        except StateTransitionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(value is not None for value in results) == 1
    state = journal.replay()
    assert len(state.events) == 2
    assert state.pending_entry_events == (reserved,)


def test_other_journal_paths_cannot_reuse_entry_claim_key(tmp_path):
    journal = journal_at(tmp_path)
    claimed = claim(journal, append(journal))
    before = journal.replay()

    def append_conflict(connection):
        return journal._append(
            connection,
            JournalEventType.SAFETY_DECISION,
            NOW,
            claimed.idempotency_key,
            claimed.execution_domain_scope,
            claimed.account_scope,
            claimed.portfolio_id,
            claimed.con_id,
            claimed.intent_fingerprint,
            None,
            {},
        )

    with pytest.raises(IdempotencyConflict):
        journal._write_transaction(append_conflict)
    assert journal.replay() == before


def test_lost_committed_response_does_not_allow_another_attempt(tmp_path, monkeypatch):
    journal = journal_at(tmp_path)
    reserved = append(journal)
    write = journal._write_transaction

    def lose_response(operation):
        write(operation)
        raise OSError("simulated response loss after durable commit")

    monkeypatch.setattr(journal, "_write_transaction", lose_response)
    with pytest.raises(OSError):
        claim(journal, reserved)
    reopened = SafetyJournal(tmp_path / "journal.db", clock=lambda: NOW)
    with pytest.raises(StateTransitionError, match="already been claimed"):
        claim(reopened, reserved)
    assert len(reopened.replay().events) == 2


def test_failure_before_commit_leaves_reservation_intact(tmp_path):
    journal = journal_at(tmp_path)
    reserved = append(journal)

    def fail(step, event):
        if step == "BEFORE_COMMIT":
            raise OSError("simulated commit failure")

    journal._fault_hook = fail
    with pytest.raises(OSError):
        claim(journal, reserved)
    journal._fault_hook = None
    assert journal.replay().events == (reserved,)
    assert claim(journal, reserved).sequence == 2


def test_replay_rejects_hash_valid_claim_with_wrong_parent(tmp_path):
    from robo_trader.safety.entry_capacity import _claim_key

    journal = journal_at(tmp_path)
    reserved = append(journal)
    claim_id = "claim-" + "a" * 32

    def append_corrupt_claim(connection):
        return journal._append(
            connection,
            JournalEventType.ENTRY_SUBMISSION_CLAIMED,
            NOW,
            _claim_key(reserved),
            reserved.execution_domain_scope,
            reserved.account_scope,
            reserved.portfolio_id,
            reserved.con_id,
            reserved.intent_fingerprint,
            claim_id,
            {
                "version": 1,
                "reservation_sequence": reserved.sequence,
                "reservation_chain_hash": "b" * 64,
                "order_ref": "entry-" + "a" * 32,
            },
        )

    journal._write_transaction(append_corrupt_claim)
    with pytest.raises(JournalIntegrityError, match="invalid entry submission claim"):
        journal.replay()


def test_claim_identifier_collision_rolls_back_before_append(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from robo_trader.safety import entry_capacity

    journal = journal_at(tmp_path)
    first = append(journal)
    second = append(journal, symbol="MSFT", con_id=272093)
    claimed = claim(journal, first)
    before = journal.replay()
    monkeypatch.setattr(
        entry_capacity,
        "uuid",
        SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=claimed.claim_id.removeprefix("claim-"))),
    )
    with pytest.raises(StateTransitionError, match="identifier already exists"):
        claim(journal, second)
    assert journal.replay() == before


def test_claim_keeps_all_pending_risk_capacity_reserved(tmp_path):
    from robo_trader.risk.entry_reservations import summarize_entry_capacity

    journal = journal_at(tmp_path)
    reserved = append(journal)
    arguments = dict(portfolio_id="alpha", symbol="AAPL", sector="Technology", held_symbols=())
    before = summarize_entry_capacity(journal.replay(), **arguments)
    claim(journal, reserved)
    after = summarize_entry_capacity(journal.replay(), **arguments)
    assert replace(after, journal_head=before.journal_head) == before


def test_bootstrap_cannot_ignore_claimed_entry(tmp_path):
    from robo_trader.reconciliation.bootstrap_producer import (
        BootstrapReconciliationBlocked,
        _crosslink_safety_journal_orders,
    )

    journal = journal_at(tmp_path)
    claim(journal, append(journal))
    with pytest.raises(BootstrapReconciliationBlocked, match="entry capacity"):
        _crosslink_safety_journal_orders(
            connection=None,
            actual_tables=set(),
            replay_state=journal.replay(),
            trade_rows=(),
            runtime=None,
            database_device=1,
            database_inode=1,
        )
