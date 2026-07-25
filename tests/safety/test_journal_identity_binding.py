"""Adversarial coverage for the journal's immutable runtime identity."""

import sqlite3

import pytest

from robo_trader.safety import (
    JournalIntegrityError,
    OrderSide,
    SafetyJournal,
)

from .conftest import ACCOUNT_A, ACCOUNT_B, make_case

BOUND_DOMAIN = "paper-domain"

SELECT_EVENT = """
    SELECT sequence, event_id, event_type, occurred_at, idempotency_key,
           execution_domain_scope, account_scope, portfolio_id, con_id,
           intent_fingerprint, claim_id, payload_json, previous_chain_hash,
           payload_hash, chain_hash, schema_version
    FROM safety_journal_events
"""

INSERT_EVENT = """
    INSERT INTO safety_journal_events (
        sequence, event_id, event_type, occurred_at, idempotency_key,
        execution_domain_scope, account_scope, portfolio_id, con_id,
        intent_fingerprint, claim_id, payload_json, previous_chain_hash,
        payload_hash, chain_hash, schema_version
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _bound_journal(path, now):
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize(
        execution_domain_scope=BOUND_DOMAIN,
        account_scope=ACCOUNT_A,
    )
    return journal


def _record_denial(journal, now, *, domain=BOUND_DOMAIN, account=ACCOUNT_A):
    intent, exposure, allocation, gates, _, _ = make_case(
        now,
        side=OrderSide.BUY,
        execution_domain_scope=domain,
        account_scope=account,
    )
    return journal.record_rejection(
        f"denied-{domain}-{account[-4:]}",
        intent,
        exposure,
        allocation,
        gates,
    )


def _event_count(path):
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        return connection.execute("SELECT COUNT(*) FROM safety_journal_events").fetchone()[0]


@pytest.mark.parametrize(
    ("domain", "account"),
    (
        ("other-paper-domain", ACCOUNT_A),
        (BOUND_DOMAIN, ACCOUNT_B),
    ),
)
def test_bound_journal_rejects_cross_identity_append_before_insert(
    tmp_path,
    now,
    domain,
    account,
):
    path = tmp_path / "bound-safety.db"
    journal = _bound_journal(path, now)

    with pytest.raises(
        JournalIntegrityError,
        match="event identity does not match bound journal identity",
    ):
        _record_denial(journal, now, domain=domain, account=account)

    assert _event_count(path) == 0
    replay = journal.replay(
        expected_execution_domain_scope=BOUND_DOMAIN,
        expected_account_scope=ACCOUNT_A,
    )
    assert replay.last_sequence == 0


def test_bound_journal_accepts_only_matching_identity(tmp_path, now):
    path = tmp_path / "bound-safety.db"
    journal = _bound_journal(path, now)

    event = _record_denial(journal, now)

    assert event.sequence == 1
    replay = journal.replay(
        expected_execution_domain_scope=BOUND_DOMAIN,
        expected_account_scope=ACCOUNT_A,
    )
    assert replay.events == (event,)


def test_bound_journal_applies_identity_to_atomic_authorization_events(tmp_path, now):
    path = tmp_path / "bound-safety.db"
    journal = _bound_journal(path, now)
    intent, exposure, allocation, gates, _, descriptor = make_case(now)

    reservation, claim, permit = journal.authorize_submission(
        "authorized",
        intent,
        exposure,
        allocation,
        gates,
        descriptor,
    )

    assert reservation.execution_domain_scope == BOUND_DOMAIN
    assert reservation.account_scope == ACCOUNT_A
    assert claim.execution_domain_scope == BOUND_DOMAIN
    assert claim.account_scope == ACCOUNT_A
    assert permit is not None
    replay = journal.replay(
        expected_execution_domain_scope=BOUND_DOMAIN,
        expected_account_scope=ACCOUNT_A,
    )
    assert replay.last_sequence == 3
    assert {(event.execution_domain_scope, event.account_scope) for event in replay.events} == {
        (BOUND_DOMAIN, ACCOUNT_A)
    }


@pytest.mark.parametrize(
    ("domain", "account"),
    (
        ("other-paper-domain", ACCOUNT_A),
        (BOUND_DOMAIN, ACCOUNT_B),
    ),
)
def test_replay_rejects_validly_hashed_event_from_another_identity(
    tmp_path,
    now,
    domain,
    account,
):
    source_path = tmp_path / "source.db"
    source = SafetyJournal(source_path, clock=lambda: now)
    source.initialize()
    _record_denial(source, now, domain=domain, account=account)

    with sqlite3.connect(f"{source_path.as_uri()}?mode=ro", uri=True) as source_connection:
        source_row = source_connection.execute(SELECT_EVENT).fetchone()

    target_path = tmp_path / "target.db"
    target = _bound_journal(target_path, now)
    with sqlite3.connect(target_path) as target_connection:
        target_connection.execute(INSERT_EVENT, source_row)
        target_connection.commit()

    with pytest.raises(
        JournalIntegrityError,
        match="persisted event identity does not match bound journal identity",
    ):
        target.replay(
            expected_execution_domain_scope=BOUND_DOMAIN,
            expected_account_scope=ACCOUNT_A,
        )

    # Every later write transaction also rejects the contaminated journal
    # before it can append another intent, decision, or lifecycle event.
    with pytest.raises(
        JournalIntegrityError,
        match="persisted event identity does not match bound journal identity",
    ):
        _record_denial(target, now)
    assert _event_count(target_path) == 1
