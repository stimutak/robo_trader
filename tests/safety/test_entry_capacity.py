"""Durable entry capacity is reservation-only and never expires into free capacity."""

import uuid
from datetime import timedelta

import pytest

from robo_trader.safety import SafetyJournal
from robo_trader.risk.entry_reservations import reserve_entry_capacity
from robo_trader.safety.journal import JournalError
from tests.safety.conftest import ACCOUNT_A
from tests.test_pr7_entry_risk_contract import NOW, _evaluate, _intent


def decision():
    return _evaluate(intent=_intent(intent_id="capacity-" + uuid.uuid4().hex))


def journal_at(path, clock=lambda: NOW):
    journal = SafetyJournal(path, clock=clock)
    journal.initialize(execution_domain_scope="paper-domain", account_scope=ACCOUNT_A)
    return journal


def reserve(journal, approved=None, head=None):
    state = journal.replay()
    return reserve_entry_capacity(
        journal,
        approved or decision(),
        sector="Technology",
        expected_head=head or (state.last_sequence, state.last_chain_hash),
    )


def test_entry_capacity_survives_reopen_and_expiry_without_submission_authority(tmp_path):
    path = tmp_path / "journal.sqlite"
    journal = journal_at(path)
    event = reserve(journal)
    assert event.claim_id is None
    reopened = journal_at(path, clock=lambda: NOW + timedelta(days=2))
    state = reopened.replay()
    assert state.pending_entry_events == (event,)
    assert state.active_reservations == ()
    with pytest.raises((JournalError, ValueError)):
        reserve(reopened)


def test_changed_journal_head_rejects_reservation(tmp_path):
    journal = journal_at(tmp_path / "journal.sqlite")
    head = (0, "0" * 64)
    reserve(journal)
    with pytest.raises(JournalError, match="head"):
        reserve(journal, head=head)
    assert journal.replay().last_sequence == 1


def test_same_decision_cannot_reserve_twice(tmp_path):
    journal = journal_at(tmp_path / "journal.sqlite")
    approved = decision()
    reserve(journal, approved)
    with pytest.raises((JournalError, ValueError)):
        reserve(journal, approved)
    assert journal.replay().last_sequence == 1


def test_unbound_journal_cannot_reserve_entry_capacity(tmp_path):
    journal = SafetyJournal(tmp_path / "journal.sqlite", clock=lambda: NOW)
    journal.initialize()
    with pytest.raises(JournalError, match="bound"):
        reserve(journal)


def test_commit_failure_leaves_no_reservation_or_reusable_decision(tmp_path):
    journal = journal_at(tmp_path / "journal.sqlite")
    approved = decision()

    def fail(*args):
        raise RuntimeError("injected commit failure")

    journal._fault_hook = fail
    with pytest.raises(RuntimeError, match="injected"):
        reserve(journal, approved)
    journal._fault_hook = None
    assert journal.replay().last_sequence == 0
    with pytest.raises(ValueError, match="consumed"):
        reserve(journal, approved)


def test_unresolved_entry_blocks_coordinator_restart(tmp_path):
    from robo_trader.safety import RuntimeStartupBlocked
    from tests.safety.test_runtime_integration import make_runtime_case

    coordinator, _, _, _ = make_runtime_case(tmp_path, NOW)
    reserve(coordinator._journal)
    with pytest.raises(RuntimeStartupBlocked) as caught:
        coordinator.start()
    assert "UNRESOLVED_ENTRY_CAPACITY_AT_STARTUP" in caught.value.reason_codes
    assert coordinator.started is False


def test_concurrent_reservations_cannot_both_commit_against_one_head(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    journal = journal_at(tmp_path / "journal.sqlite")
    decisions = [decision(), decision()]

    def attempt(approved):
        try:
            reserve(journal, approved, head=(0, "0" * 64))
            return True
        except JournalError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, decisions))
    assert sorted(results) == [False, True]
    assert len(journal.replay().pending_entry_events) == 1


def test_pending_entry_blocks_same_contract_reduction_authority(tmp_path):
    from tests.safety.test_runtime_integration import make_runtime_case

    coordinator, _, request, snapshot = make_runtime_case(tmp_path, NOW)
    coordinator.start()
    reserve(coordinator._journal)
    with pytest.raises(JournalError, match="entry capacity"):
        coordinator.authorize("reduction-after-entry", request, snapshot)
    assert coordinator._journal.replay().last_sequence == 1


def test_bootstrap_cannot_ignore_pending_entry_capacity(tmp_path):
    from robo_trader.reconciliation.bootstrap_producer import (
        _crosslink_safety_journal_orders,
        BootstrapReconciliationBlocked,
    )

    journal = journal_at(tmp_path / "journal.sqlite")
    reserve(journal)
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


def test_read_only_status_reports_unresolved_entry_capacity(tmp_path):
    import scripts.manage_paper_safety_journal as script
    from tests.test_pr2b3_paper_safety_status import _environment

    env = _environment(tmp_path)
    runtime = script._paper_contract(env)
    journal = SafetyJournal(runtime.safety_journal_path, clock=lambda: NOW)
    journal.initialize(
        execution_domain_scope=runtime.safety_execution_domain_scope,
        account_scope=runtime.safety_account_scope,
    )
    reserve(journal)
    before = journal.replay()
    status = script.paper_safety_status(env)
    assert status["status"] == "BLOCKED"
    assert status["unresolved_count"] == 1
    assert status["reason_codes"] == ["UNRESOLVED_ENTRY_CAPACITY"]
    assert status["unresolved_reservations"][0]["phase"] == "ENTRY_CAPACITY_RESERVED"
    assert journal.replay() == before


def test_existing_reduction_blocks_new_entry_reservation(tmp_path):
    from tests.safety.test_runtime_integration import make_runtime_case

    coordinator, _, request, snapshot = make_runtime_case(tmp_path, NOW)
    coordinator.start()
    coordinator.authorize("reduction-before-entry", request, snapshot)
    state = coordinator._journal.replay()
    with pytest.raises(JournalError, match="unresolved reduction"):
        reserve(coordinator._journal)
    assert coordinator._journal.replay() == state


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantity", True),
        ("quantity", 0),
        ("notional_usd", "NaN"),
        ("version", True),
        ("sector", ""),
        ("transport_generation", " x "),
        ("expires_at", "2026-07-28T14:00:00Z"),
        ("broker_contract", [265598, "AAPL", "MSFT", "STK", "USD", "SMART", "NASDAQ", "NMS"]),
    ],
)
def test_replay_rejects_semantically_invalid_entry_payload_even_with_valid_hash_chain(
    tmp_path, field, value
):
    import json
    from robo_trader.safety.models import canonical_json, sha256_text

    source = journal_at(tmp_path / "source.sqlite")
    event = reserve(source)
    payload = json.loads(event.payload_json)
    payload[field] = value
    corrupted = journal_at(tmp_path / "corrupted.sqlite")
    # Fault injection bypasses the adapter while retaining valid SQL and hashes.
    corrupted._write_transaction(
        lambda connection: corrupted._append(
            connection,
            event.event_type,
            event.occurred_at,
            event.idempotency_key,
            event.execution_domain_scope,
            event.account_scope,
            event.portfolio_id,
            event.con_id,
            sha256_text(canonical_json(payload)),
            None,
            payload,
        )
    )
    with pytest.raises(JournalError, match="invalid entry capacity"):
        corrupted.replay()
