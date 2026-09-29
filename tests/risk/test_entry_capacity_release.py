"""Entry capacity is released only by committed and independently accounted outcomes."""

from dataclasses import replace
from decimal import Decimal

import pytest

from robo_trader.safety import SafetyJournal
from tests.risk.test_entry_accounting_confirmation import _setup
from tests.test_paper_entry_terminal_record import case  # noqa: F401
from tests.test_paper_entry_persistence import entry_db  # noqa: F401


async def _prepared(entry_db, case, tmp_path):
    receipt, risk, accounting = await _setup(entry_db, case, tmp_path)
    confirmation = await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    journal = SafetyJournal(entry_db[1].safety_journal_path)
    state = journal.replay()
    return receipt, accounting, confirmation, journal, (state.last_sequence, state.last_chain_hash)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FILLED", "REJECTED", "CANCELLED", "EXPIRED"])
async def test_release_survives_restart_and_exact_retry(entry_db, case, tmp_path, status):
    from robo_trader.paper_entry_release import release_entry_capacity
    from robo_trader.paper_reduction_submitter import LocalPaperOrderStatus
    from robo_trader.risk.entry_reservations import summarize_entry_capacity

    if status != "FILLED":
        case["outcome"] = replace(
            case["outcome"],
            status=LocalPaperOrderStatus(status),
            filled_quantity=Decimal("0"),
            remaining_quantity=case["outcome"].requested_quantity,
            exact_fill_price=None,
            fill_evidence=None,
        )
    receipt, accounting, confirmation, journal, head = await _prepared(entry_db, case, tmp_path)
    released = release_entry_capacity(
        journal,
        receipt,
        confirmation,
        database=entry_db[0],
        accounting=accounting,
        expected_head=head,
    )
    replay = SafetyJournal(journal.database_path).replay()
    assert replay.pending_entry_events == ()
    assert replay.events[-1] == released
    totals = summarize_entry_capacity(
        replay, portfolio_id="portfolio-alpha", symbol="AAPL", sector="Technology", held_symbols=()
    )
    assert totals.pending_account_notional_usd == Decimal("0")
    confirmation = await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    assert (
        release_entry_capacity(
            journal,
            receipt,
            confirmation,
            database=entry_db[0],
            accounting=accounting,
            expected_head=head,
        )
        == released
    )
    assert journal.replay() == replay


@pytest.mark.asyncio
async def test_release_rejects_stale_head_without_changing_capacity(entry_db, case, tmp_path):
    from robo_trader.paper_entry_release import release_entry_capacity
    from robo_trader.safety.journal import StateTransitionError

    receipt, accounting, confirmation, journal, _ = await _prepared(entry_db, case, tmp_path)
    before = journal.replay()
    with pytest.raises(StateTransitionError):
        release_entry_capacity(
            journal,
            receipt,
            confirmation,
            database=entry_db[0],
            accounting=accounting,
            expected_head=(0, "0" * 64),
        )
    assert journal.replay() == before


@pytest.mark.asyncio
async def test_commit_failure_retains_capacity_and_retry_needs_fresh_confirmation(
    entry_db, case, tmp_path
):
    from robo_trader.paper_entry_release import release_entry_capacity
    from robo_trader.safety.models import ValidationError

    receipt, accounting, confirmation, journal, head = await _prepared(entry_db, case, tmp_path)
    before = journal.replay()

    def fail(stage, event):
        raise OSError("release commit failed")

    journal._fault_hook = fail
    with pytest.raises(OSError, match="release commit failed"):
        release_entry_capacity(
            journal,
            receipt,
            confirmation,
            database=entry_db[0],
            accounting=accounting,
            expected_head=head,
        )
    journal._fault_hook = None
    assert journal.replay() == before
    with pytest.raises(ValidationError):
        release_entry_capacity(
            journal,
            receipt,
            confirmation,
            database=entry_db[0],
            accounting=accounting,
            expected_head=head,
        )
    fresh = await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    release_entry_capacity(
        journal, receipt, fresh, database=entry_db[0], accounting=accounting, expected_head=head
    )
    assert journal.replay().pending_entry_events == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("forged", ["receipt", "confirmation"])
async def test_copied_evidence_cannot_release(entry_db, case, tmp_path, forged):
    from robo_trader.paper_entry_release import release_entry_capacity
    from robo_trader.safety.models import ValidationError

    receipt, accounting, confirmation, journal, head = await _prepared(entry_db, case, tmp_path)
    before = journal.replay()
    if forged == "receipt":
        receipt = replace(receipt)
    else:
        confirmation = replace(confirmation)
    with pytest.raises(ValidationError):
        release_entry_capacity(
            journal,
            receipt,
            confirmation,
            database=entry_db[0],
            accounting=accounting,
            expected_head=head,
        )
    assert journal.replay() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["parent", "quantity", "status", "accounting", "duplicate"])
async def test_replay_rejects_hash_valid_invalid_release(entry_db, case, tmp_path, mutation):
    import json
    from robo_trader.paper_entry_release import release_entry_capacity
    from robo_trader.safety.journal import JournalIntegrityError

    receipt, accounting, confirmation, journal, head = await _prepared(entry_db, case, tmp_path)
    before = journal.replay()
    event = release_entry_capacity(
        journal,
        receipt,
        confirmation,
        database=entry_db[0],
        accounting=accounting,
        expected_head=head,
    )
    payload = json.loads(event.payload_json)
    if mutation == "parent":
        payload["claim_chain_hash"] = "a" * 64
    elif mutation == "quantity":
        payload["filled_quantity"] = "5"
    elif mutation == "status":
        payload["terminal_status"] = "PENDING"
    elif mutation == "accounting":
        payload["gross_filled_notional"] = "0"

    from robo_trader.safety.entry_release import validate_entry_release_event
    from robo_trader.safety.journal import IdempotencyConflict

    if mutation != "duplicate":
        with pytest.raises(JournalIntegrityError):
            validate_entry_release_event(event, payload, before.events[0], before.events[1])

    # Exercise a hash-valid row through the append machinery. Reusing the real
    # release key is also prohibited before a duplicate can reach storage.
    def append_bad(conn):
        return journal._append(
            conn,
            event.event_type,
            event.occurred_at,
            (
                event.idempotency_key
                if mutation == "duplicate"
                else event.idempotency_key + "-corrupt"
            ),
            event.execution_domain_scope,
            event.account_scope,
            event.portfolio_id,
            event.con_id,
            event.intent_fingerprint,
            event.claim_id,
            payload,
        )

    if mutation == "duplicate":
        with pytest.raises(IdempotencyConflict):
            journal._write_transaction(append_bad)
    else:
        journal._write_transaction(append_bad)
        with pytest.raises(JournalIntegrityError):
            SafetyJournal(journal.database_path).replay()


@pytest.mark.asyncio
async def test_release_key_cannot_be_reused_by_another_journal_path(entry_db, case, tmp_path):
    from robo_trader.paper_entry_release import release_entry_capacity
    from robo_trader.safety.models import JournalEventType
    from robo_trader.safety.journal import IdempotencyConflict

    receipt, accounting, confirmation, journal, head = await _prepared(entry_db, case, tmp_path)
    event = release_entry_capacity(
        journal,
        receipt,
        confirmation,
        database=entry_db[0],
        accounting=accounting,
        expected_head=head,
    )
    before = journal.replay()

    def reuse(conn):
        return journal._append(
            conn,
            JournalEventType.SAFETY_DECISION,
            event.occurred_at,
            event.idempotency_key,
            event.execution_domain_scope,
            event.account_scope,
            event.portfolio_id,
            event.con_id,
            event.intent_fingerprint,
            None,
            {},
        )

    with pytest.raises(IdempotencyConflict):
        journal._write_transaction(reuse)
    assert journal.replay() == before


@pytest.mark.asyncio
async def test_retry_rejects_hash_valid_release_that_misstates_actual_fill(
    entry_db, case, tmp_path
):
    from robo_trader.paper_entry_release import release_entry_capacity
    from robo_trader.safety.journal import StateTransitionError

    receipt, accounting, confirmation, journal, head = await _prepared(entry_db, case, tmp_path)
    event = release_entry_capacity(
        journal,
        receipt,
        confirmation,
        database=entry_db[0],
        accounting=accounting,
        expected_head=head,
    )
    rewrite_release_as_zero_fill(journal, event)
    confirmation = await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    with pytest.raises(StateTransitionError):
        release_entry_capacity(
            journal,
            receipt,
            confirmation,
            database=entry_db[0],
            accounting=accounting,
            expected_head=head,
        )


@pytest.mark.asyncio
async def test_release_is_exact_under_small_decimal_context(entry_db, case, tmp_path):
    from decimal import localcontext
    from robo_trader.paper_entry_release import release_entry_capacity

    receipt, accounting, confirmation, journal, head = await _prepared(entry_db, case, tmp_path)
    with localcontext() as context:
        context.prec = 2
        release_entry_capacity(
            journal,
            receipt,
            confirmation,
            database=entry_db[0],
            accounting=accounting,
            expected_head=head,
        )
        assert journal.replay().pending_entry_events == ()


def rewrite_release_as_zero_fill(journal, event):
    """Corrupt only a temporary test journal while keeping its hash chain valid."""
    import json
    import sqlite3
    from robo_trader.safety.models import canonical_json, sha256_text, utc_to_text

    payload = json.loads(event.payload_json)
    payload.update(
        terminal_status="REJECTED",
        filled_quantity="0",
        fill_price=None,
        fill_execution_id=None,
        fill_notional="0",
    )
    encoded = canonical_json(payload)
    payload_hash = sha256_text(encoded)
    chain_hash = journal._chain_hash(
        event.sequence,
        event.event_type,
        utc_to_text(event.occurred_at),
        event.idempotency_key,
        event.execution_domain_scope,
        event.account_scope,
        event.portfolio_id,
        event.con_id,
        event.intent_fingerprint,
        event.claim_id,
        payload_hash,
        event.previous_chain_hash,
        event.schema_version,
    )
    with sqlite3.connect(journal.database_path) as conn:
        triggers = conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name='safety_journal_events'"
        ).fetchall()
        for name, _ in triggers:
            conn.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        conn.execute(
            "UPDATE safety_journal_events SET payload_json=?,payload_hash=?,chain_hash=? WHERE sequence=?",
            (encoded, payload_hash, chain_hash, event.sequence),
        )
        for _, sql in triggers:
            conn.execute(sql)
