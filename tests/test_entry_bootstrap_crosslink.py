"""Reconciliation crosslinks released entries to actual temporary ledger storage."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import sqlite3

import pytest

from robo_trader.paper_entry_receipt import recover_committed_entry_receipt
from robo_trader.paper_entry_release import release_entry_capacity
from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
from robo_trader.safety import SafetyJournal
from robo_trader.reconciliation.bootstrap_producer import (
    _crosslink_safety_journal_orders,
    BootstrapReconciliationBlocked,
)
from tests.risk.test_paper_entry_ledger_snapshot import bootstrapped_entry  # noqa: F401
from tests.test_exact_state_bootstrap import _bootstrap_evidence_keys  # noqa: F401
from tests.risk.test_daily_filled_notional import MutableClock, _service
from tests.test_paper_entry_storage_replay import _commit, _corrupt


async def _release(entry_db, case, tmp_path):
    database, runtime, _ = entry_db
    record, _ = await _commit(entry_db, case)
    journal = SafetyJournal(runtime.safety_journal_path)
    receipt = await recover_committed_entry_receipt(
        record, database=database, runtime_contract=runtime, journal=journal
    )
    ledger = _service(
        tmp_path / "daily.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        portfolio_id="default",
        clock=MutableClock(datetime.now(timezone.utc)),
    )
    accounting = PaperFillAccounting(runtime, {"default": ledger})
    confirmation = await accounting.confirm_entry_settlement(receipt, database=database)
    head = journal.replay()
    release_entry_capacity(
        journal,
        receipt,
        confirmation,
        database=database,
        accounting=accounting,
        expected_head=(head.last_sequence, head.last_chain_hash),
    )
    return journal


def _crosslink(entry_db, journal):
    database, runtime, _ = entry_db
    with sqlite3.connect(database.db_path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        trades = tuple(
            conn.execute(
                "SELECT id,portfolio_id,symbol,side,quantity,price,notional,slippage,commission,pnl,timestamp FROM trades ORDER BY id"
            )
        )
        result = _crosslink_safety_journal_orders(
            connection=conn,
            actual_tables=tables,
            replay_state=journal.replay(),
            trade_rows=trades,
            runtime=runtime,
            database_device=database.db_path.stat().st_dev,
            database_inode=database.db_path.stat().st_ino,
        )
        assert conn.total_changes == 0
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FILLED", "REJECTED", "CANCELLED", "EXPIRED"])
async def test_released_entry_crosslinks_without_writes(bootstrapped_entry, tmp_path, status):
    from robo_trader.paper_reduction_submitter import LocalPaperOrderStatus

    entry_db, case, _ = bootstrapped_entry
    if status != "FILLED":
        case["outcome"] = replace(
            case["outcome"],
            status=LocalPaperOrderStatus(status),
            filled_quantity=Decimal("0"),
            remaining_quantity=case["outcome"].requested_quantity,
            exact_fill_price=None,
            fill_evidence=None,
        )
    journal = await _release(entry_db, case, tmp_path)
    assert _crosslink(entry_db, journal) == (True, True, 1, int(status == "FILLED"))


@pytest.mark.asyncio
async def test_unreleased_entry_stays_blocked(bootstrapped_entry):
    entry_db, case, _ = bootstrapped_entry
    await _commit(entry_db, case)
    with pytest.raises(BootstrapReconciliationBlocked, match="unresolved entry"):
        _crosslink(entry_db, entry_db[2])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table,sql",
    [
        ("trades", "UPDATE trades SET quantity=5"),
        ("paper_fifo_settlement_links", "DELETE FROM paper_fifo_settlement_links"),
        (
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET receipt_fingerprint='bad'",
        ),
    ],
)
async def test_released_entry_with_damaged_storage_is_blocked(
    bootstrapped_entry, tmp_path, table, sql
):
    entry_db, case, _ = bootstrapped_entry
    journal = await _release(entry_db, case, tmp_path)
    async with entry_db[0].get_connection() as conn:
        await _corrupt(conn, table, sql)
    with pytest.raises(BootstrapReconciliationBlocked):
        _crosslink(entry_db, journal)


@pytest.mark.asyncio
async def test_full_held_bootstrap_collection_validates_released_entry(
    bootstrapped_entry, tmp_path
):
    from robo_trader.reconciliation.bootstrap_producer import _collect_wal_visible_ledger

    entry_db, case, _ = bootstrapped_entry
    await _release(entry_db, case, tmp_path)
    with _collect_wal_visible_ledger(
        entry_db[1], observed_at=datetime.now(timezone.utc)
    ) as session:
        assert session.evidence.terminal_settlement_count == 1
        assert session.evidence.terminal_fill_count == 1
        assert session._connection.row_factory is sqlite3.Row
        assert session._connection.total_changes == 0
        with pytest.raises(sqlite3.DatabaseError):
            session._connection.execute("PRAGMA foreign_keys=OFF")
        session.assert_unchanged_after_receiver_claim()


@pytest.mark.asyncio
async def test_hash_valid_release_must_match_the_actual_committed_fill(
    bootstrapped_entry, tmp_path
):
    from tests.risk.test_entry_capacity_release import rewrite_release_as_zero_fill

    entry_db, case, _ = bootstrapped_entry
    journal = await _release(entry_db, case, tmp_path)
    rewrite_release_as_zero_fill(journal, journal.replay().events[-1])
    # Structural journal replay still succeeds: DB crosslink must reject the
    # false no-fill assertion despite its valid hashes and unchanged references.
    assert journal.replay().pending_entry_events == ()
    with pytest.raises(BootstrapReconciliationBlocked, match="entry terminal crosslink"):
        _crosslink(entry_db, journal)
