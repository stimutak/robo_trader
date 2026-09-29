"""Additive terminal kinds preserve actual committed reduction evidence."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.database_migrations import (
    apply_exact_state_migrations,
    assert_paper_settlement_hot_schema,
)
from tests.test_pr2b3_terminal_settlement_persistence import _request, _runtime_contract, _seed


@pytest.mark.asyncio
async def test_v3_upgrade_preserves_reduction_and_replay(tmp_path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        request = _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        receipt = await database.commit_paper_reduction_outcome(request, runtime_contract=contract)
        async with database.get_connection() as conn:
            columns = [
                row[1]
                for row in await (
                    await conn.execute("PRAGMA table_info(paper_reduction_settlements)")
                ).fetchall()
            ]
            assert "settlement_kind" in columns
            # Only this synthetic test ledger is downgraded to the historical
            # v3 schema; all pre-existing payload/receipt/FIFO data is retained.
            await conn.execute(
                "ALTER TABLE paper_reduction_settlements DROP COLUMN settlement_kind"
            )
            await conn.execute(
                "DELETE FROM rt_schema_migrations WHERE component='paper_exact_state' AND version=4"
            )
            before = await (
                await conn.execute("SELECT * FROM paper_reduction_settlements")
            ).fetchall()
            await conn.commit()
            await conn.execute("BEGIN IMMEDIATE")
            await apply_exact_state_migrations(conn)
            await conn.rollback()
            assert "settlement_kind" not in {
                row[1]
                for row in await (
                    await conn.execute("PRAGMA table_info(paper_reduction_settlements)")
                ).fetchall()
            }
            assert (
                await (await conn.execute("SELECT * FROM paper_reduction_settlements")).fetchall()
                == before
            )
            await conn.execute("BEGIN IMMEDIATE")
            await apply_exact_state_migrations(conn)
            await assert_paper_settlement_hot_schema(conn)
            await conn.commit()
            after = await (
                await conn.execute("SELECT * FROM paper_reduction_settlements")
            ).fetchall()
            assert [row[:-1] for row in after] == before
            assert [row[-1] for row in after] == ["REDUCTION"]
            await apply_exact_state_migrations(conn)
            await conn.commit()
            assert (
                await (await conn.execute("SELECT * FROM paper_reduction_settlements")).fetchall()
                == after
            )
        replay = [r async for r in database.iter_paper_terminal_receipts(runtime_contract=contract)]
        assert len(replay) == 1
        assert replay[0].fingerprint() == receipt.fingerprint()
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "entry", "OTHER", 1])
async def test_invalid_kind_rejected_by_sql(tmp_path, kind):
    database = AsyncTradingDatabase(tmp_path / "kinds.db", pool_size=1)
    await database.initialize()
    try:
        async with database.get_connection() as conn:
            with pytest.raises(sqlite3.IntegrityError):
                await conn.execute(
                    """INSERT INTO paper_reduction_settlements
                    (settlement_id,execution_domain_scope,account_scope,portfolio_id,con_id,
                    symbol,reservation_id,claim_id,order_ref,protective_quote_payload,
                    request_fingerprint,request_payload_json,terminal_status,database_path,
                    database_identity,database_device,database_inode,committed_at,
                    receipt_fingerprint,schema_version,settlement_kind)
                    VALUES ('id','paper','account','p',1,'A','r','c','o','{}','h','{}',
                    'REJECTED','path','identity',1,1,'time','hash',1,?)""",
                    (kind,),
                )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_reduction_outbox_and_writer_reject_relabelled_reduction(tmp_path):
    from robo_trader.database_migrations import _PAPER_REDUCTION_SETTLEMENT_TRIGGER_SQL
    from robo_trader.safety.models import ValidationError

    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        request = _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        await database.commit_paper_reduction_outcome(request, runtime_contract=contract)
        async with database.get_connection() as conn:
            await conn.execute("DROP TRIGGER paper_reduction_settlements_no_update")
            await conn.execute("UPDATE paper_reduction_settlements SET settlement_kind='ENTRY'")
            await conn.execute(
                _PAPER_REDUCTION_SETTLEMENT_TRIGGER_SQL["paper_reduction_settlements_no_update"]
            )
            await conn.commit()
        with pytest.raises(ValidationError, match="mixed settlement recovery"):
            [r async for r in database.iter_paper_terminal_receipts(runtime_contract=contract)]
        from robo_trader.paper_terminal_settlement import PaperTerminalSettlementError

        with pytest.raises(PaperTerminalSettlementError):
            await database.commit_paper_reduction_outcome(request, runtime_contract=contract)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_missing_kind_constraint_rejected_by_hot_audit(tmp_path):
    database = AsyncTradingDatabase(tmp_path / "tampered.db", pool_size=1)
    await database.initialize()
    try:
        async with database.get_connection() as conn:
            await conn.execute(
                "ALTER TABLE paper_reduction_settlements DROP COLUMN settlement_kind"
            )
            await conn.execute(
                "ALTER TABLE paper_reduction_settlements ADD COLUMN settlement_kind TEXT NOT NULL DEFAULT 'REDUCTION'"
            )
            await conn.execute("BEGIN IMMEDIATE")
            with pytest.raises(RuntimeError, match="definition is malformed"):
                await assert_paper_settlement_hot_schema(conn)
    finally:
        await database.close()


def test_bootstrap_crosslink_rejects_unverifiable_entry_history():
    from types import SimpleNamespace

    from robo_trader.reconciliation.bootstrap_producer import (
        BootstrapReconciliationBlocked,
        _crosslink_safety_journal_orders,
    )

    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE paper_reduction_settlements (settlement_kind TEXT)")
        connection.execute("INSERT INTO paper_reduction_settlements VALUES ('ENTRY')")
        with pytest.raises(BootstrapReconciliationBlocked, match="entry terminal crosslink"):
            _crosslink_safety_journal_orders(
                connection=connection,
                actual_tables={"paper_reduction_settlements"},
                replay_state=SimpleNamespace(),
                trade_rows=(),
                runtime=None,
                database_device=1,
                database_inode=1,
            )
