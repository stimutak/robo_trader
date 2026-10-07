"""Durable entry validation must never submit, append, or repair a fill."""

from datetime import datetime, timezone

import pytest

from robo_trader.paper_entry_persistence import stage_entry_settlement
from robo_trader.paper_entry_settlement import build_paper_entry_terminal_record
from robo_trader.paper_entry_storage_replay import read_entry_settlement
from robo_trader.safety.models import ValidationError
from tests.test_paper_entry_persistence import _snapshot, entry_db  # noqa: F401
from tests.test_paper_entry_terminal_record import case  # noqa: F401


async def _commit(entry_db, case):
    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        await conn.execute("BEGIN IMMEDIATE")
        staged = await stage_entry_settlement(
            conn,
            record,
            database=database,
            runtime_contract=runtime,
            journal=journal,
            committed_at=datetime.now(timezone.utc),
        )
        await conn.commit()
    return record, staged


@pytest.mark.asyncio
async def test_lost_response_can_read_original_storage_without_writes(entry_db, case):
    database, runtime, journal = entry_db
    record, original = await _commit(entry_db, case)
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        changes = conn.total_changes
        await conn.execute("PRAGMA query_only=ON")
        await conn.execute("BEGIN")
        try:
            result = await read_entry_settlement(
                conn, record, database=database, runtime_contract=runtime, journal=journal
            )
            assert result.settlement_id == original.settlement_id
            assert result.trade_id == original.trade_id
            assert result.fingerprint == original.fingerprint
            assert conn.total_changes == changes
            assert await _snapshot(conn) == before
        finally:
            await conn.rollback()
            await conn.execute("PRAGMA query_only=OFF")


@pytest.mark.asyncio
async def test_missing_terminal_does_not_create_anything(entry_db, case):
    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        await conn.execute("BEGIN")
        with pytest.raises(ValidationError, match="absent"):
            await read_entry_settlement(
                conn, record, database=database, runtime_contract=runtime, journal=journal
            )
        await conn.rollback()
        assert await _snapshot(conn) == before


async def _corrupt(conn, table, sql, values=()):
    triggers = await (
        await conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,)
        )
    ).fetchall()
    for name, _ in triggers:
        await conn.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
    await conn.execute(sql, values)
    for _, definition in triggers:
        await conn.execute(definition)
    await conn.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table,sql,values",
    [
        (
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET settlement_kind='REDUCTION'",
            (),
        ),
        (
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET receipt_fingerprint=?",
            ("0" * 64,),
        ),
        (
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET request_payload_json='{}'",
            (),
        ),
        (
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET portfolio_id='other'",
            (),
        ),
        (
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET database_inode=database_inode+1",
            (),
        ),
        (
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET protective_quote_payload='{}'",
            (),
        ),
        ("trades", "UPDATE trades SET price=334", ()),
        ("trades", "UPDATE trades SET side='SELL'", ()),
        ("paper_fifo_settlement_links", "DELETE FROM paper_fifo_settlement_links", ()),
        (
            "paper_fifo_settlement_links",
            "UPDATE paper_fifo_settlement_links SET fifo_state_fingerprint=?",
            ("0" * 64,),
        ),
        ("fifo_fills", "UPDATE fifo_fills SET quantity_text='7'", ()),
    ],
)
async def test_changed_storage_rejects_without_repair(entry_db, case, table, sql, values):
    database, runtime, journal = entry_db
    record, _ = await _commit(entry_db, case)
    async with database.get_connection() as conn:
        await _corrupt(conn, table, sql, values)
        before = await _snapshot(conn)
        changes = conn.total_changes
        await conn.execute("PRAGMA query_only=ON")
        await conn.execute("BEGIN")
        try:
            with pytest.raises(ValidationError):
                await read_entry_settlement(
                    conn, record, database=database, runtime_contract=runtime, journal=journal
                )
            assert conn.total_changes == changes
            assert await _snapshot(conn) == before
        finally:
            await conn.rollback()
            await conn.execute("PRAGMA query_only=OFF")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["REJECTED", "CANCELLED", "EXPIRED"])
async def test_zero_fill_recovery_never_claims_a_trade(entry_db, case, status):
    from dataclasses import replace
    from decimal import Decimal

    from robo_trader.paper_reduction_submitter import LocalPaperOrderStatus

    database, runtime, journal = entry_db
    case["outcome"] = replace(
        case["outcome"],
        status=LocalPaperOrderStatus(status),
        filled_quantity=Decimal("0"),
        remaining_quantity=case["outcome"].requested_quantity,
        exact_fill_price=None,
        fill_evidence=None,
    )
    record, original = await _commit(entry_db, case)
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        await conn.execute("PRAGMA query_only=ON")
        await conn.execute("BEGIN")
        try:
            result = await read_entry_settlement(
                conn, record, database=database, runtime_contract=runtime, journal=journal
            )
            assert result.trade_id is None
            assert result.settlement_id == original.settlement_id
            assert await _snapshot(conn) == before
        finally:
            await conn.rollback()
            await conn.execute("PRAGMA query_only=OFF")


@pytest.mark.asyncio
async def test_recovery_works_with_reopened_database_and_no_quote_producer(entry_db, case):
    import gc

    from robo_trader.database_async import AsyncTradingDatabase

    database, runtime, journal = entry_db
    record, original = await _commit(entry_db, case)
    del case["quote_producer"]
    gc.collect()
    reopened = AsyncTradingDatabase(database.db_path, pool_size=1)
    await reopened.initialize()
    try:
        async with reopened.get_connection() as conn:
            await conn.execute("BEGIN")
            result = await read_entry_settlement(
                conn, record, database=reopened, runtime_contract=runtime, journal=journal
            )
            assert result.fingerprint == original.fingerprint
            await conn.rollback()
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_same_claim_with_different_accounting_cannot_replay_or_overwrite(entry_db, case):
    from dataclasses import replace
    from decimal import Decimal

    database, runtime, journal = entry_db
    await _commit(entry_db, case)
    case["account"] = replace(case["account"], cash=Decimal("99999"))
    altered = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        await conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValidationError, match="differs"):
            await stage_entry_settlement(
                conn,
                altered,
                database=database,
                runtime_contract=runtime,
                journal=journal,
                committed_at=datetime.now(timezone.utc),
            )
        await conn.commit()
        assert await _snapshot(conn) == before


@pytest.mark.asyncio
async def test_database_path_replacement_during_recovery_rejects(
    entry_db, case, monkeypatch, tmp_path
):
    import sqlite3

    database, runtime, journal = entry_db
    record, _ = await _commit(entry_db, case)
    original = database._sqlite_descriptor_identity
    parked = tmp_path / "parked-ledger.db"
    calls = 0

    async def replaced(conn):
        nonlocal calls
        calls += 1
        if calls == 2:
            database.db_path.rename(parked)
            sqlite3.connect(database.db_path).close()
        return await original(conn)

    monkeypatch.setattr(database, "_sqlite_descriptor_identity", replaced)
    async with database.get_connection() as conn:
        await conn.execute("BEGIN")
        try:
            with pytest.raises(ValidationError, match="verification failed"):
                await read_entry_settlement(
                    conn, record, database=database, runtime_contract=runtime, journal=journal
                )
        finally:
            if parked.exists():
                database.db_path.unlink()
                parked.rename(database.db_path)
            await conn.rollback()
