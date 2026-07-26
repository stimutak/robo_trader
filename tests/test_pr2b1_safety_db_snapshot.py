"""Focused tests for the PR 2B.1 cross-portfolio safety snapshot."""

import asyncio
import sqlite3
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import aiosqlite
import pytest

from robo_trader.config import RuntimeContract
from robo_trader.database_async import (
    AsyncTradingDatabase,
    SafetyAllocationSnapshotError,
)


def _timestamp(*, age_seconds: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()


@pytest.fixture
async def allocation_db(tmp_path):
    database = AsyncTradingDatabase(db_path=tmp_path / "safety-allocation.db", pool_size=2)
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


async def _insert_portfolio(
    database: AsyncTradingDatabase, portfolio_id: str, *, active: int = 1
) -> None:
    async with database.get_connection() as conn:
        await conn.execute(
            """
            INSERT INTO portfolios (id, name, starting_cash, active)
            VALUES (?, ?, 100000, ?)
            """,
            (portfolio_id, portfolio_id, active),
        )
        await conn.commit()


async def _insert_position(
    database: AsyncTradingDatabase,
    portfolio_id: str,
    quantity: object,
    *,
    symbol: str = "AAPL",
    timestamp: str | None = None,
) -> None:
    async with database.get_connection() as conn:
        await conn.execute(
            """
            INSERT INTO positions
                (portfolio_id, symbol, quantity, avg_cost, market_price, timestamp)
            VALUES (?, ?, ?, 100.0, 100.0, ?)
            """,
            (portfolio_id, symbol, quantity, timestamp or _timestamp()),
        )
        await conn.commit()


async def _snapshot(
    database: AsyncTradingDatabase,
    symbol: str = "AAPL",
):
    return await database.get_safety_allocation_snapshot(
        symbol,
        runtime_contract=RuntimeContract(
            environment="dev",
            execution_mode="paper",
            execution_source="paper_simulator",
            ibkr_host="127.0.0.1",
            ibkr_port=4002,
            ibkr_readonly=True,
            database_path=str(database.db_path),
            account_alias="***1234",
            account_type="paper",
            model_artifact_set="test-models",
            build_id="test-build",
            state_namespace="paper",
        ),
    )


@pytest.mark.asyncio
async def test_snapshot_computes_exact_aggregate_offsetting_and_zero_allocations(
    allocation_db,
):
    await _insert_portfolio(allocation_db, "long_book")
    await _insert_portfolio(allocation_db, "short_book")
    await _insert_portfolio(allocation_db, "inactive_book", active=0)
    await _insert_position(allocation_db, "long_book", 10)
    await _insert_position(allocation_db, "short_book", -3)

    before_snapshot = datetime.now(timezone.utc)
    snapshot = await _snapshot(allocation_db, "aapl")
    after_snapshot = datetime.now(timezone.utc)

    assert snapshot.symbol == "AAPL"
    assert snapshot.aggregate_allocated_quantity == Decimal("7")
    assert snapshot.has_offsetting_allocations is True
    assert snapshot.complete is True
    assert snapshot.observed_at.tzinfo is timezone.utc
    assert before_snapshot <= snapshot.observed_at <= after_snapshot
    assert snapshot.snapshot_id.startswith("allocation-db-")
    assert not hasattr(snapshot, "con_id")
    assert all(not hasattr(row, "con_id") for row in snapshot.allocations)

    allocations = {allocation.portfolio_id: allocation for allocation in snapshot.allocations}
    assert allocations["long_book"].quantity == Decimal("10")
    assert allocations["short_book"].quantity == Decimal("-3")
    assert allocations["default"].quantity == Decimal("0")
    assert allocations["default"].updated_at is None
    assert allocations["inactive_book"].quantity == Decimal("0")
    assert all(type(row.quantity) is Decimal for row in snapshot.allocations)

    with pytest.raises(FrozenInstanceError):
        snapshot.complete = False
    with pytest.raises(FrozenInstanceError):
        snapshot.allocations[0].quantity = Decimal("999")

    second = await _snapshot(allocation_db)
    assert second.snapshot_id != snapshot.snapshot_id
    assert second.observed_at >= snapshot.observed_at


@pytest.mark.asyncio
async def test_snapshot_is_one_coherent_read_during_concurrent_writer(
    allocation_db,
):
    await _insert_portfolio(allocation_db, "book_a")
    await _insert_portfolio(allocation_db, "book_b")
    await _insert_position(allocation_db, "book_a", 10)
    await _insert_position(allocation_db, "book_b", -4)

    writer_started = asyncio.Event()

    async def writer() -> None:
        async with aiosqlite.connect(allocation_db.db_path) as conn:
            await conn.execute("PRAGMA busy_timeout=5000")
            for iteration in range(40):
                quantities = (20, -14) if iteration % 2 else (10, -4)
                await conn.execute("BEGIN IMMEDIATE")
                await conn.execute(
                    """
                    UPDATE positions
                    SET quantity = ?, timestamp = ?
                    WHERE portfolio_id = 'book_a' AND symbol = 'AAPL'
                    """,
                    (quantities[0], _timestamp()),
                )
                writer_started.set()
                # A non-transactional or per-portfolio reader could observe a
                # torn aggregate during this yield.
                await asyncio.sleep(0)
                await conn.execute(
                    """
                    UPDATE positions
                    SET quantity = ?, timestamp = ?
                    WHERE portfolio_id = 'book_b' AND symbol = 'AAPL'
                    """,
                    (quantities[1], _timestamp()),
                )
                await conn.commit()
                await asyncio.sleep(0)

    writer_task = asyncio.create_task(writer())
    await writer_started.wait()
    snapshots = [await _snapshot(allocation_db) for _ in range(40)]
    await writer_task

    for snapshot in snapshots:
        quantities = {row.portfolio_id: row.quantity for row in snapshot.allocations}
        assert snapshot.aggregate_allocated_quantity == Decimal("6")
        assert snapshot.has_offsetting_allocations is True
        assert (quantities["book_a"], quantities["book_b"]) in {
            (Decimal("10"), Decimal("-4")),
            (Decimal("20"), Decimal("-14")),
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("quantity", ["shares", 1.5])
async def test_snapshot_rejects_non_integer_quantity_storage(allocation_db, quantity):
    await _insert_portfolio(allocation_db, "malformed")
    await _insert_position(allocation_db, "malformed", quantity)

    with pytest.raises(SafetyAllocationSnapshotError, match="not stored as an integer"):
        await _snapshot(allocation_db)


@pytest.mark.asyncio
async def test_snapshot_rejects_noncanonical_stored_symbol(allocation_db):
    await _insert_portfolio(allocation_db, "malformed")
    await _insert_position(allocation_db, "malformed", 10, symbol="aapl")

    with pytest.raises(SafetyAllocationSnapshotError, match="noncanonical or mismatched symbol"):
        await _snapshot(allocation_db)


@pytest.mark.asyncio
async def test_snapshot_rejects_invalid_portfolio_id(allocation_db):
    await _insert_portfolio(allocation_db, "bad id")
    await _insert_position(allocation_db, "bad id", 10)

    with pytest.raises(SafetyAllocationSnapshotError, match="invalid portfolio_id"):
        await _snapshot(allocation_db)


@pytest.mark.asyncio
async def test_snapshot_rejects_orphaned_position(allocation_db):
    await _insert_position(allocation_db, "orphan", 10)

    with pytest.raises(SafetyAllocationSnapshotError, match="orphaned"):
        await _snapshot(allocation_db)


@pytest.mark.asyncio
async def test_snapshot_distinguishes_historical_mutation_time_from_fresh_observation(
    allocation_db,
):
    await _insert_portfolio(allocation_db, "stale")
    await _insert_position(allocation_db, "stale", 10, timestamp=_timestamp(age_seconds=86400))

    snapshot = await _snapshot(allocation_db)
    allocation = next(row for row in snapshot.allocations if row.portfolio_id == "stale")
    assert snapshot.observed_at - allocation.updated_at > timedelta(hours=23)
    assert datetime.now(timezone.utc) - snapshot.observed_at < timedelta(seconds=1)

    async with allocation_db.get_connection() as conn:
        await conn.execute("""
            UPDATE positions
            SET timestamp = 'not-a-time'
            WHERE portfolio_id = 'stale'
            """)
        await conn.commit()

    with pytest.raises(SafetyAllocationSnapshotError, match="invalid timestamp"):
        await _snapshot(allocation_db)

    async with allocation_db.get_connection() as conn:
        await conn.execute(
            """
            UPDATE positions
            SET timestamp = ?
            WHERE portfolio_id = 'stale'
            """,
            ((datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),),
        )
        await conn.commit()

    with pytest.raises(SafetyAllocationSnapshotError, match="future timestamp"):
        await _snapshot(allocation_db)


@pytest.mark.asyncio
async def test_snapshot_rejects_duplicate_allocation_rows(tmp_path):
    database_path = tmp_path / "duplicate-allocations.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute("""
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                market_price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

    database = AsyncTradingDatabase(database_path, pool_size=1)
    await database.initialize()
    try:
        await _insert_portfolio(database, "duplicate")
        await _insert_position(database, "duplicate", 10)
        await _insert_position(database, "duplicate", 20)

        with pytest.raises(SafetyAllocationSnapshotError, match="duplicate allocation"):
            await _snapshot(database)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_snapshot_does_not_delete_or_modify_user_data(allocation_db):
    await _insert_portfolio(allocation_db, "preserved")
    await _insert_position(allocation_db, "preserved", 11)
    await _insert_position(allocation_db, "preserved", 5, symbol="MSFT")

    def read_rows():
        with sqlite3.connect(allocation_db.db_path) as conn:
            return conn.execute("""
                SELECT id, portfolio_id, symbol, quantity, avg_cost,
                       market_price, timestamp
                FROM positions
                ORDER BY id
                """).fetchall()

    before = read_rows()
    await _snapshot(allocation_db)
    after = read_rows()

    assert after == before


@pytest.mark.asyncio
async def test_snapshot_uses_one_select_on_a_read_only_connection(allocation_db, monkeypatch):
    await _insert_portfolio(allocation_db, "observed")
    await _insert_position(allocation_db, "observed", 4)

    statements = []
    connect_calls = []
    original_execute = aiosqlite.Connection.execute
    original_connect = aiosqlite.connect

    async def tracked_execute(self, sql, *args, **kwargs):
        statements.append(sql.strip())
        return await original_execute(self, sql, *args, **kwargs)

    def tracked_connect(database, *args, **kwargs):
        connect_calls.append((database, kwargs.copy()))
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(aiosqlite.Connection, "execute", tracked_execute)
    monkeypatch.setattr(aiosqlite, "connect", tracked_connect)

    await _snapshot(allocation_db)

    select_calls = [statement for statement in statements if statement.upper().startswith("SELECT")]
    assert len(select_calls) == 1
    assert sum(statement.upper() == "BEGIN" for statement in statements) == 1
    assert not any(
        statement.upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE"))
        for statement in statements
    )
    assert len(connect_calls) == 1
    database_uri, connect_kwargs = connect_calls[0]
    assert database_uri.endswith("?mode=ro")
    assert connect_kwargs["uri"] is True
