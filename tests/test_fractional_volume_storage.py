from decimal import Decimal

import pytest
import pytest_asyncio

from robo_trader.database_async import AsyncTradingDatabase
from tests.test_fractional_volume_contract import batch


@pytest_asyncio.fixture
async def database(tmp_path):
    db = AsyncTradingDatabase(tmp_path / "volume.db", pool_size=1)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["10.0000000000000001", "1000000000000000001.25", "0", "1e127"])
async def test_v2_store_is_exact_idempotent_and_preserves_v1(database, raw):
    old = batch(10, version=1).storage_rows()
    await database.batch_store_market_data(old)
    async with database.get_connection() as conn:
        before = await (await conn.execute("SELECT * FROM canonical_market_data")).fetchall()
    rows = batch(raw).storage_rows()
    await database.batch_store_market_data(rows)
    await database.batch_store_market_data(rows)
    async with database.get_connection() as conn:
        saved = await (
            await conn.execute(
                "SELECT volume, typeof(volume), volume_unit FROM canonical_market_data_v2"
            )
        ).fetchall()
        after = await (await conn.execute("SELECT * FROM canonical_market_data")).fetchall()
    assert before == after
    assert len(saved) == 1
    assert saved[0][1:] == ("text", "unknown")
    assert Decimal(saved[0][0]) == Decimal(raw)


@pytest.mark.asyncio
async def test_v2_conflicting_batch_does_not_partially_write(database):
    original = batch("10.25").storage_rows()[0]
    await database.batch_store_market_data([original])
    fresh = {**original, "timestamp": "2026-07-23T15:00:00+00:00"}
    conflict = {**original, "volume": "10.2500000000000001"}
    with pytest.raises(ValueError, match="conflicts"):
        await database.batch_store_market_data([fresh, conflict])
    async with database.get_connection() as conn:
        saved = await (
            await conn.execute("SELECT timestamp, volume FROM canonical_market_data_v2")
        ).fetchall()
    assert saved == [(original["timestamp"], "10.25")]


@pytest.mark.asyncio
async def test_mixed_versions_rejected_before_writes(database):
    with pytest.raises(ValueError, match="mix.*version"):
        await database.batch_store_market_data(
            batch(10, version=1).storage_rows() + batch("10.25").storage_rows()
        )
    async with database.get_connection() as conn:
        assert await (
            await conn.execute("SELECT COUNT(*) FROM canonical_market_data")
        ).fetchone() == (0,)
        assert await (
            await conn.execute("SELECT COUNT(*) FROM canonical_market_data_v2")
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_adding_v2_table_to_existing_database_preserves_v1_schema_and_rows(database):
    await database.batch_store_market_data(batch(10, version=1).storage_rows())
    async with database.get_connection() as conn:
        before = await (await conn.execute("SELECT * FROM canonical_market_data")).fetchall()
        schema_before = await (
            await conn.execute("SELECT sql FROM sqlite_master WHERE name = 'canonical_market_data'")
        ).fetchone()
        # Only this test's temporary database: simulate a pre-v2 installation.
        await conn.execute("DROP TABLE canonical_market_data_v2")
        await conn.commit()
    await database.close()
    reopened = AsyncTradingDatabase(database.db_path, pool_size=1)
    await reopened.initialize()
    try:
        async with reopened.get_connection() as conn:
            after = await (await conn.execute("SELECT * FROM canonical_market_data")).fetchall()
            schema_after = await (
                await conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = 'canonical_market_data'"
                )
            ).fetchone()
            assert await (
                await conn.execute("SELECT COUNT(*) FROM canonical_market_data_v2")
            ).fetchone() == (0,)
        assert after == before
        assert schema_after == schema_before
    finally:
        await reopened.close()
