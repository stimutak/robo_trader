import pytest

from robo_trader.market_data_contract import MarketDataContractError
from sync_db_reader import MarketDataReadError, SyncDatabaseReader
from tests.test_fractional_volume_contract import batch
from tests.test_fractional_volume_storage import database  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe", [None, "1 min"])
async def test_readers_select_v2_on_tie_without_mixing_v1(database, timeframe):
    old = batch(10, version=1).storage_rows()[0]
    await database.batch_store_market_data([old, {**old, "timestamp": "2026-07-23T15:00:00+00:00"}])
    await database.batch_store_market_data(batch("10.0000000000000001").storage_rows())
    sync = SyncDatabaseReader(str(database.db_path))
    for result in (
        await database.get_latest_market_data("AAPL", timeframe=timeframe),
        sync.get_latest_market_data("AAPL", timeframe=timeframe),
    ):
        assert len(result) == 1
        assert result[0]["schema_version"] == 2
        assert result[0]["volume"] == "10.0000000000000001"
        assert result[0]["volume_unit"] == "unknown"


@pytest.mark.asyncio
async def test_newer_v1_series_remains_readable(database):
    old = batch("10.25").storage_rows()[0]
    await database.batch_store_market_data([{**old, "timestamp": "2026-07-23T15:00:00+00:00"}])
    await database.batch_store_market_data(batch(20, version=1).storage_rows())
    sync = SyncDatabaseReader(str(database.db_path))
    for result in (
        await database.get_latest_market_data("AAPL"),
        sync.get_latest_market_data("AAPL"),
    ):
        assert len(result) == 1
        assert result[0]["schema_version"] == 1
        assert type(result[0]["volume"]) is int
        assert "volume_unit" not in result[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["volume", "version"])
async def test_corrupt_selected_v2_does_not_fallback(database, corruption):
    await database.batch_store_market_data(batch(10, version=1).storage_rows())
    await database.batch_store_market_data(batch("10.25").storage_rows())
    async with database.get_connection() as conn:
        if corruption == "volume":
            await conn.execute("UPDATE canonical_market_data_v2 SET volume = 'garbage'")
        else:
            await conn.execute("PRAGMA ignore_check_constraints = ON")
            await conn.execute("UPDATE canonical_market_data_v2 SET schema_version = 3")
        await conn.commit()
    with pytest.raises(MarketDataContractError):
        await database.get_latest_market_data("AAPL")
    with pytest.raises(MarketDataReadError):
        SyncDatabaseReader(str(database.db_path)).get_latest_market_data("AAPL")


@pytest.mark.asyncio
async def test_sync_reader_supports_pre_v2_database(database):
    await database.batch_store_market_data(batch(10, version=1).storage_rows())
    async with database.get_connection() as conn:
        await conn.execute("DROP TABLE canonical_market_data_v2")
        await conn.commit()
    result = SyncDatabaseReader(str(database.db_path)).get_latest_market_data("AAPL")
    assert result[0]["schema_version"] == 1
    assert result[0]["volume"] == 10


@pytest.mark.asyncio
async def test_wrong_kind_v2_object_is_not_treated_as_legacy_database(database):
    await database.batch_store_market_data(batch(10, version=1).storage_rows())
    await database.batch_store_market_data(batch("10.25").storage_rows())
    async with database.get_connection() as conn:
        await conn.execute("ALTER TABLE canonical_market_data_v2 RENAME TO saved_v2")
        await conn.execute("CREATE VIEW canonical_market_data_v2 AS SELECT * FROM saved_v2")
        await conn.commit()
    with pytest.raises(MarketDataContractError, match="table"):
        await database.get_latest_market_data("AAPL")
    with pytest.raises(MarketDataReadError) as caught:
        SyncDatabaseReader(str(database.db_path)).get_latest_market_data("AAPL")
    assert isinstance(caught.value.__cause__, MarketDataContractError)
    assert "not a table" in str(caught.value.__cause__)
