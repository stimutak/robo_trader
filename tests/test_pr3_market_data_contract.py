"""PR 3 canonical market-data contract and persistence tests."""

import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from robo_trader.clients.subprocess_ibkr_client import (
    SubprocessIBKRClient,
    _WorkerGeneration,
)
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.market_data_contract import (
    MAX_MARKET_DATA_AGE_SECONDS,
    BarQualityFlag,
    CanonicalBarBatch,
    MarketDataContractError,
    MarketSession,
    MarketSessionPolicy,
    canonicalize_historical_bars,
    market_data_max_age_seconds,
)
from robo_trader.websocket_client import WebSocketClient
from sync_db_reader import MarketDataReadError, SyncDatabaseReader


def _lineage(*, retrieval: datetime | None = None) -> SimpleNamespace:
    timestamp = retrieval or datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc)
    return SimpleNamespace(
        con_id=265598,
        symbol="AAPL",
        exchange="SMART",
        primary_exchange="NASDAQ",
        broker_timestamp=timestamp,
        retrieval_timestamp=timestamp,
        transport_generation="generation-1",
    )


def _record(timestamp: str, *, volume: int = 100) -> dict:
    return {
        "date": timestamp,
        "open": 100,
        "high": 102,
        "low": 99,
        "close": 101,
        "volume": volume,
    }


def test_canonical_batch_binds_bars_to_exact_lineage_and_source_time() -> None:
    batch = canonicalize_historical_bars(
        symbol="AAPL",
        records=[
            _record("2026-07-23T15:00:00+00:00"),
            _record("2026-07-23T15:01:00+00:00", volume=0),
        ],
        lineage=_lineage(),
        bar_size="1 min",
        use_rth=True,
        what_to_show="TRADES",
        now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
    )

    assert type(batch) is CanonicalBarBatch
    assert batch.contract.session_policy is MarketSessionPolicy.REGULAR_ONLY
    assert batch.contract.con_id == 265598
    assert batch.bars[-1].session is MarketSession.REGULAR
    assert batch.bars[-1].quality_flags == (BarQualityFlag.ZERO_VOLUME,)
    frame = batch.to_frame()
    assert frame.attrs["canonical_bar_batch"] is batch
    assert str(frame.index.tz) == "UTC"
    assert frame.index[-1].to_pydatetime() == datetime(2026, 7, 23, 15, 1, tzinfo=timezone.utc)
    storage = batch.storage_rows()[-1]
    assert storage["timestamp"] == "2026-07-23T15:01:00+00:00"
    assert storage["interval_seconds"] == 60
    assert storage["timezone_name"] == "UTC"
    assert storage["session_policy"] == "regular-only"
    assert storage["broker_timestamp"] == "2026-07-23T15:02:00+00:00"
    assert storage["timestamp_semantics"] == "bar-start"
    assert storage["use_rth"] is True
    assert storage["what_to_show"] == "TRADES"


def test_session_policy_rejects_extended_bar_from_regular_response() -> None:
    kwargs = {
        "symbol": "AAPL",
        "records": [_record("2026-07-23T12:00:00+00:00")],
        "lineage": _lineage(retrieval=datetime(2026, 7, 23, 12, 1, tzinfo=timezone.utc)),
        "bar_size": "1 min",
        "what_to_show": "TRADES",
        "now": datetime(2026, 7, 23, 12, 1, tzinfo=timezone.utc),
    }
    with pytest.raises(MarketDataContractError, match="regular-hours"):
        canonicalize_historical_bars(**kwargs, use_rth=True)

    extended = canonicalize_historical_bars(**kwargs, use_rth=False)
    assert extended.bars[0].session is MarketSession.PRE_MARKET


@pytest.mark.parametrize(
    "records",
    [
        [
            _record("2026-07-23T15:00:00+00:00"),
            _record("2026-07-23T15:03:00+00:00"),
        ],
        [_record("2026-07-03T15:00:00+00:00")],
    ],
)
def test_gap_and_market_holiday_bars_fail_closed(records: list[dict]) -> None:
    now = (
        datetime.fromisoformat(records[-1]["date"])
        if len(records) == 1
        else datetime(2026, 7, 23, 15, 4, tzinfo=timezone.utc)
    )
    with pytest.raises(MarketDataContractError):
        canonicalize_historical_bars(
            symbol="AAPL",
            records=records,
            lineage=_lineage(retrieval=now),
            bar_size="1 min",
            use_rth=True,
            what_to_show="TRADES",
            now=now,
        )


@pytest.mark.asyncio
async def test_canonical_refreshes_accumulate_by_contract_timeframe_and_time(
    tmp_path: Path,
) -> None:
    database = AsyncTradingDatabase(tmp_path / "market-data.db", pool_size=1)
    await database.initialize()
    try:
        one_minute = canonicalize_historical_bars(
            symbol="AAPL",
            records=[_record("2026-07-23T15:00:00+00:00")],
            lineage=_lineage(),
            bar_size="1 min",
            use_rth=True,
            what_to_show="TRADES",
            now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
        )
        five_minute = canonicalize_historical_bars(
            symbol="AAPL",
            records=[_record("2026-07-23T15:00:00+00:00")],
            lineage=_lineage(),
            bar_size="5 mins",
            use_rth=True,
            what_to_show="TRADES",
            now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
        )

        await database.batch_store_market_data(one_minute.storage_rows())
        await database.batch_store_market_data(one_minute.storage_rows())
        await database.batch_store_market_data(five_minute.storage_rows())

        async with database.get_connection() as connection:
            count = await (
                await connection.execute("SELECT COUNT(*) FROM canonical_market_data")
            ).fetchone()
        assert count == (2,)
        latest = await database.get_latest_market_data("AAPL", limit=10)
        assert {row["timeframe"] for row in latest} == {"1 min"}
        assert all(row["source"] == "ibkr-historical-trades" for row in latest)
        assert all(row["con_id"] == 265598 for row in latest)
        assert all(row["transport_generation"] == "generation-1" for row in latest)
        assert all(row["freshness_status"] == "stale" for row in latest)
        sync_reader = SyncDatabaseReader(str(tmp_path / "market-data.db"))
        sync_latest = sync_reader.get_latest_market_data("AAPL", limit=10)
        assert {row["timeframe"] for row in sync_latest} == {"1 min"}
        assert all(row["source"] == "ibkr-historical-trades" for row in sync_latest)
        assert all(row["freshness_status"] == "stale" for row in sync_latest)
        explicit_five = await database.get_latest_market_data("AAPL", limit=10, timeframe="5 mins")
        assert {row["timeframe"] for row in explicit_five} == {"5 mins"}
        assert {
            row["timeframe"]
            for row in sync_reader.get_latest_market_data("AAPL", limit=10, timeframe="5 mins")
        } == {"5 mins"}
    finally:
        await database.close()


def test_canonical_values_remain_exact_before_dataframe_projection() -> None:
    record = _record("2026-07-23T15:00:00+00:00")
    record["close"] = "101.1250"
    batch = canonicalize_historical_bars(
        symbol="AAPL",
        records=[record],
        lineage=_lineage(),
        bar_size="1 min",
        use_rth=True,
        what_to_show="TRADES",
        now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
    )
    assert batch.bars[0].close == Decimal("101.1250")


def test_canonical_contract_rejects_non_trade_history() -> None:
    with pytest.raises(MarketDataContractError, match="must be TRADES"):
        canonicalize_historical_bars(
            symbol="AAPL",
            records=[_record("2026-07-23T15:00:00+00:00")],
            lineage=_lineage(),
            bar_size="1 min",
            use_rth=True,
            what_to_show="MIDPOINT",
            now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("configured", ["nan", "inf", "3600", "86401"])
def test_market_data_freshness_configuration_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
) -> None:
    monkeypatch.setenv("MARKET_DATA_MAX_AGE_SECONDS", configured)
    with pytest.raises(MarketDataContractError, match="finite, positive"):
        canonicalize_historical_bars(
            symbol="AAPL",
            records=[_record("2026-07-23T15:00:00+00:00")],
            lineage=_lineage(),
            bar_size="1 min",
            use_rth=True,
            what_to_show="TRADES",
            now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
        )
    assert MAX_MARKET_DATA_AGE_SECONDS == 86_400


def test_market_data_freshness_accepts_reasonable_timeframe_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKET_DATA_MAX_AGE_SECONDS", "200")
    assert market_data_max_age_seconds(60) == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "operator-spoofed"),
        ("timestamp", "2026-07-23T15:00:00"),
        ("high", 98),
        ("quality_flags", "unrecognized"),
        ("what_to_show", "MIDPOINT"),
    ],
)
async def test_canonical_database_admission_rejects_malformed_rows(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    database = AsyncTradingDatabase(tmp_path / f"invalid-{field}.db", pool_size=1)
    await database.initialize()
    batch = canonicalize_historical_bars(
        symbol="AAPL",
        records=[_record("2026-07-23T15:00:00+00:00")],
        lineage=_lineage(),
        bar_size="1 min",
        use_rth=True,
        what_to_show="TRADES",
        now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
    )
    row = batch.storage_rows()[0]
    row[field] = value
    try:
        with pytest.raises(ValueError, match="is not canonical"):
            await database.batch_store_market_data([row])
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_cleanup_preserves_canonical_audit_and_scoped_call_is_not_global(
    tmp_path: Path,
) -> None:
    database = AsyncTradingDatabase(tmp_path / "cleanup.db", pool_size=1)
    await database.initialize()
    batch = canonicalize_historical_bars(
        symbol="AAPL",
        records=[_record("2026-07-23T15:00:00+00:00")],
        lineage=_lineage(),
        bar_size="1 min",
        use_rth=True,
        what_to_show="TRADES",
        now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
    )
    try:
        await database.batch_store_market_data(batch.storage_rows())
        await database.store_market_data(
            "AAPL",
            datetime(2020, 1, 2, tzinfo=timezone.utc),
            100,
            101,
            99,
            100,
            1,
        )
        await database.cleanup_old_data(days_to_keep=0, portfolio_id="default")
        async with database.get_connection() as connection:
            canonical_count = (
                await (
                    await connection.execute("SELECT COUNT(*) FROM canonical_market_data")
                ).fetchone()
            )[0]
            legacy_count = (
                await (await connection.execute("SELECT COUNT(*) FROM market_data")).fetchone()
            )[0]
        assert canonical_count == 1
        assert legacy_count == 1

        await database.cleanup_old_data(days_to_keep=0)
        async with database.get_connection() as connection:
            canonical_count = (
                await (
                    await connection.execute("SELECT COUNT(*) FROM canonical_market_data")
                ).fetchone()
            )[0]
            legacy_count = (
                await (await connection.execute("SELECT COUNT(*) FROM market_data")).fetchone()
            )[0]
        assert canonical_count == 1
        assert legacy_count == 0
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_tampered_canonical_row_fails_async_read_and_api_returns_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "tampered.db"
    database = AsyncTradingDatabase(database_path, pool_size=1)
    await database.initialize()
    batch = canonicalize_historical_bars(
        symbol="AAPL",
        records=[_record("2026-07-23T15:00:00+00:00")],
        lineage=_lineage(),
        bar_size="1 min",
        use_rth=True,
        what_to_show="TRADES",
        now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
    )
    try:
        await database.batch_store_market_data(batch.storage_rows())
        async with database.get_connection() as connection:
            await connection.execute(
                "UPDATE canonical_market_data " "SET broker_timestamp = '2020-01-02T15:00:00+00:00'"
            )
            await connection.commit()
        with pytest.raises(MarketDataContractError, match="clock-skew"):
            await database.get_latest_market_data("AAPL", limit=1)
    finally:
        await database.close()

    monkeypatch.setenv("RT_DB_PATH", str(database_path))
    with patch.dict(os.environ, {"DASH_AUTH_ENABLED": "false"}, clear=False):
        import app as app_module

    response = app_module.app.test_client().get("/api/market-data/AAPL")
    assert response.status_code == 503
    assert response.get_json() == {"error": "market_data_unavailable"}


def test_sync_reader_raises_typed_error_for_unavailable_database(tmp_path: Path) -> None:
    reader = SyncDatabaseReader(str(tmp_path / "missing" / "market.db"))
    with pytest.raises(MarketDataReadError, match="unavailable"):
        reader.get_latest_market_data("AAPL", limit=1)


@pytest.mark.parametrize(
    ("reader_result", "expected_status"),
    [([], 404), (MarketDataReadError("database unavailable"), 503)],
)
def test_market_data_api_distinguishes_empty_from_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    reader_result: object,
    expected_status: int,
) -> None:
    with patch.dict(os.environ, {"DASH_AUTH_ENABLED": "false"}, clear=False):
        import app as app_module

    if isinstance(reader_result, BaseException):
        replacement = patch.object(
            SyncDatabaseReader,
            "get_latest_market_data",
            side_effect=reader_result,
        )
    else:
        replacement = patch.object(
            SyncDatabaseReader,
            "get_latest_market_data",
            return_value=reader_result,
        )
    with replacement:
        response = app_module.app.test_client().get("/api/market-data/AAPL")
    assert response.status_code == expected_status
    if expected_status == 503:
        assert response.get_json() == {"error": "market_data_unavailable"}


@pytest.mark.asyncio
async def test_client_returns_bars_and_lineage_as_one_canonical_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "bars": [_record("2026-07-23T15:00:00+00:00")],
        "requested_symbol": "AAPL",
        "qualified_contract": {
            "con_id": 265598,
            "symbol": "AAPL",
            "local_symbol": "AAPL",
            "security_type": "STK",
            "exchange": "SMART",
            "primary_exchange": "NASDAQ",
            "currency": "USD",
            "trading_class": "NMS",
        },
        "broker_timestamp": now.isoformat(),
        "retrieval_timestamp": now.isoformat(),
    }
    client = SubprocessIBKRClient()
    generation = _WorkerGeneration(
        generation_id="generation-1",
        process=SimpleNamespace(poll=lambda: 0),
    )
    client._generation = generation
    client._connected = True
    client._connection_generation_id = generation.generation_id
    client._connection_identity = ("127.0.0.1", 4002, 7, True)
    execute = AsyncMock(return_value=payload)
    monkeypatch.setattr(client, "_execute_command_unlocked", execute)
    from robo_trader import market_data_contract

    canonicalize = market_data_contract.canonicalize_historical_bars

    def canonicalize_at_fixture_time(**kwargs):
        return canonicalize(
            **kwargs,
            now=datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc),
        )

    monkeypatch.setattr(
        "robo_trader.clients.subprocess_ibkr_client.canonicalize_historical_bars",
        canonicalize_at_fixture_time,
    )

    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")

    assert type(batch) is CanonicalBarBatch
    assert batch.contract.transport_generation == generation.generation_id
    assert batch.contract.con_id == 265598
    assert batch.bars[0].timestamp == datetime(2026, 7, 23, 15, tzinfo=timezone.utc)


def test_websocket_market_update_preserves_event_and_freshness_lineage() -> None:
    client = WebSocketClient(max_queue_size=2)

    client.send_market_update(
        "AAPL",
        101.25,
        event_timestamp="2026-07-23T15:00:00+00:00",
        retrieval_timestamp="2026-07-23T15:00:02+00:00",
        source="ibkr-historical-trades",
        session="regular",
        timeframe="1 min",
        freshness_status="fresh",
    )

    message = client.message_queue.get_nowait()
    assert message["event_timestamp"] == "2026-07-23T15:00:00+00:00"
    assert message["retrieval_timestamp"] == "2026-07-23T15:00:02+00:00"
    assert message["source"] == "ibkr-historical-trades"
    assert message["freshness_status"] == "fresh"
