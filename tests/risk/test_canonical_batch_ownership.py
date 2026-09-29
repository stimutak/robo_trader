"""Only the producing client's latest unmodified canonical batch is current."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient, _WorkerGeneration
from robo_trader.market_data_contract import canonicalize_historical_bars
from tests.test_pr3_market_data_contract import _record


@pytest.fixture
def client(monkeypatch):
    now = datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc)
    payload = {
        "bars": [_record("2026-07-23T15:00:00+00:00")],
        "bar_schema_version": 2,
        "requested_symbol": "AAPL",
        "qualified_contract": dict(
            con_id=265598,
            symbol="AAPL",
            local_symbol="AAPL",
            security_type="STK",
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
            trading_class="NMS",
        ),
        "broker_timestamp": now.isoformat(),
        "retrieval_timestamp": now.isoformat(),
    }
    value = SubprocessIBKRClient()
    value._generation = _WorkerGeneration(
        generation_id="generation-1", process=SimpleNamespace(poll=lambda: 0)
    )
    value._connected = True
    value._connection_generation_id = "generation-1"
    value._connection_identity = ("127.0.0.1", 4002, 7, True)
    monkeypatch.setattr(value, "_execute_command_unlocked", AsyncMock(return_value=payload))
    monkeypatch.setattr(
        "robo_trader.clients.subprocess_ibkr_client.canonicalize_historical_bars",
        lambda **kwargs: canonicalize_historical_bars(**kwargs, now=now),
    )
    return value


@pytest.mark.asyncio
async def test_owned_batch_rejects_copy_and_other_producer(client):
    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    assert batch.contract.schema_version == 2
    assert type(batch.bars[0].volume) is Decimal
    assert client.assert_current_canonical_batch(batch) is batch
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(replace(batch))
    with pytest.raises(ValueError):
        SubprocessIBKRClient().assert_current_canonical_batch(batch)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["price", "timestamp", "contract", "bars"])
async def test_owned_batch_rejects_structurally_valid_mutation(client, mutation):
    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    if mutation == "price":
        object.__setattr__(batch.bars[0], "close", Decimal("100"))
    elif mutation == "timestamp":
        object.__setattr__(
            batch.bars[0], "timestamp", datetime(2026, 7, 23, 14, 59, tzinfo=timezone.utc)
        )
    elif mutation == "contract":
        object.__setattr__(batch.contract, "primary_exchange", "NYSE")
    else:
        object.__setattr__(batch, "bars", list(batch.bars))
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(batch)


@pytest.mark.asyncio
async def test_refresh_and_disconnect_invalidate_old_batches(client):
    old = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    new = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(old)
    assert client.assert_current_canonical_batch(new) is new
    client._invalidate_historical_lineage()
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(new)


@pytest.mark.asyncio
async def test_generation_change_and_poison_reject_owned_data(client):
    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    client._generation.poisoned_reason = "test disconnection"
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(batch)


@pytest.mark.asyncio
async def test_runner_execution_lookup_requires_current_producer(client):
    from robo_trader.runner_async import AsyncRunner

    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    runner = object.__new__(AsyncRunner)
    runner.ib = client
    runner._canonical_bar_batches = {"AAPL": (batch, batch.to_frame())}
    assert runner._canonical_batch_for_execution("AAPL") is batch
    await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    assert runner._canonical_batch_for_execution("AAPL") is None
    runner.ib = SimpleNamespace(assert_current_canonical_batch=lambda value: value)
    assert runner._canonical_batch_for_execution("AAPL") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["contract", "timestamp"])
async def test_malformed_owned_objects_fail_with_validation_error(client, field):
    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    object.__setattr__(batch if field == "contract" else batch.bars[0], field, None)
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(batch)


@pytest.mark.asyncio
async def test_raw_refresh_and_replaced_worker_invalidate(client):
    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    await client.get_historical_bars("AAPL", bar_size="1 min")
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(batch)
    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    client._generation = _WorkerGeneration(
        generation_id="generation-1", process=SimpleNamespace(poll=lambda: 0)
    )
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(batch)


@pytest.mark.asyncio
async def test_runner_rejects_malformed_cached_contract(client):
    from robo_trader.runner_async import AsyncRunner

    batch = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    runner = object.__new__(AsyncRunner)
    runner.ib = client
    runner._canonical_bar_batches = {"AAPL": (batch, batch.to_frame())}
    object.__setattr__(batch, "contract", None)
    assert runner._canonical_batch_for_execution("AAPL") is None


@pytest.mark.asyncio
async def test_timeframe_and_session_batches_coexist_until_matching_refresh(client):
    minute = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    five = await client.get_canonical_historical_bars("AAPL", bar_size="5 mins")
    extended = await client.get_canonical_historical_bars("AAPL", bar_size="1 min", use_rth=False)
    for batch in (minute, five, extended):
        assert client.assert_current_canonical_batch(batch) is batch
    await client.get_historical_bars("AAPL", bar_size="1 min", use_rth=True)
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(minute)
    assert client.assert_current_canonical_batch(five) is five
    assert client.assert_current_canonical_batch(extended) is extended


@pytest.mark.asyncio
async def test_other_window_retrieval_time_does_not_relabel_owned_history(client):
    minute = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    from copy import deepcopy

    payload = deepcopy(client._execute_command_unlocked.return_value)
    payload["retrieval_timestamp"] = "2026-07-23T15:02:01+00:00"
    payload["broker_timestamp"] = payload["retrieval_timestamp"]
    client._execute_command_unlocked.return_value = payload
    await client.get_canonical_historical_bars("AAPL", bar_size="5 mins")
    assert client.assert_current_canonical_batch(minute) is minute
    assert minute.contract.retrieval_time == datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("con_id", 999), ("trading_class", "NEW")])
async def test_contract_identity_change_invalidates_every_window(client, field, value):
    minute = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    five = await client.get_canonical_historical_bars("AAPL", bar_size="5 mins")
    from copy import deepcopy

    payload = deepcopy(client._execute_command_unlocked.return_value)
    payload["qualified_contract"][field] = value
    client._execute_command_unlocked.return_value = payload
    await client.get_canonical_historical_bars("AAPL", bar_size="15 mins")
    for batch in (minute, five):
        with pytest.raises(ValueError):
            client.assert_current_canonical_batch(batch)


@pytest.mark.asyncio
async def test_failed_canonical_refresh_invalidates_only_matching_window(client, monkeypatch):
    minute = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    five = await client.get_canonical_historical_bars("AAPL", bar_size="5 mins")

    def invalid_bars(**kwargs):
        raise ValueError("test incomplete canonical response")

    monkeypatch.setattr(
        "robo_trader.clients.subprocess_ibkr_client.canonicalize_historical_bars", invalid_bars
    )
    with pytest.raises(ValueError, match="incomplete"):
        await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(minute)
    assert client.assert_current_canonical_batch(five) is five


@pytest.mark.asyncio
async def test_contract_identity_round_trip_does_not_restore_old_window(client):
    from copy import deepcopy

    minute = await client.get_canonical_historical_bars("AAPL", bar_size="1 min")
    original = deepcopy(client._execute_command_unlocked.return_value)
    changed = deepcopy(original)
    changed["qualified_contract"]["trading_class"] = "NEW"
    client._execute_command_unlocked.return_value = changed
    await client.get_canonical_historical_bars("AAPL", bar_size="5 mins")
    client._execute_command_unlocked.return_value = original
    await client.get_canonical_historical_bars("AAPL", bar_size="5 mins")
    with pytest.raises(ValueError):
        client.assert_current_canonical_batch(minute)
