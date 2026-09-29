from decimal import Decimal, localcontext

import pytest

from robo_trader.clients.exact_historical_decoder import ExactHistoricalDecoder, ExactHistoricalIB
from robo_trader.clients.ibkr_subprocess_worker import _historical_volume


def message(volume="10", *, req_id="123"):
    return [
        "17",
        req_id,
        "start",
        "end",
        "1",
        "1784817000",
        "100",
        "101",
        "99",
        "100",
        volume,
        "100",
        "3",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["1000000000000000001", "10.0000000000000001", "10.25", "0"])
async def test_wire_volume_is_exact_through_real_wrapper(raw):
    ib = ExactHistoricalIB()
    assert type(ib.client.decoder) is ExactHistoricalDecoder
    pending = ib.wrapper.startReq(123)
    with localcontext() as context:
        context.prec = 6
        ib.client.decoder.interpret(message(raw))
        rows = await pending
        assert type(rows[0].volume) is Decimal
        assert rows[0].volume == Decimal(raw)
        assert Decimal(_historical_volume(rows[0].volume)) == Decimal(raw)
    assert 123 not in ib.wrapper._results


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-1", "garbage"])
async def test_invalid_volume_fails_only_matching_request(raw):
    ib = ExactHistoricalIB()
    pending = ib.wrapper.startReq(123)
    other = ib.wrapper.startReq(124)
    ib.client.decoder.interpret(message(raw))
    with pytest.raises(ValueError, match="Invalid historical response"):
        await pending
    assert not other.done()
    assert 123 not in ib.wrapper._results
    assert 123 not in ib.wrapper._futures
    other.cancel()


@pytest.mark.asyncio
async def test_malformed_second_bar_never_publishes_first():
    ib = ExactHistoricalIB()
    rows = []
    pending = ib.wrapper.startReq(123, container=rows)
    fields = message()
    fields[4] = "2"
    fields += message("bad")[5:]
    ib.client.decoder.interpret(fields)
    with pytest.raises(ValueError):
        await pending
    assert rows == []


@pytest.mark.asyncio
async def test_truncated_message_fails_instead_of_completing_empty():
    ib = ExactHistoricalIB()
    pending = ib.wrapper.startReq(123)
    ib.client.decoder.interpret(message()[:-1])
    with pytest.raises(ValueError, match="bar count"):
        await pending


@pytest.mark.asyncio
async def test_malformed_second_date_fails_before_publication():
    ib = ExactHistoricalIB()
    rows = []
    pending = ib.wrapper.startReq(123, container=rows)
    fields = message()
    fields[4] = "2"
    second = message()[5:]
    second[0] = "bad-date"
    fields += second
    ib.client.decoder.interpret(fields)
    assert pending.done()
    with pytest.raises(ValueError):
        await pending
    assert rows == []


def test_worker_uses_exact_decoder_client():
    from robo_trader.clients import ibkr_subprocess_worker as worker

    assert worker.IB is ExactHistoricalIB


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["1000000000000000001", "10.0000000000000001"])
async def test_wire_volume_through_worker_and_canonical_contract(monkeypatch, raw):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from robo_trader.clients import ibkr_subprocess_worker as worker
    from robo_trader.market_data_contract import canonicalize_historical_bars
    from tests.test_pr3_market_data_contract import _lineage

    ib = ExactHistoricalIB()
    now = datetime(2026, 7, 23, 15, 2, tzinfo=timezone.utc)
    contract = SimpleNamespace(
        symbol="AAPL",
        localSymbol="AAPL",
        conId=265598,
        secType="STK",
        exchange="SMART",
        primaryExchange="NASDAQ",
        currency="USD",
        tradingClass="NMS",
    )

    async def historical(*args, **kwargs):
        pending = ib.wrapper.startReq(123)
        fields = message(raw)
        fields[5] = str(int(now.timestamp()) - 60)
        ib.client.decoder.interpret(fields)
        return await pending

    monkeypatch.setattr(ib, "isConnected", lambda: True)
    monkeypatch.setattr(ib, "reqHistoricalDataAsync", historical)
    monkeypatch.setattr(worker, "ib", ib)
    monkeypatch.setattr(worker, "_qualify_one_contract", AsyncMock(return_value=contract))
    monkeypatch.setattr(worker, "_request_broker_time", AsyncMock(return_value=now))
    response = await worker.handle_get_historical_bars({"symbol": "AAPL"})
    assert response["status"] == "success"
    batch = canonicalize_historical_bars(
        symbol="AAPL",
        records=response["data"]["bars"],
        lineage=_lineage(),
        schema_version=response["data"]["bar_schema_version"],
        bar_size="1 min",
        use_rth=True,
        what_to_show="TRADES",
        now=now,
    )
    assert batch.contract.schema_version == 2
    assert batch.bars[0].volume == Decimal(raw)
