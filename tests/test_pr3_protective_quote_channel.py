"""PR 3 tests for the independent live protective quote channel."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.clients import ibkr_subprocess_worker as worker
from robo_trader.clients import subprocess_ibkr_client as client_module
from robo_trader.clients.subprocess_ibkr_client import (
    IBKRTransportPoisonedError,
    SubprocessIBKRClient,
    _WorkerGeneration,
)
from robo_trader.market_data_contract import (
    BrokerProtectiveQuote,
    MarketDataIdentityError,
    MarketDataSource,
    MarketSession,
)
from robo_trader.paper_reduction_gateway import (
    PaperReductionGateway,
    PaperReductionGatewayError,
)
from robo_trader.protective_quote_evidence import (
    MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH,
    ProtectiveQuoteSource,
)
from robo_trader.risk_manager import Position
from robo_trader.runner_async import AsyncRunner
from robo_trader.stop_loss_monitor import StopLossMonitor, StopStatus


@pytest.fixture(autouse=True)
def _isolate_worker_protective_subscription_state():
    """Keep the worker's process-global subscription cache test-local."""

    worker._clear_protective_tick_subscriptions()
    yield
    worker._clear_protective_tick_subscriptions()


def _contract(symbol: str = "AAPL", con_id: int = 265598) -> SimpleNamespace:
    return SimpleNamespace(
        conId=con_id,
        symbol=symbol,
        localSymbol=symbol,
        secType="STK",
        currency="USD",
        exchange="SMART",
        primaryExchange="NASDAQ",
        tradingClass="NMS",
    )


@pytest.mark.asyncio
async def test_worker_emits_only_qualified_live_last_trade_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    contract = _contract()

    class FakeIB:
        def isConnected(self) -> bool:
            return True

        async def qualifyContractsAsync(self, requested):
            assert requested.symbol == "AAPL"
            return [contract]

        async def reqCurrentTimeAsync(self):
            return now

        def reqTickByTickData(
            self,
            qualified,
            tick_type,
            *,
            numberOfTicks,
            ignoreSize,
        ):
            assert qualified is contract
            assert tick_type == "Last"
            assert numberOfTicks == 0
            assert ignoreSize is False
            return SimpleNamespace(
                contract=contract,
                time=now,
                last=999.0,
                tickByTicks=[SimpleNamespace(time=now, price=187.25)],
            )

        def cancelTickByTickData(self, qualified, tick_type):
            assert qualified is contract
            assert tick_type == "Last"
            return True

    monkeypatch.setattr(worker, "ib", FakeIB())
    monkeypatch.setattr(worker, "WORKER_GENERATION_ID", "generation-1")
    monkeypatch.setattr(worker, "get_market_session", lambda _timestamp: "regular")

    response = await worker.handle_get_protective_quotes({"symbols": ["AAPL"]})

    assert response["status"] == "success"
    quote = response["data"]["quotes"][0]
    assert quote["price"] == "187.25"
    assert quote["con_id"] == 265598
    assert quote["source"] == "ibkr-live-last-trade"
    assert quote["market_data_type"] == 1
    assert quote["source_timestamp"] == now.isoformat()


@pytest.mark.asyncio
async def test_worker_never_pairs_generic_ticker_time_with_stale_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    contract = _contract()

    class FakeIB:
        def isConnected(self) -> bool:
            return True

        async def qualifyContractsAsync(self, _requested):
            return [contract]

        async def reqCurrentTimeAsync(self):
            return now

        def reqTickByTickData(self, _contract, _tick_type, **_kwargs):
            return SimpleNamespace(
                contract=contract,
                time=now,
                last=100.00,
                tickByTicks=[
                    SimpleNamespace(
                        time=now,
                        price=Decimal("95.1250"),
                    )
                ],
            )

        def cancelTickByTickData(self, _contract, _tick_type):
            return True

    monkeypatch.setattr(worker, "ib", FakeIB())
    monkeypatch.setattr(worker, "WORKER_GENERATION_ID", "generation-1")
    monkeypatch.setattr(worker, "get_market_session", lambda _timestamp: "regular")

    response = await worker.handle_get_protective_quotes({"symbols": ["AAPL"]})

    quote = response["data"]["quotes"][0]
    assert quote["price"] == "95.125"
    assert quote["price"] != "100"


@pytest.mark.asyncio
async def test_worker_reuses_subscription_and_replays_every_new_event_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    event_time = now.replace(microsecond=0)
    contract = _contract()
    first_tick = SimpleNamespace(
        time=event_time,
        price=Decimal("100"),
        size=1,
        exchange="NASDAQ",
    )
    ticker = SimpleNamespace(contract=contract, tickByTicks=[first_tick])

    class FakeIB:
        def __init__(self):
            self.qualify_count = 0
            self.request_count = 0

        def isConnected(self) -> bool:
            return True

        async def qualifyContractsAsync(self, _requested):
            self.qualify_count += 1
            return [contract]

        async def reqCurrentTimeAsync(self):
            return now

        def reqTickByTickData(self, _contract, _tick_type, **_kwargs):
            self.request_count += 1
            return ticker

        def cancelTickByTickData(self, _contract, _tick_type):
            return None

    fake_ib = FakeIB()
    monkeypatch.setattr(worker, "ib", fake_ib)
    monkeypatch.setattr(worker, "WORKER_GENERATION_ID", "generation-1")
    monkeypatch.setattr(worker, "get_market_session", lambda _timestamp: "regular")

    first = await worker.handle_get_protective_quotes(
        {"symbols": ["AAPL"], "active_symbols": ["AAPL"]}
    )
    first_id = first["data"]["quotes"][0]["source_event_id"]

    # Distinct broker events can have identical trade fields. Arrival identity,
    # not a value-only fingerprint, must keep both observable and unique.
    ticker.tickByTicks.extend(
        [
            SimpleNamespace(
                time=event_time,
                price=Decimal("95"),
                size=2,
                exchange="NASDAQ",
            ),
            SimpleNamespace(
                time=event_time,
                price=Decimal("95"),
                size=2,
                exchange="NASDAQ",
            ),
        ]
    )
    replay = await worker.handle_get_protective_quotes(
        {"symbols": ["AAPL"], "active_symbols": ["AAPL"]}
    )
    replay_ids = [quote["source_event_id"] for quote in replay["data"]["quotes"]]

    assert [quote["price"] for quote in replay["data"]["quotes"]] == ["95", "95"]
    assert len(set(replay_ids)) == 2
    assert first_id not in replay_ids
    assert fake_ib.qualify_count == 1
    assert fake_ib.request_count == 1
    assert len(ticker.tickByTicks) == 1

    fallback = await worker.handle_get_protective_quotes(
        {"symbols": ["AAPL"], "active_symbols": ["AAPL"]}
    )
    assert fallback["data"]["quotes"][0]["source_event_id"] == replay_ids[-1]
    assert fake_ib.qualify_count == 1
    assert fake_ib.request_count == 1


@pytest.mark.asyncio
async def test_worker_uses_account_active_set_across_fetch_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    contracts = {
        "AAPL": _contract("AAPL", 1),
        "MSFT": _contract("MSFT", 2),
    }

    class FakeIB:
        def __init__(self):
            self.requested = []
            self.cancelled = []

        def isConnected(self) -> bool:
            return True

        async def qualifyContractsAsync(self, requested):
            return [contracts[requested.symbol]]

        async def reqCurrentTimeAsync(self):
            return now

        def reqTickByTickData(self, contract, _tick_type, **_kwargs):
            self.requested.append(contract.symbol)
            return SimpleNamespace(
                contract=contract,
                tickByTicks=[SimpleNamespace(time=now, price=100)],
            )

        def cancelTickByTickData(self, contract, _tick_type):
            self.cancelled.append(contract.symbol)
            return None

    fake_ib = FakeIB()
    clock = {"now": 0.0}
    monkeypatch.setattr(worker, "ib", fake_ib)
    monkeypatch.setattr(worker, "get_market_session", lambda _timestamp: "regular")
    monkeypatch.setattr(worker, "_protective_request_monotonic", lambda: clock["now"])

    active = ["AAPL", "MSFT"]
    assert (
        await worker.handle_get_protective_quotes({"symbols": ["AAPL"], "active_symbols": active})
    )["status"] == "success"
    assert (
        await worker.handle_get_protective_quotes({"symbols": ["MSFT"], "active_symbols": active})
    )["status"] == "success"

    assert fake_ib.requested == ["AAPL", "MSFT"]
    assert fake_ib.cancelled == []

    clock["now"] = 16.0
    assert (
        await worker.handle_get_protective_quotes({"symbols": ["AAPL"], "active_symbols": ["AAPL"]})
    )["status"] == "success"
    assert fake_ib.requested == ["AAPL", "MSFT"]
    assert fake_ib.cancelled == ["MSFT"]
    assert set(worker._PROTECTIVE_SYMBOL_CON_IDS) == {"AAPL"}


@pytest.mark.asyncio
async def test_worker_paces_same_instrument_across_rapid_active_set_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    contract = _contract()
    ticker = SimpleNamespace(
        contract=contract,
        tickByTicks=[SimpleNamespace(time=now, price=100)],
    )
    clock = {"now": 0.0}

    class FakeIB:
        def __init__(self):
            self.qualify_count = 0
            self.request_count = 0
            self.cancel_count = 0

        def isConnected(self) -> bool:
            return True

        async def qualifyContractsAsync(self, _requested):
            self.qualify_count += 1
            return [contract]

        async def reqCurrentTimeAsync(self):
            return now

        def reqTickByTickData(
            self,
            _contract,
            _tick_type,
            *,
            numberOfTicks,
            ignoreSize,
        ):
            assert numberOfTicks == 0
            assert ignoreSize is False
            self.request_count += 1
            return ticker

        def cancelTickByTickData(self, _contract, _tick_type):
            self.cancel_count += 1
            return None

    fake_ib = FakeIB()
    monkeypatch.setattr(worker, "ib", fake_ib)
    monkeypatch.setattr(worker, "get_market_session", lambda _timestamp: "regular")
    monkeypatch.setattr(worker, "_protective_request_monotonic", lambda: clock["now"])

    started = await worker.handle_get_protective_quotes(
        {"symbols": ["AAPL"], "active_symbols": ["AAPL"]}
    )
    assert started["status"] == "success"
    assert worker._PROTECTIVE_TICK_REQUEST_TIMES == {contract.conId: 0.0}

    clock["now"] = 5.0
    removed = await worker.handle_get_protective_quotes({"symbols": [], "active_symbols": []})
    assert removed["status"] == "success"
    assert fake_ib.cancel_count == 0

    clock["now"] = 6.0
    readded = await worker.handle_get_protective_quotes(
        {"symbols": ["AAPL"], "active_symbols": ["AAPL"]}
    )
    assert readded["status"] == "success"
    assert fake_ib.qualify_count == 1
    assert fake_ib.request_count == 1

    clock["now"] = 15.0
    retired = await worker.handle_get_protective_quotes({"symbols": [], "active_symbols": []})
    assert retired["status"] == "success"
    assert fake_ib.cancel_count == 1
    assert worker._PROTECTIVE_TICK_SUBSCRIPTIONS == {}
    assert worker._PROTECTIVE_TICK_REQUEST_TIMES == {}
    assert worker._PROTECTIVE_SYMBOL_CON_IDS == {}


@pytest.mark.asyncio
async def test_worker_fails_closed_and_retains_state_when_cancellation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    contract = _contract()

    class FakeIB:
        def isConnected(self) -> bool:
            return True

        async def qualifyContractsAsync(self, _requested):
            return [contract]

        async def reqCurrentTimeAsync(self):
            return now

        def reqTickByTickData(self, _contract, _tick_type, **_kwargs):
            return SimpleNamespace(
                contract=contract,
                tickByTicks=[SimpleNamespace(time=now, price=100)],
            )

        def cancelTickByTickData(self, _contract, _tick_type):
            raise RuntimeError("broker cancellation failed")

    clock = {"now": 0.0}
    monkeypatch.setattr(worker, "ib", FakeIB())
    monkeypatch.setattr(worker, "get_market_session", lambda _timestamp: "regular")
    monkeypatch.setattr(worker, "_protective_request_monotonic", lambda: clock["now"])
    initial = await worker.handle_get_protective_quotes(
        {"symbols": ["AAPL"], "active_symbols": ["AAPL"]}
    )
    assert initial["status"] == "success"

    clock["now"] = 15.0
    cancelled = await worker.handle_get_protective_quotes({"symbols": [], "active_symbols": []})

    assert cancelled["status"] == "error"
    assert "broker cancellation failed" in cancelled["error"]
    assert set(worker._PROTECTIVE_SYMBOL_CON_IDS) == {"AAPL"}
    assert worker._PROTECTIVE_TICK_REQUEST_TIMES == {contract.conId: 0.0}


def _client_response(*, market_data_type: int = 1) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    timestamp = now.isoformat()
    return {
        "quotes": [
            {
                "schema_version": 1,
                "symbol": "AAPL",
                "con_id": 265598,
                "exchange": "SMART",
                "primary_exchange": "NASDAQ",
                "currency": "USD",
                "security_type": "STK",
                "price": "187.25",
                "source_timestamp": timestamp,
                "retrieval_timestamp": timestamp,
                "session": "regular",
                "source": "ibkr-live-last-trade",
                "source_event_id": "protective:generation-1:1",
                "market_data_type": market_data_type,
            }
        ],
        "broker_time_before": timestamp,
        "broker_time_after": timestamp,
        "retrieval_timestamp": timestamp,
    }


def _broker_quote(
    *,
    symbol: str = "AAPL",
    con_id: int = 265598,
    price: Decimal = Decimal("187.2500"),
    generation: str = "generation-1",
    source_timestamp: datetime | None = None,
    source_event_id: str | None = None,
    session: MarketSession = MarketSession.REGULAR,
) -> BrokerProtectiveQuote:
    now = datetime.now(timezone.utc)
    event_time = source_timestamp or now
    return BrokerProtectiveQuote(
        schema_version=1,
        symbol=symbol,
        con_id=con_id,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
        price=price,
        source_timestamp=event_time,
        retrieval_timestamp=now,
        session=session,
        source=MarketDataSource.IBKR_LIVE_LAST_TRADE,
        source_event_id=source_event_id
        or (
            f"protective:{generation}:1"
            if symbol == "AAPL"
            else f"protective:{generation}:{symbol}:1"
        ),
        transport_generation=generation,
        market_data_type=1,
    )


def _connected_client(
    monkeypatch: pytest.MonkeyPatch,
    response: dict,
    *,
    generation_id: str = "generation-1",
):
    monkeypatch.setattr(client_module, "get_market_session", lambda _timestamp: "regular")
    client = SubprocessIBKRClient()
    generation = _WorkerGeneration(
        generation_id=generation_id,
        process=SimpleNamespace(poll=lambda: 0),
    )
    client._generation = generation
    client._connected = True
    client._connection_generation_id = generation.generation_id
    client._connection_identity = ("127.0.0.1", 4002, 7, True)
    execute = AsyncMock(return_value=response)
    monkeypatch.setattr(client, "_execute_command_unlocked", execute)
    return client, generation


@pytest.mark.asyncio
async def test_fixed_event_id_accepts_worst_case_generation_and_sequence_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    contract = _contract()
    # The monitor's producer-owned evidence contract admits generation IDs up
    # to 128 characters; exercise that exact downstream boundary.
    generation_id = "g" * 128

    class FakeIB:
        def isConnected(self) -> bool:
            return True

        async def qualifyContractsAsync(self, _requested):
            return [contract]

        async def reqCurrentTimeAsync(self):
            return now

        def reqTickByTickData(
            self,
            _contract,
            _tick_type,
            *,
            numberOfTicks,
            ignoreSize,
        ):
            assert numberOfTicks == 0
            assert ignoreSize is False
            return SimpleNamespace(
                contract=contract,
                tickByTicks=[
                    SimpleNamespace(
                        time=now,
                        price=Decimal("95"),
                        size=10**100,
                        exchange="X" * 256,
                    )
                ],
            )

        def cancelTickByTickData(self, _contract, _tick_type):
            return None

    monkeypatch.setattr(worker, "ib", FakeIB())
    monkeypatch.setattr(worker, "WORKER_GENERATION_ID", generation_id)
    monkeypatch.setattr(worker, "_PROTECTIVE_TICK_EVENT_SEQUENCE", iter([10**1000]))
    monkeypatch.setattr(worker, "get_market_session", lambda _timestamp: "regular")

    worker_response = await worker.handle_get_protective_quotes(
        {"symbols": ["AAPL"], "active_symbols": ["AAPL"]}
    )
    assert worker_response["status"] == "success"
    source_event_id = worker_response["data"]["quotes"][0]["source_event_id"]
    assert len(source_event_id) == len("protective:v1:") + 64
    assert len(source_event_id) <= MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH

    client, _generation = _connected_client(
        monkeypatch,
        worker_response["data"],
        generation_id=generation_id,
    )
    quote = (await client.get_protective_quotes(["AAPL"]))[0]
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )

    accepted = await monitor.update_price(
        quote.symbol,
        quote.price,
        source_timestamp=quote.source_timestamp,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=quote.con_id,
        transport_generation=quote.transport_generation,
        source_event_id=quote.source_event_id,
    )

    assert accepted is True
    evidence = monitor.get_protective_quote_evidence("AAPL")
    assert evidence is not None
    assert evidence.source_event_id == source_event_id
    assert evidence.transport_generation == generation_id


@pytest.mark.asyncio
async def test_shared_source_event_id_boundary_is_accepted_at_128_and_rejected_at_129() -> None:
    event_time = datetime.now(timezone.utc)
    boundary_id = "x" * MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH
    quote = _broker_quote(
        price=Decimal("101"),
        source_timestamp=event_time,
        source_event_id=boundary_id,
    )
    assert quote.source_event_id == boundary_id

    with pytest.raises(MarketDataIdentityError, match="source_event_id is malformed"):
        _broker_quote(
            price=Decimal("101"),
            source_timestamp=event_time,
            source_event_id="x" * (MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH + 1),
        )

    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )
    accepted = await monitor.update_price(
        quote.symbol,
        quote.price,
        source_timestamp=quote.source_timestamp,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=quote.con_id,
        transport_generation=quote.transport_generation,
        source_event_id=boundary_id,
    )
    rejected = await monitor.update_price(
        quote.symbol,
        quote.price,
        source_timestamp=quote.source_timestamp,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=quote.con_id,
        transport_generation=quote.transport_generation,
        source_event_id="x" * (MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH + 1),
    )

    assert accepted is True
    assert rejected is False


@pytest.mark.asyncio
async def test_client_binds_quote_to_exact_current_transport_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, generation = _connected_client(monkeypatch, _client_response())

    quotes = await client.get_protective_quotes(["AAPL"])

    assert len(quotes) == 1
    quote = quotes[0]
    assert type(quote) is BrokerProtectiveQuote
    assert quote.transport_generation == generation.generation_id
    assert quote.source is MarketDataSource.IBKR_LIVE_LAST_TRADE
    assert str(quote.price) == "187.25"


@pytest.mark.asyncio
async def test_client_sends_full_account_active_set_with_each_fetch_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _generation = _connected_client(monkeypatch, _client_response())

    await client.get_protective_quotes(
        ["AAPL"],
        active_symbols=["AAPL", "MSFT"],
    )

    command = client._execute_command_unlocked.await_args.args[0]
    assert command == {
        "command": "get_protective_quotes",
        "params": {
            "symbols": ["AAPL"],
            "active_symbols": ["AAPL", "MSFT"],
        },
    }


@pytest.mark.asyncio
async def test_client_empty_fetch_explicitly_cancels_all_account_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _client_response()
    response["quotes"] = []
    client, _generation = _connected_client(monkeypatch, response)

    quotes = await client.get_protective_quotes([], active_symbols=[])

    assert quotes == ()
    command = client._execute_command_unlocked.await_args.args[0]
    assert command["params"] == {"symbols": [], "active_symbols": []}


@pytest.mark.asyncio
async def test_client_rejects_fetch_symbol_missing_from_account_active_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _generation = _connected_client(monkeypatch, _client_response())

    with pytest.raises(ValueError, match="not account-active"):
        await client.get_protective_quotes(["AAPL"], active_symbols=["MSFT"])

    client._execute_command_unlocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_canonical_client_rejects_non_trade_bars_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _generation = _connected_client(monkeypatch, {})

    with pytest.raises(ValueError, match="exact what_to_show='TRADES'"):
        await client.get_canonical_historical_bars(
            "AAPL",
            what_to_show="MIDPOINT",
        )

    client._execute_command_unlocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_accepts_ordered_multi_event_replay_with_unique_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _client_response()
    retrieval = datetime.fromisoformat(response["retrieval_timestamp"])
    first = dict(
        response["quotes"][0],
        price="95",
        source_timestamp=(retrieval - timedelta(seconds=2)).isoformat(),
        source_event_id="protective:generation-1:1",
    )
    second = dict(
        response["quotes"][0],
        price="96",
        source_timestamp=(retrieval - timedelta(seconds=1)).isoformat(),
        source_event_id="protective:generation-1:2",
    )
    response["quotes"] = [first, second]
    client, _generation = _connected_client(monkeypatch, response)

    quotes = await client.get_protective_quotes(["AAPL"])

    assert [quote.price for quote in quotes] == [Decimal("95"), Decimal("96")]
    assert [quote.source_event_id for quote in quotes] == [
        "protective:generation-1:1",
        "protective:generation-1:2",
    ]


@pytest.mark.asyncio
async def test_client_poisons_generation_for_out_of_order_event_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _client_response()
    retrieval = datetime.fromisoformat(response["retrieval_timestamp"])
    response["quotes"] = [
        dict(
            response["quotes"][0],
            source_timestamp=(retrieval - timedelta(seconds=1)).isoformat(),
            source_event_id="protective:generation-1:1",
        ),
        dict(
            response["quotes"][0],
            source_timestamp=(retrieval - timedelta(seconds=2)).isoformat(),
            source_event_id="protective:generation-1:2",
        ),
    ]
    client, generation = _connected_client(monkeypatch, response)

    with pytest.raises(IBKRTransportPoisonedError, match="out of order"):
        await client.get_protective_quotes(["AAPL"])

    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_non_live_quote_poisons_the_exact_worker_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, generation = _connected_client(
        monkeypatch,
        _client_response(market_data_type=3),
    )

    with pytest.raises(IBKRTransportPoisonedError, match="not live"):
        await client.get_protective_quotes(["AAPL"])

    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_session_timestamp_mismatch_poisons_the_exact_worker_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _client_response()
    client, generation = _connected_client(monkeypatch, response)
    monkeypatch.setattr(client_module, "get_market_session", lambda _timestamp: "after-hours")

    with pytest.raises(IBKRTransportPoisonedError, match="session contradicts"):
        await client.get_protective_quotes(["AAPL"])

    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_oversized_event_identity_poisons_the_exact_worker_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _client_response()
    response["quotes"][0]["source_event_id"] = "x" * (MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH + 1)
    client, generation = _connected_client(monkeypatch, response)

    with pytest.raises(IBKRTransportPoisonedError, match="exceeds its bound"):
        await client.get_protective_quotes(["AAPL"])

    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_gateway_distributes_exact_decimal_quote_to_attached_monitor() -> None:
    quote = _broker_quote()

    class FakeClient:
        async def get_protective_quotes(self, symbols, *, active_symbols=None):
            assert tuple(symbols) == ("AAPL",)
            assert active_symbols is None or tuple(active_symbols) == ("AAPL",)
            return (quote,)

        async def stop(self):
            return None

    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._client = FakeClient()
    gateway._protective_quote_producers = {"default": monitor}

    result = await gateway.refresh_protective_quotes(("AAPL",))

    assert result == (quote,)
    evidence = monitor.get_protective_quote_evidence("AAPL")
    assert evidence is not None
    assert evidence.price == Decimal("187.2500")
    assert evidence.source is ProtectiveQuoteSource.LIVE_BROKER
    assert evidence.transport_generation == "generation-1"


@pytest.mark.asyncio
async def test_gateway_ordered_intrapoll_replay_latches_crossing_before_recovery() -> None:
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )
    stop = await monitor.add_stop_loss(
        "AAPL",
        Position(symbol="AAPL", quantity=10, avg_price=Decimal("100")),
        stop_percent=0.02,
    )
    event_time = datetime.now(timezone.utc)
    crossing = _broker_quote(
        price=Decimal("95"),
        source_timestamp=event_time,
        source_event_id="protective:generation-1:crossing",
    )
    recovery = _broker_quote(
        price=Decimal("105"),
        source_timestamp=event_time,
        source_event_id="protective:generation-1:recovery",
    )

    class FakeClient:
        async def get_protective_quotes(self, symbols, *, active_symbols=None):
            assert tuple(symbols) == ("AAPL",)
            assert active_symbols is None or tuple(active_symbols) == ("AAPL",)
            return (crossing, recovery)

        async def stop(self):
            return None

    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._client = FakeClient()
    gateway._protective_quote_producers = {"default": monitor}

    published = await gateway.refresh_protective_quotes(("AAPL",))

    assert [quote.price for quote in published] == [Decimal("95"), Decimal("105")]
    assert stop.status is StopStatus.TRIGGERED
    assert stop.trigger_price == 95.0
    latched = monitor._latched_stop_crossings[id(stop)]
    assert latched.trigger_price == 95.0
    assert latched.quote_evidence is not None
    assert latched.quote_evidence.price == Decimal("95")
    current = monitor.get_protective_quote_evidence("AAPL")
    assert current is not None
    assert current.price == Decimal("105")
    assert current.source_event_id == "protective:generation-1:recovery"
    assert monitor.last_prices["AAPL"] == 105.0


@pytest.mark.asyncio
async def test_gateway_distributes_one_account_quote_to_each_portfolio_monitor() -> None:
    quote = _broker_quote()

    class FakeClient:
        async def get_protective_quotes(self, _symbols, *, active_symbols=None):
            return (quote,)

        async def stop(self):
            return None

    monitors = {
        portfolio_id: StopLossMonitor(
            execute_reduction=AsyncMock(),
            risk_manager=None,
            portfolio_id=portfolio_id,
        )
        for portfolio_id in ("growth", "income")
    }
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._client = FakeClient()
    gateway._protective_quote_producers = monitors

    await gateway.refresh_protective_quotes(("AAPL",))

    for portfolio_id, monitor in monitors.items():
        evidence = monitor.get_protective_quote_evidence("AAPL")
        assert evidence is not None
        assert evidence.portfolio_id == portfolio_id
        assert evidence.price == quote.price


@pytest.mark.asyncio
async def test_runner_opens_feed_gate_only_after_gateway_owned_quote_refresh() -> None:
    quote = _broker_quote()

    class FakeClient:
        async def get_protective_quotes(self, _symbols, *, active_symbols=None):
            return (quote,)

        async def stop(self):
            return None

    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._client = FakeClient()
    gateway._protective_quote_producers = {"default": monitor}
    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.paper_reduction_gateway = gateway
    runner.stop_loss_monitor = monitor
    runner.latest_prices = {}
    runner.latest_price_times = {}
    runner.latest_price_sources = {}
    runner._protective_feed_status = {}

    assert runner._has_live_protective_feed("AAPL") is False
    assert await runner._refresh_live_protective_quotes(("AAPL",)) is True
    assert runner._has_live_protective_feed("AAPL") is True
    assert runner.latest_price_sources["AAPL"] == "live_protective"


@pytest.mark.asyncio
async def test_gateway_refresh_failure_revokes_evidence_and_retires_generation() -> None:
    quote = _broker_quote()

    class FakeClient:
        def __init__(self):
            self.fail = False
            self.stop_count = 0

        async def get_protective_quotes(self, _symbols, *, active_symbols=None):
            if self.fail:
                raise RuntimeError("poisoned protective transport")
            return (quote,)

        async def stop(self):
            self.stop_count += 1

    client = FakeClient()
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._client = client
    gateway._protective_quote_producers = {"default": monitor}
    await gateway.refresh_protective_quotes(("AAPL",))
    assert monitor.get_protective_quote_evidence("AAPL") is not None

    client.fail = True
    with pytest.raises(RuntimeError, match="poisoned protective transport"):
        await gateway.refresh_protective_quotes(("AAPL",))

    assert monitor.get_protective_quote_evidence("AAPL") is None
    assert gateway._started is False
    assert gateway._diagnostic_recovery_required is True
    assert client.stop_count == 1


@pytest.mark.asyncio
async def test_gateway_chunks_fetches_without_shrinking_account_active_set() -> None:
    symbols = tuple("Q" + chr(65 + (index // 26)) + chr(65 + (index % 26)) for index in range(130))

    class FakeClient:
        def __init__(self):
            self.calls = []

        async def get_protective_quotes(self, requested, *, active_symbols=None):
            self.calls.append((tuple(requested), tuple(active_symbols or ())))
            return tuple(
                _broker_quote(
                    symbol=symbol,
                    con_id=300000 + symbols.index(symbol),
                    source_event_id=f"protective:generation-1:{symbol}",
                )
                for symbol in requested
            )

        async def stop(self):
            return None

    client = FakeClient()
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._client = client
    gateway._protective_quote_producers = {"default": monitor}

    quotes = await gateway.refresh_protective_quotes(symbols)

    assert len(quotes) == 130
    assert [len(call[0]) for call in client.calls] == [64, 64, 2]
    assert all(call[1] == symbols for call in client.calls)


@pytest.mark.asyncio
async def test_entry_quote_that_crosses_stop_blocks_entry_before_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robo_trader.safety import readiness as paper_readiness

    monkeypatch.setattr(paper_readiness, "PAPER_TERMINAL_SETTLEMENT_READY", True)
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )
    stop = await monitor.add_stop_loss(
        "AAPL",
        Position(symbol="AAPL", quantity=10, avg_price=Decimal("100")),
        stop_percent=0.02,
    )
    crossing = _broker_quote(price=Decimal("95"))

    class FakeClient:
        async def ping(self):
            return True

        async def get_protective_quotes(self, requested, *, active_symbols=None):
            assert tuple(requested) == ("AAPL",)
            assert tuple(active_symbols or ()) == ("AAPL",)
            return (crossing,)

        async def stop(self):
            return None

    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._client = FakeClient()
    gateway._protective_quote_producers = {"default": monitor}
    entered = False

    with pytest.raises(PaperReductionGatewayError, match="protective reduction is pending"):
        async with gateway.serialize_entry("AAPL"):
            entered = True

    assert entered is False
    assert stop.status is StopStatus.TRIGGERED
    assert await monitor.has_pending_reduction() is True
    assert gateway._started is True


@pytest.mark.asyncio
async def test_entry_refresh_crossing_in_another_portfolio_blocks_before_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robo_trader.safety import readiness as paper_readiness

    monkeypatch.setattr(paper_readiness, "PAPER_TERMINAL_SETTLEMENT_READY", True)
    entry_monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="entry",
    )
    protected_monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="protected",
    )
    protected_stop = await protected_monitor.add_stop_loss(
        "MSFT",
        Position(symbol="MSFT", quantity=10, avg_price=Decimal("100")),
        stop_percent=0.02,
    )
    quotes = {
        "AAPL": _broker_quote(
            symbol="AAPL",
            con_id=265598,
            price=Decimal("187.25"),
            source_event_id="protective:generation-1:aapl",
        ),
        "MSFT": _broker_quote(
            symbol="MSFT",
            con_id=272093,
            price=Decimal("95"),
            source_event_id="protective:generation-1:msft-crossing",
        ),
    }

    class FakeClient:
        async def ping(self):
            return True

        async def get_protective_quotes(self, requested, *, active_symbols=None):
            assert tuple(requested) == ("AAPL", "MSFT")
            assert tuple(active_symbols or ()) == ("AAPL", "MSFT")
            return tuple(quotes[symbol] for symbol in requested)

        async def stop(self):
            return None

    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._account_order_gate = asyncio.Lock()
    gateway._client = FakeClient()
    gateway._protective_quote_producers = {
        "entry": entry_monitor,
        "protected": protected_monitor,
    }
    entered = False

    with pytest.raises(PaperReductionGatewayError, match="protective reduction is pending"):
        async with gateway.serialize_entry("AAPL"):
            entered = True

    assert entered is False
    assert protected_stop.status is StopStatus.TRIGGERED
    assert protected_stop.trigger_price == 95.0
    assert await protected_monitor.has_pending_reduction() is True


@pytest.mark.asyncio
async def test_retired_generation_waits_before_same_symbol_resubscribe() -> None:
    clock = {"now": 100.0}
    sleeps: list[float] = []
    request_times: list[float] = []
    quote = _broker_quote()

    async def controlled_sleep(delay: float) -> None:
        sleeps.append(delay)
        clock["now"] += delay

    class FakeClient:
        async def get_protective_quotes(self, requested, *, active_symbols=None):
            request_times.append(clock["now"])
            return (quote,)

        async def stop(self):
            return None

    gateway = object.__new__(PaperReductionGateway)
    gateway._client = FakeClient()
    gateway._monotonic = lambda: clock["now"]
    gateway._sleep = controlled_sleep
    gateway._generation_symbol_first_request = {}
    gateway._generation_subscribed_symbols = set()
    gateway._resubscribe_not_before = 0.0
    gateway._tick_resubscribe_cooldown_seconds = 15.0

    await gateway._fetch_protective_quotes_locked(
        ("AAPL",),
        active_symbols=("AAPL",),
    )
    await gateway._stop_client_owned()
    await gateway._fetch_protective_quotes_locked(
        ("AAPL",),
        active_symbols=("AAPL",),
    )

    assert sleeps == [15.0]
    assert request_times == [100.0, 115.0]
    assert request_times[1] - request_times[0] >= 15.0


@pytest.mark.asyncio
async def test_cooldown_starts_after_delayed_worker_command_completion() -> None:
    clock = {"now": 100.0}
    command_starts: list[float] = []
    sleeps: list[float] = []
    quote = _broker_quote()

    async def controlled_sleep(delay: float) -> None:
        sleeps.append(delay)
        clock["now"] += delay

    class FakeClient:
        def __init__(self):
            self.call_count = 0

        async def get_protective_quotes(self, requested, *, active_symbols=None):
            self.call_count += 1
            command_starts.append(clock["now"])
            if self.call_count == 1:
                # Simulate qualification and broker-time work before the
                # worker reaches reqTickByTickData and completes its command.
                clock["now"] += 14.0
            return (quote,)

        async def stop(self):
            return None

    gateway = object.__new__(PaperReductionGateway)
    gateway._client = FakeClient()
    gateway._monotonic = lambda: clock["now"]
    gateway._sleep = controlled_sleep
    gateway._generation_symbol_first_request = {}
    gateway._generation_subscribed_symbols = set()
    gateway._resubscribe_not_before = 0.0
    gateway._tick_resubscribe_cooldown_seconds = 15.0

    await gateway._fetch_protective_quotes_locked(
        ("AAPL",),
        active_symbols=("AAPL",),
    )
    assert clock["now"] == 114.0
    await gateway._stop_client_owned()
    await gateway._fetch_protective_quotes_locked(
        ("AAPL",),
        active_symbols=("AAPL",),
    )

    assert sleeps == [15.0]
    assert command_starts == [100.0, 129.0]


@pytest.mark.asyncio
async def test_cooldown_uses_delayed_first_request_time_from_later_chunk() -> None:
    symbols = tuple(f"Q{index:03d}" for index in range(65))
    delayed_symbol = symbols[-1]
    clock = {"now": 100.0}
    request_times: dict[str, list[float]] = {}
    sleeps: list[float] = []

    async def controlled_sleep(delay: float) -> None:
        sleeps.append(delay)
        clock["now"] += delay

    class FakeClient:
        def __init__(self):
            self.call_count = 0

        async def get_protective_quotes(self, requested, *, active_symbols=None):
            self.call_count += 1
            for symbol in requested:
                request_times.setdefault(symbol, []).append(clock["now"])
            response = tuple(SimpleNamespace(symbol=symbol) for symbol in requested)
            if self.call_count == 1:
                clock["now"] += 14.0
            return response

        async def stop(self):
            return None

    gateway = object.__new__(PaperReductionGateway)
    gateway._client = FakeClient()
    gateway._monotonic = lambda: clock["now"]
    gateway._sleep = controlled_sleep
    gateway._generation_symbol_first_request = {}
    gateway._generation_subscribed_symbols = set()
    gateway._resubscribe_not_before = 0.0
    gateway._tick_resubscribe_cooldown_seconds = 15.0

    await gateway._fetch_protective_quotes_locked(
        symbols,
        active_symbols=symbols,
    )
    assert request_times[delayed_symbol] == [114.0]
    clock["now"] += 1.0
    await gateway._stop_client_owned()
    await gateway._fetch_protective_quotes_locked(
        (delayed_symbol,),
        active_symbols=(delayed_symbol,),
    )

    assert sleeps == [14.0]
    assert request_times[delayed_symbol] == [114.0, 129.0]


@pytest.mark.asyncio
async def test_feed_task_is_singleton_and_cancellable() -> None:
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=None,
        portfolio_id="default",
    )
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._protective_quote_producers = {"default": monitor}
    gateway._protective_feed_task = None
    gateway._protective_feed_enabled = False
    gateway._protective_feed_interval_seconds = 3600.0
    gateway._protective_feed_max_recovery_attempts = 3

    gateway.start_protective_feed()
    task = gateway._protective_feed_task
    gateway.start_protective_feed()

    assert task is gateway._protective_feed_task
    assert task is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_feed_recovers_before_failure_limit_without_quarantine() -> None:
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._terminal_quarantine_reason = None
    gateway._bindings = {}
    gateway._protective_quote_producers = {}
    gateway._protective_feed_enabled = True
    gateway._protective_feed_interval_seconds = 0
    gateway._protective_feed_max_recovery_attempts = 3
    gateway.refresh_protective_quotes = AsyncMock(
        side_effect=[RuntimeError("transient feed failure"), ()]
    )

    async def stop_after_recovery(_seconds: float) -> None:
        if gateway.refresh_protective_quotes.await_count == 2:
            gateway._protective_feed_enabled = False

    gateway._sleep = stop_after_recovery

    await gateway._protective_feed_loop()

    assert gateway.refresh_protective_quotes.await_count == 2
    assert gateway.terminal_quarantine_reason is None
    assert gateway.can_attempt_order_admission is True


@pytest.mark.asyncio
async def test_feed_terminal_failure_quarantines_and_alerts_every_runner() -> None:
    alerts = {portfolio_id: AsyncMock() for portfolio_id in ("growth", "income")}
    producers = {
        portfolio_id: SimpleNamespace(emergency_shutdown=callback)
        for portfolio_id, callback in alerts.items()
    }
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway._terminal_quarantine_reason = None
    gateway._bindings = {}
    gateway._protective_quote_producers = producers
    gateway._protective_feed_enabled = True
    gateway._protective_feed_interval_seconds = 0
    gateway._protective_feed_max_recovery_attempts = 3
    gateway.refresh_protective_quotes = AsyncMock(
        side_effect=RuntimeError("persistent feed failure")
    )
    sleep = AsyncMock()
    gateway._sleep = sleep

    await gateway._protective_feed_loop()

    assert gateway.refresh_protective_quotes.await_count == 3
    assert sleep.await_count == 2
    assert gateway._protective_feed_enabled is False
    assert gateway.can_attempt_order_admission is False
    assert "3 consecutive refresh failures" in gateway.terminal_quarantine_reason
    for callback in alerts.values():
        callback.assert_awaited_once_with(gateway.terminal_quarantine_reason)
