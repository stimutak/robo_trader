from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from robo_trader.clients.subprocess_ibkr_client import (
    IBKRError,
    IBKRTransportPoisonedError,
    QualifiedStockContractLineage,
    SubprocessIBKRClient,
    _WorkerGeneration,
)


def _generation(identifier: str) -> _WorkerGeneration:
    return _WorkerGeneration(identifier, SimpleNamespace(poll=lambda: 0))


def _bind_connected_generation(
    client: SubprocessIBKRClient,
    generation: _WorkerGeneration,
) -> None:
    client._generation = generation
    client._connected = True
    client._connection_generation_id = generation.generation_id
    client._connection_identity = ("127.0.0.1", 4002, 123, True)


def _historical_data() -> dict:
    return {
        "bars": [
            {
                "date": "2026-07-25T14:30:00+00:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
            }
        ],
        "requested_symbol": "AAPL",
        "qualified_contract": {
            "con_id": 265598,
            "symbol": "AAPL",
            "local_symbol": "AAPL",
            "security_type": "STK",
            "currency": "USD",
            "exchange": "SMART",
            "primary_exchange": "NASDAQ",
            "trading_class": "NMS",
        },
        "broker_timestamp": "2026-07-25T14:31:00+00:00",
        "retrieval_timestamp": "2026-07-25T14:31:01+00:00",
    }


@pytest.mark.asyncio
async def test_existing_bars_api_preserves_immutable_full_contract_lineage(monkeypatch):
    client = SubprocessIBKRClient()
    generation = _generation("generation-one")
    _bind_connected_generation(client, generation)
    response = _historical_data()

    async def execute(command, timeout):
        assert timeout == 60.0
        assert command["params"]["symbol"] == "AAPL"
        return response

    monkeypatch.setattr(client, "_execute_command_unlocked", execute)

    bars = await client.get_historical_bars("aapl")
    assert bars == response["bars"]

    lineage = client.get_cached_historical_lineage("AAPL")
    assert lineage == QualifiedStockContractLineage(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
        broker_timestamp=datetime(2026, 7, 25, 14, 31, tzinfo=timezone.utc),
        retrieval_timestamp=datetime(2026, 7, 25, 14, 31, 1, tzinfo=timezone.utc),
        transport_generation="generation-one",
    )
    with pytest.raises(FrozenInstanceError):
        lineage.con_id = 1
    assert not hasattr(lineage, "__dict__")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("con_id", None),
        ("con_id", True),
        ("con_id", 0),
        ("symbol", "MSFT"),
        ("symbol", " aapl "),
        ("local_symbol", "AAPL ALIAS"),
        ("security_type", "OPT"),
        ("currency", "EUR"),
        ("exchange", "NYSE"),
        ("primary_exchange", ""),
        ("trading_class", None),
    ],
)
def test_malformed_or_missing_contract_identity_poisons_and_is_not_cached(field, value):
    client = SubprocessIBKRClient()
    generation = _generation("generation-one")
    _bind_connected_generation(client, generation)
    data = _historical_data()
    if value is None:
        data["qualified_contract"].pop(field)
    else:
        data["qualified_contract"][field] = value

    with pytest.raises(IBKRTransportPoisonedError):
        client._validate_historical_response("AAPL", data, generation)

    assert generation.poisoned_reason
    assert client._historical_lineage_by_symbol == {}


@pytest.mark.parametrize(
    "field",
    ["broker_timestamp", "retrieval_timestamp"],
)
def test_naive_or_missing_lineage_timestamp_fails_closed(field):
    client = SubprocessIBKRClient()
    generation = _generation("generation-one")
    _bind_connected_generation(client, generation)
    data = _historical_data()
    data[field] = "2026-07-25T14:31:00"

    with pytest.raises(IBKRTransportPoisonedError, match="timezone-aware"):
        client._validate_historical_response("AAPL", data, generation)

    assert client._historical_lineage_by_symbol == {}


def test_cached_lineage_is_invalidated_and_rejected_after_generation_change():
    client = SubprocessIBKRClient()
    old_generation = _generation("generation-one")
    _bind_connected_generation(client, old_generation)
    client._validate_historical_response("AAPL", _historical_data(), old_generation)
    assert client.get_cached_historical_lineage("AAPL").transport_generation == "generation-one"

    _bind_connected_generation(client, _generation("generation-two"))

    with pytest.raises(IBKRTransportPoisonedError, match="stale worker generation"):
        client.get_cached_historical_lineage("AAPL")
    assert client._historical_lineage_generation_id is None
    assert client._historical_lineage_by_symbol == {}


def test_stale_generation_response_is_rejected_without_repopulating_cache():
    client = SubprocessIBKRClient()
    stale_generation = _generation("generation-one")
    _bind_connected_generation(client, _generation("generation-two"))

    with pytest.raises(IBKRTransportPoisonedError, match="stale worker generation"):
        client._validate_historical_response(
            "AAPL",
            _historical_data(),
            stale_generation,
        )

    assert stale_generation.poisoned_reason
    assert client._historical_lineage_by_symbol == {}


def test_disconnect_invalidates_lineage_and_same_worker_reconnect_cannot_reuse_it():
    client = SubprocessIBKRClient()
    generation = _generation("generation-one")
    _bind_connected_generation(client, generation)
    client._validate_historical_response("AAPL", _historical_data(), generation)

    assert client._clear_cached_connection_state(generation=generation)
    assert client._historical_lineage_by_symbol == {}
    with pytest.raises(IBKRTransportPoisonedError, match="stale worker generation"):
        client.get_cached_historical_lineage("AAPL")

    _bind_connected_generation(client, generation)
    with pytest.raises(IBKRError, match="No current qualified-contract lineage"):
        client.get_cached_historical_lineage("AAPL")


@pytest.mark.parametrize("case", ["broker_after_retrieval", "future_retrieval"])
def test_contract_lineage_rejects_timestamp_ordering_or_clock_skew(case):
    client = SubprocessIBKRClient()
    generation = _generation("generation-one")
    _bind_connected_generation(client, generation)
    data = _historical_data()
    if case == "broker_after_retrieval":
        data["broker_timestamp"] = "2026-07-25T14:40:00+00:00"
    else:
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        data["retrieval_timestamp"] = future.isoformat()

    with pytest.raises(IBKRTransportPoisonedError):
        client._validate_historical_response("AAPL", data, generation)

    assert client._historical_lineage_by_symbol == {}


@pytest.mark.parametrize("broker_offset_seconds", [-121, 121])
def test_contract_lineage_rejects_absolute_broker_retrieval_skew(
    broker_offset_seconds,
):
    client = SubprocessIBKRClient()
    generation = _generation("generation-one")
    _bind_connected_generation(client, generation)
    data = _historical_data()
    retrieval = datetime.fromisoformat(data["retrieval_timestamp"])
    data["broker_timestamp"] = (retrieval + timedelta(seconds=broker_offset_seconds)).isoformat()

    with pytest.raises(IBKRTransportPoisonedError, match="clock-skew tolerance"):
        client._validate_historical_response("AAPL", data, generation)

    assert generation.poisoned_reason
    assert client._historical_lineage_by_symbol == {}


@pytest.mark.parametrize("broker_offset_seconds", [-120, 120])
def test_contract_lineage_accepts_absolute_skew_boundary(broker_offset_seconds):
    client = SubprocessIBKRClient()
    generation = _generation("generation-one")
    _bind_connected_generation(client, generation)
    data = _historical_data()
    retrieval = datetime.fromisoformat(data["retrieval_timestamp"])
    data["broker_timestamp"] = (retrieval + timedelta(seconds=broker_offset_seconds)).isoformat()

    client._validate_historical_response("AAPL", data, generation)

    lineage = client.get_cached_historical_lineage("AAPL")
    assert abs((lineage.retrieval_timestamp - lineage.broker_timestamp).total_seconds()) == 120
