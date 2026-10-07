"""Explicit synthetic transport ownership for runner admission unit tests."""

from types import SimpleNamespace

from robo_trader.clients.subprocess_ibkr_client import (
    QualifiedStockContractLineage,
    SubprocessIBKRClient,
    _WorkerGeneration,
)


def bind_test_canonical_batch(runner, batch):
    contract = batch.contract
    client = SubprocessIBKRClient()
    generation = _WorkerGeneration(
        generation_id=contract.transport_generation, process=SimpleNamespace(poll=lambda: None)
    )
    client._generation = generation
    client._connected = True
    client._connection_generation_id = generation.generation_id
    client._connection_identity = ("127.0.0.1", 4002, 7, True)
    lineage = QualifiedStockContractLineage(
        con_id=contract.con_id,
        symbol=contract.symbol,
        local_symbol=contract.symbol,
        security_type="STK",
        currency="USD",
        exchange=contract.exchange,
        primary_exchange=contract.primary_exchange,
        trading_class="NMS",
        broker_timestamp=contract.broker_time,
        retrieval_timestamp=contract.retrieval_time,
        transport_generation=contract.transport_generation,
    )
    client._cache_historical_lineage(generation, lineage)
    client._check_canonical_batch(batch, publishing_generation=generation)
    runner.ib = client
