from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from robo_trader.clients.subprocess_ibkr_client import (
    QualifiedStockContractLineage,
    SubprocessIBKRClient,
    _WorkerGeneration,
)
from robo_trader.database_async import (
    AsyncTradingDatabase,
    SafetyAllocationSnapshot,
    SafetyPortfolioAllocation,
)
from robo_trader.safety import (
    AccountPosition,
    OrderSide,
    OrderType,
    PaperExecutionIdentity,
    ReconciliationStatus,
    RuntimeOrderRequest,
    SafetyJournal,
    SafetyRuntimeCoordinator,
    TimeInForce,
)
from robo_trader.safety_runtime_evidence import (
    BrokerAccountPositionSnapshot,
    BrokerOpenOrderEvidence,
    BrokerReconciliationSnapshot,
    TrustedEvidenceAssemblyError,
    assemble_trusted_safety_evidence,
)

ACCOUNT_SCOPE = "acct_v1_" + ("0123456789abcdef" * 4)


@pytest.mark.asyncio
async def test_trusted_adapter_preserves_db_and_broker_lineage_end_to_end(tmp_path):
    database = AsyncTradingDatabase(db_path=tmp_path / "paper.db", pool_size=2)
    await database.initialize()
    try:
        now = datetime.now(timezone.utc)
        async with database.get_connection() as connection:
            await connection.execute(
                """
                INSERT INTO positions
                    (portfolio_id, symbol, quantity, avg_cost, market_price, timestamp)
                VALUES ('default', 'AAPL', 2, 100.0, 100.0, ?)
                """,
                (now.isoformat(),),
            )
            await connection.commit()
        allocation = await database.get_safety_allocation_snapshot("AAPL")
    finally:
        await database.close()

    client = SubprocessIBKRClient()
    generation = _WorkerGeneration(
        "worker-generation-one",
        SimpleNamespace(poll=lambda: 0),
    )
    client._generation = generation
    client._connected = True
    client._connection_generation_id = generation.generation_id
    client._connection_identity = ("127.0.0.1", 4002, 123, True)
    response = {
        "bars": [],
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
        "broker_timestamp": now.isoformat(),
        "retrieval_timestamp": now.isoformat(),
    }
    client._validate_historical_response("AAPL", response, generation)
    lineage = client.get_cached_historical_lineage("AAPL")

    account = BrokerAccountPositionSnapshot(
        observed_at=now,
        snapshot_id="broker-account-snapshot-1",
        source="ibkr-account-updates",
        transport_generation=generation.generation_id,
        positions=(AccountPosition(265598, "AAPL", Decimal("2")),),
        complete=True,
    )
    orders = BrokerOpenOrderEvidence(
        observed_at=now,
        snapshot_id="broker-open-orders-1",
        source="ibkr-all-open-orders",
        transport_generation=generation.generation_id,
        active_con_ids=(),
        complete=True,
        all_clients=True,
        stable=True,
    )
    reconciliation = BrokerReconciliationSnapshot(
        observed_at=now,
        snapshot_id="broker-reconciliation-1",
        source="ibkr-readonly-reconciliation",
        transport_generation=generation.generation_id,
        status=ReconciliationStatus.PASSED,
    )
    identity = PaperExecutionIdentity("paper-simulator-v1", ACCOUNT_SCOPE)

    contract, snapshot = assemble_trusted_safety_evidence(
        identity,
        lineage,
        allocation,
        account,
        orders,
        reconciliation,
    )

    assert snapshot.allocation_snapshot_id == allocation.snapshot_id
    assert snapshot.allocation_observed_at == allocation.observed_at
    assert snapshot.snapshot_id == account.snapshot_id
    assert snapshot.open_orders.snapshot_id == orders.snapshot_id
    assert snapshot.reconciliation_snapshot_id == reconciliation.snapshot_id
    assert contract.transport_generation == generation.generation_id
    assert snapshot.portfolio_allocations[0].con_id == contract.con_id

    evaluation_time = datetime.now(timezone.utc)
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: evaluation_time)
    journal.initialize(
        execution_domain_scope=identity.execution_domain_scope,
        account_scope=identity.account_scope,
    )
    coordinator = SafetyRuntimeCoordinator(
        identity,
        journal,
        clock=lambda: evaluation_time,
    )
    coordinator.start()
    authorization = coordinator.authorize(
        "trusted-adapter-close",
        RuntimeOrderRequest(
            portfolio_id="default",
            contract=contract,
            side=OrderSide.SELL,
            quantity=Decimal("1"),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_ref="trusted-adapter-close",
        ),
        snapshot,
    )
    assert authorization.decision.allowed
    assert authorization.allocation_snapshot_id == allocation.snapshot_id


def test_trusted_adapter_rejects_cross_generation_broker_evidence():
    now = datetime.now(timezone.utc)
    account = BrokerAccountPositionSnapshot(
        observed_at=now,
        snapshot_id="account",
        source="ibkr-account",
        transport_generation="generation-a",
        positions=(AccountPosition(1, "AAPL", Decimal("1")),),
        complete=True,
    )
    orders = BrokerOpenOrderEvidence(
        observed_at=now,
        snapshot_id="orders",
        source="ibkr-orders",
        transport_generation="generation-b",
        active_con_ids=(),
        complete=True,
        all_clients=True,
        stable=True,
    )
    lineage = QualifiedStockContractLineage(
        con_id=1,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
        broker_timestamp=now,
        retrieval_timestamp=now,
        transport_generation="generation-a",
    )
    allocation = SafetyAllocationSnapshot(
        snapshot_id="allocation",
        observed_at=now,
        symbol="AAPL",
        allocations=(
            SafetyPortfolioAllocation(
                portfolio_id="default",
                symbol="AAPL",
                quantity=Decimal("1"),
                updated_at=now,
            ),
        ),
        aggregate_allocated_quantity=Decimal("1"),
        has_offsetting_allocations=False,
        complete=True,
    )
    assert account.transport_generation != orders.transport_generation
    with pytest.raises(TrustedEvidenceAssemblyError, match="multiple transport generations"):
        # Exact producer-type validation happens before any field is dereferenced
        # from an untrusted placeholder allocation/contract.
        assemble_trusted_safety_evidence(
            PaperExecutionIdentity("paper-simulator-v1", ACCOUNT_SCOPE),
            lineage,
            allocation,
            account,
            orders,
            BrokerReconciliationSnapshot(
                observed_at=now,
                snapshot_id="reconciliation",
                source="ibkr-reconciliation",
                transport_generation="generation-a",
                status=ReconciliationStatus.PASSED,
            ),
        )
