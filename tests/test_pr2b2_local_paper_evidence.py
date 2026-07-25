"""Local-ledger authority for paper-only reduction safety evidence."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from robo_trader.broker_safety_evidence import (
    BrokerSafetyContract,
    _issue_broker_contract_snapshot_capability,
    _produce_broker_contract_safety_snapshot,
)

# Import the subprocess client first to follow the package's established
# initialization order and avoid the reconciliation/broker-evidence cycle.
from robo_trader.clients.subprocess_ibkr_client import (  # noqa: F401
    SubprocessIBKRClient,
)
from robo_trader.config import RuntimeContract, _derive_safety_account_scope
from robo_trader.database_async import (
    AsyncTradingDatabase,
    SafetyAllocationSnapshotError,
)
from robo_trader.reconciliation.identity import validate_runtime_safety
from robo_trader.safety import (
    OrderSide,
    OrderType,
    PaperExecutionIdentity,
    ReconciliationStatus,
    RuntimeAuthorizationBlocked,
    RuntimeOrderRequest,
    SafetyJournal,
    SafetyRuntimeCoordinator,
    TimeInForce,
)
from robo_trader.safety.models import ValidationError
from robo_trader.safety_runtime_evidence import (
    TrustedEvidenceAssemblyError,
    assemble_local_paper_safety_evidence,
)

ACCOUNT = "DU1234567"
SCOPE_KEY = "0123456789abcdef" * 4
ACCOUNT_SCOPE = _derive_safety_account_scope(SCOPE_KEY, ACCOUNT)
GENERATION = "local-paper-generation-1"


def _runtime_contract(
    database_path: Path,
    *,
    execution_mode: str = "paper",
    state_namespace: str = "paper",
) -> RuntimeContract:
    return RuntimeContract(
        environment="dev",
        execution_mode=execution_mode,
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(database_path),
        account_alias="***4567",
        account_type="paper",
        model_artifact_set="tests",
        build_id="tests",
        state_namespace=state_namespace,
        safety_account_scope=ACCOUNT_SCOPE,
        safety_execution_domain_scope="paper-simulator-v1",
        safety_journal_path=str(database_path.with_suffix(".safety.db")),
    )


def _runtime_context(tmp_path: Path, monkeypatch, contract: RuntimeContract):
    ibc_path = tmp_path / "config" / "ibc" / "config.ini"
    ibc_path.parent.mkdir(parents=True, exist_ok=True)
    ibc_path.write_text("ReadOnlyApi=yes\nTradingMode=paper\n", encoding="utf-8")
    monkeypatch.setattr(
        "robo_trader.config.load_runtime_contract_from_env",
        lambda _environment: contract,
    )
    return validate_runtime_safety(
        tmp_path,
        {
            "IBKR_ACCOUNT": ACCOUNT,
            "IBKR_CLIENT_ID": "1",
            "IBKR_RECONCILIATION_CLIENT_ID": "7",
            "SAFETY_ACCOUNT_SCOPE_KEY": SCOPE_KEY,
        },
    )


async def _allocation_snapshot(
    database_path: Path,
    quantities: dict[str, int],
    *,
    symbol: str,
    contract: RuntimeContract,
):
    database = AsyncTradingDatabase(database_path, pool_size=1)
    await database.initialize()
    try:
        observed_at = datetime.now(timezone.utc).isoformat()
        async with database.get_connection() as connection:
            for portfolio_id, quantity in quantities.items():
                await connection.execute(
                    """
                    INSERT OR IGNORE INTO portfolios
                        (id, name, starting_cash, active)
                    VALUES (?, ?, 100000, 1)
                    """,
                    (portfolio_id, portfolio_id.title()),
                )
                await connection.execute(
                    """
                    INSERT INTO positions
                        (portfolio_id, symbol, quantity, avg_cost, market_price, timestamp)
                    VALUES (?, ?, ?, 100, 100, ?)
                    """,
                    (portfolio_id, symbol, quantity, observed_at),
                )
            await connection.commit()
        return await database.get_safety_allocation_snapshot(
            symbol,
            runtime_contract=contract,
        )
    finally:
        await database.close()


def _broker_contract_snapshot(
    runtime_context,
    *,
    symbol: str,
    observed_at: datetime,
):
    capability = _issue_broker_contract_snapshot_capability(
        runtime_context,
        connection_identity=("127.0.0.1", 4002, 7, True),
        transport_generation=GENERATION,
        requested_symbol=symbol,
    )
    contract = BrokerSafetyContract(
        con_id=272093 if symbol == "MSFT" else 265598,
        symbol=symbol,
        local_symbol=symbol,
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
    )
    return _produce_broker_contract_safety_snapshot(
        capability=capability,
        broker_time_before=observed_at,
        broker_time_after=observed_at,
        retrieved_at=observed_at,
        snapshot_id=f"local-paper-contract-{symbol}-{observed_at.timestamp()}",
        source="ibkr-subprocess-contract-safety-v1",
        qualified_contract=contract,
    )


def _identity() -> PaperExecutionIdentity:
    return PaperExecutionIdentity("paper-simulator-v1", ACCOUNT_SCOPE)


def _forge_exact_snapshot(snapshot):
    forged = object.__new__(type(snapshot))
    for model_field in fields(snapshot):
        object.__setattr__(forged, model_field.name, getattr(snapshot, model_field.name))
    return forged


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_quantity", "side"),
    [(10, OrderSide.SELL), (-10, OrderSide.BUY_TO_COVER)],
)
async def test_unheld_broker_symbol_authorizes_local_ledger_reduction(
    tmp_path: Path,
    monkeypatch,
    local_quantity: int,
    side: OrderSide,
) -> None:
    contract = _runtime_contract(tmp_path / "paper.db")
    allocation = await _allocation_snapshot(
        Path(contract.database_path),
        {"default": local_quantity},
        symbol="MSFT",
        contract=contract,
    )
    runtime_context = _runtime_context(tmp_path, monkeypatch, contract)
    broker = _broker_contract_snapshot(
        runtime_context,
        symbol="MSFT",
        observed_at=allocation.observed_at,
    )

    authoritative_contract, snapshot = assemble_local_paper_safety_evidence(
        _identity(),
        contract,
        broker,
        allocation,
    )

    assert not hasattr(broker, "positions")
    assert snapshot.account_positions[0].quantity == Decimal(local_quantity)
    assert snapshot.account_positions[0].con_id == authoritative_contract.con_id
    assert snapshot.reconciliation_status is ReconciliationStatus.PASSED

    evaluated_at = datetime.now(timezone.utc)
    journal = SafetyJournal(
        tmp_path / f"safety-{side.value}.db",
        clock=lambda: evaluated_at,
    )
    journal.initialize(
        execution_domain_scope=_identity().execution_domain_scope,
        account_scope=ACCOUNT_SCOPE,
    )
    coordinator = SafetyRuntimeCoordinator(
        _identity(),
        journal,
        clock=lambda: evaluated_at,
    )
    coordinator.start()
    authorization = coordinator.authorize(
        f"local-{side.value}",
        RuntimeOrderRequest(
            portfolio_id="default",
            contract=authoritative_contract,
            side=side,
            quantity=Decimal("3"),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_ref=f"local-{side.value}",
        ),
        snapshot,
    )

    assert authorization.decision.allowed


@pytest.mark.asyncio
async def test_aggregate_local_allocation_becomes_account_position(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract = _runtime_contract(tmp_path / "aggregate.db")
    allocation = await _allocation_snapshot(
        Path(contract.database_path),
        {"default": 7, "growth": 3},
        symbol="AAPL",
        contract=contract,
    )
    broker = _broker_contract_snapshot(
        _runtime_context(tmp_path, monkeypatch, contract),
        symbol="AAPL",
        observed_at=allocation.observed_at,
    )

    _, snapshot = assemble_local_paper_safety_evidence(_identity(), contract, broker, allocation)

    assert allocation.aggregate_allocated_quantity == Decimal("10")
    assert len(snapshot.account_positions) == 1
    assert snapshot.account_positions[0].quantity == Decimal("10")
    assert {row.portfolio_id: row.quantity for row in snapshot.portfolio_allocations} == {
        "default": Decimal("7"),
        "growth": Decimal("3"),
    }


@pytest.mark.asyncio
async def test_copied_and_forged_producer_snapshots_are_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract = _runtime_contract(tmp_path / "provenance.db")
    allocation = await _allocation_snapshot(
        Path(contract.database_path),
        {"default": 5},
        symbol="AAPL",
        contract=contract,
    )
    broker = _broker_contract_snapshot(
        _runtime_context(tmp_path, monkeypatch, contract),
        symbol="AAPL",
        observed_at=allocation.observed_at,
    )

    for untrusted_broker in (replace(broker), _forge_exact_snapshot(broker)):
        with pytest.raises(ValidationError, match="not producer-owned"):
            assemble_local_paper_safety_evidence(
                _identity(), contract, untrusted_broker, allocation
            )
    for untrusted_allocation in (
        replace(allocation),
        _forge_exact_snapshot(allocation),
    ):
        with pytest.raises(SafetyAllocationSnapshotError, match="producer-owned"):
            assemble_local_paper_safety_evidence(
                _identity(), contract, broker, untrusted_allocation
            )


@pytest.mark.asyncio
async def test_offsetting_local_allocations_fail_reconciliation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract = _runtime_contract(tmp_path / "offsetting.db")
    allocation = await _allocation_snapshot(
        Path(contract.database_path),
        {"default": 10, "hedge": -5},
        symbol="AAPL",
        contract=contract,
    )
    broker = _broker_contract_snapshot(
        _runtime_context(tmp_path, monkeypatch, contract),
        symbol="AAPL",
        observed_at=allocation.observed_at,
    )

    authoritative_contract, snapshot = assemble_local_paper_safety_evidence(
        _identity(), contract, broker, allocation
    )

    assert allocation.has_offsetting_allocations is True
    assert snapshot.account_positions[0].quantity == Decimal("5")
    assert snapshot.reconciliation_status is ReconciliationStatus.FAILED

    evaluated_at = datetime.now(timezone.utc)
    journal = SafetyJournal(tmp_path / "offsetting-safety.db", clock=lambda: evaluated_at)
    journal.initialize(
        execution_domain_scope=_identity().execution_domain_scope,
        account_scope=ACCOUNT_SCOPE,
    )
    coordinator = SafetyRuntimeCoordinator(
        _identity(),
        journal,
        clock=lambda: evaluated_at,
    )
    coordinator.start()
    with pytest.raises(RuntimeAuthorizationBlocked):
        coordinator.authorize(
            "offsetting-reduction",
            RuntimeOrderRequest(
                portfolio_id="default",
                contract=authoritative_contract,
                side=OrderSide.SELL,
                quantity=Decimal("1"),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                order_ref="offsetting-reduction",
            ),
            snapshot,
        )


@pytest.mark.asyncio
async def test_database_path_drift_is_rejected(tmp_path: Path, monkeypatch) -> None:
    expected = _runtime_contract(tmp_path / "expected.db")
    other = _runtime_contract(tmp_path / "other.db")
    allocation = await _allocation_snapshot(
        Path(other.database_path),
        {"default": 4},
        symbol="AAPL",
        contract=other,
    )
    broker = _broker_contract_snapshot(
        _runtime_context(tmp_path, monkeypatch, expected),
        symbol="AAPL",
        observed_at=allocation.observed_at,
    )

    with pytest.raises(TrustedEvidenceAssemblyError, match="ledger path"):
        assemble_local_paper_safety_evidence(_identity(), expected, broker, allocation)


@pytest.mark.asyncio
async def test_database_identity_drift_is_rejected(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "identity.db"
    expected = _runtime_contract(database_path)
    drifted = _runtime_contract(
        database_path,
        execution_mode="backtest",
        state_namespace="backtest",
    )
    allocation = await _allocation_snapshot(
        database_path,
        {"default": 4},
        symbol="AAPL",
        contract=drifted,
    )
    broker = _broker_contract_snapshot(
        _runtime_context(tmp_path, monkeypatch, expected),
        symbol="AAPL",
        observed_at=allocation.observed_at,
    )

    assert allocation.database_path == expected.database_path
    assert allocation.database_identity != expected.database_identity
    with pytest.raises(TrustedEvidenceAssemblyError, match="ledger identity"):
        assemble_local_paper_safety_evidence(_identity(), expected, broker, allocation)


@pytest.mark.asyncio
async def test_symbol_drift_is_rejected(tmp_path: Path, monkeypatch) -> None:
    contract = _runtime_contract(tmp_path / "symbol.db")
    allocation = await _allocation_snapshot(
        Path(contract.database_path),
        {"default": 4},
        symbol="AAPL",
        contract=contract,
    )
    broker = _broker_contract_snapshot(
        _runtime_context(tmp_path, monkeypatch, contract),
        symbol="MSFT",
        observed_at=allocation.observed_at,
    )

    with pytest.raises(TrustedEvidenceAssemblyError, match="different symbols"):
        assemble_local_paper_safety_evidence(_identity(), contract, broker, allocation)


@pytest.mark.asyncio
async def test_collection_time_drift_is_rejected(tmp_path: Path, monkeypatch) -> None:
    contract = _runtime_contract(tmp_path / "time.db")
    allocation = await _allocation_snapshot(
        Path(contract.database_path),
        {"default": 4},
        symbol="AAPL",
        contract=contract,
    )
    broker = _broker_contract_snapshot(
        _runtime_context(tmp_path, monkeypatch, contract),
        symbol="AAPL",
        observed_at=allocation.observed_at + timedelta(seconds=31),
    )

    with pytest.raises(TrustedEvidenceAssemblyError, match="collection window"):
        assemble_local_paper_safety_evidence(_identity(), contract, broker, allocation)
