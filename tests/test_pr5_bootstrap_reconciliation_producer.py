from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from robo_trader.config import RuntimeContract
from robo_trader.reconciliation import bootstrap_producer as producer_module
from robo_trader.reconciliation.bootstrap_producer import (
    BOOTSTRAP_RECONCILIATION_STATUS,
    BootstrapLedgerEvidence,
    BootstrapReconciliationBlocked,
    UnsignedBootstrapReconciliation,
    produce_bootstrap_reconciliation,
)
from robo_trader.reconciliation.domain import (
    BrokerCollectionEvidence,
    BrokerCollectionKind,
    BrokerEvidenceCompleteness,
    BrokerOrderCollection,
    BrokerOrderSide,
    ExecutionDomainScope,
    NormalizedBrokerAccount,
    NormalizedBrokerExecution,
    NormalizedBrokerOrder,
    NormalizedBrokerPosition,
    NormalizedBrokerSnapshot,
)
from robo_trader.reconciliation.policy import (
    DifferenceKind,
    DifferenceMateriality,
    ReconciliationCoverage,
    ReconciliationDifference,
    ReconciliationStatus,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
ACCOUNT_SCOPE = "acct_v1_" + "0123456789abcdef" * 4
OTHER_ACCOUNT_SCOPE = "acct_v1_" + "fedcba9876543210" * 4


def _runtime(database: Path) -> RuntimeContract:
    return RuntimeContract(
        environment="dev",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(database),
        account_alias="***1234",
        account_type="paper",
        model_artifact_set="test-models",
        build_id="test-build",
        state_namespace="paper",
        safety_account_scope=ACCOUNT_SCOPE,
        safety_execution_domain_scope="paper-simulator-v1",
        safety_journal_path=str(database.with_name("safety-journal.db")),
    )


def _coverage(**overrides: bool) -> ReconciliationCoverage:
    values = {
        "broker_account": True,
        "broker_positions": True,
        "broker_open_orders": True,
        "broker_completed_orders": True,
        "broker_executions": True,
        "broker_commissions": True,
        "ledger_positions": True,
        "ledger_orders": True,
        "ledger_executions": True,
        "ledger_cash": True,
    }
    values.update(overrides)
    return ReconciliationCoverage(**values)


def _collection_evidence(
    kind: BrokerCollectionKind,
    count: int,
    observed_at: datetime,
    *,
    scope: str,
) -> BrokerCollectionEvidence:
    digest = hashlib.sha256(
        f"{scope}:{kind.value}:{count}:{observed_at.isoformat()}".encode()
    ).hexdigest()
    return BrokerCollectionEvidence(
        account_scope=scope,
        collection=kind,
        evidence_id="broker-collection-v1-" + digest,
        result_count=count,
        observed_at=observed_at,
    )


def _position(*, scope: str = ACCOUNT_SCOPE) -> NormalizedBrokerPosition:
    return NormalizedBrokerPosition(
        account_scope=scope,
        con_id=265598,
        symbol="AAPL",
        currency="USD",
        signed_quantity=Decimal("2"),
        average_cost=Decimal("210.125"),
        observed_at=NOW - timedelta(seconds=1),
    )


def _order(
    *,
    scope: str = ACCOUNT_SCOPE,
    collection: BrokerOrderCollection = BrokerOrderCollection.OPEN,
) -> NormalizedBrokerOrder:
    completed = collection is BrokerOrderCollection.COMPLETED
    return NormalizedBrokerOrder(
        account_scope=scope,
        collection=collection,
        broker_order_id=7,
        client_id=4,
        con_id=265598,
        symbol="AAPL",
        side=BrokerOrderSide.SELL,
        total_quantity=Decimal("2"),
        filled_quantity=Decimal("2") if completed else Decimal("1"),
        remaining_quantity=Decimal("0") if completed else Decimal("1"),
        status="Filled" if completed else "Submitted",
        observed_at=NOW - timedelta(seconds=1),
        permanent_id=70,
    )


def _execution(*, scope: str = ACCOUNT_SCOPE) -> NormalizedBrokerExecution:
    return NormalizedBrokerExecution(
        account_scope=scope,
        execution_id="0001.abc",
        con_id=265598,
        symbol="AAPL",
        side=BrokerOrderSide.SELL,
        quantity=Decimal("2"),
        price=Decimal("215.50"),
        executed_at=NOW - timedelta(seconds=1),
        broker_order_id=7,
        permanent_id=70,
        commission=Decimal("1.23"),
        commission_currency="USD",
    )


def _snapshot(
    *,
    scope: str = ACCOUNT_SCOPE,
    positions: tuple[NormalizedBrokerPosition, ...] = (),
    orders: tuple[NormalizedBrokerOrder, ...] = (),
    executions: tuple[NormalizedBrokerExecution, ...] = (),
    complete: bool = True,
    retrieved_at: datetime = NOW - timedelta(seconds=1),
) -> NormalizedBrokerSnapshot:
    completeness = BrokerEvidenceCompleteness(
        account=True,
        positions=True,
        open_orders=True,
        completed_orders=complete,
        executions=True,
        commissions=True,
    )
    counts = {
        BrokerCollectionKind.POSITIONS: len(positions),
        BrokerCollectionKind.OPEN_ORDERS: sum(
            order.collection is BrokerOrderCollection.OPEN for order in orders
        ),
        BrokerCollectionKind.COMPLETED_ORDERS: sum(
            order.collection is BrokerOrderCollection.COMPLETED for order in orders
        ),
        BrokerCollectionKind.EXECUTIONS: len(executions),
        BrokerCollectionKind.COMMISSIONS: len(executions),
    }
    evidence = tuple(
        _collection_evidence(kind, counts[kind], retrieved_at, scope=scope)
        for kind in BrokerCollectionKind
        if complete or kind is not BrokerCollectionKind.COMPLETED_ORDERS
    )
    return NormalizedBrokerSnapshot(
        account=NormalizedBrokerAccount(
            account_scope=scope,
            account_alias="***1234",
            account_type="paper",
            base_currency="USD",
            total_cash=Decimal("25000.00"),
            buying_power=Decimal("100000.00"),
            observed_at=retrieved_at,
        ),
        observed_from=retrieved_at - timedelta(seconds=1),
        observed_through=retrieved_at,
        retrieved_at=retrieved_at,
        completeness=completeness,
        collection_evidence=evidence,
        positions=positions,
        orders=orders,
        executions=executions,
    )


def _ledger_evidence(
    database: Path,
    runtime: RuntimeContract,
    *,
    coverage: ReconciliationCoverage | None = None,
    differences: tuple[ReconciliationDifference, ...] = (),
    observed_at: datetime = NOW - timedelta(seconds=1),
    portfolio_ids: tuple[str, ...] = ("default",),
    known_portfolio_ids: tuple[str, ...] | None = None,
    covered_portfolio_ids: tuple[str, ...] | None = None,
) -> BootstrapLedgerEvidence:
    metadata = os.lstat(database)
    return BootstrapLedgerEvidence(
        runtime_fingerprint=runtime.fingerprint,
        execution_domain_scope=ExecutionDomainScope.PAPER_SIMULATOR,
        account_scope=ACCOUNT_SCOPE,
        database_path=str(database),
        database_identity=runtime.database_identity,
        database_device=metadata.st_dev,
        database_inode=metadata.st_ino,
        database_size=metadata.st_size,
        database_mtime_ns=metadata.st_mtime_ns,
        database_ctime_ns=metadata.st_ctime_ns,
        portfolio_ids=portfolio_ids,
        known_portfolio_ids=known_portfolio_ids or portfolio_ids,
        active_portfolio_ids=portfolio_ids,
        covered_portfolio_ids=covered_portfolio_ids or portfolio_ids,
        local_simulator_positions_count=2,
        legacy_snapshot_hash="ab" * 32,
        observed_at=observed_at,
        coverage=coverage or _coverage(),
        differences=differences,
    )


class Receiver:
    def __init__(self) -> None:
        self.results: list[UnsignedBootstrapReconciliation] = []

    def receive_unsigned_bootstrap_reconciliation(
        self,
        result: UnsignedBootstrapReconciliation,
    ) -> UnsignedBootstrapReconciliation:
        self.results.append(result)
        return result


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.db"
    path.write_bytes(b"immutable-ledger-evidence")
    return path


def _produce(
    database: Path,
    *,
    snapshot: NormalizedBrokerSnapshot | None = None,
    evidence: BootstrapLedgerEvidence | None = None,
    runtime: RuntimeContract | None = None,
    receiver: Receiver | None = None,
    now: datetime = NOW,
) -> tuple[UnsignedBootstrapReconciliation, Receiver]:
    contract = runtime or _runtime(database)
    sink = receiver or Receiver()
    result = produce_bootstrap_reconciliation(
        snapshot or _snapshot(),
        evidence or _ledger_evidence(database, contract),
        contract,
        sink,
        now=now,
    )
    return result, sink


def test_clean_stage_binds_all_evidence_and_is_non_authorizing(database: Path) -> None:
    snapshot = _snapshot()
    result, receiver = _produce(database, snapshot=snapshot)

    assert receiver.results == [result]
    assert result.status == BOOTSTRAP_RECONCILIATION_STATUS
    assert result.reconciliation_status is ReconciliationStatus.PASSED
    assert result.broker_snapshot_id == snapshot.snapshot_id
    assert (
        result.broker_snapshot_hash
        == hashlib.sha256(snapshot.canonical_payload().encode()).hexdigest()
    )
    assert result.legacy_snapshot_hash == "ab" * 32
    assert result.portfolio_ids == ("default",)
    assert len(result.broker_collection_evidence_ids) == len(BrokerCollectionKind)
    assert result.local_simulator_positions_count == 2
    assert result.broker_positions_count == 0
    assert result.broker_open_orders_count == 0
    assert result.mutated_state is False
    assert result.authorizes_startup is False
    assert result.execution_domain_scope == "paper-simulator-v1"
    payload = json.loads(result.canonical_payload())
    assert payload["snapshot_id"] == result.snapshot_id
    assert payload["authorizes_startup"] is False
    assert payload["mutated_state"] is False


def test_local_simulator_positions_are_not_compared_to_zero_ibkr_exposure(
    database: Path,
) -> None:
    runtime = _runtime(database)
    evidence = replace(
        _ledger_evidence(database, runtime),
        local_simulator_positions_count=17,
    )

    result, _ = _produce(database, runtime=runtime, evidence=evidence)

    assert result.reconciliation_status is ReconciliationStatus.PASSED
    assert result.local_simulator_positions_count == 17
    assert result.broker_positions_count == 0


def test_completed_broker_history_is_allowed_but_remains_diagnostic(database: Path) -> None:
    snapshot = _snapshot(
        orders=(_order(collection=BrokerOrderCollection.COMPLETED),),
        executions=(_execution(),),
    )

    result, _ = _produce(database, snapshot=snapshot)

    assert result.reconciliation_status is ReconciliationStatus.PASSED
    assert result.broker_positions_count == 0
    assert result.broker_open_orders_count == 0


@pytest.mark.parametrize("difference_kind", [DifferenceKind.UNKNOWN, DifferenceKind.CASH_MISMATCH])
def test_unknown_or_material_local_difference_blocks_without_receiver_call(
    database: Path,
    difference_kind: DifferenceKind,
) -> None:
    runtime = _runtime(database)
    materiality = (
        DifferenceMateriality.UNKNOWN
        if difference_kind is DifferenceKind.UNKNOWN
        else DifferenceMateriality.MATERIAL
    )
    difference = ReconciliationDifference(
        kind=difference_kind,
        materiality=materiality,
        reason_code="LOCAL_STATE_NOT_RECONCILED",
        subject="local_ledger",
    )
    evidence = _ledger_evidence(database, runtime, differences=(difference,))
    receiver = Receiver()

    with pytest.raises(BootstrapReconciliationBlocked, match="unknown, or material"):
        _produce(database, runtime=runtime, evidence=evidence, receiver=receiver)

    assert receiver.results == []


@pytest.mark.parametrize("exposure", ["position", "open_order"])
def test_ibkr_exposure_blocks_instead_of_becoming_local_equality_evidence(
    database: Path,
    exposure: str,
) -> None:
    snapshot = (
        _snapshot(positions=(_position(),))
        if exposure == "position"
        else _snapshot(orders=(_order(),))
    )
    receiver = Receiver()

    with pytest.raises(BootstrapReconciliationBlocked):
        _produce(database, snapshot=snapshot, receiver=receiver)

    assert receiver.results == []


def test_stale_or_incomplete_broker_snapshot_blocks(database: Path) -> None:
    for snapshot in (
        _snapshot(retrieved_at=NOW - timedelta(minutes=5)),
        _snapshot(complete=False),
    ):
        receiver = Receiver()
        with pytest.raises(BootstrapReconciliationBlocked):
            _produce(database, snapshot=snapshot, receiver=receiver)
        assert receiver.results == []


def test_wrong_account_blocks_without_leaking_or_delivering(database: Path) -> None:
    receiver = Receiver()

    with pytest.raises(BootstrapReconciliationBlocked, match="account identity") as raised:
        _produce(database, snapshot=_snapshot(scope=OTHER_ACCOUNT_SCOPE), receiver=receiver)

    assert receiver.results == []
    assert "DU" not in str(raised.value)


def test_incomplete_local_coverage_and_portfolio_coverage_block(database: Path) -> None:
    runtime = _runtime(database)
    cases = (
        _ledger_evidence(database, runtime, coverage=_coverage(ledger_cash=False)),
        _ledger_evidence(
            database,
            runtime,
            portfolio_ids=("default",),
            known_portfolio_ids=("default", "secondary"),
            covered_portfolio_ids=("default",),
        ),
    )
    for evidence in cases:
        receiver = Receiver()
        with pytest.raises(BootstrapReconciliationBlocked, match="coverage is incomplete"):
            _produce(database, runtime=runtime, evidence=evidence, receiver=receiver)
        assert receiver.results == []


def test_stale_local_evidence_blocks_without_delivery(database: Path) -> None:
    runtime = _runtime(database)
    evidence = _ledger_evidence(
        database,
        runtime,
        observed_at=NOW - timedelta(minutes=5),
    )
    receiver = Receiver()

    with pytest.raises(BootstrapReconciliationBlocked, match="future or stale"):
        _produce(database, runtime=runtime, evidence=evidence, receiver=receiver)

    assert receiver.results == []


def test_database_content_or_inode_drift_blocks_without_delivery(
    database: Path,
    tmp_path: Path,
) -> None:
    runtime = _runtime(database)
    evidence = _ledger_evidence(database, runtime)
    receiver = Receiver()
    database.write_bytes(b"changed-ledger-state")

    with pytest.raises(BootstrapReconciliationBlocked, match="database changed"):
        _produce(database, runtime=runtime, evidence=evidence, receiver=receiver)
    assert receiver.results == []

    database.unlink()
    replacement = tmp_path / "replacement.db"
    replacement.write_bytes(b"immutable-ledger-evidence")
    replacement.replace(database)
    with pytest.raises(BootstrapReconciliationBlocked, match="database changed"):
        _produce(database, runtime=runtime, evidence=evidence, receiver=receiver)
    assert receiver.results == []


def test_database_drift_during_policy_evaluation_blocks_before_delivery(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(database)
    evidence = _ledger_evidence(database, runtime)
    receiver = Receiver()
    real_evaluate = producer_module.evaluate_paper_simulator_reconciliation

    def evaluate_then_drift(*args, **kwargs):
        verdict = real_evaluate(*args, **kwargs)
        database.write_bytes(b"drifted-during-policy-evaluation")
        return verdict

    monkeypatch.setattr(
        producer_module,
        "evaluate_paper_simulator_reconciliation",
        evaluate_then_drift,
    )

    with pytest.raises(BootstrapReconciliationBlocked, match="database changed"):
        _produce(database, runtime=runtime, evidence=evidence, receiver=receiver)

    assert receiver.results == []


def test_runtime_binding_drift_blocks_without_delivery(database: Path) -> None:
    runtime = _runtime(database)
    evidence = _ledger_evidence(database, runtime)
    changed = replace(runtime, build_id="different-build")
    receiver = Receiver()

    with pytest.raises(BootstrapReconciliationBlocked, match="validated runtime"):
        _produce(database, runtime=changed, evidence=evidence, receiver=receiver)

    assert receiver.results == []


def test_receiver_is_narrow_and_no_signer_key_or_artifact_path_is_accepted(
    database: Path,
) -> None:
    parameters = inspect.signature(produce_bootstrap_reconciliation).parameters
    assert set(parameters) == {
        "snapshot",
        "ledger_evidence",
        "runtime_contract",
        "receiver",
        "now",
        "max_age_seconds",
    }
    assert all(
        fragment not in name
        for name in parameters
        for fragment in ("sign", "key", "artifact", "json", "output_path")
    )

    with pytest.raises(BootstrapReconciliationBlocked, match="receiver capability"):
        produce_bootstrap_reconciliation(
            _snapshot(),
            _ledger_evidence(database, _runtime(database)),
            _runtime(database),
            object(),  # type: ignore[arg-type]
            now=NOW,
        )


def test_unsigned_result_is_immutable_and_canonical(database: Path) -> None:
    first, _ = _produce(database)
    second, _ = _produce(database)

    assert first.canonical_payload() == second.canonical_payload()
    assert first.snapshot_id == second.snapshot_id
    with pytest.raises((FrozenInstanceError, TypeError)):
        first.authorizes_startup = True  # type: ignore[misc]
