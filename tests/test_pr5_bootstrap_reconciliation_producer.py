from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import sqlite3
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from robo_trader.config import RuntimeContract
from robo_trader.financial_state_bootstrap import inspect_legacy_state
from robo_trader.reconciliation import bootstrap_producer as producer_module
from robo_trader.reconciliation.bootstrap_producer import (
    BOOTSTRAP_RECONCILIATION_STATUS,
    BootstrapReconciliationBlocked,
    UnsignedBootstrapReconciliation,
    _issue_test_producer_capability,
    _produce_bootstrap_reconciliation_for_test,
    assert_and_consume_producer_owned_bootstrap_reconciliation,
)
from robo_trader.reconciliation.domain import (
    BrokerCollectionEvidence,
    BrokerCollectionKind,
    BrokerEvidenceCompleteness,
    BrokerOrderCollection,
    BrokerOrderSide,
    NormalizedBrokerAccount,
    NormalizedBrokerExecution,
    NormalizedBrokerOrder,
    NormalizedBrokerPosition,
    NormalizedBrokerSnapshot,
)
from robo_trader.reconciliation.policy import ReconciliationStatus

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
ACCOUNT_SCOPE = "acct_v1_" + "0123456789abcdef" * 4
OTHER_ACCOUNT_SCOPE = "acct_v1_" + "fedcba9876543210" * 4


def _legacy_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                market_price REAL,
                timestamp DATETIME
            );
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                notional REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                pnl REAL DEFAULT NULL,
                timestamp DATETIME
            );
            CREATE TABLE account (
                portfolio_id TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME
            );
            CREATE TABLE equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL,
                date TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL DEFAULT 0,
                positions_value REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME
            );
            INSERT INTO portfolios VALUES ('default', 'Default');
            INSERT INTO account VALUES
              ('default', 1000, 1200, 0, 0, 200, '2026-07-28T11:59:00+00:00');
            INSERT INTO positions(portfolio_id,symbol,quantity,avg_cost,market_price,timestamp)
            VALUES
              ('default','AAPL',2,100,110,'2026-07-28T11:59:00+00:00'),
              ('default','MSFT',3,200,210,'2026-07-28T11:59:00+00:00');
            INSERT INTO trades(portfolio_id,symbol,side,quantity,price,notional,timestamp)
            VALUES
              ('default','AAPL','BUY',2,100,200,'2026-07-28T11:58:00+00:00');
            INSERT INTO equity_history(
                portfolio_id,date,equity,cash,positions_value,realized_pnl,unrealized_pnl,timestamp
            ) VALUES
              ('default','2026-07-28',1200,1000,200,0,200,
               '2026-07-28T11:59:00+00:00');
        """)


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
            total_cash=Decimal("25000"),
            buying_power=Decimal("100000"),
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


@dataclass(frozen=True, slots=True)
class FakeVerifiedBrokerEnvelope:
    snapshot: NormalizedBrokerSnapshot
    snapshot_id: str
    snapshot_hash: str
    artifact_hash: str
    runtime_fingerprint: str
    account_scope: str
    receipt_id: str
    public_key_fingerprint: str
    issued_at: datetime
    expires_at: datetime


def _envelope(
    snapshot: NormalizedBrokerSnapshot, runtime: RuntimeContract
) -> FakeVerifiedBrokerEnvelope:
    artifact_hash = hashlib.sha256(snapshot.canonical_payload().encode()).hexdigest()
    return FakeVerifiedBrokerEnvelope(
        snapshot=snapshot,
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=artifact_hash,
        artifact_hash="56" * 32,
        runtime_fingerprint=runtime.fingerprint,
        account_scope=ACCOUNT_SCOPE,
        receipt_id="bevr-v2-" + "12" * 32,
        public_key_fingerprint="34" * 32,
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )


class FakeBrokerVerifier:
    def __init__(self, *envelopes: FakeVerifiedBrokerEnvelope) -> None:
        self._owned = {id(value): value for value in envelopes}

    def consume(self, value: object) -> FakeVerifiedBrokerEnvelope:
        owned = self._owned.pop(id(value), None)
        if owned is not value:
            raise BootstrapReconciliationBlocked("test envelope is not verifier-owned")
        return cast(FakeVerifiedBrokerEnvelope, value)


class ClaimingReceiver:
    def __init__(self, *, before_claim=None) -> None:
        self.results: list[UnsignedBootstrapReconciliation] = []
        self.before_claim = before_claim

    def receive_unsigned_bootstrap_reconciliation(
        self,
        result: UnsignedBootstrapReconciliation,
    ) -> UnsignedBootstrapReconciliation:
        if self.before_claim is not None:
            self.before_claim()
        claimed = assert_and_consume_producer_owned_bootstrap_reconciliation(result)
        self.results.append(claimed)
        return claimed


class BypassReceiver:
    def __init__(self) -> None:
        self.results: list[UnsignedBootstrapReconciliation] = []

    def receive_unsigned_bootstrap_reconciliation(
        self,
        result: UnsignedBootstrapReconciliation,
    ) -> str:
        self.results.append(result)
        return "bypassed"


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.db"
    _legacy_database(path)
    return path


def _produce(
    database: Path,
    *,
    snapshot: NormalizedBrokerSnapshot | None = None,
    envelope: FakeVerifiedBrokerEnvelope | None = None,
    runtime: RuntimeContract | None = None,
    receiver: object | None = None,
    clock: Callable[[], datetime] | None = None,
    verifier: FakeBrokerVerifier | None = None,
) -> tuple[
    producer_module.BootstrapReconciliationDelivery[object],
    Any,
    FakeVerifiedBrokerEnvelope,
]:
    contract = runtime or _runtime(database)
    verified = envelope or _envelope(snapshot or _snapshot(), contract)
    authority = verifier or FakeBrokerVerifier(verified)
    sink = receiver or ClaimingReceiver()
    capability = _issue_test_producer_capability(
        clock=clock or (lambda: NOW),
        broker_consumer=cast(
            Callable[[object], producer_module.VerifiedBrokerEvidenceEnvelope],
            authority.consume,
        ),
    )
    delivery: producer_module.BootstrapReconciliationDelivery[object] = (
        _produce_bootstrap_reconciliation_for_test(
            verified,
            contract,
            sink,  # type: ignore[arg-type]
            capability,
        )
    )
    return delivery, sink, verified


def test_clean_stage_collects_ledger_and_binds_non_authorizing_result(database: Path) -> None:
    expected_legacy_hash = str(inspect_legacy_state(database)["snapshot_hash"])

    delivery, receiver, envelope = _produce(database)

    assert isinstance(receiver, ClaimingReceiver)
    result = receiver.results[0]
    assert delivery.receiver_result is result
    assert delivery.local_position_identities == (("default", "AAPL"), ("default", "MSFT"))
    assert result.status == BOOTSTRAP_RECONCILIATION_STATUS
    assert result.reconciliation_status is ReconciliationStatus.PASSED
    assert result.broker_snapshot_id == envelope.snapshot_id
    assert result.broker_snapshot_hash == envelope.snapshot_hash
    assert result.broker_artifact_hash == envelope.artifact_hash
    assert result.broker_artifact_hash != result.broker_snapshot_hash
    assert result.legacy_snapshot_hash == expected_legacy_hash
    assert result.portfolio_ids == ("default",)
    assert result.local_simulator_positions_count == 2
    assert result.broker_positions_count == 0
    assert result.broker_open_orders_count == 0
    assert result.mutated_state is False
    assert result.authorizes_startup is False
    assert result.execution_domain_scope == "paper-simulator-v1"


def test_public_interface_accepts_no_snapshot_ledger_clock_or_freshness_authority() -> None:
    parameters = inspect.signature(producer_module.produce_bootstrap_reconciliation).parameters
    assert set(parameters) == {"verified_broker_evidence", "runtime_contract", "receiver"}
    assert all(
        forbidden not in parameters
        for forbidden in ("snapshot", "ledger_evidence", "now", "clock", "max_age_seconds")
    )


def test_local_simulator_positions_are_not_compared_to_zero_ibkr(database: Path) -> None:
    delivery, receiver, _ = _produce(database)
    result = receiver.results[0]

    assert result.reconciliation_status is ReconciliationStatus.PASSED
    assert result.local_simulator_positions_count == 2
    assert result.broker_positions_count == 0
    assert delivery.local_position_identities == result.local_position_identities


def test_completed_broker_history_remains_diagnostic_not_simulator_equality(
    database: Path,
) -> None:
    snapshot = _snapshot(
        orders=(_order(collection=BrokerOrderCollection.COMPLETED),),
        executions=(_execution(),),
    )

    _, receiver, _ = _produce(database, snapshot=snapshot)

    assert receiver.results[0].reconciliation_status is ReconciliationStatus.PASSED


@pytest.mark.parametrize(
    "snapshot",
    [
        _snapshot(positions=(_position(),)),
        _snapshot(orders=(_order(),)),
        _snapshot(complete=False),
        _snapshot(retrieved_at=NOW - timedelta(minutes=5)),
        _snapshot(scope=OTHER_ACCOUNT_SCOPE),
    ],
)
def test_exposure_incomplete_stale_or_wrong_account_blocks(
    database: Path,
    snapshot: NormalizedBrokerSnapshot,
) -> None:
    receiver = ClaimingReceiver()

    with pytest.raises(BootstrapReconciliationBlocked):
        _produce(database, snapshot=snapshot, receiver=receiver)

    assert receiver.results == []


def test_stale_internal_clock_blocks_without_receiver_call(database: Path) -> None:
    clock_values = iter((NOW, NOW + timedelta(seconds=31)))
    receiver = ClaimingReceiver()

    with pytest.raises(BootstrapReconciliationBlocked, match="became stale"):
        _produce(database, receiver=receiver, clock=lambda: next(clock_values))

    assert receiver.results == []


def test_forged_or_replayed_verified_envelope_blocks(database: Path) -> None:
    runtime = _runtime(database)
    envelope = _envelope(_snapshot(), runtime)
    verifier = FakeBrokerVerifier(envelope)
    _produce(database, runtime=runtime, envelope=envelope, verifier=verifier)

    with pytest.raises(BootstrapReconciliationBlocked):
        _produce(database, runtime=runtime, envelope=envelope, verifier=verifier)

    forged = replace(envelope)
    with pytest.raises(BootstrapReconciliationBlocked):
        _produce(database, runtime=runtime, envelope=forged, verifier=verifier)


def test_public_producer_rejects_structurally_valid_caller_envelope(database: Path) -> None:
    runtime = _runtime(database)
    forged = _envelope(_snapshot(), runtime)
    receiver = ClaimingReceiver()

    with pytest.raises(BootstrapReconciliationBlocked, match="not verifier-owned"):
        producer_module.produce_bootstrap_reconciliation(forged, runtime, receiver)

    assert receiver.results == []


def test_forged_typed_ledger_evidence_is_rejected(database: Path, monkeypatch) -> None:
    forged = object.__new__(producer_module._CollectedLedgerEvidence)

    @contextmanager
    def forged_collector(*args, **kwargs):
        del args, kwargs
        yield forged

    monkeypatch.setattr(producer_module, "_collect_wal_visible_ledger", forged_collector)
    receiver = ClaimingReceiver()

    with pytest.raises(BootstrapReconciliationBlocked, match="collector-owned"):
        _produce(database, receiver=receiver)

    assert receiver.results == []


def test_receiver_must_independently_claim_ownership(database: Path) -> None:
    receiver = BypassReceiver()

    with pytest.raises(BootstrapReconciliationBlocked, match="independently claim"):
        _produce(database, receiver=receiver)

    assert len(receiver.results) == 1
    with pytest.raises(BootstrapReconciliationBlocked, match="not producer-owned"):
        assert_and_consume_producer_owned_bootstrap_reconciliation(receiver.results[0])


def test_result_direct_construction_copy_and_replay_fail(database: Path) -> None:
    _, receiver, _ = _produce(database)
    assert isinstance(receiver, ClaimingReceiver)
    result = receiver.results[0]
    copied = replace(result)

    with pytest.raises(BootstrapReconciliationBlocked, match="not producer-owned"):
        assert_and_consume_producer_owned_bootstrap_reconciliation(result)
    with pytest.raises(BootstrapReconciliationBlocked, match="not producer-owned"):
        assert_and_consume_producer_owned_bootstrap_reconciliation(copied)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy(result)
    with pytest.raises(TypeError, match="cannot be copied"):
        deepcopy(result)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(result)
    with pytest.raises((TypeError, BootstrapReconciliationBlocked)):
        UnsignedBootstrapReconciliation()  # type: ignore[call-arg]


def test_wal_visible_commit_is_included_in_snapshot_hash(database: Path) -> None:
    writer = sqlite3.connect(database)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO positions(portfolio_id,symbol,quantity,avg_cost,market_price,timestamp) "
            "VALUES ('default','NVDA',4,300,310,'2026-07-28T11:59:30+00:00')"
        )
        writer.commit()
        wal_path = Path(f"{database}-wal")
        assert wal_path.exists() and wal_path.stat().st_size > 0

        delivery, receiver, _ = _produce(database)

        result = receiver.results[0]
        assert ("default", "NVDA") in delivery.local_position_identities
        assert result.legacy_snapshot_hash == inspect_legacy_state(database)["snapshot_hash"]
    finally:
        writer.close()


def test_wal_only_commit_after_snapshot_before_delivery_blocks(database: Path) -> None:
    writer = sqlite3.connect(database)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")

    def commit_during_receiver() -> None:
        writer.execute(
            "INSERT INTO trades(portfolio_id,symbol,side,quantity,price,notional,timestamp) "
            "VALUES ('default','MSFT','BUY',1,210,210,'2026-07-28T12:00:00+00:00')"
        )
        writer.commit()

    clock_calls = 0

    def clock_with_commit() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 2:
            commit_during_receiver()
        return NOW

    receiver = ClaimingReceiver()
    try:
        with pytest.raises(BootstrapReconciliationBlocked, match="ledger WAL changed"):
            _produce(database, receiver=receiver, clock=clock_with_commit)
        assert receiver.results == []
    finally:
        writer.close()


def test_malformed_or_partial_database_blocks_without_delivery(tmp_path: Path) -> None:
    database = tmp_path / "partial.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE account(portfolio_id TEXT)")
    receiver = ClaimingReceiver()

    with pytest.raises(BootstrapReconciliationBlocked):
        _produce(database, receiver=receiver)

    assert receiver.results == []
