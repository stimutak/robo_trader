from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import aiosqlite
import pytest

from robo_trader import bootstrap_evidence_receivers as receiver_module
from robo_trader.bootstrap_evidence_receivers import (
    ProtectiveMarkBundleIdentity,
    VerifiedBrokerEvidenceEnvelope,
)
from robo_trader.config import RuntimeContract
from robo_trader.reconciliation import runtime_evidence as runtime_evidence_module
from robo_trader.reconciliation.domain import (
    BrokerCollectionEvidence,
    BrokerCollectionKind,
    BrokerEvidenceCompleteness,
    NormalizedBrokerAccount,
    NormalizedBrokerSnapshot,
    canonical_timestamp,
    fingerprint,
)
from robo_trader.reconciliation.persistence import (
    OperatorResolutionKind,
    ReconciliationPersistence,
    ReconciliationPersistenceError,
)
from robo_trader.reconciliation.policy import (
    DifferenceKind,
    DifferenceMateriality,
    ExpectedTimingLagProof,
    ReconciliationCoverage,
    ReconciliationDifference,
    ReconciliationStatus,
    evaluate_paper_simulator_reconciliation,
)
from robo_trader.reconciliation.runtime_evidence import (
    RuntimeReconciliationEvidenceError,
    VerifiedRuntimeReconciliationEvidence,
    assert_and_consume_verified_runtime_reconciliation_evidence,
    bind_verified_runtime_reconciliation_evidence,
)
from robo_trader.reconciliation.service import (
    ReconciliationComparison,
    ReconciliationService,
    ReconciliationServiceBlocked,
    ReconciliationServiceState,
    ReconciliationTrigger,
)
from robo_trader.reconciliation_migrations import RECONCILIATION_COMPONENT

NOW = datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc)
ACCOUNT_SCOPE = "acct_v1_" + "0123456789abcdef" * 4


def _collection_evidence(
    collection: BrokerCollectionKind,
    observed_at: datetime,
) -> BrokerCollectionEvidence:
    digest = hashlib.sha256(f"{collection.value}:{observed_at.isoformat()}".encode()).hexdigest()
    return BrokerCollectionEvidence(
        account_scope=ACCOUNT_SCOPE,
        collection=collection,
        evidence_id=f"broker-collection-v1-{digest}",
        result_count=0,
        observed_at=observed_at,
    )


def _snapshot(
    *,
    retrieved_at: datetime = NOW - timedelta(seconds=1),
    complete: bool = True,
) -> NormalizedBrokerSnapshot:
    completeness = BrokerEvidenceCompleteness(
        account=True,
        positions=True,
        open_orders=True,
        completed_orders=True,
        executions=True,
        commissions=complete,
    )
    evidence = tuple(
        _collection_evidence(kind, retrieved_at)
        for kind in BrokerCollectionKind
        if complete or kind is not BrokerCollectionKind.COMMISSIONS
    )
    return NormalizedBrokerSnapshot(
        account=NormalizedBrokerAccount(
            account_scope=ACCOUNT_SCOPE,
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
    )


def _coverage(**changes: bool) -> ReconciliationCoverage:
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
    values.update(changes)
    return ReconciliationCoverage(**values)


@pytest.fixture(autouse=True)
def _accept_registered_test_runtime_context(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_evidence_module,
        "assert_validated_runtime_safety_context",
        lambda context: context,
    )


def _registered_evidence(
    database_path,
    *,
    snapshot: NormalizedBrokerSnapshot | None = None,
    expires_at: datetime = NOW + timedelta(seconds=29),
) -> VerifiedRuntimeReconciliationEvidence:
    path = database_path.resolve()
    path.touch(exist_ok=True)
    metadata = os.lstat(path)
    broker_snapshot = snapshot or _snapshot()
    contract = RuntimeContract(
        environment="development",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(path),
        account_alias="***1234",
        account_type="paper",
        model_artifact_set="test-models",
        build_id="test-build",
        state_namespace="paper",
        safety_account_scope=ACCOUNT_SCOPE,
        safety_execution_domain_scope="paper-simulator-v1",
    )
    context = SimpleNamespace(runtime_contract=contract, account_alias="***1234")
    snapshot_hash = hashlib.sha256(broker_snapshot.canonical_payload().encode("utf-8")).hexdigest()
    return runtime_evidence_module._register(
        VerifiedRuntimeReconciliationEvidence(
            snapshot=broker_snapshot,
            snapshot_id=broker_snapshot.snapshot_id,
            snapshot_hash=snapshot_hash,
            bundle_id="bootstrap-evidence-bundle-v1-" + "a1" * 32,
            runtime_fingerprint=contract.fingerprint,
            account_scope=ACCOUNT_SCOPE,
            account_alias="***1234",
            database_path=str(path),
            database_identity=contract.database_identity,
            database_device=metadata.st_dev,
            database_inode=metadata.st_ino,
            broker_artifact_hash="b2" * 32,
            broker_receipt_id="receipt-v1-" + "c3" * 32,
            broker_public_key_fingerprint="d4" * 32,
            issued_at=NOW - timedelta(microseconds=1),
            expires_at=expires_at,
            _runtime_context=context,
            _marker=runtime_evidence_module._CAPABILITY_MARKER,
        )
    )


class _EvidenceSource:
    def __init__(self, database_path, snapshot: NormalizedBrokerSnapshot) -> None:
        self.database_path = database_path
        self.snapshot = snapshot
        self.calls = 0
        self.closed = False
        self.error: Exception | None = None
        self.close_error: Exception | None = None
        self.close_started = None
        self.close_release = None
        self.expires_at = NOW + timedelta(seconds=29)
        self.prepared_evidence: object | None = None

    async def collect_verified_evidence(
        self, *, max_age_seconds: float
    ) -> VerifiedRuntimeReconciliationEvidence:
        assert max_age_seconds == 30.0
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.prepared_evidence is not None:
            evidence = self.prepared_evidence
            self.prepared_evidence = None
            return evidence  # type: ignore[return-value]
        return _registered_evidence(
            self.database_path,
            snapshot=self.snapshot,
            expires_at=self.expires_at,
        )

    async def close(self) -> None:
        if self.close_started is not None:
            self.close_started.set()
        if self.close_release is not None:
            await self.close_release.wait()
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class _Clock:
    def __init__(self, current: datetime = NOW) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def _comparison(
    differences: tuple[ReconciliationDifference, ...] = (),
    *,
    coverage: ReconciliationCoverage | None = None,
    timing_lag_proofs: tuple[ExpectedTimingLagProof, ...] = (),
):
    def compare(
        snapshot: NormalizedBrokerSnapshot,
        trigger: ReconciliationTrigger,
    ) -> ReconciliationComparison:
        assert snapshot.account.account_scope == ACCOUNT_SCOPE
        assert type(trigger) is ReconciliationTrigger
        return ReconciliationComparison(
            coverage or _coverage(),
            differences,
            timing_lag_proofs,
        )

    return compare


def _difference(kind: DifferenceKind) -> ReconciliationDifference:
    materiality = (
        DifferenceMateriality.UNKNOWN
        if kind is DifferenceKind.UNKNOWN
        else DifferenceMateriality.MATERIAL
    )
    return ReconciliationDifference(
        kind=kind,
        materiality=materiality,
        reason_code=f"TEST_{kind.value.upper()}",
        subject="AAPL",
    )


async def _service(
    tmp_path,
    *,
    snapshot: NormalizedBrokerSnapshot | None = None,
    comparison=None,
    clock: _Clock | None = None,
) -> tuple[ReconciliationService, _EvidenceSource, ReconciliationPersistence]:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    source = _EvidenceSource(database_path, snapshot or _snapshot())
    persistence = ReconciliationPersistence(database_path)
    service = ReconciliationService(
        evidence_source=source,
        comparison_source=comparison or _comparison(),
        persistence=persistence,
        expected_account_scope=ACCOUNT_SCOPE,
        max_age_seconds=30,
        periodic_interval_seconds=15,
        clock=clock or _Clock(),
    )
    return service, source, persistence


@pytest.mark.asyncio
async def test_clean_startup_is_durable_and_entry_eligible(tmp_path) -> None:
    service, _, _ = await _service(tmp_path)

    outcome = await service.reconcile_startup()

    assert outcome.trigger is ReconciliationTrigger.STARTUP
    assert outcome.state is ReconciliationServiceState.READY
    assert outcome.verdict.status is ReconciliationStatus.PASSED
    assert outcome.entry_eligible is True
    assert service.entry_eligible(at=NOW) is True
    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        assert connection.execute(
            "SELECT component, version FROM rt_schema_migrations"
        ).fetchall() == [(RECONCILIATION_COMPONENT, 1)]
        assert connection.execute(
            "SELECT trigger_type, status, entry_eligible FROM rt_reconciliation_runs"
        ).fetchall() == [("startup", "passed", 1)]
        assert connection.execute(
            "SELECT COUNT(*) FROM rt_reconciliation_snapshots"
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_component_migration_preserves_unrelated_schema_and_rows(tmp_path) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute(
            "CREATE TABLE legacy_user_history (event_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        await connection.execute(
            "INSERT INTO legacy_user_history VALUES ('event-1', 'irreplaceable')"
        )
        await connection.commit()
    persistence = ReconciliationPersistence(database_path)

    await persistence.initialize()
    await persistence.initialize()

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT * FROM legacy_user_history").fetchall() == [
            ("event-1", "irreplaceable")
        ]
        assert connection.execute(
            "SELECT component, version FROM rt_schema_migrations"
        ).fetchall() == [(RECONCILIATION_COMPONENT, 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "comparison", "reason_code"),
    [
        (
            _snapshot(retrieved_at=NOW - timedelta(minutes=1)),
            _comparison(),
            "BROKER_EVIDENCE_STALE",
        ),
        (_snapshot(complete=False), _comparison(), "BROKER_EVIDENCE_INCOMPLETE"),
        (
            _snapshot(),
            _comparison((_difference(DifferenceKind.QUANTITY_MISMATCH),)),
            "TEST_QUANTITY_MISMATCH",
        ),
        (
            _snapshot(),
            _comparison((_difference(DifferenceKind.UNKNOWN),)),
            "TEST_UNKNOWN",
        ),
        (
            _snapshot(),
            _comparison(coverage=_coverage(ledger_cash=False)),
            "LOCAL_COMPARISON_INCOMPLETE",
        ),
    ],
)
async def test_stale_incomplete_material_and_unknown_evidence_quarantine(
    tmp_path,
    snapshot: NormalizedBrokerSnapshot,
    comparison,
    reason_code: str,
) -> None:
    service, _, _ = await _service(
        tmp_path,
        snapshot=snapshot,
        comparison=comparison,
    )

    outcome = await service.reconcile_startup()

    assert outcome.state is ReconciliationServiceState.QUARANTINED
    assert outcome.entry_eligible is False
    assert service.entry_eligible(at=NOW) is False
    assert reason_code in {difference.reason_code for difference in outcome.verdict.differences}
    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        assert connection.execute(
            "SELECT status, entry_eligible FROM rt_reconciliation_runs"
        ).fetchone() == ("quarantined", 0)
        assert reason_code in {
            row[0]
            for row in connection.execute(
                "SELECT reason_code FROM rt_reconciliation_differences"
            ).fetchall()
        }


@pytest.mark.asyncio
async def test_reconnect_and_due_periodic_generations_are_serialized(tmp_path) -> None:
    clock = _Clock()
    service, source, _ = await _service(tmp_path, clock=clock)
    await service.reconcile_startup()
    clock.advance(1)
    await service.reconcile_reconnect()
    clock.advance(5)

    assert await service.reconcile_periodic_if_due() is None
    clock.advance(10)
    periodic = await service.reconcile_periodic_if_due()

    assert periodic is not None
    assert periodic.trigger is ReconciliationTrigger.PERIODIC
    assert source.calls == 3
    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        assert connection.execute(
            "SELECT trigger_type FROM rt_reconciliation_runs ORDER BY completed_at, rowid"
        ).fetchall() == [("startup",), ("reconnect",), ("periodic",)]


@pytest.mark.asyncio
async def test_eligibility_expires_without_waiting_for_another_periodic_run(tmp_path) -> None:
    clock = _Clock()
    service, _, _ = await _service(tmp_path, clock=clock)
    await service.reconcile_startup()

    clock.advance(31)

    assert service.entry_eligible() is False


@pytest.mark.asyncio
async def test_partial_migration_fails_closed_without_using_runtime_state(tmp_path) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute("""
            CREATE TABLE rt_schema_migrations (
                component TEXT NOT NULL,
                version INTEGER NOT NULL,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                PRIMARY KEY(component, version)
            )
        """)
        await connection.execute(
            "INSERT INTO rt_schema_migrations VALUES (?, 1, 'incomplete', ?)",
            (RECONCILIATION_COMPONENT, "2026-07-28T14:00:00Z"),
        )
        await connection.commit()
    source = _EvidenceSource(database_path, _snapshot())
    service = ReconciliationService(
        evidence_source=source,
        comparison_source=_comparison(),
        persistence=ReconciliationPersistence(database_path),
        expected_account_scope=ACCOUNT_SCOPE,
    )

    with pytest.raises(ReconciliationServiceBlocked, match="initialization failed"):
        await service.reconcile_startup()

    assert service.state is ReconciliationServiceState.QUARANTINED
    assert service.entry_eligible(at=NOW) is False
    assert source.calls == 0


@pytest.mark.asyncio
async def test_duplicate_append_is_idempotent_not_a_duplicate_callback(tmp_path) -> None:
    persistence = ReconciliationPersistence((tmp_path / "reconciliation.sqlite3").resolve())
    await persistence.initialize()
    snapshot = _snapshot()
    verdict = evaluate_paper_simulator_reconciliation(
        snapshot,
        _coverage(),
        expected_account_scope=ACCOUNT_SCOPE,
        now=NOW,
        max_age_seconds=30,
    )
    runtime_evidence = _registered_evidence(
        tmp_path / "reconciliation.sqlite3",
        snapshot=snapshot,
    )

    first = await persistence.append_reconciliation(
        trigger_type="startup",
        runtime_evidence=runtime_evidence,
        verdict=verdict,
        started_at=NOW,
        completed_at=NOW,
        eligible_until=NOW + timedelta(seconds=29),
    )
    second = await persistence.append_reconciliation(
        trigger_type="startup",
        runtime_evidence=runtime_evidence,
        verdict=verdict,
        started_at=NOW,
        completed_at=NOW,
        eligible_until=NOW + timedelta(seconds=29),
    )

    assert second == first
    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM rt_reconciliation_runs").fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM rt_reconciliation_snapshots"
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_operator_resolution_appends_without_replacing_quarantine(tmp_path) -> None:
    service, _, persistence = await _service(
        tmp_path,
        comparison=_comparison((_difference(DifferenceKind.UNKNOWN),)),
    )
    outcome = await service.reconcile_startup()
    difference_id = outcome.persisted.difference_ids[0]

    event = await persistence.append_operator_resolution(
        run_id=outcome.persisted.run_id,
        difference_id=difference_id,
        resolution_kind=OperatorResolutionKind.INVESTIGATION_NOTE,
        operator_id="operator@example.com",
        reason="Broker evidence reviewed; external investigation remains open.",
        evidence_reference="ticket/RT-205",
        created_at=NOW + timedelta(seconds=1),
    )

    assert event.difference_id == difference_id
    assert service.state is ReconciliationServiceState.QUARANTINED
    assert service.entry_eligible(at=NOW + timedelta(seconds=1)) is False
    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        assert connection.execute(
            "SELECT status, entry_eligible FROM rt_reconciliation_runs"
        ).fetchone() == ("quarantined", 0)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE rt_reconciliation_operator_resolutions SET reason = 'replacement'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM rt_reconciliation_differences")


@pytest.mark.asyncio
async def test_service_failure_clears_prior_entry_eligibility(tmp_path) -> None:
    service, source, _ = await _service(tmp_path)
    await service.reconcile_startup()
    source.error = RuntimeError("diagnostic collection failed")

    with pytest.raises(ReconciliationServiceBlocked, match="failed closed"):
        await service.reconcile_reconnect()

    assert service.state is ReconciliationServiceState.QUARANTINED
    assert service.latest_outcome is None
    assert service.entry_eligible(at=NOW) is False


@pytest.mark.asyncio
async def test_fabricated_snapshot_source_is_rejected_before_comparison(tmp_path) -> None:
    service, source, _ = await _service(tmp_path)
    source.prepared_evidence = _snapshot()

    with pytest.raises(ReconciliationServiceBlocked, match="failed closed"):
        await service.reconcile_startup()

    assert service.state is ReconciliationServiceState.QUARANTINED
    assert service.entry_eligible(at=NOW) is False


def test_core_broker_and_bundle_assertions_issue_one_shot_runtime_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    sqlite3.connect(database_path).close()
    metadata = os.lstat(database_path)
    snapshot = _snapshot()
    snapshot_hash = hashlib.sha256(snapshot.canonical_payload().encode("utf-8")).hexdigest()
    contract = RuntimeContract(
        environment="development",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(database_path),
        account_alias="***1234",
        account_type="paper",
        model_artifact_set="test-models",
        build_id="test-build",
        state_namespace="paper",
        safety_account_scope=ACCOUNT_SCOPE,
        safety_execution_domain_scope="paper-simulator-v1",
    )
    context = SimpleNamespace(runtime_contract=contract, account_alias="***1234")
    broker = VerifiedBrokerEvidenceEnvelope(
        snapshot=snapshot,
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot_hash,
        runtime_fingerprint=contract.fingerprint,
        account_scope=ACCOUNT_SCOPE,
        receipt_id="receipt-v1-" + "c3" * 32,
        public_key_fingerprint="d4" * 32,
        artifact_hash="b2" * 32,
        issued_at=NOW - timedelta(microseconds=1),
        expires_at=NOW + timedelta(seconds=29),
        _marker=receiver_module._VERIFIED_BROKER_MARKER,
    )
    bundle = ProtectiveMarkBundleIdentity(
        receiver_type=object,
        bundle_id="bootstrap-evidence-bundle-v1-" + "a1" * 32,
        reconciliation_snapshot_id="reconciliation-v1-" + "f6" * 32,
        broker_snapshot_id=snapshot.snapshot_id,
        broker_snapshot_hash=snapshot_hash,
        broker_artifact_hash=broker.artifact_hash,
        broker_receipt_id=broker.receipt_id,
        broker_public_key_fingerprint=broker.public_key_fingerprint,
        runtime_fingerprint=contract.fingerprint,
        account_scope=ACCOUNT_SCOPE,
        database_identity=contract.database_identity,
        database_device=metadata.st_dev,
        database_inode=metadata.st_ino,
    )
    asserted = []

    def assert_broker(value):
        asserted.append("broker")
        assert value is broker
        return broker

    def assert_bundle(value, *, runtime_contract):
        asserted.append("bundle")
        assert value is marker
        assert runtime_contract is contract
        return bundle

    marker = object()
    monkeypatch.setattr(
        runtime_evidence_module,
        "assert_and_consume_verified_broker_evidence",
        assert_broker,
    )
    monkeypatch.setattr(
        runtime_evidence_module,
        "assert_protective_mark_receiver_capability",
        assert_bundle,
    )

    evidence = bind_verified_runtime_reconciliation_evidence(broker, context, marker)

    assert asserted == ["broker", "bundle"]
    assert evidence.database_device == metadata.st_dev
    assert evidence.database_inode == metadata.st_ino
    assert assert_and_consume_verified_runtime_reconciliation_evidence(evidence) is evidence
    with pytest.raises(RuntimeReconciliationEvidenceError, match="already consumed"):
        assert_and_consume_verified_runtime_reconciliation_evidence(evidence)


@pytest.mark.asyncio
async def test_replaced_database_blocks_exact_prepared_evidence(tmp_path) -> None:
    service, source, _ = await _service(tmp_path)
    await service.initialize()
    database_path = tmp_path / "reconciliation.sqlite3"
    source.prepared_evidence = _registered_evidence(database_path)
    replacement = tmp_path / "replacement.sqlite3"
    sqlite3.connect(replacement).close()
    os.replace(replacement, database_path)

    with pytest.raises(ReconciliationServiceBlocked, match="failed closed"):
        await service.reconcile_startup()

    assert service.state is ReconciliationServiceState.QUARANTINED


@pytest.mark.asyncio
async def test_persisted_snapshot_contains_full_runtime_and_broker_binding(tmp_path) -> None:
    service, _, _ = await _service(tmp_path)
    outcome = await service.reconcile_startup()

    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        row = connection.execute(
            """
            SELECT runtime_fingerprint, account_scope, account_alias,
                   database_identity, database_device, database_inode,
                   broker_artifact_hash, broker_receipt_id,
                   broker_public_key_fingerprint, bundle_id, snapshot_hash
            FROM rt_reconciliation_snapshots WHERE snapshot_id = ?
            """,
            (outcome.persisted.snapshot_id,),
        ).fetchone()

    assert row is not None
    assert row[0]
    assert row[1:3] == (ACCOUNT_SCOPE, "***1234")
    assert row[3]
    assert type(row[4]) is int and type(row[5]) is int
    assert row[6] == "b2" * 32
    assert row[7] == "receipt-v1-" + "c3" * 32
    assert row[8] == "d4" * 32
    assert row[9].startswith("bootstrap-evidence-bundle-v1-")
    assert len(row[10]) == 64


@pytest.mark.asyncio
async def test_eligibility_expires_at_relied_on_timing_proof(tmp_path) -> None:
    snapshot = _snapshot()
    broker_event_id = "broker-event-v1-" + "e5" * 32
    lag = ReconciliationDifference(
        kind=DifferenceKind.EXPECTED_TIMING_LAG,
        materiality=DifferenceMateriality.INFORMATIONAL,
        reason_code="BROKER_ORDER_EVENT_PENDING",
        subject="AAPL",
        evidence_ids=(broker_event_id,),
    )
    proof = ExpectedTimingLagProof.from_trusted_producer(
        broker_snapshot_id=snapshot.snapshot_id,
        reason_code=lag.reason_code,
        subject=lag.subject,
        broker_event_id=broker_event_id,
        started_at=NOW - timedelta(seconds=1),
        expires_at=NOW + timedelta(seconds=5),
    )
    service, _, _ = await _service(
        tmp_path,
        snapshot=snapshot,
        comparison=_comparison((lag,), timing_lag_proofs=(proof,)),
    )

    outcome = await service.reconcile_startup()

    assert outcome.state is ReconciliationServiceState.DEGRADED
    assert outcome.eligible_until == proof.expires_at
    assert service.entry_eligible(at=NOW + timedelta(seconds=5)) is True
    assert service.entry_eligible(at=NOW + timedelta(seconds=5, microseconds=1)) is False
    assert service.state is ReconciliationServiceState.QUARANTINED


@pytest.mark.asyncio
async def test_clock_rollback_quarantines_entry_eligibility(tmp_path) -> None:
    service, _, _ = await _service(tmp_path)
    outcome = await service.reconcile_startup()

    assert service.entry_eligible(at=outcome.verdict.checked_at) is True
    assert (
        service.entry_eligible(at=outcome.verdict.checked_at - timedelta(microseconds=1)) is False
    )
    assert service.state is ReconciliationServiceState.QUARANTINED


@pytest.mark.asyncio
async def test_resolution_composite_fk_rejects_difference_from_another_run(tmp_path) -> None:
    clock = _Clock()
    service, _, _ = await _service(
        tmp_path,
        comparison=_comparison((_difference(DifferenceKind.UNKNOWN),)),
        clock=clock,
    )
    first = await service.reconcile_startup()
    clock.advance(1)
    second = await service.reconcile_reconnect()

    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                """
                INSERT INTO rt_reconciliation_operator_resolutions(
                    resolution_id, schema_version, run_id, difference_id,
                    resolution_kind, operator_id, reason, evidence_reference, created_at
                ) VALUES (?, 1, ?, ?, 'investigation_note', 'operator',
                          'cross-run target must fail', NULL, ?)
                """,
                (
                    "reconciliation-resolution-v1-cross-run",
                    first.persisted.run_id,
                    second.persisted.difference_ids[0],
                    canonical_timestamp(NOW + timedelta(seconds=2)),
                ),
            )


@pytest.mark.asyncio
async def test_close_cancellation_shields_cleanup_before_closing(tmp_path) -> None:
    service, source, _ = await _service(tmp_path)
    source.close_started = asyncio.Event()
    source.close_release = asyncio.Event()
    close_task = asyncio.create_task(service.close())
    await source.close_started.wait()

    close_task.cancel()
    await asyncio.sleep(0)
    assert service.state is ReconciliationServiceState.CLOSING
    assert source.closed is False
    source.close_release.set()

    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert source.closed is True
    assert service.state is ReconciliationServiceState.CLOSED


@pytest.mark.asyncio
async def test_close_failure_remains_retryable(tmp_path) -> None:
    service, source, _ = await _service(tmp_path)
    source.close_error = RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await service.close()
    assert service.state is ReconciliationServiceState.CLOSING

    source.close_error = None
    await service.close()
    assert source.closed is True
    assert service.state is ReconciliationServiceState.CLOSED


@pytest.mark.asyncio
async def test_operator_resolution_retry_is_idempotent_and_conflict_fails(tmp_path) -> None:
    service, _, persistence = await _service(
        tmp_path,
        comparison=_comparison((_difference(DifferenceKind.UNKNOWN),)),
    )
    outcome = await service.reconcile_startup()
    difference_id = outcome.persisted.difference_ids[0]
    arguments = {
        "run_id": outcome.persisted.run_id,
        "difference_id": difference_id,
        "resolution_kind": OperatorResolutionKind.INVESTIGATION_NOTE,
        "operator_id": "operator@example.com",
        "reason": "Reviewed once and retained for another reconciliation.",
        "evidence_reference": "ticket/RT-206",
        "created_at": NOW + timedelta(seconds=1),
    }

    first = await persistence.append_operator_resolution(**arguments)
    second = await persistence.append_operator_resolution(**arguments)

    assert second == first
    conflict_arguments = {
        **arguments,
        "reason": "A different deterministic resolution payload for conflict testing.",
        "created_at": NOW + timedelta(seconds=2),
    }
    payload = {
        "created_at": canonical_timestamp(conflict_arguments["created_at"]),
        "difference_id": difference_id,
        "evidence_reference": conflict_arguments["evidence_reference"],
        "operator_id": conflict_arguments["operator_id"],
        "reason": conflict_arguments["reason"],
        "resolution_kind": OperatorResolutionKind.INVESTIGATION_NOTE.value,
        "run_id": outcome.persisted.run_id,
        "schema_version": 1,
    }
    conflict_id = fingerprint("reconciliation-resolution-v1", payload)
    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO rt_reconciliation_operator_resolutions(
                resolution_id, schema_version, run_id, difference_id,
                resolution_kind, operator_id, reason, evidence_reference, created_at
            ) VALUES (?, 1, ?, ?, 'investigation_note', 'other-operator',
                      'conflicting stored evidence', ?, ?)
            """,
            (
                conflict_id,
                outcome.persisted.run_id,
                difference_id,
                conflict_arguments["evidence_reference"],
                canonical_timestamp(conflict_arguments["created_at"]),
            ),
        )
        connection.commit()

    with pytest.raises(ReconciliationPersistenceError, match="conflicting evidence"):
        await persistence.append_operator_resolution(**conflict_arguments)
    with sqlite3.connect(tmp_path / "reconciliation.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM rt_reconciliation_operator_resolutions"
        ).fetchone() == (2,)
