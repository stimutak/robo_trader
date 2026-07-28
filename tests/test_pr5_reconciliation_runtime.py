from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import aiosqlite
import pytest

from robo_trader.reconciliation.domain import (
    BrokerCollectionEvidence,
    BrokerCollectionKind,
    BrokerEvidenceCompleteness,
    NormalizedBrokerAccount,
    NormalizedBrokerSnapshot,
)
from robo_trader.reconciliation.persistence import (
    OperatorResolutionKind,
    ReconciliationPersistence,
)
from robo_trader.reconciliation.policy import (
    DifferenceKind,
    DifferenceMateriality,
    ReconciliationCoverage,
    ReconciliationDifference,
    ReconciliationStatus,
    evaluate_paper_simulator_reconciliation,
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


class _SnapshotSource:
    def __init__(self, snapshot: NormalizedBrokerSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0
        self.closed = False
        self.error: Exception | None = None

    async def collect_normalized_snapshot(
        self, *, max_age_seconds: float
    ) -> NormalizedBrokerSnapshot:
        assert max_age_seconds == 30.0
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.snapshot

    async def close(self) -> None:
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
):
    def compare(
        snapshot: NormalizedBrokerSnapshot,
        trigger: ReconciliationTrigger,
    ) -> ReconciliationComparison:
        assert snapshot.account.account_scope == ACCOUNT_SCOPE
        assert type(trigger) is ReconciliationTrigger
        return ReconciliationComparison(coverage or _coverage(), differences)

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
) -> tuple[ReconciliationService, _SnapshotSource, ReconciliationPersistence]:
    source = _SnapshotSource(snapshot or _snapshot())
    persistence = ReconciliationPersistence((tmp_path / "reconciliation.sqlite3").resolve())
    service = ReconciliationService(
        snapshot_source=source,
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
    source = _SnapshotSource(_snapshot())
    service = ReconciliationService(
        snapshot_source=source,
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

    first = await persistence.append_reconciliation(
        trigger_type="startup",
        snapshot=snapshot,
        verdict=verdict,
        started_at=NOW,
        completed_at=NOW,
    )
    second = await persistence.append_reconciliation(
        trigger_type="startup",
        snapshot=snapshot,
        verdict=verdict,
        started_at=NOW,
        completed_at=NOW,
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
