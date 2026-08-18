from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import pickle
import sqlite3
import weakref
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from robo_trader import bootstrap_evidence_receivers as receiver_module
from robo_trader import reconciliation_migrations as migrations_module
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
    monkeypatch.setattr(runtime_evidence_module, "_comparison_clock", lambda: NOW)
    monkeypatch.setattr(
        runtime_evidence_module,
        "assert_exact_state_runtime_sources_unchanged",
        lambda evidence, runtime_contract: None,
    )
    with runtime_evidence_module._COMPARISON_CONSUMPTION_LOCK:
        runtime_evidence_module._CONSUMED_COMPARISON_LINEAGES.clear()
        runtime_evidence_module._COMPARISON_LAST_CLOCK = None


def _exact_state_source() -> runtime_evidence_module.ExactStateBootstrapEvidence:
    source = object.__new__(runtime_evidence_module.ExactStateBootstrapEvidence)
    object.__setattr__(source, "_producer_digest", "91" * 32)
    return source


def _registered_evidence(
    database_path,
    *,
    snapshot: NormalizedBrokerSnapshot | None = None,
    expires_at: datetime = NOW + timedelta(seconds=29),
    coverage: ReconciliationCoverage | None = None,
    differences: tuple[ReconciliationDifference, ...] = (),
    timing_lag_proofs: tuple[ExpectedTimingLagProof, ...] = (),
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
            comparison_coverage=coverage or _coverage(),
            differences=differences,
            timing_lag_proofs=timing_lag_proofs,
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
            broker_evidence_expires_at=expires_at,
            reconciliation_snapshot_id="bootstrap-reconciliation-v1-" + "e5" * 32,
            reconciliation_artifact_path=str(path.parent / "reconciliation_report.json"),
            reconciliation_artifact_hash="f6" * 32,
            reconciliation_receipt_id="bevr-v2-" + "a7" * 32,
            reconciliation_public_key_fingerprint="b8" * 32,
            reconciliation_signature_ed25519="c2lnbmF0dXJl",
            reconciliation_evidence_issued_at=NOW - timedelta(microseconds=1),
            reconciliation_evidence_expires_at=expires_at,
            issued_at=NOW - timedelta(microseconds=1),
            expires_at=expires_at,
            _runtime_context=context,
            _exact_state_evidence=_exact_state_source(),
            _marker=runtime_evidence_module._CAPABILITY_MARKER,
        )
    )


class _EvidenceSource:
    def __init__(self, database_path, snapshot: NormalizedBrokerSnapshot, comparison) -> None:
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
        self.comparison = comparison

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
            coverage=self.comparison.coverage,
            differences=self.comparison.differences,
            timing_lag_proofs=self.comparison.timing_lag_proofs,
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
    return SimpleNamespace(
        coverage=coverage or _coverage(),
        differences=differences,
        timing_lag_proofs=timing_lag_proofs,
    )


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
    source = _EvidenceSource(database_path, snapshot or _snapshot(), comparison or _comparison())
    persistence = ReconciliationPersistence(database_path)
    await persistence.migrate_for_operator(operator_confirmed=True)
    service = ReconciliationService(
        evidence_source=source,
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
        ).fetchall() == [(RECONCILIATION_COMPONENT, 1), (RECONCILIATION_COMPONENT, 2)]
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

    await persistence.migrate_for_operator(operator_confirmed=True)
    await persistence.initialize()

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT * FROM legacy_user_history").fetchall() == [
            ("event-1", "irreplaceable")
        ]
        assert connection.execute(
            "SELECT component, version FROM rt_schema_migrations"
        ).fetchall() == [(RECONCILIATION_COMPONENT, 1), (RECONCILIATION_COMPONENT, 2)]


@pytest.mark.asyncio
async def test_runtime_initialize_is_read_only_and_operator_migration_is_explicit(
    tmp_path,
) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE irreplaceable_history(value TEXT NOT NULL)")
        connection.execute("INSERT INTO irreplaceable_history VALUES ('preserve-me')")
    persistence = ReconciliationPersistence(database_path)
    before = database_path.read_bytes()

    with pytest.raises(ReconciliationPersistenceError, match="schema is unavailable"):
        await persistence.initialize()

    assert database_path.read_bytes() == before
    assert not Path(f"{database_path}-wal").exists()
    assert not Path(f"{database_path}-shm").exists()
    with pytest.raises(ReconciliationPersistenceError, match="operator confirmation"):
        await persistence.migrate_for_operator(operator_confirmed=False)
    assert database_path.read_bytes() == before

    await persistence.migrate_for_operator(operator_confirmed=True)
    await persistence.initialize()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT * FROM irreplaceable_history").fetchall() == [
            ("preserve-me",)
        ]


@pytest.mark.asyncio
async def test_exact_old_v1_schema_upgrades_additively_without_row_loss(tmp_path) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    account_scope = ACCOUNT_SCOPE
    payload = "{}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("BEGIN IMMEDIATE")
        await migrations_module._migration_v1(connection)
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
            "INSERT INTO rt_schema_migrations VALUES (?, 1, 'old-v1', ?)",
            (RECONCILIATION_COMPONENT, canonical_timestamp(NOW)),
        )
        await connection.execute(
            "INSERT INTO rt_reconciliation_snapshots VALUES (?, 1, ?, ?, ?, ?, 1, ?, ?, ?)",
            (
                "snapshot-old-v1",
                account_scope,
                canonical_timestamp(NOW),
                canonical_timestamp(NOW),
                canonical_timestamp(NOW),
                payload,
                digest,
                canonical_timestamp(NOW),
            ),
        )
        await connection.execute(
            """
            INSERT INTO rt_reconciliation_runs VALUES (
                ?, 1, 'startup', ?, ?, ?, ?, ?, 'quarantined',
                1, 1, 1, 0, ?, ?, ?
            )
            """,
            (
                "run-old-v1",
                "snapshot-old-v1",
                "verdict-old-v1",
                account_scope,
                canonical_timestamp(NOW),
                canonical_timestamp(NOW),
                payload,
                payload,
                digest,
            ),
        )
        await connection.execute(
            """
            INSERT INTO rt_reconciliation_differences VALUES (
                ?, ?, 0, 'unknown', 'unknown', 'OLD_V1_UNKNOWN',
                'AAPL', '[]', ?, ?, ?
            )
            """,
            (
                "difference-old-v1",
                "run-old-v1",
                payload,
                digest,
                canonical_timestamp(NOW),
            ),
        )
        await connection.execute(
            "INSERT INTO rt_reconciliation_operator_resolutions VALUES (?, 1, ?, ?, ?, ?, ?, NULL, ?)",
            (
                "resolution-old-v1",
                "run-old-v1",
                "difference-old-v1",
                "investigation_note",
                "operator",
                "preserve exact old v1 evidence",
                canonical_timestamp(NOW),
            ),
        )
        await connection.commit()

    await ReconciliationPersistence(database_path).migrate_for_operator(operator_confirmed=True)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT snapshot_id FROM rt_reconciliation_snapshots"
        ).fetchall() == [("snapshot-old-v1",)]
        assert connection.execute("SELECT run_id FROM rt_reconciliation_runs").fetchall() == [
            ("run-old-v1",)
        ]
        assert connection.execute(
            "SELECT difference_id FROM rt_reconciliation_differences"
        ).fetchall() == [("difference-old-v1",)]
        assert connection.execute(
            "SELECT resolution_id, run_id, difference_id "
            "FROM rt_reconciliation_operator_resolution_bindings"
        ).fetchall() == [("resolution-old-v1", "run-old-v1", "difference-old-v1")]
        assert connection.execute(
            "SELECT version FROM rt_schema_migrations WHERE component=? ORDER BY version",
            (RECONCILIATION_COMPONENT,),
        ).fetchall() == [(1,), (2,)]


def _rewrite_schema_definition(
    database_path,
    object_type: str,
    name: str,
    old: str,
    new: str,
) -> None:
    assert object_type in {"table", "trigger"}
    with sqlite3.connect(database_path) as connection:
        definition = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type=? AND name=?",
            (object_type, name),
        ).fetchone()
        assert definition is not None
        assert definition[0].count(old) == 1
        replacement = definition[0].replace(old, new, 1)
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_master SET sql=? WHERE type=? AND name=?",
            (replacement, object_type, name),
        )
        connection.execute(f"PRAGMA schema_version = {schema_version + 1}")
        connection.execute("PRAGMA writable_schema = OFF")
        connection.commit()


def _rewrite_table_definition(
    database_path,
    table: str,
    old: str,
    new: str,
) -> None:
    _rewrite_schema_definition(database_path, "table", table, old, new)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "old", "new"),
    [
        (
            "rt_reconciliation_snapshot_lineage",
            "snapshot_id TEXT PRIMARY KEY NOT NULL",
            "snapshot_id TEXT UNIQUE NOT NULL",
        ),
        (
            "rt_reconciliation_snapshot_lineage",
            "schema_version INTEGER NOT NULL CHECK (schema_version = 2)",
            "schema_version INTEGER NOT NULL",
        ),
        (
            "rt_reconciliation_snapshot_lineage",
            "length(snapshot_hash) = 64",
            "length(snapshot_hash) >= 1",
        ),
        (
            "rt_reconciliation_snapshot_lineage",
            "broker_receipt_id TEXT NOT NULL",
            "broker_receipt_id TEXT",
        ),
        (
            "rt_reconciliation_snapshot_lineage",
            "broker_public_key_fingerprint TEXT NOT NULL",
            "broker_public_key_fingerprint TEXT",
        ),
        (
            "rt_reconciliation_snapshot_lineage",
            "REFERENCES rt_reconciliation_snapshots(snapshot_id)",
            "REFERENCES rt_reconciliation_snapshots(snapshot_id) ON DELETE CASCADE",
        ),
        (
            "rt_reconciliation_snapshot_lineage",
            "persisted_at TEXT NOT NULL,",
            "persisted_at TEXT NOT NULL, unexpected_column TEXT,",
        ),
        (
            "rt_reconciliation_run_eligibility",
            "run_id TEXT PRIMARY KEY NOT NULL",
            "run_id TEXT PRIMARY KEY",
        ),
        (
            "rt_reconciliation_run_eligibility",
            "eligible_until TEXT NOT NULL",
            "eligible_until TEXT",
        ),
        (
            "rt_reconciliation_operator_resolution_bindings",
            "resolution_id TEXT PRIMARY KEY NOT NULL",
            "resolution_id TEXT PRIMARY KEY",
        ),
        (
            "rt_reconciliation_operator_resolution_bindings",
            "FOREIGN KEY(run_id, difference_id)",
            "FOREIGN KEY(difference_id, run_id)",
        ),
    ],
)
async def test_v2_constraint_mutation_matrix_fails_closed(
    tmp_path,
    table: str,
    old: str,
    new: str,
) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    persistence = ReconciliationPersistence(database_path)
    await persistence.migrate_for_operator(operator_confirmed=True)
    _rewrite_table_definition(database_path, table, old, new)

    with pytest.raises((RuntimeError, sqlite3.DatabaseError), match="malformed"):
        await persistence.initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "old", "new"),
    [
        (
            "rt_reconciliation_snapshots",
            "schema_version INTEGER NOT NULL CHECK (schema_version = 1)",
            "schema_version INTEGER NOT NULL",
        ),
        (
            "rt_reconciliation_snapshots",
            "payload_json TEXT NOT NULL CHECK (json_valid(payload_json))",
            "payload_json TEXT NOT NULL",
        ),
        (
            "rt_reconciliation_runs",
            "evidence_fresh INTEGER NOT NULL CHECK (evidence_fresh IN (0, 1))",
            "evidence_fresh INTEGER NOT NULL",
        ),
        (
            "rt_reconciliation_runs",
            "FOREIGN KEY(snapshot_id) REFERENCES rt_reconciliation_snapshots(snapshot_id)",
            "FOREIGN KEY(snapshot_id) REFERENCES rt_reconciliation_snapshots(snapshot_id) "
            "ON DELETE CASCADE",
        ),
        (
            "rt_reconciliation_differences",
            "ordinal INTEGER NOT NULL CHECK (ordinal >= 0)",
            "ordinal INTEGER NOT NULL",
        ),
        (
            "rt_reconciliation_differences",
            "UNIQUE(run_id, ordinal)",
            "UNIQUE(ordinal, run_id)",
        ),
        (
            "rt_reconciliation_operator_resolutions",
            "reason TEXT NOT NULL CHECK (length(trim(reason)) >= 10)",
            "reason TEXT NOT NULL",
        ),
        (
            "rt_reconciliation_operator_resolutions",
            "evidence_reference TEXT,",
            "evidence_reference TEXT, unexpected_column TEXT,",
        ),
    ],
)
async def test_v1_constraint_mutation_matrix_fails_closed(
    tmp_path,
    table: str,
    old: str,
    new: str,
) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    persistence = ReconciliationPersistence(database_path)
    await persistence.migrate_for_operator(operator_confirmed=True)
    _rewrite_table_definition(database_path, table, old, new)

    with pytest.raises((RuntimeError, sqlite3.DatabaseError), match="malformed"):
        await persistence.initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table",
    [
        "rt_reconciliation_snapshots",
        "rt_reconciliation_runs",
        "rt_reconciliation_differences",
        "rt_reconciliation_operator_resolutions",
    ],
)
async def test_every_v1_append_only_trigger_is_exact(tmp_path, table: str) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    persistence = ReconciliationPersistence(database_path)
    await persistence.migrate_for_operator(operator_confirmed=True)
    trigger = f"{table}_no_delete"
    _rewrite_schema_definition(
        database_path,
        "trigger",
        trigger,
        "BEFORE DELETE",
        "AFTER DELETE",
    )

    with pytest.raises(RuntimeError, match=f"trigger {trigger} is malformed"):
        await persistence.initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table",
    [
        "rt_reconciliation_snapshot_lineage",
        "rt_reconciliation_run_eligibility",
        "rt_reconciliation_operator_resolution_bindings",
    ],
)
@pytest.mark.parametrize(
    ("suffix", "old", "new"),
    [
        ("no_update", "BEFORE UPDATE", "AFTER UPDATE"),
        ("no_delete", "BEFORE DELETE", "AFTER DELETE"),
    ],
)
async def test_every_v2_append_only_trigger_is_exact(
    tmp_path,
    table: str,
    suffix: str,
    old: str,
    new: str,
) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    persistence = ReconciliationPersistence(database_path)
    await persistence.migrate_for_operator(operator_confirmed=True)
    trigger = f"{table}_{suffix}"
    _rewrite_schema_definition(
        database_path,
        "trigger",
        trigger,
        old,
        new,
    )

    with pytest.raises(RuntimeError, match=f"trigger {trigger} is malformed"):
        await persistence.initialize()


@pytest.mark.asyncio
async def test_v2_composite_identity_index_mutation_fails_closed(tmp_path) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    persistence = ReconciliationPersistence(database_path)
    await persistence.migrate_for_operator(operator_confirmed=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX rt_reconciliation_differences_run_difference_uq")
        connection.execute(
            "CREATE UNIQUE INDEX rt_reconciliation_differences_run_difference_uq "
            "ON rt_reconciliation_differences(difference_id, run_id)"
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="indexes are malformed"):
        await persistence.initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "identity_column"),
    [
        ("rt_reconciliation_snapshot_lineage", "snapshot_id"),
        ("rt_reconciliation_run_eligibility", "run_id"),
    ],
)
async def test_v2_duplicate_null_identity_probe_fails_closed(
    tmp_path,
    table: str,
    identity_column: str,
) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    persistence = ReconciliationPersistence(database_path)
    await persistence.migrate_for_operator(operator_confirmed=True)
    constrained = f"{identity_column} TEXT PRIMARY KEY NOT NULL"
    nullable = f"{identity_column} TEXT PRIMARY KEY"
    _rewrite_table_definition(database_path, table, constrained, nullable)

    with sqlite3.connect(database_path) as connection:
        columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
        names = [str(row[1]) for row in columns]
        values = []
        for row in columns:
            name = str(row[1])
            if name == identity_column:
                values.append(None)
            elif name == "schema_version":
                values.append(2)
            elif str(row[2]).upper() == "INTEGER":
                values.append(1)
            elif name.endswith("hash") or name.endswith("fingerprint"):
                values.append("a" * 64)
            else:
                values.append("probe")
        placeholders = ", ".join("?" for _ in names)
        connection.executemany(
            f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders})",
            (values, values),
        )
        connection.commit()
        assert connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {identity_column} IS NULL"
        ).fetchone() == (2,)

    _rewrite_table_definition(database_path, table, nullable, constrained)
    with pytest.raises(RuntimeError, match="identity is malformed"):
        await persistence.initialize()


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
    source = _EvidenceSource(database_path, _snapshot(), _comparison())
    service = ReconciliationService(
        evidence_source=source,
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
    await persistence.migrate_for_operator(operator_confirmed=True)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch_kind", ["material", "timing_lag"])
async def test_authenticated_mismatch_is_persisted_with_timing_expiry(
    tmp_path,
    monkeypatch,
    mismatch_kind: str,
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
        reconciliation_snapshot_id="bootstrap-reconciliation-v1-" + "e5" * 32,
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
    reconciliation_receipt = SimpleNamespace(
        artifact_kind="reconciliation_report",
        artifact_sha256="f6" * 32,
        runtime_fingerprint=contract.fingerprint,
        account_scope=ACCOUNT_SCOPE,
        issued_at=NOW - timedelta(microseconds=1),
        expires_at=NOW + timedelta(seconds=29),
        receipt_id="bevr-v2-" + "a7" * 32,
        public_key_fingerprint="b8" * 32,
        signature_ed25519="c2lnbmF0dXJl",
    )
    if mismatch_kind == "material":
        signed_difference = _difference(DifferenceKind.QUANTITY_MISMATCH)
        signed_proofs: tuple[ExpectedTimingLagProof, ...] = ()
        signed_status = ReconciliationStatus.QUARANTINED
    else:
        broker_event_id = "broker-event-v1-" + "e5" * 32
        signed_difference = ReconciliationDifference(
            kind=DifferenceKind.EXPECTED_TIMING_LAG,
            materiality=DifferenceMateriality.INFORMATIONAL,
            reason_code="BROKER_ORDER_EVENT_PENDING",
            subject="AAPL",
            evidence_ids=(broker_event_id,),
        )
        signed_proofs = (
            ExpectedTimingLagProof.from_trusted_producer(
                broker_snapshot_id=snapshot.snapshot_id,
                reason_code=signed_difference.reason_code,
                subject=signed_difference.subject,
                broker_event_id=broker_event_id,
                started_at=NOW - timedelta(seconds=1),
                expires_at=NOW + timedelta(seconds=5),
            ),
        )
        signed_status = ReconciliationStatus.DEGRADED
    exact_state = SimpleNamespace(
        _producer_digest="91" * 32,
        authentication_receipts=(reconciliation_receipt,),
        reconciliation_status=signed_status,
        reconciliation_coverage=_coverage(),
        reconciliation_snapshot_id=bundle.reconciliation_snapshot_id,
        reconciliation_artifact_path=str(database_path.parent / "reconciliation_report.json"),
        reconciliation_report_hash=reconciliation_receipt.artifact_sha256,
        reconciliation_generated_at=NOW - timedelta(microseconds=2),
        bundle_id=bundle.bundle_id,
        runtime_fingerprint=contract.fingerprint,
        account_scope=ACCOUNT_SCOPE,
        database_path=str(database_path),
        database_identity=contract.database_identity,
        database_device=metadata.st_dev,
        database_inode=metadata.st_ino,
        broker_snapshot_id=broker.snapshot_id,
        broker_snapshot_hash=broker.artifact_hash,
        reconciliation_differences=(signed_difference,),
        reconciliation_timing_lag_proofs=signed_proofs,
    )

    def assert_broker(value):
        asserted.append("broker")
        assert value is broker
        return broker

    def assert_bundle(value, *, runtime_contract):
        asserted.append("bundle")
        assert value is marker
        assert runtime_contract is contract
        return bundle

    def assert_exact(value, runtime_contract):
        asserted.append("reconciliation")
        assert value is exact_state
        assert runtime_contract is contract
        return exact_state

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
    monkeypatch.setattr(
        runtime_evidence_module,
        "assert_verified_exact_state_reconciliation_evidence",
        assert_exact,
    )
    monkeypatch.setattr(
        runtime_evidence_module,
        "ExactStateBootstrapEvidence",
        SimpleNamespace,
    )
    revalidated = []
    monkeypatch.setattr(
        runtime_evidence_module,
        "assert_exact_state_runtime_sources_unchanged",
        lambda evidence, runtime_contract: revalidated.append((evidence, runtime_contract)),
    )

    evidence = bind_verified_runtime_reconciliation_evidence(
        broker,
        exact_state,
        context,
        marker,
    )

    assert asserted == ["broker", "reconciliation", "bundle"]
    assert evidence.database_device == metadata.st_dev
    assert evidence.database_inode == metadata.st_ino
    assert evidence.differences == (signed_difference,)
    with pytest.raises(RuntimeReconciliationEvidenceError, match="already consumed"):
        bind_verified_runtime_reconciliation_evidence(
            broker,
            exact_state,
            context,
            marker,
        )
    service, source, _ = await _service(tmp_path)
    source.prepared_evidence = evidence
    outcome = await service.reconcile_startup()

    expected_state = (
        ReconciliationServiceState.QUARANTINED
        if mismatch_kind == "material"
        else ReconciliationServiceState.DEGRADED
    )
    assert outcome.state is expected_state
    assert outcome.entry_eligible is (mismatch_kind == "timing_lag")
    assert revalidated == [(exact_state, contract), (exact_state, contract)]
    with sqlite3.connect(database_path) as connection:
        persisted_rows = connection.execute(
            "SELECT kind, materiality, reason_code FROM rt_reconciliation_differences"
        ).fetchall()
    assert persisted_rows == [
        (
            signed_difference.kind.value,
            signed_difference.materiality.value,
            signed_difference.reason_code,
        )
    ]
    if mismatch_kind == "timing_lag":
        assert service.entry_eligible(at=NOW + timedelta(seconds=5)) is True
        assert service.entry_eligible(at=NOW + timedelta(seconds=5, microseconds=1)) is False
        assert service.state is ReconciliationServiceState.QUARANTINED
    with pytest.raises(RuntimeReconciliationEvidenceError, match="already consumed"):
        assert_and_consume_verified_runtime_reconciliation_evidence(evidence)


def test_runtime_capability_is_python310_compatible_immutable_and_nonserializable(
    tmp_path,
) -> None:
    evidence = _registered_evidence(tmp_path / "reconciliation.sqlite3")

    assert weakref.ref(evidence)() is evidence
    assert not hasattr(evidence, "__dict__")
    with pytest.raises(FrozenInstanceError):
        evidence.bundle_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(evidence)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(evidence)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(evidence)


@pytest.mark.parametrize("mutation", ["missing", "substituted"])
def test_runtime_capability_rejects_raw_exact_state_reference_mutation(
    tmp_path,
    monkeypatch,
    mutation: str,
) -> None:
    evidence = _registered_evidence(tmp_path / "reconciliation.sqlite3")
    revalidated = []
    monkeypatch.setattr(
        runtime_evidence_module,
        "assert_exact_state_runtime_sources_unchanged",
        lambda source, contract: revalidated.append((source, contract)),
    )
    replacement = None if mutation == "missing" else _exact_state_source()

    object.__setattr__(evidence, "_exact_state_evidence", replacement)

    with pytest.raises(
        RuntimeReconciliationEvidenceError,
        match="forged, changed, or already consumed",
    ):
        assert_and_consume_verified_runtime_reconciliation_evidence(evidence)
    assert revalidated == []


def test_consumed_comparison_cache_is_bounded_and_evicts_only_expired(
    monkeypatch,
) -> None:
    current = [NOW]
    monkeypatch.setattr(runtime_evidence_module, "_MAX_CONSUMED_COMPARISON_LINEAGES", 3)
    monkeypatch.setattr(runtime_evidence_module, "_comparison_clock", lambda: current[0])
    for ordinal in range(3):
        runtime_evidence_module._consume_comparison_lineage(
            (f"snapshot-{ordinal}", f"receipt-{ordinal}", f"signature-{ordinal}"),
            expires_at=NOW + timedelta(seconds=10),
        )
    with pytest.raises(RuntimeReconciliationEvidenceError, match="safety bound"):
        runtime_evidence_module._consume_comparison_lineage(
            ("snapshot-full", "receipt-full", "signature-full"),
            expires_at=NOW + timedelta(seconds=10),
        )
    assert len(runtime_evidence_module._CONSUMED_COMPARISON_LINEAGES) == 3

    current[0] = NOW + timedelta(seconds=11)
    runtime_evidence_module._consume_comparison_lineage(
        ("snapshot-new", "receipt-new", "signature-new"),
        expires_at=NOW + timedelta(seconds=20),
    )
    assert runtime_evidence_module._CONSUMED_COMPARISON_LINEAGES == {
        ("snapshot-new", "receipt-new", "signature-new"): NOW + timedelta(seconds=20)
    }


def test_consumed_comparison_cache_fails_closed_on_clock_rollback(monkeypatch) -> None:
    current = [NOW]
    monkeypatch.setattr(runtime_evidence_module, "_comparison_clock", lambda: current[0])
    runtime_evidence_module._consume_comparison_lineage(
        ("snapshot-first", "receipt-first", "signature-first"),
        expires_at=NOW + timedelta(seconds=10),
    )
    current[0] = NOW - timedelta(microseconds=1)

    with pytest.raises(RuntimeReconciliationEvidenceError, match="clock moved backwards"):
        runtime_evidence_module._consume_comparison_lineage(
            ("snapshot-rollback", "receipt-rollback", "signature-rollback"),
            expires_at=NOW + timedelta(seconds=10),
        )
    assert len(runtime_evidence_module._CONSUMED_COMPARISON_LINEAGES) == 1


@pytest.mark.asyncio
async def test_public_comparison_callable_is_not_an_accepted_service_boundary(tmp_path) -> None:
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()
    source = _EvidenceSource(database_path, _snapshot(), _comparison())
    with pytest.raises(TypeError, match="comparison_source"):
        ReconciliationService(  # type: ignore[call-arg]
            evidence_source=source,
            comparison_source=lambda *_: _comparison(),
            persistence=ReconciliationPersistence(database_path),
            expected_account_scope=ACCOUNT_SCOPE,
        )


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
async def test_entry_eligibility_revalidates_database_inode_every_time(tmp_path) -> None:
    service, _, _ = await _service(tmp_path)
    await service.reconcile_startup()
    assert service.entry_eligible(at=NOW) is True
    database_path = tmp_path / "reconciliation.sqlite3"
    replacement = tmp_path / "replacement-after-reconcile.sqlite3"
    sqlite3.connect(replacement).close()
    os.replace(replacement, database_path)

    assert service.entry_eligible(at=NOW) is False
    assert service.state is ReconciliationServiceState.QUARANTINED


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_kind", ["database", "exact_state"])
async def test_post_commit_replacement_blocks_outcome(tmp_path, replacement_kind: str) -> None:
    service, _, persistence = await _service(tmp_path)
    database_path = (tmp_path / "reconciliation.sqlite3").resolve()

    class _ReplaceAfterCommitPersistence:
        async def initialize(self) -> None:
            await persistence.initialize()

        async def append_reconciliation(self, **kwargs):
            result = await persistence.append_reconciliation(**kwargs)
            if replacement_kind == "database":
                replacement = tmp_path / "replacement-after-commit.sqlite3"
                sqlite3.connect(replacement).close()
                os.replace(replacement, database_path)
            else:
                object.__setattr__(
                    kwargs["runtime_evidence"],
                    "_exact_state_evidence",
                    _exact_state_source(),
                )
            return result

    service._persistence = _ReplaceAfterCommitPersistence()  # type: ignore[assignment]

    with pytest.raises(ReconciliationServiceBlocked, match="failed closed"):
        await service.reconcile_startup()

    assert service.state is ReconciliationServiceState.QUARANTINED
    assert service.latest_outcome is None
    assert service.entry_eligible(at=NOW) is False


@pytest.mark.asyncio
async def test_post_persistence_timing_proof_deadline_blocks_positive_outcome(tmp_path) -> None:
    clock = _Clock()
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
    service, _, persistence = await _service(
        tmp_path,
        snapshot=snapshot,
        comparison=_comparison((lag,), timing_lag_proofs=(proof,)),
        clock=clock,
    )

    class _AdvancePastDeadlineAfterPersistence:
        async def initialize(self) -> None:
            await persistence.initialize()

        async def append_reconciliation(self, **kwargs):
            result = await persistence.append_reconciliation(**kwargs)
            clock.current = proof.expires_at + timedelta(microseconds=1)
            return result

    service._persistence = _AdvancePastDeadlineAfterPersistence()  # type: ignore[assignment]

    with pytest.raises(ReconciliationServiceBlocked, match="eligibility expired"):
        await service.reconcile_startup()

    assert service.state is ReconciliationServiceState.QUARANTINED
    assert service.latest_outcome is None
    assert service.entry_eligible(at=clock.current) is False


@pytest.mark.asyncio
async def test_post_persistence_clock_rollback_blocks_positive_outcome(tmp_path) -> None:
    clock = _Clock()
    service, _, persistence = await _service(tmp_path, clock=clock)

    class _RollBackClockAfterPersistence:
        async def initialize(self) -> None:
            await persistence.initialize()

        async def append_reconciliation(self, **kwargs):
            result = await persistence.append_reconciliation(**kwargs)
            clock.current = kwargs["completed_at"] - timedelta(microseconds=1)
            return result

    service._persistence = _RollBackClockAfterPersistence()  # type: ignore[assignment]

    with pytest.raises(ReconciliationServiceBlocked, match="clock moved backwards"):
        await service.reconcile_startup()

    assert service.state is ReconciliationServiceState.QUARANTINED
    assert service.latest_outcome is None
    assert service.entry_eligible(at=NOW) is False


@pytest.mark.asyncio
async def test_comparison_substitution_and_capability_reuse_are_rejected(tmp_path) -> None:
    service, source, _ = await _service(tmp_path)
    evidence = _registered_evidence(tmp_path / "reconciliation.sqlite3")
    object.__setattr__(evidence, "comparison_coverage", _coverage(ledger_cash=False))
    source.prepared_evidence = evidence

    with pytest.raises(ReconciliationServiceBlocked, match="failed closed"):
        await service.reconcile_startup()
    assert service.state is ReconciliationServiceState.QUARANTINED

    replay_directory = tmp_path / "replay"
    replay_directory.mkdir()
    replay_service, replay_source, _ = await _service(replay_directory)
    replay = _registered_evidence(replay_directory / "reconciliation.sqlite3")
    replay_source.prepared_evidence = replay
    await replay_service.reconcile_startup()
    replay_source.prepared_evidence = replay
    with pytest.raises(ReconciliationServiceBlocked, match="failed closed"):
        await replay_service.reconcile_reconnect()


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
                   broker_public_key_fingerprint, bundle_id, snapshot_hash,
                   reconciliation_snapshot_id, reconciliation_artifact_hash,
                   reconciliation_receipt_id,
                   reconciliation_public_key_fingerprint,
                   reconciliation_signature_ed25519
            FROM rt_reconciliation_snapshots AS snapshots
            JOIN rt_reconciliation_snapshot_lineage AS lineage USING(snapshot_id)
            WHERE snapshot_id = ?
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
    assert row[11].startswith("bootstrap-reconciliation-v1-")
    assert row[12] == "f6" * 32
    assert row[13] == "bevr-v2-" + "a7" * 32
    assert row[14] == "b8" * 32
    assert row[15] == "c2lnbmF0dXJl"


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
