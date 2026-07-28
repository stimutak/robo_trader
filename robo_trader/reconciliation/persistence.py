"""Append-only persistence for broker reconciliation evidence and operator notes."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import aiosqlite

from robo_trader.reconciliation_migrations import (
    apply_reconciliation_migrations,
    assert_reconciliation_schema,
)

from .domain import (
    DOMAIN_SCHEMA_VERSION,
    NormalizedBrokerSnapshot,
    ReconciliationDomainError,
    _timestamp,
    canonical_json,
    canonical_timestamp,
    fingerprint,
)
from .policy import ReconciliationDifference, ReconciliationVerdict

_OPERATOR_ID = re.compile(r"^[A-Za-z0-9._@:-]{1,64}$")
_EVIDENCE_REFERENCE = re.compile(r"^[A-Za-z0-9._:/-]{1,256}$")
_ACCOUNT_FRAGMENT = re.compile(r"(?:DU|U)\d{4,}", re.IGNORECASE)


class ReconciliationPersistenceError(ReconciliationDomainError):
    """Durable reconciliation evidence could not be safely recorded."""


class OperatorResolutionKind(str, Enum):
    """Non-authorizing annotations; none changes or removes prior evidence."""

    ACKNOWLEDGED = "acknowledged"
    EXTERNAL_REMEDIATION_RECORDED = "external_remediation_recorded"
    INVESTIGATION_NOTE = "investigation_note"


@dataclass(frozen=True, slots=True)
class PersistedReconciliation:
    run_id: str
    snapshot_id: str
    verdict_id: str
    difference_ids: tuple[str, ...]
    entry_eligible: bool


@dataclass(frozen=True, slots=True)
class OperatorResolutionEvent:
    resolution_id: str
    run_id: str
    difference_id: str
    resolution_kind: OperatorResolutionKind
    operator_id: str
    reason: str
    evidence_reference: str | None
    created_at: datetime
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def canonical_dict(self) -> dict[str, object]:
        return {
            "created_at": canonical_timestamp(self.created_at),
            "difference_id": self.difference_id,
            "evidence_reference": self.evidence_reference,
            "operator_id": self.operator_id,
            "reason": self.reason,
            "resolution_kind": self.resolution_kind.value,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
        }


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_text(
    value: object,
    field_name: str,
    *,
    pattern: re.Pattern[str],
) -> str:
    if not isinstance(value, str) or value != value.strip() or not pattern.fullmatch(value):
        raise ReconciliationPersistenceError(f"{field_name} is malformed")
    if _ACCOUNT_FRAGMENT.search(value):
        raise ReconciliationPersistenceError(f"{field_name} contains raw account identity")
    return value


class ReconciliationPersistence:
    """Own a dedicated SQLite connection per atomic reconciliation append."""

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path) or not database_path.is_absolute():
            raise ReconciliationPersistenceError("reconciliation database path must be absolute")
        self._database_path = database_path

    async def _connect(self) -> aiosqlite.Connection:
        connection = await aiosqlite.connect(self._database_path, isolation_level=None)
        await connection.execute("PRAGMA foreign_keys = ON")
        foreign_keys = await connection.execute("PRAGMA foreign_keys")
        if await foreign_keys.fetchone() != (1,):
            await connection.close()
            raise ReconciliationPersistenceError(
                "reconciliation database cannot enforce foreign keys"
            )
        return connection

    async def initialize(self) -> None:
        """Register and validate this component without touching legacy rows."""

        connection = await self._connect()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await apply_reconciliation_migrations(connection)
            await assert_reconciliation_schema(connection)
            await connection.execute("COMMIT")
        except BaseException:
            await connection.execute("ROLLBACK")
            raise
        finally:
            await connection.close()

    async def append_reconciliation(
        self,
        *,
        trigger_type: str,
        snapshot: NormalizedBrokerSnapshot,
        verdict: ReconciliationVerdict,
        started_at: datetime,
        completed_at: datetime,
    ) -> PersistedReconciliation:
        """Atomically append one snapshot, verdict, and all difference rows."""

        if type(snapshot) is not NormalizedBrokerSnapshot:
            raise ReconciliationPersistenceError("normalized broker snapshot is required")
        if type(verdict) is not ReconciliationVerdict:
            raise ReconciliationPersistenceError("normalized reconciliation verdict is required")
        if verdict.broker_snapshot_id != snapshot.snapshot_id:
            raise ReconciliationPersistenceError("verdict does not bind the broker snapshot")
        started = _timestamp(started_at, "reconciliation started_at")
        completed = _timestamp(completed_at, "reconciliation completed_at")
        if completed < started or verdict.checked_at < started or verdict.checked_at > completed:
            raise ReconciliationPersistenceError("reconciliation run chronology is invalid")
        allowed_triggers = {
            "startup",
            "reconnect",
            "periodic",
            "before_live",
            "ambiguous_order",
        }
        if trigger_type not in allowed_triggers:
            raise ReconciliationPersistenceError("reconciliation trigger is invalid")

        snapshot_payload = snapshot.canonical_payload()
        verdict_payload = verdict.canonical_payload()
        run_id = fingerprint(
            "reconciliation-run-v1",
            {
                "completed_at": canonical_timestamp(completed),
                "started_at": canonical_timestamp(started),
                "trigger_type": trigger_type,
                "verdict_id": verdict.verdict_id,
            },
        )
        difference_rows: list[tuple[str, ReconciliationDifference, str]] = []
        for ordinal, difference in enumerate(verdict.differences):
            payload = canonical_json(difference.canonical_dict())
            difference_id = fingerprint(
                "reconciliation-difference-v1",
                {"ordinal": ordinal, "payload": difference.canonical_dict(), "run_id": run_id},
            )
            difference_rows.append((difference_id, difference, payload))

        connection = await self._connect()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await assert_reconciliation_schema(connection)
            existing = await connection.execute(
                """
                SELECT payload_sha256, payload_json
                FROM rt_reconciliation_snapshots WHERE snapshot_id = ?
                """,
                (snapshot.snapshot_id,),
            )
            existing_row = await existing.fetchone()
            if existing_row is None:
                await connection.execute(
                    """
                    INSERT INTO rt_reconciliation_snapshots(
                        snapshot_id, schema_version, account_scope, observed_from,
                        observed_through, retrieved_at, complete, payload_json,
                        payload_sha256, persisted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.schema_version,
                        snapshot.account.account_scope,
                        canonical_timestamp(snapshot.observed_from),
                        canonical_timestamp(snapshot.observed_through),
                        canonical_timestamp(snapshot.retrieved_at),
                        int(snapshot.completeness.complete),
                        snapshot_payload,
                        _sha256(snapshot_payload),
                        canonical_timestamp(completed),
                    ),
                )
            elif tuple(existing_row) != (_sha256(snapshot_payload), snapshot_payload):
                raise ReconciliationPersistenceError(
                    "stored snapshot identity has conflicting evidence"
                )

            entry_eligible = not verdict.quarantine_required
            existing_run = await connection.execute(
                """
                SELECT snapshot_id, trigger_type, verdict_id, verdict_payload_json,
                       verdict_sha256, entry_eligible
                FROM rt_reconciliation_runs WHERE run_id = ?
                """,
                (run_id,),
            )
            existing_run_row = await existing_run.fetchone()
            if existing_run_row is not None:
                expected_run_row = (
                    snapshot.snapshot_id,
                    trigger_type,
                    verdict.verdict_id,
                    verdict_payload,
                    _sha256(verdict_payload),
                    int(entry_eligible),
                )
                if tuple(existing_run_row) != expected_run_row:
                    raise ReconciliationPersistenceError(
                        "stored reconciliation run has conflicting evidence"
                    )
                existing_differences = await connection.execute(
                    """
                    SELECT difference_id FROM rt_reconciliation_differences
                    WHERE run_id = ? ORDER BY ordinal
                    """,
                    (run_id,),
                )
                existing_ids = tuple(str(row[0]) for row in await existing_differences.fetchall())
                expected_ids = tuple(row[0] for row in difference_rows)
                if existing_ids != expected_ids:
                    raise ReconciliationPersistenceError(
                        "stored reconciliation differences are incomplete"
                    )
                await connection.execute("COMMIT")
                return PersistedReconciliation(
                    run_id=run_id,
                    snapshot_id=snapshot.snapshot_id,
                    verdict_id=verdict.verdict_id,
                    difference_ids=existing_ids,
                    entry_eligible=entry_eligible,
                )
            await connection.execute(
                """
                INSERT INTO rt_reconciliation_runs(
                    run_id, schema_version, trigger_type, snapshot_id, verdict_id,
                    expected_account_scope, started_at, completed_at, status,
                    evidence_fresh, comparison_complete, quarantine_required,
                    entry_eligible, coverage_json, verdict_payload_json, verdict_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    verdict.schema_version,
                    trigger_type,
                    snapshot.snapshot_id,
                    verdict.verdict_id,
                    verdict.expected_account_scope,
                    canonical_timestamp(started),
                    canonical_timestamp(completed),
                    verdict.status.value,
                    int(verdict.evidence_fresh),
                    int(verdict.comparison_complete),
                    int(verdict.quarantine_required),
                    int(entry_eligible),
                    canonical_json(verdict.coverage.canonical_dict()),
                    verdict_payload,
                    _sha256(verdict_payload),
                ),
            )
            for ordinal, (difference_id, difference, payload) in enumerate(difference_rows):
                await connection.execute(
                    """
                    INSERT INTO rt_reconciliation_differences(
                        difference_id, run_id, ordinal, kind, materiality, reason_code,
                        subject, evidence_ids_json, payload_json, payload_sha256, persisted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        difference_id,
                        run_id,
                        ordinal,
                        difference.kind.value,
                        difference.materiality.value,
                        difference.reason_code,
                        difference.subject,
                        canonical_json(list(difference.evidence_ids)),
                        payload,
                        _sha256(payload),
                        canonical_timestamp(completed),
                    ),
                )
            await connection.execute("COMMIT")
        except BaseException:
            await connection.execute("ROLLBACK")
            raise
        finally:
            await connection.close()
        return PersistedReconciliation(
            run_id=run_id,
            snapshot_id=snapshot.snapshot_id,
            verdict_id=verdict.verdict_id,
            difference_ids=tuple(row[0] for row in difference_rows),
            entry_eligible=entry_eligible,
        )

    async def append_operator_resolution(
        self,
        *,
        run_id: str,
        difference_id: str,
        resolution_kind: OperatorResolutionKind,
        operator_id: str,
        reason: str,
        created_at: datetime,
        evidence_reference: str | None = None,
    ) -> OperatorResolutionEvent:
        """Append an operator note; it never changes the recorded run or eligibility."""

        if type(resolution_kind) is not OperatorResolutionKind:
            raise ReconciliationPersistenceError("operator resolution kind is invalid")
        operator = _strict_text(operator_id, "operator_id", pattern=_OPERATOR_ID)
        if not isinstance(reason, str) or reason != reason.strip() or len(reason) < 10:
            raise ReconciliationPersistenceError("operator resolution reason is too short")
        if _ACCOUNT_FRAGMENT.search(reason):
            raise ReconciliationPersistenceError(
                "operator resolution reason contains raw account identity"
            )
        evidence = None
        if evidence_reference is not None:
            evidence = _strict_text(
                evidence_reference,
                "evidence_reference",
                pattern=_EVIDENCE_REFERENCE,
            )
        created = _timestamp(created_at, "operator resolution created_at")
        payload = {
            "created_at": canonical_timestamp(created),
            "difference_id": difference_id,
            "evidence_reference": evidence,
            "operator_id": operator,
            "reason": reason,
            "resolution_kind": resolution_kind.value,
            "run_id": run_id,
            "schema_version": DOMAIN_SCHEMA_VERSION,
        }
        resolution_id = fingerprint("reconciliation-resolution-v1", payload)
        event = OperatorResolutionEvent(
            resolution_id=resolution_id,
            run_id=run_id,
            difference_id=difference_id,
            resolution_kind=resolution_kind,
            operator_id=operator,
            reason=reason,
            evidence_reference=evidence,
            created_at=created,
        )
        connection = await self._connect()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await assert_reconciliation_schema(connection)
            target = await connection.execute(
                """
                SELECT 1 FROM rt_reconciliation_differences
                WHERE difference_id = ? AND run_id = ?
                """,
                (difference_id, run_id),
            )
            if await target.fetchone() is None:
                raise ReconciliationPersistenceError(
                    "operator resolution target does not exist in the run"
                )
            await connection.execute(
                """
                INSERT INTO rt_reconciliation_operator_resolutions(
                    resolution_id, schema_version, run_id, difference_id,
                    resolution_kind, operator_id, reason, evidence_reference, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.resolution_id,
                    event.schema_version,
                    event.run_id,
                    event.difference_id,
                    event.resolution_kind.value,
                    event.operator_id,
                    event.reason,
                    event.evidence_reference,
                    canonical_timestamp(event.created_at),
                ),
            )
            await connection.execute("COMMIT")
        except BaseException:
            await connection.execute("ROLLBACK")
            raise
        finally:
            await connection.close()
        return event
