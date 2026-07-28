"""Component-scoped, append-only schema for durable reconciliation evidence."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

import aiosqlite

RECONCILIATION_COMPONENT = "broker_reconciliation"
RECONCILIATION_SCHEMA_VERSION = 1

_TABLES = (
    "rt_reconciliation_snapshots",
    "rt_reconciliation_runs",
    "rt_reconciliation_differences",
    "rt_reconciliation_operator_resolutions",
)


async def _migration_v1(connection: aiosqlite.Connection) -> None:
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS rt_reconciliation_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL CHECK (schema_version = 1),
            account_scope TEXT NOT NULL CHECK (
                length(account_scope) = 72
                AND substr(account_scope, 1, 8) = 'acct_v1_'
                AND substr(account_scope, 9) NOT GLOB '*[^0-9a-f]*'
            ),
            account_alias TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL CHECK (
                length(snapshot_hash) = 64 AND snapshot_hash = lower(snapshot_hash)
            ),
            bundle_id TEXT NOT NULL,
            runtime_fingerprint TEXT NOT NULL,
            database_path TEXT NOT NULL,
            database_identity TEXT NOT NULL,
            database_device INTEGER NOT NULL,
            database_inode INTEGER NOT NULL,
            broker_artifact_hash TEXT NOT NULL CHECK (
                length(broker_artifact_hash) = 64
                AND broker_artifact_hash = lower(broker_artifact_hash)
            ),
            broker_receipt_id TEXT NOT NULL,
            broker_public_key_fingerprint TEXT NOT NULL CHECK (
                length(broker_public_key_fingerprint) = 64
                AND broker_public_key_fingerprint = lower(broker_public_key_fingerprint)
            ),
            broker_evidence_expires_at TEXT NOT NULL,
            observed_from TEXT NOT NULL,
            observed_through TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
            payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
            payload_sha256 TEXT NOT NULL CHECK (
                length(payload_sha256) = 64 AND payload_sha256 = lower(payload_sha256)
            ),
            persisted_at TEXT NOT NULL
        )
    """)
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS rt_reconciliation_runs (
            run_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL CHECK (schema_version = 1),
            trigger_type TEXT NOT NULL CHECK (
                trigger_type IN ('startup', 'reconnect', 'periodic',
                                 'before_live', 'ambiguous_order')
            ),
            snapshot_id TEXT NOT NULL,
            verdict_id TEXT NOT NULL,
            expected_account_scope TEXT NOT NULL CHECK (
                length(expected_account_scope) = 72
                AND substr(expected_account_scope, 1, 8) = 'acct_v1_'
                AND substr(expected_account_scope, 9) NOT GLOB '*[^0-9a-f]*'
            ),
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            eligible_until TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('passed', 'degraded', 'quarantined')),
            evidence_fresh INTEGER NOT NULL CHECK (evidence_fresh IN (0, 1)),
            comparison_complete INTEGER NOT NULL CHECK (comparison_complete IN (0, 1)),
            quarantine_required INTEGER NOT NULL CHECK (quarantine_required IN (0, 1)),
            entry_eligible INTEGER NOT NULL CHECK (entry_eligible IN (0, 1)),
            coverage_json TEXT NOT NULL CHECK (json_valid(coverage_json)),
            verdict_payload_json TEXT NOT NULL CHECK (json_valid(verdict_payload_json)),
            verdict_sha256 TEXT NOT NULL CHECK (
                length(verdict_sha256) = 64 AND verdict_sha256 = lower(verdict_sha256)
            ),
            CHECK (completed_at >= started_at),
            CHECK (
                (status = 'quarantined' AND quarantine_required = 1 AND entry_eligible = 0)
                OR
                (status != 'quarantined' AND quarantine_required = 0 AND entry_eligible = 1)
            ),
            FOREIGN KEY(snapshot_id) REFERENCES rt_reconciliation_snapshots(snapshot_id)
        )
    """)
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS rt_reconciliation_differences (
            difference_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            kind TEXT NOT NULL CHECK (
                kind IN ('expected_timing_lag', 'recoverable_missing_event',
                         'duplicate_event', 'account_mismatch', 'quantity_mismatch',
                         'cash_mismatch', 'unknown')
            ),
            materiality TEXT NOT NULL CHECK (
                materiality IN ('informational', 'material', 'unknown')
            ),
            reason_code TEXT NOT NULL,
            subject TEXT NOT NULL,
            evidence_ids_json TEXT NOT NULL CHECK (json_valid(evidence_ids_json)),
            payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
            payload_sha256 TEXT NOT NULL CHECK (
                length(payload_sha256) = 64 AND payload_sha256 = lower(payload_sha256)
            ),
            persisted_at TEXT NOT NULL,
            CHECK (
                (kind = 'expected_timing_lag' AND materiality = 'informational')
                OR (kind = 'unknown' AND materiality = 'unknown')
                OR (kind NOT IN ('expected_timing_lag', 'unknown')
                    AND materiality = 'material')
            ),
            UNIQUE(run_id, ordinal),
            UNIQUE(run_id, difference_id),
            FOREIGN KEY(run_id) REFERENCES rt_reconciliation_runs(run_id)
        )
    """)
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS rt_reconciliation_operator_resolutions (
            resolution_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL CHECK (schema_version = 1),
            run_id TEXT NOT NULL,
            difference_id TEXT NOT NULL,
            resolution_kind TEXT NOT NULL CHECK (
                resolution_kind IN ('acknowledged', 'external_remediation_recorded',
                                    'investigation_note')
            ),
            operator_id TEXT NOT NULL CHECK (length(trim(operator_id)) BETWEEN 1 AND 64),
            reason TEXT NOT NULL CHECK (length(trim(reason)) >= 10),
            evidence_reference TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES rt_reconciliation_runs(run_id),
            FOREIGN KEY(run_id, difference_id)
                REFERENCES rt_reconciliation_differences(run_id, difference_id)
        )
    """)
    for table in _TABLES:
        await connection.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'reconciliation evidence is append-only');
            END
        """)
        await connection.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'reconciliation evidence is append-only');
            END
        """)


_MIGRATIONS: tuple[tuple[int, Callable[[aiosqlite.Connection], Awaitable[None]]], ...] = (
    (1, _migration_v1),
)

_EXPECTED_COLUMNS = {
    "rt_reconciliation_snapshots": {
        "snapshot_id",
        "schema_version",
        "account_scope",
        "account_alias",
        "snapshot_hash",
        "bundle_id",
        "runtime_fingerprint",
        "database_path",
        "database_identity",
        "database_device",
        "database_inode",
        "broker_artifact_hash",
        "broker_receipt_id",
        "broker_public_key_fingerprint",
        "broker_evidence_expires_at",
        "observed_from",
        "observed_through",
        "retrieved_at",
        "complete",
        "payload_json",
        "payload_sha256",
        "persisted_at",
    },
    "rt_reconciliation_runs": {
        "run_id",
        "schema_version",
        "trigger_type",
        "snapshot_id",
        "verdict_id",
        "expected_account_scope",
        "started_at",
        "completed_at",
        "eligible_until",
        "status",
        "evidence_fresh",
        "comparison_complete",
        "quarantine_required",
        "entry_eligible",
        "coverage_json",
        "verdict_payload_json",
        "verdict_sha256",
    },
    "rt_reconciliation_differences": {
        "difference_id",
        "run_id",
        "ordinal",
        "kind",
        "materiality",
        "reason_code",
        "subject",
        "evidence_ids_json",
        "payload_json",
        "payload_sha256",
        "persisted_at",
    },
    "rt_reconciliation_operator_resolutions": {
        "resolution_id",
        "schema_version",
        "run_id",
        "difference_id",
        "resolution_kind",
        "operator_id",
        "reason",
        "evidence_reference",
        "created_at",
    },
}

_REQUIRED_TABLE_SQL = {
    "rt_reconciliation_snapshots": (
        "schema_version integer not null check (schema_version = 1)",
        "length(account_scope) = 72",
        "length(snapshot_hash) = 64 and snapshot_hash = lower(snapshot_hash)",
        "length(broker_artifact_hash) = 64 and broker_artifact_hash = "
        "lower(broker_artifact_hash)",
        "check (json_valid(payload_json))",
        "length(payload_sha256) = 64 and payload_sha256 = lower(payload_sha256)",
    ),
    "rt_reconciliation_runs": (
        "trigger_type in ('startup', 'reconnect', 'periodic', 'before_live', " "'ambiguous_order')",
        "status in ('passed', 'degraded', 'quarantined')",
        "eligible_until text not null",
        "status = 'quarantined' and quarantine_required = 1 and entry_eligible = 0",
        "foreign key(snapshot_id) references rt_reconciliation_snapshots(snapshot_id)",
    ),
    "rt_reconciliation_differences": (
        "kind = 'expected_timing_lag' and materiality = 'informational'",
        "kind = 'unknown' and materiality = 'unknown'",
        "unique(run_id, ordinal)",
        "unique(run_id, difference_id)",
        "foreign key(run_id) references rt_reconciliation_runs(run_id)",
    ),
    "rt_reconciliation_operator_resolutions": (
        "resolution_kind in ('acknowledged', 'external_remediation_recorded', "
        "'investigation_note')",
        "length(trim(reason)) >= 10",
        "foreign key(run_id) references rt_reconciliation_runs(run_id)",
        "foreign key(run_id, difference_id) references "
        "rt_reconciliation_differences(run_id, difference_id)",
    ),
}


def _normalized_sql(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _canonical_schema_sql(value: object) -> str:
    return _normalized_sql(value).replace("if not exists ", "")


async def _table_columns(connection: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await connection.execute(f"PRAGMA table_info({table})")
    return {str(row[1]) for row in await cursor.fetchall()}


async def apply_reconciliation_migrations(connection: aiosqlite.Connection) -> None:
    """Apply missing reconciliation migrations in the caller-owned transaction."""

    await connection.execute("PRAGMA foreign_keys = ON")
    foreign_keys = await connection.execute("PRAGMA foreign_keys")
    if await foreign_keys.fetchone() != (1,):
        raise RuntimeError("reconciliation schema requires foreign-key enforcement")
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS rt_schema_migrations (
            component TEXT NOT NULL,
            version INTEGER NOT NULL,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            PRIMARY KEY(component, version)
        )
    """)
    cursor = await connection.execute(
        "SELECT version FROM rt_schema_migrations WHERE component = ? ORDER BY version",
        (RECONCILIATION_COMPONENT,),
    )
    applied = [int(row[0]) for row in await cursor.fetchall()]
    if any(version < 1 or version > RECONCILIATION_SCHEMA_VERSION for version in applied):
        raise RuntimeError("reconciliation migration version is unknown")
    for version, migration in _MIGRATIONS:
        if version in applied:
            continue
        await migration(connection)
        await connection.execute(
            """
            INSERT INTO rt_schema_migrations(component, version, description, applied_at)
            VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            """,
            (RECONCILIATION_COMPONENT, version, "append-only broker reconciliation evidence"),
        )


async def assert_reconciliation_schema(connection: aiosqlite.Connection) -> None:
    """Fail closed when registered schema evidence or append-only guards drift."""

    foreign_keys = await connection.execute("PRAGMA foreign_keys")
    if await foreign_keys.fetchone() != (1,):
        raise RuntimeError("reconciliation schema requires foreign-key enforcement")
    migrations = await connection.execute(
        "SELECT version FROM rt_schema_migrations WHERE component = ? ORDER BY version",
        (RECONCILIATION_COMPONENT,),
    )
    if [int(row[0]) for row in await migrations.fetchall()] != list(
        range(1, RECONCILIATION_SCHEMA_VERSION + 1)
    ):
        raise RuntimeError("reconciliation migration evidence is incomplete or unknown")
    migration_columns = await connection.execute("PRAGMA table_info(rt_schema_migrations)")
    migration_shape = {
        str(row[1]): (str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in await migration_columns.fetchall()
    }
    if migration_shape != {
        "component": ("TEXT", 1, 1),
        "version": ("INTEGER", 1, 2),
        "description": ("TEXT", 1, 0),
        "applied_at": ("TEXT", 1, 0),
    }:
        raise RuntimeError("shared component migration registry is malformed")
    for table, expected in _EXPECTED_COLUMNS.items():
        if await _table_columns(connection, table) != expected:
            raise RuntimeError(f"reconciliation table {table} is malformed")
    table_rows = await connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
    )
    table_sql = {str(row[0]): _canonical_schema_sql(row[1]) for row in await table_rows.fetchall()}
    for table, fragments in _REQUIRED_TABLE_SQL.items():
        if any(_normalized_sql(fragment) not in table_sql.get(table, "") for fragment in fragments):
            raise RuntimeError(f"reconciliation table {table} constraints are malformed")
    trigger_rows = await connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND tbl_name IN (?, ?, ?, ?)",
        _TABLES,
    )
    triggers = {str(row[0]): _canonical_schema_sql(row[1]) for row in await trigger_rows.fetchall()}
    expected_names = {
        f"{table}_{suffix}" for table in _TABLES for suffix in ("no_update", "no_delete")
    }
    if set(triggers) != expected_names:
        raise RuntimeError("reconciliation append-only trigger set is malformed")
    for table in _TABLES:
        for suffix, operation in (("no_update", "update"), ("no_delete", "delete")):
            name = f"{table}_{suffix}"
            expected_trigger = _normalized_sql(f"""
                CREATE TRIGGER {name}
                BEFORE {operation.upper()} ON {table}
                BEGIN
                    SELECT RAISE(ABORT, 'reconciliation evidence is append-only');
                END
                """)
            if triggers[name] != expected_trigger:
                raise RuntimeError(f"reconciliation trigger {name} is malformed")
    violations = await connection.execute("PRAGMA foreign_key_check")
    if await violations.fetchone() is not None:
        raise RuntimeError("reconciliation schema contains foreign-key violations")
