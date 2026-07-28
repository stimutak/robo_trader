"""Ordered, component-scoped schema migrations for exact paper state.

This deliberately does not reuse the legacy ``schema_migrations`` integer.
That table is owned by the old multiuser migration and an unrelated high
version would incorrectly make that migration appear complete.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

import aiosqlite

EXACT_STATE_COMPONENT = "paper_exact_state"
EXACT_STATE_SCHEMA_VERSION = 1


async def _columns(connection: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await connection.execute(f"PRAGMA table_info({table})")
    return {str(row[1]) for row in await cursor.fetchall()}


async def _add_column(
    connection: aiosqlite.Connection,
    table: str,
    definition: str,
) -> None:
    name = definition.split(maxsplit=1)[0]
    if name not in await _columns(connection, table):
        await connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


async def _migration_v1(connection: aiosqlite.Connection) -> None:
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS administrator_actions (
            action_id TEXT PRIMARY KEY,
            action_type TEXT NOT NULL,
            reason TEXT NOT NULL CHECK (length(trim(reason)) >= 10),
            evidence_hash TEXT NOT NULL CHECK (
                length(evidence_hash) = 64 AND evidence_hash = lower(evidence_hash)
            ),
            created_at TEXT NOT NULL
        )
    """)
    await connection.execute("""
        CREATE TRIGGER IF NOT EXISTS administrator_actions_no_update
        BEFORE UPDATE ON administrator_actions
        BEGIN
            SELECT RAISE(ABORT, 'administrator actions are append-only');
        END
    """)
    await connection.execute("""
        CREATE TRIGGER IF NOT EXISTS administrator_actions_no_delete
        BEFORE DELETE ON administrator_actions
        BEGIN
            SELECT RAISE(ABORT, 'administrator actions are append-only');
        END
    """)
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS paper_state_bootstraps (
            bootstrap_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL CHECK (schema_version = 1),
            execution_domain_scope TEXT NOT NULL CHECK (
                execution_domain_scope = 'paper-simulator-v1'
            ),
            account_scope TEXT NOT NULL,
            portfolio_id TEXT NOT NULL,
            reconciliation_snapshot_id TEXT NOT NULL,
            reconciliation_report_hash TEXT NOT NULL,
            broker_snapshot_hash TEXT NOT NULL,
            legacy_snapshot_hash TEXT NOT NULL,
            database_path TEXT NOT NULL,
            database_identity TEXT NOT NULL,
            database_device INTEGER NOT NULL,
            database_inode INTEGER NOT NULL,
            effective_at TEXT NOT NULL,
            candidate_payload_json TEXT NOT NULL,
            candidate_fingerprint TEXT NOT NULL UNIQUE,
            operator_action_id TEXT NOT NULL UNIQUE,
            committed_at TEXT NOT NULL,
            UNIQUE(execution_domain_scope, account_scope, portfolio_id),
            FOREIGN KEY(portfolio_id) REFERENCES portfolios(id),
            FOREIGN KEY(operator_action_id) REFERENCES administrator_actions(action_id)
        )
    """)
    await connection.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_state_bootstraps_no_update
        BEFORE UPDATE ON paper_state_bootstraps
        BEGIN
            SELECT RAISE(ABORT, 'paper state bootstraps are append-only');
        END
    """)
    await connection.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_state_bootstraps_no_delete
        BEFORE DELETE ON paper_state_bootstraps
        BEGIN
            SELECT RAISE(ABORT, 'paper state bootstraps are append-only');
        END
    """)
    await _add_column(
        connection,
        "paper_account_settlement_state",
        "origin_bootstrap_id TEXT REFERENCES paper_state_bootstraps(bootstrap_id)",
    )
    await _add_column(
        connection,
        "paper_position_settlement_state",
        "origin_bootstrap_id TEXT REFERENCES paper_state_bootstraps(bootstrap_id)",
    )


_MIGRATIONS: tuple[tuple[int, Callable[[aiosqlite.Connection], Awaitable[None]]], ...] = (
    (1, _migration_v1),
)

_EXPECTED_COLUMNS = {
    "rt_schema_migrations": {
        "component": ("TEXT", 1),
        "version": ("INTEGER", 2),
        "description": ("TEXT", 0),
        "applied_at": ("TEXT", 0),
    },
    "administrator_actions": {
        "action_id": ("TEXT", 1),
        "action_type": ("TEXT", 0),
        "reason": ("TEXT", 0),
        "evidence_hash": ("TEXT", 0),
        "created_at": ("TEXT", 0),
    },
    "paper_state_bootstraps": {
        "bootstrap_id": ("TEXT", 1),
        "schema_version": ("INTEGER", 0),
        "execution_domain_scope": ("TEXT", 0),
        "account_scope": ("TEXT", 0),
        "portfolio_id": ("TEXT", 0),
        "reconciliation_snapshot_id": ("TEXT", 0),
        "reconciliation_report_hash": ("TEXT", 0),
        "broker_snapshot_hash": ("TEXT", 0),
        "legacy_snapshot_hash": ("TEXT", 0),
        "database_path": ("TEXT", 0),
        "database_identity": ("TEXT", 0),
        "database_device": ("INTEGER", 0),
        "database_inode": ("INTEGER", 0),
        "effective_at": ("TEXT", 0),
        "candidate_payload_json": ("TEXT", 0),
        "candidate_fingerprint": ("TEXT", 0),
        "operator_action_id": ("TEXT", 0),
        "committed_at": ("TEXT", 0),
    },
}

_EXPECTED_FOREIGN_KEYS = {
    "paper_state_bootstraps": {
        ("portfolio_id", "portfolios", "id", "NO ACTION", "NO ACTION", "NONE"),
        (
            "operator_action_id",
            "administrator_actions",
            "action_id",
            "NO ACTION",
            "NO ACTION",
            "NONE",
        ),
    },
    "paper_account_settlement_state": {
        (
            "origin_bootstrap_id",
            "paper_state_bootstraps",
            "bootstrap_id",
            "NO ACTION",
            "NO ACTION",
            "NONE",
        ),
    },
    "paper_position_settlement_state": {
        (
            "origin_bootstrap_id",
            "paper_state_bootstraps",
            "bootstrap_id",
            "NO ACTION",
            "NO ACTION",
            "NONE",
        ),
    },
}

_EXPECTED_UNIQUE_COLUMN_SETS = {
    frozenset({"candidate_fingerprint"}),
    frozenset({"operator_action_id"}),
    frozenset({"execution_domain_scope", "account_scope", "portfolio_id"}),
}

_EXPECTED_TABLE_SQL = {
    "administrator_actions": """
        CREATE TABLE administrator_actions (
            action_id TEXT PRIMARY KEY,
            action_type TEXT NOT NULL,
            reason TEXT NOT NULL CHECK (length(trim(reason)) >= 10),
            evidence_hash TEXT NOT NULL CHECK (
                length(evidence_hash) = 64 AND evidence_hash = lower(evidence_hash)
            ),
            created_at TEXT NOT NULL
        )
    """,
    "paper_state_bootstraps": """
        CREATE TABLE paper_state_bootstraps (
            bootstrap_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL CHECK (schema_version = 1),
            execution_domain_scope TEXT NOT NULL CHECK (
                execution_domain_scope = 'paper-simulator-v1'
            ),
            account_scope TEXT NOT NULL,
            portfolio_id TEXT NOT NULL,
            reconciliation_snapshot_id TEXT NOT NULL,
            reconciliation_report_hash TEXT NOT NULL,
            broker_snapshot_hash TEXT NOT NULL,
            legacy_snapshot_hash TEXT NOT NULL,
            database_path TEXT NOT NULL,
            database_identity TEXT NOT NULL,
            database_device INTEGER NOT NULL,
            database_inode INTEGER NOT NULL,
            effective_at TEXT NOT NULL,
            candidate_payload_json TEXT NOT NULL,
            candidate_fingerprint TEXT NOT NULL UNIQUE,
            operator_action_id TEXT NOT NULL UNIQUE,
            committed_at TEXT NOT NULL,
            UNIQUE(execution_domain_scope, account_scope, portfolio_id),
            FOREIGN KEY(portfolio_id) REFERENCES portfolios(id),
            FOREIGN KEY(operator_action_id) REFERENCES administrator_actions(action_id)
        )
    """,
}

_EXPECTED_TRIGGER_SQL = {
    "administrator_actions_no_update": """
        CREATE TRIGGER administrator_actions_no_update
        BEFORE UPDATE ON administrator_actions
        BEGIN
            SELECT RAISE(ABORT, 'administrator actions are append-only');
        END
    """,
    "administrator_actions_no_delete": """
        CREATE TRIGGER administrator_actions_no_delete
        BEFORE DELETE ON administrator_actions
        BEGIN
            SELECT RAISE(ABORT, 'administrator actions are append-only');
        END
    """,
    "paper_state_bootstraps_no_update": """
        CREATE TRIGGER paper_state_bootstraps_no_update
        BEFORE UPDATE ON paper_state_bootstraps
        BEGIN
            SELECT RAISE(ABORT, 'paper state bootstraps are append-only');
        END
    """,
    "paper_state_bootstraps_no_delete": """
        CREATE TRIGGER paper_state_bootstraps_no_delete
        BEFORE DELETE ON paper_state_bootstraps
        BEGIN
            SELECT RAISE(ABORT, 'paper state bootstraps are append-only');
        END
    """,
}


def _normalized_sql(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


async def _table_info(
    connection: aiosqlite.Connection,
    table: str,
) -> dict[str, tuple[str, int, int]]:
    cursor = await connection.execute(f"PRAGMA table_info({table})")
    return {
        str(row[1]): (str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in await cursor.fetchall()
    }


async def _foreign_keys(
    connection: aiosqlite.Connection,
    table: str,
) -> set[tuple[str, str, str, str, str, str]]:
    cursor = await connection.execute(f"PRAGMA foreign_key_list({table})")
    return {
        (
            str(row[3]),
            str(row[2]),
            str(row[4]),
            str(row[5]),
            str(row[6]),
            str(row[7]),
        )
        for row in await cursor.fetchall()
    }


async def _unique_column_sets(
    connection: aiosqlite.Connection,
    table: str,
) -> set[frozenset[str]]:
    cursor = await connection.execute(f"PRAGMA index_list({table})")
    indexes = await cursor.fetchall()
    result: set[frozenset[str]] = set()
    for index in indexes:
        if int(index[2]) != 1:
            continue
        detail = await connection.execute(f"PRAGMA index_info({index[1]})")
        result.add(frozenset(str(row[2]) for row in await detail.fetchall()))
    return result


async def apply_exact_state_migrations(connection: aiosqlite.Connection) -> None:
    """Apply every missing exact-state migration in one caller-owned transaction."""

    await connection.execute("PRAGMA foreign_keys = ON")
    cursor = await connection.execute("PRAGMA foreign_keys")
    if await cursor.fetchone() != (1,):
        raise RuntimeError("SQLite foreign-key enforcement could not be enabled")
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
        (EXACT_STATE_COMPONENT,),
    )
    applied = {int(row[0]) for row in await cursor.fetchall()}
    for version, migration in _MIGRATIONS:
        if version in applied:
            continue
        await migration(connection)
        await connection.execute(
            """
            INSERT INTO rt_schema_migrations(component, version, description, applied_at)
            VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            """,
            (
                EXACT_STATE_COMPONENT,
                version,
                "sealed exact paper-simulator accounting bootstrap",
            ),
        )


async def assert_exact_state_schema(connection: aiosqlite.Connection) -> None:
    """Prove complete exact-state structure even when version 1 is recorded."""

    foreign_keys = await connection.execute("PRAGMA foreign_keys")
    if await foreign_keys.fetchone() != (1,):
        raise RuntimeError("exact-state schema requires foreign-key enforcement")

    migration_rows = await connection.execute(
        "SELECT version FROM rt_schema_migrations WHERE component = ? ORDER BY version",
        (EXACT_STATE_COMPONENT,),
    )
    if [int(row[0]) for row in await migration_rows.fetchall()] != [EXACT_STATE_SCHEMA_VERSION]:
        raise RuntimeError("exact-state migration version evidence is incomplete or unknown")

    required = {
        "rt_schema_migrations",
        "administrator_actions",
        "paper_state_bootstraps",
        "paper_account_settlement_state",
        "paper_position_settlement_state",
    }
    cursor = await connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    tables = {str(row[0]) for row in await cursor.fetchall()}
    if not required.issubset(tables):
        raise RuntimeError("exact-state schema is incomplete")

    for table, expected in _EXPECTED_COLUMNS.items():
        actual = await _table_info(connection, table)
        if set(actual) != set(expected):
            raise RuntimeError(f"exact-state {table} columns are malformed")
        for column, (declared_type, primary_key) in expected.items():
            actual_type, not_null, actual_primary_key = actual[column]
            if (
                actual_type != declared_type
                or actual_primary_key != primary_key
                or (primary_key == 0 and not_null != 1)
            ):
                raise RuntimeError(f"exact-state {table}.{column} is malformed")

    for table in ("paper_account_settlement_state", "paper_position_settlement_state"):
        lineage = (await _table_info(connection, table)).get("origin_bootstrap_id")
        if lineage is None or lineage[0] != "TEXT":
            raise RuntimeError(f"exact-state {table} bootstrap lineage is malformed")

    for table, expected in _EXPECTED_FOREIGN_KEYS.items():
        if not expected.issubset(await _foreign_keys(connection, table)):
            raise RuntimeError(f"exact-state {table} foreign keys are malformed")

    unique_sets = await _unique_column_sets(connection, "paper_state_bootstraps")
    if not _EXPECTED_UNIQUE_COLUMN_SETS.issubset(unique_sets):
        raise RuntimeError("exact-state bootstrap uniqueness constraints are malformed")

    table_sql_rows = await connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
        ("administrator_actions", "paper_state_bootstraps"),
    )
    table_sql = {str(row[0]): _normalized_sql(row[1]) for row in await table_sql_rows.fetchall()}
    for table, expected_sql in _EXPECTED_TABLE_SQL.items():
        if table_sql.get(table) != _normalized_sql(expected_sql):
            raise RuntimeError(f"exact-state {table} constraints are malformed")

    trigger_rows = await connection.execute(
        """
        SELECT name, sql FROM sqlite_master
        WHERE type = 'trigger' AND tbl_name IN (?, ?)
        """,
        ("administrator_actions", "paper_state_bootstraps"),
    )
    trigger_sql = {str(row[0]): _normalized_sql(row[1]) for row in await trigger_rows.fetchall()}
    if set(trigger_sql) != set(_EXPECTED_TRIGGER_SQL):
        raise RuntimeError("exact-state protected-table trigger set is malformed")
    for name, expected_sql in _EXPECTED_TRIGGER_SQL.items():
        if trigger_sql.get(name) != _normalized_sql(expected_sql):
            raise RuntimeError(f"exact-state trigger {name} is missing or malformed")

    violations = await connection.execute("PRAGMA foreign_key_check")
    if await violations.fetchone() is not None:
        raise RuntimeError("exact-state schema contains foreign-key violations")
