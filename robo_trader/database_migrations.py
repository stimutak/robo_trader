"""Ordered, component-scoped schema migrations for exact paper state.

This deliberately does not reuse the legacy ``schema_migrations`` integer.
That table is owned by the old multiuser migration and an unrelated high
version would incorrectly make that migration appear complete.
"""

from __future__ import annotations

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
    """Fail closed if a partial migration could be mistaken for readiness."""

    required = {
        "administrator_actions",
        "paper_state_bootstraps",
        "paper_account_settlement_state",
        "paper_position_settlement_state",
    }
    cursor = await connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    tables = {str(row[0]) for row in await cursor.fetchall()}
    if not required.issubset(tables):
        raise RuntimeError("exact-state schema is incomplete")
    if "origin_bootstrap_id" not in await _columns(connection, "paper_account_settlement_state"):
        raise RuntimeError("exact account bootstrap lineage is missing")
    if "origin_bootstrap_id" not in await _columns(connection, "paper_position_settlement_state"):
        raise RuntimeError("exact position bootstrap lineage is missing")
