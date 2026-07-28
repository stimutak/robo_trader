"""Fixture-only migration for the dormant PR4A FIFO accounting schema.

The deliberately narrow entry point refuses ordinary database filenames and
is not imported by any runtime database module.  A later PR must provide a
separately reviewed, backup-bound production migration before this schema can
be applied to an operational ledger.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

FIFO_ACCOUNTING_COMPONENT = "fifo_accounting"
FIFO_ACCOUNTING_SCHEMA_VERSION = 1
FIXTURE_DATABASE_SUFFIX = ".fifo-fixture.sqlite3"


class FifoFixtureMigrationError(RuntimeError):
    """A fixture database cannot safely accept the FIFO schema."""


def _positive_decimal_check(column: str) -> str:
    return f"""
        {column} NOT GLOB '*[^0-9.]*'
        AND length({column}) - length(replace({column}, '.', '')) <= 1
        AND {column} NOT LIKE '.%'
        AND {column} NOT LIKE '%.'
        AND {column} <> '0'
        AND ({column} NOT LIKE '0%' OR {column} LIKE '0.%')
        AND (instr({column}, '.') = 0 OR substr({column}, -1) <> '0')
    """


def _signed_decimal_check(column: str) -> str:
    positive = _positive_decimal_check(column)
    negative = _positive_decimal_check(f"substr({column}, 2)")
    return f"""
        {column} = '0'
        OR ({positive})
        OR (substr({column}, 1, 1) = '-' AND ({negative}))
    """


_TABLE_SQL = {
    "fifo_schema_migrations": """
        CREATE TABLE fifo_schema_migrations (
            component TEXT NOT NULL,
            version INTEGER NOT NULL CHECK (version > 0),
            description TEXT NOT NULL CHECK (length(trim(description)) > 0),
            applied_at TEXT NOT NULL CHECK (applied_at LIKE '%Z'),
            PRIMARY KEY(component, version)
        )
    """,
    "fifo_accounting_epochs": """
        CREATE TABLE fifo_accounting_epochs (
            epoch_id TEXT PRIMARY KEY CHECK (
                length(epoch_id) = 39 AND substr(epoch_id, 1, 7) = 'fepoch-'
                AND substr(epoch_id, 8) NOT GLOB '*[^0-9a-f]*'
            ),
            schema_version INTEGER NOT NULL CHECK (schema_version = 1),
            execution_domain_scope TEXT NOT NULL CHECK (length(trim(execution_domain_scope)) > 0),
            account_scope TEXT NOT NULL CHECK (length(trim(account_scope)) > 0),
            portfolio_id TEXT NOT NULL CHECK (length(trim(portfolio_id)) > 0),
            origin_kind TEXT NOT NULL CHECK (
                origin_kind IN ('EMPTY_LEDGER', 'LEGACY_AGGREGATE_OPENING_BALANCE')
            ),
            source_fingerprint TEXT NOT NULL UNIQUE CHECK (
                length(source_fingerprint) = 64 AND source_fingerprint = lower(source_fingerprint)
                AND source_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            effective_at TEXT NOT NULL CHECK (effective_at LIKE '%Z'),
            created_at TEXT NOT NULL CHECK (created_at LIKE '%Z'),
            UNIQUE(execution_domain_scope, account_scope, portfolio_id)
        )
    """,
    "fifo_fills": f"""
        CREATE TABLE fifo_fills (
            fill_id TEXT PRIMARY KEY CHECK (
                length(fill_id) = 38 AND substr(fill_id, 1, 6) = 'ffill-'
                AND substr(fill_id, 7) NOT GLOB '*[^0-9a-f]*'
            ),
            epoch_id TEXT NOT NULL,
            event_sequence INTEGER NOT NULL CHECK (event_sequence > 0),
            execution_id TEXT NOT NULL CHECK (length(trim(execution_id)) > 0),
            idempotency_key TEXT NOT NULL CHECK (length(trim(idempotency_key)) > 0),
            con_id INTEGER NOT NULL CHECK (con_id > 0),
            symbol TEXT NOT NULL CHECK (length(symbol) BETWEEN 1 AND 32),
            side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
            quantity_text TEXT NOT NULL CHECK (
                length(quantity_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('quantity_text')})
            ),
            price_text TEXT NOT NULL CHECK (
                length(price_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('price_text')})
            ),
            occurred_at TEXT NOT NULL CHECK (occurred_at LIKE '%Z'),
            recorded_at TEXT NOT NULL CHECK (recorded_at LIKE '%Z'),
            payload_fingerprint TEXT NOT NULL CHECK (
                length(payload_fingerprint) = 64 AND payload_fingerprint = lower(payload_fingerprint)
                AND payload_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            UNIQUE(epoch_id, event_sequence),
            UNIQUE(epoch_id, execution_id),
            UNIQUE(epoch_id, idempotency_key),
            UNIQUE(epoch_id, fill_id),
            FOREIGN KEY(epoch_id) REFERENCES fifo_accounting_epochs(epoch_id)
        )
    """,
    "fifo_commissions": """
        CREATE TABLE fifo_commissions (
            commission_id TEXT PRIMARY KEY CHECK (
                length(commission_id) = 38 AND substr(commission_id, 1, 6) = 'fcomm-'
                AND substr(commission_id, 7) NOT GLOB '*[^0-9a-f]*'
            ),
            epoch_id TEXT NOT NULL,
            fill_id TEXT NOT NULL UNIQUE,
            amount_minor INTEGER NOT NULL CHECK (
                amount_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            currency TEXT NOT NULL CHECK (currency = 'USD'),
            minor_unit_exponent INTEGER NOT NULL CHECK (minor_unit_exponent = 2),
            recorded_at TEXT NOT NULL CHECK (recorded_at LIKE '%Z'),
            UNIQUE(epoch_id, commission_id),
            FOREIGN KEY(epoch_id, fill_id) REFERENCES fifo_fills(epoch_id, fill_id)
        )
    """,
    "fifo_lot_openings": f"""
        CREATE TABLE fifo_lot_openings (
            lot_id TEXT PRIMARY KEY CHECK (
                length(lot_id) = 37 AND substr(lot_id, 1, 5) = 'flot-'
                AND substr(lot_id, 6) NOT GLOB '*[^0-9a-f]*'
            ),
            epoch_id TEXT NOT NULL,
            opening_fill_id TEXT NOT NULL,
            lot_ordinal INTEGER NOT NULL CHECK (lot_ordinal >= 0),
            con_id INTEGER NOT NULL CHECK (con_id > 0),
            symbol TEXT NOT NULL CHECK (length(symbol) BETWEEN 1 AND 32),
            direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
            opened_quantity_text TEXT NOT NULL CHECK (
                length(opened_quantity_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('opened_quantity_text')})
            ),
            open_price_text TEXT NOT NULL CHECK (
                length(open_price_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('open_price_text')})
            ),
            opening_commission_minor INTEGER NOT NULL CHECK (
                opening_commission_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            opened_sequence INTEGER NOT NULL CHECK (opened_sequence > 0),
            opened_at TEXT NOT NULL CHECK (opened_at LIKE '%Z'),
            UNIQUE(epoch_id, lot_id),
            UNIQUE(epoch_id, opening_fill_id, lot_ordinal),
            FOREIGN KEY(epoch_id, opening_fill_id)
                REFERENCES fifo_fills(epoch_id, fill_id)
        )
    """,
    "fifo_lot_matches": f"""
        CREATE TABLE fifo_lot_matches (
            match_id TEXT PRIMARY KEY CHECK (
                length(match_id) = 39 AND substr(match_id, 1, 7) = 'fmatch-'
                AND substr(match_id, 8) NOT GLOB '*[^0-9a-f]*'
            ),
            epoch_id TEXT NOT NULL,
            closing_fill_id TEXT NOT NULL,
            opening_lot_id TEXT NOT NULL,
            match_ordinal INTEGER NOT NULL CHECK (match_ordinal >= 0),
            matched_quantity_text TEXT NOT NULL CHECK (
                length(matched_quantity_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('matched_quantity_text')})
            ),
            opening_price_text TEXT NOT NULL CHECK (
                length(opening_price_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('opening_price_text')})
            ),
            closing_price_text TEXT NOT NULL CHECK (
                length(closing_price_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('closing_price_text')})
            ),
            opening_commission_minor INTEGER NOT NULL CHECK (
                opening_commission_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            closing_commission_minor INTEGER NOT NULL CHECK (
                closing_commission_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            gross_pnl_text TEXT NOT NULL CHECK (
                length(gross_pnl_text) BETWEEN 1 AND 96
                AND ({_signed_decimal_check('gross_pnl_text')})
            ),
            realized_pnl_text TEXT NOT NULL CHECK (
                length(realized_pnl_text) BETWEEN 1 AND 96
                AND ({_signed_decimal_check('realized_pnl_text')})
            ),
            matched_at TEXT NOT NULL CHECK (matched_at LIKE '%Z'),
            UNIQUE(epoch_id, closing_fill_id, match_ordinal),
            UNIQUE(epoch_id, match_id),
            FOREIGN KEY(epoch_id, closing_fill_id)
                REFERENCES fifo_fills(epoch_id, fill_id),
            FOREIGN KEY(epoch_id, opening_lot_id)
                REFERENCES fifo_lot_openings(epoch_id, lot_id)
        )
    """,
    "fifo_position_snapshots": f"""
        CREATE TABLE fifo_position_snapshots (
            snapshot_id TEXT PRIMARY KEY CHECK (
                length(snapshot_id) = 38 AND substr(snapshot_id, 1, 6) = 'fsnap-'
                AND substr(snapshot_id, 7) NOT GLOB '*[^0-9a-f]*'
            ),
            epoch_id TEXT NOT NULL,
            source_fill_id TEXT NOT NULL UNIQUE,
            event_sequence INTEGER NOT NULL CHECK (event_sequence > 0),
            con_id INTEGER NOT NULL CHECK (con_id > 0),
            symbol TEXT NOT NULL CHECK (length(symbol) BETWEEN 1 AND 32),
            signed_quantity_text TEXT NOT NULL CHECK (
                length(signed_quantity_text) BETWEEN 1 AND 64
                AND ({_signed_decimal_check('signed_quantity_text')})
            ),
            open_cost_text TEXT CHECK (
                open_cost_text IS NULL OR (
                    length(open_cost_text) BETWEEN 1 AND 96
                    AND ({_positive_decimal_check('open_cost_text')})
                )
            ),
            open_lot_count INTEGER NOT NULL CHECK (open_lot_count >= 0),
            cumulative_realized_pnl_text TEXT NOT NULL CHECK (
                length(cumulative_realized_pnl_text) BETWEEN 1 AND 96
                AND ({_signed_decimal_check('cumulative_realized_pnl_text')})
            ),
            cumulative_commission_minor INTEGER NOT NULL CHECK (
                cumulative_commission_minor BETWEEN -1000000000000000 AND 1000000000000000
            ),
            previous_snapshot_id TEXT,
            previous_state_fingerprint TEXT,
            state_fingerprint TEXT NOT NULL CHECK (
                length(state_fingerprint) = 64 AND state_fingerprint = lower(state_fingerprint)
                AND state_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            created_at TEXT NOT NULL CHECK (created_at LIKE '%Z'),
            UNIQUE(epoch_id, event_sequence),
            UNIQUE(epoch_id, snapshot_id),
            FOREIGN KEY(epoch_id, source_fill_id)
                REFERENCES fifo_fills(epoch_id, fill_id),
            FOREIGN KEY(epoch_id, previous_snapshot_id)
                REFERENCES fifo_position_snapshots(epoch_id, snapshot_id),
            CHECK (
                (previous_snapshot_id IS NULL AND previous_state_fingerprint IS NULL)
                OR
                (previous_snapshot_id IS NOT NULL AND previous_state_fingerprint IS NOT NULL)
            )
        )
    """,
}

_APPEND_ONLY_TABLES = tuple(_TABLE_SQL)


def _trigger_sql(table: str, operation: str) -> str:
    return f"""
        CREATE TRIGGER {table}_no_{operation.lower()}
        BEFORE {operation} ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{table} is append-only');
        END
    """


_TRIGGER_SQL = {
    f"{table}_no_{operation.lower()}": _trigger_sql(table, operation)
    for table in _APPEND_ONLY_TABLES
    for operation in ("UPDATE", "DELETE")
}


def _normalize_sql(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _database_path(connection: sqlite3.Connection) -> Optional[Path]:
    rows = connection.execute("PRAGMA database_list").fetchall()
    main_rows = [row for row in rows if row[1] == "main"]
    if len(main_rows) != 1:
        raise FifoFixtureMigrationError("fixture connection has ambiguous main database")
    raw_path = str(main_rows[0][2])
    return None if raw_path == "" else Path(raw_path).resolve(strict=True)


def _assert_fixture_target(
    connection: sqlite3.Connection,
    expected_path: Optional[Path],
) -> None:
    actual_path = _database_path(connection)
    if expected_path is None:
        if actual_path is not None:
            raise FifoFixtureMigrationError("only an in-memory database may omit expected_path")
        return
    expected = Path(expected_path)
    if expected.name.endswith(FIXTURE_DATABASE_SUFFIX) is False:
        raise FifoFixtureMigrationError(
            f"fixture database name must end with {FIXTURE_DATABASE_SUFFIX}"
        )
    if expected.is_symlink():
        raise FifoFixtureMigrationError("fixture database path must not be a symlink")
    resolved = expected.resolve(strict=True)
    if actual_path != resolved:
        raise FifoFixtureMigrationError("fixture connection does not match expected_path")
    if resolved.stat().st_nlink != 1:
        raise FifoFixtureMigrationError("fixture database must not have hard link aliases")


def assert_fifo_accounting_schema(connection: sqlite3.Connection) -> None:
    """Fail closed unless the complete PR4A fixture schema is exact."""

    if connection.execute("PRAGMA foreign_keys").fetchone() != (1,):
        raise FifoFixtureMigrationError("FIFO accounting requires foreign-key enforcement")

    rows = connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()
    actual_tables = {str(name): _normalize_sql(sql) for name, sql in rows}
    for name, statement in _TABLE_SQL.items():
        if actual_tables.get(name) != _normalize_sql(statement):
            raise FifoFixtureMigrationError(f"FIFO accounting table {name} is malformed")

    rows = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
    ).fetchall()
    actual_triggers = {str(name): _normalize_sql(sql) for name, sql in rows}
    for name, statement in _TRIGGER_SQL.items():
        if actual_triggers.get(name) != _normalize_sql(statement):
            raise FifoFixtureMigrationError(f"FIFO accounting trigger {name} is malformed")

    versions = connection.execute(
        "SELECT version FROM fifo_schema_migrations WHERE component = ? ORDER BY version",
        (FIFO_ACCOUNTING_COMPONENT,),
    ).fetchall()
    if versions != [(FIFO_ACCOUNTING_SCHEMA_VERSION,)]:
        raise FifoFixtureMigrationError("FIFO accounting migration evidence is incomplete")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise FifoFixtureMigrationError("FIFO accounting schema has foreign-key violations")


def migrate_fifo_fixture_database(
    connection: sqlite3.Connection,
    *,
    expected_path: Optional[Path],
) -> None:
    """Apply the sole PR4A migration to an explicitly identified fixture DB.

    This function intentionally rejects normal database filenames.  It is not a
    production migration and must never be pointed at ``trading_data.db``.
    """

    _assert_fixture_target(connection, expected_path)
    if connection.in_transaction:
        raise FifoFixtureMigrationError("fixture migration requires an idle connection")
    connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("PRAGMA foreign_keys").fetchone() != (1,):
        raise FifoFixtureMigrationError("could not enable fixture foreign keys")

    existing = connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','trigger') " "AND name LIKE 'fifo_%'"
    ).fetchall()
    if existing:
        assert_fifo_accounting_schema(connection)
        return

    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in _TABLE_SQL.values():
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO fifo_schema_migrations(component, version, description, applied_at)
            VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            """,
            (
                FIFO_ACCOUNTING_COMPONENT,
                FIFO_ACCOUNTING_SCHEMA_VERSION,
                "dormant exact append-only FIFO accounting foundation",
            ),
        )
        for statement in _TRIGGER_SQL.values():
            connection.execute(statement)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    assert_fifo_accounting_schema(connection)
