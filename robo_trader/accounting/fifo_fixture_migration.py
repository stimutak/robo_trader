"""Fixture-only migration for the dormant PR4A FIFO accounting schema.

The deliberately narrow entry point refuses ordinary database filenames and
is not imported by any runtime database module.  A later PR must provide a
separately reviewed, backup-bound production migration before this schema can
be applied to an operational ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

FIFO_ACCOUNTING_COMPONENT = "fifo_accounting"
FIFO_ACCOUNTING_SCHEMA_VERSION = 2
FIFO_ACCOUNTING_MIGRATIONS = (
    (1, "dormant exact append-only FIFO accounting foundation"),
    (2, "sealed legacy aggregate opening balances"),
)
FIFO_ACCOUNTING_SCHEMA_VERSIONS = tuple(version for version, _ in FIFO_ACCOUNTING_MIGRATIONS)
FIXTURE_DATABASE_SUFFIX = ".fifo-fixture.sqlite3"


class FifoFixtureMigrationError(RuntimeError):
    """A fixture database cannot safely accept the FIFO schema."""


def _legacy_opening_manifest_hash(rows: Sequence[Sequence[object]]) -> str:
    """Bind one sealed candidate to its complete ordered opening-lot set."""

    payload = [list(row) for row in rows]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


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
            version INTEGER NOT NULL CHECK (typeof(version) = 'integer' AND version > 0),
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
            schema_version INTEGER NOT NULL CHECK (
                typeof(schema_version) = 'integer' AND schema_version = 1
            ),
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
    "fifo_legacy_bootstrap_lineage": """
        CREATE TABLE fifo_legacy_bootstrap_lineage (
            epoch_id TEXT PRIMARY KEY,
            bootstrap_id TEXT NOT NULL UNIQUE CHECK (
                length(bootstrap_id) = 38 AND substr(bootstrap_id, 1, 6) = 'pboot-'
                AND substr(bootstrap_id, 7) NOT GLOB '*[^0-9a-f]*'
            ),
            candidate_fingerprint TEXT NOT NULL UNIQUE CHECK (
                length(candidate_fingerprint) = 64
                AND candidate_fingerprint = lower(candidate_fingerprint)
                AND candidate_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            opening_manifest_count INTEGER NOT NULL CHECK (
                typeof(opening_manifest_count) = 'integer' AND opening_manifest_count >= 0
            ),
            opening_manifest_hash TEXT NOT NULL CHECK (
                length(opening_manifest_hash) = 64
                AND opening_manifest_hash = lower(opening_manifest_hash)
                AND opening_manifest_hash NOT GLOB '*[^0-9a-f]*'
            ),
            reconciliation_snapshot_id TEXT NOT NULL CHECK (
                length(trim(reconciliation_snapshot_id)) > 0
            ),
            reconciliation_report_hash TEXT NOT NULL CHECK (
                length(reconciliation_report_hash) = 64
                AND reconciliation_report_hash = lower(reconciliation_report_hash)
                AND reconciliation_report_hash NOT GLOB '*[^0-9a-f]*'
            ),
            broker_snapshot_hash TEXT NOT NULL CHECK (
                length(broker_snapshot_hash) = 64
                AND broker_snapshot_hash = lower(broker_snapshot_hash)
                AND broker_snapshot_hash NOT GLOB '*[^0-9a-f]*'
            ),
            legacy_snapshot_hash TEXT NOT NULL CHECK (
                length(legacy_snapshot_hash) = 64
                AND legacy_snapshot_hash = lower(legacy_snapshot_hash)
                AND legacy_snapshot_hash NOT GLOB '*[^0-9a-f]*'
            ),
            operator_action_id TEXT NOT NULL UNIQUE CHECK (
                length(trim(operator_action_id)) > 0
            ),
            recorded_at TEXT NOT NULL CHECK (recorded_at LIKE '%Z'),
            FOREIGN KEY(epoch_id) REFERENCES fifo_accounting_epochs(epoch_id)
        )
    """,
    "fifo_epoch_account_baselines": f"""
        CREATE TABLE fifo_epoch_account_baselines (
            epoch_id TEXT PRIMARY KEY,
            cash_text TEXT NOT NULL CHECK (
                length(cash_text) BETWEEN 1 AND 96
                AND ({_signed_decimal_check('cash_text')})
            ),
            realized_pnl_text TEXT NOT NULL CHECK (
                length(realized_pnl_text) BETWEEN 1 AND 96
                AND ({_signed_decimal_check('realized_pnl_text')})
            ),
            daily_pnl_text TEXT NOT NULL CHECK (
                length(daily_pnl_text) BETWEEN 1 AND 96
                AND ({_signed_decimal_check('daily_pnl_text')})
            ),
            daily_pnl_baseline_text TEXT NOT NULL CHECK (
                length(daily_pnl_baseline_text) BETWEEN 1 AND 96
                AND ({_signed_decimal_check('daily_pnl_baseline_text')})
            ),
            daily_pnl_date TEXT NOT NULL CHECK (
                length(daily_pnl_date) = 10
                AND daily_pnl_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
            ),
            recorded_at TEXT NOT NULL CHECK (recorded_at LIKE '%Z'),
            FOREIGN KEY(epoch_id) REFERENCES fifo_accounting_epochs(epoch_id)
        )
    """,
    "fifo_opening_balances": f"""
        CREATE TABLE fifo_opening_balances (
            opening_balance_id TEXT PRIMARY KEY CHECK (
                length(opening_balance_id) = 38
                AND substr(opening_balance_id, 1, 6) = 'fobal-'
                AND substr(opening_balance_id, 7) NOT GLOB '*[^0-9a-f]*'
            ),
            epoch_id TEXT NOT NULL,
            con_id INTEGER NOT NULL CHECK (typeof(con_id) = 'integer' AND con_id > 0),
            symbol TEXT NOT NULL CHECK (length(symbol) BETWEEN 1 AND 32),
            direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
            opened_quantity_text TEXT NOT NULL CHECK (
                length(opened_quantity_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('opened_quantity_text')})
            ),
            cost_basis_text TEXT NOT NULL CHECK (
                length(cost_basis_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('cost_basis_text')})
            ),
            mark_price_text TEXT NOT NULL CHECK (
                length(mark_price_text) BETWEEN 1 AND 64
                AND ({_positive_decimal_check('mark_price_text')})
            ),
            mark_observed_at TEXT NOT NULL CHECK (mark_observed_at LIKE '%Z'),
            mark_evidence_fingerprint TEXT NOT NULL CHECK (
                length(mark_evidence_fingerprint) = 64
                AND mark_evidence_fingerprint = lower(mark_evidence_fingerprint)
                AND mark_evidence_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            recorded_at TEXT NOT NULL CHECK (recorded_at LIKE '%Z'),
            UNIQUE(epoch_id, opening_balance_id),
            UNIQUE(epoch_id, con_id),
            UNIQUE(epoch_id, symbol),
            FOREIGN KEY(epoch_id) REFERENCES fifo_accounting_epochs(epoch_id)
        )
    """,
    "fifo_fills": f"""
        CREATE TABLE fifo_fills (
            fill_id TEXT PRIMARY KEY CHECK (
                length(fill_id) = 38 AND substr(fill_id, 1, 6) = 'ffill-'
                AND substr(fill_id, 7) NOT GLOB '*[^0-9a-f]*'
            ),
            epoch_id TEXT NOT NULL,
            event_sequence INTEGER NOT NULL CHECK (
                typeof(event_sequence) = 'integer' AND event_sequence > 0
            ),
            execution_id TEXT NOT NULL CHECK (length(trim(execution_id)) > 0),
            idempotency_key TEXT NOT NULL CHECK (length(trim(idempotency_key)) > 0),
            con_id INTEGER NOT NULL CHECK (typeof(con_id) = 'integer' AND con_id > 0),
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
                typeof(amount_minor) = 'integer'
                AND amount_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            currency TEXT NOT NULL CHECK (currency = 'USD'),
            minor_unit_exponent INTEGER NOT NULL CHECK (
                typeof(minor_unit_exponent) = 'integer' AND minor_unit_exponent = 2
            ),
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
            opening_fill_id TEXT,
            opening_balance_id TEXT,
            lot_ordinal INTEGER NOT NULL CHECK (
                typeof(lot_ordinal) = 'integer' AND lot_ordinal >= 0
            ),
            con_id INTEGER NOT NULL CHECK (typeof(con_id) = 'integer' AND con_id > 0),
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
                typeof(opening_commission_minor) = 'integer'
                AND opening_commission_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            opened_sequence INTEGER NOT NULL CHECK (
                typeof(opened_sequence) = 'integer' AND opened_sequence >= 0
            ),
            opened_at TEXT NOT NULL CHECK (opened_at LIKE '%Z'),
            UNIQUE(epoch_id, lot_id),
            UNIQUE(epoch_id, opening_fill_id, lot_ordinal),
            UNIQUE(epoch_id, opening_balance_id),
            FOREIGN KEY(epoch_id, opening_fill_id)
                REFERENCES fifo_fills(epoch_id, fill_id),
            FOREIGN KEY(epoch_id, opening_balance_id)
                REFERENCES fifo_opening_balances(epoch_id, opening_balance_id),
            CHECK (
                (opening_fill_id IS NOT NULL AND opening_balance_id IS NULL
                    AND opened_sequence > 0)
                OR
                (opening_fill_id IS NULL AND opening_balance_id IS NOT NULL
                    AND lot_ordinal = 0 AND opening_commission_minor = 0
                    AND opened_sequence = 0)
            )
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
            match_ordinal INTEGER NOT NULL CHECK (
                typeof(match_ordinal) = 'integer' AND match_ordinal >= 0
            ),
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
                typeof(opening_commission_minor) = 'integer'
                AND opening_commission_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            closing_commission_minor INTEGER NOT NULL CHECK (
                typeof(closing_commission_minor) = 'integer'
                AND closing_commission_minor BETWEEN -1000000000000 AND 1000000000000
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
            event_sequence INTEGER NOT NULL CHECK (
                typeof(event_sequence) = 'integer' AND event_sequence > 0
            ),
            con_id INTEGER NOT NULL CHECK (typeof(con_id) = 'integer' AND con_id > 0),
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
            open_lot_count INTEGER NOT NULL CHECK (
                typeof(open_lot_count) = 'integer' AND open_lot_count >= 0
            ),
            cumulative_realized_pnl_text TEXT NOT NULL CHECK (
                length(cumulative_realized_pnl_text) BETWEEN 1 AND 96
                AND ({_signed_decimal_check('cumulative_realized_pnl_text')})
            ),
            cumulative_commission_minor INTEGER NOT NULL CHECK (
                typeof(cumulative_commission_minor) = 'integer'
                AND cumulative_commission_minor BETWEEN -1000000000000000 AND 1000000000000000
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


_INSERT_CONFLICT_PREDICATES = {
    "fifo_schema_migrations": "component = NEW.component AND version = NEW.version",
    "fifo_accounting_epochs": """
        epoch_id = NEW.epoch_id
        OR source_fingerprint = NEW.source_fingerprint
        OR (
            execution_domain_scope = NEW.execution_domain_scope
            AND account_scope = NEW.account_scope
            AND portfolio_id = NEW.portfolio_id
        )
    """,
    "fifo_legacy_bootstrap_lineage": "epoch_id = NEW.epoch_id OR bootstrap_id = NEW.bootstrap_id OR candidate_fingerprint = NEW.candidate_fingerprint OR operator_action_id = NEW.operator_action_id",
    "fifo_epoch_account_baselines": "epoch_id = NEW.epoch_id",
    "fifo_opening_balances": """
        opening_balance_id = NEW.opening_balance_id
        OR (epoch_id = NEW.epoch_id AND con_id = NEW.con_id)
        OR (epoch_id = NEW.epoch_id AND symbol = NEW.symbol)
    """,
    "fifo_fills": """
        fill_id = NEW.fill_id
        OR (epoch_id = NEW.epoch_id AND event_sequence = NEW.event_sequence)
        OR (epoch_id = NEW.epoch_id AND execution_id = NEW.execution_id)
        OR (epoch_id = NEW.epoch_id AND idempotency_key = NEW.idempotency_key)
    """,
    "fifo_commissions": "commission_id = NEW.commission_id OR fill_id = NEW.fill_id",
    "fifo_lot_openings": """
        lot_id = NEW.lot_id
        OR (
            epoch_id = NEW.epoch_id
            AND opening_fill_id = NEW.opening_fill_id
            AND lot_ordinal = NEW.lot_ordinal
        )
        OR (epoch_id = NEW.epoch_id AND opening_balance_id = NEW.opening_balance_id)
    """,
    "fifo_lot_matches": """
        match_id = NEW.match_id
        OR (
            epoch_id = NEW.epoch_id
            AND closing_fill_id = NEW.closing_fill_id
            AND match_ordinal = NEW.match_ordinal
        )
    """,
    "fifo_position_snapshots": """
        snapshot_id = NEW.snapshot_id
        OR source_fill_id = NEW.source_fill_id
        OR (epoch_id = NEW.epoch_id AND event_sequence = NEW.event_sequence)
    """,
}


def _insert_conflict_guard_sql(table: str) -> str:
    # Every interpolated identifier and predicate comes from the closed constants above.
    predicate = _INSERT_CONFLICT_PREDICATES[table]
    return "\n".join(
        (
            f"CREATE TRIGGER {table}_no_replace",
            f"BEFORE INSERT ON {table}",
            "WHEN EXISTS (",
            f"SELECT 1 FROM {table}",  # nosec B608
            f"WHERE {predicate}",
            ")",
            "BEGIN",
            f"SELECT RAISE(ABORT, '{table} identity is append-only');",
            "END",
        )
    )


_TRIGGER_SQL = {
    f"{table}_no_{operation.lower()}": _trigger_sql(table, operation)
    for table in _APPEND_ONLY_TABLES
    for operation in ("UPDATE", "DELETE")
}
_TRIGGER_SQL.update(
    {f"{table}_no_replace": _insert_conflict_guard_sql(table) for table in _APPEND_ONLY_TABLES}
)


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


def _trigger_references_fifo_table(sql: object) -> bool:
    if type(sql) is not str:
        return False
    return any(
        re.search(rf"(?<![a-z0-9_]){re.escape(table)}(?![a-z0-9_])", sql, re.IGNORECASE) is not None
        for table in _TABLE_SQL
    )


def _foreign_fifo_triggers(rows: Sequence[Sequence[object]]) -> set[str]:
    return {
        str(name)
        for name, table, sql in rows
        if str(name) not in _TRIGGER_SQL
        and (
            str(name).lower().startswith("fifo_")
            or str(table).lower() in _TABLE_SQL
            or _trigger_references_fifo_table(sql)
        )
    }


def _assert_no_temp_fifo_objects(connection: sqlite3.Connection) -> None:
    shadow_query = "SELECT type, name, tbl_name, sql FROM temp.sqlite_master " "ORDER BY type, name"
    shadows = connection.execute(shadow_query).fetchall()
    if any(
        str(row[1]).lower().startswith("fifo_")
        or str(row[2]).lower() in _TABLE_SQL
        or (str(row[0]) == "trigger" and _trigger_references_fifo_table(row[3]))
        for row in shadows
    ):
        raise FifoFixtureMigrationError("temporary FIFO objects cannot shadow durable main state")


def _pragma_foreign_keys_enabled(connection: sqlite3.Connection) -> bool:
    row = connection.execute("PRAGMA foreign_keys").fetchone()
    return row is not None and type(row[0]) is int and row[0] == 1


def assert_fifo_accounting_schema(
    connection: sqlite3.Connection,
    *,
    allow_other_objects: bool = False,
) -> None:
    """Fail closed unless every FIFO object is exact.

    Fixture callers retain the original closed-world check.  PR4B's reviewed
    bootstrap may validate the same objects inside a legacy ledger containing
    unrelated pre-existing tables.
    """

    _assert_no_temp_fifo_objects(connection)
    if not _pragma_foreign_keys_enabled(connection):
        raise FifoFixtureMigrationError("FIFO accounting requires foreign-key enforcement")

    rows = connection.execute(
        "SELECT name, sql FROM main.sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    actual_tables = {str(name): _normalize_sql(sql) for name, sql in rows}
    missing_tables = set(_TABLE_SQL).difference(actual_tables)
    unexpected_tables = set(actual_tables).difference(_TABLE_SQL)
    if missing_tables:
        raise FifoFixtureMigrationError(
            f"FIFO accounting table {sorted(missing_tables)[0]} is missing"
        )
    if unexpected_tables and not allow_other_objects:
        raise FifoFixtureMigrationError(
            f"FIFO fixture contains unexpected table {sorted(unexpected_tables)[0]}"
        )
    for name, statement in _TABLE_SQL.items():
        if actual_tables.get(name) != _normalize_sql(statement):
            raise FifoFixtureMigrationError(f"FIFO accounting table {name} is malformed")

    rows = connection.execute(
        "SELECT name, tbl_name, sql FROM main.sqlite_master WHERE type = 'trigger'"
    ).fetchall()
    actual_triggers = {str(name): _normalize_sql(sql) for name, _, sql in rows}
    missing_triggers = set(_TRIGGER_SQL).difference(actual_triggers)
    unexpected_triggers = set(actual_triggers).difference(_TRIGGER_SQL)
    if missing_triggers:
        raise FifoFixtureMigrationError(
            f"FIFO accounting trigger {sorted(missing_triggers)[0]} is missing"
        )
    if unexpected_triggers and not allow_other_objects:
        raise FifoFixtureMigrationError(
            f"FIFO fixture contains unexpected trigger {sorted(unexpected_triggers)[0]}"
        )
    protected_foreign_triggers = _foreign_fifo_triggers(rows)
    if protected_foreign_triggers:
        raise FifoFixtureMigrationError("foreign triggers cannot target FIFO accounting tables")
    for name, statement in _TRIGGER_SQL.items():
        if actual_triggers.get(name) != _normalize_sql(statement):
            raise FifoFixtureMigrationError(f"FIFO accounting trigger {name} is malformed")

    unexpected_schema = connection.execute(
        "SELECT type, name, tbl_name FROM main.sqlite_master "
        "WHERE type IN ('view','index') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    if unexpected_schema and not allow_other_objects:
        raise FifoFixtureMigrationError("FIFO fixture contains unexpected schema objects")
    protected_foreign_schema = [
        row
        for row in unexpected_schema
        if str(row[1]).lower().startswith("fifo_") or str(row[2]).lower() in _TABLE_SQL
    ]
    if protected_foreign_schema:
        raise FifoFixtureMigrationError(
            "foreign schema objects cannot target FIFO accounting tables"
        )

    version_rows = connection.execute(
        "SELECT version,description FROM main.fifo_schema_migrations "
        "WHERE component = ? ORDER BY version",
        (FIFO_ACCOUNTING_COMPONENT,),
    ).fetchall()
    migrations = [(row[0], row[1]) for row in version_rows]
    if migrations != list(FIFO_ACCOUNTING_MIGRATIONS) or any(
        type(version) is not int for version, _ in migrations
    ):
        raise FifoFixtureMigrationError("FIFO accounting migration evidence is incomplete")
    if connection.execute("PRAGMA main.foreign_key_check").fetchone() is not None:
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
    _assert_no_temp_fifo_objects(connection)
    if connection.in_transaction:
        raise FifoFixtureMigrationError("fixture migration requires an idle connection")
    connection.execute("PRAGMA foreign_keys = ON")
    if not _pragma_foreign_keys_enabled(connection):
        raise FifoFixtureMigrationError("could not enable fixture foreign keys")

    try:
        connection.execute("BEGIN IMMEDIATE")
        schema_objects = connection.execute(
            "SELECT type, name FROM main.sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        fifo_objects = [row for row in schema_objects if str(row[1]).startswith("fifo_")]
        if fifo_objects:
            assert_fifo_accounting_schema(connection)
            connection.commit()
            return
        if schema_objects:
            raise FifoFixtureMigrationError(
                "fixture migration requires an empty database or an exact FIFO fixture"
            )
        for statement in _TABLE_SQL.values():
            connection.execute(statement)
        connection.executemany(
            """
            INSERT INTO main.fifo_schema_migrations(component, version, description, applied_at)
            VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            """,
            tuple(
                (FIFO_ACCOUNTING_COMPONENT, version, description)
                for version, description in FIFO_ACCOUNTING_MIGRATIONS
            ),
        )
        for statement in _TRIGGER_SQL.values():
            connection.execute(statement)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    assert_fifo_accounting_schema(connection)
