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
EXACT_STATE_SCHEMA_VERSION = 3


async def _columns(connection: aiosqlite.Connection, table: str) -> set[str]:
    quoted = '"' + table.replace('"', '""') + '"'
    cursor = await connection.execute(f"PRAGMA main.table_info({quoted})")
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


async def _migration_v2(connection: aiosqlite.Connection) -> None:
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS exact_bootstrap_evidence_consumptions (
            receipt_id TEXT PRIMARY KEY,
            bootstrap_id TEXT NOT NULL,
            artifact_kind TEXT NOT NULL CHECK (
                artifact_kind IN ('broker_snapshot','reconciliation_report','protective_mark')
            ),
            producer_id TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            runtime_fingerprint TEXT NOT NULL,
            account_scope TEXT NOT NULL,
            consumed_at TEXT NOT NULL,
            FOREIGN KEY(bootstrap_id) REFERENCES paper_state_bootstraps(bootstrap_id)
        )
    """)
    await connection.execute("""
        CREATE TRIGGER IF NOT EXISTS exact_bootstrap_evidence_consumptions_no_update
        BEFORE UPDATE ON exact_bootstrap_evidence_consumptions
        BEGIN
            SELECT RAISE(ABORT, 'bootstrap evidence consumptions are append-only');
        END
    """)
    await connection.execute("""
        CREATE TRIGGER IF NOT EXISTS exact_bootstrap_evidence_consumptions_no_delete
        BEFORE DELETE ON exact_bootstrap_evidence_consumptions
        BEGIN
            SELECT RAISE(ABORT, 'bootstrap evidence consumptions are append-only');
        END
    """)


async def _migration_v3(connection: aiosqlite.Connection) -> None:
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS paper_fifo_settlement_links (
            settlement_id TEXT PRIMARY KEY,
            request_fingerprint TEXT NOT NULL UNIQUE,
            epoch_id TEXT NOT NULL,
            fill_id TEXT NOT NULL UNIQUE,
            event_sequence INTEGER NOT NULL CHECK (
                typeof(event_sequence) = 'integer' AND event_sequence > 0
            ),
            execution_id TEXT NOT NULL,
            commission_minor INTEGER NOT NULL CHECK (
                typeof(commission_minor) = 'integer'
                AND commission_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            commission_currency TEXT NOT NULL CHECK (commission_currency = 'USD'),
            commission_source TEXT NOT NULL CHECK (
                commission_source = 'LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1'
            ),
            fifo_state_fingerprint TEXT NOT NULL CHECK (
                length(fifo_state_fingerprint) = 64
                AND fifo_state_fingerprint = lower(fifo_state_fingerprint)
                AND fifo_state_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL CHECK (committed_at LIKE '%Z'),
            UNIQUE(epoch_id, event_sequence),
            UNIQUE(epoch_id, execution_id),
            FOREIGN KEY(settlement_id)
                REFERENCES paper_reduction_settlements(settlement_id),
            FOREIGN KEY(epoch_id, fill_id)
                REFERENCES fifo_fills(epoch_id, fill_id)
        )
    """)
    await connection.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_fifo_settlement_links_no_update
        BEFORE UPDATE ON paper_fifo_settlement_links
        BEGIN
            SELECT RAISE(ABORT, 'paper FIFO settlement links are append-only');
        END
    """)
    await connection.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_fifo_settlement_links_no_delete
        BEFORE DELETE ON paper_fifo_settlement_links
        BEGIN
            SELECT RAISE(ABORT, 'paper FIFO settlement links are append-only');
        END
    """)


_MIGRATIONS: tuple[tuple[int, Callable[[aiosqlite.Connection], Awaitable[None]]], ...] = (
    (1, _migration_v1),
    (2, _migration_v2),
    (3, _migration_v3),
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
    "exact_bootstrap_evidence_consumptions": {
        "receipt_id": ("TEXT", 1),
        "bootstrap_id": ("TEXT", 0),
        "artifact_kind": ("TEXT", 0),
        "producer_id": ("TEXT", 0),
        "artifact_sha256": ("TEXT", 0),
        "runtime_fingerprint": ("TEXT", 0),
        "account_scope": ("TEXT", 0),
        "consumed_at": ("TEXT", 0),
    },
    "paper_fifo_settlement_links": {
        "settlement_id": ("TEXT", 1),
        "request_fingerprint": ("TEXT", 0),
        "epoch_id": ("TEXT", 0),
        "fill_id": ("TEXT", 0),
        "event_sequence": ("INTEGER", 0),
        "execution_id": ("TEXT", 0),
        "commission_minor": ("INTEGER", 0),
        "commission_currency": ("TEXT", 0),
        "commission_source": ("TEXT", 0),
        "fifo_state_fingerprint": ("TEXT", 0),
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
    "exact_bootstrap_evidence_consumptions": {
        (
            "bootstrap_id",
            "paper_state_bootstraps",
            "bootstrap_id",
            "NO ACTION",
            "NO ACTION",
            "NONE",
        ),
    },
    "paper_fifo_settlement_links": {
        (
            "settlement_id",
            "paper_reduction_settlements",
            "settlement_id",
            "NO ACTION",
            "NO ACTION",
            "NONE",
        ),
        ("epoch_id", "fifo_fills", "epoch_id", "NO ACTION", "NO ACTION", "NONE"),
        ("fill_id", "fifo_fills", "fill_id", "NO ACTION", "NO ACTION", "NONE"),
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
    "exact_bootstrap_evidence_consumptions": """
        CREATE TABLE exact_bootstrap_evidence_consumptions (
            receipt_id TEXT PRIMARY KEY,
            bootstrap_id TEXT NOT NULL,
            artifact_kind TEXT NOT NULL CHECK (
                artifact_kind IN ('broker_snapshot','reconciliation_report','protective_mark')
            ),
            producer_id TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            runtime_fingerprint TEXT NOT NULL,
            account_scope TEXT NOT NULL,
            consumed_at TEXT NOT NULL,
            FOREIGN KEY(bootstrap_id) REFERENCES paper_state_bootstraps(bootstrap_id)
        )
    """,
    "paper_fifo_settlement_links": """
        CREATE TABLE paper_fifo_settlement_links (
            settlement_id TEXT PRIMARY KEY,
            request_fingerprint TEXT NOT NULL UNIQUE,
            epoch_id TEXT NOT NULL,
            fill_id TEXT NOT NULL UNIQUE,
            event_sequence INTEGER NOT NULL CHECK (
                typeof(event_sequence) = 'integer' AND event_sequence > 0
            ),
            execution_id TEXT NOT NULL,
            commission_minor INTEGER NOT NULL CHECK (
                typeof(commission_minor) = 'integer'
                AND commission_minor BETWEEN -1000000000000 AND 1000000000000
            ),
            commission_currency TEXT NOT NULL CHECK (commission_currency = 'USD'),
            commission_source TEXT NOT NULL CHECK (
                commission_source = 'LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1'
            ),
            fifo_state_fingerprint TEXT NOT NULL CHECK (
                length(fifo_state_fingerprint) = 64
                AND fifo_state_fingerprint = lower(fifo_state_fingerprint)
                AND fifo_state_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL CHECK (committed_at LIKE '%Z'),
            UNIQUE(epoch_id, event_sequence),
            UNIQUE(epoch_id, execution_id),
            FOREIGN KEY(settlement_id)
                REFERENCES paper_reduction_settlements(settlement_id),
            FOREIGN KEY(epoch_id, fill_id)
                REFERENCES fifo_fills(epoch_id, fill_id)
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
    "exact_bootstrap_evidence_consumptions_no_update": """
        CREATE TRIGGER exact_bootstrap_evidence_consumptions_no_update
        BEFORE UPDATE ON exact_bootstrap_evidence_consumptions
        BEGIN
            SELECT RAISE(ABORT, 'bootstrap evidence consumptions are append-only');
        END
    """,
    "exact_bootstrap_evidence_consumptions_no_delete": """
        CREATE TRIGGER exact_bootstrap_evidence_consumptions_no_delete
        BEFORE DELETE ON exact_bootstrap_evidence_consumptions
        BEGIN
            SELECT RAISE(ABORT, 'bootstrap evidence consumptions are append-only');
        END
    """,
    "paper_fifo_settlement_links_no_update": """
        CREATE TRIGGER paper_fifo_settlement_links_no_update
        BEFORE UPDATE ON paper_fifo_settlement_links
        BEGIN
            SELECT RAISE(ABORT, 'paper FIFO settlement links are append-only');
        END
    """,
    "paper_fifo_settlement_links_no_delete": """
        CREATE TRIGGER paper_fifo_settlement_links_no_delete
        BEFORE DELETE ON paper_fifo_settlement_links
        BEGIN
            SELECT RAISE(ABORT, 'paper FIFO settlement links are append-only');
        END
    """,
}

_PAPER_SETTLEMENT_HOT_COLUMNS = {
    "trades": {
        "id",
        "portfolio_id",
        "symbol",
        "side",
        "quantity",
        "price",
        "notional",
        "slippage",
        "commission",
        "pnl",
        "timestamp",
    },
    "positions": {
        "id",
        "portfolio_id",
        "symbol",
        "quantity",
        "avg_cost",
        "market_price",
        "timestamp",
    },
    "account": {
        "portfolio_id",
        "cash",
        "equity",
        "daily_pnl",
        "realized_pnl",
        "unrealized_pnl",
        "timestamp",
    },
    "paper_reduction_settlements": {
        "settlement_id",
        "execution_domain_scope",
        "account_scope",
        "portfolio_id",
        "con_id",
        "symbol",
        "reservation_id",
        "claim_id",
        "order_ref",
        "protective_quote_payload",
        "request_fingerprint",
        "request_payload_json",
        "terminal_status",
        "trade_id",
        "database_path",
        "database_identity",
        "database_device",
        "database_inode",
        "committed_at",
        "receipt_fingerprint",
        "schema_version",
    },
    "paper_account_settlement_state": {
        "portfolio_id",
        "cash_text",
        "realized_pnl_text",
        "daily_pnl_text",
        "daily_pnl_baseline_text",
        "daily_pnl_date",
        "updated_at",
        "source_settlement_id",
        "origin_bootstrap_id",
    },
    "paper_position_settlement_state": {
        "portfolio_id",
        "symbol",
        "cost_basis_text",
        "mark_price_text",
        "source_settlement_id",
        "updated_at",
        "origin_bootstrap_id",
    },
    "paper_fifo_settlement_links": set(_EXPECTED_COLUMNS["paper_fifo_settlement_links"]),
}

_PAPER_SETTLEMENT_HOT_COLUMN_TYPES = {
    "trades": {
        "id": ("INTEGER", 1),
        "portfolio_id": ("TEXT", 0),
        "symbol": ("TEXT", 0),
        "side": ("TEXT", 0),
        "quantity": ("INTEGER", 0),
        "price": ("REAL", 0),
        "notional": ("REAL", 0),
        "slippage": ("REAL", 0),
        "commission": ("REAL", 0),
        "pnl": ("REAL", 0),
        "timestamp": ("DATETIME", 0),
    },
    "positions": {
        "id": ("INTEGER", 1),
        "portfolio_id": ("TEXT", 0),
        "symbol": ("TEXT", 0),
        "quantity": ("INTEGER", 0),
        "avg_cost": ("REAL", 0),
        "market_price": ("REAL", 0),
        "timestamp": ("DATETIME", 0),
    },
    "account": {
        "portfolio_id": ("TEXT", 1),
        "cash": ("REAL", 0),
        "equity": ("REAL", 0),
        "daily_pnl": ("REAL", 0),
        "realized_pnl": ("REAL", 0),
        "unrealized_pnl": ("REAL", 0),
        "timestamp": ("DATETIME", 0),
    },
    "paper_reduction_settlements": {
        "settlement_id": ("TEXT", 1),
        "execution_domain_scope": ("TEXT", 0),
        "account_scope": ("TEXT", 0),
        "portfolio_id": ("TEXT", 0),
        "con_id": ("INTEGER", 0),
        "symbol": ("TEXT", 0),
        "reservation_id": ("TEXT", 0),
        "claim_id": ("TEXT", 0),
        "order_ref": ("TEXT", 0),
        "protective_quote_payload": ("TEXT", 0),
        "request_fingerprint": ("TEXT", 0),
        "request_payload_json": ("TEXT", 0),
        "terminal_status": ("TEXT", 0),
        "trade_id": ("INTEGER", 0),
        "database_path": ("TEXT", 0),
        "database_identity": ("TEXT", 0),
        "database_device": ("INTEGER", 0),
        "database_inode": ("INTEGER", 0),
        "committed_at": ("TEXT", 0),
        "receipt_fingerprint": ("TEXT", 0),
        "schema_version": ("INTEGER", 0),
    },
    "paper_account_settlement_state": {
        "portfolio_id": ("TEXT", 1),
        "cash_text": ("TEXT", 0),
        "realized_pnl_text": ("TEXT", 0),
        "daily_pnl_text": ("TEXT", 0),
        "daily_pnl_baseline_text": ("TEXT", 0),
        "daily_pnl_date": ("TEXT", 0),
        "updated_at": ("TEXT", 0),
        "source_settlement_id": ("TEXT", 0),
        "origin_bootstrap_id": ("TEXT", 0),
    },
    "paper_position_settlement_state": {
        "portfolio_id": ("TEXT", 1),
        "symbol": ("TEXT", 2),
        "cost_basis_text": ("TEXT", 0),
        "mark_price_text": ("TEXT", 0),
        "source_settlement_id": ("TEXT", 0),
        "updated_at": ("TEXT", 0),
        "origin_bootstrap_id": ("TEXT", 0),
    },
    "paper_fifo_settlement_links": dict(_EXPECTED_COLUMNS["paper_fifo_settlement_links"]),
}

_PAPER_REDUCTION_SETTLEMENT_TRIGGER_SQL = {
    "paper_reduction_settlements_no_update": """
        CREATE TRIGGER paper_reduction_settlements_no_update
        BEFORE UPDATE ON paper_reduction_settlements
        BEGIN
            SELECT RAISE(ABORT, 'paper reduction settlements are append-only');
        END
    """,
    "paper_reduction_settlements_no_delete": """
        CREATE TRIGGER paper_reduction_settlements_no_delete
        BEFORE DELETE ON paper_reduction_settlements
        BEGIN
            SELECT RAISE(ABORT, 'paper reduction settlements are append-only');
        END
    """,
}

_PAPER_SETTLEMENT_HOT_TRIGGER_SQL = {
    **_PAPER_REDUCTION_SETTLEMENT_TRIGGER_SQL,
    "paper_fifo_settlement_links_no_update": _EXPECTED_TRIGGER_SQL[
        "paper_fifo_settlement_links_no_update"
    ],
    "paper_fifo_settlement_links_no_delete": _EXPECTED_TRIGGER_SQL[
        "paper_fifo_settlement_links_no_delete"
    ],
}

_PAPER_SETTLEMENT_HOT_TABLE_SQL = {
    "trades": """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id TEXT NOT NULL DEFAULT 'default',
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            notional REAL DEFAULT 0,
            slippage REAL DEFAULT 0,
            commission REAL DEFAULT 0,
            pnl REAL DEFAULT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "positions": """
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id TEXT NOT NULL DEFAULT 'default',
            symbol TEXT NOT NULL CHECK (length(symbol) BETWEEN 1 AND 32),
            quantity INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            market_price REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(portfolio_id, symbol)
        )
    """,
    "account": """
        CREATE TABLE account (
            portfolio_id TEXT PRIMARY KEY DEFAULT 'default',
            cash REAL NOT NULL,
            equity REAL NOT NULL,
            daily_pnl REAL DEFAULT 0,
            realized_pnl REAL DEFAULT 0,
            unrealized_pnl REAL DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "paper_reduction_settlements": """
        CREATE TABLE paper_reduction_settlements (
            settlement_id TEXT PRIMARY KEY,
            execution_domain_scope TEXT NOT NULL,
            account_scope TEXT NOT NULL,
            portfolio_id TEXT NOT NULL,
            con_id INTEGER NOT NULL CHECK (con_id > 0),
            symbol TEXT NOT NULL,
            reservation_id TEXT NOT NULL UNIQUE,
            claim_id TEXT NOT NULL UNIQUE,
            order_ref TEXT NOT NULL,
            protective_quote_payload TEXT NOT NULL,
            request_fingerprint TEXT NOT NULL,
            request_payload_json TEXT NOT NULL,
            terminal_status TEXT NOT NULL,
            trade_id INTEGER,
            database_path TEXT NOT NULL,
            database_identity TEXT NOT NULL,
            database_device INTEGER NOT NULL,
            database_inode INTEGER NOT NULL,
            committed_at TEXT NOT NULL,
            receipt_fingerprint TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            UNIQUE(execution_domain_scope, account_scope, order_ref),
            FOREIGN KEY(trade_id) REFERENCES trades(id)
        )
    """,
    "paper_account_settlement_state": """
        CREATE TABLE paper_account_settlement_state (
            portfolio_id TEXT PRIMARY KEY,
            cash_text TEXT NOT NULL,
            realized_pnl_text TEXT NOT NULL,
            daily_pnl_text TEXT NOT NULL,
            daily_pnl_baseline_text TEXT NOT NULL,
            daily_pnl_date TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_settlement_id TEXT,
            origin_bootstrap_id TEXT REFERENCES paper_state_bootstraps(bootstrap_id),
            FOREIGN KEY(source_settlement_id)
                REFERENCES paper_reduction_settlements(settlement_id)
        )
    """,
    "paper_position_settlement_state": """
        CREATE TABLE paper_position_settlement_state (
            portfolio_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            cost_basis_text TEXT NOT NULL,
            mark_price_text TEXT,
            source_settlement_id TEXT,
            updated_at TEXT NOT NULL,
            origin_bootstrap_id TEXT REFERENCES paper_state_bootstraps(bootstrap_id),
            PRIMARY KEY (portfolio_id, symbol),
            FOREIGN KEY(source_settlement_id)
                REFERENCES paper_reduction_settlements(settlement_id)
        )
    """,
    "paper_fifo_settlement_links": _EXPECTED_TABLE_SQL["paper_fifo_settlement_links"],
}

_PAPER_SETTLEMENT_HOT_INDEX_SQL = {
    "idx_positions_portfolio": """
        CREATE INDEX idx_positions_portfolio ON positions (portfolio_id)
    """,
    "idx_trades_portfolio": """
        CREATE INDEX idx_trades_portfolio ON trades (portfolio_id)
    """,
    "idx_trades_portfolio_symbol": """
        CREATE INDEX idx_trades_portfolio_symbol
        ON trades (portfolio_id, symbol, timestamp DESC)
    """,
    "idx_paper_reduction_settlement_scope": """
        CREATE INDEX idx_paper_reduction_settlement_scope
        ON paper_reduction_settlements
           (execution_domain_scope, account_scope, portfolio_id, symbol)
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
    quoted = '"' + table.replace('"', '""') + '"'
    cursor = await connection.execute(f"PRAGMA main.table_info({quoted})")
    return {
        str(row[1]): (str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in await cursor.fetchall()
    }


async def _foreign_keys(
    connection: aiosqlite.Connection,
    table: str,
) -> set[tuple[str, str, str, str, str, str]]:
    quoted = '"' + table.replace('"', '""') + '"'
    cursor = await connection.execute(f"PRAGMA main.foreign_key_list({quoted})")
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
    quoted = '"' + table.replace('"', '""') + '"'
    cursor = await connection.execute(f"PRAGMA main.index_list({quoted})")
    indexes = await cursor.fetchall()
    result: set[frozenset[str]] = set()
    for index in indexes:
        if int(index[2]) != 1:
            continue
        index_name = '"' + str(index[1]).replace('"', '""') + '"'
        detail = await connection.execute(f"PRAGMA main.index_info({index_name})")
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
        "SELECT version FROM main.rt_schema_migrations WHERE component = ? ORDER BY version",
        (EXACT_STATE_COMPONENT,),
    )
    if [int(row[0]) for row in await migration_rows.fetchall()] != list(
        range(1, EXACT_STATE_SCHEMA_VERSION + 1)
    ):
        raise RuntimeError("exact-state migration version evidence is incomplete or unknown")

    required = {
        "rt_schema_migrations",
        "administrator_actions",
        "paper_state_bootstraps",
        "exact_bootstrap_evidence_consumptions",
        "paper_fifo_settlement_links",
        "paper_account_settlement_state",
        "paper_position_settlement_state",
    }
    cursor = await connection.execute("SELECT name FROM main.sqlite_master WHERE type = 'table'")
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

    for table, expected_foreign_keys in _EXPECTED_FOREIGN_KEYS.items():
        if not expected_foreign_keys.issubset(await _foreign_keys(connection, table)):
            raise RuntimeError(f"exact-state {table} foreign keys are malformed")

    unique_sets = await _unique_column_sets(connection, "paper_state_bootstraps")
    if not _EXPECTED_UNIQUE_COLUMN_SETS.issubset(unique_sets):
        raise RuntimeError("exact-state bootstrap uniqueness constraints are malformed")
    fifo_link_unique_sets = await _unique_column_sets(
        connection,
        "paper_fifo_settlement_links",
    )
    expected_fifo_link_unique_sets = {
        frozenset({"request_fingerprint"}),
        frozenset({"fill_id"}),
        frozenset({"epoch_id", "event_sequence"}),
        frozenset({"epoch_id", "execution_id"}),
    }
    if not expected_fifo_link_unique_sets.issubset(fifo_link_unique_sets):
        raise RuntimeError("exact-state FIFO settlement link uniqueness is malformed")

    table_sql_rows = await connection.execute(
        "SELECT name, sql FROM main.sqlite_master WHERE type = 'table'"
    )
    table_sql = {str(row[0]): _normalized_sql(row[1]) for row in await table_sql_rows.fetchall()}
    for table, expected_sql in _EXPECTED_TABLE_SQL.items():
        if table_sql.get(table) != _normalized_sql(expected_sql):
            raise RuntimeError(f"exact-state {table} constraints are malformed")

    trigger_rows = await connection.execute(
        """
        SELECT name, sql FROM main.sqlite_master
        WHERE type = 'trigger' AND tbl_name IN (?, ?, ?, ?)
        """,
        (
            "administrator_actions",
            "paper_state_bootstraps",
            "exact_bootstrap_evidence_consumptions",
            "paper_fifo_settlement_links",
        ),
    )
    trigger_sql = {str(row[0]): _normalized_sql(row[1]) for row in await trigger_rows.fetchall()}
    if set(trigger_sql) != set(_EXPECTED_TRIGGER_SQL):
        raise RuntimeError("exact-state protected-table trigger set is malformed")
    for name, expected_sql in _EXPECTED_TRIGGER_SQL.items():
        if trigger_sql.get(name) != _normalized_sql(expected_sql):
            raise RuntimeError(f"exact-state trigger {name} is missing or malformed")

    violations = await connection.execute("PRAGMA main.foreign_key_check")
    if await violations.fetchone() is not None:
        raise RuntimeError("exact-state schema contains foreign-key violations")


async def assert_paper_settlement_hot_schema(connection: aiosqlite.Connection) -> None:
    """Revalidate every table and trigger touched by terminal settlement.

    The caller holds ``BEGIN IMMEDIATE``.  Main-schema qualification prevents
    temporary objects from redirecting a statement, while this audit rejects
    both temp shadows and any persistent trigger capable of adding side
    effects to the atomic settlement transaction.
    """

    if not connection.in_transaction:
        raise RuntimeError("paper settlement schema audit requires an active transaction")
    await assert_exact_state_schema(connection)

    hot_tables = frozenset(_PAPER_SETTLEMENT_HOT_COLUMNS)
    temporary = await connection.execute(
        "SELECT type,name,tbl_name FROM temp.sqlite_master ORDER BY type,name"
    )
    for object_type, name, table_name in await temporary.fetchall():
        if str(name).lower() in hot_tables or (
            str(object_type).lower() == "trigger" and str(table_name).lower() in hot_tables
        ):
            raise RuntimeError("temporary objects cannot shadow paper settlement state")

    main_tables = await connection.execute("SELECT name FROM main.sqlite_master WHERE type='table'")
    present = {str(row[0]) for row in await main_tables.fetchall()}
    if not hot_tables.issubset(present):
        raise RuntimeError("paper settlement hot schema is incomplete")
    for table, required_columns in _PAPER_SETTLEMENT_HOT_COLUMNS.items():
        actual = await _table_info(connection, table)
        if set(actual) != required_columns:
            raise RuntimeError(f"paper settlement hot table {table} is malformed")
        for column, (expected_type, expected_primary_key) in _PAPER_SETTLEMENT_HOT_COLUMN_TYPES[
            table
        ].items():
            actual_type, _, actual_primary_key = actual[column]
            if actual_type != expected_type or actual_primary_key != expected_primary_key:
                raise RuntimeError(f"paper settlement hot table {table} is malformed")

    table_rows = await connection.execute(
        """
        SELECT name,sql FROM main.sqlite_master
        WHERE type='table' AND lower(name) IN (?,?,?,?,?,?,?)
        ORDER BY name
        """,
        tuple(sorted(hot_tables)),
    )
    table_sql = {str(name): _normalized_sql(sql) for name, sql in await table_rows.fetchall()}
    if set(table_sql) != set(_PAPER_SETTLEMENT_HOT_TABLE_SQL):
        raise RuntimeError("paper settlement hot-table definition set is malformed")
    for name, expected_sql in _PAPER_SETTLEMENT_HOT_TABLE_SQL.items():
        if table_sql.get(name) != _normalized_sql(expected_sql):
            raise RuntimeError(f"paper settlement hot table {name} definition is malformed")

    index_rows = await connection.execute(
        """
        SELECT name,sql FROM main.sqlite_master
        WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%'
          AND lower(tbl_name) IN (?,?,?,?,?,?,?)
        ORDER BY name
        """,
        tuple(sorted(hot_tables)),
    )
    index_sql = {str(name): _normalized_sql(sql) for name, sql in await index_rows.fetchall()}
    if set(index_sql) != set(_PAPER_SETTLEMENT_HOT_INDEX_SQL):
        raise RuntimeError("paper settlement hot-table index set is malformed")
    for name, expected_sql in _PAPER_SETTLEMENT_HOT_INDEX_SQL.items():
        if index_sql.get(name) != _normalized_sql(expected_sql):
            raise RuntimeError(f"paper settlement hot-table index {name} is malformed")

    triggers = await connection.execute(
        """
        SELECT name,sql FROM main.sqlite_master
        WHERE type='trigger' AND lower(tbl_name) IN (?,?,?,?,?,?,?)
        ORDER BY name
        """,
        tuple(sorted(hot_tables)),
    )
    trigger_sql = {str(name): _normalized_sql(sql) for name, sql in await triggers.fetchall()}
    if set(trigger_sql) != set(_PAPER_SETTLEMENT_HOT_TRIGGER_SQL):
        raise RuntimeError("paper settlement hot-table trigger set is malformed")
    for name, expected_sql in _PAPER_SETTLEMENT_HOT_TRIGGER_SQL.items():
        if trigger_sql.get(name) != _normalized_sql(expected_sql):
            raise RuntimeError(f"paper settlement trigger {name} is malformed")
