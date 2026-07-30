"""Focused read-only diagnostics for unresolved local-paper safety authority."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

import scripts.manage_paper_safety_journal as journal_script
from robo_trader.accounting.fifo import FillSide
from robo_trader.accounting.fifo_fixture_migration import (
    _TABLE_SQL,
    _TRIGGER_SQL,
    FIFO_ACCOUNTING_COMPONENT,
    FIFO_ACCOUNTING_MIGRATIONS,
    _legacy_opening_manifest_hash,
)
from robo_trader.accounting.fifo_runtime import (
    LOCAL_PAPER_COMMISSION_SOURCE,
    RuntimePaperFillEvidence,
    append_runtime_fill_in_transaction,
)
from robo_trader.config import _derive_safety_account_scope
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.paper_terminal_settlement import (
    PaperTerminalSettlementReceipt,
    PaperTerminalSettlementRequest,
)
from robo_trader.runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from robo_trader.safety import (
    JournalEventType,
    OrderSide,
    PaperExecutionIdentity,
    ReplayReservation,
    SafetyJournal,
    SafetyRuntimeCoordinator,
    SubmissionClaim,
    TerminalOrderStatus,
    canonical_json,
)
from tests.safety.conftest import make_case


def _environment(tmp_path: Path) -> dict[str, str]:
    scope_key = "0123456789abcdef" * 4
    return {
        "EXECUTION_MODE": "paper",
        "TRADING_MODE": "paper",
        "ENVIRONMENT": "dev",
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "4002",
        "IBKR_READONLY": "true",
        "IBKR_CLIENT_ID": "123",
        "IBKR_ACCOUNT": "DU_TEST_PAPER",
        "IBKR_APPROVED_ACCOUNTS": "DU_TEST_PAPER",
        "IBKR_ACCOUNT_TYPE": "paper",
        "RT_STATE_NAMESPACE": "paper",
        "RT_DB_PATH": str(tmp_path / "paper-ledger.db"),
        "SAFETY_ACCOUNT_SCOPE_KEY": scope_key,
        "SAFETY_ACCOUNT_SCOPE": _derive_safety_account_scope(
            scope_key,
            "DU_TEST_PAPER",
        ),
        "SAFETY_JOURNAL_PATH": str(tmp_path / "paper-safety.db"),
        "MODEL_ARTIFACT_SET": "test-models",
        "BUILD_ID": "test-build",
    }


def _snapshot(path: Path) -> tuple[str, int, int, int]:
    metadata = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata.st_mtime_ns,
        metadata.st_size,
        metadata.st_ino,
    )


def _artifact_snapshot(path: Path) -> tuple[str, int, int]:
    metadata = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata.st_size,
        metadata.st_ino,
    )


def _directory_snapshot(path: Path) -> dict[str, tuple[str, int, int]]:
    return {
        item.name: _artifact_snapshot(item)
        for item in path.iterdir()
        if item.is_file() and not item.is_symlink()
    }


def _sqlite_family_snapshot(path: Path) -> dict[str, tuple[str, int, int] | None]:
    return {
        suffix: (
            _artifact_snapshot(Path(f"{path}{suffix}"))
            if Path(f"{path}{suffix}").exists()
            else None
        )
        for suffix in ("", "-wal", "-shm", "-journal")
    }


def _sqlite_durable_snapshot(path: Path) -> dict[str, tuple[str, int, int] | None]:
    """Snapshot durable pages; SQLite may update transient WAL shared memory on reads."""

    family = _sqlite_family_snapshot(path)
    snapshot = {suffix: family[suffix] for suffix in ("", "-wal", "-journal")}
    for suffix in ("-wal", "-journal"):
        artifact = snapshot[suffix]
        if artifact is not None and artifact[1] == 0:
            snapshot[suffix] = None
    return snapshot


def _non_journal_directory_snapshot(
    directory: Path,
    journal_path: Path,
) -> dict[str, tuple[str, int, int]]:
    return {
        name: value
        for name, value in _directory_snapshot(directory).items()
        if name
        not in {f"{journal_path.name}{suffix}" for suffix in ("", "-wal", "-shm", "-journal")}
    }


class _AvailableLifecycleLock:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.acquired = False
        self.released = False

    def acquire(self) -> bool:
        self.acquired = True
        return self.available

    def release(self) -> None:
        self.released = True


def _seed_unresolved_claim(
    environ: dict[str, str],
    *,
    outcome_unknown: bool,
) -> tuple[ReplayReservation, SubmissionClaim]:
    at = datetime.now(timezone.utc) - timedelta(seconds=30)
    journal = SafetyJournal(environ["SAFETY_JOURNAL_PATH"], clock=lambda: at)
    journal.initialize(
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
    )
    intent, exposure, allocation, gates, _, descriptor = make_case(
        at,
        account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
    )
    reservation, claim, permit = journal.authorize_submission(
        "status-test",
        intent,
        exposure,
        allocation,
        gates,
        descriptor,
    )
    if outcome_unknown:
        journal.consume_submission_permit(permit)
        journal.mark_outcome_unknown("status-test", intent.fingerprint())
    replayed = journal.replay(
        expected_execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        expected_account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
    ).active_reservations[0]
    assert replayed.reservation_id == reservation.reservation_id
    return replayed, claim


def _create_settlement_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                notional REAL NOT NULL,
                slippage REAL NOT NULL,
                commission REAL NOT NULL,
                pnl REAL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE positions (
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                market_price REAL,
                timestamp TEXT,
                UNIQUE(portfolio_id, symbol)
            );
            CREATE TABLE account (
                portfolio_id TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE paper_account_settlement_state (
                portfolio_id TEXT PRIMARY KEY,
                cash_text TEXT NOT NULL,
                realized_pnl_text TEXT NOT NULL,
                daily_pnl_text TEXT NOT NULL,
                daily_pnl_baseline_text TEXT NOT NULL,
                daily_pnl_date TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_settlement_id TEXT
            );
            CREATE TABLE paper_position_settlement_state (
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                cost_basis_text TEXT NOT NULL,
                mark_price_text TEXT,
                source_settlement_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (portfolio_id, symbol)
            );
            """)
        connection.execute("""
            CREATE TABLE paper_reduction_settlements (
                settlement_id TEXT PRIMARY KEY,
                execution_domain_scope TEXT NOT NULL,
                account_scope TEXT NOT NULL,
                portfolio_id TEXT NOT NULL,
                con_id INTEGER NOT NULL,
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
                schema_version INTEGER NOT NULL
            )
            """)
        for statement in _TABLE_SQL.values():
            connection.execute(statement)
        connection.executemany(
            """
            INSERT INTO fifo_schema_migrations(component,version,description,applied_at)
            VALUES (?,?,?,'2020-01-01T00:00:00.000000Z')
            """,
            tuple(
                (FIFO_ACCOUNTING_COMPONENT, version, description)
                for version, description in FIFO_ACCOUNTING_MIGRATIONS
            ),
        )
        for statement in _TRIGGER_SQL.values():
            connection.execute(statement)
        connection.executescript("""
            CREATE TABLE paper_fifo_settlement_links (
                settlement_id TEXT PRIMARY KEY,
                request_fingerprint TEXT NOT NULL UNIQUE,
                epoch_id TEXT NOT NULL,
                fill_id TEXT NOT NULL UNIQUE,
                event_sequence INTEGER NOT NULL,
                execution_id TEXT NOT NULL,
                commission_minor INTEGER NOT NULL,
                commission_currency TEXT NOT NULL,
                commission_source TEXT NOT NULL,
                fifo_state_fingerprint TEXT NOT NULL,
                committed_at TEXT NOT NULL,
                UNIQUE(epoch_id,event_sequence),
                UNIQUE(epoch_id,execution_id),
                FOREIGN KEY(settlement_id) REFERENCES paper_reduction_settlements(settlement_id),
                FOREIGN KEY(epoch_id,fill_id) REFERENCES fifo_fills(epoch_id,fill_id)
            );
            CREATE TRIGGER paper_fifo_settlement_links_no_update
            BEFORE UPDATE ON paper_fifo_settlement_links
            BEGIN
                SELECT RAISE(ABORT, 'paper FIFO settlement links are append-only');
            END;
            CREATE TRIGGER paper_fifo_settlement_links_no_delete
            BEFORE DELETE ON paper_fifo_settlement_links
            BEGIN
                SELECT RAISE(ABORT, 'paper FIFO settlement links are append-only');
            END;
            """)


def _insert_exact_settlement(
    environ: dict[str, str],
    reservation: ReplayReservation,
    claim: SubmissionClaim,
) -> PaperTerminalSettlementRequest:
    ledger_path = Path(environ["RT_DB_PATH"])
    contract = journal_script._paper_contract(environ)
    quote_payload = json.dumps(
        {
            "con_id": reservation.con_id,
            "portfolio_id": reservation.portfolio_id,
            "price": "100",
            "receipt_monotonic": float(10.0).hex(),
            "receipt_order": 1,
            "source": "live-broker",
            "source_event_id": "paper-safety-status-test",
            "source_timestamp": (
                datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
            ),
            "symbol": reservation.symbol,
            "transport_generation": "status-test-generation",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    request = PaperTerminalSettlementRequest(
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
        portfolio_id=reservation.portfolio_id,
        con_id=reservation.con_id,
        symbol=reservation.symbol,
        reservation_id=reservation.reservation_id,
        claim_id=reservation.claim_id,
        claim_sequence=reservation.claim_sequence,
        submission_descriptor_fingerprint=(reservation.submission_descriptor_fingerprint),
        protective_quote_fingerprint=hashlib.sha256(quote_payload.encode()).hexdigest(),
        protective_quote_payload=quote_payload,
        order_ref=reservation.order_ref,
        side=OrderSide.SELL,
        requested_quantity=Decimal("2"),
        filled_quantity=Decimal("2"),
        remaining_quantity=Decimal("0"),
        expected_pre_position_quantity=Decimal("5"),
        expected_post_position_quantity=Decimal("3"),
        expected_pre_aggregate_quantity=Decimal("10"),
        expected_post_aggregate_quantity=Decimal("8"),
        expected_pre_cash=Decimal("10000"),
        expected_post_cash=Decimal("10202.50"),
        expected_pre_realized_pnl=Decimal("0"),
        expected_post_realized_pnl=Decimal("2.50"),
        expected_pre_daily_pnl=Decimal("0"),
        expected_post_daily_pnl=Decimal("2.50"),
        expected_daily_pnl_baseline=Decimal("0"),
        expected_daily_pnl_date=datetime.now(timezone.utc).date().isoformat(),
        expected_position_cost_basis=Decimal("100"),
        expected_pre_position_mark_price=Decimal("100"),
        expected_pre_position_source_settlement_id=None,
        terminal_status=TerminalOrderStatus.FILLED,
        fill_price=Decimal("101.25"),
        outcome_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        fill_execution_id="lpfill-" + ("8" * 32),
        fill_commission_minor=0,
        fill_commission_currency="USD",
        fill_commission_source=LOCAL_PAPER_COMMISSION_SOURCE,
    )
    database_metadata = ledger_path.stat()
    settlement_id = "pset-" + ("2" * 32)
    trade_id = 7
    committed_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )
    receipt_payload = canonical_json(
        {
            "committed_at": committed_at,
            "database_device": database_metadata.st_dev,
            "database_identity": contract.database_identity,
            "database_inode": database_metadata.st_ino,
            "database_path": str(ledger_path),
            "request_fingerprint": request.fingerprint(),
            "schema_version": 1,
            "settlement_id": settlement_id,
            "trade_id": trade_id,
        }
    )
    receipt_fingerprint = hashlib.sha256(receipt_payload.encode("utf-8")).hexdigest()
    with sqlite3.connect(ledger_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        epoch_id = "fepoch-" + ("1" * 32)
        effective_at = "2020-01-01T00:00:00.000000Z"
        opening_balance_id = "fobal-" + ("3" * 32)
        connection.execute(
            """
            INSERT INTO fifo_accounting_epochs VALUES (
                ?,1,?,?,?,'LEGACY_AGGREGATE_OPENING_BALANCE',?,?,?
            )
            """,
            (
                epoch_id,
                request.execution_domain_scope,
                request.account_scope,
                request.portfolio_id,
                "4" * 64,
                effective_at,
                effective_at,
            ),
        )
        connection.execute(
            "INSERT INTO fifo_epoch_account_baselines VALUES (?,?,?,?,?,?,?)",
            (
                epoch_id,
                "10000",
                "0",
                "0",
                "0",
                request.expected_daily_pnl_date,
                effective_at,
            ),
        )
        manifest_row = (
            opening_balance_id,
            "flot-" + ("a" * 32),
            request.con_id,
            request.symbol,
            "LONG",
            "5",
            "100",
            "100",
            effective_at,
            "9" * 64,
            0,
            0,
            effective_at,
        )
        connection.execute(
            "INSERT INTO fifo_legacy_bootstrap_lineage VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                epoch_id,
                "pboot-" + ("5" * 32),
                "4" * 64,
                1,
                _legacy_opening_manifest_hash([manifest_row]),
                "status-reconciliation",
                "6" * 64,
                "7" * 64,
                "8" * 64,
                "status-administrator-action",
                effective_at,
            ),
        )
        connection.execute(
            "INSERT INTO fifo_opening_balances VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                opening_balance_id,
                epoch_id,
                request.con_id,
                request.symbol,
                "LONG",
                "5",
                "100",
                "100",
                effective_at,
                "9" * 64,
                effective_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO fifo_lot_openings VALUES (
                ?,?,NULL,?,0,?,?,?,'5','100',0,0,?
            )
            """,
            (
                "flot-" + ("a" * 32),
                epoch_id,
                opening_balance_id,
                request.con_id,
                request.symbol,
                "LONG",
                effective_at,
            ),
        )
        fifo_projection = append_runtime_fill_in_transaction(
            connection,
            RuntimePaperFillEvidence(
                execution_domain_scope=request.execution_domain_scope,
                account_scope=request.account_scope,
                portfolio_id=request.portfolio_id,
                con_id=request.con_id,
                symbol=request.symbol,
                side=FillSide.SELL,
                quantity=request.filled_quantity,
                price=request.fill_price,
                execution_id=request.fill_execution_id,
                idempotency_key=request.fingerprint(),
                commission_minor=request.fill_commission_minor,
                commission_currency=request.fill_commission_currency,
                commission_source=request.fill_commission_source,
                occurred_at=request.outcome_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO trades (
                id, portfolio_id, symbol, side, quantity, price, notional,
                slippage, commission, pnl, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
            """,
            (
                trade_id,
                request.portfolio_id,
                request.symbol,
                request.side.value,
                int(request.filled_quantity),
                float(request.fill_price),
                float(request.fill_price * request.filled_quantity),
                float(request.expected_post_realized_pnl - request.expected_pre_realized_pnl),
                committed_at,
            ),
        )
        connection.executemany(
            """
            INSERT INTO positions (
                portfolio_id, symbol, quantity, avg_cost, market_price, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    request.portfolio_id,
                    request.symbol,
                    int(request.expected_post_position_quantity),
                    float(request.expected_position_cost_basis),
                    float(request.protective_mark_price),
                    committed_at,
                ),
                (
                    "portfolio-b",
                    request.symbol,
                    int(
                        request.expected_post_aggregate_quantity
                        - request.expected_post_position_quantity
                    ),
                    99.0,
                    float(request.protective_mark_price),
                    committed_at,
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO account (
                portfolio_id, cash, equity, daily_pnl, realized_pnl,
                unrealized_pnl, timestamp
            ) VALUES (?, ?, 10000, ?, ?, 0, ?)
            """,
            (
                request.portfolio_id,
                float(request.expected_post_cash),
                float(request.expected_post_daily_pnl),
                float(request.expected_post_realized_pnl),
                committed_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO paper_position_settlement_state (
                portfolio_id, symbol, cost_basis_text, mark_price_text,
                source_settlement_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request.portfolio_id,
                request.symbol,
                str(request.expected_position_cost_basis),
                str(request.protective_mark_price),
                settlement_id,
                committed_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO paper_reduction_settlements (
                settlement_id, execution_domain_scope, account_scope,
                portfolio_id, con_id, symbol, reservation_id, claim_id,
                order_ref, protective_quote_payload, request_fingerprint,
                request_payload_json,
                terminal_status, trade_id, database_path, database_identity,
                database_device, database_inode, committed_at,
                receipt_fingerprint, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                settlement_id,
                request.execution_domain_scope,
                request.account_scope,
                request.portfolio_id,
                request.con_id,
                request.symbol,
                request.reservation_id,
                request.claim_id,
                request.order_ref,
                request.protective_quote_payload,
                request.fingerprint(),
                request.canonical_payload(),
                request.terminal_status.value,
                trade_id,
                str(ledger_path),
                contract.database_identity,
                database_metadata.st_dev,
                database_metadata.st_ino,
                committed_at,
                receipt_fingerprint,
                1,
            ),
        )
        connection.execute(
            """
            INSERT INTO paper_account_settlement_state (
                portfolio_id, cash_text, realized_pnl_text, daily_pnl_text,
                daily_pnl_baseline_text, daily_pnl_date,
                updated_at, source_settlement_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.portfolio_id,
                str(request.expected_post_cash.normalize()),
                str(request.expected_post_realized_pnl.normalize()),
                str(request.expected_post_daily_pnl.normalize()),
                str(request.expected_daily_pnl_baseline.normalize()),
                request.expected_daily_pnl_date,
                committed_at,
                settlement_id,
            ),
        )
        connection.execute(
            """
            INSERT INTO paper_fifo_settlement_links VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                settlement_id,
                request.fingerprint(),
                fifo_projection.epoch_id,
                fifo_projection.fill_id,
                fifo_projection.event_sequence,
                request.fill_execution_id,
                request.fill_commission_minor,
                request.fill_commission_currency,
                request.fill_commission_source,
                fifo_projection.state_fingerprint,
                committed_at,
            ),
        )
    return request


def _zero_fill_request(
    environ: dict[str, str],
    reservation: ReplayReservation,
    terminal_status: TerminalOrderStatus,
) -> PaperTerminalSettlementRequest:
    quote_payload = json.dumps(
        {
            "con_id": reservation.con_id,
            "portfolio_id": reservation.portfolio_id,
            "price": "101.25",
            "receipt_monotonic": float(10.0).hex(),
            "receipt_order": 1,
            "source": "live-broker",
            "source_event_id": "paper-zero-fill-status-test",
            "source_timestamp": (
                datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
            ),
            "symbol": reservation.symbol,
            "transport_generation": "zero-fill-status-generation",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return PaperTerminalSettlementRequest(
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
        portfolio_id=reservation.portfolio_id,
        con_id=reservation.con_id,
        symbol=reservation.symbol,
        reservation_id=reservation.reservation_id,
        claim_id=reservation.claim_id,
        claim_sequence=reservation.claim_sequence,
        submission_descriptor_fingerprint=reservation.submission_descriptor_fingerprint,
        protective_quote_fingerprint=hashlib.sha256(quote_payload.encode()).hexdigest(),
        protective_quote_payload=quote_payload,
        order_ref=reservation.order_ref,
        side=OrderSide.SELL,
        requested_quantity=Decimal("2"),
        filled_quantity=Decimal("0"),
        remaining_quantity=Decimal("2"),
        expected_pre_position_quantity=Decimal("5"),
        expected_post_position_quantity=Decimal("5"),
        expected_pre_aggregate_quantity=Decimal("10"),
        expected_post_aggregate_quantity=Decimal("10"),
        expected_pre_cash=Decimal("10000"),
        expected_post_cash=Decimal("10000"),
        expected_pre_realized_pnl=Decimal("3.25"),
        expected_post_realized_pnl=Decimal("3.25"),
        expected_pre_daily_pnl=Decimal("4.5"),
        expected_post_daily_pnl=Decimal("4.5"),
        expected_daily_pnl_baseline=Decimal("-1.25"),
        expected_daily_pnl_date=datetime.now(timezone.utc).date().isoformat(),
        expected_position_cost_basis=Decimal("100"),
        expected_pre_position_mark_price=Decimal("101"),
        expected_pre_position_source_settlement_id=None,
        terminal_status=terminal_status,
        fill_price=None,
        outcome_at=datetime.now(timezone.utc) - timedelta(seconds=10),
    )


async def _commit_zero_fill_case(
    tmp_path: Path,
    terminal_status: TerminalOrderStatus,
) -> tuple[
    dict[str, str],
    ReplayReservation,
    PaperTerminalSettlementRequest,
    PaperTerminalSettlementReceipt,
]:
    environ = _environment(tmp_path)
    reservation, _ = _seed_unresolved_claim(environ, outcome_unknown=True)
    contract = journal_script._paper_contract(environ)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        async with database.get_connection() as connection:
            await connection.executemany(
                "INSERT INTO portfolios (id, name) VALUES (?, ?)",
                (("portfolio-a", "Portfolio A"), ("portfolio-b", "Portfolio B")),
            )
            await connection.commit()
        for portfolio_id in ("portfolio-a", "portfolio-b"):
            await database.update_position(
                "AAPL",
                5,
                Decimal("100"),
                Decimal("101"),
                portfolio_id=portfolio_id,
            )
        await database.update_account(
            Decimal("10000"),
            Decimal("10500"),
            daily_pnl=Decimal("4.5"),
            realized_pnl=Decimal("3.25"),
            portfolio_id="portfolio-a",
            daily_pnl_baseline=Decimal("-1.25"),
        )
        request = _zero_fill_request(environ, reservation, terminal_status)
        receipt = await database.commit_paper_reduction_outcome(
            request,
            runtime_contract=contract,
        )
        assert receipt.trade_id is None
        return environ, reservation, request, receipt
    finally:
        await database.close()


def _prepared_status_case(
    tmp_path: Path,
) -> tuple[
    dict[str, str],
    ReplayReservation,
    SubmissionClaim,
    PaperTerminalSettlementRequest,
]:
    environ = _environment(tmp_path)
    reservation, claim = _seed_unresolved_claim(environ, outcome_unknown=True)
    _create_settlement_schema(Path(environ["RT_DB_PATH"]))
    request = _insert_exact_settlement(environ, reservation, claim)
    return environ, reservation, claim, request


def _resign_settlement_receipt(connection: sqlite3.Connection) -> None:
    row = connection.execute("""
        SELECT settlement_id, request_fingerprint, trade_id, database_path,
               database_identity, database_device, database_inode, committed_at,
               schema_version
        FROM paper_reduction_settlements
        """).fetchone()
    assert row is not None
    (
        settlement_id,
        request_fingerprint,
        trade_id,
        database_path,
        database_identity,
        database_device,
        database_inode,
        committed_at,
        schema_version,
    ) = row
    payload = canonical_json(
        {
            "committed_at": committed_at,
            "database_device": database_device,
            "database_identity": database_identity,
            "database_inode": database_inode,
            "database_path": database_path,
            "request_fingerprint": request_fingerprint,
            "schema_version": schema_version,
            "settlement_id": settlement_id,
            "trade_id": trade_id,
        }
    )
    connection.execute(
        "UPDATE paper_reduction_settlements SET receipt_fingerprint = ?",
        (hashlib.sha256(payload.encode("utf-8")).hexdigest(),),
    )


def test_status_json_is_redacted_exact_and_strictly_read_only(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    environ, reservation, claim, request = _prepared_status_case(tmp_path)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    ledger_path = Path(environ["RT_DB_PATH"])
    before_directory = _directory_snapshot(tmp_path)
    before_journal = _sqlite_family_snapshot(journal_path)
    before_ledger = _sqlite_family_snapshot(ledger_path)

    report = journal_script.paper_safety_status(environ)
    serialized = json.dumps(report, sort_keys=True)
    assert report["schema_version"] == 1
    assert report["status"] == "BLOCKED"
    assert report["reason_codes"] == ["UNRESOLVED_PAPER_SUBMISSION_AUTHORITY"]
    assert report["unresolved_count"] == 1
    item = report["unresolved_reservations"][0]
    assert item["symbol"] == "AAPL"
    assert item["portfolio_id"] == "portfolio-a"
    assert item["phase"] == "OUTCOME_UNKNOWN"
    assert item["outcome_unknown"] is True
    assert item["quarantined"] is True
    assert item["exact_local_settlement_exists"] is True
    assert item["local_settlement_status"] == "MATCH"
    assert item["age_seconds"] >= 0
    assert len(item["reservation_id_sha256"]) == 64
    assert len(item["claim_id_sha256"]) == 64
    assert len(item["order_ref_sha256"]) == 64
    assert "LOCAL_TERMINAL_SETTLEMENT_PRESENT" in item["reason_codes"]

    for sensitive in (
        reservation.reservation_id,
        claim.claim_id,
        request.order_ref,
        environ["SAFETY_ACCOUNT_SCOPE"],
        environ["SAFETY_ACCOUNT_SCOPE_KEY"],
        environ["SAFETY_JOURNAL_PATH"],
        environ["RT_DB_PATH"],
        environ["IBKR_ACCOUNT"],
    ):
        assert sensitive not in serialized

    monkeypatch.setattr(journal_script, "_resolved_environment", lambda: environ)
    assert journal_script.main(["status", "--json"]) == 0
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["status"] == "BLOCKED"
    assert cli_report["unresolved_reservations"][0]["exact_local_settlement_exists"] is True
    assert _directory_snapshot(tmp_path) == before_directory
    assert _sqlite_family_snapshot(journal_path) == before_journal
    assert _sqlite_family_snapshot(ledger_path) == before_ledger


def test_status_rejects_nonmatching_local_settlement_receipt(tmp_path: Path) -> None:
    environ, _, _, _ = _prepared_status_case(tmp_path)
    ledger_path = Path(environ["RT_DB_PATH"])
    with sqlite3.connect(ledger_path) as connection:
        connection.execute(
            "UPDATE paper_reduction_settlements SET receipt_fingerprint = ?",
            ("f" * 64,),
        )
    before_directory = _directory_snapshot(tmp_path)
    before_journal = _sqlite_family_snapshot(Path(environ["SAFETY_JOURNAL_PATH"]))
    before_ledger = _sqlite_family_snapshot(ledger_path)

    item = journal_script.paper_safety_status(environ)["unresolved_reservations"][0]

    assert item["exact_local_settlement_exists"] is False
    assert item["local_settlement_status"] == "MISMATCH"
    assert "LOCAL_TERMINAL_SETTLEMENT_MISMATCH" in item["reason_codes"]
    assert _directory_snapshot(tmp_path) == before_directory
    assert _sqlite_family_snapshot(Path(environ["SAFETY_JOURNAL_PATH"])) == before_journal
    assert _sqlite_family_snapshot(ledger_path) == before_ledger


@pytest.mark.parametrize(
    "corruption",
    [
        "FABRICATED_TRADE_ID",
        "MISSING_TRADE",
        "TAMPERED_TRADE",
        "TAMPERED_POSITION",
        "TAMPERED_AGGREGATE",
        "TAMPERED_POSITION_BASIS",
        "TAMPERED_POSITION_MARK",
        "TAMPERED_POSITION_SOURCE",
        "TAMPERED_LEGACY_POSITION_MARK",
        "FUTURE_POSITION_UPDATED",
        "TAMPERED_EXACT_ACCOUNT",
        "TAMPERED_ACCOUNT_BASELINE",
        "FUTURE_ACCOUNT_DATE",
        "TAMPERED_ACCOUNT_SOURCE",
        "TAMPERED_LEGACY_ACCOUNT",
        "AMBIGUOUS_TRADE",
        "TAMPERED_FIFO_LINK",
        "MISSING_FIFO_LINK_SCHEMA",
        "MISSING_MARK_SCHEMA",
        "MISSING_ACCOUNT_BASELINE_SCHEMA",
        "OUTBOX_ONLY",
    ],
)
def test_status_and_recovery_reject_partial_or_forged_atomic_projection(
    tmp_path: Path,
    monkeypatch,
    corruption: str,
) -> None:
    environ, _, _, request = _prepared_status_case(tmp_path)
    ledger_path = Path(environ["RT_DB_PATH"])
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    with sqlite3.connect(ledger_path) as connection:
        if corruption == "FABRICATED_TRADE_ID":
            connection.execute("UPDATE paper_reduction_settlements SET trade_id = 999")
            _resign_settlement_receipt(connection)
        elif corruption == "MISSING_TRADE":
            connection.execute("DELETE FROM trades WHERE id = 7")
        elif corruption == "TAMPERED_TRADE":
            connection.execute("UPDATE trades SET pnl = pnl + 1 WHERE id = 7")
        elif corruption == "TAMPERED_POSITION":
            connection.execute(
                """
                UPDATE positions SET quantity = quantity + 1
                WHERE portfolio_id = ? AND symbol = ?
                """,
                (request.portfolio_id, request.symbol),
            )
        elif corruption == "TAMPERED_AGGREGATE":
            connection.execute(
                """
                UPDATE positions SET quantity = quantity + 1
                WHERE portfolio_id = 'portfolio-b' AND symbol = ?
                """,
                (request.symbol,),
            )
        elif corruption == "TAMPERED_POSITION_BASIS":
            connection.execute("UPDATE paper_position_settlement_state SET cost_basis_text = '99'")
        elif corruption == "TAMPERED_POSITION_MARK":
            connection.execute("UPDATE paper_position_settlement_state SET mark_price_text = '99'")
        elif corruption == "TAMPERED_POSITION_SOURCE":
            connection.execute(
                "UPDATE paper_position_settlement_state SET source_settlement_id = NULL"
            )
        elif corruption == "TAMPERED_LEGACY_POSITION_MARK":
            connection.execute("UPDATE positions SET market_price = market_price + 1")
        elif corruption == "FUTURE_POSITION_UPDATED":
            connection.execute("""
                UPDATE paper_position_settlement_state
                SET updated_at = '2099-01-01T00:00:00.000000Z'
                """)
        elif corruption == "TAMPERED_EXACT_ACCOUNT":
            connection.execute("UPDATE paper_account_settlement_state SET cash_text = '10203.5'")
        elif corruption == "TAMPERED_ACCOUNT_BASELINE":
            connection.execute(
                "UPDATE paper_account_settlement_state SET daily_pnl_baseline_text = '1'"
            )
        elif corruption == "FUTURE_ACCOUNT_DATE":
            connection.execute(
                "UPDATE paper_account_settlement_state SET daily_pnl_date = '2099-01-01'"
            )
        elif corruption == "TAMPERED_ACCOUNT_SOURCE":
            connection.execute(
                "UPDATE paper_account_settlement_state SET source_settlement_id = NULL"
            )
        elif corruption == "TAMPERED_LEGACY_ACCOUNT":
            connection.execute("UPDATE account SET cash = cash + 1")
        elif corruption == "AMBIGUOUS_TRADE":
            connection.execute("""
                INSERT INTO trades (
                    id, portfolio_id, symbol, side, quantity, price, notional,
                    slippage, commission, pnl, timestamp
                ) SELECT 8, portfolio_id, symbol, side, quantity, price, notional,
                         slippage, commission, pnl, timestamp
                  FROM trades WHERE id = 7
                """)
        elif corruption == "TAMPERED_FIFO_LINK":
            connection.execute("DROP TRIGGER paper_fifo_settlement_links_no_update")
            connection.execute(
                "UPDATE paper_fifo_settlement_links SET fifo_state_fingerprint = ?",
                ("0" * 64,),
            )
        elif corruption == "MISSING_FIFO_LINK_SCHEMA":
            connection.execute("DROP TABLE paper_fifo_settlement_links")
        elif corruption == "MISSING_MARK_SCHEMA":
            connection.execute(
                "ALTER TABLE paper_position_settlement_state DROP COLUMN mark_price_text"
            )
        elif corruption == "MISSING_ACCOUNT_BASELINE_SCHEMA":
            connection.execute("""
                ALTER TABLE paper_account_settlement_state
                DROP COLUMN daily_pnl_baseline_text
                """)
        elif corruption == "OUTBOX_ONLY":
            connection.executescript("""
                DROP TABLE trades;
                DROP TABLE positions;
                DROP TABLE account;
                DROP TABLE paper_account_settlement_state;
                DROP TABLE paper_position_settlement_state;
                """)
        else:  # pragma: no cover - parameter exhaustiveness guard
            raise AssertionError(corruption)

    before_journal = _sqlite_family_snapshot(journal_path)
    before_ledger = _sqlite_family_snapshot(ledger_path)
    item = journal_script.paper_safety_status(environ)["unresolved_reservations"][0]
    expected_status = (
        "SCHEMA_MISSING"
        if corruption
        in {
            "OUTBOX_ONLY",
            "MISSING_MARK_SCHEMA",
            "MISSING_ACCOUNT_BASELINE_SCHEMA",
            "MISSING_FIFO_LINK_SCHEMA",
        }
        else "MISMATCH"
    )
    assert item["local_settlement_status"] == expected_status
    assert item["exact_local_settlement_exists"] is False

    _enable_stopped_offline_recovery(monkeypatch)
    with pytest.raises(
        RuntimeError,
        match=f"local terminal settlement status is {expected_status}",
    ):
        journal_script.recover_exact_local_paper_settlement(
            environ,
            confirmation=journal_script.RECOVER_CONFIRMATION,
        )
    assert _sqlite_family_snapshot(journal_path) == before_journal
    assert _sqlite_family_snapshot(ledger_path) == before_ledger


def test_status_subprocess_loads_no_broker_module_and_mutates_no_database(
    tmp_path: Path,
) -> None:
    environ, _, _, _ = _prepared_status_case(tmp_path)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    ledger_path = Path(environ["RT_DB_PATH"])
    before_directory = _directory_snapshot(tmp_path)
    before_journal = _sqlite_family_snapshot(journal_path)
    before_ledger = _sqlite_family_snapshot(ledger_path)
    child_environment = dict(os.environ)
    child_environment.update(environ)
    program = """
import json
import sys
import scripts.manage_paper_safety_journal as command

result = command.main(["status", "--json"])
for module_name in sys.modules:
    if module_name == "ib_insync" or module_name.startswith("ib_insync."):
        raise SystemExit(91)
    if module_name == "ibapi" or module_name.startswith("ibapi."):
        raise SystemExit(92)
    if module_name.startswith("robo_trader.clients"):
        raise SystemExit(93)
raise SystemExit(result)
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=journal_script.PROJECT_ROOT,
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "BLOCKED"
    serialized = json.dumps(report, sort_keys=True)
    for sensitive in (
        environ["SAFETY_ACCOUNT_SCOPE_KEY"],
        environ["SAFETY_JOURNAL_PATH"],
        environ["RT_DB_PATH"],
    ):
        assert sensitive not in serialized
    assert _directory_snapshot(tmp_path) == before_directory
    assert _sqlite_family_snapshot(journal_path) == before_journal
    assert _sqlite_family_snapshot(ledger_path) == before_ledger


def _enable_stopped_offline_recovery(monkeypatch) -> _AvailableLifecycleLock:
    lock = _AvailableLifecycleLock()
    monkeypatch.setattr(journal_script, "RuntimeLifecycleLock", lambda: lock)
    monkeypatch.setattr(journal_script, "_assert_gateway_stopped", lambda: None)
    return lock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    [
        TerminalOrderStatus.CANCELLED,
        TerminalOrderStatus.REJECTED,
        TerminalOrderStatus.EXPIRED,
    ],
)
async def test_zero_fill_commit_release_failure_recovers_offline_exactly_once(
    tmp_path: Path,
    monkeypatch,
    terminal_status: TerminalOrderStatus,
) -> None:
    environ, reservation, request, receipt = await _commit_zero_fill_case(
        tmp_path,
        terminal_status,
    )
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    ledger_path = Path(environ["RT_DB_PATH"])
    before_ledger = _sqlite_durable_snapshot(ledger_path)
    unresolved = SafetyJournal(journal_path).replay().active_reservations
    assert len(unresolved) == 1
    assert unresolved[0].reservation_id == reservation.reservation_id
    assert unresolved[0].outcome_unknown is True

    item = journal_script.paper_safety_status(environ)["unresolved_reservations"][0]
    assert item["local_settlement_status"] == "MATCH"
    _enable_stopped_offline_recovery(monkeypatch)
    recovered = journal_script.recover_exact_local_paper_settlement(
        environ,
        confirmation=journal_script.RECOVER_CONFIRMATION,
    )

    assert recovered.terminal_sequence is not None
    assert SafetyJournal(journal_path).replay().active_reservations == ()
    assert _sqlite_durable_snapshot(ledger_path) == before_ledger
    assert request.terminal_status is terminal_status
    assert receipt.request.fingerprint() == request.fingerprint()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "ZERO_SOURCE_MISSING",
        "ZERO_TRADE_FORGED",
        "ZERO_POSITION_TAMPERED",
        "ZERO_POSITION_MARK_TAMPERED",
        "ZERO_POSITION_SOURCE_WRONG",
        "ZERO_POSITION_TIMESTAMP_FUTURE",
        "ZERO_LEGACY_MARK_TAMPERED",
        "ZERO_EXACT_ACCOUNT_TAMPERED",
        "ZERO_ACCOUNT_BASELINE_TAMPERED",
        "ZERO_ACCOUNT_DATE_FUTURE",
        "ZERO_EXACT_TIMESTAMP_FUTURE",
        "ZERO_LEGACY_TIMESTAMP_FUTURE",
    ],
)
async def test_zero_fill_recovery_rejects_forged_or_changed_projection(
    tmp_path: Path,
    monkeypatch,
    corruption: str,
) -> None:
    environ, _, request, receipt = await _commit_zero_fill_case(
        tmp_path,
        TerminalOrderStatus.CANCELLED,
    )
    ledger_path = Path(environ["RT_DB_PATH"])
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    future = (
        (receipt.committed_at + timedelta(seconds=1))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    with sqlite3.connect(ledger_path) as connection:
        if corruption == "ZERO_SOURCE_MISSING":
            connection.execute(
                "UPDATE paper_account_settlement_state SET source_settlement_id = NULL"
            )
        elif corruption == "ZERO_TRADE_FORGED":
            connection.execute(
                """
                INSERT INTO trades (
                    portfolio_id, symbol, side, quantity, price, notional,
                    slippage, commission, pnl, timestamp
                ) VALUES (?, ?, 'SELL', 1, 101.25, 101.25, 0, 0, 1.25, ?)
                """,
                (request.portfolio_id, request.symbol, receipt.committed_at.isoformat()),
            )
            connection.execute(
                "UPDATE trades SET timestamp = ? WHERE id = last_insert_rowid()",
                (receipt.committed_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),),
            )
        elif corruption == "ZERO_POSITION_TAMPERED":
            connection.execute(
                """
                UPDATE positions SET quantity = quantity - 1
                WHERE portfolio_id = ? AND symbol = ?
                """,
                (request.portfolio_id, request.symbol),
            )
        elif corruption == "ZERO_POSITION_MARK_TAMPERED":
            connection.execute("UPDATE paper_position_settlement_state SET mark_price_text = '102'")
        elif corruption == "ZERO_POSITION_SOURCE_WRONG":
            connection.execute("""
                UPDATE paper_position_settlement_state
                SET source_settlement_id = 'pset-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                """)
        elif corruption == "ZERO_POSITION_TIMESTAMP_FUTURE":
            connection.execute(
                "UPDATE paper_position_settlement_state SET updated_at = ?",
                (future,),
            )
        elif corruption == "ZERO_LEGACY_MARK_TAMPERED":
            connection.execute("UPDATE positions SET market_price = market_price + 1")
        elif corruption == "ZERO_EXACT_ACCOUNT_TAMPERED":
            connection.execute("UPDATE paper_account_settlement_state SET daily_pnl_text = '4.6'")
        elif corruption == "ZERO_ACCOUNT_BASELINE_TAMPERED":
            connection.execute(
                "UPDATE paper_account_settlement_state SET daily_pnl_baseline_text = '0'"
            )
        elif corruption == "ZERO_ACCOUNT_DATE_FUTURE":
            connection.execute(
                "UPDATE paper_account_settlement_state SET daily_pnl_date = '2099-01-01'"
            )
        elif corruption == "ZERO_EXACT_TIMESTAMP_FUTURE":
            connection.execute(
                "UPDATE paper_account_settlement_state SET updated_at = ?",
                (future,),
            )
        elif corruption == "ZERO_LEGACY_TIMESTAMP_FUTURE":
            connection.execute("UPDATE account SET timestamp = ?", (future,))
        else:  # pragma: no cover - parameter exhaustiveness guard
            raise AssertionError(corruption)

    before_journal = _sqlite_family_snapshot(journal_path)
    before_ledger = _sqlite_durable_snapshot(ledger_path)
    item = journal_script.paper_safety_status(environ)["unresolved_reservations"][0]
    assert item["local_settlement_status"] == "MISMATCH"
    _enable_stopped_offline_recovery(monkeypatch)
    with pytest.raises(RuntimeError, match="local terminal settlement status is MISMATCH"):
        journal_script.recover_exact_local_paper_settlement(
            environ,
            confirmation=journal_script.RECOVER_CONFIRMATION,
        )
    assert _sqlite_family_snapshot(journal_path) == before_journal
    assert _sqlite_durable_snapshot(ledger_path) == before_ledger


def test_exact_offline_recovery_appends_once_starts_clean_and_cannot_resubmit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    environ, reservation, claim, request = _prepared_status_case(tmp_path)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    ledger_path = Path(environ["RT_DB_PATH"])
    before_ledger = _sqlite_family_snapshot(ledger_path)
    before_non_journal = _non_journal_directory_snapshot(tmp_path, journal_path)
    before_state = SafetyJournal(journal_path).replay()
    lock = _enable_stopped_offline_recovery(monkeypatch)
    monkeypatch.setattr(journal_script, "_resolved_environment", lambda: environ)

    assert (
        journal_script.main(
            [
                "recover-exact-local-settlement",
                "--confirm",
                journal_script.RECOVER_CONFIRMATION,
                "--json",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    serialized = json.dumps(report, sort_keys=True)
    assert report["status"] == "RECOVERED"
    assert report["appended_terminal_events"] == 1
    assert report["terminal_sequence"] == before_state.last_sequence + 1
    for sensitive in (
        reservation.reservation_id,
        claim.claim_id,
        request.order_ref,
        environ["SAFETY_ACCOUNT_SCOPE"],
        environ["SAFETY_ACCOUNT_SCOPE_KEY"],
        environ["SAFETY_JOURNAL_PATH"],
        environ["RT_DB_PATH"],
        environ["IBKR_ACCOUNT"],
    ):
        assert sensitive not in serialized
    assert lock.acquired is True
    assert lock.released is True
    assert _sqlite_family_snapshot(ledger_path) == before_ledger
    assert _non_journal_directory_snapshot(tmp_path, journal_path) == before_non_journal

    journal = SafetyJournal(journal_path)
    state = journal.replay(
        expected_execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        expected_account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
    )
    assert state.active_reservations == ()
    terminal_events = [
        event for event in state.events if event.event_type is JournalEventType.TERMINAL_RECONCILED
    ]
    assert len(terminal_events) == 1
    coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
            environ["SAFETY_ACCOUNT_SCOPE"],
        ),
        journal,
    )
    coordinator.start()
    assert coordinator.started is True

    intent, exposure, allocation, gates, _, descriptor = make_case(
        reservation.acquired_at,
        account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
    )
    _, replayed_claim, permit = journal.authorize_submission(
        "status-test",
        intent,
        exposure,
        allocation,
        gates,
        descriptor,
    )
    assert replayed_claim.granted is False
    assert permit is None

    stable_journal = _artifact_snapshot(journal_path)
    assert (
        journal_script.main(
            [
                "recover-exact-local-settlement",
                "--confirm",
                journal_script.RECOVER_CONFIRMATION,
                "--json",
            ]
        )
        == 2
    )
    capsys.readouterr()
    assert _artifact_snapshot(journal_path) == stable_journal
    assert _sqlite_family_snapshot(ledger_path) == before_ledger


def test_offline_recovery_subprocess_loads_no_broker_and_changes_only_journal(
    tmp_path: Path,
) -> None:
    environ, reservation, claim, request = _prepared_status_case(tmp_path)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    ledger_path = Path(environ["RT_DB_PATH"])
    before_ledger = _sqlite_family_snapshot(ledger_path)
    before_non_journal = _non_journal_directory_snapshot(tmp_path, journal_path)
    child_environment = dict(os.environ)
    child_environment.update(environ)
    program = """
import sys
import scripts.manage_paper_safety_journal as command

class AvailableLock:
    def acquire(self):
        return True
    def release(self):
        return None

command.RuntimeLifecycleLock = AvailableLock
command._assert_gateway_stopped = lambda: None
result = command.main([
    "recover-exact-local-settlement",
    "--confirm",
    command.RECOVER_CONFIRMATION,
    "--json",
])
for module_name in sys.modules:
    if module_name == "ib_insync" or module_name.startswith("ib_insync."):
        raise SystemExit(91)
    if module_name == "ibapi" or module_name.startswith("ibapi."):
        raise SystemExit(92)
    if module_name.startswith("robo_trader.clients"):
        raise SystemExit(93)
raise SystemExit(result)
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=journal_script.PROJECT_ROOT,
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "RECOVERED"
    serialized = json.dumps(report, sort_keys=True)
    for sensitive in (
        reservation.reservation_id,
        claim.claim_id,
        request.order_ref,
        environ["SAFETY_ACCOUNT_SCOPE"],
        environ["SAFETY_ACCOUNT_SCOPE_KEY"],
        environ["SAFETY_JOURNAL_PATH"],
        environ["RT_DB_PATH"],
        environ["IBKR_ACCOUNT"],
    ):
        assert sensitive not in serialized
    assert _sqlite_family_snapshot(ledger_path) == before_ledger
    assert _non_journal_directory_snapshot(tmp_path, journal_path) == before_non_journal
    assert SafetyJournal(journal_path).replay().active_reservations == ()


@pytest.mark.parametrize("settlement_state", ["ABSENT", "MISMATCH", "AMBIGUOUS"])
def test_offline_recovery_failures_leave_every_artifact_unchanged(
    tmp_path: Path,
    monkeypatch,
    settlement_state: str,
) -> None:
    if settlement_state == "ABSENT":
        environ = _environment(tmp_path)
        reservation, _ = _seed_unresolved_claim(environ, outcome_unknown=True)
        _create_settlement_schema(Path(environ["RT_DB_PATH"]))
    else:
        environ, reservation, _, _ = _prepared_status_case(tmp_path)
    ledger_path = Path(environ["RT_DB_PATH"])
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    with sqlite3.connect(ledger_path) as connection:
        if settlement_state == "MISMATCH":
            connection.execute(
                "UPDATE paper_reduction_settlements SET receipt_fingerprint = ?",
                ("f" * 64,),
            )
        elif settlement_state == "AMBIGUOUS":
            row = connection.execute("SELECT * FROM paper_reduction_settlements").fetchone()
            columns = [
                item[1]
                for item in connection.execute("PRAGMA table_info(paper_reduction_settlements)")
            ]
            duplicate = dict(zip(columns, row))
            duplicate["settlement_id"] = "pset-" + ("3" * 32)
            duplicate["reservation_id"] = "res-" + ("4" * 32)
            duplicate["claim_id"] = "claim-" + ("5" * 32)
            placeholders = ",".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO paper_reduction_settlements ({','.join(columns)}) "
                f"VALUES ({placeholders})",
                tuple(duplicate[column] for column in columns),
            )

    before_directory = _directory_snapshot(tmp_path)
    before_journal = _sqlite_family_snapshot(journal_path)
    before_ledger = _sqlite_family_snapshot(ledger_path)
    _enable_stopped_offline_recovery(monkeypatch)

    with pytest.raises(RuntimeError, match="offline recovery blocked"):
        journal_script.recover_exact_local_paper_settlement(
            environ,
            confirmation=journal_script.RECOVER_CONFIRMATION,
        )

    item = journal_script.paper_safety_status(environ)["unresolved_reservations"][0]
    expected = "IDENTITY_CONFLICT" if settlement_state == "AMBIGUOUS" else settlement_state
    assert item["local_settlement_status"] == expected
    assert item["reservation_id_sha256"] == journal_script._redacted_identifier(
        "reservation_id", reservation.reservation_id
    )
    assert _directory_snapshot(tmp_path) == before_directory
    assert _sqlite_family_snapshot(journal_path) == before_journal
    assert _sqlite_family_snapshot(ledger_path) == before_ledger


def test_offline_recovery_requires_exact_confirmation_lock_and_stopped_gateway(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environ, _, _, _ = _prepared_status_case(tmp_path)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    ledger_path = Path(environ["RT_DB_PATH"])
    before_directory = _directory_snapshot(tmp_path)
    before_journal = _sqlite_family_snapshot(journal_path)
    before_ledger = _sqlite_family_snapshot(ledger_path)

    with pytest.raises(ValueError, match=journal_script.RECOVER_CONFIRMATION):
        journal_script.recover_exact_local_paper_settlement(
            environ,
            confirmation="yes",
        )

    unavailable = _AvailableLifecycleLock(available=False)
    monkeypatch.setattr(journal_script, "RuntimeLifecycleLock", lambda: unavailable)
    with pytest.raises(RuntimeError, match="lifecycle lock"):
        journal_script.recover_exact_local_paper_settlement(
            environ,
            confirmation=journal_script.RECOVER_CONFIRMATION,
        )

    available = _AvailableLifecycleLock()
    monkeypatch.setattr(journal_script, "RuntimeLifecycleLock", lambda: available)
    monkeypatch.setattr(
        journal_script,
        "_assert_gateway_stopped",
        lambda: (_ for _ in ()).throw(RuntimeError("Gateway must remain stopped")),
    )
    with pytest.raises(RuntimeError, match="Gateway must remain stopped"):
        journal_script.recover_exact_local_paper_settlement(
            environ,
            confirmation=journal_script.RECOVER_CONFIRMATION,
        )
    assert available.released is True
    assert _directory_snapshot(tmp_path) == before_directory
    assert _sqlite_family_snapshot(journal_path) == before_journal
    assert _sqlite_family_snapshot(ledger_path) == before_ledger


@pytest.mark.parametrize("fault_stage", ["AFTER_APPEND", "BEFORE_COMMIT"])
def test_offline_recovery_journal_crash_boundary_rolls_back_and_can_retry_once(
    tmp_path: Path,
    monkeypatch,
    fault_stage: str,
) -> None:
    environ, _, _, _ = _prepared_status_case(tmp_path)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    ledger_path = Path(environ["RT_DB_PATH"])
    before_journal = _artifact_snapshot(journal_path)
    before_ledger = _sqlite_family_snapshot(ledger_path)
    real_journal = SafetyJournal

    def fail(stage, _event) -> None:
        if stage == fault_stage:
            raise RuntimeError(f"injected {fault_stage} recovery crash")

    monkeypatch.setattr(
        journal_script,
        "SafetyJournal",
        lambda path: real_journal(path, fault_hook=fail),
    )
    _enable_stopped_offline_recovery(monkeypatch)
    with pytest.raises(RuntimeError, match="injected"):
        journal_script.recover_exact_local_paper_settlement(
            environ,
            confirmation=journal_script.RECOVER_CONFIRMATION,
        )
    assert _artifact_snapshot(journal_path) == before_journal
    assert _sqlite_family_snapshot(ledger_path) == before_ledger

    monkeypatch.setattr(journal_script, "SafetyJournal", real_journal)
    recovered = journal_script.recover_exact_local_paper_settlement(
        environ,
        confirmation=journal_script.RECOVER_CONFIRMATION,
    )
    assert recovered.terminal_sequence is not None
    state = real_journal(journal_path).replay()
    assert state.active_reservations == ()
    assert (
        len(
            [
                event
                for event in state.events
                if event.event_type is JournalEventType.TERMINAL_RECONCILED
            ]
        )
        == 1
    )
    assert _sqlite_family_snapshot(ledger_path) == before_ledger
