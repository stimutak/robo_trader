from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.config import RuntimeContract, _derive_safety_account_scope
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.financial_state_bootstrap import (
    ExactBootstrapAccount,
    ExactBootstrapPosition,
    ExactStateBootstrapBackupReceipt,
    ExactStateBootstrapCandidate,
    ExactStateBootstrapError,
    inspect_legacy_state,
    load_exact_state_bootstrap_evidence,
    sqlite_table_evidence,
    verified_file_sha256,
)

ACCOUNT_SCOPE = _derive_safety_account_scope("0123456789abcdef" * 4, "DU_TEST_PAPER")


def _legacy_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                starting_cash REAL NOT NULL DEFAULT 100000,
                symbols TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                max_position_pct REAL,
                max_daily_loss_pct REAL,
                max_open_positions INTEGER,
                stop_loss_pct REAL,
                trailing_stop_pct REAL,
                use_trailing_stop INTEGER,
                enabled_strategies TEXT,
                min_confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL DEFAULT 'default',
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                market_price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(portfolio_id, symbol)
            );
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
            );
            CREATE TABLE account (
                portfolio_id TEXT PRIMARY KEY DEFAULT 'default',
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL DEFAULT 'default',
                date TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL DEFAULT 0,
                positions_value REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(portfolio_id, date)
            );
            INSERT INTO portfolios(id,name) VALUES ('default','Default Portfolio');
            INSERT INTO positions(portfolio_id,symbol,quantity,avg_cost,market_price,timestamp)
            VALUES
              ('default','NVDA',9,210.96,326.70,'2026-07-20 22:23:02'),
              ('default','TSLA',2,370.81,203.45,'2026-07-20 22:23:02');
            INSERT INTO account VALUES
              ('default',96739.16,100086.36,226.30,226.30,706.94,'2026-07-20 22:23:02');
            INSERT INTO trades(portfolio_id,symbol,side,quantity,price,notional,timestamp)
            VALUES
              ('default','NVDA','BUY',9,210.96,1898.64,'2026-07-10 21:18:53'),
              ('default','TSLA','BUY',2,370.81,741.62,'2026-07-20 19:40:45');
            INSERT INTO equity_history(
                portfolio_id,date,equity,cash,positions_value,realized_pnl,unrealized_pnl,timestamp
            ) VALUES
              ('default','2026-07-20',100086.36,96739.16,3347.20,226.30,706.94,
               '2026-07-20 22:23:02');
        """)


def _runtime_contract(path: Path) -> RuntimeContract:
    return RuntimeContract(
        environment="test",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(path),
        account_alias="***PER",
        account_type="paper",
        model_artifact_set="test",
        build_id="test-build",
        state_namespace="paper",
        safety_account_scope=ACCOUNT_SCOPE,
        safety_execution_domain_scope="paper-simulator-v1",
    )


def _write_artifact(path: Path, payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _candidate_bundle(
    path: Path,
    artifact_root: Path,
    *,
    legacy_hash: str | None = None,
) -> tuple[ExactStateBootstrapCandidate, object, RuntimeContract]:
    effective = datetime.now(timezone.utc).replace(microsecond=0)
    runtime_contract = _runtime_contract(path)
    legacy_snapshot_hash = legacy_hash or str(inspect_legacy_state(path)["snapshot_hash"])
    broker_path = artifact_root / "broker.json"
    broker_hash = _write_artifact(
        broker_path,
        {
            "schema_version": 1,
            "snapshot_id": "broker-zero-exposure-v1",
            "observed_at": (effective - timedelta(seconds=2)).isoformat(),
            "broker_time_before": (effective - timedelta(seconds=3)).isoformat(),
            "broker_time_after": (effective - timedelta(seconds=1)).isoformat(),
            "runtime_fingerprint": runtime_contract.fingerprint,
            "execution_domain_scope": "paper-simulator-v1",
            "account_scope": ACCOUNT_SCOPE,
            "broker_host": "127.0.0.1",
            "broker_port": 4002,
            "read_only": True,
            "managed_account_count": 1,
            "positions": [],
            "open_orders": [],
        },
    )
    mark_specs = (
        ("NVDA", "326.7", "mark-nvda-v1", 123),
        ("TSLA", "369.57", "mark-tsla-v1", 456),
    )
    mark_paths: list[Path] = []
    mark_hashes: dict[str, str] = {}
    mark_observed_at = effective - timedelta(seconds=5)
    for symbol, price, event_id, con_id in mark_specs:
        mark_path = artifact_root / f"mark-{symbol}.json"
        mark_hashes[symbol] = _write_artifact(
            mark_path,
            {
                "schema_version": 1,
                "portfolio_id": "default",
                "symbol": symbol,
                "price_text": price,
                "observed_at": mark_observed_at.isoformat(),
                "source": "pr3-validated-market-data-v1",
                "source_event_id": event_id,
                "con_id": con_id,
                "runtime_fingerprint": runtime_contract.fingerprint,
                "execution_domain_scope": "paper-simulator-v1",
                "account_scope": ACCOUNT_SCOPE,
            },
        )
        mark_paths.append(mark_path)
    metadata = path.stat()
    reconciliation_path = artifact_root / "reconciliation.json"
    reconciliation_hash = _write_artifact(
        reconciliation_path,
        {
            "schema_version": 1,
            "snapshot_id": "recon-zero-exposure-v1",
            "generated_at": (effective - timedelta(seconds=1)).isoformat(),
            "runtime_fingerprint": runtime_contract.fingerprint,
            "execution_domain_scope": "paper-simulator-v1",
            "account_scope": ACCOUNT_SCOPE,
            "database_path": str(path),
            "database_identity": runtime_contract.database_identity,
            "database_device": metadata.st_dev,
            "database_inode": metadata.st_ino,
            "portfolio_ids": ["default"],
            "legacy_snapshot_hash": legacy_snapshot_hash,
            "broker_snapshot_id": "broker-zero-exposure-v1",
            "broker_snapshot_hash": broker_hash,
            "status": "BOOTSTRAP_EVIDENCE_COMPLETE",
            "authorizes_startup": False,
            "mutated_state": False,
            "managed_account_count": 1,
            "broker_positions_count": 0,
            "broker_open_orders_count": 0,
        },
    )
    candidate = ExactStateBootstrapCandidate(
        bootstrap_id="pboot-" + ("a" * 32),
        execution_domain_scope="paper-simulator-v1",
        account_scope=ACCOUNT_SCOPE,
        portfolio_id="default",
        database_path=str(path),
        database_identity=runtime_contract.database_identity,
        reconciliation_snapshot_id="recon-zero-exposure-v1",
        reconciliation_report_hash=reconciliation_hash,
        broker_snapshot_hash=broker_hash,
        legacy_snapshot_hash=legacy_snapshot_hash,
        broker_position_count=0,
        broker_open_order_count=0,
        effective_at=effective,
        account=ExactBootstrapAccount(
            cash=Decimal("96739.16"),
            realized_pnl=Decimal("0"),
            daily_pnl=Decimal("0"),
            daily_pnl_baseline=Decimal("1039.18"),
            daily_pnl_date=effective.date(),
        ),
        positions=(
            ExactBootstrapPosition(
                symbol="NVDA",
                quantity=9,
                cost_basis=Decimal("210.96"),
                mark_price=Decimal("326.70"),
                mark_observed_at=effective - timedelta(seconds=5),
                mark_evidence_fingerprint=mark_hashes["NVDA"],
            ),
            ExactBootstrapPosition(
                symbol="TSLA",
                quantity=2,
                cost_basis=Decimal("370.81"),
                # The known-bad legacy 203.45 mark is deliberately not adopted.
                mark_price=Decimal("369.57"),
                mark_observed_at=effective - timedelta(seconds=5),
                mark_evidence_fingerprint=mark_hashes["TSLA"],
            ),
        ),
    )
    evidence = load_exact_state_bootstrap_evidence(
        reconciliation_path=reconciliation_path,
        broker_snapshot_path=broker_path,
        protective_mark_paths=mark_paths,
        expected_runtime_contract=runtime_contract,
    )
    return candidate, evidence, runtime_contract


def _backup_receipt(
    source: Path,
    target: Path,
    candidate: ExactStateBootstrapCandidate,
) -> ExactStateBootstrapBackupReceipt:
    with sqlite3.connect(source) as source_connection, sqlite3.connect(target) as target_connection:
        source_connection.backup(target_connection)
    target.chmod(0o600)
    backup_hash, backup_metadata = verified_file_sha256(target, "test backup")
    source_metadata = source.stat()
    with sqlite3.connect(target.as_uri() + "?mode=ro", uri=True) as connection:
        row_counts, table_hashes = sqlite_table_evidence(connection)
    return ExactStateBootstrapBackupReceipt(
        schema_version=1,
        created_at=datetime.now(timezone.utc),
        candidate_fingerprint=candidate.fingerprint(),
        source_path=str(source),
        source_device=source_metadata.st_dev,
        source_inode=source_metadata.st_ino,
        backup_path=str(target),
        backup_device=backup_metadata.st_dev,
        backup_inode=backup_metadata.st_ino,
        integrity_check="ok",
        source_snapshot_hash=candidate.legacy_snapshot_hash,
        row_counts=row_counts,
        table_hashes=table_hashes,
        backup_content_hash=backup_hash,
    )


def _legacy_rows(path: Path) -> dict[str, list[tuple]]:
    with sqlite3.connect(path) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in ("account", "positions", "trades", "equity_history")
        }


@pytest.mark.asyncio
async def test_bootstrap_is_insert_only_exact_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, evidence, runtime_contract = _candidate_bundle(path, tmp_path)
    before = _legacy_rows(path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    backup_receipt = _backup_receipt(path, tmp_path / "backup.db", candidate)
    try:
        receipt = await database.apply_exact_state_bootstrap(
            candidate,
            evidence=evidence,
            backup_receipt=backup_receipt,
            operator_reason="Seal the reviewed legacy simulator accounting epoch.",
            runtime_contract=runtime_contract,
        )
        replay = await database.apply_exact_state_bootstrap(
            candidate,
            evidence=evidence,
            backup_receipt=backup_receipt,
            operator_reason="Seal the reviewed legacy simulator accounting epoch.",
            runtime_contract=runtime_contract,
        )
        assert replay == receipt
        positions = await database.get_positions(runtime_contract=runtime_contract)
        account = await database.get_account_info(runtime_contract=runtime_contract)
    finally:
        await database.close()

    assert _legacy_rows(path) == before
    assert account["cash_exact"] == Decimal("96739.16")
    assert account["bootstrap_lineage_valid"] is True
    assert {row["symbol"]: row["market_price_exact"] for row in positions} == {
        "NVDA": Decimal("326.7"),
        "TSLA": Decimal("369.57"),
    }
    assert all(row["bootstrap_lineage_valid"] is True for row in positions)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM paper_state_bootstraps").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM administrator_actions").fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_reduction_settlements"
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_changed_legacy_snapshot_rolls_back_every_bootstrap_row(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, evidence, runtime_contract = _candidate_bundle(
        path,
        tmp_path,
        legacy_hash="f" * 64,
    )
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    backup_receipt = _backup_receipt(path, tmp_path / "backup.db", candidate)
    try:
        with pytest.raises(ExactStateBootstrapError, match="backup does not restore"):
            await database.apply_exact_state_bootstrap(
                candidate,
                evidence=evidence,
                backup_receipt=backup_receipt,
                operator_reason="Reject a candidate whose reviewed ledger changed.",
                runtime_contract=runtime_contract,
            )
    finally:
        await database.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM paper_state_bootstraps").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM administrator_actions").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_account_settlement_state"
        ).fetchone() == (0,)


def test_candidate_rejects_broker_exposure_and_stale_marks(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    base, _, _ = _candidate_bundle(path, tmp_path)
    values = base.canonical_dict()
    values["broker_position_count"] = 1
    with pytest.raises(ExactStateBootstrapError, match="zero exposure"):
        ExactStateBootstrapCandidate.from_mapping(values)

    values = base.canonical_dict()
    values["positions"][0]["mark_observed_at"] = (
        base.effective_at - timedelta(minutes=6)
    ).isoformat()
    with pytest.raises(ExactStateBootstrapError, match="future or stale"):
        ExactStateBootstrapCandidate.from_mapping(values)

    values = base.canonical_dict()
    values["account"]["cash_text"] = 96739.16
    with pytest.raises(ExactStateBootstrapError, match="not an exact decimal"):
        ExactStateBootstrapCandidate.from_mapping(values)


@pytest.mark.asyncio
async def test_runner_rejects_explicit_missing_bootstrap_lineage(tmp_path: Path) -> None:
    from robo_trader.portfolio import Portfolio
    from robo_trader.runner_async import AsyncRunner, UnprotectedExistingPositionsError

    runner = object.__new__(AsyncRunner)
    runner.cfg = SimpleNamespace(runtime_contract=_runtime_contract(tmp_path / "runner.db"))
    runner.portfolio_id = "default"
    runner.positions = {}
    runner.db = SimpleNamespace(
        get_account_info=AsyncMock(
            return_value={
                "cash_exact": Decimal("100000"),
                "realized_pnl_exact": Decimal("0"),
                "daily_pnl_exact": Decimal("0"),
                "daily_pnl_baseline_exact": Decimal("0"),
                "daily_pnl_date_exact": datetime.utcnow().date(),
                "source_settlement_id": None,
                "bootstrap_lineage_valid": False,
            }
        ),
        get_positions=AsyncMock(
            return_value=[
                {
                    "symbol": "AAPL",
                    "quantity": 1,
                    "avg_cost": 100.0,
                    "market_price_exact": Decimal("101"),
                    "bootstrap_lineage_valid": False,
                }
            ]
        ),
    )
    runner.portfolio = Portfolio(100_000)
    runner.stop_loss_monitor = None
    runner.use_trailing_stop = False
    runner.stop_loss_percent = 0.02
    runner.trailing_stop_pct = 0.05
    runner.use_advanced_risk = False
    runner.advanced_risk = None
    runner.daily_pnl = 0.0
    runner.latest_prices = {}
    runner.latest_price_sources = {}
    runner.latest_price_times = {}
    runner._daily_pnl_date = None
    runner._starting_unrealized_today = 0.0
    runner._starting_unrealized_today_exact = Decimal("0")

    with pytest.raises(UnprotectedExistingPositionsError):
        await runner.load_existing_positions()


def test_legacy_inspection_reads_wal_and_hashes_trade_and_equity_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wal.db"
    _legacy_database(path)
    initial = inspect_legacy_state(path)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("UPDATE trades SET price = price + 1 WHERE id = 1")
        connection.execute("UPDATE equity_history SET equity = equity + 1 WHERE id = 1")
        connection.commit()

        observed = inspect_legacy_state(path)
        assert observed["trade_count"] == 2
        assert observed["snapshot_hash"] != initial["snapshot_hash"]
    finally:
        connection.close()


def test_candidate_rejects_inconsistent_exact_daily_pnl(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, _, _ = _candidate_bundle(path, tmp_path)
    raw = candidate.canonical_dict()
    raw["account"]["daily_pnl_baseline_text"] = "0"

    with pytest.raises(ExactStateBootstrapError, match="does not reconcile"):
        ExactStateBootstrapCandidate.from_mapping(raw)


def test_evidence_loader_rejects_tampered_broker_and_wrong_runtime_scope(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime_contract = _candidate_bundle(path, tmp_path)
    broker_path = tmp_path / "broker.json"
    broker = json.loads(broker_path.read_text(encoding="utf-8"))
    broker["positions"] = [{"symbol": "AAPL", "quantity": "1"}]
    _write_artifact(broker_path, broker)

    with pytest.raises(ExactStateBootstrapError, match="zero paper exposure"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=broker_path,
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime_contract,
        )

    _, _, runtime_contract = _candidate_bundle(path, tmp_path)
    wrong_scope = replace(
        runtime_contract,
        safety_account_scope=_derive_safety_account_scope(
            "fedcba9876543210" * 4,
            "DU_OTHER_PAPER",
        ),
    )
    with pytest.raises(ExactStateBootstrapError, match="runtime evidence"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=broker_path,
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=wrong_scope,
        )


@pytest.mark.asyncio
async def test_future_exact_position_inherits_bootstrap_lineage(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, evidence, runtime_contract = _candidate_bundle(path, tmp_path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    receipt = _backup_receipt(path, tmp_path / "backup.db", candidate)
    try:
        await database.apply_exact_state_bootstrap(
            candidate,
            evidence=evidence,
            backup_receipt=receipt,
            operator_reason="Seal exact state before testing inherited lineage.",
            runtime_contract=runtime_contract,
        )
        await database.update_position(
            "AAPL",
            1,
            Decimal("100"),
            Decimal("101"),
            portfolio_id="default",
        )
        positions = await database.get_positions(runtime_contract=runtime_contract)
    finally:
        await database.close()

    assert {row["symbol"]: row["bootstrap_lineage_valid"] for row in positions} == {
        "AAPL": True,
        "NVDA": True,
        "TSLA": True,
    }


@pytest.mark.asyncio
async def test_copied_database_cannot_reuse_old_inode_lineage(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, evidence, runtime_contract = _candidate_bundle(path, tmp_path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    receipt = _backup_receipt(path, tmp_path / "backup.db", candidate)
    try:
        await database.apply_exact_state_bootstrap(
            candidate,
            evidence=evidence,
            backup_receipt=receipt,
            operator_reason="Seal exact state before testing inode replacement.",
            runtime_contract=runtime_contract,
        )
    finally:
        await database.close()

    replacement = tmp_path / "replacement.db"
    shutil.copy2(path, replacement)
    old_inode = path.stat().st_ino
    os.replace(replacement, path)
    assert path.stat().st_ino != old_inode

    copied = AsyncTradingDatabase(path, pool_size=1)
    await copied.initialize()
    try:
        account = await copied.get_account_info(runtime_contract=runtime_contract)
        positions = await copied.get_positions(runtime_contract=runtime_contract)
    finally:
        await copied.close()
    assert account["bootstrap_lineage_valid"] is False
    assert all(row["bootstrap_lineage_valid"] is False for row in positions)


@pytest.mark.asyncio
async def test_runner_rejects_flat_account_without_positive_lineage(tmp_path: Path) -> None:
    from robo_trader.portfolio import Portfolio
    from robo_trader.runner_async import AsyncRunner, UnprotectedExistingPositionsError

    runner = object.__new__(AsyncRunner)
    runner.cfg = SimpleNamespace(runtime_contract=_runtime_contract(tmp_path / "runner.db"))
    runner.portfolio_id = "default"
    runner.positions = {}
    runner.db = SimpleNamespace(
        get_account_info=AsyncMock(
            return_value={
                "cash_exact": Decimal("100000"),
                "realized_pnl_exact": Decimal("0"),
                "daily_pnl_exact": Decimal("0"),
                "daily_pnl_baseline_exact": Decimal("0"),
                "daily_pnl_date_exact": datetime.utcnow().date(),
                "source_settlement_id": None,
                "bootstrap_lineage_valid": None,
            }
        ),
        get_positions=AsyncMock(return_value=[]),
    )
    runner.portfolio = Portfolio(100_000)
    runner._starting_unrealized_today_exact = Decimal("0")
    runner._starting_unrealized_today = 0.0

    with pytest.raises(UnprotectedExistingPositionsError):
        await runner.load_existing_positions()


@pytest.mark.asyncio
async def test_tampered_backup_blocks_before_bootstrap_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, evidence, runtime_contract = _candidate_bundle(path, tmp_path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    backup_path = tmp_path / "backup.db"
    receipt = _backup_receipt(path, backup_path, candidate)
    with sqlite3.connect(backup_path) as connection:
        connection.execute("UPDATE trades SET price = price + 1 WHERE id = 1")
    try:
        with pytest.raises(ExactStateBootstrapError, match="does not restore"):
            await database.apply_exact_state_bootstrap(
                candidate,
                evidence=evidence,
                backup_receipt=receipt,
                operator_reason="Reject a backup modified after verification.",
                runtime_contract=runtime_contract,
            )
    finally:
        await database.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM paper_state_bootstraps").fetchone() == (0,)
