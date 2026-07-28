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
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from robo_trader.bootstrap_evidence_auth import (
    PUBLIC_KEY_ENV,
    BootstrapEvidenceAuthenticationError,
    emit_broker_snapshot_receipt,
    emit_protective_mark_receipt,
    emit_reconciliation_report_receipt,
)
from robo_trader.config import RuntimeContract, _derive_safety_account_scope
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.financial_state_bootstrap import (
    ExactBootstrapAccount,
    ExactBootstrapPosition,
    ExactStateBootstrapBackupReceipt,
    ExactStateBootstrapCandidate,
    ExactStateBootstrapCommittedBackupInvalid,
    ExactStateBootstrapError,
    inspect_legacy_state,
    load_exact_state_bootstrap_evidence,
    sqlite_table_evidence,
    verified_file_sha256,
)

ACCOUNT_SCOPE = _derive_safety_account_scope("0123456789abcdef" * 4, "DU_TEST_PAPER")
_TEST_PRIVATE_KEYS: dict[str, Path] = {}


@pytest.fixture(autouse=True)
def _bootstrap_evidence_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    producers = ("broker_snapshot", "reconciliation_report", "protective_mark")
    for kind in producers:
        private_key = Ed25519PrivateKey.generate()
        private_path = tmp_path / f"{kind}.private.pem"
        public_path = tmp_path / f"{kind}.public.pem"
        private_path.write_bytes(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        public_path.write_bytes(
            private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        private_path.chmod(0o400)
        public_path.chmod(0o400)
        _TEST_PRIVATE_KEYS[kind] = private_path
        monkeypatch.setenv(PUBLIC_KEY_ENV[kind], str(public_path))
    monkeypatch.delenv("SAFETY_ACCOUNT_SCOPE_KEY", raising=False)


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


def _write_artifact(
    path: Path,
    payload: dict[str, object],
    *,
    artifact_kind: str,
    issued_at: datetime | None = None,
) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    path.chmod(0o600)
    artifact_hash = hashlib.sha256(raw).hexdigest()
    receipt_path = path.with_name(path.name + ".auth.json")
    if receipt_path.exists():
        receipt_path.unlink()
    producer = {
        "broker_snapshot": emit_broker_snapshot_receipt,
        "reconciliation_report": emit_reconciliation_report_receipt,
        "protective_mark": emit_protective_mark_receipt,
    }[artifact_kind]
    producer(
        artifact_path=path,
        private_key_path=_TEST_PRIVATE_KEYS[artifact_kind],
        runtime_fingerprint=str(payload["runtime_fingerprint"]),
        account_scope=str(payload["account_scope"]),
        issued_at=issued_at,
    )
    return artifact_hash


def _candidate_bundle(
    path: Path,
    artifact_root: Path,
    *,
    legacy_hash: str | None = None,
    flat: bool = False,
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
        artifact_kind="broker_snapshot",
    )
    mark_specs = (
        ()
        if flat
        else (
            ("NVDA", "326.7", "mark-nvda-v1", 123),
            ("TSLA", "369.57", "mark-tsla-v1", 456),
        )
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
            artifact_kind="protective_mark",
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
        artifact_kind="reconciliation_report",
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
            cash=Decimal("100000") if flat else Decimal("96739.16"),
            realized_pnl=Decimal("0"),
            daily_pnl=Decimal("0"),
            daily_pnl_baseline=Decimal("0") if flat else Decimal("1039.18"),
            daily_pnl_date=effective.date(),
        ),
        positions=(
            ()
            if flat
            else (
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
            )
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
    target.chmod(0o400)
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
async def test_bootstrap_is_insert_only_exact_and_receipt_replay_is_rejected(
    tmp_path: Path,
) -> None:
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
        with pytest.raises(ExactStateBootstrapError, match="receipt replay"):
            load_exact_state_bootstrap_evidence(
                reconciliation_path=tmp_path / "reconciliation.json",
                broker_snapshot_path=tmp_path / "broker.json",
                protective_mark_paths=[
                    tmp_path / "mark-NVDA.json",
                    tmp_path / "mark-TSLA.json",
                ],
                expected_runtime_contract=runtime_contract,
            )
        with pytest.raises(ExactStateBootstrapError, match="receipt replay"):
            await database.apply_exact_state_bootstrap(
                candidate,
                evidence=evidence,
                backup_receipt=backup_receipt,
                operator_reason="Seal the reviewed legacy simulator accounting epoch.",
                runtime_contract=runtime_contract,
            )
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
            "SELECT COUNT(*) FROM exact_bootstrap_evidence_consumptions "
            "WHERE bootstrap_id=? AND runtime_fingerprint=? AND account_scope=?",
            (candidate.bootstrap_id, runtime_contract.fingerprint, ACCOUNT_SCOPE),
        ).fetchone() == (4,)
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
    _write_artifact(broker_path, broker, artifact_kind="broker_snapshot")

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
    backup_path.chmod(0o600)
    with backup_path.open("r+b") as stream:
        stream.seek(100)
        original = stream.read(1)
        stream.seek(100)
        stream.write(bytes([original[0] ^ 0x01]))
    try:
        with pytest.raises(ExactStateBootstrapError, match="identity|restore"):
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


def test_external_json_without_authenticated_receipt_has_no_authority(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    (tmp_path / "broker.json.auth.json").unlink()
    with pytest.raises(ExactStateBootstrapError, match="authentication receipt"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=tmp_path / "broker.json",
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


def test_current_authentication_cannot_refresh_thirty_day_old_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    broker_path = tmp_path / "broker.json"
    broker = json.loads(broker_path.read_text())
    stale = datetime.now(timezone.utc) - timedelta(days=30)
    broker["observed_at"] = stale.isoformat()
    broker["broker_time_before"] = (stale - timedelta(seconds=1)).isoformat()
    broker["broker_time_after"] = (stale + timedelta(seconds=1)).isoformat()
    broker_hash = _write_artifact(broker_path, broker, artifact_kind="broker_snapshot")
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation = json.loads(reconciliation_path.read_text())
    reconciliation["broker_snapshot_hash"] = broker_hash
    _write_artifact(
        reconciliation_path,
        reconciliation,
        artifact_kind="reconciliation_report",
    )
    with pytest.raises(ExactStateBootstrapError, match="wall clock"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=reconciliation_path,
            broker_snapshot_path=broker_path,
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


def test_expired_producer_receipt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    broker_path = tmp_path / "broker.json"
    broker = json.loads(broker_path.read_text())
    _write_artifact(
        broker_path,
        broker,
        artifact_kind="broker_snapshot",
        issued_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    with pytest.raises(ExactStateBootstrapError, match="expired"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=broker_path,
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("schema_version", True), ("managed_account_count", True)],
)
def test_broker_integer_fields_reject_json_booleans(
    tmp_path: Path, field: str, invalid: object
) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    broker_path = tmp_path / "broker.json"
    broker = json.loads(broker_path.read_text())
    broker[field] = invalid
    _write_artifact(broker_path, broker, artifact_kind="broker_snapshot")
    with pytest.raises(ExactStateBootstrapError, match="JSON integer"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=broker_path,
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


@pytest.mark.parametrize("field", ["broker_positions_count", "database_device", "database_inode"])
def test_reconciliation_integer_fields_reject_json_floats(tmp_path: Path, field: str) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation = json.loads(reconciliation_path.read_text())
    reconciliation[field] = float(reconciliation[field])
    _write_artifact(
        reconciliation_path,
        reconciliation,
        artifact_kind="reconciliation_report",
    )
    with pytest.raises(ExactStateBootstrapError, match="JSON integer"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=reconciliation_path,
            broker_snapshot_path=tmp_path / "broker.json",
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


def test_fabricated_current_authentication_receipt_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    receipt_path = tmp_path / "broker.json.auth.json"
    monkeypatch.setenv("SAFETY_ACCOUNT_SCOPE_KEY", "fedcba9876543210" * 4)
    receipt = json.loads(receipt_path.read_text())
    receipt["issued_at"] = datetime.now(timezone.utc).isoformat()
    receipt["signature_ed25519"] = "AAAA"
    _write_artifact_receipt = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt_path.chmod(0o600)
    receipt_path.write_text(_write_artifact_receipt)
    receipt_path.chmod(0o400)
    with pytest.raises(ExactStateBootstrapError, match="signature is invalid"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=tmp_path / "broker.json",
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


def test_production_emitters_bind_all_three_distinct_producers(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, evidence, _ = _candidate_bundle(path, tmp_path)
    assert {
        (receipt.artifact_kind, receipt.producer_id) for receipt in evidence.authentication_receipts
    } == {
        ("broker_snapshot", "robotrader-broker-snapshot-producer-v1"),
        ("reconciliation_report", "robotrader-reconciliation-producer-v1"),
        ("protective_mark", "robotrader-protective-mark-producer-v1"),
    }
    assert len({receipt.receipt_id for receipt in evidence.authentication_receipts}) == 4


def test_bootstrap_consumer_refuses_private_signing_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    monkeypatch.setenv(
        "BOOTSTRAP_BROKER_EVIDENCE_PRIVATE_KEY_PATH",
        str(_TEST_PRIVATE_KEYS["broker_snapshot"]),
    )
    with pytest.raises(ExactStateBootstrapError, match="refuses producer signing-key presence"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=tmp_path / "broker.json",
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


@pytest.mark.parametrize("attack", ["wrong_key", "wrong_kind"])
def test_wrong_producer_key_or_kind_cannot_authenticate_broker(tmp_path: Path, attack: str) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    receipt_path = tmp_path / "broker.json.auth.json"
    receipt_path.unlink()
    if attack == "wrong_key":
        emit_broker_snapshot_receipt(
            artifact_path=tmp_path / "broker.json",
            private_key_path=_TEST_PRIVATE_KEYS["protective_mark"],
            runtime_fingerprint=runtime.fingerprint,
            account_scope=ACCOUNT_SCOPE,
        )
    else:
        emit_protective_mark_receipt(
            artifact_path=tmp_path / "broker.json",
            private_key_path=_TEST_PRIVATE_KEYS["protective_mark"],
            runtime_fingerprint=runtime.fingerprint,
            account_scope=ACCOUNT_SCOPE,
        )
    with pytest.raises(ExactStateBootstrapError, match="wrong key|wrong producer|wrong.*kind"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=tmp_path / "broker.json",
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


@pytest.mark.parametrize("attack", ["writable", "hardlink"])
def test_public_verification_key_permissions_and_identity_are_strict(
    tmp_path: Path, attack: str
) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    public_path = Path(os.environ[PUBLIC_KEY_ENV["broker_snapshot"]])
    if attack == "writable":
        public_path.chmod(0o600)
    else:
        os.link(public_path, tmp_path / "broker-public-alias.pem")
    with pytest.raises(ExactStateBootstrapError, match="sealed owner file"):
        load_exact_state_bootstrap_evidence(
            reconciliation_path=tmp_path / "reconciliation.json",
            broker_snapshot_path=tmp_path / "broker.json",
            protective_mark_paths=[tmp_path / "mark-NVDA.json", tmp_path / "mark-TSLA.json"],
            expected_runtime_contract=runtime,
        )


def test_producer_private_key_must_be_sealed_owner_readonly(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    _, _, runtime = _candidate_bundle(path, tmp_path)
    private_path = _TEST_PRIVATE_KEYS["broker_snapshot"]
    private_path.chmod(0o600)
    (tmp_path / "broker.json.auth.json").unlink()
    with pytest.raises(BootstrapEvidenceAuthenticationError, match="sealed owner file"):
        emit_broker_snapshot_receipt(
            artifact_path=tmp_path / "broker.json",
            private_key_path=private_path,
            runtime_fingerprint=runtime.fingerprint,
            account_scope=ACCOUNT_SCOPE,
        )


@pytest.mark.asyncio
async def test_loaded_evidence_copy_cannot_change_authenticated_claims(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, evidence, runtime = _candidate_bundle(path, tmp_path)
    copied = replace(evidence, broker_observed_at=datetime.now(timezone.utc))
    database = AsyncTradingDatabase(path, pool_size=1)

    await database.initialize()
    receipt = _backup_receipt(path, tmp_path / "copy-backup.db", candidate)
    try:
        with pytest.raises(ExactStateBootstrapError, match="not producer-owned"):
            await database.apply_exact_state_bootstrap(
                candidate,
                evidence=copied,
                backup_receipt=receipt,
                operator_reason="Reject copied evidence with changed claims.",
                runtime_contract=runtime,
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_flat_fresh_database_bootstraps_and_runtime_lineage_is_true(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    await database.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_account_settlement_state"
        ).fetchone() == (0,)
    candidate, evidence, runtime = _candidate_bundle(path, tmp_path, flat=True)
    receipt = _backup_receipt(path, tmp_path / "fresh-backup.db", candidate)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    try:
        await database.apply_exact_state_bootstrap(
            candidate,
            evidence=evidence,
            backup_receipt=receipt,
            operator_reason="Seal a reviewed flat fresh paper ledger.",
            runtime_contract=runtime,
        )
        account = await database.get_account_info(runtime_contract=runtime)
        positions = await database.get_positions(runtime_contract=runtime)
    finally:
        await database.close()
    assert account["bootstrap_lineage_valid"] is True
    assert positions == []


@pytest.mark.asyncio
async def test_matching_unlineaged_exact_state_is_adopted_without_value_rewrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "upgrade.db"
    _legacy_database(path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    candidate, evidence, runtime = _candidate_bundle(path, tmp_path)
    async with database.get_connection() as connection:
        await connection.execute(
            """
            INSERT INTO paper_account_settlement_state VALUES
            ('default','96739.16','0','0','1039.18',?,? ,NULL,NULL)
            """,
            (candidate.account.daily_pnl_date.isoformat(), datetime.now(timezone.utc).isoformat()),
        )
        for position in candidate.positions:
            await connection.execute(
                """
                INSERT INTO paper_position_settlement_state VALUES
                ('default',?,?,?,NULL,?,NULL)
                """,
                (
                    position.symbol,
                    position.public_dict()["cost_basis_text"],
                    position.public_dict()["mark_price_text"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        await connection.commit()
    before = None
    with sqlite3.connect(path) as connection:
        before = connection.execute(
            "SELECT cash_text,realized_pnl_text,daily_pnl_text,daily_pnl_baseline_text,daily_pnl_date "
            "FROM paper_account_settlement_state"
        ).fetchone()
    receipt = _backup_receipt(path, tmp_path / "upgrade-backup.db", candidate)
    try:
        await database.apply_exact_state_bootstrap(
            candidate,
            evidence=evidence,
            backup_receipt=receipt,
            operator_reason="Adopt exact current-main state after full evidence review.",
            runtime_contract=runtime,
        )
    finally:
        await database.close()
    with sqlite3.connect(path) as connection:
        after = connection.execute(
            "SELECT cash_text,realized_pnl_text,daily_pnl_text,daily_pnl_baseline_text,daily_pnl_date "
            "FROM paper_account_settlement_state"
        ).fetchone()
        assert before == after
        assert connection.execute(
            "SELECT origin_bootstrap_id FROM paper_account_settlement_state"
        ).fetchone() == (candidate.bootstrap_id,)
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_position_settlement_state " "WHERE origin_bootstrap_id = ?",
            (candidate.bootstrap_id,),
        ).fetchone() == (2,)


@pytest.mark.asyncio
async def test_mismatched_unlineaged_exact_state_blocks_with_zero_mutation(tmp_path: Path) -> None:
    path = tmp_path / "upgrade-mismatch.db"
    _legacy_database(path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    candidate, evidence, runtime = _candidate_bundle(path, tmp_path)
    async with database.get_connection() as connection:
        await connection.execute(
            """
            INSERT INTO paper_account_settlement_state VALUES
            ('default','1','0','0','1039.18',?,?,NULL,NULL)
            """,
            (candidate.account.daily_pnl_date.isoformat(), datetime.now(timezone.utc).isoformat()),
        )
        await connection.commit()
    receipt = _backup_receipt(path, tmp_path / "mismatch-backup.db", candidate)
    try:
        with pytest.raises(ExactStateBootstrapError, match="does not exactly match"):
            await database.apply_exact_state_bootstrap(
                candidate,
                evidence=evidence,
                backup_receipt=receipt,
                operator_reason="Reject mismatched exact current-main state.",
                runtime_contract=runtime,
            )
    finally:
        await database.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT cash_text,origin_bootstrap_id FROM paper_account_settlement_state"
        ).fetchone() == ("1", None)
        assert connection.execute("SELECT COUNT(*) FROM paper_state_bootstraps").fetchone() == (0,)


@pytest.mark.asyncio
async def test_post_commit_alias_corruption_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, evidence, runtime = _candidate_bundle(path, tmp_path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    backup_path = tmp_path / "sealed-backup.db"
    receipt = _backup_receipt(path, backup_path, candidate)
    original = backup_path.stat()
    alias = tmp_path / "backup-alias.db"

    def corrupt_after_commit(step: str) -> None:
        if step != "AFTER_EXACT_BOOTSTRAP_COMMIT":
            return
        os.link(backup_path, alias)
        alias.chmod(0o600)
        with alias.open("r+b") as stream:
            stream.seek(100)
            value = stream.read(1)
            stream.seek(100)
            stream.write(bytes([value[0] ^ 1]))
        alias.chmod(0o400)
        os.utime(alias, ns=(original.st_atime_ns, original.st_mtime_ns))

    database._paper_settlement_fault_hook = corrupt_after_commit
    try:
        with pytest.raises(ExactStateBootstrapCommittedBackupInvalid) as captured:
            await database.apply_exact_state_bootstrap(
                candidate,
                evidence=evidence,
                backup_receipt=receipt,
                operator_reason="Prove post-commit backup corruption blocks success.",
                runtime_contract=runtime,
            )
        assert captured.value.status == "COMMITTED_BACKUP_INVALID"
        assert captured.value.mutated_state is True
        assert captured.value.safe_retry is False
        assert captured.value.bootstrap_id == candidate.bootstrap_id
        assert captured.value.candidate_fingerprint == candidate.fingerprint()
    finally:
        await database.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM paper_state_bootstraps").fetchone() == (1,)


@pytest.mark.asyncio
async def test_portfolio_upsert_preserves_bootstrap_foreign_key_row(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    candidate, evidence, runtime = _candidate_bundle(path, tmp_path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    receipt = _backup_receipt(path, tmp_path / "backup.db", candidate)
    try:
        await database.apply_exact_state_bootstrap(
            candidate,
            evidence=evidence,
            backup_receipt=receipt,
            operator_reason="Seal before updating the portfolio definition.",
            runtime_contract=runtime,
        )
        await database.upsert_portfolio({"id": "default", "name": "Renamed Safely"})
    finally:
        await database.close()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        assert connection.execute("SELECT name FROM portfolios WHERE id='default'").fetchone() == (
            "Renamed Safely",
        )
        assert connection.execute("SELECT bootstrap_id FROM paper_state_bootstraps").fetchone() == (
            candidate.bootstrap_id,
        )
