from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.financial_state_bootstrap import (
    ExactBootstrapAccount,
    ExactBootstrapPosition,
    ExactStateBootstrapCandidate,
    ExactStateBootstrapError,
    inspect_legacy_state,
)

ACCOUNT_SCOPE = "acct_v1_" + ("1" * 64)


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


def _database_identity(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"paper:{digest}"


def _candidate(path: Path, *, legacy_hash: str | None = None) -> ExactStateBootstrapCandidate:
    effective = datetime.now(timezone.utc).replace(microsecond=0)
    return ExactStateBootstrapCandidate(
        bootstrap_id="pboot-" + ("a" * 32),
        execution_domain_scope="paper-simulator-v1",
        account_scope=ACCOUNT_SCOPE,
        portfolio_id="default",
        database_path=str(path),
        database_identity=_database_identity(path),
        reconciliation_snapshot_id="recon-zero-exposure-v1",
        reconciliation_report_hash="2" * 64,
        broker_snapshot_hash="3" * 64,
        legacy_snapshot_hash=(legacy_hash or str(inspect_legacy_state(path)["snapshot_hash"])),
        broker_position_count=0,
        broker_open_order_count=0,
        effective_at=effective,
        account=ExactBootstrapAccount(
            cash=Decimal("96739.16"),
            realized_pnl=Decimal("0"),
            daily_pnl=Decimal("0"),
            daily_pnl_baseline=Decimal("0"),
            daily_pnl_date=effective.date(),
        ),
        positions=(
            ExactBootstrapPosition(
                symbol="NVDA",
                quantity=9,
                cost_basis=Decimal("210.96"),
                mark_price=Decimal("326.70"),
                mark_observed_at=effective - timedelta(seconds=5),
                mark_evidence_fingerprint="4" * 64,
            ),
            ExactBootstrapPosition(
                symbol="TSLA",
                quantity=2,
                cost_basis=Decimal("370.81"),
                # The known-bad legacy 203.45 mark is deliberately not adopted.
                mark_price=Decimal("369.57"),
                mark_observed_at=effective - timedelta(seconds=5),
                mark_evidence_fingerprint="5" * 64,
            ),
        ),
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
    candidate = _candidate(path)
    before = _legacy_rows(path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    try:
        receipt = await database.apply_exact_state_bootstrap(
            candidate,
            operator_reason="Seal the reviewed legacy simulator accounting epoch.",
        )
        replay = await database.apply_exact_state_bootstrap(
            candidate,
            operator_reason="Seal the reviewed legacy simulator accounting epoch.",
        )
        assert replay == receipt
        positions = await database.get_positions()
        account = await database.get_account_info()
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
    candidate = _candidate(path, legacy_hash="f" * 64)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    try:
        with pytest.raises(ExactStateBootstrapError, match="changed after"):
            await database.apply_exact_state_bootstrap(
                candidate,
                operator_reason="Reject a candidate whose reviewed ledger changed.",
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
    base = _candidate(path)
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
async def test_runner_rejects_explicit_missing_bootstrap_lineage() -> None:
    from types import SimpleNamespace

    from robo_trader.portfolio import Portfolio
    from robo_trader.runner_async import AsyncRunner, UnprotectedExistingPositionsError

    runner = object.__new__(AsyncRunner)
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
