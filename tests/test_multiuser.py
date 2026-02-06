"""Tests for multiuser/multi-portfolio support."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from robo_trader.multiuser.migration import MultiuserMigration
from robo_trader.multiuser.portfolio_config import PortfolioConfig, load_portfolio_configs


# ──────────────────────────────────────────────
# PortfolioConfig Tests
# ──────────────────────────────────────────────

class TestPortfolioConfig:
    def test_basic_creation(self):
        cfg = PortfolioConfig(id="test", name="Test Portfolio", starting_cash=50000, symbols=["AAPL", "MSFT"])
        assert cfg.id == "test"
        assert cfg.name == "Test Portfolio"
        assert cfg.starting_cash == 50000
        assert cfg.symbols == ["AAPL", "MSFT"]
        assert cfg.active is True

    def test_risk_override_fallback(self):
        cfg = PortfolioConfig(id="test", name="Test", max_position_pct=0.04)
        # Has override
        assert cfg.get_risk_param("max_position_pct", 0.02) == 0.04
        # Falls back to global
        assert cfg.get_risk_param("max_daily_loss_pct", 0.005) == 0.005

    def test_to_dict_roundtrip(self):
        original = PortfolioConfig(
            id="aggressive",
            name="Aggressive Growth",
            starting_cash=75000,
            symbols=["NVDA", "TSLA"],
            max_position_pct=0.04,
            enabled_strategies=["momentum", "ml_enhanced"],
        )
        d = original.to_dict()
        restored = PortfolioConfig.from_dict(d)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.starting_cash == original.starting_cash
        assert restored.symbols == original.symbols
        assert restored.max_position_pct == original.max_position_pct
        assert restored.enabled_strategies == original.enabled_strategies

    def test_from_dict_with_string_symbols(self):
        cfg = PortfolioConfig.from_dict({"id": "test", "name": "T", "symbols": "AAPL,MSFT,SPY"})
        assert cfg.symbols == ["AAPL", "MSFT", "SPY"]

    def test_from_dict_with_list_symbols(self):
        cfg = PortfolioConfig.from_dict({"id": "test", "name": "T", "symbols": ["AAPL", "MSFT"]})
        assert cfg.symbols == ["AAPL", "MSFT"]


class TestLoadPortfolioConfigs:
    def test_default_portfolio_when_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PORTFOLIOS", None)
            configs = load_portfolio_configs()
            assert len(configs) == 1
            assert configs[0].id == "default"
            assert configs[0].name == "Default Portfolio"

    def test_default_portfolio_uses_symbols_env(self):
        env = {"SYMBOLS": "NVDA,AMD,TSLA", "DEFAULT_CASH": "75000"}
        with patch.dict(os.environ, env, clear=True):
            configs = load_portfolio_configs()
            assert len(configs) == 1
            assert configs[0].symbols == ["NVDA", "AMD", "TSLA"]
            assert configs[0].starting_cash == 75000

    def test_multi_portfolio_from_json(self):
        portfolios = [
            {"id": "aggressive", "name": "Aggressive", "starting_cash": 50000, "symbols": "NVDA,TSLA"},
            {"id": "conservative", "name": "Conservative", "starting_cash": 50000, "symbols": "AAPL,MSFT"},
        ]
        env = {"PORTFOLIOS": json.dumps(portfolios)}
        with patch.dict(os.environ, env, clear=True):
            configs = load_portfolio_configs()
            assert len(configs) == 2
            assert configs[0].id == "aggressive"
            assert configs[1].id == "conservative"
            assert configs[0].symbols == ["NVDA", "TSLA"]
            assert configs[1].symbols == ["AAPL", "MSFT"]

    def test_duplicate_id_raises(self):
        portfolios = [
            {"id": "same", "name": "A", "symbols": "AAPL"},
            {"id": "same", "name": "B", "symbols": "MSFT"},
        ]
        env = {"PORTFOLIOS": json.dumps(portfolios)}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Duplicate portfolio id"):
                load_portfolio_configs()

    def test_missing_id_raises(self):
        portfolios = [{"name": "No ID", "symbols": "AAPL"}]
        env = {"PORTFOLIOS": json.dumps(portfolios)}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="must have an 'id' field"):
                load_portfolio_configs()

    def test_invalid_json_raises(self):
        env = {"PORTFOLIOS": "not valid json"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Invalid PORTFOLIOS JSON"):
                load_portfolio_configs()

    def test_empty_array_raises(self):
        env = {"PORTFOLIOS": "[]"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="non-empty JSON array"):
                load_portfolio_configs()


# ──────────────────────────────────────────────
# Migration Tests
# ──────────────────────────────────────────────

@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    db_path.unlink(missing_ok=True)
    # Cleanup backup files too
    for backup in db_path.parent.glob(f"{db_path.stem}.backup_*"):
        backup.unlink(missing_ok=True)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def create_legacy_schema(db_path: Path):
    """Create the legacy (pre-multiuser) database schema with test data."""
    import aiosqlite

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")

        # Positions table (legacy - no portfolio_id)
        await conn.execute("""
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                market_price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol)
            )
        """)

        # Trades table (legacy)
        await conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        """)

        # Account table (legacy - id=1)
        await conn.execute("""
            CREATE TABLE account (
                id INTEGER PRIMARY KEY,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Equity history (legacy)
        await conn.execute("""
            CREATE TABLE equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                equity REAL NOT NULL,
                cash REAL DEFAULT 0,
                positions_value REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Signals (legacy)
        await conn.execute("""
            CREATE TABLE signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert test data
        await conn.execute(
            "INSERT INTO positions (symbol, quantity, avg_cost, market_price) VALUES (?, ?, ?, ?)",
            ("AAPL", 100, 180.50, 185.00),
        )
        await conn.execute(
            "INSERT INTO positions (symbol, quantity, avg_cost, market_price) VALUES (?, ?, ?, ?)",
            ("NVDA", 50, 450.00, 470.00),
        )
        await conn.execute(
            "INSERT INTO trades (symbol, side, quantity, price, notional) VALUES (?, ?, ?, ?, ?)",
            ("AAPL", "BUY", 100, 180.50, 18050.0),
        )
        await conn.execute(
            "INSERT INTO account (id, cash, equity) VALUES (1, 80000, 100000)"
        )
        await conn.execute(
            "INSERT INTO equity_history (date, equity, cash, positions_value) VALUES (?, ?, ?, ?)",
            ("2026-02-05", 100000, 80000, 20000),
        )
        await conn.execute(
            "INSERT INTO signals (symbol, strategy, signal_type, strength) VALUES (?, ?, ?, ?)",
            ("AAPL", "momentum", "BUY", 0.8),
        )

        await conn.commit()


class TestMultiuserMigration:
    @pytest.mark.asyncio
    async def test_migration_on_legacy_db(self, temp_db):
        """Test that migration correctly adds portfolio_id to all tables."""
        import aiosqlite

        # Create legacy schema with data
        await create_legacy_schema(temp_db)

        # Run migration
        migration = MultiuserMigration(temp_db)
        assert await migration.needs_migration() is True
        result = await migration.migrate(default_cash=100000)
        assert result is True

        # Verify migration applied
        assert await migration.needs_migration() is False

        # Verify data integrity
        async with aiosqlite.connect(temp_db) as conn:
            # Check portfolios table exists with default entry
            cursor = await conn.execute("SELECT id, name, starting_cash FROM portfolios")
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "default"

            # Check positions have portfolio_id
            cursor = await conn.execute("SELECT portfolio_id, symbol, quantity FROM positions")
            rows = await cursor.fetchall()
            assert len(rows) == 2
            assert all(row[0] == "default" for row in rows)

            # Check trades have portfolio_id
            cursor = await conn.execute("SELECT portfolio_id, symbol, side FROM trades")
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "default"

            # Check account uses portfolio_id instead of id=1
            cursor = await conn.execute("SELECT portfolio_id, cash, equity FROM account")
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "default"
            assert rows[0][1] == 80000  # cash preserved

            # Check equity_history has portfolio_id
            cursor = await conn.execute("SELECT portfolio_id, date, equity FROM equity_history")
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "default"

            # Check signals have portfolio_id
            cursor = await conn.execute("SELECT portfolio_id, symbol, strategy FROM signals")
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "default"

    @pytest.mark.asyncio
    async def test_migration_is_idempotent(self, temp_db):
        """Running migration twice should be safe."""
        await create_legacy_schema(temp_db)

        migration = MultiuserMigration(temp_db)
        result1 = await migration.migrate()
        assert result1 is True

        result2 = await migration.migrate()
        assert result2 is False  # Already applied

    @pytest.mark.asyncio
    async def test_migration_creates_backup(self, temp_db):
        """Migration should create a backup file."""
        await create_legacy_schema(temp_db)

        migration = MultiuserMigration(temp_db)
        await migration.migrate()

        backups = list(temp_db.parent.glob(f"{temp_db.stem}.backup_premultiuser_*"))
        assert len(backups) >= 1

    @pytest.mark.asyncio
    async def test_migration_on_empty_db(self, temp_db):
        """Migration on a fresh/empty database should work."""
        import aiosqlite

        # Create empty DB with just the tables (no data)
        async with aiosqlite.connect(temp_db) as conn:
            await conn.execute("""
                CREATE TABLE positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    market_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            """)
            await conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            """)
            await conn.execute("""
                CREATE TABLE account (
                    id INTEGER PRIMARY KEY,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    equity REAL NOT NULL,
                    cash REAL DEFAULT 0,
                    positions_value REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.commit()

        migration = MultiuserMigration(temp_db)
        result = await migration.migrate()
        assert result is True

    @pytest.mark.asyncio
    async def test_unique_constraint_portfolio_symbol(self, temp_db):
        """Two portfolios can hold the same symbol independently."""
        import aiosqlite

        await create_legacy_schema(temp_db)
        migration = MultiuserMigration(temp_db)
        await migration.migrate()

        async with aiosqlite.connect(temp_db) as conn:
            # Portfolio A has AAPL (from migration, portfolio_id='default')
            # Portfolio B can also have AAPL
            await conn.execute(
                "INSERT INTO positions (portfolio_id, symbol, quantity, avg_cost) VALUES (?, ?, ?, ?)",
                ("portfolio_b", "AAPL", 50, 190.00),
            )
            await conn.commit()

            cursor = await conn.execute("SELECT portfolio_id, symbol, quantity FROM positions WHERE symbol = 'AAPL'")
            rows = await cursor.fetchall()
            assert len(rows) == 2
            portfolios = {row[0] for row in rows}
            assert portfolios == {"default", "portfolio_b"}

    @pytest.mark.asyncio
    async def test_no_db_file_returns_false(self, temp_db):
        """If no database file exists, migration returns False."""
        temp_db.unlink()
        migration = MultiuserMigration(temp_db)
        result = await migration.migrate()
        assert result is False
