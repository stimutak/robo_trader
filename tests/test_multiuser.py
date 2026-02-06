"""Tests for multiuser/multi-portfolio support."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.multiuser.db_proxy import PortfolioScopedDB
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


# ──────────────────────────────────────────────
# AsyncTradingDatabase Multi-Portfolio Tests
# ──────────────────────────────────────────────

class TestAsyncDatabaseMultiPortfolio:
    """Test that all DB methods correctly scope by portfolio_id."""

    @pytest.fixture
    async def db(self, temp_db):
        """Create an initialized async database."""
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()
        yield database
        await database.close()

    @pytest.mark.asyncio
    async def test_positions_scoped_by_portfolio(self, db):
        """Positions are isolated per portfolio."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
        await db.update_position("AAPL", 50, 190.0, portfolio_id="portfolio_b")

        pos_a = await db.get_positions(portfolio_id="portfolio_a")
        pos_b = await db.get_positions(portfolio_id="portfolio_b")

        assert len(pos_a) == 1
        assert pos_a[0]["quantity"] == 100
        assert pos_a[0]["avg_cost"] == 180.0

        assert len(pos_b) == 1
        assert pos_b[0]["quantity"] == 50
        assert pos_b[0]["avg_cost"] == 190.0

    @pytest.mark.asyncio
    async def test_close_position_only_affects_portfolio(self, db):
        """Closing position in one portfolio doesn't affect another."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
        await db.update_position("AAPL", 50, 190.0, portfolio_id="portfolio_b")

        # Close AAPL in portfolio_a
        await db.update_position("AAPL", 0, 0, portfolio_id="portfolio_a")

        pos_a = await db.get_positions(portfolio_id="portfolio_a")
        pos_b = await db.get_positions(portfolio_id="portfolio_b")

        assert len(pos_a) == 0  # Closed
        assert len(pos_b) == 1  # Still open

    @pytest.mark.asyncio
    async def test_trades_scoped_by_portfolio(self, db):
        """Trades are isolated per portfolio."""
        await db.record_trade("AAPL", "BUY", 100, 180.0, portfolio_id="portfolio_a")
        await db.record_trade("MSFT", "BUY", 50, 300.0, portfolio_id="portfolio_b")

        trades_a = await db.get_recent_trades(portfolio_id="portfolio_a")
        trades_b = await db.get_recent_trades(portfolio_id="portfolio_b")

        assert len(trades_a) == 1
        assert trades_a[0]["symbol"] == "AAPL"

        assert len(trades_b) == 1
        assert trades_b[0]["symbol"] == "MSFT"

    @pytest.mark.asyncio
    async def test_has_recent_buy_trade_scoped(self, db):
        """Recent buy check is scoped to portfolio."""
        await db.record_trade("AAPL", "BUY", 100, 180.0, portfolio_id="portfolio_a")

        assert await db.has_recent_buy_trade("AAPL", 600, portfolio_id="portfolio_a") is True
        assert await db.has_recent_buy_trade("AAPL", 600, portfolio_id="portfolio_b") is False

    @pytest.mark.asyncio
    async def test_account_scoped_by_portfolio(self, db):
        """Account info is isolated per portfolio."""
        await db.update_account(50000, 55000, portfolio_id="portfolio_a")
        await db.update_account(80000, 82000, portfolio_id="portfolio_b")

        acct_a = await db.get_account_info(portfolio_id="portfolio_a")
        acct_b = await db.get_account_info(portfolio_id="portfolio_b")

        assert acct_a["cash"] == 50000
        assert acct_a["equity"] == 55000

        assert acct_b["cash"] == 80000
        assert acct_b["equity"] == 82000

    @pytest.mark.asyncio
    async def test_equity_history_scoped_by_portfolio(self, db):
        """Equity history is isolated per portfolio."""
        await db.save_equity_snapshot(55000, 50000, 5000, portfolio_id="portfolio_a")
        await db.save_equity_snapshot(82000, 80000, 2000, portfolio_id="portfolio_b")

        hist_a = await db.get_equity_history(portfolio_id="portfolio_a")
        hist_b = await db.get_equity_history(portfolio_id="portfolio_b")

        assert len(hist_a) == 1
        assert hist_a[0]["equity"] == 55000

        assert len(hist_b) == 1
        assert hist_b[0]["equity"] == 82000

    @pytest.mark.asyncio
    async def test_default_portfolio_backward_compat(self, db):
        """Calling without portfolio_id uses 'default' portfolio."""
        await db.update_position("AAPL", 100, 180.0)  # No portfolio_id = 'default'
        pos = await db.get_positions()  # No portfolio_id = 'default'

        assert len(pos) == 1
        assert pos[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_portfolios(self, db):
        """get_portfolios returns all portfolio definitions."""
        await db.upsert_portfolio({"id": "aggressive", "name": "Aggressive Growth", "starting_cash": 50000, "symbols": "NVDA,TSLA"})
        await db.upsert_portfolio({"id": "conservative", "name": "Conservative", "starting_cash": 50000, "symbols": "AAPL,MSFT"})

        portfolios = await db.get_portfolios()
        ids = {p["id"] for p in portfolios}
        assert "default" in ids
        assert "aggressive" in ids
        assert "conservative" in ids

    @pytest.mark.asyncio
    async def test_signals_scoped_by_portfolio(self, db):
        """Signals are scoped to portfolio."""
        await db.record_signal("AAPL", "momentum", "BUY", 0.8, portfolio_id="portfolio_a")
        await db.record_signal("MSFT", "ml_enhanced", "SELL", 0.7, portfolio_id="portfolio_b")

        # Signals are in the signals table but filtered by portfolio
        # We can verify by reading from DB directly
        import aiosqlite
        async with aiosqlite.connect(db._db.db_path if hasattr(db, '_db') else db.db_path) as conn:
            cursor = await conn.execute(
                "SELECT portfolio_id, symbol FROM signals ORDER BY symbol"
            )
            rows = await cursor.fetchall()
            assert len(rows) == 2
            assert rows[0] == ("portfolio_a", "AAPL")
            assert rows[1] == ("portfolio_b", "MSFT")


# ──────────────────────────────────────────────
# PortfolioScopedDB Proxy Tests
# ──────────────────────────────────────────────

class TestPortfolioScopedDB:
    """Test the portfolio-scoped DB proxy."""

    @pytest.fixture
    async def raw_db(self, temp_db):
        """Create a raw async database."""
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()
        yield database
        await database.close()

    @pytest.mark.asyncio
    async def test_proxy_auto_injects_portfolio_id(self, raw_db):
        """Proxy automatically injects portfolio_id into scoped methods."""
        proxy = PortfolioScopedDB(raw_db, portfolio_id="aggressive")

        # Write via proxy (should use portfolio_id="aggressive")
        await proxy.update_position("NVDA", 100, 450.0)
        await proxy.record_trade("NVDA", "BUY", 100, 450.0)
        await proxy.update_account(50000, 55000)

        # Read via proxy (should filter to portfolio_id="aggressive")
        pos = await proxy.get_positions()
        assert len(pos) == 1
        assert pos[0]["symbol"] == "NVDA"

        trades = await proxy.get_recent_trades()
        assert len(trades) == 1
        assert trades[0]["symbol"] == "NVDA"

        acct = await proxy.get_account_info()
        assert acct["cash"] == 50000

    @pytest.mark.asyncio
    async def test_proxy_isolation_between_portfolios(self, raw_db):
        """Two proxies with different portfolio_ids are isolated."""
        proxy_a = PortfolioScopedDB(raw_db, portfolio_id="aggressive")
        proxy_b = PortfolioScopedDB(raw_db, portfolio_id="conservative")

        await proxy_a.update_position("NVDA", 100, 450.0)
        await proxy_b.update_position("AAPL", 50, 180.0)

        pos_a = await proxy_a.get_positions()
        pos_b = await proxy_b.get_positions()

        assert len(pos_a) == 1
        assert pos_a[0]["symbol"] == "NVDA"

        assert len(pos_b) == 1
        assert pos_b[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_proxy_passes_through_global_methods(self, raw_db):
        """Non-scoped methods pass through to underlying DB."""
        proxy = PortfolioScopedDB(raw_db, portfolio_id="aggressive")

        # health_check is a global method (not portfolio-scoped)
        result = await proxy.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_proxy_explicit_portfolio_id_overrides(self, raw_db):
        """Explicitly passing portfolio_id overrides the proxy default."""
        proxy = PortfolioScopedDB(raw_db, portfolio_id="aggressive")

        # Write to a different portfolio explicitly
        await proxy.update_position("MSFT", 200, 400.0, portfolio_id="other")

        # Proxy's default portfolio should have nothing
        pos_default = await proxy.get_positions()
        assert len(pos_default) == 0

        # The explicit portfolio should have the position
        pos_other = await proxy.get_positions(portfolio_id="other")
        assert len(pos_other) == 1
        assert pos_other[0]["symbol"] == "MSFT"
