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


# ──────────────────────────────────────────────
# Extended Database Isolation Tests
# ──────────────────────────────────────────────

class TestDatabaseIsolationEdgeCases:
    """Edge cases for cross-portfolio data isolation."""

    @pytest.fixture
    async def db(self, temp_db):
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()
        yield database
        await database.close()

    @pytest.mark.asyncio
    async def test_has_recent_sell_trade_scoped(self, db):
        """has_recent_sell_trade is scoped to portfolio."""
        await db.record_trade("AAPL", "SELL", 100, 195.0, portfolio_id="portfolio_a")

        assert await db.has_recent_sell_trade("AAPL", 600, portfolio_id="portfolio_a") is True
        assert await db.has_recent_sell_trade("AAPL", 600, portfolio_id="portfolio_b") is False

    @pytest.mark.asyncio
    async def test_get_all_positions_returns_all_portfolios(self, db):
        """get_all_positions returns positions from ALL portfolios with portfolio_id."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
        await db.update_position("MSFT", 50, 300.0, portfolio_id="portfolio_b")
        await db.update_position("AAPL", 25, 185.0, portfolio_id="portfolio_b")

        all_pos = await db.get_all_positions()
        assert len(all_pos) == 3

        # Verify portfolio_id field is present
        portfolio_ids = {p["portfolio_id"] for p in all_pos}
        assert portfolio_ids == {"portfolio_a", "portfolio_b"}

        # Verify both AAPL positions have correct quantities
        aapl_positions = [p for p in all_pos if p["symbol"] == "AAPL"]
        assert len(aapl_positions) == 2
        qtys = {p["quantity"] for p in aapl_positions}
        assert qtys == {100, 25}

    @pytest.mark.asyncio
    async def test_get_all_positions_excludes_closed(self, db):
        """get_all_positions excludes positions with quantity=0."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
        await db.update_position("MSFT", 0, 0, portfolio_id="portfolio_b")

        all_pos = await db.get_all_positions()
        assert len(all_pos) == 1
        assert all_pos[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_equity_snapshots_same_date_different_portfolios(self, db):
        """Two portfolios can save equity for the same date independently."""
        date = "2026-02-06"
        await db.save_equity_snapshot(55000, 50000, 5000, snapshot_date=date, portfolio_id="portfolio_a")
        await db.save_equity_snapshot(82000, 80000, 2000, snapshot_date=date, portfolio_id="portfolio_b")

        hist_a = await db.get_equity_history(portfolio_id="portfolio_a")
        hist_b = await db.get_equity_history(portfolio_id="portfolio_b")

        assert len(hist_a) == 1
        assert hist_a[0]["equity"] == 55000
        assert hist_a[0]["date"] == date

        assert len(hist_b) == 1
        assert hist_b[0]["equity"] == 82000
        assert hist_b[0]["date"] == date

    @pytest.mark.asyncio
    async def test_equity_snapshot_upsert_same_date_same_portfolio(self, db):
        """Saving equity twice for same date+portfolio replaces the old value."""
        date = "2026-02-06"
        await db.save_equity_snapshot(50000, 48000, 2000, snapshot_date=date, portfolio_id="portfolio_a")
        await db.save_equity_snapshot(52000, 48000, 4000, snapshot_date=date, portfolio_id="portfolio_a")

        hist = await db.get_equity_history(portfolio_id="portfolio_a")
        assert len(hist) == 1
        assert hist[0]["equity"] == 52000  # Updated value

    @pytest.mark.asyncio
    async def test_trades_symbol_filter_scoped_to_portfolio(self, db):
        """get_recent_trades with symbol filter respects portfolio scoping."""
        await db.record_trade("AAPL", "BUY", 100, 180.0, portfolio_id="portfolio_a")
        await db.record_trade("AAPL", "BUY", 50, 185.0, portfolio_id="portfolio_b")

        trades_a = await db.get_recent_trades(symbol="AAPL", portfolio_id="portfolio_a")
        trades_b = await db.get_recent_trades(symbol="AAPL", portfolio_id="portfolio_b")

        assert len(trades_a) == 1
        assert trades_a[0]["quantity"] == 100

        assert len(trades_b) == 1
        assert trades_b[0]["quantity"] == 50

    @pytest.mark.asyncio
    async def test_fifo_pnl_scoped_to_portfolio(self, db):
        """SELL trade PnL calculation only considers BUY trades from same portfolio."""
        # Portfolio A bought AAPL at 180
        await db.record_trade("AAPL", "BUY", 100, 180.0, portfolio_id="portfolio_a")
        # Portfolio B bought AAPL at 200 (higher cost basis)
        await db.record_trade("AAPL", "BUY", 100, 200.0, portfolio_id="portfolio_b")

        # Sell from portfolio A at 190 - should use A's cost basis (180), PnL = +$1000
        await db.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
        await db.record_trade("AAPL", "SELL", 100, 190.0, portfolio_id="portfolio_a")

        trades_a = await db.get_recent_trades(symbol="AAPL", portfolio_id="portfolio_a")
        sell_trade = [t for t in trades_a if t["side"] == "SELL"][0]
        assert sell_trade["pnl"] == pytest.approx(1000.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_account_update_preserves_other_portfolio(self, db):
        """Updating one portfolio's account doesn't touch the other."""
        await db.update_account(50000, 55000, daily_pnl=500, portfolio_id="portfolio_a")
        await db.update_account(80000, 82000, daily_pnl=1000, portfolio_id="portfolio_b")

        # Update portfolio_a again
        await db.update_account(48000, 53000, daily_pnl=-200, portfolio_id="portfolio_a")

        # portfolio_b should be unchanged
        acct_b = await db.get_account_info(portfolio_id="portfolio_b")
        assert acct_b["cash"] == 80000
        assert acct_b["daily_pnl"] == 1000

        acct_a = await db.get_account_info(portfolio_id="portfolio_a")
        assert acct_a["cash"] == 48000
        assert acct_a["daily_pnl"] == -200

    @pytest.mark.asyncio
    async def test_position_replace_same_portfolio_symbol(self, db):
        """Updating a position for same portfolio+symbol replaces it (not duplicates)."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
        await db.update_position("AAPL", 150, 185.0, portfolio_id="portfolio_a")

        pos = await db.get_positions(portfolio_id="portfolio_a")
        assert len(pos) == 1
        assert pos[0]["quantity"] == 150
        assert pos[0]["avg_cost"] == 185.0

    @pytest.mark.asyncio
    async def test_get_position_single_symbol_scoped(self, db):
        """get_position for a single symbol is scoped to portfolio."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
        await db.update_position("AAPL", 50, 190.0, portfolio_id="portfolio_b")

        pos_a = await db.get_position("AAPL", portfolio_id="portfolio_a")
        pos_b = await db.get_position("AAPL", portfolio_id="portfolio_b")

        assert pos_a is not None
        assert pos_a["quantity"] == 100

        assert pos_b is not None
        assert pos_b["quantity"] == 50

    @pytest.mark.asyncio
    async def test_get_position_returns_none_for_other_portfolio(self, db):
        """get_position returns None when symbol exists only in another portfolio."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")

        pos = await db.get_position("AAPL", portfolio_id="portfolio_b")
        assert pos is None


# ──────────────────────────────────────────────
# Non-Existent Portfolio Query Tests
# ──────────────────────────────────────────────

class TestNonExistentPortfolioQueries:
    """Verify graceful behavior when querying a portfolio that has no data."""

    @pytest.fixture
    async def db(self, temp_db):
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()
        yield database
        await database.close()

    @pytest.mark.asyncio
    async def test_get_positions_empty_for_unknown_portfolio(self, db):
        """Querying positions for a non-existent portfolio returns empty list."""
        pos = await db.get_positions(portfolio_id="nonexistent")
        assert pos == []

    @pytest.mark.asyncio
    async def test_get_recent_trades_empty_for_unknown_portfolio(self, db):
        """Querying trades for a non-existent portfolio returns empty list."""
        trades = await db.get_recent_trades(portfolio_id="nonexistent")
        assert trades == []

    @pytest.mark.asyncio
    async def test_get_account_info_empty_for_unknown_portfolio(self, db):
        """Account info for non-existent portfolio returns empty dict."""
        acct = await db.get_account_info(portfolio_id="nonexistent")
        assert acct == {}

    @pytest.mark.asyncio
    async def test_get_equity_history_empty_for_unknown_portfolio(self, db):
        """Equity history for non-existent portfolio returns empty list."""
        hist = await db.get_equity_history(portfolio_id="nonexistent")
        assert hist == []

    @pytest.mark.asyncio
    async def test_has_recent_buy_trade_false_for_unknown_portfolio(self, db):
        """Recent buy check returns False for non-existent portfolio."""
        result = await db.has_recent_buy_trade("AAPL", 600, portfolio_id="nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_has_recent_sell_trade_false_for_unknown_portfolio(self, db):
        """Recent sell check returns False for non-existent portfolio."""
        result = await db.has_recent_sell_trade("AAPL", 600, portfolio_id="nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_position_none_for_unknown_portfolio(self, db):
        """get_position returns None for non-existent portfolio."""
        pos = await db.get_position("AAPL", portfolio_id="nonexistent")
        assert pos is None


# ──────────────────────────────────────────────
# Extended PortfolioScopedDB Tests
# ──────────────────────────────────────────────

class TestPortfolioScopedDBExtended:
    """Test all scoped methods via the proxy."""

    @pytest.fixture
    async def raw_db(self, temp_db):
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()
        yield database
        await database.close()

    @pytest.mark.asyncio
    async def test_proxy_has_recent_buy_trade(self, raw_db):
        """has_recent_buy_trade correctly scoped via proxy."""
        proxy_a = PortfolioScopedDB(raw_db, portfolio_id="alpha")
        proxy_b = PortfolioScopedDB(raw_db, portfolio_id="beta")

        await proxy_a.record_trade("AAPL", "BUY", 100, 180.0)

        assert await proxy_a.has_recent_buy_trade("AAPL", 600) is True
        assert await proxy_b.has_recent_buy_trade("AAPL", 600) is False

    @pytest.mark.asyncio
    async def test_proxy_has_recent_sell_trade(self, raw_db):
        """has_recent_sell_trade correctly scoped via proxy."""
        proxy_a = PortfolioScopedDB(raw_db, portfolio_id="alpha")
        proxy_b = PortfolioScopedDB(raw_db, portfolio_id="beta")

        await proxy_a.record_trade("AAPL", "SELL", 100, 195.0)

        assert await proxy_a.has_recent_sell_trade("AAPL", 600) is True
        assert await proxy_b.has_recent_sell_trade("AAPL", 600) is False

    @pytest.mark.asyncio
    async def test_proxy_record_signal(self, raw_db):
        """record_signal correctly scoped via proxy."""
        proxy_a = PortfolioScopedDB(raw_db, portfolio_id="alpha")
        proxy_b = PortfolioScopedDB(raw_db, portfolio_id="beta")

        await proxy_a.record_signal("AAPL", "momentum", "BUY", 0.8)
        await proxy_b.record_signal("MSFT", "ml_enhanced", "SELL", 0.7)

        # Verify isolation via raw DB query
        import aiosqlite
        async with aiosqlite.connect(raw_db.db_path) as conn:
            cursor = await conn.execute(
                "SELECT portfolio_id, symbol FROM signals ORDER BY symbol"
            )
            rows = await cursor.fetchall()
            assert len(rows) == 2
            assert rows[0] == ("alpha", "AAPL")
            assert rows[1] == ("beta", "MSFT")

    @pytest.mark.asyncio
    async def test_proxy_save_equity_snapshot(self, raw_db):
        """save_equity_snapshot correctly scoped via proxy."""
        proxy_a = PortfolioScopedDB(raw_db, portfolio_id="alpha")
        proxy_b = PortfolioScopedDB(raw_db, portfolio_id="beta")

        await proxy_a.save_equity_snapshot(55000, 50000, 5000)
        await proxy_b.save_equity_snapshot(82000, 80000, 2000)

        hist_a = await proxy_a.get_equity_history()
        hist_b = await proxy_b.get_equity_history()

        assert len(hist_a) == 1
        assert hist_a[0]["equity"] == 55000
        assert len(hist_b) == 1
        assert hist_b[0]["equity"] == 82000

    @pytest.mark.asyncio
    async def test_proxy_upsert_portfolio_passes_through(self, raw_db):
        """upsert_portfolio passes through proxy (not portfolio-scoped kwarg)."""
        proxy = PortfolioScopedDB(raw_db, portfolio_id="alpha")

        # upsert_portfolio takes portfolio_data dict (id is inside dict, not a kwarg)
        await proxy.upsert_portfolio({
            "id": "alpha",
            "name": "Alpha Portfolio",
            "starting_cash": 75000,
            "symbols": "AAPL,NVDA",
        })

        portfolios = await raw_db.get_portfolios()
        alpha = [p for p in portfolios if p["id"] == "alpha"]
        assert len(alpha) == 1
        assert alpha[0]["name"] == "Alpha Portfolio"

    @pytest.mark.asyncio
    async def test_proxy_portfolio_id_attribute(self, raw_db):
        """Proxy exposes its portfolio_id."""
        proxy = PortfolioScopedDB(raw_db, portfolio_id="my_portfolio")
        assert proxy.portfolio_id == "my_portfolio"

    @pytest.mark.asyncio
    async def test_proxy_underlying_db_accessible(self, raw_db):
        """Proxy gives access to the underlying DB via _db."""
        proxy = PortfolioScopedDB(raw_db, portfolio_id="test")
        assert proxy._db is raw_db

    @pytest.mark.asyncio
    async def test_proxy_get_position_single_symbol(self, raw_db):
        """get_position (single symbol) works through proxy."""
        proxy = PortfolioScopedDB(raw_db, portfolio_id="alpha")

        await proxy.update_position("AAPL", 100, 180.0)

        pos = await proxy.get_position("AAPL")
        assert pos is not None
        assert pos["quantity"] == 100

        # Different portfolio should not see it
        proxy_b = PortfolioScopedDB(raw_db, portfolio_id="beta")
        pos_b = await proxy_b.get_position("AAPL")
        assert pos_b is None

    @pytest.mark.asyncio
    async def test_multiple_proxies_concurrent_operations(self, raw_db):
        """Multiple proxies operating concurrently don't interfere."""
        proxy_a = PortfolioScopedDB(raw_db, portfolio_id="alpha")
        proxy_b = PortfolioScopedDB(raw_db, portfolio_id="beta")
        proxy_c = PortfolioScopedDB(raw_db, portfolio_id="gamma")

        # Write positions concurrently
        await asyncio.gather(
            proxy_a.update_position("AAPL", 100, 180.0),
            proxy_b.update_position("AAPL", 50, 185.0),
            proxy_c.update_position("AAPL", 25, 190.0),
        )

        # Verify isolation
        pos_a = await proxy_a.get_positions()
        pos_b = await proxy_b.get_positions()
        pos_c = await proxy_c.get_positions()

        assert len(pos_a) == 1 and pos_a[0]["quantity"] == 100
        assert len(pos_b) == 1 and pos_b[0]["quantity"] == 50
        assert len(pos_c) == 1 and pos_c[0]["quantity"] == 25


# ──────────────────────────────────────────────
# Portfolio Config Validation Edge Cases
# ──────────────────────────────────────────────

class TestPortfolioConfigEdgeCases:
    """Edge cases for portfolio configuration."""

    def test_from_dict_empty_symbols_string(self):
        """Empty symbols string produces empty list."""
        cfg = PortfolioConfig.from_dict({"id": "test", "name": "T", "symbols": ""})
        assert cfg.symbols == []

    def test_from_dict_whitespace_symbols(self):
        """Whitespace in symbols is stripped."""
        cfg = PortfolioConfig.from_dict({"id": "test", "name": "T", "symbols": "  AAPL  ,  MSFT  "})
        assert cfg.symbols == ["AAPL", "MSFT"]

    def test_from_dict_missing_name_uses_id(self):
        """Missing name falls back to id."""
        cfg = PortfolioConfig.from_dict({"id": "my_portfolio"})
        assert cfg.name == "my_portfolio"

    def test_from_dict_all_risk_overrides(self):
        """All risk override fields are preserved in from_dict."""
        data = {
            "id": "test",
            "name": "Test",
            "max_position_pct": 0.05,
            "max_daily_loss_pct": 0.01,
            "max_open_positions": 5,
            "stop_loss_pct": 3.0,
            "trailing_stop_pct": 5.0,
            "use_trailing_stop": True,
            "min_confidence": 0.6,
        }
        cfg = PortfolioConfig.from_dict(data)
        assert cfg.max_position_pct == 0.05
        assert cfg.max_daily_loss_pct == 0.01
        assert cfg.max_open_positions == 5
        assert cfg.stop_loss_pct == 3.0
        assert cfg.trailing_stop_pct == 5.0
        assert cfg.use_trailing_stop is True
        assert cfg.min_confidence == 0.6

    def test_from_dict_strategy_string_split(self):
        """enabled_strategies as comma-separated string is split correctly."""
        cfg = PortfolioConfig.from_dict({
            "id": "test",
            "name": "T",
            "enabled_strategies": "momentum, ml_enhanced , pairs",
        })
        assert cfg.enabled_strategies == ["momentum", "ml_enhanced", "pairs"]

    def test_from_dict_extra_fields_ignored(self):
        """Unknown fields in dict don't cause errors."""
        cfg = PortfolioConfig.from_dict({
            "id": "test",
            "name": "T",
            "unknown_field": "value",
            "future_feature": True,
        })
        assert cfg.id == "test"
        assert not hasattr(cfg, "unknown_field")

    def test_to_dict_inactive_portfolio(self):
        """Inactive portfolio serializes active=False correctly."""
        cfg = PortfolioConfig(id="disabled", name="Disabled", active=False)
        d = cfg.to_dict()
        assert d["active"] is False

    def test_get_risk_param_nonexistent_attribute(self):
        """get_risk_param with non-existent attribute returns global default."""
        cfg = PortfolioConfig(id="test", name="Test")
        result = cfg.get_risk_param("nonexistent_param", 42)
        assert result == 42

    def test_to_dict_none_strategies(self):
        """to_dict handles None enabled_strategies."""
        cfg = PortfolioConfig(id="test", name="T")
        d = cfg.to_dict()
        assert d["enabled_strategies"] is None

    def test_inactive_portfolio_in_multi_config(self):
        """Inactive portfolios are loaded but identifiable."""
        portfolios = [
            {"id": "active", "name": "Active", "active": True, "symbols": "AAPL"},
            {"id": "disabled", "name": "Disabled", "active": False, "symbols": "MSFT"},
        ]
        env = {"PORTFOLIOS": json.dumps(portfolios)}
        with patch.dict(os.environ, env, clear=True):
            configs = load_portfolio_configs()
            assert len(configs) == 2
            active_configs = [c for c in configs if c.active]
            assert len(active_configs) == 1
            assert active_configs[0].id == "active"

    def test_portfolio_not_json_object(self):
        """Non-object items in PORTFOLIOS array raise ValueError."""
        env = {"PORTFOLIOS": json.dumps(["just_a_string"])}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="JSON object"):
                load_portfolio_configs()


# ──────────────────────────────────────────────
# Upsert Portfolio Tests
# ──────────────────────────────────────────────

class TestUpsertPortfolio:
    """Test portfolio definition insert and update behavior."""

    @pytest.fixture
    async def db(self, temp_db):
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()
        yield database
        await database.close()

    @pytest.mark.asyncio
    async def test_upsert_creates_new_portfolio(self, db):
        """upsert_portfolio creates a new portfolio definition."""
        await db.upsert_portfolio({
            "id": "new_portfolio",
            "name": "New Portfolio",
            "starting_cash": 75000,
            "symbols": "AAPL,NVDA",
            "active": True,
        })

        portfolios = await db.get_portfolios()
        new = [p for p in portfolios if p["id"] == "new_portfolio"]
        assert len(new) == 1
        assert new[0]["name"] == "New Portfolio"
        assert new[0]["starting_cash"] == 75000

    @pytest.mark.asyncio
    async def test_upsert_updates_existing_portfolio(self, db):
        """upsert_portfolio updates an existing portfolio definition."""
        await db.upsert_portfolio({
            "id": "evolving",
            "name": "Version 1",
            "starting_cash": 50000,
            "symbols": "AAPL",
        })

        # Update it
        await db.upsert_portfolio({
            "id": "evolving",
            "name": "Version 2",
            "starting_cash": 75000,
            "symbols": "AAPL,NVDA,TSLA",
        })

        portfolios = await db.get_portfolios()
        evolving = [p for p in portfolios if p["id"] == "evolving"]
        assert len(evolving) == 1
        assert evolving[0]["name"] == "Version 2"
        assert evolving[0]["starting_cash"] == 75000

    @pytest.mark.asyncio
    async def test_upsert_preserves_risk_overrides(self, db):
        """upsert_portfolio preserves risk override fields."""
        await db.upsert_portfolio({
            "id": "risky",
            "name": "Risky",
            "max_position_pct": 0.08,
            "max_daily_loss_pct": 0.02,
            "max_open_positions": 10,
            "stop_loss_pct": 5.0,
            "trailing_stop_pct": 7.0,
            "use_trailing_stop": True,
            "min_confidence": 0.4,
        })

        portfolios = await db.get_portfolios()
        risky = [p for p in portfolios if p["id"] == "risky"][0]
        assert risky["max_position_pct"] == 0.08
        assert risky["max_daily_loss_pct"] == 0.02
        assert risky["max_open_positions"] == 10
        assert risky["use_trailing_stop"] is True

    @pytest.mark.asyncio
    async def test_upsert_default_portfolio_exists(self, db):
        """The 'default' portfolio is created during DB init."""
        portfolios = await db.get_portfolios()
        default = [p for p in portfolios if p["id"] == "default"]
        assert len(default) == 1
        assert default[0]["name"] == "Default Portfolio"


# ──────────────────────────────────────────────
# Migration Edge Cases
# ──────────────────────────────────────────────

class TestMigrationEdgeCases:
    """Additional migration edge cases."""

    @pytest.mark.asyncio
    async def test_migration_version_recorded_correctly(self, temp_db):
        """Migration version is recorded in schema_migrations table."""
        import aiosqlite

        await create_legacy_schema(temp_db)
        migration = MultiuserMigration(temp_db)
        await migration.migrate()

        async with aiosqlite.connect(temp_db) as conn:
            cursor = await conn.execute(
                "SELECT version, description FROM schema_migrations"
            )
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == 1  # Version 1
            assert "multi-portfolio" in rows[0][1].lower()

    @pytest.mark.asyncio
    async def test_migration_preserves_null_market_price(self, temp_db):
        """Migration preserves NULL market_price values."""
        import aiosqlite

        await create_legacy_schema(temp_db)

        # Add a position with NULL market_price
        async with aiosqlite.connect(temp_db) as conn:
            await conn.execute(
                "INSERT INTO positions (symbol, quantity, avg_cost) VALUES (?, ?, ?)",
                ("TSLA", 30, 250.00),
            )
            await conn.commit()

        migration = MultiuserMigration(temp_db)
        await migration.migrate()

        async with aiosqlite.connect(temp_db) as conn:
            cursor = await conn.execute(
                "SELECT symbol, market_price FROM positions WHERE symbol = 'TSLA'"
            )
            row = await cursor.fetchone()
            assert row[0] == "TSLA"
            assert row[1] is None  # NULL preserved

    @pytest.mark.asyncio
    async def test_migration_preserves_trade_pnl(self, temp_db):
        """Migration preserves PnL values on trades."""
        import aiosqlite

        await create_legacy_schema(temp_db)

        async with aiosqlite.connect(temp_db) as conn:
            await conn.execute(
                "INSERT INTO trades (symbol, side, quantity, price, notional, pnl) VALUES (?, ?, ?, ?, ?, ?)",
                ("AAPL", "SELL", 50, 195.0, 9750.0, 750.0),
            )
            await conn.commit()

        migration = MultiuserMigration(temp_db)
        await migration.migrate()

        async with aiosqlite.connect(temp_db) as conn:
            cursor = await conn.execute(
                "SELECT symbol, side, pnl FROM trades WHERE side = 'SELL'"
            )
            row = await cursor.fetchone()
            assert row[0] == "AAPL"
            assert row[2] == 750.0  # PnL preserved

    @pytest.mark.asyncio
    async def test_migration_backup_is_valid_sqlite(self, temp_db):
        """Backup file created by migration is a valid SQLite database."""
        import aiosqlite

        await create_legacy_schema(temp_db)
        migration = MultiuserMigration(temp_db)
        await migration.migrate()

        backups = list(temp_db.parent.glob(f"{temp_db.stem}.backup_premultiuser_*"))
        assert len(backups) >= 1

        # Verify backup is valid SQLite
        async with aiosqlite.connect(backups[0]) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM positions")
            count = (await cursor.fetchone())[0]
            assert count == 2  # Original 2 positions

    @pytest.mark.asyncio
    async def test_migration_with_many_trades(self, temp_db):
        """Migration handles a moderate number of trades correctly."""
        import aiosqlite

        await create_legacy_schema(temp_db)

        # Insert 500 additional trades
        async with aiosqlite.connect(temp_db) as conn:
            for i in range(500):
                symbol = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOG"][i % 5]
                side = "BUY" if i % 3 != 0 else "SELL"
                price = 100.0 + (i % 50)
                await conn.execute(
                    "INSERT INTO trades (symbol, side, quantity, price, notional) VALUES (?, ?, ?, ?, ?)",
                    (symbol, side, 10, price, 10 * price),
                )
            await conn.commit()

        migration = MultiuserMigration(temp_db)
        await migration.migrate()

        async with aiosqlite.connect(temp_db) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM trades")
            count = (await cursor.fetchone())[0]
            assert count == 501  # 1 original + 500 added

            # All should have portfolio_id='default'
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM trades WHERE portfolio_id = 'default'"
            )
            default_count = (await cursor.fetchone())[0]
            assert default_count == 501


# ──────────────────────────────────────────────
# SyncDatabaseReader Portfolio Scoping Tests
# ──────────────────────────────────────────────

class TestSyncDatabaseReaderPortfolioScoping:
    """Test that SyncDatabaseReader correctly scopes queries by portfolio_id."""

    @pytest.fixture
    async def populated_db(self, temp_db):
        """Create a database with data in multiple portfolios."""
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()

        # Portfolio A data
        await database.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
        await database.update_position("NVDA", 50, 450.0, portfolio_id="portfolio_a")
        await database.record_trade("AAPL", "BUY", 100, 180.0, portfolio_id="portfolio_a")
        await database.update_account(50000, 55000, portfolio_id="portfolio_a")
        await database.save_equity_snapshot(55000, 50000, 5000, snapshot_date="2026-02-06", portfolio_id="portfolio_a")
        await database.record_signal("AAPL", "momentum", "BUY", 0.8, portfolio_id="portfolio_a")

        # Portfolio B data
        await database.update_position("MSFT", 200, 300.0, portfolio_id="portfolio_b")
        await database.record_trade("MSFT", "BUY", 200, 300.0, portfolio_id="portfolio_b")
        await database.update_account(80000, 82000, portfolio_id="portfolio_b")
        await database.save_equity_snapshot(82000, 80000, 2000, snapshot_date="2026-02-06", portfolio_id="portfolio_b")
        await database.record_signal("MSFT", "ml_enhanced", "SELL", 0.7, portfolio_id="portfolio_b")

        # Portfolio definitions
        await database.upsert_portfolio({"id": "portfolio_a", "name": "Portfolio A", "symbols": "AAPL,NVDA"})
        await database.upsert_portfolio({"id": "portfolio_b", "name": "Portfolio B", "symbols": "MSFT"})

        await database.close()
        yield temp_db

    def test_get_positions_scoped(self, populated_db):
        """SyncDatabaseReader.get_positions scoped to portfolio."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        pos_a = reader.get_positions(portfolio_id="portfolio_a")
        pos_b = reader.get_positions(portfolio_id="portfolio_b")

        assert len(pos_a) == 2
        symbols_a = {p["symbol"] for p in pos_a}
        assert symbols_a == {"AAPL", "NVDA"}

        assert len(pos_b) == 1
        assert pos_b[0]["symbol"] == "MSFT"

    def test_get_all_positions_cross_portfolio(self, populated_db):
        """SyncDatabaseReader.get_all_positions returns all portfolios."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        all_pos = reader.get_all_positions()
        assert len(all_pos) == 3

        portfolio_ids = {p["portfolio_id"] for p in all_pos}
        assert portfolio_ids == {"portfolio_a", "portfolio_b"}

    def test_get_recent_trades_scoped(self, populated_db):
        """SyncDatabaseReader.get_recent_trades scoped to portfolio."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        trades_a = reader.get_recent_trades(portfolio_id="portfolio_a")
        trades_b = reader.get_recent_trades(portfolio_id="portfolio_b")

        assert len(trades_a) == 1
        assert trades_a[0]["symbol"] == "AAPL"

        assert len(trades_b) == 1
        assert trades_b[0]["symbol"] == "MSFT"

    def test_get_account_info_scoped(self, populated_db):
        """SyncDatabaseReader.get_account_info scoped to portfolio."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        acct_a = reader.get_account_info(portfolio_id="portfolio_a")
        acct_b = reader.get_account_info(portfolio_id="portfolio_b")

        assert acct_a["cash"] == 50000
        assert acct_b["cash"] == 80000

    def test_get_equity_history_scoped(self, populated_db):
        """SyncDatabaseReader.get_equity_history scoped to portfolio."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        hist_a = reader.get_equity_history(portfolio_id="portfolio_a")
        hist_b = reader.get_equity_history(portfolio_id="portfolio_b")

        assert len(hist_a) == 1
        assert hist_a[0]["equity"] == 55000

        assert len(hist_b) == 1
        assert hist_b[0]["equity"] == 82000

    def test_get_signals_scoped(self, populated_db):
        """SyncDatabaseReader.get_signals scoped to portfolio."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        signals_a = reader.get_signals(hours=24, portfolio_id="portfolio_a")
        signals_b = reader.get_signals(hours=24, portfolio_id="portfolio_b")

        assert len(signals_a) == 1
        assert signals_a[0]["symbol"] == "AAPL"

        assert len(signals_b) == 1
        assert signals_b[0]["symbol"] == "MSFT"

    def test_get_portfolios_returns_all(self, populated_db):
        """SyncDatabaseReader.get_portfolios returns all portfolios."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        portfolios = reader.get_portfolios()
        ids = {p["id"] for p in portfolios}
        assert "default" in ids
        assert "portfolio_a" in ids
        assert "portfolio_b" in ids

    def test_default_portfolio_backward_compat(self, populated_db):
        """SyncDatabaseReader defaults to 'default' portfolio when no portfolio_id."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        # No data in 'default' portfolio, should return empty
        pos = reader.get_positions()  # No portfolio_id = 'default'
        assert pos == []

    def test_nonexistent_portfolio_returns_empty(self, populated_db):
        """SyncDatabaseReader returns empty for non-existent portfolio."""
        from sync_db_reader import SyncDatabaseReader
        reader = SyncDatabaseReader(db_path=str(populated_db))

        pos = reader.get_positions(portfolio_id="nonexistent")
        trades = reader.get_recent_trades(portfolio_id="nonexistent")
        hist = reader.get_equity_history(portfolio_id="nonexistent")

        assert pos == []
        assert trades == []
        assert hist == []


# ──────────────────────────────────────────────
# Portfolio ID Edge Cases
# ──────────────────────────────────────────────

class TestPortfolioIdEdgeCases:
    """Test edge cases for portfolio_id values."""

    @pytest.fixture
    async def db(self, temp_db):
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()
        yield database
        await database.close()

    @pytest.mark.asyncio
    async def test_portfolio_id_case_sensitive(self, db):
        """Portfolio IDs are case-sensitive."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="Alpha")
        await db.update_position("AAPL", 50, 190.0, portfolio_id="alpha")

        pos_upper = await db.get_positions(portfolio_id="Alpha")
        pos_lower = await db.get_positions(portfolio_id="alpha")

        assert len(pos_upper) == 1
        assert pos_upper[0]["quantity"] == 100

        assert len(pos_lower) == 1
        assert pos_lower[0]["quantity"] == 50

    @pytest.mark.asyncio
    async def test_portfolio_id_with_special_chars(self, db):
        """Portfolio IDs with hyphens and underscores work."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="my-portfolio_v2")

        pos = await db.get_positions(portfolio_id="my-portfolio_v2")
        assert len(pos) == 1
        assert pos[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_portfolio_id_with_numbers(self, db):
        """Portfolio IDs with numbers work."""
        await db.update_position("AAPL", 100, 180.0, portfolio_id="user123_portfolio456")

        pos = await db.get_positions(portfolio_id="user123_portfolio456")
        assert len(pos) == 1

    @pytest.mark.asyncio
    async def test_many_portfolios_isolation(self, db):
        """20 portfolios with the same symbol maintain complete isolation."""
        num_portfolios = 20

        # Create positions in 20 portfolios
        for i in range(num_portfolios):
            pid = f"portfolio_{i}"
            await db.update_position("AAPL", i + 1, 180.0 + i, portfolio_id=pid)

        # Verify each portfolio has the correct quantity
        for i in range(num_portfolios):
            pid = f"portfolio_{i}"
            pos = await db.get_positions(portfolio_id=pid)
            assert len(pos) == 1
            assert pos[0]["quantity"] == i + 1
            assert pos[0]["avg_cost"] == 180.0 + i

        # Verify get_all_positions returns all 20
        all_pos = await db.get_all_positions()
        assert len(all_pos) == num_portfolios


# ──────────────────────────────────────────────
# Concurrent Cross-Portfolio Operations
# ──────────────────────────────────────────────

class TestConcurrentCrossPortfolio:
    """Test concurrent operations across multiple portfolios."""

    @pytest.fixture
    async def db(self, temp_db):
        database = AsyncTradingDatabase(db_path=temp_db)
        await database.initialize()
        yield database
        await database.close()

    @pytest.mark.asyncio
    async def test_concurrent_position_writes(self, db):
        """Concurrent position writes to different portfolios succeed."""
        tasks = [
            db.update_position("AAPL", 100, 180.0, portfolio_id=f"portfolio_{i}")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        all_pos = await db.get_all_positions()
        assert len(all_pos) == 10

    @pytest.mark.asyncio
    async def test_concurrent_trade_recording(self, db):
        """Concurrent trade recording to different portfolios doesn't lose data."""
        tasks = [
            db.record_trade("AAPL", "BUY", 10 + i, 180.0, portfolio_id=f"portfolio_{i}")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        # Verify each portfolio has exactly 1 trade
        for i in range(10):
            trades = await db.get_recent_trades(portfolio_id=f"portfolio_{i}")
            assert len(trades) == 1
            assert trades[0]["quantity"] == 10 + i

    @pytest.mark.asyncio
    async def test_concurrent_account_updates(self, db):
        """Concurrent account updates to different portfolios are isolated."""
        tasks = [
            db.update_account(50000 + i * 1000, 55000 + i * 1000, portfolio_id=f"portfolio_{i}")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        for i in range(10):
            acct = await db.get_account_info(portfolio_id=f"portfolio_{i}")
            assert acct["cash"] == 50000 + i * 1000
            assert acct["equity"] == 55000 + i * 1000

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, db):
        """Concurrent mixed operations (positions, trades, account) across portfolios."""
        async def portfolio_operations(pid, qty, price):
            await db.update_position("AAPL", qty, price, portfolio_id=pid)
            await db.record_trade("AAPL", "BUY", qty, price, portfolio_id=pid)
            await db.update_account(100000 - qty * price, 100000, portfolio_id=pid)

        tasks = [
            portfolio_operations(f"portfolio_{i}", 10 * (i + 1), 180.0)
            for i in range(5)
        ]
        await asyncio.gather(*tasks)

        for i in range(5):
            pid = f"portfolio_{i}"
            pos = await db.get_positions(portfolio_id=pid)
            trades = await db.get_recent_trades(portfolio_id=pid)
            acct = await db.get_account_info(portfolio_id=pid)

            assert len(pos) == 1
            assert pos[0]["quantity"] == 10 * (i + 1)
            assert len(trades) == 1
            assert acct is not None
