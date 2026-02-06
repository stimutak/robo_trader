"""
Async database integration for trading data persistence.

This implements Phase 1 F3: Async Database Operations
- Converts all SQLite operations to async using aiosqlite
- Implements connection pooling for database access
- Ensures no event loop blocking
- Multi-portfolio support: all user-scoped tables partitioned by portfolio_id
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiosqlite

from robo_trader.database_validator import DatabaseValidator, ValidationError
from robo_trader.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("trading_data.db")

# Default portfolio ID for backward compatibility
DEFAULT_PORTFOLIO_ID = "default"


class AsyncTradingDatabase:
    """Async database manager for trading data persistence."""

    def __init__(self, db_path: Path = DB_PATH, pool_size: int = 5):
        """Initialize async database with connection pooling."""
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: List[aiosqlite.Connection] = []
        self._available: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._initialized = False
        self._lock = asyncio.Lock()
        self._closed = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize database and connection pool."""
        async with self._lock:
            if self._initialized:
                return

            # Create database and tables
            await self._init_database()

            # Create connection pool
            for _ in range(self.pool_size):
                conn = await aiosqlite.connect(self.db_path)
                await conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
                await conn.execute("PRAGMA busy_timeout=5000")  # Wait up to 5s on contention
                self._pool.append(conn)
                await self._available.put(conn)

            self._initialized = True
            logger.info(
                f"Async database initialized at {self.db_path} with {self.pool_size} connections"
            )

    async def close(self):
        """Close all connections in the pool."""
        async with self._lock:
            if not self._initialized:
                return

            # Close all connections in the pool
            for conn in self._pool:
                try:
                    # Ensure no transactions are left open
                    if getattr(conn, "in_transaction", False):
                        await conn.rollback()
                    await conn.close()
                except Exception as e:
                    logger.debug(f"Error closing connection: {e}")

            # Clear the pool and queue
            self._pool.clear()

            # Clear the available queue
            while not self._available.empty():
                try:
                    self._available.get_nowait()
                except asyncio.QueueEmpty:
                    break

            self._initialized = False
            self._closed = True
            logger.info("Closed all database connections")

    async def health_check(self) -> bool:
        """Check if database connections are healthy."""
        if not self._initialized or self._closed:
            return False

        try:
            async with self.get_connection() as conn:
                await conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    async def ensure_connection(self):
        """Ensure database is connected and healthy."""
        if not await self.health_check():
            logger.info("Database unhealthy, reinitializing...")
            await self.close()
            self._closed = False
            await self.initialize()

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Database is closed")

        if not self._initialized:
            await self.initialize()

        conn = await asyncio.wait_for(self._available.get(), timeout=10.0)
        try:
            # Test connection before use
            await conn.execute("SELECT 1")
            yield conn
        except Exception as e:
            logger.warning(f"Connection error: {e}")
            # Try to create a new connection to replace the bad one
            try:
                await conn.close()
                new_conn = await aiosqlite.connect(self.db_path)
                await new_conn.execute("PRAGMA journal_mode=WAL")
                await new_conn.execute("PRAGMA busy_timeout=5000")
                await self._available.put(new_conn)
            except Exception as replace_error:
                logger.error(f"Failed to replace bad connection: {replace_error}")
                await self._available.put(conn)  # Put back the original connection
            raise
        finally:
            # Ensure no transaction remains open on pooled connections
            try:
                if getattr(conn, "in_transaction", False):
                    await conn.rollback()
            except Exception as e:
                logger.debug(f"Rollback on pooled connection failed: {e}")

            # Only put back if not closed
            if not self._closed:
                await self._available.put(conn)

    async def _init_database(self) -> None:
        """Create tables if they don't exist.

        Tables with portfolio_id (user-scoped):
            positions, trades, account, equity_history, signals
        Tables without portfolio_id (global/shared):
            ticks, features, market_data
        """
        async with aiosqlite.connect(self.db_path) as conn:
            # Set WAL and busy timeout on the initializer connection too, to avoid rollback journal usage
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")

            # Portfolios table (multi-portfolio support)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolios (
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
                )
            """
            )

            # Insert default portfolio if not exists
            await conn.execute(
                """
                INSERT OR IGNORE INTO portfolios (id, name, starting_cash)
                VALUES ('default', 'Default Portfolio', 100000)
            """
            )

            # Schema migrations table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Positions table (portfolio-scoped)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    market_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(portfolio_id, symbol)
                )
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_positions_portfolio
                ON positions (portfolio_id)
            """
            )

            # Tick data table (global - shared across portfolios)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ticks (
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    bid REAL,
                    ask REAL,
                    last REAL,
                    bid_size INTEGER,
                    ask_size INTEGER,
                    last_size INTEGER,
                    volume INTEGER,
                    PRIMARY KEY (timestamp, symbol)
                )
            """
            )

            # Create index for efficient queries
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ticks_symbol
                ON ticks (symbol, timestamp DESC)
            """
            )

            # Features table (global - shared across portfolios)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS features (
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    rsi REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    atr REAL,
                    vwap REAL,
                    obv REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    sma_200 REAL,
                    volume_ratio REAL,
                    spread_bps REAL,
                    trend_strength REAL,
                    mean_reversion_signal REAL,
                    breakout_signal REAL,
                    PRIMARY KEY (timestamp, symbol)
                )
            """
            )

            # Create index for efficient queries
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_features_symbol
                ON features (symbol, timestamp DESC)
            """
            )

            # Trades table (portfolio-scoped)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
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
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_portfolio
                ON trades (portfolio_id)
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_portfolio_symbol
                ON trades (portfolio_id, symbol, timestamp DESC)
            """
            )

            # Migrations for existing tables
            migrations = [
                "ALTER TABLE trades ADD COLUMN pnl REAL DEFAULT NULL",
                "ALTER TABLE trades ADD COLUMN notional REAL DEFAULT 0",
                "ALTER TABLE trades ADD COLUMN portfolio_id TEXT DEFAULT 'default'",
                "ALTER TABLE positions ADD COLUMN portfolio_id TEXT DEFAULT 'default'",
            ]
            for migration in migrations:
                try:
                    await conn.execute(migration)
                except Exception:
                    pass  # Column already exists

            # Account table (portfolio-scoped, keyed by portfolio_id)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account (
                    portfolio_id TEXT PRIMARY KEY DEFAULT 'default',
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Equity history table for portfolio value over time (portfolio-scoped)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_history (
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
                )
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_equity_history_portfolio
                ON equity_history (portfolio_id, date)
            """
            )

            # Market data table (global - shared across portfolios)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER,
                    timestamp DATETIME NOT NULL,
                    UNIQUE(symbol, timestamp)
                )
            """
            )

            # Strategy signals table (portfolio-scoped)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_signals_portfolio
                ON signals (portfolio_id)
            """
            )

            # Insert default account if not exists
            await conn.execute(
                """
                INSERT OR IGNORE INTO account (portfolio_id, cash, equity)
                VALUES ('default', 100000, 100000)
            """
            )

            await conn.commit()

    async def update_position(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float,
        market_price: Optional[float] = None,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Update or insert a position asynchronously."""
        # Validate inputs
        try:
            symbol = DatabaseValidator.validate_symbol(symbol)
            quantity = DatabaseValidator.validate_quantity(quantity, allow_negative=True)
            # Skip price validation when closing position (quantity=0)
            if quantity != 0:
                avg_cost = DatabaseValidator.validate_price(avg_cost, field_name="avg_cost")
                if market_price is not None:
                    market_price = DatabaseValidator.validate_price(
                        market_price, field_name="market_price"
                    )
        except ValidationError as e:
            logger.error(f"Position update validation failed: {e}")
            raise

        async with self.get_connection() as conn:
            if quantity == 0:
                # Close position - delete from database
                await conn.execute(
                    "DELETE FROM positions WHERE portfolio_id = ? AND symbol = ?",
                    (portfolio_id, symbol),
                )
                logger.info(f"Closed position for {symbol} (portfolio={portfolio_id})")
            else:
                # Update or insert position
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (portfolio_id, symbol, quantity, avg_cost, market_price)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (portfolio_id, symbol, quantity, avg_cost, market_price),
                )

            await conn.commit()
            logger.debug(f"Updated position: {symbol} qty={quantity} avg={avg_cost} (portfolio={portfolio_id})")

    async def _calculate_fifo_pnl(
        self, conn, symbol: str, sell_quantity: int, sell_price: float,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> float:
        """
        Calculate realized P&L for a SELL trade using weighted average cost.

        Note: Despite the name, this uses weighted average cost basis (not strict
        FIFO lot matching) for simplicity. The average cost is calculated across
        all BUY trades for the symbol within this portfolio.

        Args:
            conn: Database connection
            symbol: Stock symbol
            sell_quantity: Number of shares being sold
            sell_price: Price per share for the sell
            portfolio_id: Portfolio to scope the calculation to

        Returns:
            Realized P&L (positive = profit, negative = loss)
        """
        # Get all BUY trades for this symbol in this portfolio, ordered by timestamp (FIFO)
        cursor = await conn.execute(
            """
            SELECT id, quantity, price FROM trades
            WHERE portfolio_id = ? AND symbol = ? AND side = 'BUY'
            ORDER BY timestamp ASC
            """,
            (portfolio_id, symbol),
        )
        buy_trades = await cursor.fetchall()

        if not buy_trades:
            # No BUY trades found - use position's avg_cost if available
            cursor = await conn.execute(
                "SELECT avg_cost FROM positions WHERE portfolio_id = ? AND symbol = ?",
                (portfolio_id, symbol),
            )
            pos = await cursor.fetchone()
            if pos and pos[0]:
                avg_cost = pos[0]
                return (sell_price - avg_cost) * sell_quantity
            # No cost basis - return 0
            logger.warning(f"No cost basis found for {symbol} SELL trade (portfolio={portfolio_id})")
            return 0.0

        # Calculate weighted average cost from BUY trades
        # For simplicity, use weighted average rather than strict FIFO lot matching
        total_shares = sum(t[1] for t in buy_trades)
        total_cost = sum(t[1] * t[2] for t in buy_trades)

        if total_shares > 0:
            avg_cost = total_cost / total_shares
            realized_pnl = (sell_price - avg_cost) * sell_quantity
            logger.debug(
                f"FIFO P&L for {symbol}: sell ${sell_price:.2f} - avg cost ${avg_cost:.2f} "
                f"x {sell_quantity} = ${realized_pnl:.2f}"
            )
            return realized_pnl

        return 0.0

    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        slippage: float = 0.0,
        commission: float = 0.0,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Record a trade asynchronously with P&L calculation for SELL trades."""
        # Validate inputs
        try:
            symbol = DatabaseValidator.validate_symbol(symbol)
            side = DatabaseValidator.validate_order_side(side)
            quantity = DatabaseValidator.validate_quantity(quantity)
            price = DatabaseValidator.validate_price(price)
            slippage = DatabaseValidator._validate_numeric(
                slippage, "slippage", min_val=0, max_val=1000
            )
            commission = DatabaseValidator._validate_numeric(
                commission, "commission", min_val=0, max_val=1000
            )
        except ValidationError as e:
            logger.error(f"Trade record validation failed: {e}")
            raise

        async with self.get_connection() as conn:
            # Calculate P&L for SELL trades (realized profit/loss)
            pnl = None
            if side.upper() in ("SELL", "BUY_TO_COVER"):
                pnl = await self._calculate_fifo_pnl(conn, symbol, quantity, price, portfolio_id)

            # Ensure consistent float type for SQLite storage
            notional = float(quantity) * float(price)
            await conn.execute(
                """
                INSERT INTO trades (portfolio_id, symbol, side, quantity, price, notional, slippage, commission, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (portfolio_id, symbol, side, quantity, price, notional, slippage, commission, pnl),
            )
            await conn.commit()

            if pnl is not None:
                logger.info(
                    f"Recorded trade: {side} {quantity} {symbol} @ {price} "
                    f"(P&L: ${pnl:,.2f}, portfolio={portfolio_id})"
                )
            else:
                logger.info(f"Recorded trade: {side} {quantity} {symbol} @ {price} (portfolio={portfolio_id})")

    async def update_account(
        self,
        cash: float,
        equity: float,
        daily_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Update account values asynchronously."""
        # Validate inputs
        try:
            account_data = {
                "cash": cash,
                "equity": equity,
                "daily_pnl": daily_pnl,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
            }
            validated_data = DatabaseValidator.validate_account_data(account_data)
            cash = validated_data.get("cash", cash)
            equity = validated_data.get("equity", equity)
            daily_pnl = validated_data.get("daily_pnl", daily_pnl)
            realized_pnl = validated_data.get("realized_pnl", realized_pnl)
            unrealized_pnl = validated_data.get("unrealized_pnl", unrealized_pnl)
        except ValidationError as e:
            logger.error(f"Account update validation failed: {e}")
            raise

        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO account
                    (portfolio_id, cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (portfolio_id, cash, equity, daily_pnl, realized_pnl, unrealized_pnl),
            )
            await conn.commit()
            logger.debug(f"Updated account: cash={cash:.2f} equity={equity:.2f} (portfolio={portfolio_id})")

    async def record_signal(
        self,
        symbol: str,
        strategy: str,
        signal_type: str,
        strength: float = 0.0,
        metadata: str = "",
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Record a strategy signal asynchronously."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO signals (portfolio_id, symbol, strategy, signal_type, strength, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (portfolio_id, symbol, strategy, signal_type, strength, metadata),
            )
            await conn.commit()
            logger.debug(f"Recorded signal: {strategy} {signal_type} for {symbol} (portfolio={portfolio_id})")

    async def store_market_data(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """Store market data bar asynchronously."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO market_data
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (symbol, timestamp, open_price, high, low, close, volume),
            )
            await conn.commit()

    async def batch_store_market_data(self, data: List[Dict]) -> None:
        """Store multiple market data bars in a batch for efficiency."""
        async with self.get_connection() as conn:
            await conn.executemany(
                """
                INSERT OR REPLACE INTO market_data
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
            """,
                data,
            )
            await conn.commit()
            logger.debug(f"Stored {len(data)} market data bars")

    async def get_position(self, symbol: str, portfolio_id: str = DEFAULT_PORTFOLIO_ID) -> Optional[Dict]:
        """Get position for a specific symbol in a portfolio."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT symbol, quantity, avg_cost, market_price
                FROM positions
                WHERE portfolio_id = ? AND symbol = ? AND quantity != 0
            """,
                (portfolio_id, symbol),
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "symbol": row[0],
                    "quantity": row[1],
                    "avg_cost": row[2],
                    "market_price": row[3],
                }
            return None

    async def get_positions(self, portfolio_id: str = DEFAULT_PORTFOLIO_ID) -> List[Dict]:
        """Get all current positions for a portfolio."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT symbol, quantity, avg_cost, market_price
                FROM positions
                WHERE portfolio_id = ? AND quantity != 0
            """,
                (portfolio_id,),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "symbol": row[0],
                    "quantity": row[1],
                    "avg_cost": row[2],
                    "market_price": row[3],
                }
                for row in rows
            ]

    async def get_all_positions(self) -> List[Dict]:
        """Get all current positions across ALL portfolios."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT portfolio_id, symbol, quantity, avg_cost, market_price
                FROM positions
                WHERE quantity != 0
            """
            )
            rows = await cursor.fetchall()
            return [
                {
                    "portfolio_id": row[0],
                    "symbol": row[1],
                    "quantity": row[2],
                    "avg_cost": row[3],
                    "market_price": row[4],
                }
                for row in rows
            ]

    async def has_recent_buy_trade(
        self, symbol: str, seconds: int = 60, portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> bool:
        """
        Check if a BUY trade for the symbol exists within the last N seconds.

        Used to prevent duplicate BUY trades across strategy systems (main + pairs).

        Args:
            symbol: Stock symbol to check (validated against symbol format)
            seconds: Time window in seconds (default 60, must be 1-86400)
            portfolio_id: Portfolio to scope the check to

        Returns:
            True if a BUY trade exists within the time window

        Raises:
            ValidationError: If symbol format is invalid
            ValueError: If seconds is not a positive integer in valid range
        """
        # Validate symbol for consistency with other methods
        symbol = DatabaseValidator.validate_symbol(symbol)

        # Validate seconds parameter - must be positive int in reasonable range
        if not isinstance(seconds, int):
            raise ValueError(f"seconds must be int, got {type(seconds).__name__}")
        if seconds <= 0 or seconds > 86400:  # Max 24 hours
            raise ValueError(f"seconds must be between 1 and 86400, got {seconds}")

        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM trades
                WHERE portfolio_id = ?
                AND symbol = ?
                AND side = 'BUY'
                AND timestamp > datetime('now', ? || ' seconds')
                """,
                (portfolio_id, symbol, f"-{seconds}"),
            )
            row = await cursor.fetchone()
            return row[0] > 0 if row else False

    async def has_recent_sell_trade(
        self, symbol: str, seconds: int = 60, portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> bool:
        """
        Check if a SELL trade for the symbol exists within the last N seconds.

        Used to prevent duplicate SELL trades in pairs trading strategy.

        Args:
            symbol: Stock symbol to check (validated against symbol format)
            seconds: Time window in seconds (default 60, must be 1-86400)
            portfolio_id: Portfolio to scope the check to

        Returns:
            True if a SELL trade exists within the time window

        Raises:
            ValidationError: If symbol format is invalid
            ValueError: If seconds is not a positive integer in valid range
        """
        # Validate symbol for consistency with other methods
        symbol = DatabaseValidator.validate_symbol(symbol)

        # Validate seconds parameter - must be positive int in reasonable range
        if not isinstance(seconds, int):
            raise ValueError(f"seconds must be int, got {type(seconds).__name__}")
        if seconds <= 0 or seconds > 86400:  # Max 24 hours
            raise ValueError(f"seconds must be between 1 and 86400, got {seconds}")

        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM trades
                WHERE portfolio_id = ?
                AND symbol = ?
                AND side = 'SELL'
                AND timestamp > datetime('now', ? || ' seconds')
                """,
                (portfolio_id, symbol, f"-{seconds}"),
            )
            row = await cursor.fetchone()
            return row[0] > 0 if row else False

    async def get_recent_trades(
        self, limit: int = 100, symbol: Optional[str] = None,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> List[Dict]:
        """Get recent trades, optionally filtered by symbol, scoped to portfolio."""
        # Validate symbol if provided
        if symbol:
            symbol = DatabaseValidator.validate_symbol(symbol)

        async with self.get_connection() as conn:
            if symbol:
                cursor = await conn.execute(
                    """
                    SELECT symbol, side, quantity, price, notional, slippage, commission, pnl, timestamp
                    FROM trades
                    WHERE portfolio_id = ? AND symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (portfolio_id, symbol, limit),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT symbol, side, quantity, price, notional, slippage, commission, pnl, timestamp
                    FROM trades
                    WHERE portfolio_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (portfolio_id, limit),
                )

            rows = await cursor.fetchall()
            return [
                {
                    "symbol": row[0],
                    "side": row[1],  # API uses 'side' for backward compatibility
                    "quantity": row[2],
                    "price": row[3],
                    "notional": row[4],
                    "slippage": row[5],
                    "commission": row[6],
                    "pnl": row[7],
                    "timestamp": row[8],
                }
                for row in rows
            ]

    async def get_account_info(self, portfolio_id: str = DEFAULT_PORTFOLIO_ID) -> Dict:
        """Get current account information for a portfolio."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp
                FROM account
                WHERE portfolio_id = ?
            """,
                (portfolio_id,),
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "cash": row[0],
                    "equity": row[1],
                    "daily_pnl": row[2],
                    "realized_pnl": row[3],
                    "unrealized_pnl": row[4],
                    "timestamp": row[5],
                }
            return {}

    async def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest market data for a symbol."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (symbol, limit),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "timestamp": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                }
                for row in rows
            ]

    async def save_equity_snapshot(
        self,
        equity: float,
        cash: float = 0.0,
        positions_value: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        snapshot_date: Optional[str] = None,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Save a daily equity snapshot for portfolio value tracking.

        This is the industry standard approach for tracking portfolio value over time.
        Called at end of each trading day or when account summary is updated.
        """
        if snapshot_date is None:
            snapshot_date = datetime.now().strftime("%Y-%m-%d")

        async with self.get_connection() as conn:
            # Use INSERT OR REPLACE to update if date already exists for this portfolio
            await conn.execute(
                """
                INSERT OR REPLACE INTO equity_history
                (portfolio_id, date, equity, cash, positions_value, realized_pnl, unrealized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (portfolio_id, snapshot_date, equity, cash, positions_value, realized_pnl, unrealized_pnl),
            )
            await conn.commit()
            logger.debug(f"Saved equity snapshot: {snapshot_date} equity={equity:.2f} (portfolio={portfolio_id})")

    async def get_equity_history(
        self, days: int = 365, portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> List[Dict]:
        """Get equity history for the specified number of days.

        Returns list of daily snapshots ordered by date ascending (oldest first).
        """
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT date, equity, cash, positions_value, realized_pnl, unrealized_pnl, timestamp
                FROM equity_history
                WHERE portfolio_id = ?
                ORDER BY date ASC
                LIMIT ?
            """,
                (portfolio_id, days),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "date": row[0],
                    "equity": row[1],
                    "cash": row[2],
                    "positions_value": row[3],
                    "realized_pnl": row[4],
                    "unrealized_pnl": row[5],
                    "timestamp": row[6],
                }
                for row in rows
            ]

    # ── Portfolio management methods ──

    async def get_portfolios(self) -> List[Dict]:
        """Get all portfolio definitions."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, name, starting_cash, symbols, active,
                       max_position_pct, max_daily_loss_pct, max_open_positions,
                       stop_loss_pct, trailing_stop_pct, use_trailing_stop,
                       enabled_strategies, min_confidence,
                       created_at, updated_at
                FROM portfolios
                ORDER BY created_at ASC
            """
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "starting_cash": row[2],
                    "symbols": row[3],
                    "active": bool(row[4]),
                    "max_position_pct": row[5],
                    "max_daily_loss_pct": row[6],
                    "max_open_positions": row[7],
                    "stop_loss_pct": row[8],
                    "trailing_stop_pct": row[9],
                    "use_trailing_stop": bool(row[10]) if row[10] is not None else None,
                    "enabled_strategies": row[11],
                    "min_confidence": row[12],
                    "created_at": row[13],
                    "updated_at": row[14],
                }
                for row in rows
            ]

    async def upsert_portfolio(self, portfolio_data: Dict) -> None:
        """Insert or update a portfolio definition."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO portfolios
                    (id, name, starting_cash, symbols, active,
                     max_position_pct, max_daily_loss_pct, max_open_positions,
                     stop_loss_pct, trailing_stop_pct, use_trailing_stop,
                     enabled_strategies, min_confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    portfolio_data["id"],
                    portfolio_data.get("name", portfolio_data["id"]),
                    portfolio_data.get("starting_cash", 100000),
                    portfolio_data.get("symbols", ""),
                    1 if portfolio_data.get("active", True) else 0,
                    portfolio_data.get("max_position_pct"),
                    portfolio_data.get("max_daily_loss_pct"),
                    portfolio_data.get("max_open_positions"),
                    portfolio_data.get("stop_loss_pct"),
                    portfolio_data.get("trailing_stop_pct"),
                    1 if portfolio_data.get("use_trailing_stop") else (0 if portfolio_data.get("use_trailing_stop") is False else None),
                    portfolio_data.get("enabled_strategies"),
                    portfolio_data.get("min_confidence"),
                ),
            )
            await conn.commit()
            logger.info(f"Upserted portfolio: {portfolio_data['id']}")

    async def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data from the database."""
        async with self.get_connection() as conn:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 86400)

            # Clean up old market data
            await conn.execute(
                "DELETE FROM market_data WHERE timestamp < ?",
                (datetime.fromtimestamp(cutoff_date),),
            )

            # Clean up old signals (global across all portfolios - old signals have no value)
            await conn.execute(
                "DELETE FROM signals WHERE timestamp < ?",
                (datetime.fromtimestamp(cutoff_date),),
            )

            # Clean up old ticks
            await conn.execute(
                "DELETE FROM ticks WHERE timestamp < ?",
                (datetime.fromtimestamp(cutoff_date),),
            )

            await conn.commit()
            logger.info(f"Cleaned up data older than {days_to_keep} days")


# Backward compatibility wrapper
def create_async_database(db_path: Path = DB_PATH) -> AsyncTradingDatabase:
    """Create an async database instance."""
    return AsyncTradingDatabase(db_path)
