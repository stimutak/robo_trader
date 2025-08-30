"""
Async database integration for trading data persistence.

This implements Phase 1 F3: Async Database Operations
- Converts all SQLite operations to async using aiosqlite
- Implements connection pooling for database access
- Ensures no event loop blocking
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiosqlite

from robo_trader.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("trading_data.db")


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
                self._pool.append(conn)
                await self._available.put(conn)

            self._initialized = True
            logger.info(
                f"Async database initialized at {self.db_path} with {self.pool_size} connections"
            )

    async def close(self):
        """Close all connections in the pool."""
        for conn in self._pool:
            await conn.close()
        self._pool.clear()
        self._initialized = False
        logger.info("Closed all database connections")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._initialized:
            await self.initialize()

        conn = await self._available.get()
        try:
            yield conn
        finally:
            await self._available.put(conn)

    async def _init_database(self) -> None:
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as conn:
            # Positions table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    market_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            """
            )

            # Tick data table
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

            # Features table
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

            # Trades table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    slippage REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Account table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Market data table
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

            # Strategy signals table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Insert default account if not exists
            await conn.execute(
                """
                INSERT OR IGNORE INTO account (id, cash, equity) 
                VALUES (1, 100000, 100000)
            """
            )

            await conn.commit()

    async def update_position(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float,
        market_price: Optional[float] = None,
    ) -> None:
        """Update or insert a position asynchronously."""
        async with self.get_connection() as conn:
            if quantity == 0:
                # Close position
                await conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                # Update or insert position
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (symbol, quantity, avg_cost, market_price)
                    VALUES (?, ?, ?, ?)
                """,
                    (symbol, quantity, avg_cost, market_price),
                )

            await conn.commit()
            logger.debug(f"Updated position: {symbol} qty={quantity} avg={avg_cost}")

    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        slippage: float = 0.0,
        commission: float = 0.0,
    ) -> None:
        """Record a trade execution asynchronously."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO trades (symbol, side, quantity, price, slippage, commission)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (symbol, side, quantity, price, slippage, commission),
            )
            await conn.commit()
            logger.info(f"Recorded trade: {side} {quantity} {symbol} @ {price}")

    async def update_account(
        self,
        cash: float,
        equity: float,
        daily_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> None:
        """Update account information asynchronously."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                UPDATE account 
                SET cash = ?, equity = ?, daily_pnl = ?, 
                    realized_pnl = ?, unrealized_pnl = ?, timestamp = CURRENT_TIMESTAMP
                WHERE id = 1
            """,
                (cash, equity, daily_pnl, realized_pnl, unrealized_pnl),
            )
            await conn.commit()
            logger.debug(f"Updated account: cash={cash:.2f} equity={equity:.2f}")

    async def record_signal(
        self,
        symbol: str,
        strategy: str,
        signal_type: str,
        strength: float = 0.0,
        metadata: str = "",
    ) -> None:
        """Record a strategy signal asynchronously."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO signals (symbol, strategy, signal_type, strength, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (symbol, strategy, signal_type, strength, metadata),
            )
            await conn.commit()
            logger.debug(f"Recorded signal: {strategy} {signal_type} for {symbol}")

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

    async def get_positions(self) -> List[Dict]:
        """Get all current positions."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT symbol, quantity, avg_cost, market_price 
                FROM positions 
                WHERE quantity != 0
            """
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

    async def get_recent_trades(self, limit: int = 100, symbol: Optional[str] = None) -> List[Dict]:
        """Get recent trades, optionally filtered by symbol."""
        async with self.get_connection() as conn:
            if symbol:
                cursor = await conn.execute(
                    """
                    SELECT symbol, side, quantity, price, slippage, commission, timestamp
                    FROM trades
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (symbol, limit),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT symbol, side, quantity, price, slippage, commission, timestamp
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            rows = await cursor.fetchall()
            return [
                {
                    "symbol": row[0],
                    "side": row[1],
                    "quantity": row[2],
                    "price": row[3],
                    "slippage": row[4],
                    "commission": row[5],
                    "timestamp": row[6],
                }
                for row in rows
            ]

    async def get_account_info(self) -> Dict:
        """Get current account information."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp
                FROM account
                WHERE id = 1
            """
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

    async def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data from the database."""
        async with self.get_connection() as conn:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 86400)

            # Clean up old market data
            await conn.execute(
                "DELETE FROM market_data WHERE timestamp < ?",
                (datetime.fromtimestamp(cutoff_date),),
            )

            # Clean up old signals
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
