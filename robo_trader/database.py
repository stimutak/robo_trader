"""
Database integration for trading data persistence.

Provides methods to store and retrieve trading data, positions, and account information.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from robo_trader.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("trading_data.db")


class TradingDatabase:
    """Manages trading data persistence in SQLite."""
    
    def __init__(self, db_path: Path = DB_PATH):
        """Initialize database connection and ensure tables exist."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    market_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            """)
            
            # Tick data table (Phase 2)
            cursor.execute("""
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
            """)
            
            # Create index for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticks_symbol 
                ON ticks (symbol, timestamp DESC)
            """)
            
            # Features table (Phase 2)
            cursor.execute("""
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
            """)
            
            # Create index for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_symbol 
                ON features (symbol, timestamp DESC)
            """)
            
            # Trades table
            cursor.execute("""
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
            """)
            
            # Account table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Market data table (for historical prices)
            cursor.execute("""
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
            """)
            
            # Strategy signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert default account if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO account (id, cash, equity) 
                VALUES (1, 100000, 100000)
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def update_position(self, symbol: str, quantity: int, avg_cost: float, 
                       market_price: Optional[float] = None) -> None:
        """Update or insert a position."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if quantity == 0:
                # Close position
                cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                # Update or insert position
                cursor.execute("""
                    INSERT OR REPLACE INTO positions (symbol, quantity, avg_cost, market_price)
                    VALUES (?, ?, ?, ?)
                """, (symbol, quantity, avg_cost, market_price))
            
            conn.commit()
            logger.debug(f"Updated position: {symbol} qty={quantity} avg={avg_cost}")
    
    def record_trade(self, symbol: str, side: str, quantity: int, price: float,
                    slippage: float = 0.0, commission: float = 0.0) -> None:
        """Record a trade execution."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (symbol, side, quantity, price, slippage, commission)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, side, quantity, price, slippage, commission))
            conn.commit()
            logger.info(f"Recorded trade: {side} {quantity} {symbol} @ {price}")
    
    def update_account(self, cash: float, equity: float, daily_pnl: float = 0.0,
                      realized_pnl: float = 0.0, unrealized_pnl: float = 0.0) -> None:
        """Update account information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE account 
                SET cash = ?, equity = ?, daily_pnl = ?, 
                    realized_pnl = ?, unrealized_pnl = ?, timestamp = CURRENT_TIMESTAMP
                WHERE id = 1
            """, (cash, equity, daily_pnl, realized_pnl, unrealized_pnl))
            conn.commit()
            logger.debug(f"Updated account: cash={cash:.2f} equity={equity:.2f}")
    
    def record_signal(self, symbol: str, strategy: str, signal_type: str,
                     strength: float = 0.0, metadata: str = "") -> None:
        """Record a strategy signal."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals (symbol, strategy, signal_type, strength, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, strategy, signal_type, strength, metadata))
            conn.commit()
            logger.debug(f"Recorded signal: {strategy} {signal_type} for {symbol}")
    
    def store_market_data(self, symbol: str, timestamp: datetime,
                         open_price: float, high: float, low: float, 
                         close: float, volume: int) -> None:
        """Store market data bar."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timestamp, open_price, high, low, close, volume))
            conn.commit()
    
    def store_tick(self, timestamp: datetime, symbol: str, bid: float, ask: float, 
                   last: float, bid_size: int = 0, ask_size: int = 0, 
                   last_size: int = 0, volume: int = 0) -> None:
        """Store tick data (Phase 2)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ticks 
                (timestamp, symbol, bid, ask, last, bid_size, ask_size, last_size, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, symbol, bid, ask, last, bid_size, ask_size, last_size, volume))
            conn.commit()
    
    def store_features(self, timestamp: datetime, symbol: str, features: Dict) -> None:
        """Store calculated features (Phase 2)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Extract features with defaults
            cursor.execute("""
                INSERT OR REPLACE INTO features 
                (timestamp, symbol, rsi, macd_line, macd_signal, macd_histogram,
                 bb_upper, bb_middle, bb_lower, atr, vwap, obv,
                 sma_20, sma_50, sma_200, volume_ratio, spread_bps,
                 trend_strength, mean_reversion_signal, breakout_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, symbol,
                features.get('rsi'), features.get('macd_line'), features.get('macd_signal'),
                features.get('macd_histogram'), features.get('bb_upper'), features.get('bb_middle'),
                features.get('bb_lower'), features.get('atr'), features.get('vwap'),
                features.get('obv'), features.get('sma_20'), features.get('sma_50'),
                features.get('sma_200'), features.get('volume_ratio'), features.get('spread_bps'),
                features.get('trend_strength'), features.get('mean_reversion_signal'),
                features.get('breakout_signal')
            ))
            conn.commit()
    
    def get_recent_ticks(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent tick data for a symbol (Phase 2)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, bid, ask, last, volume
                FROM ticks
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))
            
            ticks = []
            for row in cursor.fetchall():
                ticks.append({
                    'timestamp': row[0],
                    'bid': row[1],
                    'ask': row[2],
                    'last': row[3],
                    'volume': row[4]
                })
            
            return ticks[::-1]  # Return in chronological order
    
    def get_latest_features(self, symbol: str) -> Optional[Dict]:
        """Get latest features for a symbol (Phase 2)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, rsi, macd_line, macd_signal, bb_upper, bb_lower,
                       atr, vwap, trend_strength, mean_reversion_signal, breakout_signal
                FROM features
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'timestamp': row[0],
                    'rsi': row[1],
                    'macd_line': row[2],
                    'macd_signal': row[3],
                    'bb_upper': row[4],
                    'bb_lower': row[5],
                    'atr': row[6],
                    'vwap': row[7],
                    'trend_strength': row[8],
                    'mean_reversion_signal': row[9],
                    'breakout_signal': row[10]
                }
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, quantity, avg_cost, market_price, timestamp
                FROM positions
                WHERE quantity != 0
            """)
            
            positions = []
            for row in cursor.fetchall():
                positions.append({
                    'symbol': row[0],
                    'quantity': row[1],
                    'avg_cost': row[2],
                    'market_price': row[3],
                    'timestamp': row[4]
                })
            
            return positions
    
    def get_today_trades(self) -> List[Dict]:
        """Get all trades from today."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, side, quantity, price, slippage, commission, timestamp
                FROM trades
                WHERE DATE(timestamp) = DATE('now')
                ORDER BY timestamp DESC
            """)
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'symbol': row[0],
                    'side': row[1],
                    'quantity': row[2],
                    'price': row[3],
                    'slippage': row[4],
                    'commission': row[5],
                    'timestamp': row[6]
                })
            
            return trades
    
    def get_account_info(self) -> Dict:
        """Get current account information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp
                FROM account
                WHERE id = 1
            """)
            
            row = cursor.fetchone()
            if row:
                return {
                    'cash': row[0],
                    'equity': row[1],
                    'daily_pnl': row[2],
                    'realized_pnl': row[3],
                    'unrealized_pnl': row[4],
                    'timestamp': row[5]
                }
            
            return {
                'cash': 100000,
                'equity': 100000,
                'daily_pnl': 0,
                'realized_pnl': 0,
                'unrealized_pnl': 0,
                'timestamp': datetime.now()
            }
    
    def calculate_daily_pnl(self) -> Tuple[float, float]:
        """Calculate today's realized and unrealized P&L."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate realized P&L from today's trades
            cursor.execute("""
                SELECT SUM(
                    CASE 
                        WHEN side = 'SELL' THEN quantity * price
                        WHEN side = 'BUY' THEN -quantity * price
                        ELSE 0
                    END
                ) as realized_pnl
                FROM trades
                WHERE DATE(timestamp) = DATE('now')
            """)
            
            realized_pnl = cursor.fetchone()[0] or 0.0
            
            # Calculate unrealized P&L from open positions
            cursor.execute("""
                SELECT SUM((market_price - avg_cost) * quantity) as unrealized_pnl
                FROM positions
                WHERE quantity != 0 AND market_price IS NOT NULL
            """)
            
            unrealized_pnl = cursor.fetchone()[0] or 0.0
            
            return realized_pnl, unrealized_pnl
    
    def get_current_day_prices(self, symbol: str) -> List[Dict]:
        """Get today's price data for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT open, high, low, close, volume, timestamp
                FROM market_data
                WHERE symbol = ? AND DATE(timestamp) = DATE('now')
                ORDER BY timestamp
            """, (symbol,))
            
            prices = []
            for i, row in enumerate(cursor.fetchall()):
                prices.append({
                    'open': row[0],
                    'high': row[1],
                    'low': row[2],
                    'close': row[3],
                    'price': row[3],  # Alias for compatibility with dashboard
                    'volume': row[4],
                    'timestamp': row[5],
                    'minute_index': i
                })
            
            return prices
    
    def get_last_trading_day_prices(self, symbol: str) -> List[Dict]:
        """Get last trading day's price data for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Get the most recent trading day
            cursor.execute("""
                SELECT DISTINCT DATE(timestamp) as trading_day
                FROM market_data
                WHERE symbol = ?
                ORDER BY trading_day DESC
                LIMIT 1
            """, (symbol,))
            
            last_day = cursor.fetchone()
            if not last_day:
                return []
            
            cursor.execute("""
                SELECT open, high, low, close, volume, timestamp
                FROM market_data
                WHERE symbol = ? AND DATE(timestamp) = ?
                ORDER BY timestamp
            """, (symbol, last_day[0]))
            
            prices = []
            for i, row in enumerate(cursor.fetchall()):
                prices.append({
                    'open': row[0],
                    'high': row[1],
                    'low': row[2],
                    'close': row[3],
                    'price': row[3],  # Alias for compatibility with dashboard
                    'volume': row[4],
                    'timestamp': row[5],
                    'minute_index': i
                })
            
            return prices
    
    def get_last_pnl_history(self) -> List[Dict]:
        """Get P&L history for the last trading session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get account history for today
            cursor.execute("""
                SELECT equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp
                FROM account
                WHERE DATE(timestamp) = DATE('now')
                ORDER BY timestamp
            """)
            
            history = []
            for i, row in enumerate(cursor.fetchall()):
                history.append({
                    'equity': row[0],
                    'daily_pnl': row[1],
                    'total_pnl': row[1],  # For compatibility with enhanced dashboard
                    'realized_pnl': row[2],
                    'unrealized_pnl': row[3],
                    'timestamp': row[4],
                    'minute_index': i
                })
            
            return history
    
    def get_today_pnl(self) -> Dict:
        """Get today's P&L summary."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get latest account info for today
            cursor.execute("""
                SELECT equity, daily_pnl, realized_pnl, unrealized_pnl
                FROM account
                WHERE DATE(timestamp) = DATE('now')
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                return {
                    'equity': row[0],
                    'daily_pnl': row[1],
                    'total_pnl': row[1],  # For compatibility
                    'realized_pnl': row[2],
                    'unrealized_pnl': row[3]
                }
            
            # Return default values if no data for today
            return {
                'equity': 100000,
                'daily_pnl': 0,
                'total_pnl': 0,
                'realized_pnl': 0,
                'unrealized_pnl': 0
            }
    
    def close(self) -> None:
        """Close database connection. For compatibility with enhanced dashboard."""
        # SQLite connections are closed automatically with context manager
        # This method exists for API compatibility
        pass