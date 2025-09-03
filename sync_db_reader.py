#!/usr/bin/env python3
"""Simple synchronous database reader for dashboard

Hardened for concurrent access with the async trader:
- Opens read-only connections with shared cache where possible
- Sets WAL mode and a reasonable busy_timeout on connect
- Uses short retry/backoff on SQLITE_BUSY/locked errors
"""

import sqlite3
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class SyncDatabaseReader:
    """Simple sync database reader to avoid locking issues"""
    
    def __init__(self, db_path='trading_data.db'):
        self.db_path = db_path
    
    def _make_uri(self, read_only: bool = True) -> str:
        """Build a SQLite URI for the database path."""
        if read_only:
            return f"file:{self.db_path}?mode=ro&cache=shared"
        return f"file:{self.db_path}?cache=shared"

    def _connect(self, read_only: bool = True) -> sqlite3.Connection:
        """Open a connection with sane defaults for concurrent reads."""
        conn = sqlite3.connect(
            self._make_uri(read_only=read_only),
            uri=True,
            timeout=5.0,           # wait for busy writer up to 5s
            isolation_level=None,  # autocommit, avoid long read txns
        )
        conn.row_factory = sqlite3.Row
        # Apply pragmas: WAL for compatibility, busy timeout, query_only when RO
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA busy_timeout=5000")
        except Exception:
            pass
        if read_only:
            try:
                conn.execute("PRAGMA query_only=ON")
            except Exception:
                pass
        return conn

    def _fetch_all(self, sql: str, params: Tuple = (), retries: int = 3) -> List[sqlite3.Row]:
        """Execute a read-only query with retry on lock contention."""
        delay = 0.1
        last_exc = None
        for attempt in range(retries):
            conn = None
            try:
                conn = self._connect(read_only=True)
                # Test connection before use
                conn.execute("SELECT 1")
                cur = conn.execute(sql, params)
                rows = cur.fetchall()
                return rows
            except sqlite3.OperationalError as e:
                last_exc = e
                msg = str(e).lower()
                if ("database is locked" in msg or "database table is locked" in msg or "resource busy" in msg) and attempt < retries - 1:
                    print(f"Database locked on attempt {attempt + 1}/{retries}, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    print(f"Database operation failed after {retries} attempts: {e}")
                    raise
            except Exception as e:
                print(f"Unexpected database error: {e}")
                raise
            finally:
                if conn is not None:
                    try:
                        # Ensure no transaction is left open
                        if conn.in_transaction:
                            conn.rollback()
                        conn.close()
                    except Exception as close_error:
                        print(f"Error closing connection: {close_error}")
                        pass

    def _fetch_one(self, sql: str, params: Tuple = (), retries: int = 3) -> Optional[sqlite3.Row]:
        rows = self._fetch_all(sql, params, retries)
        return rows[0] if rows else None
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions"""
        try:
            rows = self._fetch_all(
                """
                SELECT symbol, quantity, avg_cost, market_price, timestamp
                FROM positions
                WHERE quantity != 0
                """
            )
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def get_recent_trades(self, limit: int = 100, symbol: Optional[str] = None) -> List[Dict]:
        """Get recent trades"""
        try:
            if symbol:
                rows = self._fetch_all(
                    """
                    SELECT * FROM trades
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (symbol, limit),
                )
            else:
                rows = self._fetch_all(
                    """
                    SELECT * FROM trades
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting trades: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            row = self._fetch_one(
                """
                SELECT cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp
                FROM account
                WHERE id = 1
                """
            )
            if row:
                return dict(row)
            return {'cash': 100000, 'equity': 100000, 'daily_pnl': 0, 'realized_pnl': 0, 'unrealized_pnl': 0}
        except Exception as e:
            print(f"Error getting account: {e}")
            return {'cash': 100000, 'equity': 100000, 'daily_pnl': 0, 'realized_pnl': 0, 'unrealized_pnl': 0}
    
    def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest market data for a symbol"""
        try:
            rows = self._fetch_all(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting market data: {e}")
            return []
    
    def get_signals(self, hours: int = 1) -> List[Dict]:
        """Get recent signals"""
        try:
            rows = self._fetch_all(
                f"""
                SELECT symbol, strategy, signal_type, strength, metadata, timestamp
                FROM signals
                WHERE timestamp >= datetime('now', '-{hours} hour')
                ORDER BY timestamp DESC
                """
            )
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting signals: {e}")
            return []
