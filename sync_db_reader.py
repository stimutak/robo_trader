#!/usr/bin/env python3
"""Simple synchronous database reader for dashboard"""

import sqlite3
from typing import Dict, List, Optional
from datetime import datetime

class SyncDatabaseReader:
    """Simple sync database reader to avoid locking issues"""
    
    def __init__(self, db_path='trading_data.db'):
        self.db_path = db_path
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=1.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT symbol, quantity, avg_cost, market_price, timestamp
                FROM positions 
                WHERE quantity != 0
            """)
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def get_recent_trades(self, limit: int = 100, symbol: Optional[str] = None) -> List[Dict]:
        """Get recent trades"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=1.0)
            conn.row_factory = sqlite3.Row
            
            if symbol:
                cursor = conn.execute("""
                    SELECT * FROM trades
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (symbol, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM trades
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting trades: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=1.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp
                FROM account
                WHERE id = 1
            """)
            row = cursor.fetchone()
            conn.close()
            if row:
                return dict(row)
            return {'cash': 100000, 'equity': 100000, 'daily_pnl': 0, 'realized_pnl': 0, 'unrealized_pnl': 0}
        except Exception as e:
            print(f"Error getting account: {e}")
            return {'cash': 100000, 'equity': 100000, 'daily_pnl': 0, 'realized_pnl': 0, 'unrealized_pnl': 0}
    
    def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest market data for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=1.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting market data: {e}")
            return []
    
    def get_signals(self, hours: int = 1) -> List[Dict]:
        """Get recent signals"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=1.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT symbol, strategy, signal_type, strength, metadata, timestamp
                FROM signals
                WHERE timestamp >= datetime('now', '-{hours} hour')
                ORDER BY timestamp DESC
            """)
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting signals: {e}")
            return []