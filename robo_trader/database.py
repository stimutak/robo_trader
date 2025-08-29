"""
Database compatibility shim for backward compatibility.
This module provides a synchronous wrapper around the async database.
"""

import asyncio
from typing import Any, Dict, List, Optional

from robo_trader.database_async import AsyncTradingDatabase


class TradingDatabase:
    """Synchronous wrapper for AsyncTradingDatabase for backward compatibility."""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.async_db = AsyncTradingDatabase(db_path)
        self._loop = None
        self._initialized = False
    
    def _ensure_loop(self):
        """Ensure we have an event loop."""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
    
    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        self._ensure_loop()
        
        if not self._initialized:
            self._loop.run_until_complete(self.async_db.initialize())
            self._initialized = True
        
        return self._loop.run_until_complete(coro)
    
    def initialize(self):
        """Initialize the database connection."""
        self._ensure_loop()
        if not self._initialized:
            self._loop.run_until_complete(self.async_db.initialize())
            self._initialized = True
    
    def close(self):
        """Close the database connection."""
        if self._initialized:
            self._run_async(self.async_db.close())
            self._initialized = False
    
    def save_market_data(self, symbol: str, data: Dict[str, Any]):
        """Save market data."""
        return self._run_async(self.async_db.save_market_data(symbol, data))
    
    def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest market data."""
        return self._run_async(self.async_db.get_latest_market_data(symbol, limit))
    
    def save_trade(self, trade_data: Dict[str, Any]):
        """Save a trade."""
        return self._run_async(self.async_db.save_trade(trade_data))
    
    def get_positions(self) -> List[Dict]:
        """Get all positions."""
        return self._run_async(self.async_db.get_positions())
    
    def update_position(self, symbol: str, quantity: int, avg_cost: float):
        """Update a position."""
        return self._run_async(self.async_db.update_position(symbol, quantity, avg_cost))
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account info."""
        return self._run_async(self.async_db.get_account_info())
    
    def update_account_info(self, account_data: Dict[str, Any]):
        """Update account info."""
        return self._run_async(self.async_db.update_account_info(account_data))
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()