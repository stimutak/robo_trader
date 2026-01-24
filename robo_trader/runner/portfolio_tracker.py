"""
Portfolio Tracker Module - Position tracking and P&L calculation.

This module handles:
- Position state management with atomic updates
- Market price updates for positions
- P&L calculation (realized and unrealized)
- Account summary updates
- Equity snapshots for historical tracking

Extracted from runner_async.py to improve modularity.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..logger import get_logger

if TYPE_CHECKING:
    from ..database_async import AsyncTradingDatabase
    from ..portfolio import Portfolio
    from ..portfolio_manager.portfolio_manager import PortfolioManager
    from ..risk.advanced_risk import AdvancedRiskManager

logger = get_logger(__name__)


class Position:
    """Represents a position in a symbol."""

    def __init__(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        entry_time: Optional[datetime] = None,
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.entry_time = entry_time or datetime.now()

    @property
    def avg_cost(self) -> float:
        """Alias for avg_price for compatibility."""
        return self.avg_price

    @property
    def market_value(self) -> float:
        """Calculate market value (requires current price)."""
        return self.quantity * self.avg_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        return (current_price - self.avg_price) * self.quantity

    def __repr__(self) -> str:
        return f"Position({self.symbol}, qty={self.quantity}, avg=${self.avg_price:.2f})"


class PortfolioTracker:
    """
    Tracks portfolio positions and calculates P&L.

    Features:
    - Atomic position updates with per-symbol locks
    - Market price tracking
    - Realized and unrealized P&L calculation
    - Account summary persistence
    - Equity history snapshots

    Usage:
        tracker = PortfolioTracker(
            portfolio=portfolio,
            db=database,
        )
        await tracker.update_position(symbol, qty, price, side)
        summary = await tracker.get_summary()
    """

    def __init__(
        self,
        portfolio: Portfolio,
        db: AsyncTradingDatabase,
        advanced_risk: Optional[AdvancedRiskManager] = None,
        portfolio_manager: Optional[PortfolioManager] = None,
    ):
        self.portfolio = portfolio
        self.db = db
        self.advanced_risk = advanced_risk
        self.portfolio_manager = portfolio_manager

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.latest_prices: Dict[str, float] = {}

        # P&L tracking
        self.daily_pnl: float = 0.0
        self.daily_executed_notional: float = 0.0

        # Locks for atomic updates
        self._position_locks: Dict[str, asyncio.Lock] = {}
        self._lock_manager = asyncio.Lock()

        # ML predictions for dashboard
        self._ml_predictions: Dict[str, Dict] = {}

    async def _get_position_lock(self, symbol: str) -> asyncio.Lock:
        """Get or create a lock for a specific symbol."""
        async with self._lock_manager:
            if symbol not in self._position_locks:
                self._position_locks[symbol] = asyncio.Lock()
            return self._position_locks[symbol]

    async def update_position(self, symbol: str, quantity: int, price: float, side: str) -> bool:
        """
        Atomically update a position.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Execution price
            side: BUY, SELL, BUY_TO_COVER, SELL_SHORT

        Returns:
            True if update successful
        """
        lock = await self._get_position_lock(symbol)
        async with lock:
            try:
                if side.upper() in ["BUY", "BUY_TO_COVER"]:
                    return await self._handle_buy(symbol, quantity, price, side)
                elif side.upper() in ["SELL", "SELL_SHORT"]:
                    return await self._handle_sell(symbol, quantity, price, side)
                else:
                    logger.error(f"Unknown side: {side}")
                    return False

            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {e}")
                return False

    async def _handle_buy(self, symbol: str, quantity: int, price: float, side: str) -> bool:
        """Handle BUY or BUY_TO_COVER."""
        if symbol in self.positions:
            pos = self.positions[symbol]

            if side.upper() == "BUY_TO_COVER" and pos.quantity < 0:
                # Covering short position
                new_qty = pos.quantity + quantity
                if new_qty == 0:
                    del self.positions[symbol]
                else:
                    self.positions[symbol] = Position(symbol, new_qty, pos.avg_price)

            elif side.upper() == "BUY" and pos.quantity >= 0:
                # Adding to long position
                total_qty = pos.quantity + quantity
                new_avg = (pos.avg_price * pos.quantity + price * quantity) / total_qty
                self.positions[symbol] = Position(symbol, total_qty, new_avg)

            else:
                logger.error(
                    f"Invalid position update: {side} on {pos.quantity} shares of {symbol}"
                )
                return False
        else:
            # New long position
            self.positions[symbol] = Position(symbol, quantity, price)

        # Update portfolio
        await self.portfolio.update_fill(symbol, side, quantity, price)

        # Update advanced risk manager
        if self.advanced_risk:
            self.advanced_risk.update_position(symbol, quantity, price, side)

        return True

    async def _handle_sell(self, symbol: str, quantity: int, price: float, side: str) -> bool:
        """Handle SELL or SELL_SHORT."""
        if symbol in self.positions:
            pos = self.positions[symbol]

            if side.upper() == "SELL" and pos.quantity > 0:
                # Selling long position
                if quantity >= pos.quantity:
                    del self.positions[symbol]
                else:
                    remaining = pos.quantity - quantity
                    self.positions[symbol] = Position(symbol, remaining, pos.avg_price)

            else:
                logger.error(
                    f"Invalid position update: {side} on {pos.quantity} shares of {symbol}"
                )
                return False

        elif side.upper() == "SELL_SHORT":
            # New short position (negative quantity)
            self.positions[symbol] = Position(symbol, -quantity, price)

        else:
            logger.error(f"Cannot {side} {symbol}: no existing position")
            return False

        # Update portfolio
        await self.portfolio.update_fill(symbol, side, quantity, price)

        # Update advanced risk manager
        if self.advanced_risk:
            self.advanced_risk.update_position(symbol, quantity, price, side)

        return True

    def update_price(self, symbol: str, price: float) -> None:
        """Update latest price for a symbol."""
        self.latest_prices[symbol] = price

    async def update_market_prices(self, market_prices: Dict[str, float]) -> None:
        """Update market prices for all positions in database."""
        try:
            for symbol, pos in self.positions.items():
                current_price = market_prices.get(symbol)
                if current_price:
                    self.latest_prices[symbol] = current_price
                    await self.db.update_position(
                        symbol=symbol,
                        quantity=pos.quantity,
                        avg_cost=pos.avg_price,
                        market_price=current_price,
                    )
                    logger.debug(f"Updated {symbol} market price to ${current_price:.2f}")
        except Exception as e:
            logger.error(f"Error updating position market prices: {e}")

    async def update_account_summary(self) -> Dict[str, float]:
        """
        Update account summary in database.

        Returns:
            Dict with equity, cash, realized_pnl, unrealized_pnl
        """
        # Get market prices for equity calculation
        market_prices = {}
        for symbol, pos in self.positions.items():
            latest_pos = await self.db.get_position(symbol)
            if latest_pos and latest_pos.get("market_price"):
                market_prices[symbol] = latest_pos["market_price"]
            else:
                market_prices[symbol] = pos.avg_price

        # Calculate values
        equity = await self.portfolio.equity(market_prices)
        unrealized = await self.portfolio.compute_unrealized(market_prices)

        # Convert Decimal to float
        equity_float = float(equity)
        unrealized_float = float(unrealized)
        cash_float = float(self.portfolio.cash)
        realized_pnl_float = float(self.portfolio.realized_pnl)

        # Update portfolio manager
        if self.portfolio_manager:
            try:
                self.portfolio_manager.update_capital(equity_float)
                if await self.portfolio_manager.should_rebalance():
                    rb = await self.portfolio_manager.rebalance()
                    logger.info(f"Rebalanced strategies at {rb['timestamp']}: {rb['new_weights']}")
            except Exception as e:
                logger.debug(f"Portfolio manager update failed: {e}")

        # Update database
        await self.db.update_account(
            cash=cash_float,
            equity=equity_float,
            daily_pnl=self.daily_pnl,
            realized_pnl=realized_pnl_float,
            unrealized_pnl=unrealized_float,
        )

        # Save equity snapshot for historical tracking
        positions_value = sum(
            pos.quantity * market_prices.get(symbol, pos.avg_price)
            for symbol, pos in self.positions.items()
        )
        await self.db.save_equity_snapshot(
            equity=equity_float,
            cash=cash_float,
            positions_value=positions_value,
            realized_pnl=realized_pnl_float,
            unrealized_pnl=unrealized_float,
        )

        logger.info(
            f"Trading cycle complete. Equity: ${equity:,.2f}, "
            f"Unrealized: ${unrealized:,.2f}, Realized: ${realized_pnl_float:,.2f}"
        )

        return {
            "equity": equity_float,
            "cash": cash_float,
            "realized_pnl": realized_pnl_float,
            "unrealized_pnl": unrealized_float,
            "positions_value": positions_value,
        }

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def get_total_value(self) -> float:
        """Get total value of all positions."""
        total = 0.0
        for symbol, pos in self.positions.items():
            price = self.latest_prices.get(symbol, pos.avg_price)
            total += pos.quantity * price
        return total

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        total = 0.0
        for symbol, pos in self.positions.items():
            price = self.latest_prices.get(symbol, pos.avg_price)
            total += pos.unrealized_pnl(price)
        return total

    def get_winners_losers(self) -> tuple:
        """Get count of winning and losing positions."""
        winners = 0
        losers = 0
        for symbol, pos in self.positions.items():
            price = self.latest_prices.get(symbol, pos.avg_price)
            if pos.unrealized_pnl(price) >= 0:
                winners += 1
            else:
                losers += 1
        return winners, losers

    def get_best_worst_positions(self) -> tuple:
        """Get best and worst performing positions."""
        if not self.positions:
            return None, None

        best = None
        worst = None
        best_pnl = float("-inf")
        worst_pnl = float("inf")

        for symbol, pos in self.positions.items():
            price = self.latest_prices.get(symbol, pos.avg_price)
            pnl = pos.unrealized_pnl(price)

            if pnl > best_pnl:
                best_pnl = pnl
                best = (symbol, pnl, (price / pos.avg_price - 1) * 100)

            if pnl < worst_pnl:
                worst_pnl = pnl
                worst = (symbol, pnl, (price / pos.avg_price - 1) * 100)

        return best, worst

    def set_ml_predictions(self, predictions: Dict[str, Dict]) -> None:
        """Set ML predictions for dashboard."""
        self._ml_predictions = predictions

    def get_ml_predictions(self) -> Dict[str, Dict]:
        """Get ML predictions."""
        return self._ml_predictions

    async def save_ml_predictions(self, file_path: str = "ml_predictions.json") -> None:
        """Save ML predictions to file for dashboard."""
        if not self._ml_predictions:
            return

        try:
            import json
            from pathlib import Path

            predictions_file = Path(file_path)
            with open(predictions_file, "w") as f:
                json.dump(self._ml_predictions, f, indent=2)
            logger.debug(f"Saved {len(self._ml_predictions)} ML predictions to {predictions_file}")
        except Exception as e:
            logger.warning(f"Failed to save ML predictions: {e}")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.daily_executed_notional = 0.0
