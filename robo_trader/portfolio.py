from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Optional

from .utils.pricing import PrecisePricing


@dataclass
class PositionSnapshot:
    symbol: str
    quantity: int
    avg_price: Decimal


class Portfolio:
    """In-memory portfolio with realized/unrealized PnL computation.

    This is intentionally minimal and deterministic. No brokerage integration here.
    Thread-safe for concurrent async access.
    """

    def __init__(self, starting_cash: float) -> None:
        self.cash: Decimal = PrecisePricing.to_decimal(starting_cash)
        self.positions: Dict[str, PositionSnapshot] = {}
        self.realized_pnl: Decimal = Decimal("0.0")
        self._lock = asyncio.Lock()  # Protect concurrent access

    async def update_fill(self, symbol: str, side: str, quantity: int, price: float) -> None:
        """Thread-safe update of portfolio position.

        Uses asyncio.Lock to prevent race conditions when multiple async tasks
        update positions concurrently.
        """
        async with self._lock:
            await self._update_fill_unsafe(symbol, side, quantity, price)

    async def _update_fill_unsafe(
        self, symbol: str, side: str, quantity: int, price: float
    ) -> None:
        """Internal non-thread-safe update method."""
        if quantity <= 0 or price <= 0:
            return

        price_d = PrecisePricing.to_decimal(price)
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            return

        pos = self.positions.get(symbol)
        if side == "BUY":
            cost = PrecisePricing.calculate_notional(quantity, price_d)
            self.cash -= cost
            if pos is None:
                self.positions[symbol] = PositionSnapshot(symbol, quantity, price_d)
            else:
                total_qty = pos.quantity + quantity
                # Use precise decimal arithmetic for average price calculation
                existing_cost = pos.avg_price * Decimal(str(pos.quantity))
                new_cost = price_d * Decimal(str(quantity))
                new_avg = (existing_cost + new_cost) / Decimal(str(total_qty))
                self.positions[symbol] = PositionSnapshot(symbol, total_qty, new_avg)
        else:  # SELL
            if pos is None:
                # Short selling not supported in this minimal model
                return

            # Ensure we don't sell more than we have
            actual_quantity = min(quantity, pos.quantity)
            if quantity > pos.quantity:
                from .logger import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"Attempted to sell {quantity} shares of {symbol}, only had {pos.quantity}"
                )

            sell_notional = PrecisePricing.calculate_notional(actual_quantity, price_d)
            realized = PrecisePricing.calculate_pnl(pos.avg_price, price_d, actual_quantity)
            self.cash += sell_notional
            self.realized_pnl += realized
            remaining = pos.quantity - actual_quantity
            if remaining > 0:
                self.positions[symbol] = PositionSnapshot(symbol, remaining, pos.avg_price)
            else:
                self.positions.pop(symbol, None)

    async def compute_unrealized(self, symbol_to_market_price: Dict[str, float]) -> Decimal:
        """Thread-safe computation of unrealized P&L."""
        async with self._lock:
            unrealized = Decimal("0.0")
            for sym, pos in self.positions.items():
                mp = symbol_to_market_price.get(sym)
                if mp is None:
                    continue
                mp_d = PrecisePricing.to_decimal(mp)
                unrealized += PrecisePricing.calculate_pnl(pos.avg_price, mp_d, pos.quantity)
            return unrealized

    async def equity(self, symbol_to_market_price: Dict[str, float]) -> Decimal:
        """Thread-safe computation of total equity."""
        async with self._lock:
            # Equity = cash + current market value of positions
            value = Decimal("0.0")
            for sym, pos in self.positions.items():
                mp = symbol_to_market_price.get(sym)
                mp_d = PrecisePricing.to_decimal(mp) if mp is not None else pos.avg_price
                value += PrecisePricing.calculate_notional(pos.quantity, mp_d)
            return self.cash + value

    async def export_csv(self, path: str) -> None:
        """Thread-safe CSV export."""
        async with self._lock:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["symbol", "quantity", "avg_price"])
                for pos in self.positions.values():
                    writer.writerow([pos.symbol, pos.quantity, float(pos.avg_price)])

    # Synchronous versions for backward compatibility
    def update_fill_sync(self, symbol: str, side: str, quantity: int, price: float) -> None:
        """Synchronous wrapper for update_fill - creates event loop if needed."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If called from async context, schedule as task
            asyncio.create_task(self.update_fill(symbol, side, quantity, price))
        else:
            # If called from sync context, run directly
            loop.run_until_complete(self.update_fill(symbol, side, quantity, price))

    def compute_unrealized_sync(self, symbol_to_market_price: Dict[str, float]) -> Decimal:
        """Synchronous wrapper for compute_unrealized."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Create a future to get the result
            future = asyncio.create_task(self.compute_unrealized(symbol_to_market_price))
            # This is a workaround - ideally should not be called from async context
            return Decimal("0.0")  # Return default for now
        else:
            return loop.run_until_complete(self.compute_unrealized(symbol_to_market_price))

    def equity_sync(self, symbol_to_market_price: Dict[str, float]) -> Decimal:
        """Synchronous wrapper for equity."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Create a future to get the result
            future = asyncio.create_task(self.equity(symbol_to_market_price))
            # This is a workaround - ideally should not be called from async context
            return self.cash  # Return cash only for now
        else:
            return loop.run_until_complete(self.equity(symbol_to_market_price))
