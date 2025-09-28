from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
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
    """

    def __init__(self, starting_cash: float) -> None:
        self.cash: Decimal = PrecisePricing.to_decimal(starting_cash)
        self.positions: Dict[str, PositionSnapshot] = {}
        self.realized_pnl: Decimal = Decimal('0.0')

    def update_fill(self, symbol: str, side: str, quantity: int, price: float) -> None:
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

    def compute_unrealized(self, symbol_to_market_price: Dict[str, float]) -> Decimal:
        unrealized = Decimal('0.0')
        for sym, pos in self.positions.items():
            mp = symbol_to_market_price.get(sym)
            if mp is None:
                continue
            mp_d = PrecisePricing.to_decimal(mp)
            unrealized += PrecisePricing.calculate_pnl(pos.avg_price, mp_d, pos.quantity)
        return unrealized

    def equity(self, symbol_to_market_price: Dict[str, float]) -> Decimal:
        # Equity = cash + current market value of positions
        value = Decimal('0.0')
        for sym, pos in self.positions.items():
            mp = symbol_to_market_price.get(sym, float(pos.avg_price))
            mp_d = PrecisePricing.to_decimal(mp)
            value += PrecisePricing.calculate_notional(pos.quantity, mp_d)
        return self.cash + value

    def export_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "quantity", "avg_price"])
            for pos in self.positions.values():
                writer.writerow([pos.symbol, pos.quantity, float(pos.avg_price)])
