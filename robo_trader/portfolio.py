from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import csv


@dataclass
class PositionSnapshot:
    symbol: str
    quantity: int
    avg_price: float


class Portfolio:
    """In-memory portfolio with realized/unrealized PnL computation.

    This is intentionally minimal and deterministic. No brokerage integration here.
    """

    def __init__(self, starting_cash: float) -> None:
        self.cash: float = float(starting_cash)
        self.positions: Dict[str, PositionSnapshot] = {}
        self.realized_pnl: float = 0.0

    def update_fill(self, symbol: str, side: str, quantity: int, price: float) -> None:
        if quantity <= 0 or price <= 0:
            return
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            return

        pos = self.positions.get(symbol)
        if side == "BUY":
            cost = price * quantity
            self.cash -= cost
            if pos is None:
                self.positions[symbol] = PositionSnapshot(symbol, quantity, price)
            else:
                total_qty = pos.quantity + quantity
                new_avg = (pos.avg_price * pos.quantity + price * quantity) / total_qty
                self.positions[symbol] = PositionSnapshot(symbol, total_qty, new_avg)
        else:  # SELL
            if pos is None:
                # Short selling not supported in this minimal model
                return
            sell_notional = price * quantity
            realized = (price - pos.avg_price) * min(quantity, pos.quantity)
            self.cash += sell_notional
            self.realized_pnl += realized
            remaining = pos.quantity - quantity
            if remaining > 0:
                self.positions[symbol] = PositionSnapshot(
                    symbol, remaining, pos.avg_price
                )
            else:
                self.positions.pop(symbol, None)

    def compute_unrealized(self, symbol_to_market_price: Dict[str, float]) -> float:
        unrealized = 0.0
        for sym, pos in self.positions.items():
            mp = symbol_to_market_price.get(sym)
            if mp is None:
                continue
            unrealized += (mp - pos.avg_price) * pos.quantity
        return unrealized

    def equity(self, symbol_to_market_price: Dict[str, float]) -> float:
        # Equity = cash + current market value of positions
        value = 0.0
        for sym, pos in self.positions.items():
            mp = symbol_to_market_price.get(sym, pos.avg_price)
            value += mp * pos.quantity
        return self.cash + value

    def export_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "quantity", "avg_price"])
            for pos in self.positions.values():
                writer.writerow([pos.symbol, pos.quantity, pos.avg_price])
