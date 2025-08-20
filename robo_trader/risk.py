from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float


class RiskManager:
    """Basic risk controls: position sizing and exposure checks.

    This is intentionally conservative to prevent outsized exposure by default.
    """

    def __init__(
        self,
        max_daily_loss: float,
        max_position_risk_pct: float,
        max_symbol_exposure_pct: float,
        max_leverage: float,
        max_order_notional: float | None = None,
        max_daily_notional: float | None = None,
    ) -> None:
        self.max_daily_loss = float(max_daily_loss)
        self.max_position_risk_pct = float(max_position_risk_pct)
        self.max_symbol_exposure_pct = float(max_symbol_exposure_pct)
        self.max_leverage = float(max_leverage)
        self.max_order_notional = float(max_order_notional) if max_order_notional is not None else None
        self.max_daily_notional = float(max_daily_notional) if max_daily_notional is not None else None

    def position_size(self, cash_available: float, entry_price: float) -> int:
        """Risk-based position size using a fraction of equity per position.

        Uses max_position_risk_pct of cash_available as notional per new position.
        """
        if entry_price <= 0 or cash_available <= 0:
            return 0
        notional = cash_available * self.max_position_risk_pct
        return max(int(notional // entry_price), 0)

    def validate_order(
        self,
        symbol: str,
        order_qty: int,
        price: float,
        equity: float,
        daily_pnl: float,
        current_positions: Dict[str, Position],
        daily_executed_notional: float = 0.0,
    ) -> Tuple[bool, str]:
        if daily_pnl <= -abs(self.max_daily_loss):
            return False, "Daily loss limit reached"
        if order_qty <= 0:
            return False, "Quantity must be positive"
        if price <= 0:
            return False, "Invalid price"

        symbol_exposure_notional = price * order_qty

        # Per-order notional ceiling
        if self.max_order_notional is not None and symbol_exposure_notional > self.max_order_notional:
            return False, "Order notional exceeds per-order limit"

        # Per-day notional ceiling
        if self.max_daily_notional is not None and (daily_executed_notional + symbol_exposure_notional) > self.max_daily_notional:
            return False, "Daily notional exceeds limit"
        max_symbol_notional = equity * self.max_symbol_exposure_pct
        if symbol_exposure_notional > max_symbol_notional:
            return False, "Symbol exposure exceeds limit"

        # Leverage check: sum of notionals / equity <= max_leverage
        existing_notional = sum(pos.quantity * pos.avg_price for pos in current_positions.values())
        total_after = existing_notional + symbol_exposure_notional
        if equity > 0 and (total_after / equity) > self.max_leverage:
            return False, "Account leverage exceeds limit"

        return True, "OK"


