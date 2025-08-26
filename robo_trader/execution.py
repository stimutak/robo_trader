from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Order:
    symbol: str
    quantity: int
    side: str  # "BUY" or "SELL"
    price: Optional[float] = None  # None implies market


class ExecutionResult:
    def __init__(
        self, ok: bool, message: str, fill_price: Optional[float] = None
    ) -> None:
        self.ok = ok
        self.message = message
        self.fill_price = fill_price


class AbstractExecutor:
    def place_order(
        self, order: Order
    ) -> ExecutionResult:  # pragma: no cover - interface
        raise NotImplementedError


class PaperExecutor(AbstractExecutor):
    """Simple in-memory simulator for order placement.

    Models symmetric slippage (in basis points) around the limit price for both
    buy and sell orders. Does not model partial fills. Intended as a safe default
    to test logic without touching live accounts.

    Args:
        slippage_bps: Slippage in basis points to apply symmetrically to fills.
                      Defaults to 0.0 (no slippage).

    Attributes:
        fills: Dictionary tracking all executed orders with timestamps and fill prices.
        slippage_bps: Configured slippage in basis points.
    """

    def __init__(self, slippage_bps: float = 0.0) -> None:
        self.fills: Dict[str, Tuple[dt.datetime, Order, float]] = {}
        self.slippage_bps = float(slippage_bps)

    def place_order(self, order: Order) -> ExecutionResult:
        if order.quantity <= 0:
            return ExecutionResult(False, "Quantity must be positive")
        side = order.side.upper()
        if side not in {"BUY", "SELL"}:
            return ExecutionResult(False, "Side must be BUY or SELL")
        # For paper, we mark fill at limit price if provided else 0.0 as placeholder
        base = order.price if order.price is not None else 0.0
        # Apply symmetric slippage in basis points
        slip = base * (self.slippage_bps / 10_000.0) if base > 0 else 0.0
        fill = base + slip if side == "BUY" else base - slip
        self.fills[f"{order.symbol}-{len(self.fills)+1}"] = (
            dt.datetime.utcnow(),
            order,
            fill,
        )
        return ExecutionResult(True, "Paper fill", fill)
