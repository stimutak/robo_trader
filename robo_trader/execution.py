from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .smart_execution.smart_executor import SmartExecutor, ExecutionParams, ExecutionAlgorithm


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
    """Enhanced paper executor with smart execution support.

    Models symmetric slippage (in basis points) around the limit price for both
    buy and sell orders. Supports smart execution algorithms when configured.

    Args:
        slippage_bps: Slippage in basis points to apply symmetrically to fills.
                      Defaults to 0.0 (no slippage).
        smart_executor: Optional SmartExecutor for advanced execution algorithms.
        use_smart_execution: Whether to use smart execution for orders.

    Attributes:
        fills: Dictionary tracking all executed orders with timestamps and fill prices.
        slippage_bps: Configured slippage in basis points.
        smart_executor: SmartExecutor instance for advanced algorithms.
        use_smart_execution: Flag to enable smart execution.
    """

    def __init__(
        self, 
        slippage_bps: float = 0.0,
        smart_executor: Optional['SmartExecutor'] = None,
        use_smart_execution: bool = False
    ) -> None:
        self.fills: Dict[str, Tuple[dt.datetime, Order, float]] = {}
        self.slippage_bps = float(slippage_bps)
        self.smart_executor = smart_executor
        self.use_smart_execution = use_smart_execution
        self._execution_cache: Dict[str, float] = {}  # Cache recent prices

    def place_order(self, order: Order) -> ExecutionResult:
        """Place order with optional smart execution."""
        if order.quantity <= 0:
            return ExecutionResult(False, "Quantity must be positive")
        side = order.side.upper()
        if side not in {"BUY", "SELL"}:
            return ExecutionResult(False, "Side must be BUY or SELL")
        
        # Use smart execution if enabled and available
        if self.use_smart_execution and self.smart_executor:
            return self._place_smart_order(order)
        
        # Standard paper execution
        return self._place_simple_order(order)
    
    def _place_simple_order(self, order: Order) -> ExecutionResult:
        """Place order with simple paper execution."""
        # For paper, we mark fill at limit price if provided else 0.0 as placeholder
        base = order.price if order.price is not None else 0.0
        # Apply symmetric slippage in basis points
        slip = base * (self.slippage_bps / 10_000.0) if base > 0 else 0.0
        fill = base + slip if order.side.upper() == "BUY" else base - slip
        self.fills[f"{order.symbol}-{len(self.fills)+1}"] = (
            dt.datetime.utcnow(),
            order,
            fill,
        )
        return ExecutionResult(True, "Paper fill", fill)
    
    def _place_smart_order(self, order: Order) -> ExecutionResult:
        """Place order using smart execution algorithms."""
        try:
            # Run async smart execution in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._execute_smart_order_async(order)
            )
            loop.close()
            return result
        except Exception as e:
            # Fallback to simple execution on error
            import structlog
            logger = structlog.get_logger(__name__)
            logger.warning(
                "Smart execution failed, falling back to simple",
                error=str(e),
                symbol=order.symbol
            )
            return self._place_simple_order(order)
    
    async def _execute_smart_order_async(self, order: Order) -> ExecutionResult:
        """Execute order using smart algorithms (async)."""
        from .smart_execution.smart_executor import ExecutionParams, ExecutionAlgorithm
        
        # Select algorithm based on order size
        algorithm = self._select_algorithm(order)
        
        # Create execution parameters
        params = ExecutionParams(
            algorithm=algorithm,
            duration_minutes=5 if order.quantity < 1000 else 15,
            slice_count=min(10, max(1, order.quantity // 100)),
            max_participation=0.15,
            urgency=0.7 if order.price else 0.5  # Higher urgency for limit orders
        )
        
        # Create and execute plan
        plan = await self.smart_executor.create_execution_plan(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            params=params
        )
        
        # Execute with paper fills
        result = await self.smart_executor.execute_plan(plan, self)
        
        # Convert to ExecutionResult
        if result.success:
            self.fills[f"{order.symbol}-{len(self.fills)+1}"] = (
                dt.datetime.utcnow(),
                order,
                result.average_price,
            )
            return ExecutionResult(True, result.message, result.average_price)
        else:
            return ExecutionResult(False, result.message)
    
    def _select_algorithm(self, order: Order) -> 'ExecutionAlgorithm':
        """Select execution algorithm based on order characteristics."""
        from .smart_execution.smart_executor import ExecutionAlgorithm
        
        # Algorithm selection logic
        if order.quantity < 500:
            return ExecutionAlgorithm.MARKET
        elif order.quantity < 2000:
            return ExecutionAlgorithm.TWAP
        elif order.quantity < 5000:
            return ExecutionAlgorithm.VWAP
        elif order.quantity < 10000:
            return ExecutionAlgorithm.ADAPTIVE
        else:
            return ExecutionAlgorithm.ICEBERG
