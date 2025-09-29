from __future__ import annotations

import asyncio
import datetime as dt
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .smart_execution.smart_executor import ExecutionAlgorithm, ExecutionParams, SmartExecutor

from robo_trader.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Order:
    symbol: str
    quantity: int
    side: str  # "BUY" or "SELL"
    price: Optional[float] = None  # None implies market


class ExecutionResult:
    def __init__(self, ok: bool, message: str, fill_price: Optional[float] = None) -> None:
        self.ok = ok
        self.message = message
        self.fill_price = fill_price


class AbstractExecutor:
    def place_order(self, order: Order) -> ExecutionResult:  # pragma: no cover - interface
        raise NotImplementedError


class BaseExecutor(AbstractExecutor):
    """Base executor with common functionality for all executor types.

    Provides shared validation, smart execution support, and algorithm selection.
    """

    def __init__(
        self,
        use_smart_execution: bool = False,
        skip_execution_delays: bool = True,
    ) -> None:
        self.use_smart_execution = use_smart_execution
        self.skip_execution_delays = skip_execution_delays
        self.smart_executor = None
        self._execution_cache: Dict[str, float] = {}

        # Initialize smart executor if enabled
        if use_smart_execution:
            from .smart_execution.smart_executor import SmartExecutor

            self.smart_executor = SmartExecutor()

    def validate_order(self, order: Order) -> Optional[ExecutionResult]:
        """Validate order parameters. Returns error result if invalid, None if valid."""
        if order.quantity <= 0:
            return ExecutionResult(False, "Quantity must be positive")

        side = order.side.upper()
        if side not in {"BUY", "SELL", "BUY_TO_COVER", "SELL_SHORT"}:
            return ExecutionResult(False, "Invalid order side")

        # Check kill switch
        kill_switch_file = "data/kill_switch.lock"
        if os.path.exists(kill_switch_file):
            logger.error(f"KILL SWITCH ACTIVE - Order blocked for {order.symbol}")
            return ExecutionResult(False, "Kill switch active")

        return None  # Order is valid

    def _select_algorithm(self, order: Order) -> "ExecutionAlgorithm":
        """Select execution algorithm based on order characteristics."""
        from .smart_execution.smart_executor import ExecutionAlgorithm

        # Algorithm selection logic (shared by all executors)
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

    async def _execute_smart_order_async(self, order: Order) -> ExecutionResult:
        """Execute order using smart algorithms (async)."""
        from .smart_execution.smart_executor import ExecutionAlgorithm
        from .smart_execution.smart_executor import ExecutionParams as ExecParams

        # Select algorithm
        algorithm = self._select_algorithm(order)

        # Create execution parameters
        params = ExecParams(
            algorithm=algorithm,
            duration_minutes=5 if order.quantity < 1000 else 15,
            slice_count=min(10, max(1, order.quantity // 100)),
            max_participation=0.15,
            urgency=0.7 if order.price else 0.5,
        )

        # Create and execute plan
        plan = await self.smart_executor.create_execution_plan(
            symbol=order.symbol, side=order.side, quantity=order.quantity, params=params
        )

        # Execute with configurable delays
        result = await self.smart_executor.execute_plan(
            plan, self, skip_delays=self.skip_execution_delays
        )

        # Convert to ExecutionResult
        if result.success:
            return ExecutionResult(True, result.message, result.average_price)
        else:
            return ExecutionResult(False, result.message)

    async def execute_order(
        self, symbol: str, side: str, quantity: int, order_type: str, limit_price: float = None
    ) -> Dict[str, Any]:
        """Execute order slice for smart executor (must be implemented by subclass)."""
        raise NotImplementedError


class PaperExecutor(BaseExecutor):
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
        smart_executor: Optional["SmartExecutor"] = None,
        use_smart_execution: bool = False,
        skip_execution_delays: bool = True,
    ) -> None:
        super().__init__(use_smart_execution, skip_execution_delays)
        self.fills: Dict[str, Tuple[dt.datetime, Order, float]] = {}
        self.slippage_bps = float(slippage_bps)
        if smart_executor:
            self.smart_executor = smart_executor

    def place_order(self, order: Order) -> ExecutionResult:
        """Place order with optional smart execution."""
        # Validate order
        validation_result = self.validate_order(order)
        if validation_result:
            return validation_result

        # Use smart execution if enabled and available
        if self.use_smart_execution and self.smart_executor:
            return self._place_smart_order(order)

        # Standard paper execution
        return self._place_simple_order(order)

    async def place_order_async(self, order: Order) -> ExecutionResult:
        """Place order asynchronously with smart execution support."""
        # Validate order
        validation_result = self.validate_order(order)
        if validation_result:
            return validation_result

        # Use smart execution if enabled and available
        if self.use_smart_execution and self.smart_executor:
            try:
                return await self._execute_smart_order_async(order)
            except Exception as e:
                logger.warning(
                    "Smart execution failed, falling back to simple",
                    error=str(e),
                    symbol=order.symbol,
                )
                return self._place_simple_order(order)

        # Standard paper execution
        return self._place_simple_order(order)

    def _place_simple_order(self, order: Order) -> ExecutionResult:
        """Place order with simple paper execution."""
        base: Optional[float]

        if order.price is not None:
            try:
                base = float(order.price)
            except (TypeError, ValueError):
                logger.error(
                    "Invalid price provided for paper order", symbol=order.symbol, price=order.price
                )
                return ExecutionResult(False, "Invalid price for paper execution")
            if base <= 0:
                logger.error(
                    "Non-positive price provided for paper order",
                    symbol=order.symbol,
                    price=order.price,
                )
                return ExecutionResult(False, "Non-positive price for paper execution")
            self._execution_cache[order.symbol] = base
        else:
            base = self._execution_cache.get(order.symbol)
            if base is None:
                logger.error(
                    "Missing reference price for market order in paper execution",
                    symbol=order.symbol,
                )
                return ExecutionResult(False, "No reference price for market order")

        # Apply symmetric slippage in basis points
        slip = base * (self.slippage_bps / 10_000.0) if self.slippage_bps else 0.0

        # Handle all order sides including short selling
        if order.side.upper() in {"BUY", "BUY_TO_COVER"}:
            fill = base + slip
        else:  # SELL or SELL_SHORT
            fill = base - slip

        self.fills[f"{order.symbol}-{len(self.fills)+1}"] = (
            dt.datetime.utcnow(),
            order,
            fill,
        )
        return ExecutionResult(True, "Paper fill", fill)

    def _place_smart_order(self, order: Order) -> ExecutionResult:
        """Place order using smart execution algorithms."""
        try:
            # Check if there's already an event loop running
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context but called synchronously
                # Log warning and fall back to simple execution
                logger.warning(
                    "Smart execution called synchronously from async context, using simple execution",
                    symbol=order.symbol,
                )
                return self._place_simple_order(order)
            except RuntimeError:
                # No event loop running, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self._execute_smart_order_async(order))
                    return result
                finally:
                    loop.close()
        except Exception as e:
            # Fallback to simple execution on error
            logger.warning(
                "Smart execution failed, falling back to simple", error=str(e), symbol=order.symbol
            )
            return self._place_simple_order(order)

    async def _execute_smart_order_async(self, order: Order) -> ExecutionResult:
        """Execute order using smart algorithms (async)."""
        result = await super()._execute_smart_order_async(order)

        # Store fill if successful
        if result.ok:
            self.fills[f"{order.symbol}-{len(self.fills)+1}"] = (
                dt.datetime.utcnow(),
                order,
                result.fill_price,
            )

        return result

    async def execute_order(
        self, symbol: str, side: str, quantity: int, order_type: str, limit_price: float = None
    ) -> Dict[str, Any]:
        """Execute order for smart executor (async method for slices)."""
        # Create an Order object
        order = Order(symbol=symbol, quantity=quantity, side=side, price=limit_price)

        # Execute using paper fills
        result = self._place_simple_order(order)

        # Return in the format expected by smart executor
        if result.ok:
            return {
                "executed_quantity": quantity,
                "price": result.fill_price,
                "timestamp": dt.datetime.now(),
            }
        else:
            return {"executed_quantity": 0, "price": 0, "timestamp": dt.datetime.now()}


class LiveExecutor(BaseExecutor):
    """Live trading executor using IBKR connection.

    This executor places real orders through Interactive Brokers.
    It includes safety checks and supports both market and limit orders.

    Args:
        ibkr_client: AsyncIBKRClient instance for order placement
        use_smart_execution: Whether to use smart execution algorithms
        skip_execution_delays: Whether to skip delays in smart execution
    """

    def __init__(
        self,
        ibkr_client: Optional[Any] = None,
        use_smart_execution: bool = False,
        skip_execution_delays: bool = False,
    ) -> None:
        super().__init__(use_smart_execution, skip_execution_delays)
        self.ibkr_client = ibkr_client

    def place_order(self, order: Order) -> ExecutionResult:
        """Place live order through IBKR (sync wrapper).

        Note: This method creates an event loop if needed.
        Use place_order_async() for proper async support.
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context but called synchronously
            logger.warning(
                "Live order placement called synchronously from async context", symbol=order.symbol
            )
            # Create a task but can't wait for it synchronously
            asyncio.create_task(self.place_order_async(order))
            return ExecutionResult(False, "Order submitted asynchronously")
        except RuntimeError:
            # No event loop running, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.place_order_async(order))
                return result
            finally:
                loop.close()

    async def place_order_async(self, order: Order) -> ExecutionResult:
        """Place live order through IBKR asynchronously."""
        # Validate order
        validation_result = self.validate_order(order)
        if validation_result:
            return validation_result

        # Check IBKR connection
        if not self.ibkr_client or not self.ibkr_client.is_connected():
            logger.error("IBKR client not connected")
            return ExecutionResult(False, "IBKR not connected")

        # Use smart execution if enabled
        if self.use_smart_execution and self.smart_executor:
            try:
                return await self._execute_smart_order_async(order)
            except Exception as e:
                logger.warning(
                    "Smart execution failed, falling back to simple",
                    error=str(e),
                    symbol=order.symbol,
                )
                return await self._place_simple_order_async(order)

        # Standard live execution
        return await self._place_simple_order_async(order)

    async def _place_simple_order_async(self, order: Order) -> ExecutionResult:
        """Place simple live order through IBKR."""
        try:
            # Map order side to IBKR action
            action = "BUY" if order.side in {"BUY", "BUY_TO_COVER"} else "SELL"

            # Determine order type
            order_type = "LMT" if order.price else "MKT"

            # Place order through IBKR
            result = await self.ibkr_client.place_order(
                symbol=order.symbol,
                action=action,
                quantity=order.quantity,
                order_type=order_type,
                limit_price=order.price,
            )

            if result and result.get("status") == "Filled":
                fill_price = result.get("avg_fill_price", order.price or 0)
                return ExecutionResult(True, "Order filled", fill_price)
            elif result and result.get("status") == "Submitted":
                return ExecutionResult(True, "Order submitted", order.price)
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response"
                return ExecutionResult(False, f"Order failed: {error_msg}")

        except Exception as e:
            logger.error(f"Failed to place live order: {e}", symbol=order.symbol)
            return ExecutionResult(False, f"Order error: {str(e)}")

    async def execute_order(
        self, symbol: str, side: str, quantity: int, order_type: str, limit_price: float = None
    ) -> Dict[str, Any]:
        """Execute order slice for smart executor."""
        # Create an Order object
        order = Order(symbol=symbol, quantity=quantity, side=side, price=limit_price)

        # Execute using live order
        result = await self._place_simple_order_async(order)

        # Return in the format expected by smart executor
        if result.ok:
            return {
                "executed_quantity": quantity,
                "price": result.fill_price,
                "timestamp": dt.datetime.now(),
            }
        else:
            return {"executed_quantity": 0, "price": 0, "timestamp": dt.datetime.now()}
