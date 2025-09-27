"""
Enhanced Order Management System with State Tracking

This module provides comprehensive order lifecycle management with
retry logic, timeout monitoring, and partial fill tracking.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from robo_trader.logger import get_logger

from .utils.market_time import get_market_time
from .utils.pricing import PrecisePricing

logger = get_logger(__name__)


class OrderStatus(Enum):
    """Order lifecycle states."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"
    TIMEOUT = "timeout"


class OrderType(Enum):
    """Order types."""

    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"
    TRAILING_STOP = "TRAIL"


@dataclass
class OrderDetails:
    """Complete order details with tracking information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    quantity: int = 0
    side: str = ""  # 'BUY' or 'SELL'
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=get_market_time)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    last_update: datetime = field(default_factory=get_market_time)
    retry_count: int = 0
    error_message: Optional[str] = None
    broker_order_id: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.ERROR,
            OrderStatus.TIMEOUT,
        ]

    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    @property
    def time_in_market(self) -> Optional[timedelta]:
        """Calculate time order has been in market."""
        if self.submitted_at:
            end_time = self.filled_at or datetime.now()
            return end_time - self.submitted_at
        return None


class OrderManager:
    """Enhanced order management with state tracking and retry logic."""

    def __init__(
        self,
        max_retries: int = 3,
        order_timeout: int = 60,  # seconds
        retry_delay_base: float = 2.0,  # exponential backoff base
        max_concurrent_orders: int = 10,
    ):
        self.max_retries = max_retries
        self.order_timeout = order_timeout
        self.retry_delay_base = retry_delay_base
        self.max_concurrent_orders = max_concurrent_orders

        # Order tracking
        self.orders: Dict[str, OrderDetails] = {}
        self.pending_orders: Set[str] = set()
        self.active_orders: Set[str] = set()

        # Monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}

        # Callbacks
        self.order_callbacks: Dict[str, List[Callable]] = {
            "submitted": [],
            "filled": [],
            "partial_fill": [],
            "cancelled": [],
            "rejected": [],
            "error": [],
            "timeout": [],
        }

        # Statistics
        self.stats = {
            "total_orders": 0,
            "successful_fills": 0,
            "partial_fills": 0,
            "cancellations": 0,
            "rejections": 0,
            "errors": 0,
            "timeouts": 0,
            "total_retries": 0,
        }

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for order events."""
        if event in self.order_callbacks:
            self.order_callbacks[event].append(callback)

    async def can_place_order(self) -> bool:
        """Check if we can place a new order."""
        active_count = len(self.pending_orders) + len(self.active_orders)
        if active_count >= self.max_concurrent_orders:
            logger.warning(
                f"Max concurrent orders reached: {active_count}/{self.max_concurrent_orders}"
            )
            return False
        return True

    async def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        executor=None,  # The execution backend
    ) -> OrderDetails:
        """Place an order with retry logic and monitoring."""

        # Check concurrent order limit
        if not await self.can_place_order():
            order = OrderDetails(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=order_type,
                status=OrderStatus.REJECTED,
                error_message="Max concurrent orders exceeded",
            )
            self.orders[order.id] = order
            return order

        # Create order
        order = OrderDetails(
            symbol=symbol,
            quantity=quantity,
            side=side.upper(),
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        # Validate order
        validation_error = self._validate_order(order)
        if validation_error:
            order.status = OrderStatus.REJECTED
            order.error_message = validation_error
            self.orders[order.id] = order
            await self._trigger_callbacks("rejected", order)
            return order

        # Track order
        self.orders[order.id] = order
        self.pending_orders.add(order.id)
        self.stats["total_orders"] += 1

        # Submit with retry logic
        success = await self._submit_with_retry(order, executor)

        if success:
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_order(order, executor))
            self.monitoring_tasks[order.id] = monitor_task

        return order

    def _validate_order(self, order: OrderDetails) -> Optional[str]:
        """Validate order parameters."""
        if order.quantity <= 0:
            return "Quantity must be positive"

        if order.side not in ["BUY", "SELL", "BUY_TO_COVER", "SELL_SHORT"]:
            return f"Invalid order side: {order.side}"

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            return "Limit order requires limit price"

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return "Stop order requires stop price"

        if order.limit_price is not None and order.limit_price <= 0:
            return "Invalid limit price"

        if order.stop_price is not None and order.stop_price <= 0:
            return "Invalid stop price"

        return None

    async def _submit_with_retry(self, order: OrderDetails, executor) -> bool:
        """Submit order with exponential backoff retry."""

        for attempt in range(self.max_retries):
            try:
                # Update retry count
                order.retry_count = attempt

                # Submit order
                if executor:
                    from .execution import Order as ExecutionOrder

                    exec_order = ExecutionOrder(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        side=order.side,
                        price=order.limit_price,
                    )

                    result = executor.place_order(exec_order)

                    if result.ok:
                        order.status = OrderStatus.SUBMITTED
                        order.submitted_at = datetime.now()
                        order.broker_order_id = getattr(result, "order_id", None)
                        self.pending_orders.discard(order.id)
                        self.active_orders.add(order.id)

                        logger.info(f"Order {order.id} submitted successfully for {order.symbol}")
                        await self._trigger_callbacks("submitted", order)
                        return True
                    else:
                        order.error_message = result.message
                        logger.warning(f"Order submission failed: {result.message}")
                else:
                    # Simulate submission for testing
                    order.status = OrderStatus.SUBMITTED
                    order.submitted_at = datetime.now()
                    self.pending_orders.discard(order.id)
                    self.active_orders.add(order.id)
                    return True

            except Exception as e:
                order.error_message = str(e)
                logger.error(
                    f"Order submission error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay_base**attempt
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    self.stats["total_retries"] += 1

        # All retries failed
        order.status = OrderStatus.ERROR
        order.last_update = datetime.now()
        self.pending_orders.discard(order.id)
        self.stats["errors"] += 1
        await self._trigger_callbacks("error", order)
        return False

    async def _monitor_order(self, order: OrderDetails, executor) -> None:
        """Monitor order until filled, cancelled, or timeout."""

        start_time = time.time()

        try:
            while not order.is_complete:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.order_timeout:
                    logger.warning(f"Order {order.id} timed out after {elapsed:.1f}s")
                    order.status = OrderStatus.TIMEOUT
                    order.last_update = datetime.now()
                    self.stats["timeouts"] += 1

                    # Attempt to cancel
                    await self.cancel_order(order.id, executor)
                    await self._trigger_callbacks("timeout", order)
                    break

                # Check order status with broker
                if executor and hasattr(executor, "get_order_status"):
                    status_update = executor.get_order_status(order.broker_order_id)
                    if status_update:
                        await self._update_order_status(order, status_update)

                # Sleep before next check
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error monitoring order {order.id}: {e}")
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            order.last_update = datetime.now()
            self.stats["errors"] += 1
            await self._trigger_callbacks("error", order)

        finally:
            # Clean up
            self.active_orders.discard(order.id)
            if order.id in self.monitoring_tasks:
                del self.monitoring_tasks[order.id]

    async def _update_order_status(self, order: OrderDetails, status_update: dict) -> None:
        """Update order status based on broker response."""

        old_status = order.status

        # Update filled quantity
        if "filled_quantity" in status_update:
            order.filled_quantity = status_update["filled_quantity"]

            if order.filled_quantity > 0 and order.filled_quantity < order.quantity:
                order.status = OrderStatus.PARTIAL_FILL
                if old_status != OrderStatus.PARTIAL_FILL:
                    await self._trigger_callbacks("partial_fill", order)
            elif order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                self.stats["successful_fills"] += 1
                await self._trigger_callbacks("filled", order)

        # Update average fill price
        if "avg_fill_price" in status_update:
            order.average_fill_price = status_update["avg_fill_price"]

        # Check for cancellation/rejection
        if status_update.get("status") == "cancelled":
            order.status = OrderStatus.CANCELLED
            self.stats["cancellations"] += 1
            await self._trigger_callbacks("cancelled", order)
        elif status_update.get("status") == "rejected":
            order.status = OrderStatus.REJECTED
            order.error_message = status_update.get("reason", "Unknown reason")
            self.stats["rejections"] += 1
            await self._trigger_callbacks("rejected", order)

        order.last_update = datetime.now()

    async def cancel_order(self, order_id: str, executor=None) -> bool:
        """Cancel an active order."""

        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        if order.is_complete:
            logger.warning(f"Cannot cancel completed order {order_id} (status: {order.status})")
            return False

        try:
            if executor and order.broker_order_id:
                # Cancel with broker
                result = executor.cancel_order(order.broker_order_id)
                if result:
                    order.status = OrderStatus.CANCELLED
                    order.last_update = datetime.now()
                    self.active_orders.discard(order_id)
                    self.stats["cancellations"] += 1
                    await self._trigger_callbacks("cancelled", order)
                    return True
            else:
                # Simulate cancellation
                order.status = OrderStatus.CANCELLED
                order.last_update = datetime.now()
                self.active_orders.discard(order_id)
                return True

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")

        return False

    async def cancel_all_orders(self, executor=None) -> int:
        """Cancel all active orders."""

        cancelled_count = 0
        active_order_ids = list(self.active_orders)

        for order_id in active_order_ids:
            if await self.cancel_order(order_id, executor):
                cancelled_count += 1

        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count

    async def _trigger_callbacks(self, event: str, order: OrderDetails) -> None:
        """Trigger callbacks for order events."""

        if event in self.order_callbacks:
            for callback in self.order_callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(order)
                    else:
                        callback(order)
                except Exception as e:
                    logger.error(f"Error in order callback for {event}: {e}")

    def get_order(self, order_id: str) -> Optional[OrderDetails]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_active_orders(self) -> List[OrderDetails]:
        """Get all active orders."""
        return [self.orders[order_id] for order_id in self.active_orders if order_id in self.orders]

    def get_orders_by_symbol(self, symbol: str) -> List[OrderDetails]:
        """Get all orders for a symbol."""
        return [order for order in self.orders.values() if order.symbol == symbol]

    def get_statistics(self) -> dict:
        """Get order management statistics."""

        stats = self.stats.copy()
        stats["active_orders"] = len(self.active_orders)
        stats["pending_orders"] = len(self.pending_orders)
        stats["total_tracked"] = len(self.orders)

        # Calculate success rate
        if stats["total_orders"] > 0:
            stats["fill_rate"] = (stats["successful_fills"] / stats["total_orders"]) * 100
            stats["error_rate"] = (stats["errors"] / stats["total_orders"]) * 100
            stats["timeout_rate"] = (stats["timeouts"] / stats["total_orders"]) * 100

        return stats

    async def cleanup(self) -> None:
        """Clean up monitoring tasks."""

        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)

        self.monitoring_tasks.clear()
        logger.info("Order manager cleaned up")
