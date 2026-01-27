"""
Active Stop-Loss Monitoring System

CRITICAL SAFETY COMPONENT: This module monitors and automatically executes stop-loss
orders to prevent excessive losses. Failure of this system could result in
significant financial loss.

Features:
- Real-time price monitoring for stop-loss triggers
- Automatic market order execution on breach
- Support for both fixed and trailing stops
- Emergency shutdown on execution failure
- Support for long and short positions
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from robo_trader.database_validator import DatabaseValidator, ValidationError
from robo_trader.execution import Order
from robo_trader.logger import get_logger
from robo_trader.risk_manager import Position

logger = get_logger(__name__)


class StopType(str, Enum):
    """Types of stop-loss orders."""

    FIXED = "fixed"
    TRAILING = "trailing"
    TRAILING_PERCENT = "trailing_percent"


class StopStatus(str, Enum):
    """Status of stop-loss orders."""

    PENDING = "pending"
    TRIGGERED = "triggered"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StopLossOrder:
    """Stop-loss order details."""

    symbol: str
    position_qty: int  # Positive for long, negative for short
    stop_price: float
    entry_price: float
    stop_type: StopType
    created_at: datetime
    status: StopStatus = StopStatus.PENDING

    # For trailing stops
    trailing_amount: Optional[float] = None  # Dollar amount for trailing
    trailing_percent: Optional[float] = None  # Percentage for trailing
    high_water_mark: Optional[float] = None  # Best price since entry

    # Execution tracking
    triggered_at: Optional[datetime] = None
    trigger_price: Optional[float] = None
    executed_at: Optional[datetime] = None
    execution_price: Optional[float] = None
    execution_order_id: Optional[str] = None

    # Risk metrics
    max_loss_amount: Optional[float] = None
    max_loss_percent: Optional[float] = None

    def __post_init__(self):
        """Calculate risk metrics after initialization."""
        if self.position_qty > 0:  # Long position
            self.max_loss_amount = abs(self.position_qty * (self.entry_price - self.stop_price))
            self.max_loss_percent = abs((self.stop_price - self.entry_price) / self.entry_price)
        else:  # Short position
            self.max_loss_amount = abs(self.position_qty * (self.stop_price - self.entry_price))
            self.max_loss_percent = abs((self.stop_price - self.entry_price) / self.entry_price)

        # Initialize high water mark for trailing stops
        if self.stop_type in [StopType.TRAILING, StopType.TRAILING_PERCENT]:
            self.high_water_mark = self.entry_price


@dataclass
class StopLossMetrics:
    """Metrics for stop-loss monitoring."""

    total_stops: int = 0
    active_stops: int = 0
    triggered_today: int = 0
    executed_today: int = 0
    failed_today: int = 0
    total_prevented_loss: float = 0.0
    largest_prevented_loss: float = 0.0
    average_trigger_time_seconds: float = 0.0
    trailing_adjustments_today: int = 0


class StopLossMonitor:
    """
    Active stop-loss monitoring and execution system.

    This is a critical safety component that prevents excessive losses by
    automatically executing stop-loss orders when price thresholds are breached.
    """

    def __init__(self, executor, risk_manager, emergency_shutdown_callback=None):
        """
        Initialize stop-loss monitor.

        Args:
            executor: Order executor for placing stop orders
            risk_manager: Risk manager for validation and limits
            emergency_shutdown_callback: Callback for emergency shutdown
        """
        self.executor = executor
        self.risk_manager = risk_manager
        self.emergency_shutdown = emergency_shutdown_callback

        # Active stop-loss orders by symbol
        self.active_stops: Dict[str, StopLossOrder] = {}

        # Historical stops for analysis
        self.stop_history: List[StopLossOrder] = []

        # Monitoring state
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Price tracking for triggers
        self.last_prices: Dict[str, float] = {}
        self.price_update_times: Dict[str, datetime] = {}

        # Metrics
        self.metrics = StopLossMetrics()
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)

        # Configuration
        self.check_interval_seconds = 1  # Check every second
        self.max_price_age_seconds = 10  # Require fresh prices
        self.max_execution_retries = 3
        self.emergency_shutdown_on_failure = True

        logger.info("Stop-loss monitor initialized")

    async def add_stop_loss(
        self,
        symbol: str,
        position: Position,
        stop_percent: float = 0.02,
        stop_type: StopType = StopType.FIXED,
        trailing_amount: Optional[float] = None,
        trailing_percent: Optional[float] = None,
    ) -> StopLossOrder:
        """
        Add stop-loss order for a position.

        Args:
            symbol: Trading symbol
            position: Current position
            stop_percent: Stop-loss percentage (default 2%)
            stop_type: Type of stop order
            trailing_amount: Dollar amount for trailing stop
            trailing_percent: Percentage for trailing stop

        Returns:
            StopLossOrder: Created stop-loss order

        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        try:
            symbol = DatabaseValidator.validate_symbol(symbol)
            stop_percent = DatabaseValidator._validate_numeric(
                stop_percent, "stop_percent", min_val=0.001, max_val=0.5
            )
        except ValidationError as e:
            logger.error(f"Invalid stop-loss parameters: {e}")
            raise

        # Calculate stop price based on position direction
        # Convert avg_price to float if it's a Decimal to avoid type mismatch
        avg_price_float = (
            float(position.avg_price)
            if isinstance(position.avg_price, Decimal)
            else position.avg_price
        )
        if position.quantity > 0:  # Long position
            stop_price = avg_price_float * (1 - stop_percent)
        else:  # Short position
            stop_price = avg_price_float * (1 + stop_percent)

        # Create stop-loss order
        stop_order = StopLossOrder(
            symbol=symbol,
            position_qty=position.quantity,
            stop_price=stop_price,
            entry_price=position.avg_price,
            stop_type=stop_type,
            created_at=datetime.now(),
            trailing_amount=trailing_amount,
            trailing_percent=trailing_percent,
        )

        # Cancel existing stop for this symbol if any
        if symbol in self.active_stops:
            old_stop = self.active_stops[symbol]
            old_stop.status = StopStatus.CANCELLED
            self.stop_history.append(old_stop)
            logger.info(f"Cancelled existing stop-loss for {symbol}")

        # Add new stop
        self.active_stops[symbol] = stop_order
        self.metrics.total_stops += 1
        self.metrics.active_stops = len(self.active_stops)

        logger.info(
            f"Stop-loss added for {symbol}: "
            f"{'LONG' if position.quantity > 0 else 'SHORT'} "
            f"{abs(position.quantity)} shares @ ${stop_price:.2f} "
            f"(max loss: ${stop_order.max_loss_amount:.2f})"
        )

        return stop_order

    async def update_price(self, symbol: str, price: float) -> None:
        """
        Update current price for a symbol.

        Args:
            symbol: Trading symbol
            price: Current market price
        """
        try:
            symbol = DatabaseValidator.validate_symbol(symbol)
            price = DatabaseValidator.validate_price(price)
        except ValidationError as e:
            logger.error(f"Invalid price update: {e}")
            return

        self.last_prices[symbol] = price
        self.price_update_times[symbol] = datetime.now()

        # Update trailing stops if needed
        if symbol in self.active_stops:
            stop = self.active_stops[symbol]
            if stop.stop_type in [StopType.TRAILING, StopType.TRAILING_PERCENT]:
                await self._update_trailing_stop(stop, price)

    async def _update_trailing_stop(self, stop: StopLossOrder, current_price: float) -> None:
        """
        Update trailing stop based on current price.

        Args:
            stop: Stop-loss order to update
            current_price: Current market price
        """
        if stop.status != StopStatus.PENDING:
            return

        # Update high water mark
        if stop.position_qty > 0:  # Long position
            if current_price > stop.high_water_mark:
                old_stop = stop.stop_price
                stop.high_water_mark = current_price

                # Adjust stop price
                if stop.stop_type == StopType.TRAILING and stop.trailing_amount:
                    stop.stop_price = current_price - stop.trailing_amount
                elif stop.stop_type == StopType.TRAILING_PERCENT and stop.trailing_percent:
                    stop.stop_price = current_price * (1 - stop.trailing_percent)

                if stop.stop_price > old_stop:
                    self.metrics.trailing_adjustments_today += 1
                    logger.debug(
                        f"Trailing stop adjusted for {stop.symbol}: "
                        f"${old_stop:.2f} -> ${stop.stop_price:.2f}"
                    )

        else:  # Short position
            if current_price < stop.high_water_mark:
                old_stop = stop.stop_price
                stop.high_water_mark = current_price

                # Adjust stop price
                if stop.stop_type == StopType.TRAILING and stop.trailing_amount:
                    stop.stop_price = current_price + stop.trailing_amount
                elif stop.stop_type == StopType.TRAILING_PERCENT and stop.trailing_percent:
                    stop.stop_price = current_price * (1 + stop.trailing_percent)

                if stop.stop_price < old_stop:
                    self.metrics.trailing_adjustments_today += 1
                    logger.debug(
                        f"Trailing stop adjusted for {stop.symbol}: "
                        f"${old_stop:.2f} -> ${stop.stop_price:.2f}"
                    )

    async def check_stops(self) -> List[StopLossOrder]:
        """
        Check all active stops and return triggered ones.

        Returns:
            List of triggered stop-loss orders
        """
        triggered = []

        for symbol, stop in list(self.active_stops.items()):
            if stop.status != StopStatus.PENDING:
                continue

            # Get current price
            current_price = self.last_prices.get(symbol)
            if not current_price:
                logger.warning(f"No price data for {symbol}, cannot check stop-loss")
                continue

            # Check price freshness
            price_age = datetime.now() - self.price_update_times.get(symbol, datetime.min)
            if price_age.total_seconds() > self.max_price_age_seconds:
                logger.warning(
                    f"Stale price data for {symbol} (age: {price_age.total_seconds():.1f}s)"
                )
                continue

            # Check if stop triggered
            triggered_flag = False

            if stop.position_qty > 0:  # Long position
                if current_price <= stop.stop_price:
                    triggered_flag = True
                    logger.warning(
                        f"STOP-LOSS TRIGGERED for {symbol} LONG: "
                        f"price ${current_price:.2f} <= stop ${stop.stop_price:.2f}"
                    )

            else:  # Short position
                if current_price >= stop.stop_price:
                    triggered_flag = True
                    logger.warning(
                        f"STOP-LOSS TRIGGERED for {symbol} SHORT: "
                        f"price ${current_price:.2f} >= stop ${stop.stop_price:.2f}"
                    )

            if triggered_flag:
                stop.status = StopStatus.TRIGGERED
                stop.triggered_at = datetime.now()
                stop.trigger_price = current_price
                triggered.append(stop)
                self.metrics.triggered_today += 1

        return triggered

    async def execute_stop_loss(self, stop: StopLossOrder) -> bool:
        """
        Execute stop-loss order immediately.

        Args:
            stop: Stop-loss order to execute

        Returns:
            bool: True if execution successful
        """
        logger.critical(
            f"EXECUTING STOP-LOSS for {stop.symbol}: "
            f"closing {'LONG' if stop.position_qty > 0 else 'SHORT'} "
            f"{abs(stop.position_qty)} shares"
        )

        # Create market order for immediate execution
        order = Order(
            symbol=stop.symbol,
            quantity=abs(stop.position_qty),
            side="SELL" if stop.position_qty > 0 else "BUY",
            price=None,  # Market order for immediate execution
        )

        # Attempt execution with retries
        for attempt in range(self.max_execution_retries):
            try:
                result = await self.executor.place_order_async(order)

                if result.ok:
                    stop.status = StopStatus.EXECUTED
                    stop.executed_at = datetime.now()
                    stop.execution_price = result.fill_price

                    # Calculate prevented loss
                    if stop.position_qty > 0:  # Long
                        prevented_loss = abs(
                            stop.position_qty * (stop.trigger_price - stop.stop_price)
                        )
                    else:  # Short
                        prevented_loss = abs(
                            stop.position_qty * (stop.stop_price - stop.trigger_price)
                        )

                    self.metrics.executed_today += 1
                    self.metrics.total_prevented_loss += prevented_loss
                    self.metrics.largest_prevented_loss = max(
                        self.metrics.largest_prevented_loss, prevented_loss
                    )

                    # Remove from active stops
                    del self.active_stops[stop.symbol]
                    self.stop_history.append(stop)
                    self.metrics.active_stops = len(self.active_stops)

                    logger.info(
                        f"Stop-loss executed successfully for {stop.symbol}: "
                        f"filled at ${result.fill_price:.2f}, "
                        f"prevented loss: ${prevented_loss:.2f}"
                    )

                    return True

                else:
                    logger.error(
                        f"Stop-loss execution failed for {stop.symbol} "
                        f"(attempt {attempt + 1}/{self.max_execution_retries}): {result.message}"
                    )

                    if attempt < self.max_execution_retries - 1:
                        await asyncio.sleep(0.5)  # Brief delay before retry

            except Exception as e:
                logger.error(
                    f"Exception during stop-loss execution for {stop.symbol} "
                    f"(attempt {attempt + 1}/{self.max_execution_retries}): {e}"
                )

                if attempt < self.max_execution_retries - 1:
                    await asyncio.sleep(0.5)

        # Execution failed after all retries
        stop.status = StopStatus.FAILED
        self.metrics.failed_today += 1

        logger.critical(
            f"CRITICAL: Stop-loss execution FAILED for {stop.symbol} after {self.max_execution_retries} attempts!"
        )

        # Trigger emergency shutdown if configured
        if self.emergency_shutdown_on_failure and self.emergency_shutdown:
            logger.critical("Triggering EMERGENCY SHUTDOWN due to stop-loss execution failure!")
            await self.emergency_shutdown("Stop-loss execution failed")

        return False

    async def monitor_stops(self) -> None:
        """
        Main monitoring loop - checks stops continuously.
        """
        logger.info("Stop-loss monitoring started")
        self.monitoring_active = True

        while self.monitoring_active:
            try:
                # Reset daily metrics if needed
                if datetime.now().date() > self.daily_reset_time.date():
                    self._reset_daily_metrics()

                # Check all stops
                triggered = await self.check_stops()

                # Execute triggered stops
                for stop in triggered:
                    success = await self.execute_stop_loss(stop)
                    if not success:
                        logger.error(f"Failed to execute stop-loss for {stop.symbol}")

                # Brief sleep before next check
                await asyncio.sleep(self.check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in stop-loss monitoring loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)

    async def start_monitoring(self) -> None:
        """Start the monitoring task."""
        if self.monitor_task and not self.monitor_task.done():
            logger.warning("Stop-loss monitoring already running")
            return

        self.monitor_task = asyncio.create_task(self.monitor_stops())
        logger.info("Stop-loss monitoring task started")

    async def stop_monitoring(self) -> None:
        """Stop the monitoring task."""
        self.monitoring_active = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stop-loss monitoring stopped")

    def get_stop_for_symbol(self, symbol: str) -> Optional[StopLossOrder]:
        """
        Get active stop-loss order for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Stop-loss order if exists
        """
        return self.active_stops.get(symbol)

    def cancel_stop(self, symbol: str) -> bool:
        """
        Cancel stop-loss order for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if stop was cancelled
        """
        if symbol in self.active_stops:
            stop = self.active_stops[symbol]
            stop.status = StopStatus.CANCELLED
            del self.active_stops[symbol]
            self.stop_history.append(stop)
            self.metrics.active_stops = len(self.active_stops)
            logger.info(f"Stop-loss cancelled for {symbol}")
            return True

        return False

    def cancel_all_stops(self) -> int:
        """
        Cancel all active stop-loss orders.

        Returns:
            Number of stops cancelled
        """
        count = 0
        for symbol in list(self.active_stops.keys()):
            if self.cancel_stop(symbol):
                count += 1

        logger.info(f"Cancelled {count} stop-loss orders")
        return count

    def get_metrics(self) -> StopLossMetrics:
        """
        Get current stop-loss metrics.

        Returns:
            Current metrics
        """
        # Calculate average trigger time
        if self.metrics.executed_today > 0:
            total_time = 0
            count = 0
            for stop in self.stop_history:
                if stop.status == StopStatus.EXECUTED and stop.triggered_at and stop.executed_at:
                    total_time += (stop.executed_at - stop.triggered_at).total_seconds()
                    count += 1

            if count > 0:
                self.metrics.average_trigger_time_seconds = total_time / count

        return self.metrics

    def _reset_daily_metrics(self) -> None:
        """Reset daily metrics."""
        self.metrics.triggered_today = 0
        self.metrics.executed_today = 0
        self.metrics.failed_today = 0
        self.metrics.trailing_adjustments_today = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        logger.info("Daily stop-loss metrics reset")

    def get_status_summary(self) -> Dict:
        """
        Get comprehensive status summary.

        Returns:
            Status dictionary
        """
        return {
            "monitoring_active": self.monitoring_active,
            "active_stops": len(self.active_stops),
            "stops_by_symbol": list(self.active_stops.keys()),
            "metrics": {
                "total_stops": self.metrics.total_stops,
                "active_stops": self.metrics.active_stops,
                "triggered_today": self.metrics.triggered_today,
                "executed_today": self.metrics.executed_today,
                "failed_today": self.metrics.failed_today,
                "total_prevented_loss": self.metrics.total_prevented_loss,
                "largest_prevented_loss": self.metrics.largest_prevented_loss,
                "average_execution_time": self.metrics.average_trigger_time_seconds,
                "trailing_adjustments": self.metrics.trailing_adjustments_today,
            },
            "last_update": datetime.now().isoformat(),
        }
