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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from robo_trader.database_validator import DatabaseValidator, ValidationError
from robo_trader.execution import ExecutionResult, Order
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
    portfolio_id: str = "default"  # Multi-portfolio support

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


@dataclass(frozen=True)
class _PendingStopTrigger:
    """Immutable, validated crossing evidence awaiting monitor-loop handling."""

    stop: StopLossOrder = field(compare=False, repr=False)
    trigger_price: float
    event_time: datetime
    receipt_monotonic: float
    receipt_order: int


class StopLossMonitor:
    """
    Active stop-loss monitoring and execution system.

    This is a critical safety component that prevents excessive losses by
    automatically executing stop-loss orders when price thresholds are breached.
    """

    def __init__(
        self,
        executor,
        risk_manager,
        emergency_shutdown_callback=None,
        portfolio_id: str = "default",
        position_closed_callback: Optional[
            Callable[["StopLossOrder", "ExecutionResult"], Awaitable[None]]
        ] = None,
    ):
        """
        Initialize stop-loss monitor.

        Args:
            executor: Order executor for placing stop orders
            risk_manager: Risk manager for validation and limits
            emergency_shutdown_callback: Callback for emergency shutdown
            portfolio_id: Portfolio this monitor is scoped to
            position_closed_callback: Optional async callback invoked AFTER a
                stop-loss execution succeeds. Receives (stop_order, result).
                Used by AsyncRunner to sync `runner.positions`,
                `portfolio.update_fill`, and DB persistence so that a phantom
                position does not block subsequent BUY/SELL signals.
                (TCN-H4 followup audit fix.)
        """
        self.executor = executor
        self.risk_manager = risk_manager
        self.emergency_shutdown = emergency_shutdown_callback
        self.portfolio_id = portfolio_id
        self.position_closed_callback = position_closed_callback

        # Active stop-loss orders by portfolio:symbol key (supports multi-portfolio)
        self.active_stops: Dict[str, StopLossOrder] = {}

        # Historical stops for analysis
        self.stop_history: List[StopLossOrder] = []

        # Monitoring state
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Price tracking for triggers
        self.last_prices: Dict[str, float] = {}
        self.price_event_times: Dict[str, datetime] = {}
        self.price_receipt_monotonic: Dict[str, float] = {}
        # Broker feeds can publish multiple quotes with the same (often
        # second-resolution) event timestamp.  Preserve their local arrival
        # order independently of the monotonic clock, whose resolution is not
        # guaranteed to distinguish adjacent callbacks.
        self.price_receipt_orders: Dict[str, int] = {}
        self._price_receipt_order = 0
        self._pending_stop_triggers: Dict[str, _PendingStopTrigger] = {}
        # Serialize quote ordering and trigger draining. No broker I/O occurs
        # while this lock is held.
        self._price_update_lock = asyncio.Lock()

        # Metrics
        self.metrics = StopLossMetrics()
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)

        # Configuration
        self.check_interval_seconds = 1  # Check every second
        self.max_price_age_seconds = 10  # Require fresh prices
        self.max_execution_retries = 3
        self.emergency_shutdown_on_failure = True

        logger.info(f"Stop-loss monitor initialized for portfolio={self.portfolio_id}")

    def _stop_key(self, symbol: str) -> str:
        """Generate composite key for stop-loss storage (portfolio:symbol)."""
        return f"{self.portfolio_id}:{symbol}"

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
            entry_price=avg_price_float,  # Use float, not Decimal
            stop_type=stop_type,
            created_at=datetime.now(),
            portfolio_id=self.portfolio_id,
            trailing_amount=trailing_amount,
            trailing_percent=trailing_percent,
        )

        # Cancel existing stop for this symbol if any
        stop_key = self._stop_key(symbol)
        if stop_key in self.active_stops:
            old_stop = self.active_stops[stop_key]
            old_stop.status = StopStatus.CANCELLED
            self.stop_history.append(old_stop)
            logger.info(
                f"Cancelled existing stop-loss for {symbol} (portfolio={self.portfolio_id})"
            )

        # Add new stop
        self._pending_stop_triggers.pop(stop_key, None)
        self.active_stops[stop_key] = stop_order
        self.metrics.total_stops += 1
        self.metrics.active_stops = len(self.active_stops)

        logger.info(
            f"Stop-loss added for {symbol}: "
            f"{'LONG' if position.quantity > 0 else 'SHORT'} "
            f"{abs(position.quantity)} shares @ ${stop_price:.2f} "
            f"(max loss: ${stop_order.max_loss_amount:.2f})"
        )

        return stop_order

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    async def update_price(
        self,
        symbol: str,
        price: float,
        *,
        source_timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Update current price for a symbol.

        Args:
            symbol: Trading symbol
            price: Current market price
            source_timestamp: Timezone-aware broker event timestamp. Missing
                event time is rejected rather than replaced by receipt time.
        """
        try:
            symbol = DatabaseValidator.validate_symbol(symbol)
            price = DatabaseValidator.validate_price(price)
        except ValidationError as e:
            logger.error(f"Invalid price update: {e}")
            return False

        if source_timestamp is None or not isinstance(source_timestamp, datetime):
            logger.error("Rejected price update for %s: missing source timestamp", symbol)
            return False
        if source_timestamp.tzinfo is None or source_timestamp.utcoffset() is None:
            logger.error("Rejected price update for %s: timezone-naive timestamp", symbol)
            return False

        event_time = source_timestamp.astimezone(timezone.utc)
        async with self._price_update_lock:
            event_age = (self._utcnow() - event_time).total_seconds()
            if event_age < 0:
                logger.error("Rejected price update for %s: future source timestamp", symbol)
                return False
            if event_age > self.max_price_age_seconds:
                logger.warning(
                    "Rejected stale price update for %s (event age: %.1fs)",
                    symbol,
                    event_age,
                )
                return False

            previous_event_time = self.price_event_times.get(symbol)
            if previous_event_time is not None and event_time < previous_event_time:
                logger.warning(
                    "Rejected out-of-order price event for %s: %s < %s",
                    symbol,
                    event_time.isoformat(),
                    previous_event_time.isoformat(),
                )
                return False

            # A sequence, rather than the clock value, is the authoritative
            # tie-breaker for quotes sharing a broker timestamp. This is
            # strictly increasing even when time.monotonic() returns the same
            # tick. Every arrival is accepted because a coarse broker timestamp
            # cannot distinguish a replay from a legitimate value revisited in
            # the same interval. Event age remains authoritative, so repeated
            # callbacks cannot make an old broker event fresh.
            receipt_monotonic = self._monotonic()
            self._price_receipt_order += 1
            receipt_order = self._price_receipt_order
            self.last_prices[symbol] = price
            self.price_event_times[symbol] = event_time
            self.price_receipt_monotonic[symbol] = receipt_monotonic
            self.price_receipt_orders[symbol] = receipt_order

            stop_key = self._stop_key(symbol)
            stop = self.active_stops.get(stop_key)
            if (
                stop is not None
                and stop.status == StopStatus.PENDING
                and stop_key not in self._pending_stop_triggers
            ):
                if stop.stop_type in [StopType.TRAILING, StopType.TRAILING_PERCENT]:
                    self._update_trailing_stop(stop, price)

                if self._price_crosses_stop(stop, price):
                    stop.status = StopStatus.TRIGGERED
                    stop.triggered_at = event_time
                    stop.trigger_price = price
                    self._pending_stop_triggers[stop_key] = _PendingStopTrigger(
                        stop=stop,
                        trigger_price=price,
                        event_time=event_time,
                        receipt_monotonic=receipt_monotonic,
                        receipt_order=receipt_order,
                    )
                    self.metrics.triggered_today += 1
                    logger.warning(
                        "Latched stop-loss crossing for %s at %.2f "
                        "(stop=%.2f event=%s receipt_order=%d)",
                        symbol,
                        price,
                        stop.stop_price,
                        event_time.isoformat(),
                        receipt_order,
                    )
        return True

    @staticmethod
    def _price_crosses_stop(stop: StopLossOrder, price: float) -> bool:
        if stop.position_qty > 0:
            return price <= stop.stop_price
        return price >= stop.stop_price

    def _update_trailing_stop(self, stop: StopLossOrder, current_price: float) -> None:
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
        async with self._price_update_lock:
            triggered = []

            # Crossings validated during quote ingestion are authoritative.
            # Drain each object once before consulting the mutable latest-price
            # cache, which may already contain a later recovery quote.
            for stop_key, evidence in list(self._pending_stop_triggers.items()):
                del self._pending_stop_triggers[stop_key]
                stop = evidence.stop
                if self.active_stops.get(stop_key) is not stop:
                    continue
                if stop.status != StopStatus.TRIGGERED:
                    continue
                # The frozen evidence remains authoritative even if unrelated
                # code touched mutable display fields on the stop object while
                # it waited for the monitor-loop drain.
                stop.trigger_price = evidence.trigger_price
                stop.triggered_at = evidence.event_time
                triggered.append(stop)

            for stop_key, stop in list(self.active_stops.items()):
                if stop.status != StopStatus.PENDING:
                    continue

                # Get current price (last_prices is keyed by bare symbol, not
                # composite key). This fallback covers a stop added after the
                # most recent accepted quote.
                current_price = self.last_prices.get(stop.symbol)
                if not current_price:
                    logger.warning(f"No price data for {stop.symbol}, cannot check stop-loss")
                    continue

                event_time = self.price_event_times.get(stop.symbol)
                receipt_time = self.price_receipt_monotonic.get(stop.symbol)
                if event_time is None or receipt_time is None:
                    logger.warning(f"No timestamp data for {stop.symbol}, cannot check stop-loss")
                    continue

                # Event age protects against stale broker data. Monotonic receipt
                # age stays correct if the host wall clock moves backwards.
                event_age_seconds = (self._utcnow() - event_time).total_seconds()
                receipt_age_seconds = self._monotonic() - receipt_time
                if (
                    event_age_seconds < 0
                    or receipt_age_seconds < 0
                    or event_age_seconds > self.max_price_age_seconds
                    or receipt_age_seconds > self.max_price_age_seconds
                ):
                    logger.warning(
                        "Stale price data for %s (event_age=%.1fs receipt_age=%.1fs)",
                        stop.symbol,
                        event_age_seconds,
                        receipt_age_seconds,
                    )
                    continue

                if self._price_crosses_stop(stop, current_price):
                    logger.warning(
                        "STOP-LOSS TRIGGERED for %s %s: price $%.2f %s stop $%.2f",
                        stop.symbol,
                        "LONG" if stop.position_qty > 0 else "SHORT",
                        current_price,
                        "<=" if stop.position_qty > 0 else ">=",
                        stop.stop_price,
                    )
                    stop.status = StopStatus.TRIGGERED
                    stop.triggered_at = event_time
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
        stop_key = self._stop_key(stop.symbol)
        if self.active_stops.get(stop_key) is not stop or stop.status in {
            StopStatus.CANCELLED,
            StopStatus.EXECUTED,
            StopStatus.FAILED,
        }:
            logger.warning(
                "Refusing obsolete stop execution for %s: status=%s",
                stop.symbol,
                stop.status.value,
            )
            return False

        logger.critical(
            f"EXECUTING STOP-LOSS for {stop.symbol}: "
            f"closing {'LONG' if stop.position_qty > 0 else 'SHORT'} "
            f"{abs(stop.position_qty)} shares"
        )

        # TCN-H5 (followup audit): pass stop.trigger_price as the order price
        # rather than None. The paper executor's market-order path requires a
        # cached reference price in _execution_cache; if the runner just restarted
        # or the symbol has been "held" for >60s, the cache is empty/stale and
        # the order fails with "No reference price for market order", causing the
        # stop to never fire. trigger_price is the price that crossed the stop
        # threshold this cycle, so it's the correct execution reference.
        # IBKR live executor still treats this as the limit price guard around
        # an immediate fill; paper executor uses it directly.
        order = Order(
            symbol=stop.symbol,
            quantity=abs(stop.position_qty),
            side="SELL" if stop.position_qty > 0 else "BUY",
            price=stop.trigger_price if stop.trigger_price is not None else stop.stop_price,
        )

        # Attempt execution with retries.
        #
        # R2-L3 (intentional): stop-loss execution does NOT route through
        # AsyncRunner._trading_blocked(). That's the gate for opening new
        # positions; stop-losses are loss-mitigation exits on positions we
        # already hold, and must execute even when normal trading is disabled
        # (e.g. extended-hours window closed, AI suppressors active, daily
        # notional cap hit). Refusing to close a losing position because new
        # entries are blocked would be the worst possible behavior.
        for attempt in range(self.max_execution_retries):
            # A prior definitive rejection may be followed by cancellation or
            # replacement during retry backoff. Revalidate immediately before
            # every broker call so an obsolete close is never resubmitted.
            if self.active_stops.get(stop_key) is not stop or stop.status in {
                StopStatus.CANCELLED,
                StopStatus.EXECUTED,
                StopStatus.FAILED,
            }:
                logger.warning(
                    "Abandoning obsolete stop retry for %s: status=%s " "attempt=%d",
                    stop.symbol,
                    stop.status.value,
                    attempt + 1,
                )
                return False
            try:
                result = await self.executor.place_order_async(order)
            except Exception as e:
                logger.error(
                    f"Exception during stop-loss execution for {stop.symbol} "
                    f"(attempt {attempt + 1}/{self.max_execution_retries}): {e}"
                )

                if attempt < self.max_execution_retries - 1:
                    await asyncio.sleep(0.5)
                continue

            if not result.ok:
                logger.error(
                    f"Stop-loss execution failed for {stop.symbol} "
                    f"(attempt {attempt + 1}/{self.max_execution_retries}): {result.message}"
                )
                if attempt < self.max_execution_retries - 1:
                    await asyncio.sleep(0.5)  # Brief delay before retry
                continue

            # Broker-fill commit point. Nothing after result.ok may re-enter
            # the retry loop: bookkeeping failures cannot make an already
            # filled closing order safe to submit again.
            try:
                stop.status = StopStatus.EXECUTED
                stop.executed_at = self._utcnow()
                stop.execution_price = result.fill_price
            except Exception as state_err:  # pragma: no cover - defensive
                logger.error(
                    "Post-fill stop state update failed for %s: %r; "
                    "order remains irrevocably filled",
                    stop.symbol,
                    state_err,
                )

            prevented_loss = 0.0
            try:
                if stop.position_qty > 0:  # Long
                    prevented_loss = abs(stop.position_qty * (stop.trigger_price - stop.stop_price))
                else:  # Short
                    prevented_loss = abs(stop.position_qty * (stop.stop_price - stop.trigger_price))
                self.metrics.executed_today += 1
                self.metrics.total_prevented_loss += prevented_loss
                self.metrics.largest_prevented_loss = max(
                    self.metrics.largest_prevented_loss, prevented_loss
                )
            except Exception as metrics_err:
                logger.error(
                    "Post-fill stop metrics failed for %s: %r; " "order remains irrevocably filled",
                    stop.symbol,
                    metrics_err,
                )

            try:
                if self.active_stops.get(stop_key) is stop:
                    del self.active_stops[stop_key]
                elif stop_key in self.active_stops:
                    logger.warning(
                        "Preserving replacement stop after filled prior stop: symbol=%s",
                        stop.symbol,
                    )
                if not any(historical is stop for historical in self.stop_history):
                    self.stop_history.append(stop)
                self.metrics.active_stops = len(self.active_stops)
            except Exception as cleanup_err:
                logger.error(
                    "Post-fill stop cleanup failed for %s: %r; "
                    "preserving current active-stop state",
                    stop.symbol,
                    cleanup_err,
                )

            logger.info(
                "Stop-loss executed successfully for %s: fill_price=%r " "prevented_loss=%.2f",
                stop.symbol,
                result.fill_price,
                prevented_loss,
            )

            # TCN-H4 (followup audit): notify the runner so it can update
            # self.positions, portfolio.update_fill, and persist to DB.
            # This callback is attempted exactly once after the broker fill.
            if self.position_closed_callback is not None:
                try:
                    await self.position_closed_callback(stop, result)
                except Exception as cb_err:  # pragma: no cover - defensive
                    logger.error(
                        f"position_closed_callback failed for {stop.symbol}: "
                        f"{cb_err} — runtime state may diverge from broker. "
                        f"Reconcile via load_existing_positions() on restart."
                    )

            return True

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
        Get active stop-loss order for a symbol in this portfolio.

        Args:
            symbol: Trading symbol

        Returns:
            Stop-loss order if exists
        """
        return self.active_stops.get(self._stop_key(symbol))

    def cancel_stop(self, symbol: str) -> bool:
        """
        Cancel stop-loss order for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if stop was cancelled
        """
        stop_key = self._stop_key(symbol)
        if stop_key in self.active_stops:
            stop = self.active_stops[stop_key]
            self._pending_stop_triggers.pop(stop_key, None)
            stop.status = StopStatus.CANCELLED
            del self.active_stops[stop_key]
            self.stop_history.append(stop)
            self.metrics.active_stops = len(self.active_stops)
            logger.info(f"Stop-loss cancelled for {symbol} (portfolio={self.portfolio_id})")
            return True

        return False

    def cancel_all_stops(self) -> int:
        """
        Cancel all active stop-loss orders.

        Returns:
            Number of stops cancelled
        """
        # TCN-H1 (followup audit): self.active_stops is keyed by _stop_key(symbol),
        # which is "<portfolio_id>:<symbol>". The previous implementation iterated
        # the keys and passed them as `symbol` to cancel_stop(), which then prefixed
        # them AGAIN ("default:default:AAPL") and lookups failed silently. Use the
        # actual `stop.symbol` attribute to recover the bare symbol.
        count = 0
        for stop in list(self.active_stops.values()):
            if self.cancel_stop(stop.symbol):
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
