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
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from robo_trader.database_validator import DatabaseValidator, ValidationError
from robo_trader.execution import ExecutionResult, Order
from robo_trader.logger import get_logger
from robo_trader.paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperTerminalOutcome,
)
from robo_trader.protective_quote_evidence import (
    ProtectiveQuoteEvidence,
    ProtectiveQuoteSource,
    ProtectiveQuoteValidationError,
    _produce_protective_quote,
    assert_producer_owned_protective_quote,
)
from robo_trader.risk_manager import Position

logger = get_logger(__name__)


def _stable_stop_order_ref(portfolio_id: str, stop: "StopLossOrder") -> str:
    """Return a fixed-width, deterministic reference for one protective stop."""

    canonical_identity = json.dumps(
        [
            portfolio_id,
            stop.symbol,
            stop.created_at.astimezone(timezone.utc).isoformat(timespec="microseconds"),
            str(stop.position_qty),
            str(Decimal(str(stop.stop_price))),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"stop:v1:{hashlib.sha256(canonical_identity).hexdigest()}"


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


def _positive_finite_number(value: object, field_name: str) -> float:
    """Return a finite positive numeric value or reject ambiguous coercions."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must not be bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return numeric


def _aware_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def validate_stop_position(
    position: object,
    expected_symbol: str,
) -> Position:
    """Validate the canonical position shape required by a protective stop."""
    if not isinstance(position, Position):
        raise ValueError("position must be a Position")
    try:
        normalized_symbol = DatabaseValidator.validate_symbol(position.symbol)
        normalized_expected = DatabaseValidator.validate_symbol(expected_symbol)
    except ValidationError as exc:
        raise ValueError("position symbol is invalid") from exc
    if (
        position.symbol != normalized_symbol
        or expected_symbol != normalized_expected
        or position.symbol != expected_symbol
    ):
        raise ValueError("position symbol does not match stop symbol")
    if type(position.quantity) is not int or position.quantity == 0:
        raise ValueError("position quantity must be a nonzero canonical int")
    _positive_finite_number(position.avg_price, "position.avg_price")
    return position


def validate_stop_loss_order(
    stop: object,
    *,
    position: Optional[Position] = None,
    now_utc: Optional[datetime] = None,
    allow_authenticated_precreation_trigger: bool = False,
) -> StopLossOrder:
    """Validate structural and economic stop invariants without mutating state."""
    if not isinstance(stop, StopLossOrder):
        raise ValueError("stop must be a StopLossOrder")
    if not isinstance(stop.stop_type, StopType):
        raise ValueError("stop_type must be a StopType")
    if not isinstance(stop.status, StopStatus):
        raise ValueError("status must be a StopStatus")
    if type(stop.position_qty) is not int or stop.position_qty == 0:
        raise ValueError("stop quantity must be a nonzero canonical int")

    _positive_finite_number(stop.entry_price, "entry_price")
    stop_price = _positive_finite_number(stop.stop_price, "stop_price")
    created_at = _aware_utc(stop.created_at, "created_at")
    current_time = _aware_utc(now_utc, "now_utc") if now_utc is not None else None
    if current_time is not None and created_at > current_time + timedelta(seconds=5):
        raise ValueError("created_at is in the future")

    trigger_price = stop.trigger_price
    triggered_at = stop.triggered_at
    requires_trigger = stop.status in {StopStatus.TRIGGERED, StopStatus.EXECUTED}
    has_partial_trigger = (trigger_price is None) != (triggered_at is None)
    if has_partial_trigger:
        raise ValueError("trigger price and timestamp must be recorded together")
    if requires_trigger and trigger_price is None:
        raise ValueError("triggered stop is missing crossing evidence")
    if stop.status is StopStatus.PENDING and trigger_price is not None:
        raise ValueError("pending stop cannot retain crossing evidence")
    if trigger_price is not None:
        _positive_finite_number(trigger_price, "trigger_price")
        trigger_time = _aware_utc(triggered_at, "triggered_at")
        if trigger_time < created_at and not allow_authenticated_precreation_trigger:
            raise ValueError("triggered_at precedes created_at")
        if current_time is not None and trigger_time > current_time + timedelta(seconds=5):
            raise ValueError("triggered_at is in the future")

    if stop.stop_type is StopType.FIXED:
        if (
            stop.trailing_amount is not None
            or stop.trailing_percent is not None
            or stop.high_water_mark is not None
        ):
            raise ValueError("fixed stop cannot contain trailing fields")
    else:
        high_water_mark = _positive_finite_number(
            stop.high_water_mark,
            "high_water_mark",
        )
        if stop.position_qty > 0 and stop_price >= high_water_mark:
            raise ValueError("long trailing stop must remain below high-water mark")
        if stop.position_qty < 0 and stop_price <= high_water_mark:
            raise ValueError("short trailing stop must remain above high-water mark")
        if stop.stop_type is StopType.TRAILING:
            _positive_finite_number(stop.trailing_amount, "trailing_amount")
            if stop.trailing_percent is not None:
                raise ValueError("amount trailing stop cannot contain trailing_percent")
        else:
            if stop.trailing_amount is not None:
                raise ValueError("percent trailing stop cannot contain trailing_amount")
            trailing_percent = _positive_finite_number(
                stop.trailing_percent,
                "trailing_percent",
            )
            if trailing_percent >= 1:
                raise ValueError("trailing_percent must be less than 1")

    if position is not None:
        validate_stop_position(position, stop.symbol)
        # Entry and average cost may legitimately differ after partial fills or
        # replacement, but both must independently remain real positive prices.
        _positive_finite_number(position.avg_price, "position.avg_price")

    # Crossing entry is not itself invalid: a profitable ratcheted stop may
    # protect above/below its original entry. The current accepted quote is
    # authoritative at assertion time.
    return stop


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
    drain_timeout_seconds: float
    drain_deadline_monotonic: float
    quote_evidence: Optional[ProtectiveQuoteEvidence] = field(
        default=None,
        compare=False,
        repr=False,
    )


class StopExecutionPhase(str, Enum):
    """Bounded monitor-owned phases after a stop trigger is latched."""

    QUEUED = "queued"
    BROKER_WAIT = "broker_wait"
    POST_FILL_SETTLEMENT = "post_fill_settlement"


@dataclass(frozen=True)
class StopExecutionPhaseRecord:
    """Immutable exact-object ownership plus a monotonic progress deadline."""

    stop: StopLossOrder = field(compare=False, repr=False)
    phase: StopExecutionPhase
    started_monotonic: float
    timeout_seconds: float
    deadline_monotonic: float


class StopLossMonitor:
    """
    Active stop-loss monitoring and execution system.

    This is a critical safety component that prevents excessive losses by
    automatically executing stop-loss orders when price thresholds are breached.
    """

    def __init__(
        self,
        execute_reduction: Callable[
            ["StopLossOrder", "Order"],
            Awaitable["ExecutionResult"],
        ],
        risk_manager,
        emergency_shutdown_callback=None,
        portfolio_id: str = "default",
        position_closed_callback: Optional[
            Callable[["StopLossOrder", "ExecutionResult"], Awaitable[None]]
        ] = None,
        order_timeout_seconds: float = 30.0,
        pending_drain_timeout_seconds: Optional[float] = None,
        queue_timeout_seconds: Optional[float] = None,
        settlement_timeout_seconds: Optional[float] = None,
    ):
        """
        Initialize stop-loss monitor.

        Args:
            execute_reduction: Narrow async callback for one authorized
                reduce-only order attempt. The monitor never receives a raw
                executor and never retries a submission.
            risk_manager: Risk manager for validation and limits
            emergency_shutdown_callback: Callback for emergency shutdown
            portfolio_id: Portfolio this monitor is scoped to
            position_closed_callback: Optional async callback invoked AFTER a
                stop-loss execution succeeds. Receives (stop_order, result).
                Used by AsyncRunner to sync `runner.positions`,
                `portfolio.update_fill`, and DB persistence so that a phantom
                position does not block subsequent BUY/SELL signals.
                (TCN-H4 followup audit fix.)
            order_timeout_seconds: Configured upper bound for one broker-order
                progress phase.
            pending_drain_timeout_seconds: Optional pending-latch drain bound.
            queue_timeout_seconds: Optional sequential queue progress bound.
            settlement_timeout_seconds: Optional post-fill callback bound.
        """
        if not callable(execute_reduction):
            raise ValueError("execute_reduction must be an async callback")
        self._execute_reduction = execute_reduction
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
        # The exact producer-owned object corresponding to each accepted quote.
        # Legacy callbacks remain explicitly unbound to a broker contract and
        # transport generation; later safety boundaries can therefore reject
        # them without trusting the mutable display dictionaries above.
        self._protective_quote_evidence: Dict[str, ProtectiveQuoteEvidence] = {}
        self._price_receipt_order = 0
        self._pending_stop_triggers: Dict[str, _PendingStopTrigger] = {}
        # Immutable crossing evidence is retained by exact stop identity until
        # tracked execution finishes. This authenticates the narrow case where
        # a newly-added stop legitimately triggers from an already-accepted
        # cached quote whose broker event time predates stop creation.
        self._latched_stop_crossings: Dict[int, _PendingStopTrigger] = {}
        # Entire triggered batches are published here synchronously before the
        # monitor awaits the first broker call. This preserves exact ownership
        # for later stops waiting behind an in-flight stop.
        self._queued_stop_orders: Dict[str, StopExecutionPhaseRecord] = {}
        # Exact-object ownership for stops whose closing order is currently
        # awaiting the broker. Runtime health checks may trust only an active
        # TRIGGERED stop that is this same tracked object.
        self._inflight_stop_orders: Dict[str, StopExecutionPhaseRecord] = {}
        # Serialize quote ordering and trigger draining. No broker I/O occurs
        # while this lock is held.
        self._price_update_lock = asyncio.Lock()

        # Metrics
        self.metrics = StopLossMetrics()
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)

        # Configuration
        self.check_interval_seconds = 1  # Check every second
        self.max_price_age_seconds = 10  # Require fresh prices
        self.max_execution_retries = 1
        self.emergency_shutdown_on_failure = True
        self.broker_attempt_timeout_seconds = self._validate_progress_timeout(
            order_timeout_seconds,
            "order_timeout_seconds",
        )
        self.pending_drain_timeout_seconds = self._validate_progress_timeout(
            (
                pending_drain_timeout_seconds
                if pending_drain_timeout_seconds is not None
                else min(
                    self.broker_attempt_timeout_seconds,
                    max(2.0, self.check_interval_seconds * 3),
                )
            ),
            "pending_drain_timeout_seconds",
        )
        self.queue_timeout_seconds = self._validate_progress_timeout(
            (
                queue_timeout_seconds
                if queue_timeout_seconds is not None
                else (self.broker_attempt_timeout_seconds + self.check_interval_seconds)
            ),
            "queue_timeout_seconds",
        )
        self.settlement_timeout_seconds = self._validate_progress_timeout(
            (
                settlement_timeout_seconds
                if settlement_timeout_seconds is not None
                else self.broker_attempt_timeout_seconds * 2
            ),
            "settlement_timeout_seconds",
        )

        logger.info(f"Stop-loss monitor initialized for portfolio={self.portfolio_id}")

    @staticmethod
    def _validate_progress_timeout(value: object, field_name: str) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{field_name} must be a positive finite number")
        return float(value)

    def _new_phase_record(
        self,
        stop: StopLossOrder,
        phase: StopExecutionPhase,
        timeout_seconds: float,
    ) -> StopExecutionPhaseRecord:
        started = self._monotonic()
        return StopExecutionPhaseRecord(
            stop=stop,
            phase=phase,
            started_monotonic=started,
            timeout_seconds=timeout_seconds,
            deadline_monotonic=started + timeout_seconds,
        )

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
            validate_stop_position(position, symbol)
        except ValidationError as e:
            logger.error(f"Invalid stop-loss parameters: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid position for stop-loss: {e}")
            raise ValidationError(str(e)) from e

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
            created_at=self._utcnow(),
            portfolio_id=self.portfolio_id,
            trailing_amount=trailing_amount,
            trailing_percent=trailing_percent,
        )
        try:
            validate_stop_loss_order(
                stop_order,
                position=position,
                now_utc=self._utcnow(),
            )
        except ValueError as e:
            logger.error(f"Invalid stop-loss structure: {e}")
            raise ValidationError(str(e)) from e

        # Cancel existing stop for this symbol if any
        stop_key = self._stop_key(symbol)
        if stop_key in self.active_stops:
            old_stop = self.active_stops[stop_key]
            old_stop.status = StopStatus.CANCELLED
            self.stop_history.append(old_stop)
            logger.info(
                f"Cancelled existing stop-loss for {symbol} (portfolio={self.portfolio_id})"
            )
            queued = self._queued_stop_orders.get(stop_key)
            inflight = self._inflight_stop_orders.get(stop_key)
            if not (
                (queued is not None and queued.stop is old_stop)
                or (inflight is not None and inflight.stop is old_stop)
            ):
                self._latched_stop_crossings.pop(id(old_stop), None)

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
        source: ProtectiveQuoteSource = ProtectiveQuoteSource.LEGACY_CALLBACK,
        con_id: Optional[int] = None,
        transport_generation: Optional[str] = None,
        source_event_id: Optional[str] = None,
    ) -> bool:
        """
        Update current price for a symbol.

        Args:
            symbol: Trading symbol
            price: Current market price
            source_timestamp: Timezone-aware broker event timestamp. Missing
                event time is rejected rather than replaced by receipt time.
            source: Explicit quote origin. LIVE_BROKER additionally requires
                contract and transport lineage.
            con_id: Qualified broker contract identifier for LIVE_BROKER.
            transport_generation: Current broker transport generation.
            source_event_id: Optional upstream event identifier.
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
            try:
                quote_evidence = _produce_protective_quote(
                    self,
                    portfolio_id=self.portfolio_id,
                    symbol=symbol,
                    price=Decimal(str(price)),
                    source_timestamp=event_time,
                    receipt_monotonic=receipt_monotonic,
                    receipt_order=receipt_order,
                    source=source,
                    con_id=con_id,
                    transport_generation=transport_generation,
                    source_event_id=source_event_id,
                )
            except ProtectiveQuoteValidationError as exc:
                logger.error("Rejected price update for %s: %s", symbol, exc)
                return False
            self.last_prices[symbol] = price
            self.price_event_times[symbol] = event_time
            self.price_receipt_monotonic[symbol] = receipt_monotonic
            self.price_receipt_orders[symbol] = receipt_order
            self._protective_quote_evidence[symbol] = quote_evidence

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
                    evidence = _PendingStopTrigger(
                        stop=stop,
                        trigger_price=price,
                        event_time=event_time,
                        receipt_monotonic=receipt_monotonic,
                        receipt_order=receipt_order,
                        drain_timeout_seconds=self.pending_drain_timeout_seconds,
                        drain_deadline_monotonic=(
                            receipt_monotonic + self.pending_drain_timeout_seconds
                        ),
                        quote_evidence=quote_evidence,
                    )
                    self._pending_stop_triggers[stop_key] = evidence
                    self._latched_stop_crossings[id(stop)] = evidence
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

    def get_protective_quote_evidence(
        self,
        symbol: str,
    ) -> Optional[ProtectiveQuoteEvidence]:
        """Return the latest exact evidence only while this monitor owns it."""

        try:
            normalized_symbol = DatabaseValidator.validate_symbol(symbol)
        except ValidationError:
            return None
        evidence = self._protective_quote_evidence.get(normalized_symbol)
        if evidence is None:
            return None
        try:
            return assert_producer_owned_protective_quote(evidence, producer=self)
        except ProtectiveQuoteValidationError:
            logger.error(
                "Protective quote ownership check failed for %s",
                normalized_symbol,
            )
            return None

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
                    receipt_order = self.price_receipt_orders.get(stop.symbol)
                    if (
                        not isinstance(receipt_order, int)
                        or isinstance(receipt_order, bool)
                        or receipt_order <= 0
                    ):
                        logger.error(
                            "Missing accepted-quote receipt lineage for %s; "
                            "refusing cached stop trigger",
                            stop.symbol,
                        )
                        stop.status = StopStatus.PENDING
                        stop.triggered_at = None
                        stop.trigger_price = None
                        continue
                    evidence = _PendingStopTrigger(
                        stop=stop,
                        trigger_price=current_price,
                        event_time=event_time,
                        receipt_monotonic=receipt_time,
                        receipt_order=receipt_order,
                        drain_timeout_seconds=self.pending_drain_timeout_seconds,
                        drain_deadline_monotonic=(
                            receipt_time + self.pending_drain_timeout_seconds
                        ),
                        quote_evidence=self._protective_quote_evidence.get(stop.symbol),
                    )
                    self._latched_stop_crossings[id(stop)] = evidence
                    triggered.append(stop)
                    self.metrics.triggered_today += 1

            return triggered

    async def execute_stop_loss(self, stop: StopLossOrder) -> bool:
        """Execute a stop and always retire only this object's phase records.

        The implementation keeps BROKER_WAIT visible during each broker await
        and POST_FILL_SETTLEMENT visible through the runner callback. This
        outer boundary then handles direct callers and the monitor wrapper
        identically, without deleting a same-symbol replacement's records.
        """
        stop_key = self._stop_key(stop.symbol)
        queued_record = self._queued_stop_orders.get(stop_key)
        if queued_record is not None and queued_record.stop is stop:
            del self._queued_stop_orders[stop_key]
        try:
            return await self._execute_stop_loss_impl(stop, stop_key)
        finally:
            self._cleanup_stop_execution_tracking(stop, stop_key)

    def _cleanup_stop_execution_tracking(
        self,
        stop: StopLossOrder,
        stop_key: str,
    ) -> None:
        """Remove exact-object execution state while preserving replacements."""
        inflight_record = self._inflight_stop_orders.get(stop_key)
        if inflight_record is not None and inflight_record.stop is stop:
            del self._inflight_stop_orders[stop_key]
        queued_record = self._queued_stop_orders.get(stop_key)
        if queued_record is not None and queued_record.stop is stop:
            del self._queued_stop_orders[stop_key]
        self._latched_stop_crossings.pop(id(stop), None)

    async def _execute_stop_loss_impl(
        self,
        stop: StopLossOrder,
        stop_key: str,
    ) -> bool:
        """
        Execute stop-loss order immediately.

        Args:
            stop: Stop-loss order to execute

        Returns:
            bool: True if execution successful
        """
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

        # The mutable stop price is never submission authority. The runner passes
        # the exact producer-owned quote latched with this crossing, and the
        # gateway alone derives the local-paper reference price from it.
        order = Order(
            symbol=stop.symbol,
            quantity=abs(stop.position_qty),
            side="SELL" if stop.position_qty > 0 else "BUY_TO_COVER",
            price=None,
            order_ref=_stable_stop_order_ref(self.portfolio_id, stop),
        )

        # One authenticated callback is the only execution path. A timeout,
        # exception, malformed result, or definitive rejection is not retried:
        # once an authorization might have crossed the submission boundary,
        # another call could duplicate an exit.
        if self.active_stops.get(stop_key) is not stop or stop.status in {
            StopStatus.CANCELLED,
            StopStatus.EXECUTED,
            StopStatus.FAILED,
        }:
            logger.warning(
                "Refusing obsolete stop submission for %s: status=%s",
                stop.symbol,
                stop.status.value,
            )
            return False
        self._inflight_stop_orders[stop_key] = self._new_phase_record(
            stop,
            StopExecutionPhase.BROKER_WAIT,
            self.broker_attempt_timeout_seconds,
        )
        try:
            result = await self._execute_reduction(stop, order)
        except Exception as exc:
            logger.error(
                "Stop-loss execution raised for %s; refusing retry: %r",
                stop.symbol,
                exc,
            )
            result = ExecutionResult(False, "Stop-loss submission failed")

        if (
            type(result) not in {ExecutionResult, LocalPaperTerminalOutcome}
            or type(result.ok) is not bool
        ):
            logger.error(
                "Stop-loss callback returned malformed result for %s; refusing retry",
                stop.symbol,
            )
            result = ExecutionResult(False, "Malformed stop-loss execution result")
        elif result.ok and (
            isinstance(result.fill_price, bool)
            or not isinstance(result.fill_price, (int, float, Decimal))
            or not math.isfinite(float(result.fill_price))
            or float(result.fill_price) <= 0
        ):
            logger.error(
                "Stop-loss callback returned an invalid fill for %s; refusing retry",
                stop.symbol,
            )
            result = ExecutionResult(False, "Invalid stop-loss fill result")

        if result.ok:

            # Settled local-paper commit point. The gateway has already
            # committed the ledger, verified runtime projection, and released
            # the safety journal. Nothing after result.ok may re-enter the
            # retry loop or submit a second simulated exit.
            self._inflight_stop_orders[stop_key] = self._new_phase_record(
                stop,
                StopExecutionPhase.POST_FILL_SETTLEMENT,
                self.settlement_timeout_seconds,
            )
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

            # Compatibility-only notification. Production PR 2B.3 wiring sets
            # this to None because the gateway owns every authoritative DB and
            # runtime mutation before it returns terminal success.
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

        # A producer-owned terminal no-fill is definitive: the exact authority
        # was consumed and released, so this object must never submit again.
        # Preserve the authenticated TRIGGERED stop in active state for
        # operator/restart recovery instead of converting it to inert FAILED;
        # check_stops intentionally does not redispatch TRIGGERED objects.
        definitive_no_fill = (
            type(result) is LocalPaperTerminalOutcome
            and result.terminal is True
            and result.status
            in {
                LocalPaperOrderStatus.REJECTED,
                LocalPaperOrderStatus.CANCELLED,
                LocalPaperOrderStatus.EXPIRED,
            }
            and result.filled_quantity == 0
        )
        if definitive_no_fill:
            self.metrics.failed_today += 1
            logger.critical(
                "CRITICAL: terminal no-fill retained triggered protection for %s; "
                "operator restart/reconciliation required",
                stop.symbol,
            )
            if self.emergency_shutdown_on_failure and self.emergency_shutdown:
                await self.emergency_shutdown(f"Stop-loss terminal no-fill: {result.status.value}")
            return False

        # Ambiguous or malformed failure after the only authorized attempt.
        stop.status = StopStatus.FAILED
        self.metrics.failed_today += 1

        logger.critical(
            f"CRITICAL: Stop-loss execution FAILED for {stop.symbol}; no retry permitted"
        )

        # Trigger emergency shutdown if configured
        if self.emergency_shutdown_on_failure and self.emergency_shutdown:
            logger.critical("Triggering EMERGENCY SHUTDOWN due to stop-loss execution failure!")
            await self.emergency_shutdown("Stop-loss execution failed")

        return False

    async def _execute_tracked_stop(self, stop: StopLossOrder) -> bool:
        """Execute one stop while publishing exact in-flight ownership."""
        return await self.execute_stop_loss(stop)

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

                # Publish the complete exact-object batch before the first
                # broker await. Later stops remain verifiably monitor-owned
                # while an earlier stop is in flight. Execution remains
                # intentionally sequential in PR1A: these phase deadlines
                # bound starvation and make runtime health fail closed. Safe
                # concurrent stop execution, including same-symbol replacement
                # and transport serialization, is explicitly PR2-owned.
                for stop in triggered:
                    self._queued_stop_orders[self._stop_key(stop.symbol)] = self._new_phase_record(
                        stop,
                        StopExecutionPhase.QUEUED,
                        self.queue_timeout_seconds,
                    )
                try:
                    for stop in triggered:
                        success = await self._execute_tracked_stop(stop)
                        if not success:
                            logger.error(f"Failed to execute stop-loss for {stop.symbol}")
                finally:
                    for stop in triggered:
                        stop_key = self._stop_key(stop.symbol)
                        queued_record = self._queued_stop_orders.get(stop_key)
                        if queued_record is not None and queued_record.stop is stop:
                            del self._queued_stop_orders[stop_key]

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
            self._latched_stop_crossings.pop(id(stop), None)
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
