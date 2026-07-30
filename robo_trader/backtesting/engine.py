"""Deterministic, event-phased portfolio backtesting.

The engine is deliberately offline-only.  A strategy observes a completed bar,
creates a decision, and that decision can first execute on the symbol's next
exact bar.  Accounting consumes the execution simulator's reported filled
quantity and commission once; it never invents fills or substitutes quotes.
The next bar's opening price is executable, but its completed total volume is
not yet known; liquidity capacity therefore uses the last completed bar volume
captured with the decision and advances only for a later retry.

Stop-loss and take-profit predicates receive the adverse and favorable
intrabar extremes respectively.  If both are touched in one completed bar, the
stop wins because OHLC does not reveal path order.  The resulting reduce-only
decision executes no earlier than the next exact bar.  Rebalancing occurs once
per observed daily, weekly, or monthly period, with exposure reductions queued
before increases in deterministic symbol order.
"""

import copy
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV = ("open", "high", "low", "close", "volume")
_FINALIZATION_POLICIES = frozenset({"liquidate", "mark_to_market"})
_ERROR_POLICIES = frozenset({"raise", "record"})


def _decimal(value: Any, name: str) -> Decimal:
    """Convert one finite numeric value without importing binary float noise."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class Position:
    """A copy-safe aggregate position snapshot.

    Quantity is signed: positive is long and negative is short.  Existing
    long-only consumers continue to see positive quantities unchanged.
    """

    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    is_open: bool = True


@dataclass(frozen=True)
class Trade:
    """One realized FIFO lot segment."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    commission: float
    duration_days: int
    trade_type: str  # 'long' or 'short'


@dataclass(frozen=True)
class BacktestResult:
    """Results from one isolated backtest run.

    The original seven fields remain in their original order.  New fields have
    defaults so constructing a result with the legacy signature stays valid.
    Returned pandas objects and collections are detached copies; a later run
    cannot mutate an earlier result.
    """

    equity_curve: pd.Series
    trades: List[Trade]
    positions: List[Position]
    metrics: Dict[str, float]
    daily_returns: pd.Series
    drawdown_series: pd.Series
    signals: pd.DataFrame
    errors: Tuple[str, ...] = field(default_factory=tuple)
    approval_eligible: bool = True
    sampling_periods_per_year: float = 0.0
    finalization_policy: str = "liquidate"


@dataclass
class _Lot:
    symbol: str
    quantity: int
    entry_price: Decimal
    entry_time: pd.Timestamp
    commission_remaining: Decimal


@dataclass
class _PendingOrder:
    symbol: str
    side: str
    remaining_quantity: int
    decision_time: pd.Timestamp
    reason: str
    reduce_only: bool = False
    close_all: bool = False
    requires_flat: bool = False
    commission_quantity: int = 0
    commission_paid: Decimal = Decimal(0)
    known_volume: Optional[float] = None
    known_volatility: float = 0.02
    known_spread: Optional[float] = None


class BacktestEngine:
    """Run deterministic long/short, multi-asset bar backtests.

    This class has no runtime, broker, database, or order authority.  Execution
    is delegated exclusively to the injected offline simulator.
    """

    def __init__(
        self,
        strategy: Any,
        execution_simulator: Any,
        initial_capital: float = 100000,
        commission: float = 0.001,
        min_commission: float = 1.0,
        position_sizer: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        use_fractional_shares: bool = False,
        max_positions: int = 10,
        rebalance_frequency: str = "daily",
        finalization_policy: str = "liquidate",
        error_policy: str = "raise",
    ):
        capital = _decimal(initial_capital, "initial_capital")
        if capital <= 0:
            raise ValueError("initial_capital must be positive")
        if isinstance(max_positions, bool) or not isinstance(max_positions, int):
            raise ValueError("max_positions must be a positive integer")
        if max_positions <= 0:
            raise ValueError("max_positions must be a positive integer")
        if finalization_policy not in _FINALIZATION_POLICIES:
            raise ValueError("finalization_policy must be liquidate or mark_to_market")
        if error_policy not in _ERROR_POLICIES:
            raise ValueError("error_policy must be raise or record")
        if rebalance_frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError("rebalance_frequency must be daily, weekly, or monthly")

        self.strategy = strategy
        self.execution_simulator = execution_simulator
        self.initial_capital = float(capital)
        # Retained for API compatibility.  Simulator-returned commission is the
        # sole commission source used by portfolio accounting.
        commission_value = _decimal(commission, "commission")
        minimum_value = _decimal(min_commission, "min_commission")
        if commission_value < 0 or minimum_value < 0:
            raise ValueError("commission values must be non-negative")
        self.commission = float(commission_value)
        self.min_commission = float(minimum_value)
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.use_fractional_shares = bool(use_fractional_shares)
        self.max_positions = max_positions
        self.rebalance_frequency = rebalance_frequency
        self.finalization_policy = finalization_policy
        self.error_policy = error_policy

        self._sampling_periods_per_year = 0.0
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset every mutable run-owned value."""

        self._cash = _decimal(self.initial_capital, "initial_capital")
        self.cash = float(self._cash)
        self._lots: Dict[str, List[_Lot]] = {}
        self._pending: List[_PendingOrder] = []
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        self.daily_returns: List[float] = []
        self.high_water_mark = self.initial_capital
        self.max_drawdown = 0.0
        self._signals: List[Dict[str, Any]] = []
        self._errors: List[str] = []
        self._approval_eligible = True
        self._last_rebalance_bucket: Optional[Tuple[int, ...]] = None
        self._known_volumes: Dict[str, float] = {}
        self._known_volatilities: Dict[str, float] = {}
        self._known_spreads: Dict[str, Optional[float]] = {}

    def run(self, data: pd.DataFrame, symbols: Optional[List[str]] = None) -> BacktestResult:
        """Run one isolated backtest.

        Input is either one OHLCV DataFrame indexed by timestamp, or a
        two-level MultiIndex frame containing one symbol and one timestamp
        level.  All input is validated before the strategy is initialized.
        """

        self._reset_state()
        frames, selected_symbols, timeline = self._normalize_data(data, symbols)
        self._sampling_periods_per_year = self._infer_periods_per_year(timeline)
        reset = getattr(self.execution_simulator, "reset", None)
        if callable(reset):
            reset()
        self.strategy.initialize(symbols=list(selected_symbols))

        last_market: Optional[pd.DataFrame] = None
        last_timestamp: Optional[pd.Timestamp] = None
        aborted = False
        for timestamp in timeline:
            try:
                current = self._market_slice(frames, timestamp)
                self._apply_corporate_actions(current, timestamp)
                self._execute_pending(current, timestamp)
                self._sync_positions()
                self._mark(current, timestamp)
                self._known_volumes.update(
                    {symbol: float(current.loc[symbol, "volume"]) for symbol in current.index}
                )
                self._known_volatilities.update(
                    {
                        symbol: float(current.loc[symbol].get("volatility", 0.02))
                        for symbol in current.index
                    }
                )
                self._known_spreads.update(
                    {symbol: self._observed_spread(current.loc[symbol]) for symbol in current.index}
                )

                if self._should_rebalance(timestamp):
                    self._queue_rebalance(current, timestamp)

                signals = self.strategy.generate_signals(
                    current.copy(deep=True), copy.deepcopy(self.positions)
                )
                if signals:
                    self._signals.append(
                        {"timestamp": timestamp, "signals": copy.deepcopy(signals)}
                    )
                    self._process_signals(signals, current, timestamp)

                self._update_positions(current, timestamp)
                if self.risk_manager:
                    self._apply_risk_management(current, timestamp)
                last_market = current
                last_timestamp = timestamp
            except Exception as exc:
                if self.error_policy == "raise":
                    raise
                message = f"{timestamp.isoformat()}: {type(exc).__name__}: {exc}"
                logger.error("Backtest aborted: %s", message)
                self._errors.append(message)
                self._approval_eligible = False
                aborted = True
                break

        # A recorded-error run preserves evidence but never continues trading
        # or becomes approval eligible.
        if not aborted and last_market is not None and last_timestamp is not None:
            if self.finalization_policy == "liquidate":
                self._pending.clear()
                self._close_all_positions(last_market, last_timestamp)
                self._sync_positions()
                if self.positions:
                    self._approval_eligible = False
                    self._errors.append("forced liquidation left an unfilled residual position")
            self._replace_final_mark(last_market, last_timestamp)

        return self._create_results(self._signals)

    @classmethod
    def _normalize_data(
        cls, data: pd.DataFrame, symbols: Optional[List[str]]
    ) -> Tuple[Dict[str, pd.DataFrame], Tuple[str, ...], pd.DatetimeIndex]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("data must not be empty")
        if data.index.has_duplicates:
            raise ValueError("data index must contain unique events")
        if not data.index.is_monotonic_increasing:
            raise ValueError("data index must be sorted in increasing order")
        cls._validate_requested_symbols(symbols)

        frames: Dict[str, pd.DataFrame]
        if isinstance(data.index, pd.MultiIndex):
            if data.index.nlevels != 2:
                raise ValueError("multi-asset data must use exactly two index levels")
            datetime_levels = [
                level
                for level in range(2)
                if isinstance(data.index.levels[level], pd.DatetimeIndex)
            ]
            if len(datetime_levels) != 1:
                raise TypeError("multi-asset data requires exactly one DatetimeIndex level")
            time_level = datetime_levels[0]
            symbol_level = 1 - time_level
            available = tuple(
                dict.fromkeys(str(value) for value in data.index.get_level_values(symbol_level))
            )
            selected = tuple(symbols) if symbols is not None else available
            unknown = sorted(set(selected) - set(available))
            if unknown:
                raise ValueError(f"unknown symbols requested: {', '.join(unknown)}")
            frames = {}
            for symbol in selected:
                mask = data.index.get_level_values(symbol_level).astype(str) == symbol
                frame = data.loc[mask].copy(deep=True)
                frame.index = pd.DatetimeIndex(frame.index.get_level_values(time_level))
                cls._validate_frame(frame, symbol)
                frames[symbol] = frame
        else:
            if not isinstance(data.index, pd.DatetimeIndex):
                raise TypeError("single-asset data must use a DatetimeIndex")
            if symbols is not None and len(symbols) != 1:
                raise ValueError("single-asset data accepts exactly one symbol")
            symbol = symbols[0] if symbols else "SINGLE"
            selected = (symbol,)
            frame = data.copy(deep=True)
            cls._validate_frame(frame, symbol)
            frames = {symbol: frame}

        if not frames:
            raise ValueError("at least one symbol must be selected")
        timeline = pd.DatetimeIndex(
            sorted(set().union(*(frame.index.tolist() for frame in frames.values())))
        )
        if timeline.empty:
            raise ValueError("data must contain at least one event")
        return frames, selected, timeline

    @staticmethod
    def _validate_requested_symbols(symbols: Optional[List[str]]) -> None:
        if symbols is None:
            return
        if not isinstance(symbols, list) or not symbols:
            raise ValueError("symbols must be a non-empty list when provided")
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
            raise ValueError("symbols must contain non-empty strings")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, symbol: str) -> None:
        if frame.empty:
            raise ValueError(f"{symbol} has no market events")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{symbol} must use a DatetimeIndex")
        if frame.index.hasnans or frame.index.has_duplicates:
            raise ValueError(f"{symbol} timestamps must be unique and valid")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{symbol} timestamps must be sorted")
        missing = [column for column in _OHLCV if column not in frame.columns]
        if missing:
            raise ValueError(f"{symbol} missing OHLCV columns: {', '.join(missing)}")
        numeric: Dict[str, np.ndarray] = {}
        for column in _OHLCV:
            try:
                values = frame[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{symbol} {column} must be numeric") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"{symbol} {column} must contain only finite values")
            numeric[column] = values
        for column in ("open", "high", "low", "close"):
            if (numeric[column] <= 0).any():
                raise ValueError(f"{symbol} {column} must be positive")
        if (numeric["volume"] < 0).any():
            raise ValueError(f"{symbol} volume must be non-negative")
        if (
            numeric["high"] < np.maximum.reduce([numeric["open"], numeric["low"], numeric["close"]])
        ).any():
            raise ValueError(f"{symbol} high violates OHLC bounds")
        if (
            numeric["low"] > np.minimum.reduce([numeric["open"], numeric["high"], numeric["close"]])
        ).any():
            raise ValueError(f"{symbol} low violates OHLC bounds")
        for optional in ("dividend", "split", "borrow_rate", "volatility"):
            if optional in frame:
                values = frame[optional].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"{symbol} {optional} must contain only finite values")
                minimum = 0 if optional != "split" else np.nextafter(0.0, 1.0)
                if (values < minimum).any():
                    raise ValueError(f"{symbol} {optional} contains an invalid value")
        has_bid = "bid" in frame
        has_ask = "ask" in frame
        if has_bid != has_ask:
            raise ValueError(f"{symbol} must provide bid and ask together")
        if has_bid:
            bids = frame["bid"].to_numpy(dtype=float)
            asks = frame["ask"].to_numpy(dtype=float)
            if (
                not np.isfinite(bids).all()
                or not np.isfinite(asks).all()
                or (bids <= 0).any()
                or (asks < bids).any()
            ):
                raise ValueError(f"{symbol} requires finite 0 < bid <= ask")

    @staticmethod
    def _observed_spread(market_data: pd.Series) -> Optional[float]:
        if "bid" not in market_data or "ask" not in market_data:
            return None
        return float(market_data["ask"] - market_data["bid"])

    def _refresh_pending_cost_inputs(self, pending: _PendingOrder, market_data: pd.Series) -> None:
        pending.known_volume = float(market_data["volume"])
        pending.known_volatility = float(market_data.get("volatility", 0.02))
        pending.known_spread = self._observed_spread(market_data)

    @staticmethod
    def _market_slice(frames: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> pd.DataFrame:
        rows = {
            symbol: frame.loc[timestamp].copy(deep=True)
            for symbol, frame in frames.items()
            if timestamp in frame.index
        }
        current = pd.DataFrame.from_dict(rows, orient="index")
        current.index.name = "symbol"
        current.attrs["timestamp"] = timestamp
        return current

    def _execute_pending(self, current: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        event_capacity: Dict[str, int] = {}
        if getattr(self.execution_simulator, "supports_shared_event_capacity", False):
            eligible = [
                pending
                for pending in self._pending
                if timestamp > pending.decision_time and pending.symbol in current.index
            ]
            for pending in eligible:
                volume = (
                    pending.known_volume
                    if pending.known_volume is not None
                    else float(current.loc[pending.symbol, "volume"])
                )
                capacity = self.execution_simulator.calculate_fill_capacity(volume)
                event_capacity[pending.symbol] = min(
                    event_capacity.get(pending.symbol, capacity), capacity
                )

        still_pending: List[_PendingOrder] = []
        ordered_pending = sorted(self._pending, key=lambda pending: not pending.reduce_only)
        for pending in ordered_pending:
            if timestamp <= pending.decision_time or pending.symbol not in current.index:
                still_pending.append(pending)
                continue
            if pending.requires_flat and pending.symbol in self.positions:
                self._refresh_pending_cost_inputs(pending, current.loc[pending.symbol])
                still_pending.append(pending)
                continue
            if pending.symbol not in self.positions and len(self.positions) >= self.max_positions:
                # State may have changed since this order was reserved.  Never
                # let a stale pending entry exceed the configured position cap.
                continue
            if pending.reduce_only:
                position = self.positions.get(pending.symbol)
                reduces_long = (
                    position is not None and position.quantity > 0 and pending.side == "sell"
                )
                reduces_short = (
                    position is not None and position.quantity < 0 and pending.side == "buy"
                )
                if not (reduces_long or reduces_short):
                    continue
                if position is None:
                    raise RuntimeError("reduce-only validation lost its position")
                if pending.close_all:
                    pending.remaining_quantity = abs(position.quantity)
                else:
                    pending.remaining_quantity = min(
                        pending.remaining_quantity, abs(position.quantity)
                    )
            requested = pending.remaining_quantity
            filled = self._simulate_and_account(
                pending.symbol,
                requested,
                pending.side,
                current.loc[pending.symbol],
                timestamp,
                parent_order=pending,
                max_fill_quantity=event_capacity.get(pending.symbol),
            )
            if pending.symbol in event_capacity:
                event_capacity[pending.symbol] -= filled
            pending.remaining_quantity -= filled
            if pending.remaining_quantity > 0:
                self._refresh_pending_cost_inputs(pending, current.loc[pending.symbol])
                still_pending.append(pending)
        self._pending = still_pending

    def _simulate_and_account(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: pd.Series,
        timestamp: pd.Timestamp,
        *,
        parent_order: Optional[_PendingOrder] = None,
        max_fill_quantity: Optional[int] = None,
    ) -> int:
        price_data = pd.DataFrame([market_data], index=pd.DatetimeIndex([timestamp]))
        position_before = self.positions.get(symbol)
        long_quantity_before = max(position_before.quantity, 0) if position_before else 0
        opens_short = side == "sell" and quantity > long_quantity_before
        short_margin_available = self._cash - sum(
            Decimal(2) * lot.entry_price * abs(lot.quantity)
            for lots in self._lots.values()
            for lot in lots
            if lot.quantity < 0
        )
        simulation_options: Dict[str, Any] = {}
        if parent_order is not None and getattr(
            self.execution_simulator, "supports_parent_order_commission", False
        ):
            simulation_options = {
                "cumulative_filled_quantity": parent_order.commission_quantity,
                "commission_paid": float(parent_order.commission_paid),
            }
        if (
            parent_order is not None
            and parent_order.known_volume is not None
            and getattr(self.execution_simulator, "supports_lagged_liquidity", False)
        ):
            simulation_options["liquidity_volume"] = parent_order.known_volume
        if parent_order is not None and getattr(
            self.execution_simulator, "supports_lagged_cost_inputs", False
        ):
            simulation_options.update(
                {
                    "liquidity_volatility": parent_order.known_volatility,
                    "liquidity_spread": parent_order.known_spread,
                    "ignore_reported_cost_inputs": True,
                }
            )
        if max_fill_quantity is not None and getattr(
            self.execution_simulator, "supports_shared_event_capacity", False
        ):
            simulation_options["max_fill_quantity"] = max_fill_quantity
        if (side == "buy" or opens_short) and getattr(
            self.execution_simulator, "supports_cash_budget", False
        ):
            simulation_options["cash_available"] = float(
                self._cash if side == "buy" else max(short_margin_available, Decimal(0))
            )
            if opens_short:
                simulation_options["reserve_short_margin"] = True
        order = self.execution_simulator.simulate_execution(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type="market",
            price_data=price_data,
            timestamp=timestamp,
            **simulation_options,
        )
        if not hasattr(order, "filled_quantity"):
            raise ValueError("execution result must report filled_quantity")
        filled = order.filled_quantity
        if isinstance(filled, bool) or not isinstance(filled, (int, np.integer)):
            raise ValueError("filled_quantity must be an integer")
        filled = int(filled)
        if filled < 0 or filled > quantity:
            raise ValueError("filled_quantity is outside the requested quantity")
        if bool(order.filled) != (filled > 0):
            raise ValueError("execution filled flag disagrees with filled_quantity")
        if hasattr(order, "remaining_quantity") and order.remaining_quantity != quantity - filled:
            raise ValueError("execution remaining_quantity is inconsistent")
        if filled == 0:
            return 0
        fill_price = _decimal(order.fill_price, "fill_price")
        commission = _decimal(order.execution_cost.commission, "commission")
        if fill_price <= 0 or commission < 0:
            raise ValueError("execution returned invalid price or commission")
        opened_short_quantity = max(filled - long_quantity_before, 0) if side == "sell" else 0
        if opened_short_quantity:
            margin_price = max(fill_price, _decimal(market_data["open"], "open price"))
            required_margin = margin_price * opened_short_quantity + commission
            if short_margin_available < required_margin:
                raise RuntimeError("execution fill exceeds available unlevered short margin")
        if side == "buy":
            cash_delta = -(fill_price * filled + commission)
            if self._cash + cash_delta < 0:
                raise RuntimeError("execution fill exceeds available cash")
        else:
            cash_delta = fill_price * filled - commission
        self._cash += cash_delta
        self.cash = float(self._cash)
        self._apply_fill_to_lots(symbol, side, filled, fill_price, commission, timestamp)
        if parent_order is not None:
            parent_order.commission_quantity += filled
            parent_order.commission_paid += commission
        self._sync_positions()
        return filled

    def _apply_fill_to_lots(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: Decimal,
        commission: Decimal,
        timestamp: pd.Timestamp,
    ) -> None:
        direction = 1 if side == "buy" else -1
        remaining = quantity
        lots = self._lots.setdefault(symbol, [])
        commission_per_share = commission / quantity
        while remaining > 0 and lots and (lots[0].quantity > 0) != (direction > 0):
            lot = lots[0]
            lot_quantity = abs(lot.quantity)
            closed = min(remaining, lot_quantity)
            entry_commission = lot.commission_remaining * Decimal(closed) / lot_quantity
            exit_commission = commission_per_share * closed
            if lot.quantity > 0:
                gross = (fill_price - lot.entry_price) * closed
                trade_type = "long"
            else:
                gross = (lot.entry_price - fill_price) * closed
                trade_type = "short"
            realized = gross - entry_commission - exit_commission
            basis = lot.entry_price * closed + entry_commission
            pnl_percent = realized / basis * 100 if basis else Decimal(0)
            self.trades.append(
                Trade(
                    symbol=symbol,
                    entry_time=lot.entry_time,
                    exit_time=timestamp,
                    entry_price=float(lot.entry_price),
                    exit_price=float(fill_price),
                    quantity=closed,
                    pnl=float(realized),
                    pnl_percent=float(pnl_percent),
                    commission=float(entry_commission + exit_commission),
                    duration_days=max(0, (timestamp - lot.entry_time).days),
                    trade_type=trade_type,
                )
            )
            lot.commission_remaining -= entry_commission
            lot.quantity += direction * closed
            remaining -= closed
            if lot.quantity == 0:
                lots.pop(0)
        if remaining:
            lots.append(
                _Lot(
                    symbol=symbol,
                    quantity=direction * remaining,
                    entry_price=fill_price,
                    entry_time=timestamp,
                    commission_remaining=commission_per_share * remaining,
                )
            )
        if not lots:
            self._lots.pop(symbol, None)

    def _sync_positions(self) -> None:
        positions: Dict[str, Position] = {}
        for symbol, lots in self._lots.items():
            quantity = sum(lot.quantity for lot in lots)
            if quantity == 0:
                continue
            absolute = sum(abs(lot.quantity) for lot in lots)
            entry = sum(lot.entry_price * abs(lot.quantity) for lot in lots) / absolute
            positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=float(entry),
                entry_time=min(lot.entry_time for lot in lots),
            )
        self.positions = positions

    def _mark(self, current: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        value = self._portfolio_value(current)
        numeric = float(value)
        previous = self.equity_curve[-1] if self.equity_curve else None
        self.equity_curve.append(numeric)
        self.timestamps.append(timestamp)
        if previous is not None:
            self.daily_returns.append(numeric / previous - 1.0 if previous else 0.0)
        self.high_water_mark = max(self.high_water_mark, numeric)
        if self.high_water_mark > 0:
            self.max_drawdown = max(
                self.max_drawdown, (self.high_water_mark - numeric) / self.high_water_mark
            )

    def _replace_final_mark(self, current: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        value = float(self._portfolio_value(current))
        if not self.equity_curve:
            self._mark(current, timestamp)
            return
        prior = self.equity_curve[-2] if len(self.equity_curve) > 1 else None
        self.equity_curve[-1] = value
        if self.daily_returns:
            self.daily_returns[-1] = value / prior - 1.0 if prior else 0.0
        self.high_water_mark = max(self.equity_curve)
        drawdowns = [
            (max(self.equity_curve[: index + 1]) - equity) / max(self.equity_curve[: index + 1])
            for index, equity in enumerate(self.equity_curve)
            if max(self.equity_curve[: index + 1]) > 0
        ]
        self.max_drawdown = max(drawdowns, default=0.0)

    def _portfolio_value(self, current: pd.DataFrame) -> Decimal:
        value = self._cash
        for symbol, position in self.positions.items():
            if symbol not in current.index:
                raise RuntimeError(f"missing exact quote for held position {symbol}")
            price = _decimal(current.loc[symbol, "close"], f"{symbol} close")
            value += price * position.quantity
        return value

    def _calculate_portfolio_value(self, current_data: pd.DataFrame) -> float:
        """Compatibility wrapper for callers of the legacy helper."""

        return float(self._portfolio_value(current_data))

    def _process_signals(
        self, signals: Dict[str, Any], current_data: pd.DataFrame, timestamp: datetime
    ) -> None:
        if not isinstance(signals, dict):
            raise ValueError("strategy signals must be a symbol mapping")
        decision_time = pd.Timestamp(timestamp)
        for symbol, signal in signals.items():
            if symbol not in current_data.index:
                raise ValueError(f"signal references unavailable symbol {symbol}")
            if not isinstance(signal, dict):
                raise ValueError(f"signal for {symbol} must be a mapping")
            action = str(signal.get("action", "")).lower()
            position = self.positions.get(symbol)
            if action == "close":
                if position:
                    side = "sell" if position.quantity > 0 else "buy"
                    self._queue_order(
                        symbol,
                        side,
                        abs(position.quantity),
                        decision_time,
                        "close",
                        reduce_only=True,
                    )
                continue
            if action in {"buy", "cover"}:
                side = "buy"
            elif action in {"sell", "short"}:
                side = "sell"
            elif action in {"hold", "none"}:
                continue
            else:
                raise ValueError(f"unsupported action {action!r} for {symbol}")

            explicit = signal.get("quantity")
            if action == "sell" and (position is None or position.quantity <= 0):
                continue
            if action == "cover" and (position is None or position.quantity >= 0):
                continue
            if explicit is None and action == "sell" and position and position.quantity > 0:
                quantity = position.quantity
            elif explicit is None and action == "cover" and position and position.quantity < 0:
                quantity = abs(position.quantity)
            else:
                quantity = (
                    self._validate_quantity(explicit, f"{symbol} signal quantity")
                    if explicit is not None
                    else self._calculate_position_size(symbol, current_data.loc[symbol], signal)
                )
            if quantity > 0:
                self._queue_order(
                    symbol,
                    side,
                    quantity,
                    decision_time,
                    action,
                    reduce_only=action in {"sell", "cover"},
                )

    @staticmethod
    def _validate_quantity(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        return int(numeric)

    def _queue_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        timestamp: pd.Timestamp,
        reason: str,
        *,
        reduce_only: bool = False,
        close_all: bool = False,
        requires_flat: bool = False,
    ) -> None:
        position = self.positions.get(symbol)
        if reduce_only and position:
            already_pending = sum(
                pending.remaining_quantity
                for pending in self._pending
                if pending.symbol == symbol and pending.side == side
            )
            quantity = min(quantity, max(abs(position.quantity) - already_pending, 0))
            if quantity == 0:
                return
        opens_new = position is None
        if opens_new:
            reserved = {
                pending.symbol for pending in self._pending if pending.symbol not in self.positions
            }
            if symbol not in reserved and len(self.positions) + len(reserved) >= self.max_positions:
                return
        self._pending.append(
            _PendingOrder(
                symbol,
                side,
                quantity,
                pd.Timestamp(timestamp),
                reason,
                reduce_only=reduce_only,
                close_all=close_all,
                requires_flat=requires_flat,
                known_volume=self._known_volumes.get(symbol),
                known_volatility=self._known_volatilities.get(symbol, 0.02),
                known_spread=self._known_spreads.get(symbol),
            )
        )

    def _cancel_pending_increases(self, symbol: str) -> None:
        self._pending = [
            pending for pending in self._pending if pending.symbol != symbol or pending.reduce_only
        ]

    def _queue_authoritative_close(
        self,
        symbol: str,
        side: str,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> None:
        position = self.positions.get(symbol)
        if position is None:
            return
        # One risk close supersedes all stale intent for the symbol.  It is
        # resized against the actual position immediately before each fill.
        self._pending = [pending for pending in self._pending if pending.symbol != symbol]
        self._queue_order(
            symbol,
            side,
            abs(position.quantity),
            timestamp,
            reason,
            reduce_only=True,
            close_all=True,
        )

    def _calculate_position_size(
        self, symbol: str, market_data: pd.Series, signal: Dict[str, Any]
    ) -> int:
        close = _decimal(market_data["close"], f"{symbol} close")
        if close <= 0:
            return 0
        if self.position_sizer:
            size = self.position_sizer.calculate_size(
                symbol,
                market_data.copy(deep=True),
                copy.deepcopy(signal),
                float(self._cash),
                copy.deepcopy(self.positions),
            )
            return self._validate_quantity(size, "position size")
        open_slots = max(1, self.max_positions - len(self.positions))
        return max(0, int((self._cash / open_slots) / close))

    def _should_rebalance(self, timestamp: datetime) -> bool:
        """Return true only for the first observed event in each configured period.

        Period transitions are based on observed events rather than weekday or
        month-day literals, so a holiday cannot suppress an entire weekly or
        monthly rebalance and intraday bars cannot schedule duplicates.
        """

        event = pd.Timestamp(timestamp)
        bucket: Tuple[int, ...]
        if self.rebalance_frequency == "daily":
            bucket = (event.year, event.month, event.day)
        elif self.rebalance_frequency == "weekly":
            iso = event.isocalendar()
            bucket = (int(iso.year), int(iso.week))
        else:
            bucket = (event.year, event.month)
        if bucket == self._last_rebalance_bucket:
            return False
        self._last_rebalance_bucket = bucket
        return True

    def _queue_rebalance(self, current: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        if not hasattr(self.strategy, "get_target_weights"):
            return
        target_weights = self.strategy.get_target_weights(
            current.copy(deep=True), copy.deepcopy(self.positions)
        )
        if target_weights is None:
            return
        self._queue_target_weights(target_weights, current, timestamp)

    def _queue_target_weights(
        self,
        target_weights: Dict[str, float],
        current: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> None:
        if not isinstance(target_weights, dict):
            raise ValueError("target weights must be a symbol mapping")
        if self._pending:
            raise RuntimeError("target weights cannot be applied while orders are pending")
        if not target_weights:
            target_weights = {symbol: 0.0 for symbol in self.positions}
        omitted_positions = set(self.positions).difference(target_weights)
        if omitted_positions:
            omitted = ", ".join(sorted(omitted_positions))
            raise ValueError(f"target weights must include every held symbol; missing: {omitted}")
        portfolio_value = self._portfolio_value(current)
        orders: List[Tuple[int, str, str, int, bool, bool]] = []
        normalized_weights: Dict[str, Decimal] = {}
        for symbol, weight_value in target_weights.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("target weight symbols must be non-empty strings")
            if symbol not in current.index:
                raise ValueError(f"target weight references unavailable symbol {symbol}")
            weight = _decimal(weight_value, f"{symbol} target weight")
            if not Decimal("-1") <= weight <= Decimal("1"):
                raise ValueError("target weights must be between -1 and 1")
            normalized_weights[symbol] = weight
        if sum(abs(weight) for weight in normalized_weights.values()) > Decimal(1):
            raise ValueError("absolute target weights must not exceed unlevered gross exposure")

        for symbol, weight in normalized_weights.items():
            price = _decimal(current.loc[symbol, "close"], f"{symbol} close")
            target = int((portfolio_value * weight) / price)
            current_quantity = self.positions.get(
                symbol, Position(symbol, 0, 0.0, timestamp)
            ).quantity
            if current_quantity and target and (current_quantity > 0) != (target > 0):
                close_side = "sell" if current_quantity > 0 else "buy"
                open_side = "buy" if target > 0 else "sell"
                orders.append((0, symbol, close_side, abs(current_quantity), True, False))
                orders.append((1, symbol, open_side, abs(target), False, True))
                continue
            delta = target - current_quantity
            if not delta:
                continue
            side = "buy" if delta > 0 else "sell"
            reduces_exposure = abs(target) < abs(current_quantity)
            # Exposure reductions run before increases so sale proceeds are
            # available to rotations.  Symbol ordering makes equivalent
            # mappings execution-identical regardless of insertion order.
            priority = 0 if reduces_exposure else 1
            orders.append((priority, symbol, side, abs(delta), reduces_exposure, False))
        for _priority, symbol, side, quantity, reduce_only, requires_flat in sorted(orders):
            self._queue_order(
                symbol,
                side,
                quantity,
                timestamp,
                "rebalance-reduce" if reduce_only else "rebalance-increase",
                reduce_only=reduce_only,
                requires_flat=requires_flat,
            )

    def _update_positions(self, current: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        for symbol, position in list(self.positions.items()):
            if symbol not in current.index:
                raise RuntimeError(f"missing exact quote for held position {symbol}")
            row = current.loc[symbol]
            adverse_price = float(row["low"] if position.quantity > 0 else row["high"])
            favorable_price = float(row["high"] if position.quantity > 0 else row["low"])
            stop_triggered = False
            take_profit_triggered = False
            if hasattr(self.strategy, "check_stop_loss"):
                stop_triggered = bool(self.strategy.check_stop_loss(position, adverse_price))
            if hasattr(self.strategy, "check_take_profit"):
                take_profit_triggered = bool(
                    self.strategy.check_take_profit(position, favorable_price)
                )
            if stop_triggered or take_profit_triggered:
                side = "sell" if position.quantity > 0 else "buy"
                # A bar does not reveal whether high or low occurred first.
                # When both thresholds were touched, the adverse stop wins.
                # The completed bar creates a decision; the reduce-only market
                # order can execute only on the symbol's next exact bar.
                reason = "risk-stop" if stop_triggered else "risk-take-profit"
                self._queue_authoritative_close(symbol, side, timestamp, reason)

    def _apply_risk_management(self, current: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        risk_manager = self.risk_manager
        if risk_manager is None:
            return
        value = float(self._portfolio_value(current))
        drawdown = (
            (self.high_water_mark - value) / self.high_water_mark if self.high_water_mark else 0
        )
        actions = risk_manager.check_risk(
            portfolio_value=value,
            positions=copy.deepcopy(self.positions),
            drawdown=drawdown,
            current_data=current.copy(deep=True),
        )
        for action in actions or []:
            action_type = action.get("type")
            if action_type == "close_all":
                for symbol, position in self.positions.items():
                    side = "sell" if position.quantity > 0 else "buy"
                    self._queue_authoritative_close(symbol, side, timestamp, "risk-close")
            elif action_type == "reduce_position":
                symbol = action["symbol"]
                if symbol not in self.positions:
                    continue
                reduction = float(action["reduction"])
                if not math.isfinite(reduction) or not 0 < reduction <= 1:
                    raise ValueError("risk reduction must be in (0, 1]")
                position = self.positions[symbol]
                quantity = max(1, int(abs(position.quantity) * reduction))
                side = "sell" if position.quantity > 0 else "buy"
                self._cancel_pending_increases(symbol)
                self._queue_order(
                    symbol,
                    side,
                    quantity,
                    timestamp,
                    "risk-reduce",
                    reduce_only=True,
                )
            else:
                raise ValueError(f"unsupported risk action {action_type!r}")

    def _close_all_positions(self, current: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        for symbol, position in list(self.positions.items()):
            if symbol not in current.index:
                raise RuntimeError(f"missing exact quote for held position {symbol}")
            side = "sell" if position.quantity > 0 else "buy"
            closing_event = current.loc[symbol].drop(labels=["bid", "ask"], errors="ignore").copy()
            # Finalization happens after the final completed bar was observed.
            # Reusing the bar's opening touch would travel backward in time, so
            # the synthetic terminal event executes at the known closing touch.
            closing_event["open"] = closing_event["close"]
            self._simulate_and_account(
                symbol, abs(position.quantity), side, closing_event, timestamp
            )

    def _execute_buy(
        self, symbol: str, quantity: float, market_data: pd.Series, timestamp: datetime
    ) -> None:
        """Compatibility helper for immediate explicit test execution."""

        self._simulate_and_account(
            symbol,
            self._validate_quantity(quantity, "quantity"),
            "buy",
            market_data,
            pd.Timestamp(timestamp),
        )

    def _execute_sell(
        self, symbol: str, quantity: float, market_data: pd.Series, timestamp: datetime
    ) -> None:
        """Compatibility helper for immediate explicit test execution."""

        self._simulate_and_account(
            symbol,
            self._validate_quantity(quantity, "quantity"),
            "sell",
            market_data,
            pd.Timestamp(timestamp),
        )

    def _close_position(self, symbol: str, market_data: pd.Series, timestamp: datetime) -> None:
        position = self.positions.get(symbol)
        if position:
            self._simulate_and_account(
                symbol,
                abs(position.quantity),
                "sell" if position.quantity > 0 else "buy",
                market_data,
                pd.Timestamp(timestamp),
            )

    def _execute_rebalance(
        self, target_weights: Dict[str, float], current_data: pd.DataFrame
    ) -> None:
        timestamp = current_data.attrs.get("timestamp")
        if timestamp is None:
            raise ValueError("current_data requires a timestamp attribute")
        # Compatibility entry point now queues; it never allows same-bar fill.
        self._queue_target_weights(target_weights, current_data, pd.Timestamp(timestamp))

    def _rebalance_portfolio(self, current_data: pd.DataFrame) -> None:
        timestamp = current_data.attrs.get("timestamp")
        if timestamp is None:
            raise ValueError("current_data requires a timestamp attribute")
        self._queue_rebalance(current_data, pd.Timestamp(timestamp))

    def _get_current_data(
        self, data: pd.DataFrame, timestamp: datetime, symbols: List[str]
    ) -> pd.DataFrame:
        frames, _, _ = self._normalize_data(data, symbols)
        return self._market_slice(frames, pd.Timestamp(timestamp))

    def _apply_corporate_actions(self, current: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        """Apply explicit dividends, splits, and short borrow fees.

        Corporate-action columns are optional.  Splits must preserve integral
        share quantities because the execution simulator only accepts integral
        orders.  Delistings are intentionally not guessed: absent quotes for a
        held symbol fail closed instead of inventing a recovery value.
        """

        for symbol in current.index:
            row = current.loc[symbol]
            split = _decimal(row.get("split", 1), f"{symbol} split")
            if split <= 0:
                raise ValueError("split must be positive")
            if split != 1:
                for lot in self._lots.get(symbol, []):
                    adjusted = Decimal(lot.quantity) * split
                    if adjusted != adjusted.to_integral_value():
                        raise RuntimeError("split creates unsupported fractional shares")
                    lot.quantity = int(adjusted)
                    lot.entry_price /= split
                for pending in self._pending:
                    if pending.symbol != symbol:
                        continue
                    adjusted = Decimal(pending.remaining_quantity) * split
                    if adjusted != adjusted.to_integral_value():
                        raise RuntimeError("split creates an unsupported fractional order")
                    pending.remaining_quantity = int(adjusted)
                    if pending.known_volume is not None:
                        pending.known_volume *= float(split)
                    if pending.known_spread is not None:
                        pending.known_spread /= float(split)
            dividend = _decimal(row.get("dividend", 0), f"{symbol} dividend")
            if dividend < 0:
                raise ValueError("dividend must be non-negative")
            lots = self._lots.get(symbol, [])
            self._cash += dividend * sum(lot.quantity for lot in lots)
            borrow_rate = _decimal(row.get("borrow_rate", 0), f"{symbol} borrow_rate")
            if borrow_rate < 0:
                raise ValueError("borrow_rate must be non-negative")
            short_quantity = abs(sum(min(lot.quantity, 0) for lot in lots))
            if short_quantity and borrow_rate:
                periods = Decimal(str(max(self._sampling_periods_per_year, 1.0)))
                self._cash -= (
                    _decimal(row["close"], f"{symbol} close")
                    * short_quantity
                    * borrow_rate
                    / periods
                )
            self.cash = float(self._cash)
        self._sync_positions()

    @staticmethod
    def _infer_periods_per_year(timestamps: pd.DatetimeIndex) -> float:
        if len(timestamps) < 2:
            return 0.0
        # ``DatetimeIndex.asi8`` follows the index's storage resolution in
        # pandas 3 (often microseconds), so normalize explicitly to ns before
        # converting to seconds.
        nanoseconds = timestamps.to_numpy(dtype="datetime64[ns]").astype("int64")
        deltas = np.diff(nanoseconds) / 1_000_000_000
        median_seconds = float(np.median(deltas))
        if not math.isfinite(median_seconds) or median_seconds <= 0:
            return 0.0
        day = 86400.0
        if median_seconds < 18 * 3600:
            return 252.0 * 6.5 * 3600.0 / median_seconds
        if median_seconds <= 2 * day:
            return 252.0
        if median_seconds <= 10 * day:
            return 52.0
        if median_seconds <= 40 * day:
            return 12.0
        return 365.2425 * day / median_seconds

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate finite, sampling-aware performance metrics."""

        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = final_equity / self.initial_capital - 1.0
        wins = [trade for trade in self.trades if trade.pnl > 0]
        losses = [trade for trade in self.trades if trade.pnl < 0]
        gross_profit = sum(trade.pnl for trade in wins)
        gross_loss = abs(sum(trade.pnl for trade in losses))
        returns = np.asarray(self.daily_returns, dtype=float)
        annualizer = (
            math.sqrt(self._sampling_periods_per_year)
            if self._sampling_periods_per_year > 0
            else 0.0
        )
        standard_deviation = float(np.std(returns)) if len(returns) > 1 else 0.0
        sharpe = (
            annualizer * float(np.mean(returns)) / standard_deviation
            if standard_deviation > 0
            else 0.0
        )
        downside = np.minimum(returns, 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
        sortino = (
            annualizer * float(np.mean(returns)) / downside_deviation
            if downside_deviation > 0
            else 0.0
        )
        calmar = total_return / self.max_drawdown if self.max_drawdown > 0 else 0.0
        return {
            "total_return": total_return,
            "total_pnl": sum(trade.pnl for trade in self.trades),
            "num_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": self.max_drawdown,
            "avg_win": float(np.mean([trade.pnl for trade in wins])) if wins else 0.0,
            "avg_loss": float(np.mean([trade.pnl for trade in losses])) if losses else 0.0,
            "avg_duration_days": (
                float(np.mean([trade.duration_days for trade in self.trades]))
                if self.trades
                else 0.0
            ),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "returns": total_return * 100.0,
            "final_equity": final_equity,
            "sampling_periods_per_year": self._sampling_periods_per_year,
        }

    def _create_results(self, signals: List[Dict]) -> BacktestResult:
        equity = pd.Series(
            list(self.equity_curve),
            index=pd.DatetimeIndex(list(self.timestamps)),
            dtype=float,
            name="equity",
        )
        returns = pd.Series(
            list(self.daily_returns),
            index=pd.DatetimeIndex(list(self.timestamps[1:])),
            dtype=float,
            name="return",
        )
        drawdown = self._calculate_drawdown_series(equity)
        signals_frame = pd.DataFrame(copy.deepcopy(signals)) if signals else pd.DataFrame()
        return BacktestResult(
            equity_curve=equity.copy(deep=True),
            trades=list(self.trades),
            positions=list(self.positions.values()),
            metrics=dict(self.calculate_metrics()),
            daily_returns=returns.copy(deep=True),
            drawdown_series=drawdown.copy(deep=True),
            signals=signals_frame.copy(deep=True),
            errors=tuple(self._errors),
            approval_eligible=self._approval_eligible and not self._errors,
            sampling_periods_per_year=self._sampling_periods_per_year,
            finalization_policy=self.finalization_policy,
        )

    @staticmethod
    def _calculate_drawdown_series(equity_series: pd.Series) -> pd.Series:
        if equity_series.empty:
            return pd.Series(dtype=float)
        running_max = equity_series.cummax()
        return (equity_series - running_max) / running_max.replace(0, np.nan)
