"""Deterministic, bar-based execution simulation for backtesting.

The simulator is deliberately isolated from runtime execution.  It consumes one
exact OHLCV event and returns a value object; it has no broker, database, or
order-submission authority.

Intrabar policy
---------------
Bars are interpreted conservatively as ``open`` followed by an unknown path
through ``high`` and ``low`` and finally ``close``.  Market orders execute at
the opening touch.  A limit that is marketable at the open receives the opening
touch (or the limit, whichever is worse); a limit reached later in the bar
executes at its limit.  A stop gapped through at the open executes at the
opening touch; a stop reached later becomes a market order at the stop touch.
No assumption is made about whether the high or low occurred first.

For a logical order carried across bars, callers pass cumulative filled
quantity and commission already paid.  The per-order minimum is then charged
once while per-share commission can accrue as cumulative fills grow.
Engine callers also pass the last completed bar's known volume as the next
bar's liquidity budget; execution-bar total volume is never used at its open.
Execution analytics define fill rate by filled/requested quantity and require
an explicit average portfolio value for turnover; they never invent one from
the fill price.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
_ORDER_TYPES = frozenset({"market", "limit", "stop"})
_SIDES = frozenset({"buy", "sell"})


@dataclass
class ExecutionCost:
    """Execution-cost breakdown.

    Spread, market impact, and slippage are adverse per-share amounts already
    reflected exactly once in ``fill_price``.  Commission is a total cash fee
    and is intentionally *not* embedded in ``fill_price``.  ``total_cost`` is
    the total cash cost for the filled quantity, including commission.
    """

    spread_cost: float
    market_impact: float
    commission: float
    slippage: float
    total_cost: float
    fill_price: float


@dataclass
class SimulatedOrder:
    """Simulated order with execution details.

    ``quantity`` remains the requested quantity for API compatibility.
    ``partial_fill`` remains public and now consistently means the executed
    quantity.  The explicit quantity fields remove ambiguity for new callers.
    """

    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    timestamp: datetime
    requested_price: float
    fill_price: float
    execution_cost: ExecutionCost
    filled: bool
    partial_fill: int = 0
    filled_quantity: int = 0
    remaining_quantity: int = 0

    @property
    def fully_filled(self) -> bool:
        """Return whether the entire requested quantity executed."""

        return self.filled and self.filled_quantity == self.quantity


class MarketImpactModel:
    """Simple deterministic participation-rate market-impact model."""

    def __init__(
        self,
        permanent_impact_factor: float = 0.1,
        temporary_impact_factor: float = 0.05,
        gamma: float = 1.5,
    ):
        self.permanent_impact = self._finite_nonnegative(
            permanent_impact_factor, "permanent_impact_factor"
        )
        self.temporary_impact = self._finite_nonnegative(
            temporary_impact_factor, "temporary_impact_factor"
        )
        self.gamma = self._finite_nonnegative(gamma, "gamma")

    @staticmethod
    def _finite_nonnegative(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and non-negative")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return numeric

    def calculate_impact(
        self, order_size: int, avg_volume: float, volatility: float, spread: float
    ) -> Tuple[float, float]:
        """Return permanent and temporary impact as fractions of price.

        ``spread`` is the spread as a fraction of price.  Both results are
        non-negative adverse costs.  A zero-volume event cannot execute and
        therefore has zero calculated impact.
        """

        if avg_volume <= 0 or order_size <= 0:
            return 0.0, 0.0

        participation_rate = min(abs(order_size) / avg_volume, 1.0)
        permanent = self.permanent_impact * (participation_rate**self.gamma) * volatility
        temporary = self.temporary_impact * math.sqrt(participation_rate) * spread
        return float(permanent), float(temporary)


class ExecutionSimulator:
    """Simulate deterministic market, limit, and stop executions on OHLCV bars.

    Random slippage comes only from an owned ``numpy.random.Generator``.  Call
    :meth:`reset` at the start of each backtest run to reproduce the same fill
    sequence.  The default seed is fixed, so even callers that do not yet call
    ``reset`` remain deterministic.
    """

    supports_parent_order_commission = True
    supports_lagged_liquidity = True
    supports_lagged_cost_inputs = True
    supports_shared_event_capacity = True
    supports_cash_budget = True

    def __init__(
        self,
        spread_model: str = "dynamic",
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        market_impact_model: Optional[MarketImpactModel] = None,
        slippage_factor: float = 0.0001,
        use_real_spreads: bool = True,
        random_seed: int = 0,
        max_volume_participation: float = 0.10,
    ):
        if spread_model not in {"fixed", "dynamic", "historical"}:
            raise ValueError("spread_model must be fixed, dynamic, or historical")

        self.spread_model = spread_model
        self.commission_per_share = self._finite_nonnegative(
            commission_per_share, "commission_per_share"
        )
        self.min_commission = self._finite_nonnegative(min_commission, "min_commission")
        self.market_impact_model = market_impact_model or MarketImpactModel()
        self.slippage_factor = self._finite_nonnegative(slippage_factor, "slippage_factor")
        if not isinstance(use_real_spreads, bool):
            raise ValueError("use_real_spreads must be a boolean")
        self.use_real_spreads = use_real_spreads

        participation = float(max_volume_participation)
        if not math.isfinite(participation) or not 0 < participation <= 1:
            raise ValueError("max_volume_participation must be in (0, 1]")
        self.max_volume_participation = participation

        self.random_seed = self._validate_seed(random_seed)
        self._rng = np.random.default_rng(self.random_seed)

        # Retained for public compatibility.  Exact-event reads deliberately do
        # not populate or consult this cache.
        self.market_data_cache: Dict = {}

    @staticmethod
    def _finite_nonnegative(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and non-negative")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return numeric

    @staticmethod
    def _validate_seed(seed: int) -> int:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        return int(seed)

    def reset(self, random_seed: Optional[int] = None) -> None:
        """Reset the owned generator for a new deterministic backtest run.

        Passing a seed selects and remembers it for subsequent no-argument
        resets.  No global NumPy random state is read or changed.
        """

        if random_seed is not None:
            self.random_seed = self._validate_seed(random_seed)
        self._rng = np.random.default_rng(self.random_seed)

    def reset_random_state(self, random_seed: Optional[int] = None) -> None:
        """Compatibility-friendly alias for :meth:`reset`."""

        self.reset(random_seed)

    def simulate_execution(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str,
        price_data: pd.DataFrame,
        timestamp: datetime,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        cumulative_filled_quantity: int = 0,
        commission_paid: float = 0.0,
        liquidity_volume: Optional[float] = None,
        max_fill_quantity: Optional[int] = None,
        cash_available: Optional[float] = None,
        liquidity_volatility: Optional[float] = None,
        liquidity_spread: Optional[float] = None,
        ignore_reported_cost_inputs: bool = False,
        reserve_short_margin: bool = False,
    ) -> SimulatedOrder:
        """Simulate an order using only the bar at ``timestamp``.

        Invalid inputs raise ``TypeError`` or ``ValueError``.  A valid order
        with no exact event, no trigger, or no participation capacity returns a
        well-formed unfilled order.  Data from earlier or later timestamps is
        never substituted.
        """

        event_timestamp = self._validate_order_and_data(
            symbol,
            quantity,
            side,
            order_type,
            price_data,
            timestamp,
            limit_price,
            stop_price,
        )
        if (
            isinstance(cumulative_filled_quantity, bool)
            or not isinstance(cumulative_filled_quantity, Integral)
            or cumulative_filled_quantity < 0
        ):
            raise ValueError("cumulative_filled_quantity must be a non-negative integer")
        normalized_commission_paid = self._finite_nonnegative(commission_paid, "commission_paid")
        normalized_liquidity_volume = (
            None
            if liquidity_volume is None
            else self._finite_nonnegative(liquidity_volume, "liquidity_volume")
        )
        if max_fill_quantity is not None and (
            isinstance(max_fill_quantity, bool)
            or not isinstance(max_fill_quantity, Integral)
            or max_fill_quantity < 0
        ):
            raise ValueError("max_fill_quantity must be a non-negative integer")
        normalized_cash_available = (
            None
            if cash_available is None
            else self._finite_nonnegative(cash_available, "cash_available")
        )
        normalized_liquidity_volatility = (
            None
            if liquidity_volatility is None
            else self._finite_nonnegative(liquidity_volatility, "liquidity_volatility")
        )
        normalized_liquidity_spread = (
            None
            if liquidity_spread is None
            else self._finite_nonnegative(liquidity_spread, "liquidity_spread")
        )
        if not isinstance(ignore_reported_cost_inputs, bool):
            raise ValueError("ignore_reported_cost_inputs must be a boolean")
        if not isinstance(reserve_short_margin, bool):
            raise ValueError("reserve_short_margin must be a boolean")
        expected_commission_paid = (
            0.0
            if cumulative_filled_quantity == 0
            else max(
                self.min_commission,
                self.commission_per_share * int(cumulative_filled_quantity),
            )
        )
        if not math.isclose(
            normalized_commission_paid,
            expected_commission_paid,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("commission_paid does not match cumulative_filled_quantity and policy")
        requested_quantity = int(quantity)
        normalized_limit = float(limit_price) if limit_price is not None else None
        normalized_stop = float(stop_price) if stop_price is not None else None
        current_data = self._get_market_data_at_time(price_data, event_timestamp)
        if current_data is None:
            logger.warning("No exact market event available for %s at %s", symbol, timestamp)
            return self._create_unfilled_order(
                symbol, requested_quantity, side, order_type, timestamp, 0.0
            )

        arrival_price = float(current_data["open"])
        reported_volume = float(current_data["volume"])
        volume = (
            reported_volume if normalized_liquidity_volume is None else normalized_liquidity_volume
        )
        volatility = (
            0.02
            if ignore_reported_cost_inputs and normalized_liquidity_volatility is None
            else (
                float(current_data.get("volatility", 0.02))
                if normalized_liquidity_volatility is None
                else normalized_liquidity_volatility
            )
        )
        spread = self._calculate_spread(
            current_data,
            arrival_price,
            liquidity_volume=volume,
            volatility=volatility,
            observed_spread=normalized_liquidity_spread,
            reported_inputs_allowed=not ignore_reported_cost_inputs,
        )
        fills, base_fill_price = self._check_order_fill(
            order_type,
            side,
            arrival_price,
            spread,
            normalized_limit,
            normalized_stop,
            current_data,
        )
        if not fills:
            return self._create_unfilled_order(
                symbol, requested_quantity, side, order_type, timestamp, arrival_price
            )

        capacity = self.calculate_fill_capacity(volume)
        if max_fill_quantity is not None:
            capacity = min(capacity, int(max_fill_quantity))
        filled_quantity = min(requested_quantity, max(0, capacity))
        if filled_quantity == 0:
            return self._create_unfilled_order(
                symbol, requested_quantity, side, order_type, timestamp, arrival_price
            )

        slippage_draw = abs(float(self._rng.normal()))

        def execution_for(candidate_quantity: int) -> Tuple[ExecutionCost, float]:
            candidate_cost = self._calculate_execution_costs(
                candidate_quantity,
                side,
                base_fill_price,
                spread,
                volume,
                volatility,
                cumulative_filled_quantity=int(cumulative_filled_quantity),
                commission_paid=normalized_commission_paid,
                slippage_draw=slippage_draw,
            )
            candidate_price = self._apply_adverse_costs(
                base_fill_price,
                side,
                candidate_quantity,
                candidate_cost,
                normalized_limit if order_type == "limit" else None,
            )
            return candidate_cost, candidate_price

        if (side == "buy" or reserve_short_margin) and normalized_cash_available is not None:
            lower, upper = 0, filled_quantity
            while lower < upper:
                candidate = (lower + upper + 1) // 2
                candidate_cost, candidate_price = execution_for(candidate)
                collateral_price = (
                    max(candidate_price, arrival_price) if reserve_short_margin else candidate_price
                )
                required_cash = collateral_price * candidate + candidate_cost.commission
                if required_cash <= normalized_cash_available + 1e-12:
                    lower = candidate
                else:
                    upper = candidate - 1
            filled_quantity = lower
            if filled_quantity == 0:
                return self._create_unfilled_order(
                    symbol, requested_quantity, side, order_type, timestamp, arrival_price
                )

        execution_cost, final_fill_price = execution_for(filled_quantity)
        if not math.isfinite(final_fill_price) or final_fill_price <= 0:
            raise ValueError("cost model produced a non-finite or non-positive fill price")
        execution_cost.fill_price = final_fill_price

        return SimulatedOrder(
            symbol=symbol,
            quantity=requested_quantity,
            side=side,
            order_type=order_type,
            timestamp=timestamp,
            requested_price=arrival_price,
            fill_price=final_fill_price,
            execution_cost=execution_cost,
            filled=True,
            partial_fill=filled_quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=requested_quantity - filled_quantity,
        )

    def calculate_fill_capacity(self, liquidity_volume: float) -> int:
        """Return the whole-share capacity available to one symbol/event."""

        volume = self._finite_nonnegative(liquidity_volume, "liquidity_volume")
        return int(math.floor(volume * self.max_volume_participation))

    def _validate_order_and_data(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str,
        price_data: pd.DataFrame,
        timestamp: datetime,
        limit_price: Optional[float],
        stop_price: Optional[float],
    ) -> pd.Timestamp:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if isinstance(quantity, bool) or not isinstance(quantity, Integral) or quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        if side not in _SIDES:
            raise ValueError("side must be 'buy' or 'sell'")
        if order_type not in _ORDER_TYPES:
            raise ValueError("order_type must be market, limit, or stop")
        if not isinstance(timestamp, (datetime, pd.Timestamp)):
            raise TypeError("timestamp must be a datetime")

        event_timestamp = pd.Timestamp(timestamp)
        if pd.isna(event_timestamp):
            raise ValueError("timestamp must not be NaT")

        if order_type == "limit":
            self._validate_required_price(limit_price, "limit_price")
        elif limit_price is not None:
            self._validate_required_price(limit_price, "limit_price")
        if order_type == "stop":
            self._validate_required_price(stop_price, "stop_price")
        elif stop_price is not None:
            self._validate_required_price(stop_price, "stop_price")

        self._validate_price_data(price_data)
        index_timezone = price_data.index.tz
        timestamp_timezone = event_timestamp.tzinfo
        if (index_timezone is None) != (timestamp_timezone is None):
            raise ValueError(
                "timestamp and market-data index must have matching timezone awareness"
            )
        if index_timezone is not None:
            event_timestamp = event_timestamp.tz_convert(index_timezone)
        return event_timestamp

    @staticmethod
    def _validate_required_price(value: Optional[float], name: str) -> float:
        if value is None:
            raise ValueError(f"{name} is required")
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and positive")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return numeric

    @staticmethod
    def _validate_price_data(price_data: pd.DataFrame) -> None:
        if not isinstance(price_data, pd.DataFrame):
            raise TypeError("price_data must be a pandas DataFrame")
        if price_data.empty:
            raise ValueError("price_data must not be empty")
        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise TypeError("price_data must use a DatetimeIndex")
        if price_data.index.hasnans or price_data.index.has_duplicates:
            raise ValueError("price_data index must contain unique, valid timestamps")
        if not price_data.index.is_monotonic_increasing:
            raise ValueError("price_data index must be sorted in increasing order")

        missing = [column for column in _REQUIRED_COLUMNS if column not in price_data.columns]
        if missing:
            raise ValueError(f"price_data missing required OHLCV columns: {', '.join(missing)}")

        numeric_columns = list(_REQUIRED_COLUMNS)
        optional_columns = [
            column for column in ("bid", "ask", "volatility") if column in price_data.columns
        ]
        for column in numeric_columns + optional_columns:
            try:
                values = price_data[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"price_data {column} must be numeric") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"price_data {column} must contain only finite values")

        for column in ("open", "high", "low", "close"):
            if (price_data[column].to_numpy(dtype=float) <= 0).any():
                raise ValueError(f"price_data {column} must be positive")
        if (price_data["volume"].to_numpy(dtype=float) < 0).any():
            raise ValueError("price_data volume must be non-negative")

        opens = price_data["open"].to_numpy(dtype=float)
        highs = price_data["high"].to_numpy(dtype=float)
        lows = price_data["low"].to_numpy(dtype=float)
        closes = price_data["close"].to_numpy(dtype=float)
        if (highs < np.maximum.reduce([opens, lows, closes])).any():
            raise ValueError("price_data high must be at least open, low, and close")
        if (lows > np.minimum.reduce([opens, highs, closes])).any():
            raise ValueError("price_data low must be at most open, high, and close")

        has_bid = "bid" in price_data.columns
        has_ask = "ask" in price_data.columns
        if has_bid != has_ask:
            raise ValueError("price_data must provide bid and ask together")
        if has_bid:
            bids = price_data["bid"].to_numpy(dtype=float)
            asks = price_data["ask"].to_numpy(dtype=float)
            if (bids <= 0).any() or (asks <= 0).any() or (asks < bids).any():
                raise ValueError("price_data requires 0 < bid <= ask")
        if "volatility" in price_data.columns:
            if (price_data["volatility"].to_numpy(dtype=float) < 0).any():
                raise ValueError("price_data volatility must be non-negative")

    def _get_market_data_at_time(
        self, price_data: pd.DataFrame, timestamp: datetime
    ) -> Optional[pd.Series]:
        """Return only the exact timestamped event; never backfill or look ahead."""

        if timestamp not in price_data.index:
            return None
        row = price_data.loc[timestamp]
        # Duplicate indexes are rejected during validation, so this is always a
        # Series rather than an ambiguous DataFrame.
        return row

    def _calculate_spread(
        self,
        market_data: pd.Series,
        mid_price: float,
        *,
        liquidity_volume: Optional[float] = None,
        volatility: Optional[float] = None,
        observed_spread: Optional[float] = None,
        reported_inputs_allowed: bool = True,
    ) -> float:
        """Calculate a finite, non-negative full bid-ask spread."""

        if self.use_real_spreads and observed_spread is not None:
            return float(observed_spread)
        if (
            reported_inputs_allowed
            and self.use_real_spreads
            and "bid" in market_data
            and "ask" in market_data
        ):
            return float(market_data["ask"] - market_data["bid"])

        if self.spread_model == "fixed":
            return 0.01
        if self.spread_model == "dynamic":
            normalized_volatility = (
                float(market_data.get("volatility", 0.02))
                if volatility is None
                else float(volatility)
            )
            volume = (
                float(market_data["volume"])
                if liquidity_volume is None
                else float(liquidity_volume)
            )
            volume_factor = max(math.log10(volume + 1.0) / 10.0, 0.05)
            return float(mid_price * (0.0001 + normalized_volatility * 0.01) / volume_factor)
        return float(mid_price * 0.0005)

    def _check_order_fill(
        self,
        order_type: str,
        side: str,
        mid_price: float,
        spread: float,
        limit_price: Optional[float],
        stop_price: Optional[float],
        market_data: pd.Series,
    ) -> Tuple[bool, float]:
        """Apply the documented open-first, otherwise-conservative bar policy."""

        half_spread = spread / 2.0
        if self.use_real_spreads and "bid" in market_data and "ask" in market_data:
            opening_touch = float(market_data["ask"] if side == "buy" else market_data["bid"])
        else:
            opening_touch = mid_price + half_spread if side == "buy" else mid_price - half_spread

        if order_type == "market":
            return True, opening_touch

        high = float(market_data["high"])
        low = float(market_data["low"])
        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price is required")
            if side == "buy":
                if opening_touch <= limit_price:
                    return True, min(opening_touch, limit_price)
                if low <= limit_price:
                    return True, float(limit_price)
            else:
                if opening_touch >= limit_price:
                    return True, max(opening_touch, limit_price)
                if high >= limit_price:
                    return True, float(limit_price)
            return False, 0.0

        if stop_price is None:
            raise ValueError("stop_price is required")
        if side == "buy":
            if mid_price >= stop_price:
                return True, max(opening_touch, float(stop_price))
            if high >= stop_price:
                return True, float(stop_price) + half_spread
        else:
            if mid_price <= stop_price:
                return True, min(opening_touch, float(stop_price))
            if low <= stop_price:
                return True, float(stop_price) - half_spread
        return False, 0.0

    def _calculate_execution_costs(
        self,
        quantity: int,
        side: str,
        fill_price: float,
        spread: float,
        volume: float,
        volatility: float,
        *,
        cumulative_filled_quantity: int,
        commission_paid: float,
        slippage_draw: Optional[float] = None,
    ) -> ExecutionCost:
        """Calculate each adverse cost component exactly once."""

        del side  # Costs are adverse magnitudes; side is applied at the price boundary.
        spread_cost = spread / 2.0
        relative_spread = spread / fill_price
        permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
            quantity, volume, volatility, relative_spread
        )
        market_impact = (permanent_impact + temporary_impact) * fill_price
        cumulative_commission = max(
            self.min_commission,
            self.commission_per_share * (cumulative_filled_quantity + quantity),
        )
        commission = max(cumulative_commission - commission_paid, 0.0)
        normalized_draw = (
            abs(float(self._rng.normal()))
            if slippage_draw is None
            else self._finite_nonnegative(slippage_draw, "slippage_draw")
        )
        slippage = fill_price * self.slippage_factor * normalized_draw
        total_cost = (spread_cost + market_impact + slippage) * quantity + commission
        return ExecutionCost(
            spread_cost=float(spread_cost),
            market_impact=float(market_impact),
            commission=float(commission),
            slippage=float(slippage),
            total_cost=float(total_cost),
            fill_price=float(fill_price),
        )

    @staticmethod
    def _apply_adverse_costs(
        base_fill_price: float,
        side: str,
        quantity: int,
        execution_cost: ExecutionCost,
        limit_price: Optional[float],
    ) -> float:
        """Apply impact/slippage without charging spread or commission twice."""

        impact = execution_cost.market_impact
        slippage = execution_cost.slippage
        if limit_price is not None:
            price_room = (
                max(float(limit_price) - base_fill_price, 0.0)
                if side == "buy"
                else max(base_fill_price - float(limit_price), 0.0)
            )
            impact = min(impact, price_room)
            slippage = min(slippage, max(price_room - impact, 0.0))

        execution_cost.market_impact = impact
        execution_cost.slippage = slippage
        execution_cost.total_cost = (
            execution_cost.spread_cost + impact + slippage
        ) * quantity + execution_cost.commission
        adverse_adjustment = impact + slippage
        if side == "buy":
            return float(base_fill_price + adverse_adjustment)
        return float(base_fill_price - adverse_adjustment)

    def _create_unfilled_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str,
        timestamp: datetime,
        price: float,
    ) -> SimulatedOrder:
        """Create an unfilled order while retaining the requested quantity."""

        return SimulatedOrder(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            timestamp=timestamp,
            requested_price=float(price),
            fill_price=0.0,
            execution_cost=ExecutionCost(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            filled=False,
            partial_fill=0,
            filled_quantity=0,
            remaining_quantity=quantity,
        )

    @staticmethod
    def _executed_quantity(trade: SimulatedOrder) -> int:
        """Read new and legacy constructed order objects consistently."""

        if trade.filled_quantity > 0:
            return trade.filled_quantity
        if trade.partial_fill > 0:
            return trade.partial_fill
        return trade.quantity if trade.filled else 0

    def calculate_portfolio_turnover(
        self, trades: List[SimulatedOrder], *, average_portfolio_value: float
    ) -> float:
        """Calculate traded notional divided by an explicit portfolio value."""

        denominator = float(average_portfolio_value)
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("average_portfolio_value must be finite and positive")

        filled_trades = [trade for trade in trades if self._executed_quantity(trade) > 0]
        if not filled_trades:
            return 0.0
        total_traded = sum(
            abs(self._executed_quantity(trade) * trade.fill_price) for trade in filled_trades
        )
        return total_traded / denominator

    def get_execution_analytics(
        self, trades: List[SimulatedOrder], *, average_portfolio_value: float
    ) -> Dict:
        """Return aggregate analytics without double-counting any cost component."""

        denominator = float(average_portfolio_value)
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("average_portfolio_value must be finite and positive")
        if not trades:
            return {}
        filled_trades = [trade for trade in trades if self._executed_quantity(trade) > 0]
        total_requested_quantity = sum(trade.quantity for trade in trades)
        if total_requested_quantity <= 0:
            raise ValueError("analytics require positive requested quantities")
        total_filled_quantity = sum(self._executed_quantity(trade) for trade in filled_trades)
        if not filled_trades:
            return {"fill_rate": 0.0}

        total_spread_cost = sum(
            trade.execution_cost.spread_cost * self._executed_quantity(trade)
            for trade in filled_trades
        )
        total_impact = sum(
            trade.execution_cost.market_impact * self._executed_quantity(trade)
            for trade in filled_trades
        )
        total_commission = sum(trade.execution_cost.commission for trade in filled_trades)
        total_slippage = sum(
            trade.execution_cost.slippage * self._executed_quantity(trade)
            for trade in filled_trades
        )
        return {
            "fill_rate": total_filled_quantity / total_requested_quantity,
            "avg_spread_cost_bps": float(
                np.mean(
                    [
                        trade.execution_cost.spread_cost / trade.fill_price * 10000
                        for trade in filled_trades
                    ]
                )
            ),
            "avg_market_impact_bps": float(
                np.mean(
                    [
                        trade.execution_cost.market_impact / trade.fill_price * 10000
                        for trade in filled_trades
                    ]
                )
            ),
            "total_spread_cost": total_spread_cost,
            "total_market_impact": total_impact,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_execution_cost": sum(trade.execution_cost.total_cost for trade in filled_trades),
            "turnover": self.calculate_portfolio_turnover(
                filled_trades, average_portfolio_value=denominator
            ),
        }
