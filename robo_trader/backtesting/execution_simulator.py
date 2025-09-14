"""
Realistic execution simulator for backtesting with market impact and slippage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionCost:
    """Breakdown of execution costs."""

    spread_cost: float
    market_impact: float
    commission: float
    slippage: float
    total_cost: float
    fill_price: float


@dataclass
class SimulatedOrder:
    """Simulated order with execution details."""

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


class MarketImpactModel:
    """
    Market impact model based on Almgren-Chriss framework.
    Estimates price impact based on order size and market conditions.
    """

    def __init__(
        self,
        permanent_impact_factor: float = 0.1,
        temporary_impact_factor: float = 0.05,
        gamma: float = 1.5,
    ):
        """
        Initialize market impact model.

        Args:
            permanent_impact_factor: Permanent price impact coefficient
            temporary_impact_factor: Temporary price impact coefficient
            gamma: Power law exponent for impact (typically 1.5-2)
        """
        self.permanent_impact = permanent_impact_factor
        self.temporary_impact = temporary_impact_factor
        self.gamma = gamma

    def calculate_impact(
        self, order_size: int, avg_volume: float, volatility: float, spread: float
    ) -> Tuple[float, float]:
        """
        Calculate market impact for an order.

        Args:
            order_size: Number of shares
            avg_volume: Average daily volume
            volatility: Price volatility (standard deviation)
            spread: Bid-ask spread

        Returns:
            Tuple of (permanent_impact, temporary_impact) in basis points
        """
        if avg_volume <= 0:
            return 0.0, 0.0

        # Participation rate (what % of volume we represent)
        participation_rate = abs(order_size) / avg_volume

        # Permanent impact (moves the price permanently)
        permanent = self.permanent_impact * (participation_rate**self.gamma) * volatility

        # Temporary impact (immediate cost, reverts over time)
        temporary = self.temporary_impact * np.sqrt(participation_rate) * spread

        return permanent, temporary


class ExecutionSimulator:
    """
    Simulates realistic order execution with:
    - Bid-ask spreads
    - Market impact
    - Slippage
    - Partial fills
    - Commission costs
    """

    def __init__(
        self,
        spread_model: str = "dynamic",
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        market_impact_model: Optional[MarketImpactModel] = None,
        slippage_factor: float = 0.0001,
        use_real_spreads: bool = True,
    ):
        """
        Initialize execution simulator.

        Args:
            spread_model: 'fixed', 'dynamic', or 'historical'
            commission_per_share: Commission per share traded
            min_commission: Minimum commission per trade
            market_impact_model: Market impact model to use
            slippage_factor: Random slippage factor (% of price)
            use_real_spreads: Use real bid-ask data if available
        """
        self.spread_model = spread_model
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.market_impact_model = market_impact_model or MarketImpactModel()
        self.slippage_factor = slippage_factor
        self.use_real_spreads = use_real_spreads

        # Cache for market data
        self.market_data_cache: Dict = {}

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
    ) -> SimulatedOrder:
        """
        Simulate order execution with realistic costs and slippage.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', or 'stop'
            price_data: DataFrame with OHLCV and optionally bid/ask data
            timestamp: Order timestamp
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders

        Returns:
            SimulatedOrder with execution details
        """
        # Get current market data
        current_data = self._get_market_data_at_time(price_data, timestamp)
        if current_data is None:
            logger.warning(f"No market data available for {symbol} at {timestamp}")
            return self._create_unfilled_order(symbol, quantity, side, order_type, timestamp, 0)

        mid_price = current_data["close"]
        volume = current_data.get("volume", 1000000)

        # Calculate spread
        spread = self._calculate_spread(current_data, mid_price)

        # Determine if order fills based on type
        fills, fill_price = self._check_order_fill(
            order_type, side, mid_price, spread, limit_price, stop_price, current_data
        )

        if not fills:
            return self._create_unfilled_order(
                symbol, quantity, side, order_type, timestamp, mid_price
            )

        # Calculate execution costs
        execution_cost = self._calculate_execution_costs(
            quantity, side, fill_price, spread, volume, current_data
        )

        # Apply costs to get final fill price
        if side == "buy":
            final_fill_price = fill_price + execution_cost.total_cost
        else:
            final_fill_price = fill_price - execution_cost.total_cost

        execution_cost.fill_price = final_fill_price

        return SimulatedOrder(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            timestamp=timestamp,
            requested_price=mid_price,
            fill_price=final_fill_price,
            execution_cost=execution_cost,
            filled=True,
            partial_fill=quantity,  # Full fill for now
        )

    def _get_market_data_at_time(
        self, price_data: pd.DataFrame, timestamp: datetime
    ) -> Optional[pd.Series]:
        """Get market data at or before timestamp."""
        # Ensure timestamp is comparable
        if hasattr(price_data.index, "tz"):
            # Handle timezone-aware index
            from pandas import Timestamp

            if not isinstance(timestamp, Timestamp):
                timestamp = Timestamp(timestamp)
            if timestamp.tz is None and price_data.index.tz is not None:
                timestamp = timestamp.tz_localize(price_data.index.tz)

        # Find the most recent data point
        if timestamp in price_data.index:
            return price_data.loc[timestamp]

        # Get most recent data before timestamp
        try:
            mask = price_data.index <= timestamp
            if mask.any():
                return price_data[mask].iloc[-1]
        except TypeError:
            # Handle comparison issues
            return price_data.iloc[-1] if not price_data.empty else None

        return None

    def _calculate_spread(self, market_data: pd.Series, mid_price: float) -> float:
        """Calculate bid-ask spread."""
        if self.use_real_spreads and "bid" in market_data and "ask" in market_data:
            return market_data["ask"] - market_data["bid"]

        if self.spread_model == "fixed":
            return 0.01  # 1 cent spread
        elif self.spread_model == "dynamic":
            # Dynamic spread based on volatility and volume
            volatility = market_data.get("volatility", 0.02)
            volume_factor = np.log10(market_data.get("volume", 1000000) + 1) / 10
            return mid_price * (0.0001 + volatility * 0.01) / volume_factor
        else:  # historical
            # Use historical average spread for the symbol
            return mid_price * 0.0005  # 5 basis points default

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
        """
        Check if order fills and at what price.

        Returns:
            Tuple of (fills: bool, fill_price: float)
        """
        if order_type == "market":
            # Market orders always fill
            if side == "buy":
                return True, mid_price + spread / 2  # Fill at ask
            else:
                return True, mid_price - spread / 2  # Fill at bid

        elif order_type == "limit":
            if limit_price is None:
                return False, 0

            # Check if limit price is met
            if side == "buy":
                ask_price = mid_price + spread / 2
                if limit_price >= ask_price:
                    # Buy limit fills at limit or better
                    return True, min(limit_price, ask_price)
            else:
                bid_price = mid_price - spread / 2
                if limit_price <= bid_price:
                    # Sell limit fills at limit or better
                    return True, max(limit_price, bid_price)

            return False, 0

        elif order_type == "stop":
            if stop_price is None:
                return False, 0

            # Check if stop is triggered
            high = market_data.get("high", mid_price)
            low = market_data.get("low", mid_price)

            if side == "buy" and high >= stop_price:
                # Buy stop triggered
                return True, max(stop_price, mid_price + spread / 2)
            elif side == "sell" and low <= stop_price:
                # Sell stop triggered
                return True, min(stop_price, mid_price - spread / 2)

            return False, 0

        return False, 0

    def _calculate_execution_costs(
        self,
        quantity: int,
        side: str,
        fill_price: float,
        spread: float,
        volume: float,
        market_data: pd.Series,
    ) -> ExecutionCost:
        """Calculate all execution costs."""
        # Spread cost (already included in fill price for market orders)
        spread_cost = spread / 2

        # Market impact
        volatility = market_data.get("volatility", 0.02)
        permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
            quantity, volume, volatility, spread
        )
        market_impact = (permanent_impact + temporary_impact) * fill_price

        # Commission
        commission = max(self.min_commission, self.commission_per_share * abs(quantity))

        # Random slippage
        slippage = fill_price * self.slippage_factor * np.random.randn()

        # Total cost per share
        total_cost = spread_cost + market_impact + commission / abs(quantity) + abs(slippage)

        return ExecutionCost(
            spread_cost=spread_cost,
            market_impact=market_impact,
            commission=commission,
            slippage=slippage,
            total_cost=total_cost,
            fill_price=fill_price,
        )

    def _create_unfilled_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str,
        timestamp: datetime,
        price: float,
    ) -> SimulatedOrder:
        """Create an unfilled order."""
        return SimulatedOrder(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            timestamp=timestamp,
            requested_price=price,
            fill_price=0,
            execution_cost=ExecutionCost(0, 0, 0, 0, 0, 0),
            filled=False,
        )

    def calculate_portfolio_turnover(self, trades: List[SimulatedOrder]) -> float:
        """Calculate portfolio turnover rate."""
        if not trades:
            return 0.0

        total_traded = sum(abs(t.quantity * t.fill_price) for t in trades if t.filled)
        # Approximate portfolio value
        avg_portfolio_value = np.mean([t.fill_price * 10000 for t in trades if t.filled])

        if avg_portfolio_value > 0:
            return total_traded / avg_portfolio_value
        return 0.0

    def get_execution_analytics(self, trades: List[SimulatedOrder]) -> Dict:
        """Get execution analytics for trades."""
        if not trades:
            return {}

        filled_trades = [t for t in trades if t.filled]

        if not filled_trades:
            return {"fill_rate": 0.0}

        total_spread_cost = sum(t.execution_cost.spread_cost * t.quantity for t in filled_trades)
        total_impact = sum(t.execution_cost.market_impact * t.quantity for t in filled_trades)
        total_commission = sum(t.execution_cost.commission for t in filled_trades)
        total_slippage = sum(abs(t.execution_cost.slippage * t.quantity) for t in filled_trades)

        return {
            "fill_rate": len(filled_trades) / len(trades),
            "avg_spread_cost_bps": np.mean(
                [t.execution_cost.spread_cost / t.fill_price * 10000 for t in filled_trades]
            ),
            "avg_market_impact_bps": np.mean(
                [t.execution_cost.market_impact / t.fill_price * 10000 for t in filled_trades]
            ),
            "total_spread_cost": total_spread_cost,
            "total_market_impact": total_impact,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_execution_cost": total_spread_cost
            + total_impact
            + total_commission
            + total_slippage,
            "turnover": self.calculate_portfolio_turnover(filled_trades),
        }
