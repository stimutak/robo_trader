"""
Advanced execution simulation for realistic backtesting.

This module implements:
- Order book simulation
- Market impact models
- Liquidity constraints
- Realistic fill algorithms
- Partial fills and order splitting
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logger import get_logger


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"


@dataclass
class OrderBookLevel:
    """Single level in order book."""

    price: float
    size: int
    orders: int


@dataclass
class OrderBook:
    """Simulated order book."""

    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_price: float
    last_size: int

    def get_spread(self) -> float:
        """Get bid-ask spread."""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    def get_mid_price(self) -> float:
        """Get mid price."""
        if self.bids and self.asks:
            return (self.asks[0].price + self.bids[0].price) / 2
        return self.last_price

    def get_imbalance(self) -> float:
        """Get order book imbalance."""
        if not self.bids or not self.asks:
            return 0.0

        bid_volume = sum(level.size for level in self.bids[:5])
        ask_volume = sum(level.size for level in self.asks[:5])

        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0.0

    def get_depth(self, levels: int = 5) -> Dict[str, float]:
        """Get order book depth metrics."""
        bid_depth = sum(level.size * level.price for level in self.bids[:levels])
        ask_depth = sum(level.size * level.price for level in self.asks[:levels])

        return {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": bid_depth + ask_depth,
            "depth_imbalance": (
                (bid_depth - ask_depth) / (bid_depth + ask_depth)
                if (bid_depth + ask_depth) > 0
                else 0
            ),
        }


class MarketImpactModel:
    """Market impact modeling for large orders."""

    def __init__(
        self,
        temporary_impact_const: float = 0.1,
        permanent_impact_const: float = 0.05,
        price_impact_exp: float = 0.5,
    ):
        self.temp_const = temporary_impact_const
        self.perm_const = permanent_impact_const
        self.price_exp = price_impact_exp
        self.logger = get_logger("execution.impact")

    def calculate_impact(
        self,
        order_size: int,
        avg_daily_volume: float,
        volatility: float,
        spread: float,
        urgency: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate market impact using Almgren-Chriss model.

        Args:
            order_size: Number of shares
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            spread: Bid-ask spread
            urgency: Trading urgency (0-1, higher = more aggressive)

        Returns:
            Impact components
        """
        if avg_daily_volume <= 0:
            return {
                "temporary_impact": 0,
                "permanent_impact": 0,
                "total_impact": 0,
                "effective_spread": spread,
            }

        # Calculate participation rate
        participation = order_size / avg_daily_volume

        # Temporary impact (will revert after execution)
        temp_impact = self.temp_const * volatility * np.power(participation, self.price_exp)

        # Permanent impact (information leakage)
        perm_impact = self.perm_const * volatility * participation

        # Adjust for urgency
        temp_impact *= 1 + urgency

        # Total expected impact
        total_impact = temp_impact + perm_impact

        # Effective spread including impact
        effective_spread = spread + total_impact

        return {
            "temporary_impact": temp_impact,
            "permanent_impact": perm_impact,
            "total_impact": total_impact,
            "effective_spread": effective_spread,
            "participation_rate": participation,
        }


class LiquidityProvider:
    """Simulate market liquidity dynamics."""

    def __init__(self):
        self.logger = get_logger("execution.liquidity")
        self.liquidity_factor = 1.0  # Multiplier for available liquidity
        self.resilience_rate = 0.1  # Rate at which liquidity replenishes

    def get_available_liquidity(
        self, symbol: str, timestamp: datetime, avg_volume: float, time_of_day_factor: float = 1.0
    ) -> float:
        """
        Get available liquidity at a point in time.

        Args:
            symbol: Trading symbol
            timestamp: Current time
            avg_volume: Average daily volume
            time_of_day_factor: Intraday liquidity pattern (0.5-2.0)

        Returns:
            Available liquidity in shares
        """
        # Base liquidity as fraction of ADV
        base_liquidity = avg_volume * 0.01  # 1% of ADV per minute

        # Adjust for time of day (U-shaped intraday pattern)
        adjusted_liquidity = base_liquidity * time_of_day_factor

        # Apply current liquidity factor
        available = adjusted_liquidity * self.liquidity_factor

        return max(0, available)

    def consume_liquidity(self, shares: int, available: float) -> Tuple[int, float]:
        """
        Consume available liquidity and update state.

        Args:
            shares: Shares to execute
            available: Available liquidity

        Returns:
            Tuple of (filled_shares, remaining_liquidity)
        """
        filled = min(shares, int(available))
        remaining = available - filled

        # Reduce liquidity factor after consumption
        if available > 0:
            consumption_rate = filled / available
            self.liquidity_factor *= 1 - consumption_rate * 0.5
            self.liquidity_factor = max(0.1, self.liquidity_factor)  # Min 10% liquidity

        return filled, remaining

    def replenish_liquidity(self, time_elapsed: float):
        """Replenish liquidity over time."""
        recovery = self.resilience_rate * time_elapsed
        self.liquidity_factor = min(1.0, self.liquidity_factor + recovery)


class ExecutionAlgorithm:
    """Base class for execution algorithms."""

    def __init__(self):
        self.logger = get_logger("execution.algo")

    def execute(
        self, order_size: int, market_data: pd.DataFrame, constraints: Dict[str, any]
    ) -> List[Dict]:
        """Execute order and return fills."""
        raise NotImplementedError


class TWAPExecutor(ExecutionAlgorithm):
    """Time-weighted average price execution."""

    def execute(
        self, order_size: int, market_data: pd.DataFrame, constraints: Dict[str, any]
    ) -> List[Dict]:
        """
        Execute order using TWAP algorithm.

        Args:
            order_size: Total shares to execute
            market_data: Market data with OHLCV
            constraints: Execution constraints

        Returns:
            List of fill dictionaries
        """
        fills = []

        # Get time horizon
        num_periods = constraints.get("periods", 10)
        max_participation = constraints.get("max_participation", 0.1)

        # Calculate slice size
        slice_size = order_size // num_periods
        remaining = order_size

        # Execute slices
        for i in range(min(num_periods, len(market_data))):
            row = market_data.iloc[i]

            # Check participation constraint
            if row["volume"] > 0:
                max_executable = int(row["volume"] * max_participation)
                executable = min(slice_size, max_executable, remaining)
            else:
                executable = 0

            if executable > 0:
                # Simulate fill at VWAP for the period
                fill_price = row.get("vwap", row["close"])

                fills.append(
                    {
                        "timestamp": row.name,
                        "quantity": executable,
                        "price": fill_price,
                        "volume": row["volume"],
                    }
                )

                remaining -= executable

            if remaining <= 0:
                break

        return fills


class VWAPExecutor(ExecutionAlgorithm):
    """Volume-weighted average price execution."""

    def execute(
        self, order_size: int, market_data: pd.DataFrame, constraints: Dict[str, any]
    ) -> List[Dict]:
        """
        Execute order using VWAP algorithm.

        Args:
            order_size: Total shares to execute
            market_data: Market data with OHLCV
            constraints: Execution constraints

        Returns:
            List of fill dictionaries
        """
        fills = []

        # Get constraints
        max_participation = constraints.get("max_participation", 0.1)

        # Calculate volume profile
        total_volume = market_data["volume"].sum()
        if total_volume <= 0:
            return fills

        remaining = order_size

        # Execute based on volume distribution
        for i, row in market_data.iterrows():
            if row["volume"] <= 0:
                continue

            # Calculate target size based on volume proportion
            volume_pct = row["volume"] / total_volume
            target_size = int(order_size * volume_pct)

            # Apply participation constraint
            max_executable = int(row["volume"] * max_participation)
            executable = min(target_size, max_executable, remaining)

            if executable > 0:
                # Fill at VWAP
                fill_price = row.get("vwap", row["close"])

                fills.append(
                    {
                        "timestamp": i,
                        "quantity": executable,
                        "price": fill_price,
                        "volume": row["volume"],
                    }
                )

                remaining -= executable

            if remaining <= 0:
                break

        return fills


class AdvancedExecutionSimulator:
    """Complete execution simulation system."""

    def __init__(self):
        self.logger = get_logger("execution.simulator")
        self.impact_model = MarketImpactModel()
        self.liquidity_provider = LiquidityProvider()

        # Execution algorithms
        self.algorithms = {"TWAP": TWAPExecutor(), "VWAP": VWAPExecutor()}

        # Metrics tracking
        self.execution_metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "partial_fills": 0,
            "avg_slippage": [],
            "avg_impact": [],
        }

    def simulate_market_order(
        self, side: str, quantity: int, order_book: OrderBook, market_impact: bool = True
    ) -> Dict[str, any]:
        """
        Simulate market order execution.

        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            order_book: Current order book
            market_impact: Whether to apply market impact

        Returns:
            Execution details
        """
        fills = []
        remaining = quantity
        total_cost = 0

        # Get relevant book side
        book_side = order_book.asks if side == "BUY" else order_book.bids

        # Walk through order book levels
        for level in book_side:
            if remaining <= 0:
                break

            # Calculate fill at this level
            fill_size = min(remaining, level.size)
            fill_cost = fill_size * level.price

            fills.append({"price": level.price, "quantity": fill_size})

            total_cost += fill_cost
            remaining -= fill_size

        # Calculate average fill price
        if quantity - remaining > 0:
            avg_price = total_cost / (quantity - remaining)
        else:
            avg_price = order_book.last_price

        # Calculate slippage
        if side == "BUY":
            slippage = (avg_price - order_book.get_mid_price()) / order_book.get_mid_price()
        else:
            slippage = (order_book.get_mid_price() - avg_price) / order_book.get_mid_price()

        # Apply market impact if enabled
        if market_impact and quantity > 0:
            impact = self.impact_model.calculate_impact(
                order_size=quantity,
                avg_daily_volume=100000,  # Default ADV
                volatility=0.02,  # 2% daily vol
                spread=order_book.get_spread(),
                urgency=0.8,  # High urgency for market orders
            )
            avg_price *= 1 + impact["total_impact"]
            slippage += impact["total_impact"]

        # Update metrics
        self.execution_metrics["total_orders"] += 1
        if remaining == 0:
            self.execution_metrics["filled_orders"] += 1
        elif remaining < quantity:
            self.execution_metrics["partial_fills"] += 1

        self.execution_metrics["avg_slippage"].append(slippage)

        return {
            "filled_quantity": quantity - remaining,
            "remaining_quantity": remaining,
            "average_price": avg_price,
            "slippage": slippage,
            "fills": fills,
            "total_cost": total_cost,
            "status": "FILLED" if remaining == 0 else "PARTIAL",
        }

    def simulate_limit_order(
        self,
        side: str,
        quantity: int,
        limit_price: float,
        order_book: OrderBook,
        time_in_force: str = "DAY",
    ) -> Dict[str, any]:
        """
        Simulate limit order execution.

        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            limit_price: Limit price
            order_book: Current order book
            time_in_force: Order time in force

        Returns:
            Execution details
        """
        # Check if order is marketable
        if side == "BUY" and order_book.asks and limit_price >= order_book.asks[0].price:
            # Marketable buy order
            return self.simulate_market_order(side, quantity, order_book)
        elif side == "SELL" and order_book.bids and limit_price <= order_book.bids[0].price:
            # Marketable sell order
            return self.simulate_market_order(side, quantity, order_book)

        # Non-marketable - add to book
        return {
            "filled_quantity": 0,
            "remaining_quantity": quantity,
            "average_price": 0,
            "slippage": 0,
            "fills": [],
            "total_cost": 0,
            "status": "PENDING",
        }

    def simulate_algo_order(
        self,
        side: str,
        quantity: int,
        algo_type: str,
        market_data: pd.DataFrame,
        constraints: Dict[str, any] = None,
    ) -> Dict[str, any]:
        """
        Simulate algorithmic order execution.

        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            algo_type: Algorithm type ('TWAP', 'VWAP', etc.)
            market_data: Historical market data
            constraints: Execution constraints

        Returns:
            Execution details
        """
        if algo_type not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algo_type}")

        # Get executor
        executor = self.algorithms[algo_type]

        # Execute with constraints
        fills = executor.execute(quantity, market_data, constraints or {})

        # Calculate summary statistics
        if fills:
            total_quantity = sum(f["quantity"] for f in fills)
            total_cost = sum(f["quantity"] * f["price"] for f in fills)
            avg_price = total_cost / total_quantity if total_quantity > 0 else 0

            # Calculate slippage vs arrival price
            arrival_price = market_data.iloc[0]["close"]
            if side == "BUY":
                slippage = (avg_price - arrival_price) / arrival_price
            else:
                slippage = (arrival_price - avg_price) / arrival_price
        else:
            total_quantity = 0
            avg_price = 0
            slippage = 0

        return {
            "filled_quantity": total_quantity,
            "remaining_quantity": quantity - total_quantity,
            "average_price": avg_price,
            "slippage": slippage,
            "fills": fills,
            "algorithm": algo_type,
            "status": "FILLED" if total_quantity >= quantity else "PARTIAL",
        }

    def get_metrics(self) -> Dict[str, any]:
        """Get execution metrics."""
        return {
            "fill_rate": self.execution_metrics["filled_orders"]
            / max(1, self.execution_metrics["total_orders"]),
            "partial_fill_rate": self.execution_metrics["partial_fills"]
            / max(1, self.execution_metrics["total_orders"]),
            "avg_slippage": (
                np.mean(self.execution_metrics["avg_slippage"])
                if self.execution_metrics["avg_slippage"]
                else 0
            ),
            "total_orders": self.execution_metrics["total_orders"],
        }
