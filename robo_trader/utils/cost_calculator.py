"""
Comprehensive Cost Calculator - Fix for Critical Bug #11: Missing Commission/Slippage

Provides accurate P&L calculations including all trading costs:
commissions, fees, slippage, spreads, and market impact.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

import structlog

from .market_time import get_market_time
from .pricing import PrecisePricing

logger = structlog.get_logger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class CostType(Enum):
    """Types of trading costs."""

    COMMISSION = "commission"
    FEES = "fees"
    SLIPPAGE = "slippage"
    SPREAD = "spread"
    MARKET_IMPACT = "market_impact"
    FINANCING = "financing"  # Overnight/margin costs


@dataclass
class CostBreakdown:
    """Detailed breakdown of trading costs."""

    commission: Decimal = Decimal("0")
    regulatory_fees: Decimal = Decimal("0")
    exchange_fees: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    spread_cost: Decimal = Decimal("0")
    market_impact: Decimal = Decimal("0")
    financing_cost: Decimal = Decimal("0")

    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost."""
        return (
            self.commission
            + self.regulatory_fees
            + self.exchange_fees
            + self.slippage
            + self.spread_cost
            + self.market_impact
            + self.financing_cost
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with float values."""
        return {
            "commission": float(self.commission),
            "regulatory_fees": float(self.regulatory_fees),
            "exchange_fees": float(self.exchange_fees),
            "slippage": float(self.slippage),
            "spread_cost": float(self.spread_cost),
            "market_impact": float(self.market_impact),
            "financing_cost": float(self.financing_cost),
            "total_cost": float(self.total_cost),
        }


@dataclass
class TradeCostConfig:
    """Configuration for trading cost calculations."""

    # Commission rates (per share or percentage)
    commission_per_share: Decimal = Decimal("0.005")  # $0.005 per share
    commission_percentage: Decimal = Decimal("0")  # Percentage of notional
    commission_minimum: Decimal = Decimal("1.00")  # Minimum commission
    commission_maximum: Decimal = Decimal("100.00")  # Maximum commission

    # Regulatory fees (percentage of notional)
    sec_fee_rate: Decimal = Decimal("0.0000278")  # SEC fee rate
    finra_taf_rate: Decimal = Decimal("0.000145")  # FINRA TAF rate

    # Exchange fees
    exchange_fee_per_share: Decimal = Decimal("0.0003")

    # Slippage models
    default_slippage_bps: Decimal = Decimal("1.0")  # 1 basis point default
    market_order_slippage_bps: Decimal = Decimal("2.0")  # Higher for market orders
    limit_order_slippage_bps: Decimal = Decimal("0.5")  # Lower for limit orders

    # Market impact (square root model)
    market_impact_coefficient: Decimal = Decimal("0.1")  # Impact coefficient

    # Financing costs (overnight positions)
    financing_rate_long: Decimal = Decimal("0.05")  # 5% annual for long positions
    financing_rate_short: Decimal = Decimal("0.08")  # 8% annual for short positions


class TradeCostCalculator:
    """
    Comprehensive trading cost calculator.

    Calculates all trading costs including:
    - Commissions (per share, percentage, min/max)
    - Regulatory fees (SEC, FINRA)
    - Exchange fees
    - Slippage estimation
    - Market impact
    - Financing costs
    """

    def __init__(self, config: Optional[TradeCostConfig] = None):
        self.config = config or TradeCostConfig()

    def calculate_commission(
        self, shares: int, notional_value: Decimal, order_type: str = "MARKET"
    ) -> Decimal:
        """
        Calculate commission costs.

        Args:
            shares: Number of shares
            notional_value: Notional value of trade
            order_type: Order type (affects commission structure)

        Returns:
            Commission amount
        """
        # Per-share commission
        per_share_commission = PrecisePricing.to_decimal(shares) * self.config.commission_per_share

        # Percentage commission
        percentage_commission = notional_value * self.config.commission_percentage

        # Take higher of the two
        commission = max(per_share_commission, percentage_commission)

        # Apply min/max limits
        commission = max(commission, self.config.commission_minimum)
        commission = min(commission, self.config.commission_maximum)

        return commission

    def calculate_regulatory_fees(self, notional_value: Decimal, side: OrderSide) -> Decimal:
        """
        Calculate regulatory fees (SEC, FINRA).

        Args:
            notional_value: Notional value of trade
            side: Order side (SELL orders have SEC fees)

        Returns:
            Total regulatory fees
        """
        fees = Decimal("0")

        # SEC fees only apply to sell orders
        if side == OrderSide.SELL:
            sec_fee = notional_value * self.config.sec_fee_rate
            fees += sec_fee

        # FINRA TAF applies to all trades
        finra_fee = notional_value * self.config.finra_taf_rate
        fees += finra_fee

        return fees

    def calculate_exchange_fees(self, shares: int) -> Decimal:
        """
        Calculate exchange fees.

        Args:
            shares: Number of shares

        Returns:
            Exchange fees
        """
        return PrecisePricing.to_decimal(shares) * self.config.exchange_fee_per_share

    def calculate_slippage(
        self,
        shares: int,
        price: Decimal,
        order_type: str = "MARKET",
        market_conditions: str = "NORMAL",
    ) -> Decimal:
        """
        Calculate slippage costs.

        Args:
            shares: Number of shares
            price: Execution price
            order_type: Order type (affects slippage)
            market_conditions: Market conditions (affects slippage)

        Returns:
            Slippage cost
        """
        # Base slippage rate based on order type
        if order_type.upper() == "MARKET":
            slippage_bps = self.config.market_order_slippage_bps
        else:
            slippage_bps = self.config.limit_order_slippage_bps

        # Adjust for market conditions
        if market_conditions.upper() == "VOLATILE":
            slippage_bps *= Decimal("2.0")
        elif market_conditions.upper() == "ILLIQUID":
            slippage_bps *= Decimal("1.5")

        # Calculate slippage amount
        notional = PrecisePricing.to_decimal(shares) * price
        slippage = notional * (slippage_bps / Decimal("10000"))  # Convert bps to decimal

        return slippage

    def calculate_spread_cost(
        self, shares: int, bid_price: Decimal, ask_price: Decimal, side: OrderSide
    ) -> Decimal:
        """
        Calculate spread crossing costs.

        Args:
            shares: Number of shares
            bid_price: Bid price
            ask_price: Ask price
            side: Order side

        Returns:
            Spread cost
        """
        spread = ask_price - bid_price

        if spread <= 0:
            return Decimal("0")

        # Market orders cross the spread
        # For buy orders, we pay the ask (spread cost vs mid)
        # For sell orders, we receive the bid (spread cost vs mid)
        mid_price = (bid_price + ask_price) / Decimal("2")

        if side == OrderSide.BUY:
            spread_cost_per_share = ask_price - mid_price
        else:
            spread_cost_per_share = mid_price - bid_price

        return PrecisePricing.to_decimal(shares) * spread_cost_per_share

    def calculate_market_impact(
        self, shares: int, price: Decimal, average_volume: int, volatility: float = 0.02
    ) -> Decimal:
        """
        Calculate market impact using square root model.

        Args:
            shares: Number of shares
            price: Stock price
            average_volume: Average daily volume
            volatility: Stock volatility

        Returns:
            Market impact cost
        """
        if average_volume <= 0:
            return Decimal("0")

        # Participation rate (trade size / average volume)
        participation_rate = Decimal(str(shares)) / Decimal(str(average_volume))

        # Square root market impact model
        # Impact = coefficient * volatility * sqrt(participation_rate)
        impact_bps = (
            self.config.market_impact_coefficient
            * Decimal(str(volatility))
            * participation_rate.sqrt()
        )

        # Convert to dollar amount
        notional = PrecisePricing.to_decimal(shares) * price
        impact = notional * impact_bps

        return impact

    def calculate_financing_cost(
        self, notional_value: Decimal, side: OrderSide, holding_days: int = 1
    ) -> Decimal:
        """
        Calculate financing costs for overnight positions.

        Args:
            notional_value: Notional value of position
            side: Position side (long/short affects rate)
            holding_days: Number of days held

        Returns:
            Financing cost
        """
        if holding_days <= 0:
            return Decimal("0")

        # Select appropriate rate
        if side == OrderSide.BUY:  # Long position
            annual_rate = self.config.financing_rate_long
        else:  # Short position
            annual_rate = self.config.financing_rate_short

        # Calculate daily cost
        daily_rate = annual_rate / Decimal("365")
        financing_cost = notional_value * daily_rate * Decimal(str(holding_days))

        return financing_cost

    def calculate_total_costs(
        self,
        shares: int,
        price: Decimal,
        side: OrderSide,
        order_type: str = "MARKET",
        bid_price: Optional[Decimal] = None,
        ask_price: Optional[Decimal] = None,
        average_volume: Optional[int] = None,
        volatility: float = 0.02,
        market_conditions: str = "NORMAL",
        holding_days: int = 0,
    ) -> CostBreakdown:
        """
        Calculate comprehensive trading costs.

        Args:
            shares: Number of shares
            price: Execution price
            side: Order side
            order_type: Order type
            bid_price: Bid price (for spread calculation)
            ask_price: Ask price (for spread calculation)
            average_volume: Average daily volume (for market impact)
            volatility: Stock volatility
            market_conditions: Market conditions
            holding_days: Days position is held

        Returns:
            Complete cost breakdown
        """
        price_decimal = PrecisePricing.to_decimal(price)
        notional_value = PrecisePricing.to_decimal(shares) * price_decimal

        costs = CostBreakdown()

        # Commission
        costs.commission = self.calculate_commission(shares, notional_value, order_type)

        # Regulatory fees
        costs.regulatory_fees = self.calculate_regulatory_fees(notional_value, side)

        # Exchange fees
        costs.exchange_fees = self.calculate_exchange_fees(shares)

        # Slippage
        costs.slippage = self.calculate_slippage(
            shares, price_decimal, order_type, market_conditions
        )

        # Spread cost (if bid/ask available)
        if bid_price is not None and ask_price is not None:
            costs.spread_cost = self.calculate_spread_cost(
                shares,
                PrecisePricing.to_decimal(bid_price),
                PrecisePricing.to_decimal(ask_price),
                side,
            )

        # Market impact (if volume data available)
        if average_volume is not None:
            costs.market_impact = self.calculate_market_impact(
                shares, price_decimal, average_volume, volatility
            )

        # Financing costs (if holding overnight)
        if holding_days > 0:
            costs.financing_cost = self.calculate_financing_cost(notional_value, side, holding_days)

        return costs

    def calculate_net_pnl(
        self,
        entry_shares: int,
        entry_price: Decimal,
        exit_shares: int,
        exit_price: Decimal,
        side: OrderSide,
        entry_costs: Optional[CostBreakdown] = None,
        exit_costs: Optional[CostBreakdown] = None,
        **kwargs,
    ) -> Dict[str, Decimal]:
        """
        Calculate net P&L including all costs.

        Args:
            entry_shares: Shares at entry
            entry_price: Entry price
            exit_shares: Shares at exit
            exit_price: Exit price
            side: Position side
            entry_costs: Entry cost breakdown (calculated if not provided)
            exit_costs: Exit cost breakdown (calculated if not provided)
            **kwargs: Additional parameters for cost calculation

        Returns:
            Dictionary with gross P&L, total costs, and net P&L
        """
        entry_price_d = PrecisePricing.to_decimal(entry_price)
        exit_price_d = PrecisePricing.to_decimal(exit_price)

        # Calculate gross P&L
        shares_traded = min(entry_shares, exit_shares)  # Handle partial exits

        if side == OrderSide.BUY:
            # Long position: profit when exit > entry
            gross_pnl = (exit_price_d - entry_price_d) * Decimal(str(shares_traded))
        else:
            # Short position: profit when entry > exit
            gross_pnl = (entry_price_d - exit_price_d) * Decimal(str(shares_traded))

        # Calculate costs if not provided
        if entry_costs is None:
            entry_costs = self.calculate_total_costs(entry_shares, entry_price, side, **kwargs)

        if exit_costs is None:
            # Exit side is opposite of entry
            exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            exit_costs = self.calculate_total_costs(exit_shares, exit_price, exit_side, **kwargs)

        total_costs = entry_costs.total_cost + exit_costs.total_cost
        net_pnl = gross_pnl - total_costs

        return {
            "gross_pnl": gross_pnl,
            "entry_costs": entry_costs.total_cost,
            "exit_costs": exit_costs.total_cost,
            "total_costs": total_costs,
            "net_pnl": net_pnl,
            "cost_percentage": (total_costs / abs(gross_pnl)) * Decimal("100")
            if gross_pnl != 0
            else Decimal("0"),
        }


class PortfolioCostTracker:
    """
    Track trading costs across an entire portfolio.
    """

    def __init__(self, calculator: Optional[TradeCostCalculator] = None):
        self.calculator = calculator or TradeCostCalculator()
        self.cost_history: List[Dict] = []

    def record_trade_costs(
        self, symbol: str, trade_id: str, costs: CostBreakdown, **metadata
    ) -> None:
        """Record trading costs for a specific trade."""
        record = {
            "symbol": symbol,
            "trade_id": trade_id,
            "timestamp": get_market_time().isoformat(),
            "costs": costs.to_dict(),
            **metadata,
        }
        self.cost_history.append(record)

    def get_cost_summary(
        self, symbol: Optional[str] = None, days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get cost summary for portfolio or specific symbol.

        Args:
            symbol: Symbol to filter by (None for all)
            days: Number of days to look back (None for all)

        Returns:
            Summary of costs by type
        """
        # Filter records
        records = self.cost_history

        if symbol:
            records = [r for r in records if r["symbol"] == symbol]

        if days:
            cutoff = get_market_time().timestamp() - (days * 24 * 3600)
            records = [
                r
                for r in records
                if get_market_time().fromisoformat(r["timestamp"]).timestamp() >= cutoff
            ]

        # Aggregate costs
        summary = {
            "total_commission": 0.0,
            "total_regulatory_fees": 0.0,
            "total_exchange_fees": 0.0,
            "total_slippage": 0.0,
            "total_spread_cost": 0.0,
            "total_market_impact": 0.0,
            "total_financing_cost": 0.0,
            "total_all_costs": 0.0,
            "trade_count": len(records),
        }

        for record in records:
            costs = record["costs"]
            summary["total_commission"] += costs["commission"]
            summary["total_regulatory_fees"] += costs["regulatory_fees"]
            summary["total_exchange_fees"] += costs["exchange_fees"]
            summary["total_slippage"] += costs["slippage"]
            summary["total_spread_cost"] += costs["spread_cost"]
            summary["total_market_impact"] += costs["market_impact"]
            summary["total_financing_cost"] += costs["financing_cost"]
            summary["total_all_costs"] += costs["total_cost"]

        return summary


# Global instance for easy access
default_cost_calculator = TradeCostCalculator()


# Convenience functions
def calculate_trade_costs(shares: int, price: float, side: str, **kwargs) -> Dict[str, float]:
    """Calculate trading costs using default calculator."""
    side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
    costs = default_cost_calculator.calculate_total_costs(
        shares, Decimal(str(price)), side_enum, **kwargs
    )
    return costs.to_dict()


def calculate_net_pnl(
    entry_shares: int, entry_price: float, exit_shares: int, exit_price: float, side: str, **kwargs
) -> Dict[str, float]:
    """Calculate net P&L using default calculator."""
    side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
    result = default_cost_calculator.calculate_net_pnl(
        entry_shares,
        Decimal(str(entry_price)),
        exit_shares,
        Decimal(str(exit_price)),
        side_enum,
        **kwargs,
    )
    return {k: float(v) for k, v in result.items()}


# Export main classes and functions
__all__ = [
    "TradeCostCalculator",
    "TradeCostConfig",
    "CostBreakdown",
    "PortfolioCostTracker",
    "OrderSide",
    "CostType",
    "default_cost_calculator",
    "calculate_trade_costs",
    "calculate_net_pnl",
]
