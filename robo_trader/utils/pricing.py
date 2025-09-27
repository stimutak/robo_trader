"""
Decimal-based Pricing Utilities - Fix for Critical Bug #3: Float Precision Errors

Provides precise decimal arithmetic for price and quantity calculations.
Prevents order rejections and wrong position sizes due to float precision issues.
"""

from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from typing import Optional, Union

# Type alias for numeric inputs
Numeric = Union[int, float, str, Decimal]


class PrecisePricing:
    """Handles all price and quantity calculations with decimal precision."""

    # Common tick sizes for different price ranges
    TICK_SIZES = {
        "penny": Decimal("0.01"),  # Most stocks
        "half_penny": Decimal("0.005"),  # Some ETFs
        "nickel": Decimal("0.05"),  # Some low-priced stocks
        "quarter": Decimal("0.25"),  # Some very low-priced stocks
        "eighth": Decimal("0.125"),  # Some bonds
    }

    @staticmethod
    def to_decimal(value: Numeric) -> Decimal:
        """Convert any numeric value to Decimal safely."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    @staticmethod
    def round_price(price: Numeric, tick_size: Numeric = "0.01") -> Decimal:
        """Round price to valid tick size."""
        price_d = PrecisePricing.to_decimal(price)
        tick_d = PrecisePricing.to_decimal(tick_size)

        if tick_d <= 0:
            raise ValueError(f"Tick size must be positive, got {tick_d}")

        return (price_d / tick_d).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick_d

    @staticmethod
    def calculate_shares(capital: Numeric, price: Numeric, round_down: bool = True) -> int:
        """Calculate shares with proper precision - no truncation errors."""
        capital_d = PrecisePricing.to_decimal(capital)
        price_d = PrecisePricing.to_decimal(price)

        if price_d <= 0:
            raise ValueError(f"Price must be positive, got {price_d}")

        if capital_d <= 0:
            return 0

        # Calculate exact shares
        shares_exact = capital_d / price_d

        # Round appropriately
        if round_down:
            shares = int(shares_exact.quantize(Decimal("1"), rounding=ROUND_DOWN))
        else:
            shares = int(shares_exact.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        return max(shares, 0)

    @staticmethod
    def calculate_notional(shares: int, price: Numeric) -> Decimal:
        """Calculate exact notional value."""
        price_d = PrecisePricing.to_decimal(price)
        return Decimal(str(shares)) * price_d

    @staticmethod
    def calculate_pnl(entry_price: Numeric, exit_price: Numeric, shares: int) -> Decimal:
        """Calculate exact P&L without float precision errors."""
        entry_d = PrecisePricing.to_decimal(entry_price)
        exit_d = PrecisePricing.to_decimal(exit_price)
        shares_d = Decimal(str(shares))

        return (exit_d - entry_d) * shares_d

    @staticmethod
    def split_order_size(total_shares: int, max_chunk_size: int) -> list[int]:
        """Split large order into smaller chunks for execution."""
        if total_shares <= 0:
            return []

        if max_chunk_size <= 0:
            raise ValueError("Max chunk size must be positive")

        if total_shares <= max_chunk_size:
            return [total_shares]

        chunks = []
        remaining = total_shares

        while remaining > 0:
            chunk_size = min(remaining, max_chunk_size)
            chunks.append(chunk_size)
            remaining -= chunk_size

        return chunks

    @staticmethod
    def validate_price_increment(price: Numeric, tick_size: Numeric = "0.01") -> bool:
        """Validate that price conforms to tick size."""
        price_d = PrecisePricing.to_decimal(price)
        tick_d = PrecisePricing.to_decimal(tick_size)

        if tick_d <= 0:
            return False

        # Check if price is a multiple of tick size
        remainder = price_d % tick_d
        return remainder == 0

    @staticmethod
    def calculate_average_price(fills: list[tuple[Numeric, int]]) -> Optional[Decimal]:
        """Calculate volume-weighted average price from fills."""
        if not fills:
            return None

        total_notional = Decimal("0")
        total_shares = 0

        for price, shares in fills:
            if shares <= 0:
                continue

            price_d = PrecisePricing.to_decimal(price)
            total_notional += price_d * Decimal(str(shares))
            total_shares += shares

        if total_shares == 0:
            return None

        return total_notional / Decimal(str(total_shares))

    @staticmethod
    def calculate_slippage_bps(intended_price: Numeric, actual_price: Numeric) -> Decimal:
        """Calculate slippage in basis points."""
        intended_d = PrecisePricing.to_decimal(intended_price)
        actual_d = PrecisePricing.to_decimal(actual_price)

        if intended_d <= 0:
            raise ValueError("Intended price must be positive")

        slippage = (actual_d - intended_d) / intended_d
        return slippage * Decimal("10000")  # Convert to basis points

    @staticmethod
    def apply_commission(
        gross_pnl: Numeric, shares: int, commission_per_share: Numeric = "0.005"
    ) -> Decimal:
        """Apply commission costs to P&L calculation."""
        gross_d = PrecisePricing.to_decimal(gross_pnl)
        commission_d = PrecisePricing.to_decimal(commission_per_share)

        # Entry + Exit commissions
        total_commission = commission_d * Decimal(str(shares)) * Decimal("2")

        return gross_d - total_commission


class OrderSizing:
    """Helper class for order sizing calculations."""

    @staticmethod
    def kelly_sizing(
        capital: Numeric, win_rate: float, avg_win: Numeric, avg_loss: Numeric
    ) -> Decimal:
        """Calculate Kelly criterion position size."""
        if not (0 < win_rate < 1):
            raise ValueError("Win rate must be between 0 and 1")

        capital_d = PrecisePricing.to_decimal(capital)
        avg_win_d = PrecisePricing.to_decimal(avg_win)
        avg_loss_d = PrecisePricing.to_decimal(avg_loss)

        if avg_loss_d <= 0 or avg_win_d <= 0:
            raise ValueError("Average win and loss must be positive")

        # Kelly formula: f = (bp - q) / b
        # where b = odds received, p = probability of win, q = probability of loss
        b = avg_win_d / avg_loss_d  # Odds
        p = Decimal(str(win_rate))
        q = Decimal("1") - p

        kelly_fraction = (b * p - q) / b

        # Apply maximum Kelly fraction for safety (typically 25% of full Kelly)
        max_kelly = Decimal("0.25")
        kelly_fraction = min(kelly_fraction, max_kelly)
        kelly_fraction = max(kelly_fraction, Decimal("0"))  # No negative sizing

        return capital_d * kelly_fraction

    @staticmethod
    def fixed_dollar_sizing(dollar_amount: Numeric, price: Numeric) -> int:
        """Calculate shares for fixed dollar amount."""
        return PrecisePricing.calculate_shares(dollar_amount, price, round_down=True)

    @staticmethod
    def percent_of_portfolio_sizing(
        portfolio_value: Numeric, percent: float, price: Numeric
    ) -> int:
        """Calculate shares for percentage of portfolio."""
        if not (0 <= percent <= 1):
            raise ValueError("Percent must be between 0 and 1")

        portfolio_d = PrecisePricing.to_decimal(portfolio_value)
        dollar_amount = portfolio_d * Decimal(str(percent))

        return PrecisePricing.calculate_shares(dollar_amount, price, round_down=True)


class RiskCalculations:
    """Risk-related calculations with decimal precision."""

    @staticmethod
    def calculate_var(
        position_value: Numeric,
        volatility: float,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
    ) -> Decimal:
        """Calculate Value at Risk (VaR)."""
        from math import sqrt

        from scipy.stats import norm

        position_d = PrecisePricing.to_decimal(position_value)

        # Z-score for confidence level
        z_score = norm.ppf(confidence_level)

        # Adjust volatility for time horizon
        adjusted_vol = volatility * sqrt(time_horizon_days)

        var = position_d * Decimal(str(adjusted_vol)) * Decimal(str(z_score))
        return var

    @staticmethod
    def position_limit_check(
        new_position_value: Numeric, current_portfolio: Numeric, max_position_percent: float = 0.1
    ) -> bool:
        """Check if position size exceeds portfolio limits."""
        new_pos_d = PrecisePricing.to_decimal(new_position_value)
        portfolio_d = PrecisePricing.to_decimal(current_portfolio)

        if portfolio_d <= 0:
            return False

        position_percent = new_pos_d / portfolio_d
        return position_percent <= Decimal(str(max_position_percent))


# Convenience functions for common operations
def round_price(price: Numeric, tick_size: Numeric = "0.01") -> float:
    """Round price to tick size and return as float."""
    return float(PrecisePricing.round_price(price, tick_size))


def calculate_shares(capital: Numeric, price: Numeric) -> int:
    """Calculate shares for given capital and price."""
    return PrecisePricing.calculate_shares(capital, price)


def calculate_notional(shares: int, price: Numeric) -> float:
    """Calculate notional value and return as float."""
    return float(PrecisePricing.calculate_notional(shares, price))


# Export main classes and functions
__all__ = [
    "PrecisePricing",
    "OrderSizing",
    "RiskCalculations",
    "round_price",
    "calculate_shares",
    "calculate_notional",
]
