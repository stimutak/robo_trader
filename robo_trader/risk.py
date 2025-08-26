"""
Enhanced risk management system with ATR-based position sizing and portfolio heat tracking.

This module provides comprehensive risk controls including:
- ATR-based position sizing
- Portfolio heat calculation
- Correlation tracking
- Pre-trade validation
- Emergency shutdown triggers
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)


class RiskViolationType(Enum):
    """Types of risk violations."""

    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_SIZE_LIMIT = "position_size_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    CORRELATION_LIMIT = "correlation_limit"
    SECTOR_EXPOSURE_LIMIT = "sector_exposure_limit"
    PORTFOLIO_HEAT_LIMIT = "portfolio_heat_limit"
    ORDER_NOTIONAL_LIMIT = "order_notional_limit"
    DAILY_NOTIONAL_LIMIT = "daily_notional_limit"
    VOLUME_LIMIT = "volume_limit"
    MARKET_CAP_LIMIT = "market_cap_limit"


@dataclass
class Position:
    """Enhanced position tracking with additional metadata."""

    symbol: str
    quantity: int
    avg_price: float
    entry_time: datetime = field(default_factory=datetime.now)
    sector: Optional[str] = None
    beta: float = 1.0
    atr: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_active: bool = False
    trailing_stop_distance: Optional[float] = None
    max_price_since_entry: Optional[float] = None

    @property
    def notional_value(self) -> float:
        """Calculate position notional value."""
        return abs(self.quantity * self.avg_price)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.is_long:
            return (current_price - self.avg_price) * self.quantity
        else:
            return (self.avg_price - current_price) * abs(self.quantity)


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""

    portfolio_heat: float
    portfolio_beta: float
    max_correlation: float
    sector_exposures: Dict[str, float]
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float


class RiskManager:
    """
    Advanced risk management system with comprehensive controls.

    Features:
    - ATR-based dynamic position sizing
    - Portfolio heat calculation
    - Correlation tracking
    - Sector exposure limits
    - Emergency shutdown triggers
    - Real-time risk monitoring
    """

    def __init__(
        self,
        max_daily_loss: float,
        max_position_risk_pct: float,
        max_symbol_exposure_pct: float,
        max_leverage: float,
        max_order_notional: Optional[float] = None,
        max_daily_notional: Optional[float] = None,
        max_portfolio_heat: float = 0.06,  # 6% portfolio heat limit
        position_sizing_method: str = "atr",  # atr, fixed, kelly
        atr_risk_factor: float = 2.0,  # Risk 2x ATR per position
        min_volume: int = 1_000_000,
        min_market_cap: float = 1_000_000_000,
        correlation_limit: float = 0.7,
        max_sector_exposure_pct: float = 0.3,
        max_open_positions: int = 20,
        enable_emergency_shutdown: bool = True,
    ) -> None:
        """
        Initialize the risk manager with enhanced parameters.

        Args:
            max_daily_loss: Maximum daily loss in dollars
            max_position_risk_pct: Maximum risk per position as % of portfolio
            max_symbol_exposure_pct: Maximum exposure per symbol as % of portfolio
            max_leverage: Maximum account leverage
            max_order_notional: Maximum notional per order
            max_daily_notional: Maximum daily trading notional
            max_portfolio_heat: Maximum portfolio heat (sum of position risks)
            position_sizing_method: Method for position sizing (atr, fixed, kelly)
            atr_risk_factor: Number of ATRs to risk per position
            min_volume: Minimum daily volume for stocks
            min_market_cap: Minimum market capitalization
            correlation_limit: Maximum correlation between positions
            max_sector_exposure_pct: Maximum exposure per sector
            max_open_positions: Maximum number of open positions
            enable_emergency_shutdown: Enable automatic emergency shutdown
        """
        self.max_daily_loss = float(max_daily_loss)
        self.max_position_risk_pct = float(max_position_risk_pct)
        self.max_symbol_exposure_pct = float(max_symbol_exposure_pct)
        self.max_leverage = float(max_leverage)
        self.max_order_notional = (
            float(max_order_notional) if max_order_notional else None
        )
        self.max_daily_notional = (
            float(max_daily_notional) if max_daily_notional else None
        )
        self.max_portfolio_heat = float(max_portfolio_heat)
        self.position_sizing_method = position_sizing_method
        self.atr_risk_factor = float(atr_risk_factor)
        self.min_volume = int(min_volume)
        self.min_market_cap = float(min_market_cap)
        self.correlation_limit = float(correlation_limit)
        self.max_sector_exposure_pct = float(max_sector_exposure_pct)
        self.max_open_positions = int(max_open_positions)
        self.enable_emergency_shutdown = enable_emergency_shutdown

        # Risk tracking
        self.violations: List[Tuple[datetime, RiskViolationType, str]] = []
        self.daily_executed_notional = 0.0
        self.portfolio_correlations: Dict[str, Dict[str, float]] = {}
        self.sector_exposures: Dict[str, float] = {}
        self.emergency_shutdown_triggered = False

        # Market data cache for calculations
        self.atr_cache: Dict[str, float] = {}
        self.volume_cache: Dict[str, int] = {}
        self.market_cap_cache: Dict[str, float] = {}
        self.beta_cache: Dict[str, float] = {}

    def update_market_data(
        self,
        symbol: str,
        atr: Optional[float] = None,
        volume: Optional[int] = None,
        market_cap: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        """Update cached market data for a symbol."""
        if atr is not None:
            self.atr_cache[symbol] = atr
        if volume is not None:
            self.volume_cache[symbol] = volume
        if market_cap is not None:
            self.market_cap_cache[symbol] = market_cap
        if beta is not None:
            self.beta_cache[symbol] = beta

    def position_size_atr(
        self,
        symbol: str,
        cash_available: float,
        entry_price: float,
        atr: Optional[float] = None,
    ) -> int:
        """
        Calculate position size using ATR-based risk management.

        Risk per trade = Account * Risk%
        Position size = Risk per trade / (ATR * Risk Factor)

        Args:
            symbol: Trading symbol
            cash_available: Available cash
            entry_price: Entry price
            atr: Average True Range (will use cached if not provided)

        Returns:
            Number of shares to trade
        """
        if entry_price <= 0 or cash_available <= 0:
            return 0

        # Get ATR from cache or parameter
        atr_value = atr or self.atr_cache.get(symbol)
        if not atr_value or atr_value <= 0:
            logger.warning(f"No ATR data for {symbol}, falling back to fixed sizing")
            return self.position_size_fixed(cash_available, entry_price)

        # Calculate risk per trade
        risk_per_trade = cash_available * self.max_position_risk_pct

        # Calculate position size based on ATR
        risk_per_share = atr_value * self.atr_risk_factor
        position_size = risk_per_trade / risk_per_share

        # Ensure we can afford the position
        max_shares_affordable = int(cash_available / entry_price)
        final_size = min(int(position_size), max_shares_affordable)

        logger.debug(
            f"ATR position sizing for {symbol}: "
            f"ATR={atr_value:.2f}, risk/share={risk_per_share:.2f}, "
            f"position={final_size} shares"
        )

        return max(final_size, 0)

    def position_size_fixed(self, cash_available: float, entry_price: float) -> int:
        """
        Fixed position sizing using percentage of equity.

        Args:
            cash_available: Available cash
            entry_price: Entry price

        Returns:
            Number of shares to trade
        """
        if entry_price <= 0 or cash_available <= 0:
            return 0
        notional = cash_available * self.max_position_risk_pct
        return max(int(notional // entry_price), 0)

    def position_size_kelly(
        self,
        symbol: str,
        cash_available: float,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> int:
        """
        Kelly Criterion position sizing.

        Kelly % = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = ratio of win to loss

        Args:
            symbol: Trading symbol
            cash_available: Available cash
            entry_price: Entry price
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)

        Returns:
            Number of shares to trade
        """
        if entry_price <= 0 or cash_available <= 0:
            return 0

        if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
            logger.warning(f"Invalid Kelly parameters for {symbol}, using fixed sizing")
            return self.position_size_fixed(cash_available, entry_price)

        # Calculate Kelly percentage
        b = avg_win / avg_loss
        q = 1 - win_rate
        kelly_pct = (win_rate * b - q) / b

        # Apply Kelly fraction (usually 0.25 for safety)
        kelly_fraction = 0.25
        position_pct = min(kelly_pct * kelly_fraction, self.max_position_risk_pct)

        if position_pct <= 0:
            logger.info(f"Kelly criterion suggests no position for {symbol}")
            return 0

        notional = cash_available * position_pct
        return max(int(notional // entry_price), 0)

    def position_size(
        self,
        cash_available: float,
        entry_price: float,
        symbol: Optional[str] = None,
        **kwargs,
    ) -> int:
        """
        Calculate position size using configured method.

        Args:
            cash_available: Available cash
            entry_price: Entry price
            symbol: Trading symbol (required for ATR/Kelly)
            **kwargs: Additional parameters for specific methods

        Returns:
            Number of shares to trade
        """
        if self.position_sizing_method == "atr" and symbol:
            return self.position_size_atr(symbol, cash_available, entry_price, **kwargs)
        elif self.position_sizing_method == "kelly" and symbol:
            return self.position_size_kelly(
                symbol, cash_available, entry_price, **kwargs
            )
        else:
            return self.position_size_fixed(cash_available, entry_price)

    def calculate_portfolio_heat(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
    ) -> float:
        """
        Calculate portfolio heat (sum of position risks).

        Portfolio heat = Sum of (Position Value * Position Risk %)

        Args:
            positions: Current positions
            current_prices: Current market prices

        Returns:
            Portfolio heat as percentage
        """
        if not positions:
            return 0.0

        total_heat = 0.0
        total_value = sum(
            abs(pos.quantity) * current_prices.get(sym, pos.avg_price)
            for sym, pos in positions.items()
        )

        if total_value <= 0:
            return 0.0

        for symbol, pos in positions.items():
            current_price = current_prices.get(symbol, pos.avg_price)
            position_value = abs(pos.quantity) * current_price

            # Calculate position risk based on stop loss or ATR
            if pos.stop_loss:
                risk_per_share = abs(current_price - pos.stop_loss)
            elif pos.atr:
                risk_per_share = pos.atr * self.atr_risk_factor
            else:
                # Default to 2% if no stop or ATR
                risk_per_share = current_price * 0.02

            position_risk = risk_per_share * abs(pos.quantity)
            position_heat = position_risk / total_value
            total_heat += position_heat

        return total_heat

    def calculate_portfolio_beta(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
    ) -> float:
        """
        Calculate weighted portfolio beta.

        Args:
            positions: Current positions
            current_prices: Current market prices

        Returns:
            Portfolio beta
        """
        if not positions:
            return 0.0

        total_value = 0.0
        weighted_beta = 0.0

        for symbol, pos in positions.items():
            current_price = current_prices.get(symbol, pos.avg_price)
            position_value = abs(pos.quantity) * current_price
            total_value += position_value

            # Use cached beta or position beta
            beta = self.beta_cache.get(symbol, pos.beta)
            weighted_beta += position_value * beta

        if total_value <= 0:
            return 1.0

        return weighted_beta / total_value

    def check_correlation_limit(
        self,
        symbol: str,
        positions: Dict[str, Position],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if adding a position would violate correlation limits.

        Args:
            symbol: Symbol to check
            positions: Current positions

        Returns:
            Tuple of (is_valid, violation_message)
        """
        if symbol not in self.portfolio_correlations:
            return True, None

        for other_symbol in positions:
            if other_symbol == symbol:
                continue

            correlation = self.portfolio_correlations.get(symbol, {}).get(
                other_symbol, 0
            )
            if abs(correlation) > self.correlation_limit:
                msg = f"Correlation between {symbol} and {other_symbol} ({correlation:.2f}) exceeds limit"
                return False, msg

        return True, None

    def check_sector_exposure(
        self,
        symbol: str,
        order_value: float,
        equity: float,
        sector: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check sector exposure limits.

        Args:
            symbol: Trading symbol
            order_value: Order notional value
            equity: Total equity
            sector: Sector classification

        Returns:
            Tuple of (is_valid, violation_message)
        """
        if not sector or equity <= 0:
            return True, None

        current_exposure = self.sector_exposures.get(sector, 0.0)
        new_exposure = (current_exposure + order_value) / equity

        if new_exposure > self.max_sector_exposure_pct:
            msg = f"Sector {sector} exposure ({new_exposure:.1%}) would exceed limit"
            return False, msg

        return True, None

    def validate_order(
        self,
        symbol: str,
        order_qty: int,
        price: float,
        equity: float,
        daily_pnl: float,
        current_positions: Dict[str, Position],
        daily_executed_notional: float = 0.0,
        current_prices: Optional[Dict[str, float]] = None,
        sector: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Comprehensive pre-trade validation.

        Args:
            symbol: Trading symbol
            order_qty: Order quantity
            price: Order price
            equity: Account equity
            daily_pnl: Daily P&L
            current_positions: Current positions
            daily_executed_notional: Daily executed notional
            current_prices: Current market prices
            sector: Sector classification

        Returns:
            Tuple of (is_valid, validation_message)
        """
        # Check emergency shutdown
        if self.emergency_shutdown_triggered:
            return False, "Emergency shutdown active"

        # Basic validation
        if order_qty <= 0:
            return False, "Quantity must be positive"
        if price <= 0:
            return False, "Invalid price"

        # Daily loss limit
        if daily_pnl <= -abs(self.max_daily_loss):
            self._record_violation(RiskViolationType.DAILY_LOSS_LIMIT, symbol)
            return False, "Daily loss limit reached"

        # Position count limit
        if len(current_positions) >= self.max_open_positions:
            return False, f"Maximum {self.max_open_positions} positions reached"

        order_notional = price * order_qty

        # Per-order notional limit
        if self.max_order_notional and order_notional > self.max_order_notional:
            self._record_violation(RiskViolationType.ORDER_NOTIONAL_LIMIT, symbol)
            return False, "Order notional exceeds per-order limit"

        # Daily notional limit
        if self.max_daily_notional:
            new_daily_total = daily_executed_notional + order_notional
            if new_daily_total > self.max_daily_notional:
                self._record_violation(RiskViolationType.DAILY_NOTIONAL_LIMIT, symbol)
                return False, "Daily notional exceeds limit"

        # Symbol exposure limit
        max_symbol_notional = equity * self.max_symbol_exposure_pct
        if order_notional > max_symbol_notional:
            self._record_violation(RiskViolationType.POSITION_SIZE_LIMIT, symbol)
            return False, "Symbol exposure exceeds limit"

        # Leverage check
        existing_notional = sum(
            pos.notional_value for pos in current_positions.values()
        )
        total_after = existing_notional + order_notional
        if equity > 0 and (total_after / equity) > self.max_leverage:
            self._record_violation(RiskViolationType.LEVERAGE_LIMIT, symbol)
            return False, "Account leverage exceeds limit"

        # Volume check
        if symbol in self.volume_cache:
            if self.volume_cache[symbol] < self.min_volume:
                self._record_violation(RiskViolationType.VOLUME_LIMIT, symbol)
                return False, f"Volume ({self.volume_cache[symbol]:,}) below minimum"

        # Market cap check
        if symbol in self.market_cap_cache:
            if self.market_cap_cache[symbol] < self.min_market_cap:
                self._record_violation(RiskViolationType.MARKET_CAP_LIMIT, symbol)
                return False, f"Market cap below ${self.min_market_cap:,.0f}"

        # Correlation check
        is_valid, msg = self.check_correlation_limit(symbol, current_positions)
        if not is_valid:
            self._record_violation(RiskViolationType.CORRELATION_LIMIT, symbol)
            return False, msg

        # Sector exposure check
        is_valid, msg = self.check_sector_exposure(
            symbol, order_notional, equity, sector
        )
        if not is_valid:
            self._record_violation(RiskViolationType.SECTOR_EXPOSURE_LIMIT, symbol)
            return False, msg

        # Portfolio heat check
        if current_prices:
            heat = self.calculate_portfolio_heat(current_positions, current_prices)
            if heat > self.max_portfolio_heat:
                self._record_violation(RiskViolationType.PORTFOLIO_HEAT_LIMIT, symbol)
                return False, f"Portfolio heat ({heat:.1%}) exceeds limit"

        return True, "OK"

    def should_emergency_shutdown(self) -> bool:
        """
        Check if emergency shutdown should be triggered.

        Returns:
            True if shutdown should be triggered
        """
        if not self.enable_emergency_shutdown:
            return False

        if self.emergency_shutdown_triggered:
            return True

        # Check for multiple violations in short time
        recent_violations = [
            v for v in self.violations if v[0] > datetime.now() - timedelta(minutes=5)
        ]

        # Trigger if 5+ violations in 5 minutes
        if len(recent_violations) >= 5:
            logger.error(
                f"Emergency shutdown triggered: {len(recent_violations)} violations in 5 minutes"
            )
            self.emergency_shutdown_triggered = True
            return True

        # Check for critical violations
        critical_types = {
            RiskViolationType.DAILY_LOSS_LIMIT,
            RiskViolationType.LEVERAGE_LIMIT,
            RiskViolationType.PORTFOLIO_HEAT_LIMIT,
        }

        if any(v[1] in critical_types for v in recent_violations):
            logger.error("Emergency shutdown triggered: Critical risk violation")
            self.emergency_shutdown_triggered = True
            return True

        return False

    def update_trailing_stop(
        self,
        position: Position,
        current_price: float,
        trailing_pct: float = 0.015,
    ) -> Optional[float]:
        """
        Update trailing stop for a position.

        Args:
            position: Position to update
            current_price: Current market price
            trailing_pct: Trailing stop percentage

        Returns:
            New stop loss price if updated
        """
        if not position.trailing_stop_active:
            return None

        # Update max price since entry
        if position.max_price_since_entry is None:
            position.max_price_since_entry = current_price
        else:
            position.max_price_since_entry = max(
                position.max_price_since_entry, current_price
            )

        # Calculate new stop
        if position.is_long:
            new_stop = position.max_price_since_entry * (1 - trailing_pct)
            if position.stop_loss is None or new_stop > position.stop_loss:
                position.stop_loss = new_stop
                return new_stop
        else:
            new_stop = position.max_price_since_entry * (1 + trailing_pct)
            if position.stop_loss is None or new_stop < position.stop_loss:
                position.stop_loss = new_stop
                return new_stop

        return None

    def calculate_risk_metrics(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        returns_history: Optional[pd.Series] = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio.

        Args:
            positions: Current positions
            current_prices: Current market prices
            returns_history: Historical returns for risk calculations

        Returns:
            RiskMetrics object with calculated values
        """
        portfolio_heat = self.calculate_portfolio_heat(positions, current_prices)
        portfolio_beta = self.calculate_portfolio_beta(positions, current_prices)

        # Find maximum correlation
        max_correlation = 0.0
        for sym1 in positions:
            for sym2 in positions:
                if sym1 != sym2:
                    corr = self.portfolio_correlations.get(sym1, {}).get(sym2, 0)
                    max_correlation = max(max_correlation, abs(corr))

        # Calculate VaR and Expected Shortfall if returns provided
        var_95 = 0.0
        expected_shortfall = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        max_drawdown = 0.0
        current_drawdown = 0.0

        if returns_history is not None and len(returns_history) > 0:
            # VaR at 95% confidence
            var_95 = np.percentile(returns_history, 5)

            # Expected Shortfall (CVaR)
            tail_returns = returns_history[returns_history <= var_95]
            if len(tail_returns) > 0:
                expected_shortfall = tail_returns.mean()

            # Sharpe Ratio (assuming 0 risk-free rate)
            if returns_history.std() > 0:
                sharpe_ratio = (
                    returns_history.mean() / returns_history.std() * np.sqrt(252)
                )

            # Sortino Ratio
            downside_returns = returns_history[returns_history < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    sortino_ratio = returns_history.mean() / downside_std * np.sqrt(252)

            # Maximum Drawdown
            cumulative = (1 + returns_history).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            current_drawdown = drawdown.iloc[-1]

        return RiskMetrics(
            portfolio_heat=portfolio_heat,
            portfolio_beta=portfolio_beta,
            max_correlation=max_correlation,
            sector_exposures=self.sector_exposures.copy(),
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
        )

    def _record_violation(self, violation_type: RiskViolationType, symbol: str) -> None:
        """Record a risk violation."""
        self.violations.append((datetime.now(), violation_type, symbol))
        logger.warning(f"Risk violation: {violation_type.value} for {symbol}")

    def reset_daily_counters(self) -> None:
        """Reset daily tracking counters."""
        self.daily_executed_notional = 0.0
        # Keep only violations from today
        today = datetime.now().date()
        self.violations = [v for v in self.violations if v[0].date() == today]

    def update_correlations(self, correlations: Dict[str, Dict[str, float]]) -> None:
        """Update portfolio correlations matrix."""
        self.portfolio_correlations = correlations

    def update_sector_exposures(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        equity: float,
    ) -> None:
        """Update sector exposure tracking."""
        self.sector_exposures.clear()

        if equity <= 0:
            return

        for symbol, pos in positions.items():
            if pos.sector:
                current_price = current_prices.get(symbol, pos.avg_price)
                position_value = abs(pos.quantity) * current_price
                exposure_pct = position_value / equity

                if pos.sector in self.sector_exposures:
                    self.sector_exposures[pos.sector] += exposure_pct
                else:
                    self.sector_exposures[pos.sector] = exposure_pct


def create_risk_manager_from_config(config) -> RiskManager:
    """
    Create RiskManager from configuration object.

    Args:
        config: Configuration object with risk settings

    Returns:
        Configured RiskManager instance
    """
    return RiskManager(
        max_daily_loss=config.risk.max_daily_loss_pct * config.default_cash,
        max_position_risk_pct=config.risk.max_position_pct,
        max_symbol_exposure_pct=config.risk.max_sector_exposure_pct,
        max_leverage=config.risk.max_leverage,
        max_order_notional=config.risk.max_order_notional,
        max_daily_notional=config.risk.max_daily_notional,
        position_sizing_method=config.risk.position_sizing_method,
        min_volume=config.risk.min_volume,
        min_market_cap=config.risk.min_market_cap,
        correlation_limit=config.risk.correlation_limit,
        max_sector_exposure_pct=config.risk.max_sector_exposure_pct,
        max_open_positions=config.risk.max_open_positions,
    )
