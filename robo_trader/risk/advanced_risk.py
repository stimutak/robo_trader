"""
Advanced Risk Management System for RoboTrader.

Implements Kelly criterion position sizing, correlation-based limits,
automated kill switches, and comprehensive risk monitoring.
"""

import asyncio
import json
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import timezone-aware market time utilities
from ..utils.market_time import get_market_time

# Handle optional dependencies gracefully
try:
    import numpy as np
    import pandas as pd

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy/Pandas not available. Some risk features will be limited.")


class RiskLevel(Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskMetrics:
    """Real-time risk metrics."""

    total_exposure: float
    leverage: float
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional VaR
    sharpe_ratio: float
    max_drawdown: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    timestamp: datetime = field(default_factory=get_market_time)


@dataclass
class KellyParameters:
    """Kelly criterion calculation parameters."""

    win_rate: float
    avg_win: float
    avg_loss: float
    kelly_fraction: float
    half_kelly: float  # Conservative Kelly (half of full Kelly)
    quarter_kelly: float  # Very conservative

    @property
    def edge(self) -> float:
        """Calculate edge (expected value)."""
        return self.win_rate * self.avg_win - (1 - self.win_rate) * abs(self.avg_loss)

    @property
    def odds(self) -> float:
        """Calculate odds (win/loss ratio)."""
        return self.avg_win / abs(self.avg_loss) if self.avg_loss != 0 else 0


class KellySizer:
    """Kelly criterion position sizing calculator."""

    def __init__(
        self,
        lookback_trades: int = 100,
        min_trades: int = 30,
        max_kelly_fraction: float = 0.25,  # Never exceed 25% even if Kelly suggests more
        use_half_kelly: bool = True,  # Use half-Kelly by default for safety
    ):
        self.lookback_trades = lookback_trades
        self.min_trades = min_trades
        self.max_kelly_fraction = max_kelly_fraction
        self.use_half_kelly = use_half_kelly

        # Trade history for Kelly calculation
        self.trade_history: deque = deque(maxlen=lookback_trades)
        self.kelly_cache: Dict[str, KellyParameters] = {}

    def add_trade(self, symbol: str, pnl: float, entry_price: float) -> None:
        """Record a completed trade for Kelly calculation."""
        self.trade_history.append(
            {
                "symbol": symbol,
                "pnl": pnl,
                "pnl_pct": pnl / entry_price if entry_price != 0 else 0,
                "timestamp": get_market_time(),
            }
        )

    def calculate_kelly(self, symbol: Optional[str] = None) -> KellyParameters:
        """
        Calculate Kelly fraction for position sizing.

        Kelly formula: f = (p * b - q) / b
        where:
            f = fraction of capital to bet
            p = probability of winning
            b = odds (amount won on win / amount lost on loss)
            q = probability of losing (1 - p)
        """
        # Filter trades for specific symbol if provided
        trades = [t for t in self.trade_history if symbol is None or t["symbol"] == symbol]

        if len(trades) < self.min_trades:
            # Not enough history, return conservative default
            return KellyParameters(
                win_rate=0.5,
                avg_win=0.02,
                avg_loss=0.02,
                kelly_fraction=0.02,
                half_kelly=0.01,
                quarter_kelly=0.005,
            )

        # Calculate win rate and average win/loss
        wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
        losses = [t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0]

        if not wins or not losses:
            # Edge case: all wins or all losses
            return KellyParameters(
                win_rate=len(wins) / len(trades),
                avg_win=sum(wins) / len(wins) if wins else 0,
                avg_loss=sum(losses) / len(losses) if losses else 0,
                kelly_fraction=0.01,  # Very conservative
                half_kelly=0.005,
                quarter_kelly=0.0025,
            )

        win_rate = len(wins) / len(trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        # Calculate Kelly fraction
        if avg_loss > 0:
            odds = avg_win / avg_loss
            kelly = (win_rate * odds - (1 - win_rate)) / odds
        else:
            kelly = 0

        # Apply safety limits
        kelly = max(0, min(kelly, self.max_kelly_fraction))

        params = KellyParameters(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            kelly_fraction=kelly,
            half_kelly=kelly * 0.5,
            quarter_kelly=kelly * 0.25,
        )

        # Cache result
        cache_key = symbol or "portfolio"
        self.kelly_cache[cache_key] = params

        return params

    def get_position_size(self, capital: float, symbol: Optional[str] = None) -> float:
        """Get recommended position size based on Kelly criterion."""
        kelly_params = self.calculate_kelly(symbol)

        if self.use_half_kelly:
            fraction = kelly_params.half_kelly
        else:
            fraction = kelly_params.kelly_fraction

        return capital * fraction


class CorrelationLimiter:
    """Manages correlation-based position limits."""

    def __init__(
        self,
        max_correlation: float = 0.7,
        max_correlated_exposure: float = 0.3,
        correlation_window: int = 60,
    ):
        self.max_correlation = max_correlation
        self.max_correlated_exposure = max_correlated_exposure
        self.correlation_window = correlation_window

        # Correlation matrix cache
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_correlation_update: Optional[datetime] = None
        self.price_history: Dict[str, deque] = {}

    def update_price(self, symbol: str, price: float) -> None:
        """Update price history for correlation calculation."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.correlation_window)
        self.price_history[symbol].append(price)

    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix from price history."""
        if not NUMPY_AVAILABLE:
            return pd.DataFrame()

        # Create DataFrame from price history
        data = {}
        for symbol, prices in self.price_history.items():
            if len(prices) >= 20:  # Minimum for meaningful correlation
                data[symbol] = list(prices)

        if len(data) < 2:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Calculate returns
        returns = df.pct_change().dropna()

        # Calculate correlation matrix
        self.correlation_matrix = returns.corr()
        self.last_correlation_update = get_market_time()

        return self.correlation_matrix

    def check_correlation_limit(
        self, symbol: str, current_positions: Dict[str, float]
    ) -> Tuple[bool, float, List[str]]:
        """
        Check if adding a position would violate correlation limits.

        Returns:
            Tuple of (is_allowed, max_correlation, highly_correlated_symbols)
        """
        if self.correlation_matrix is None or symbol not in self.correlation_matrix:
            return True, 0.0, []

        highly_correlated = []
        max_corr = 0.0

        for pos_symbol, exposure in current_positions.items():
            if pos_symbol != symbol and pos_symbol in self.correlation_matrix:
                correlation = abs(self.correlation_matrix.loc[symbol, pos_symbol])

                if correlation > self.max_correlation:
                    highly_correlated.append(pos_symbol)
                    max_corr = max(max_corr, correlation)

        # Check if total correlated exposure exceeds limit
        if highly_correlated:
            correlated_exposure = sum(current_positions.get(s, 0) for s in highly_correlated)

            if correlated_exposure > self.max_correlated_exposure:
                return False, max_corr, highly_correlated

        return True, max_corr, highly_correlated

    def get_correlation_penalty(self, symbol: str, current_positions: Dict[str, float]) -> float:
        """Calculate position size penalty based on correlations."""
        _, max_corr, correlated_symbols = self.check_correlation_limit(symbol, current_positions)

        if not correlated_symbols:
            return 1.0  # No penalty

        # Linear penalty based on correlation
        penalty = 1.0 - (max_corr - 0.5) * 2  # Scale from 0.5 to 1.0 correlation
        return max(0.3, penalty)  # Minimum 30% of original size


class KillSwitch:
    """Automated kill switch for emergency risk management."""

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,  # 5% daily loss limit
        max_position_loss_pct: float = 0.02,  # 2% per position loss limit
        max_consecutive_losses: int = 5,
        max_drawdown_pct: float = 0.10,  # 10% drawdown limit
        cooldown_minutes: int = 60,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_loss_pct = max_position_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_minutes = cooldown_minutes

        # Tracking variables
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.consecutive_losses = 0
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None

        # Position tracking
        self.position_pnl: Dict[str, float] = {}
        self.position_entry: Dict[str, Tuple[float, datetime]] = {}

    def update_equity(self, equity: float) -> None:
        """Update peak equity for drawdown calculation."""
        self.peak_equity = max(self.peak_equity, equity)

    def check_daily_loss(self, current_equity: float, starting_equity: float) -> bool:
        """Check if daily loss limit exceeded with robust validation."""
        # Validate inputs
        if not self._validate_equity_inputs(current_equity, starting_equity):
            self.trigger("Invalid equity data for daily loss check", "Data validation failed")
            return True  # Fail safe - block trading on invalid data

        # Validate configuration
        if not self._validate_daily_loss_config():
            self.trigger("Invalid daily loss configuration", "Config validation failed")
            return True  # Fail safe

        daily_loss_pct = (starting_equity - current_equity) / starting_equity

        if daily_loss_pct > self.max_daily_loss_pct:
            self.trigger("Daily loss limit exceeded", f"{daily_loss_pct:.2%}")
            return True
        return False

    def check_drawdown(self, current_equity: float) -> bool:
        """Check if drawdown limit exceeded."""
        if self.peak_equity <= 0:
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown > self.max_drawdown_pct:
            self.trigger("Maximum drawdown exceeded", f"{drawdown:.2%}")
            return True
        return False

    def check_consecutive_losses(self, trade_result: float) -> bool:
        """Check consecutive loss limit with robust validation."""
        # Validate trade result
        if not isinstance(trade_result, (int, float)) or not np.isfinite(trade_result):
            self.trigger("Invalid trade result data", f"Got {type(trade_result)}: {trade_result}")
            return True  # Fail safe

        # Validate configuration
        if not isinstance(self.max_consecutive_losses, int) or self.max_consecutive_losses <= 0:
            self.trigger("Invalid consecutive loss configuration", f"max_consecutive_losses: {self.max_consecutive_losses}")
            return True  # Fail safe

        if trade_result < 0:
            self.consecutive_losses += 1

            if self.consecutive_losses >= self.max_consecutive_losses:
                self.trigger(
                    "Consecutive loss limit exceeded", f"{self.consecutive_losses} losses in a row"
                )
                return True
        else:
            self.consecutive_losses = 0

        return False

    def check_position_loss(self, symbol: str, current_price: float) -> bool:
        """Check if any position exceeds loss limit with robust validation."""
        # Validate inputs
        if not isinstance(symbol, str) or not symbol.strip():
            self.trigger("Invalid symbol for position loss check", f"Symbol: {symbol}")
            return True  # Fail safe

        if not isinstance(current_price, (int, float)) or current_price <= 0 or not np.isfinite(current_price):
            self.trigger(f"Invalid price for {symbol}", f"Price: {current_price}")
            return True  # Fail safe

        # Validate configuration
        if not isinstance(self.max_position_loss_pct, (int, float)) or self.max_position_loss_pct <= 0:
            self.trigger("Invalid position loss configuration", f"max_position_loss_pct: {self.max_position_loss_pct}")
            return True  # Fail safe

        if symbol not in self.position_entry:
            return False

        entry_price, entry_time = self.position_entry[symbol]

        # Validate entry price
        if not isinstance(entry_price, (int, float)) or entry_price <= 0 or not np.isfinite(entry_price):
            self.trigger(f"Invalid entry price for {symbol}", f"Entry: {entry_price}")
            return True  # Fail safe

        loss_pct = (entry_price - current_price) / entry_price

        if loss_pct > self.max_position_loss_pct:
            self.trigger(f"Position loss limit exceeded for {symbol}", f"{loss_pct:.2%} loss")
            return True

        return False

    def trigger(self, reason: str, details: str = "") -> None:
        """Trigger the kill switch."""
        self.triggered = True
        self.trigger_time = get_market_time()
        self.trigger_reason = f"{reason}: {details}" if details else reason

        # Log critical event
        print(f"🚨 KILL SWITCH TRIGGERED: {self.trigger_reason}")

    def reset(self) -> None:
        """Reset kill switch after cooldown period."""
        if not self.triggered:
            return

        if self.trigger_time:
            elapsed = get_market_time() - self.trigger_time
            if elapsed > timedelta(minutes=self.cooldown_minutes):
                self.triggered = False
                self.trigger_time = None
                self.trigger_reason = None
                self.consecutive_losses = 0
                print(f"✅ Kill switch reset after {self.cooldown_minutes} minute cooldown")

    def is_active(self) -> bool:
        """Check if kill switch is currently active."""
        return self.triggered

    def can_trade(self) -> Tuple[bool, Optional[str]]:
        """Check if trading is allowed."""
        if self.triggered:
            return False, self.trigger_reason
        return True, None

    def _validate_equity_inputs(self, current_equity: float, starting_equity: float) -> bool:
        """
        Validate equity inputs for daily loss calculation.

        Args:
            current_equity: Current account equity
            starting_equity: Starting account equity

        Returns:
            True if inputs are valid
        """
        # Check types and basic validity
        if not isinstance(current_equity, (int, float)) or not isinstance(starting_equity, (int, float)):
            return False

        # Check for negative or zero starting equity
        if starting_equity <= 0:
            return False

        # Check for negative current equity (unusual but possible)
        if current_equity < 0:
            return False

        # Check for infinite or NaN values
        if not np.isfinite(current_equity) or not np.isfinite(starting_equity):
            return False

        # Sanity check: current equity shouldn't be more than 10x starting (prevents manipulation)
        if current_equity > starting_equity * 10:
            return False

        return True

    def _validate_daily_loss_config(self) -> bool:
        """
        Validate daily loss configuration.

        Returns:
            True if configuration is valid
        """
        if not isinstance(self.max_daily_loss_pct, (int, float)):
            return False

        if self.max_daily_loss_pct <= 0:
            return False

        # Reasonable limit check (prevent extreme values)
        if self.max_daily_loss_pct > 0.50:  # 50% daily loss seems unreasonable
            return False

        if not np.isfinite(self.max_daily_loss_pct):
            return False

        return True


class AdvancedRiskManager:
    """Advanced risk management system with Kelly sizing and kill switches."""

    def __init__(
        self,
        config: dict,
        enable_kelly: bool = True,
        enable_correlation_limits: bool = True,
        enable_kill_switch: bool = True,
    ):
        self.config = config
        self.enable_kelly = enable_kelly
        self.enable_correlation_limits = enable_correlation_limits
        self.enable_kill_switch = enable_kill_switch

        # Initialize components
        self.kelly_sizer = KellySizer() if enable_kelly else None
        self.correlation_limiter = CorrelationLimiter() if enable_correlation_limits else None
        self.kill_switch = KillSwitch() if enable_kill_switch else None

        # Risk metrics tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_metrics: Optional[RiskMetrics] = None

        # Position and PnL tracking
        self.positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.starting_capital = config.get("starting_capital", 100000)
        self.current_capital = self.starting_capital

    async def calculate_position_size(
        self, symbol: str, signal_strength: float, current_price: float, atr: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Calculate position size using Kelly criterion and risk limits.

        Returns:
            Dict with position_size, risk_metrics, and warnings
        """
        result = {
            "position_size": 0,
            "position_value": 0,
            "risk_per_trade": 0,
            "kelly_fraction": 0,
            "correlation_penalty": 1.0,
            "warnings": [],
            "blocked": False,
            "block_reason": None,
        }

        # Check kill switch first
        if self.kill_switch:
            can_trade, reason = self.kill_switch.can_trade()
            if not can_trade:
                result["blocked"] = True
                result["block_reason"] = reason
                return result

        # Calculate base position size using Kelly
        if self.kelly_sizer:
            kelly_params = self.kelly_sizer.calculate_kelly(symbol)
            base_position_value = self.kelly_sizer.get_position_size(self.current_capital, symbol)
            result["kelly_fraction"] = (
                kelly_params.half_kelly
                if self.kelly_sizer.use_half_kelly
                else kelly_params.kelly_fraction
            )

            # Add Kelly metrics to result
            result["kelly_metrics"] = {
                "win_rate": kelly_params.win_rate,
                "edge": kelly_params.edge,
                "odds": kelly_params.odds,
            }
        else:
            # Fallback to fixed fraction
            base_position_value = self.current_capital * 0.02
            result["kelly_fraction"] = 0.02

        # Apply signal strength scaling
        base_position_value *= signal_strength

        # Check correlation limits
        if self.correlation_limiter:
            current_exposures = {
                s: p["value"] / self.current_capital for s, p in self.positions.items()
            }

            allowed, max_corr, correlated = self.correlation_limiter.check_correlation_limit(
                symbol, current_exposures
            )

            if not allowed:
                result["warnings"].append(f"High correlation ({max_corr:.2f}) with {correlated}")

                # Apply correlation penalty
                penalty = self.correlation_limiter.get_correlation_penalty(
                    symbol, current_exposures
                )
                base_position_value *= penalty
                result["correlation_penalty"] = penalty

        # Apply maximum position limits
        max_position_value = self.current_capital * self.config.get("max_position_pct", 0.1)
        position_value = min(base_position_value, max_position_value)

        # Calculate shares
        position_size = int(position_value / current_price)

        # Calculate risk metrics
        if atr:
            stop_distance = atr * 2
            risk_per_share = stop_distance
            risk_per_trade = position_size * risk_per_share
            result["risk_per_trade"] = risk_per_trade
            result["stop_loss"] = current_price - stop_distance

            # Check if risk exceeds limits
            max_risk = self.current_capital * self.config.get("max_risk_per_trade", 0.02)
            if risk_per_trade > max_risk:
                # Reduce position size to meet risk limit
                position_size = int(max_risk / risk_per_share)
                position_value = position_size * current_price
                result["warnings"].append(f"Position reduced to meet risk limit")

        result["position_size"] = position_size
        result["position_value"] = position_value

        return result

    def update_position(self, symbol: str, quantity: int, price: float, side: str) -> None:
        """Update position tracking."""
        if side.upper() in ["BUY", "BUY_TO_COVER"]:
            if symbol in self.positions:
                # Update existing position
                pos = self.positions[symbol]
                total_qty = pos["quantity"] + quantity
                if total_qty != 0:
                    new_avg = (pos["avg_price"] * pos["quantity"] + price * quantity) / total_qty
                    self.positions[symbol] = {
                        "quantity": total_qty,
                        "avg_price": new_avg,
                        "value": total_qty * price,
                        "entry_time": pos.get("entry_time", get_market_time()),
                    }
                else:
                    del self.positions[symbol]
            else:
                # New position
                self.positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": price,
                    "value": quantity * price,
                    "entry_time": get_market_time(),
                }

                # Track for kill switch
                if self.kill_switch:
                    self.kill_switch.position_entry[symbol] = (price, get_market_time())

        elif side.upper() in ["SELL", "SELL_SHORT"]:
            if symbol in self.positions:
                pos = self.positions[symbol]

                # Calculate PnL
                pnl = (price - pos["avg_price"]) * min(quantity, pos["quantity"])
                self.daily_pnl += pnl
                self.total_pnl += pnl

                # Update Kelly sizer with trade result
                if self.kelly_sizer:
                    self.kelly_sizer.add_trade(symbol, pnl, pos["avg_price"])

                # Check kill switch conditions
                if self.kill_switch:
                    self.kill_switch.check_consecutive_losses(pnl)

                # Update or remove position
                remaining = pos["quantity"] - quantity
                if remaining > 0:
                    self.positions[symbol]["quantity"] = remaining
                    self.positions[symbol]["value"] = remaining * price
                else:
                    del self.positions[symbol]
                    if self.kill_switch and symbol in self.kill_switch.position_entry:
                        del self.kill_switch.position_entry[symbol]

    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """Update positions with current market prices."""
        for symbol, price in prices.items():
            # Update correlation limiter
            if self.correlation_limiter:
                self.correlation_limiter.update_price(symbol, price)

            # Check position stop losses
            if self.kill_switch and symbol in self.positions:
                self.kill_switch.check_position_loss(symbol, price)

            # Update position values
            if symbol in self.positions:
                self.positions[symbol]["value"] = self.positions[symbol]["quantity"] * price

    def calculate_current_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics."""
        total_exposure = sum(abs(p["value"]) for p in self.positions.values())

        # Calculate leverage
        leverage = total_exposure / self.current_capital if self.current_capital > 0 else 0

        # Simple VaR calculation (would be more sophisticated with full returns history)
        position_values = [p["value"] for p in self.positions.values()]
        if NUMPY_AVAILABLE and position_values:
            var_95 = np.percentile(position_values, 5) if len(position_values) > 1 else 0
            cvar_95 = np.mean([v for v in position_values if v <= var_95]) if var_95 else 0
        else:
            var_95 = 0
            cvar_95 = 0

        # Calculate other metrics
        sharpe = self._calculate_sharpe_ratio()
        max_dd = self._calculate_max_drawdown()
        conc_risk = self._calculate_concentration_risk()

        metrics = RiskMetrics(
            total_exposure=total_exposure,
            leverage=leverage,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            correlation_risk=0.0,  # Would need correlation matrix
            concentration_risk=conc_risk,
            liquidity_risk=0.0,  # Would need volume data
        )

        self.current_metrics = metrics
        self.metrics_history.append(metrics)

        return metrics

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns history."""
        if len(self.metrics_history) < 20:
            return 0.0

        if NUMPY_AVAILABLE:
            returns = []
            for i in range(1, len(self.metrics_history)):
                prev_equity = self.starting_capital + sum(
                    p["value"] - p["quantity"] * p["avg_price"] for p in self.positions.values()
                )
                curr_equity = self.current_capital
                returns.append((curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0)

            if returns:
                return (
                    np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                )

        return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if self.kill_switch and self.kill_switch.peak_equity > 0:
            return (
                self.kill_switch.peak_equity - self.current_capital
            ) / self.kill_switch.peak_equity
        return 0.0

    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk (Herfindahl index)."""
        if not self.positions:
            return 0.0

        total_value = sum(abs(p["value"]) for p in self.positions.values())
        if total_value == 0:
            return 0.0

        # Calculate Herfindahl index
        herfindahl = sum((abs(p["value"]) / total_value) ** 2 for p in self.positions.values())

        return herfindahl

    def get_risk_dashboard(self) -> Dict:
        """Get comprehensive risk dashboard data."""
        metrics = self.calculate_current_metrics()

        dashboard = {
            "current_metrics": {
                "total_exposure": metrics.total_exposure,
                "leverage": metrics.leverage,
                "var_95": metrics.var_95,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "concentration_risk": metrics.concentration_risk,
            },
            "positions": {
                symbol: {
                    "quantity": pos["quantity"],
                    "value": pos["value"],
                    "pnl": pos["value"] - pos["quantity"] * pos["avg_price"],
                    "pnl_pct": (pos["value"] - pos["quantity"] * pos["avg_price"])
                    / (pos["quantity"] * pos["avg_price"])
                    if pos["quantity"] * pos["avg_price"] != 0
                    else 0,
                }
                for symbol, pos in self.positions.items()
            },
            "kelly_parameters": {},
            "kill_switch": {
                "active": self.kill_switch.is_active() if self.kill_switch else False,
                "reason": self.kill_switch.trigger_reason if self.kill_switch else None,
                "consecutive_losses": self.kill_switch.consecutive_losses
                if self.kill_switch
                else 0,
            },
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
        }

        # Add Kelly parameters if available
        if self.kelly_sizer:
            for symbol in list(self.positions.keys())[:5]:  # Top 5 positions
                params = self.kelly_sizer.calculate_kelly(symbol)
                dashboard["kelly_parameters"][symbol] = {
                    "kelly_fraction": params.kelly_fraction,
                    "win_rate": params.win_rate,
                    "edge": params.edge,
                }

        return dashboard

    def save_state(self, filepath: Path) -> None:
        """Save risk manager state to file."""
        state = {
            "positions": self.positions,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "current_capital": self.current_capital,
            "kill_switch_triggered": self.kill_switch.triggered if self.kill_switch else False,
            "kelly_trades": list(self.kelly_sizer.trade_history) if self.kelly_sizer else [],
            "timestamp": get_market_time().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, filepath: Path) -> None:
        """Load risk manager state from file."""
        if not filepath.exists():
            return

        with open(filepath, "r") as f:
            state = json.load(f)

        self.positions = state.get("positions", {})
        self.daily_pnl = state.get("daily_pnl", 0)
        self.total_pnl = state.get("total_pnl", 0)
        self.current_capital = state.get("current_capital", self.starting_capital)

        # Restore Kelly trade history
        if self.kelly_sizer and "kelly_trades" in state:
            for trade in state["kelly_trades"]:
                self.kelly_sizer.trade_history.append(trade)

        # Restore kill switch state
        if self.kill_switch and state.get("kill_switch_triggered"):
            self.kill_switch.triggered = True
            self.kill_switch.trigger_time = datetime.fromisoformat(state["timestamp"])


# Async monitoring task
async def risk_monitor_task(risk_manager: AdvancedRiskManager, interval: int = 60):
    """Async task to continuously monitor risk metrics."""
    while True:
        try:
            metrics = risk_manager.calculate_current_metrics()

            # Check risk levels
            risk_level = RiskLevel.LOW

            if metrics.leverage > 2:
                risk_level = RiskLevel.HIGH
            elif metrics.leverage > 1.5:
                risk_level = RiskLevel.MEDIUM

            if metrics.max_drawdown > 0.08:
                risk_level = RiskLevel.CRITICAL
            elif metrics.max_drawdown > 0.05:
                risk_level = RiskLevel.HIGH

            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                print(
                    f"⚠️ Risk Alert: {risk_level.value.upper()} - Leverage: {metrics.leverage:.2f}, Drawdown: {metrics.max_drawdown:.2%}"
                )

            # Reset kill switch if cooldown expired
            if risk_manager.kill_switch:
                risk_manager.kill_switch.reset()

            # Update correlations periodically
            if risk_manager.correlation_limiter:
                risk_manager.correlation_limiter.calculate_correlations()

        except Exception as e:
            print(f"Error in risk monitor: {e}")

        await asyncio.sleep(interval)
