"""
Kelly Criterion Position Sizing Module.

Implements optimal position sizing based on the Kelly criterion formula
with safety modifications for practical trading.
"""

import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy/Pandas not available. Using simplified Kelly calculations.")


@dataclass
class KellyResult:
    """Result from Kelly criterion calculation."""

    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    recommended: float
    win_probability: float
    win_loss_ratio: float
    expected_value: float
    sample_size: int
    confidence: float


class OptimalKellySizer:
    """
    Advanced Kelly criterion calculator with multiple safety features.

    Features:
    - Dynamic Kelly fraction based on confidence
    - Regime-aware adjustments
    - Volatility scaling
    - Maximum position limits
    """

    def __init__(
        self,
        min_sample_size: int = 30,
        max_kelly: float = 0.25,
        confidence_threshold: float = 0.6,
        use_fractional_kelly: float = 0.5,  # Default to half-Kelly
        volatility_lookback: int = 20,
        regime_lookback: int = 60,
    ):
        self.min_sample_size = min_sample_size
        self.max_kelly = max_kelly
        self.confidence_threshold = confidence_threshold
        self.use_fractional_kelly = use_fractional_kelly
        self.volatility_lookback = volatility_lookback
        self.regime_lookback = regime_lookback

        # Trade history by symbol and strategy
        self.trade_history: Dict[str, deque] = {}
        self.strategy_history: Dict[str, deque] = {}

        # Cached calculations
        self.kelly_cache: Dict[str, KellyResult] = {}
        self.last_calculation: Dict[str, datetime] = {}

        # Market regime tracking
        self.market_returns: deque = deque(maxlen=regime_lookback)
        self.current_regime: str = "normal"

    def add_trade_result(
        self,
        symbol: str,
        pnl_pct: float,
        strategy: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a trade result for Kelly calculation."""
        timestamp = timestamp or datetime.now()

        trade = {"pnl_pct": pnl_pct, "timestamp": timestamp, "strategy": strategy}

        # Add to symbol history
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=200)
        self.trade_history[symbol].append(trade)

        # Add to strategy history
        if strategy:
            if strategy not in self.strategy_history:
                self.strategy_history[strategy] = deque(maxlen=200)
            self.strategy_history[strategy].append(trade)

    def calculate_kelly(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        override_confidence: Optional[float] = None,
    ) -> KellyResult:
        """
        Calculate Kelly fraction with advanced features.

        Args:
            symbol: Specific symbol to calculate for
            strategy: Specific strategy to calculate for
            override_confidence: Override the calculated confidence

        Returns:
            KellyResult with various Kelly fractions and metrics
        """
        # Get relevant trade history
        if symbol and symbol in self.trade_history:
            trades = list(self.trade_history[symbol])
        elif strategy and strategy in self.strategy_history:
            trades = list(self.strategy_history[strategy])
        else:
            # Use all trades
            all_trades = []
            for history in self.trade_history.values():
                all_trades.extend(history)
            trades = all_trades

        # Check sample size
        if len(trades) < self.min_sample_size:
            return self._default_kelly_result(len(trades))

        # Calculate win/loss statistics
        wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
        losses = [t["pnl_pct"] for t in trades if t["pnl_pct"] < 0]

        if not wins or not losses:
            return self._default_kelly_result(len(trades))

        # Basic Kelly inputs
        win_prob = len(wins) / len(trades)
        avg_win = np.mean(wins) if NUMPY_AVAILABLE else sum(wins) / len(wins)
        avg_loss = abs(np.mean(losses)) if NUMPY_AVAILABLE else abs(sum(losses) / len(losses))
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Calculate raw Kelly fraction
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        if win_loss_ratio > 0:
            kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        else:
            kelly = 0

        # Calculate confidence based on sample size and consistency
        confidence = self._calculate_confidence(trades, win_prob, override_confidence)

        # Apply regime adjustments
        regime_multiplier = self._get_regime_multiplier()

        # Apply volatility scaling
        vol_scalar = self._calculate_volatility_scalar(trades)

        # Calculate adjusted Kelly
        adjusted_kelly = kelly * confidence * regime_multiplier * vol_scalar

        # Apply maximum limit
        full_kelly = min(max(0, adjusted_kelly), self.max_kelly)

        # Calculate fractional Kelly variants
        half_kelly = full_kelly * 0.5
        quarter_kelly = full_kelly * 0.25

        # Determine recommended fraction
        if confidence >= self.confidence_threshold:
            recommended = full_kelly * self.use_fractional_kelly
        else:
            # Use more conservative fraction for low confidence
            recommended = quarter_kelly

        # Calculate expected value
        expected_value = win_prob * avg_win - (1 - win_prob) * avg_loss

        result = KellyResult(
            full_kelly=full_kelly,
            half_kelly=half_kelly,
            quarter_kelly=quarter_kelly,
            recommended=recommended,
            win_probability=win_prob,
            win_loss_ratio=win_loss_ratio,
            expected_value=expected_value,
            sample_size=len(trades),
            confidence=confidence,
        )

        # Cache result
        cache_key = f"{symbol or 'all'}_{strategy or 'all'}"
        self.kelly_cache[cache_key] = result
        self.last_calculation[cache_key] = datetime.now()

        return result

    def _calculate_confidence(
        self, trades: List[dict], win_prob: float, override: Optional[float] = None
    ) -> float:
        """Calculate confidence level for Kelly fraction."""
        if override is not None:
            return override

        # Base confidence on sample size
        sample_confidence = min(len(trades) / 100, 1.0)

        if NUMPY_AVAILABLE:
            # Calculate standard error of win rate
            se = np.sqrt(win_prob * (1 - win_prob) / len(trades))

            # Calculate consistency (lower variance is better)
            returns = [t["pnl_pct"] for t in trades]
            consistency = 1 / (1 + np.std(returns))

            # Check for statistical significance (simplified t-test)
            t_stat = (win_prob - 0.5) / se if se > 0 else 0
            stat_confidence = min(abs(t_stat) / 2, 1.0)

            # Combine confidence factors
            confidence = sample_confidence * 0.3 + consistency * 0.3 + stat_confidence * 0.4
        else:
            # Simplified confidence calculation
            confidence = sample_confidence * (0.5 + abs(win_prob - 0.5))

        return min(confidence, 1.0)

    def _get_regime_multiplier(self) -> float:
        """Get Kelly multiplier based on market regime."""
        if not self.market_returns or len(self.market_returns) < 20:
            return 1.0

        if NUMPY_AVAILABLE:
            returns = list(self.market_returns)
            vol = np.std(returns)
            trend = np.mean(returns)

            # Detect regime
            if vol > np.percentile([abs(r) for r in returns], 80):
                self.current_regime = "high_volatility"
                return 0.5  # Reduce Kelly in high volatility
            elif trend < np.percentile(returns, 20):
                self.current_regime = "bear"
                return 0.7  # Reduce Kelly in bear market
            elif trend > np.percentile(returns, 80):
                self.current_regime = "bull"
                return 1.1  # Slightly increase in strong bull
            else:
                self.current_regime = "normal"
                return 1.0

        return 1.0

    def _calculate_volatility_scalar(self, trades: List[dict]) -> float:
        """Calculate volatility-based scaling factor."""
        if len(trades) < self.volatility_lookback:
            return 1.0

        recent_trades = trades[-self.volatility_lookback :]
        recent_returns = [t["pnl_pct"] for t in recent_trades]

        if NUMPY_AVAILABLE:
            recent_vol = np.std(recent_returns)
            historical_vol = np.std([t["pnl_pct"] for t in trades])

            if historical_vol > 0:
                vol_ratio = recent_vol / historical_vol
                # Reduce Kelly if recent volatility is high
                if vol_ratio > 1.5:
                    return 0.7
                elif vol_ratio > 1.2:
                    return 0.85
                elif vol_ratio < 0.8:
                    return 1.1  # Increase slightly if volatility is low

        return 1.0

    def _default_kelly_result(self, sample_size: int) -> KellyResult:
        """Return conservative default Kelly result."""
        return KellyResult(
            full_kelly=0.02,
            half_kelly=0.01,
            quarter_kelly=0.005,
            recommended=0.01,
            win_probability=0.5,
            win_loss_ratio=1.0,
            expected_value=0,
            sample_size=sample_size,
            confidence=0.3,
        )

    def update_market_returns(self, market_return: float) -> None:
        """Update market returns for regime detection."""
        self.market_returns.append(market_return)

    def get_position_size(
        self,
        capital: float,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        max_position_value: Optional[float] = None,
    ) -> Tuple[float, KellyResult]:
        """
        Get recommended position size in dollars.

        Returns:
            Tuple of (position_value, kelly_result)
        """
        kelly_result = self.calculate_kelly(symbol, strategy)

        position_value = capital * kelly_result.recommended

        # Apply maximum position limit if specified
        if max_position_value:
            position_value = min(position_value, max_position_value)

        return position_value, kelly_result

    def get_portfolio_allocation(
        self, capital: float, symbols: List[str], max_positions: int = 10
    ) -> Dict[str, float]:
        """
        Get Kelly-based portfolio allocation across multiple symbols.

        Returns:
            Dict mapping symbols to position values
        """
        allocations = {}
        kelly_results = {}

        # Calculate Kelly for each symbol
        for symbol in symbols:
            result = self.calculate_kelly(symbol)
            if result.expected_value > 0:  # Only consider positive EV
                kelly_results[symbol] = result

        # Sort by expected value
        sorted_symbols = sorted(
            kelly_results.keys(), key=lambda s: kelly_results[s].expected_value, reverse=True
        )

        # Select top N symbols
        selected = sorted_symbols[:max_positions]

        # Calculate raw allocations
        total_kelly = sum(kelly_results[s].recommended for s in selected)

        if total_kelly > 1.0:
            # Scale down proportionally if total exceeds 100%
            scale_factor = 0.95 / total_kelly  # Leave 5% cash
        else:
            scale_factor = 1.0

        # Assign allocations
        for symbol in selected:
            allocation = kelly_results[symbol].recommended * scale_factor
            allocations[symbol] = capital * allocation

        return allocations

    def calculate_optimal_leverage(
        self, symbol: Optional[str] = None, strategy: Optional[str] = None
    ) -> float:
        """
        Calculate optimal leverage based on Kelly criterion.

        Returns:
            Optimal leverage ratio (1.0 = no leverage)
        """
        result = self.calculate_kelly(symbol, strategy)

        # Only use leverage if Kelly > 1 and confidence is high
        if result.full_kelly > 1.0 and result.confidence > 0.8:
            # Conservative leverage: never exceed 2x
            return min(result.full_kelly, 2.0)

        return 1.0

    def should_skip_trade(
        self, symbol: str, signal_confidence: float, min_edge: float = 0.01
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be skipped based on Kelly criterion.

        Returns:
            Tuple of (should_skip, reason)
        """
        result = self.calculate_kelly(symbol)

        # Skip if negative expected value
        if result.expected_value < 0:
            return True, f"Negative EV: {result.expected_value:.3f}"

        # Skip if expected value below minimum edge
        if result.expected_value < min_edge:
            return True, f"Insufficient edge: {result.expected_value:.3f} < {min_edge}"

        # Skip if Kelly fraction too small
        if result.recommended < 0.005:
            return True, f"Kelly too small: {result.recommended:.3f}"

        # Skip if confidence too low
        combined_confidence = result.confidence * signal_confidence
        if combined_confidence < 0.4:
            return True, f"Low confidence: {combined_confidence:.2f}"

        return False, "Trade acceptable"

    def get_dashboard_metrics(self) -> Dict:
        """Get Kelly metrics for dashboard display."""
        metrics = {
            "current_regime": self.current_regime,
            "symbol_kelly": {},
            "strategy_kelly": {},
            "portfolio_metrics": {
                "total_symbols": len(self.trade_history),
                "total_strategies": len(self.strategy_history),
                "avg_sample_size": 0,
            },
        }

        # Get Kelly for top symbols
        for symbol in list(self.trade_history.keys())[:10]:
            result = self.calculate_kelly(symbol)
            metrics["symbol_kelly"][symbol] = {
                "recommended": result.recommended,
                "win_rate": result.win_probability,
                "ev": result.expected_value,
                "confidence": result.confidence,
            }

        # Get Kelly for strategies
        for strategy in self.strategy_history.keys():
            result = self.calculate_kelly(strategy=strategy)
            metrics["strategy_kelly"][strategy] = {
                "recommended": result.recommended,
                "win_rate": result.win_probability,
                "ev": result.expected_value,
                "confidence": result.confidence,
            }

        # Calculate average sample size
        if self.trade_history:
            total_trades = sum(len(h) for h in self.trade_history.values())
            metrics["portfolio_metrics"]["avg_sample_size"] = total_trades / len(self.trade_history)

        return metrics

    def apply_risk_scaling(self, kelly_fraction: float, risk_scaling: float = 0.25) -> float:
        """Apply risk scaling to Kelly fraction for conservative sizing."""
        return kelly_fraction * risk_scaling
