"""
Strategy Manager for multi-strategy trading.

Combines signals from multiple strategies using voting or weighted prioritization.
Includes correlation guards to prevent overlapping exposures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from robo_trader.logger import get_logger

logger = get_logger(__name__)


class SignalStrength(Enum):
    """Signal strength levels for strategy outputs."""

    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class StrategySignal:
    """Signal output from a strategy."""

    symbol: str
    strategy_name: str
    signal: SignalStrength
    confidence: float  # 0.0 to 1.0
    metadata: Optional[Dict] = None


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""

    name: str
    function: Callable[[pd.DataFrame], pd.DataFrame]
    weight: float = 1.0
    enabled: bool = True
    params: Optional[Dict] = None


class StrategyManager:
    """
    Manages multiple trading strategies and combines their signals.

    Features:
    - Register multiple strategies with weights
    - Combine signals using voting or weighted average
    - Apply correlation guards
    - Track strategy performance
    """

    def __init__(
        self, combination_method: str = "weighted", correlation_threshold: float = 0.7
    ):
        """
        Initialize the strategy manager.

        Args:
            combination_method: How to combine signals ("vote", "weighted", "priority")
            correlation_threshold: Maximum allowed correlation between positions
        """
        self.strategies: List[StrategyConfig] = []
        self.combination_method = combination_method
        self.correlation_threshold = correlation_threshold
        self.position_correlations: Dict[str, Dict[str, float]] = {}
        self.strategy_performance: Dict[str, Dict] = {}

    def register_strategy(self, config: StrategyConfig) -> None:
        """Register a new strategy."""
        if config.enabled:
            self.strategies.append(config)
            self.strategy_performance[config.name] = {
                "signals": 0,
                "successful": 0,
                "total_return": 0.0,
            }
            logger.info(f"Registered strategy: {config.name} (weight={config.weight})")

    def unregister_strategy(self, name: str) -> None:
        """Remove a strategy by name."""
        self.strategies = [s for s in self.strategies if s.name != name]
        logger.info(f"Unregistered strategy: {name}")

    def evaluate_symbol(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        """
        Evaluate all strategies for a symbol and combine signals.

        Args:
            symbol: The symbol to evaluate
            df: DataFrame with market data

        Returns:
            Combined strategy signal
        """
        if not self.strategies:
            return StrategySignal(symbol, "NONE", SignalStrength.NEUTRAL, 0.0)

        signals = []

        # Collect signals from all strategies
        for strategy in self.strategies:
            try:
                # Call strategy function with params if provided
                if strategy.params:
                    result = strategy.function(df, **strategy.params)
                else:
                    result = strategy.function(df)

                # Extract signal from result
                if not result.empty and "signal" in result.columns:
                    last_signal = result["signal"].iloc[-1]
                    confidence = (
                        result.get("confidence", pd.Series([1.0])).iloc[-1]
                        if "confidence" in result.columns
                        else 1.0
                    )

                    # Convert numeric signal to SignalStrength
                    if last_signal > 1.5:
                        strength = SignalStrength.STRONG_BUY
                    elif last_signal > 0.5:
                        strength = SignalStrength.BUY
                    elif last_signal < -1.5:
                        strength = SignalStrength.STRONG_SELL
                    elif last_signal < -0.5:
                        strength = SignalStrength.SELL
                    else:
                        strength = SignalStrength.NEUTRAL

                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy_name=strategy.name,
                            signal=strength,
                            confidence=confidence * strategy.weight,
                        )
                    )

                    # Track strategy activity
                    self.strategy_performance[strategy.name]["signals"] += 1

            except Exception as e:
                logger.error(f"Error in strategy {strategy.name} for {symbol}: {e}")
                continue

        if not signals:
            return StrategySignal(symbol, "NONE", SignalStrength.NEUTRAL, 0.0)

        # Combine signals based on method
        combined = self._combine_signals(signals)

        # Apply correlation guard
        if self._check_correlation_guard(symbol, combined):
            logger.info(
                f"Correlation guard triggered for {symbol}, neutralizing signal"
            )
            combined.signal = SignalStrength.NEUTRAL
            combined.confidence *= 0.5

        return combined

    def _combine_signals(self, signals: List[StrategySignal]) -> StrategySignal:
        """Combine multiple signals based on the configured method."""
        if not signals:
            return StrategySignal("", "COMBINED", SignalStrength.NEUTRAL, 0.0)

        symbol = signals[0].symbol

        if self.combination_method == "vote":
            # Simple majority voting
            votes = {}
            for signal in signals:
                votes[signal.signal] = votes.get(signal.signal, 0) + 1

            winning_signal = max(votes, key=votes.get)
            confidence = votes[winning_signal] / len(signals)

            return StrategySignal(
                symbol=symbol,
                strategy_name="VOTE_COMBINED",
                signal=winning_signal,
                confidence=confidence,
            )

        elif self.combination_method == "weighted":
            # Weighted average of signals
            weighted_sum = 0.0
            weight_total = 0.0

            for signal in signals:
                weighted_sum += signal.signal.value * signal.confidence
                weight_total += signal.confidence

            if weight_total == 0:
                return StrategySignal(
                    symbol, "WEIGHTED_COMBINED", SignalStrength.NEUTRAL, 0.0
                )

            avg_signal = weighted_sum / weight_total

            # Convert back to SignalStrength
            if avg_signal > 1.5:
                strength = SignalStrength.STRONG_BUY
            elif avg_signal > 0.5:
                strength = SignalStrength.BUY
            elif avg_signal < -1.5:
                strength = SignalStrength.STRONG_SELL
            elif avg_signal < -0.5:
                strength = SignalStrength.SELL
            else:
                strength = SignalStrength.NEUTRAL

            return StrategySignal(
                symbol=symbol,
                strategy_name="WEIGHTED_COMBINED",
                signal=strength,
                confidence=min(1.0, weight_total / len(signals)),
            )

        elif self.combination_method == "priority":
            # Use highest confidence signal
            best_signal = max(signals, key=lambda s: s.confidence)
            best_signal.strategy_name = f"PRIORITY_{best_signal.strategy_name}"
            return best_signal

        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _check_correlation_guard(self, symbol: str, signal: StrategySignal) -> bool:
        """
        Check if adding this position would violate correlation limits.

        Returns:
            True if correlation guard should block the trade
        """
        if signal.signal in [SignalStrength.NEUTRAL]:
            return False

        # Check correlations with existing positions
        if symbol in self.position_correlations:
            for other_symbol, correlation in self.position_correlations[symbol].items():
                if abs(correlation) > self.correlation_threshold:
                    logger.warning(
                        f"High correlation ({correlation:.2f}) between {symbol} and {other_symbol}"
                    )
                    return True

        return False

    def update_correlations(self, correlations: Dict[str, Dict[str, float]]) -> None:
        """Update the correlation matrix for position symbols."""
        self.position_correlations = correlations

    def record_trade_result(
        self,
        strategy_name: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        success: bool,
    ) -> None:
        """Record the result of a trade for performance tracking."""
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            if success:
                perf["successful"] += 1

            return_pct = (exit_price - entry_price) / entry_price
            perf["total_return"] += return_pct

            logger.debug(
                f"Recorded trade result for {strategy_name}: "
                f"{symbol} return={return_pct:.2%}"
            )

    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary for all strategies."""
        summary = {}
        for name, perf in self.strategy_performance.items():
            win_rate = (
                perf["successful"] / perf["signals"] if perf["signals"] > 0 else 0
            )
            avg_return = (
                perf["total_return"] / perf["signals"] if perf["signals"] > 0 else 0
            )

            summary[name] = {
                "signals": perf["signals"],
                "win_rate": win_rate,
                "avg_return": avg_return,
                "total_return": perf["total_return"],
            }

        return summary


def create_default_manager() -> StrategyManager:
    """Create a strategy manager with default strategies."""
    from robo_trader.strategies import sma_crossover_signals

    manager = StrategyManager(combination_method="weighted")

    # Register SMA crossover with different timeframes
    manager.register_strategy(
        StrategyConfig(
            name="SMA_FAST",
            function=sma_crossover_signals,
            weight=1.0,
            params={"fast": 5, "slow": 10},
        )
    )

    manager.register_strategy(
        StrategyConfig(
            name="SMA_MEDIUM",
            function=sma_crossover_signals,
            weight=0.8,
            params={"fast": 10, "slow": 20},
        )
    )

    manager.register_strategy(
        StrategyConfig(
            name="SMA_SLOW",
            function=sma_crossover_signals,
            weight=0.6,
            params={"fast": 20, "slow": 50},
        )
    )

    return manager
