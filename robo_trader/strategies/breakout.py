"""
Breakout trading strategy.

This strategy identifies and trades breakouts from
consolidation patterns with volume confirmation.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..features.engine import FeatureSet
from ..logger import get_logger
from .framework import Signal, SignalType, Strategy

logger = get_logger(__name__)


class BreakoutStrategy(Strategy):
    """
    Breakout strategy with volume confirmation.

    Features:
    - Support/resistance level detection
    - Consolidation pattern recognition
    - Volume surge confirmation
    - ATR-based volatility breakouts
    - False breakout filtering
    """

    def __init__(
        self,
        symbols: List[str],
        # Breakout parameters
        lookback_periods: int = 20,
        breakout_threshold: float = 0.02,  # 2% above resistance
        consolidation_periods: int = 10,
        volatility_threshold: float = 0.5,  # Low vol for consolidation
        # Volume parameters
        volume_surge_multiplier: float = 1.5,
        volume_ma_period: int = 20,
        # Confirmation parameters
        confirmation_bars: int = 2,
        retest_tolerance: float = 0.005,  # 0.5% tolerance for retest
        # Risk parameters
        stop_loss_pct: float = 0.03,  # 3% stop loss
        target_multiplier: float = 2.5,  # Risk:reward target
        false_breakout_threshold: float = 0.01,  # 1% pullback = false
        **kwargs,
    ):
        """
        Initialize breakout strategy.

        Args:
            symbols: Symbols to trade
            lookback_periods: Periods for support/resistance
            breakout_threshold: Minimum breakout percentage
            consolidation_periods: Min periods in consolidation
            volatility_threshold: Max volatility for consolidation
            volume_surge_multiplier: Volume confirmation multiplier
            volume_ma_period: Volume MA calculation period
            confirmation_bars: Bars to confirm breakout
            retest_tolerance: Price tolerance for retest
            stop_loss_pct: Stop loss percentage
            target_multiplier: Target distance multiplier
            false_breakout_threshold: Threshold for false breakout
        """
        super().__init__(
            name="Breakout",
            symbols=symbols,
            lookback_period=lookback_periods * 2,
            **kwargs,
        )

        self.lookback_periods = lookback_periods
        self.breakout_threshold = breakout_threshold
        self.consolidation_periods = consolidation_periods
        self.volatility_threshold = volatility_threshold

        self.volume_surge_multiplier = volume_surge_multiplier
        self.volume_ma_period = volume_ma_period

        self.confirmation_bars = confirmation_bars
        self.retest_tolerance = retest_tolerance

        self.stop_loss_pct = stop_loss_pct
        self.target_multiplier = target_multiplier
        self.false_breakout_threshold = false_breakout_threshold

        # Track breakout levels
        self.resistance_levels: Dict[str, float] = {}
        self.support_levels: Dict[str, float] = {}
        self.pending_breakouts: Dict[str, Dict[str, Any]] = {}
        self.consolidation_counts: Dict[str, int] = {}

    async def _initialize(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Initialize with historical support/resistance levels."""
        logger.info(
            "breakout.initializing",
            symbols=len(historical_data),
            lookback=self.lookback_periods,
        )

        for symbol, data in historical_data.items():
            if len(data) >= self.lookback_periods:
                # Calculate initial levels
                highs = data["high"].tail(self.lookback_periods)
                lows = data["low"].tail(self.lookback_periods)

                self.resistance_levels[symbol] = highs.max()
                self.support_levels[symbol] = lows.min()
                self.consolidation_counts[symbol] = 0

    async def _generate_signals(
        self, market_data: Dict[str, pd.DataFrame], features: Dict[str, FeatureSet]
    ) -> List[Signal]:
        """Generate breakout trading signals."""
        signals = []

        for symbol in self.symbols:
            if symbol not in market_data or symbol not in features:
                continue

            data = market_data[symbol]
            feature_set = features[symbol]

            if len(data) < self.lookback_periods:
                continue

            # Update support/resistance levels
            self._update_levels(symbol, data)

            # Check for consolidation
            is_consolidating = self._detect_consolidation(data, feature_set)

            if is_consolidating:
                self.consolidation_counts[symbol] += 1
            else:
                self.consolidation_counts[symbol] = 0

            # Check pending breakouts
            signal = self._check_pending_breakout(symbol, data, feature_set)
            if signal:
                signals.append(signal)
                continue

            # Look for new breakout opportunities
            signal = self._detect_breakout(
                symbol=symbol,
                data=data,
                feature_set=feature_set,
                consolidation_count=self.consolidation_counts[symbol],
            )

            if signal:
                signals.append(signal)

                logger.debug(
                    "breakout.signal",
                    symbol=symbol,
                    type=signal.signal_type.value,
                    level=signal.metadata.get("breakout_level"),
                    strength=signal.strength,
                )

        return signals

    def _update_levels(self, symbol: str, data: pd.DataFrame) -> None:
        """Update support and resistance levels."""
        recent_data = data.tail(self.lookback_periods)

        # Rolling high/low
        self.resistance_levels[symbol] = recent_data["high"].max()
        self.support_levels[symbol] = recent_data["low"].min()

        # Look for pivot points
        pivots = self._find_pivot_points(data)
        if pivots:
            # Update with strongest pivot levels
            resistance_pivots = [p for p in pivots if p > data["close"].iloc[-1]]
            support_pivots = [p for p in pivots if p < data["close"].iloc[-1]]

            if resistance_pivots:
                self.resistance_levels[symbol] = min(resistance_pivots)
            if support_pivots:
                self.support_levels[symbol] = max(support_pivots)

    def _find_pivot_points(
        self, data: pd.DataFrame, min_touches: int = 2
    ) -> List[float]:
        """Find price levels with multiple touches."""
        if len(data) < 50:
            return []

        pivots = []
        price_levels = np.concatenate([data["high"].values, data["low"].values])

        # Cluster nearby prices
        tolerance = data["close"].iloc[-1] * 0.005  # 0.5% tolerance

        for level in np.unique(np.round(price_levels / tolerance) * tolerance):
            touches = np.sum(np.abs(price_levels - level) < tolerance)
            if touches >= min_touches:
                pivots.append(level)

        return pivots

    def _detect_consolidation(self, data: pd.DataFrame, features: FeatureSet) -> bool:
        """Detect if price is consolidating."""
        if len(data) < self.consolidation_periods:
            return False

        recent_data = data.tail(self.consolidation_periods)

        # Check volatility
        if features.atr and recent_data["close"].iloc[-1] > 0:
            atr_pct = features.atr / recent_data["close"].iloc[-1]
            if atr_pct > self.volatility_threshold * 0.01:
                return False

        # Check price range
        high_low_range = (
            recent_data["high"].max() - recent_data["low"].min()
        ) / recent_data["close"].mean()

        # Check Bollinger Band squeeze
        squeeze = False
        if features.bb_upper and features.bb_lower and features.bb_middle:
            bb_width = (features.bb_upper - features.bb_lower) / features.bb_middle
            squeeze = bb_width < 0.04  # Less than 4% width

        return high_low_range < 0.05 or squeeze  # Less than 5% range

    def _detect_breakout(
        self,
        symbol: str,
        data: pd.DataFrame,
        feature_set: FeatureSet,
        consolidation_count: int,
    ) -> Optional[Signal]:
        """Detect breakout opportunities."""
        current_price = data["close"].iloc[-1]
        current_high = data["high"].iloc[-1]
        current_low = data["low"].iloc[-1]
        has_position = symbol in self.state.positions

        if has_position:
            return None

        # Need consolidation period first
        if consolidation_count < self.consolidation_periods // 2:
            return None

        # Check for resistance breakout
        resistance = self.resistance_levels.get(symbol, float("inf"))
        if current_high > resistance * (1 + self.breakout_threshold):
            # Check volume confirmation
            if self._check_volume_surge(data):
                # Add to pending breakouts for confirmation
                self.pending_breakouts[symbol] = {
                    "type": "resistance",
                    "level": resistance,
                    "breakout_bar": len(data) - 1,
                    "high": current_high,
                    "confirmation_needed": self.confirmation_bars,
                }

                return self._create_breakout_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    current_price=current_price,
                    breakout_level=resistance,
                    feature_set=feature_set,
                    volume_confirmed=True,
                )

        # Check for support breakdown (short)
        if self.enable_shorts:
            support = self.support_levels.get(symbol, 0)
            if current_low < support * (1 - self.breakout_threshold):
                if self._check_volume_surge(data):
                    self.pending_breakouts[symbol] = {
                        "type": "support",
                        "level": support,
                        "breakout_bar": len(data) - 1,
                        "low": current_low,
                        "confirmation_needed": self.confirmation_bars,
                    }

                    return self._create_breakout_signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        current_price=current_price,
                        breakout_level=support,
                        feature_set=feature_set,
                        volume_confirmed=True,
                    )

        return None

    def _check_pending_breakout(
        self, symbol: str, data: pd.DataFrame, feature_set: FeatureSet
    ) -> Optional[Signal]:
        """Check if pending breakout is confirmed or failed."""
        if symbol not in self.pending_breakouts:
            return None

        pending = self.pending_breakouts[symbol]
        current_price = data["close"].iloc[-1]
        bars_since = len(data) - 1 - pending["breakout_bar"]

        # Check for false breakout
        if pending["type"] == "resistance":
            if current_price < pending["level"] * (1 - self.false_breakout_threshold):
                # False breakout - remove pending
                del self.pending_breakouts[symbol]
                logger.debug(
                    "breakout.false_breakout", symbol=symbol, level=pending["level"]
                )
                return None
        else:  # Support breakdown
            if current_price > pending["level"] * (1 + self.false_breakout_threshold):
                del self.pending_breakouts[symbol]
                return None

        # Check if confirmed
        if bars_since >= pending["confirmation_needed"]:
            del self.pending_breakouts[symbol]
            # Breakout confirmed, but signal already sent

        return None

    def _check_volume_surge(self, data: pd.DataFrame) -> bool:
        """Check for volume surge confirmation."""
        if "volume" not in data.columns or len(data) < self.volume_ma_period:
            return True  # No volume data, assume confirmed

        current_volume = data["volume"].iloc[-1]
        volume_ma = data["volume"].rolling(self.volume_ma_period).mean().iloc[-1]

        return current_volume > volume_ma * self.volume_surge_multiplier

    def _create_breakout_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        current_price: float,
        breakout_level: float,
        feature_set: FeatureSet,
        volume_confirmed: bool,
    ) -> Signal:
        """Create breakout entry signal."""
        # Calculate stops and targets
        if signal_type == SignalType.BUY:
            stop_loss = breakout_level * (1 - self.stop_loss_pct)
            risk = current_price - stop_loss
            take_profit = current_price + (risk * self.target_multiplier)
        else:  # SELL/SHORT
            stop_loss = breakout_level * (1 + self.stop_loss_pct)
            risk = stop_loss - current_price
            take_profit = current_price - (risk * self.target_multiplier)

        # Calculate signal strength
        breakout_strength = abs(current_price - breakout_level) / breakout_level
        volume_strength = 1.0 if volume_confirmed else 0.7
        atr_strength = 1.0

        if feature_set.atr:
            # Stronger signal if breaking out with increasing volatility
            atr_change = feature_set.atr / current_price
            atr_strength = min(1.0 + atr_change * 10, 1.5)

        signal_strength = min(
            breakout_strength * volume_strength * atr_strength * 10, 1.0
        )

        return Signal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_type,
            strength=signal_strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=self.target_multiplier,
            rationale=f"Breakout from {breakout_level:.2f}",
            metadata={
                "breakout_level": breakout_level,
                "breakout_type": (
                    "resistance" if signal_type == SignalType.BUY else "support"
                ),
                "volume_confirmed": volume_confirmed,
                "consolidation_bars": self.consolidation_counts.get(symbol, 0),
                "atr": feature_set.atr,
            },
        )
