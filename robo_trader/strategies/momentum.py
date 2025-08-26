"""
Enhanced momentum trading strategy.

This strategy uses multiple technical indicators to identify
strong momentum moves with proper risk management.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .framework import Strategy, Signal, SignalType
from ..features.engine import FeatureSet
from ..logger import get_logger

logger = get_logger(__name__)


class EnhancedMomentumStrategy(Strategy):
    """
    Enhanced momentum strategy using technical indicators.

    Features:
    - RSI for overbought/oversold conditions
    - MACD for trend confirmation
    - Volume analysis for momentum validation
    - ATR-based stop loss and targets
    - Multiple timeframe analysis
    """

    def __init__(
        self,
        symbols: List[str],
        # RSI parameters
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        # MACD parameters
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # Volume parameters
        volume_ma_period: int = 20,
        volume_threshold: float = 1.5,
        # Risk parameters
        atr_multiplier: float = 2.0,
        risk_reward_min: float = 1.8,
        # Signal strength
        min_signal_strength: float = 0.6,
        **kwargs,
    ):
        """
        Initialize enhanced momentum strategy.

        Args:
            symbols: List of symbols to trade
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            volume_ma_period: Volume MA period
            volume_threshold: Volume spike threshold
            atr_multiplier: ATR multiplier for stops
            risk_reward_min: Minimum risk:reward ratio
            min_signal_strength: Minimum signal strength to trade
        """
        super().__init__(
            name="EnhancedMomentum",
            symbols=symbols,
            lookback_period=max(100, macd_slow * 2),
            **kwargs,
        )

        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # MACD parameters
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        # Volume parameters
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold

        # Risk parameters
        self.atr_multiplier = atr_multiplier
        self.risk_reward_min = risk_reward_min
        self.min_signal_strength = min_signal_strength

        # Internal state
        self.momentum_scores: Dict[str, float] = {}
        self.trend_strength: Dict[str, float] = {}

    async def _initialize(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Initialize strategy with historical data."""
        logger.info(
            "momentum.initializing",
            symbols=len(historical_data),
            lookback=self.lookback_period,
        )

        # Calculate initial momentum scores
        for symbol, data in historical_data.items():
            if len(data) >= self.min_data_points:
                self.momentum_scores[symbol] = 0.0
                self.trend_strength[symbol] = 0.0

    async def _generate_signals(
        self, market_data: Dict[str, pd.DataFrame], features: Dict[str, FeatureSet]
    ) -> List[Signal]:
        """Generate momentum-based trading signals."""
        signals = []

        for symbol in self.symbols:
            if symbol not in market_data or symbol not in features:
                continue

            data = market_data[symbol]
            feature_set = features[symbol]

            if len(data) < self.min_data_points:
                continue

            # Get latest values
            current_price = data["close"].iloc[-1]
            current_volume = data["volume"].iloc[-1] if "volume" in data else 0

            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(
                data, feature_set, current_volume
            )

            self.momentum_scores[symbol] = momentum_score

            # Generate signal based on momentum
            signal = self._evaluate_momentum_signal(
                symbol=symbol,
                momentum_score=momentum_score,
                feature_set=feature_set,
                current_price=current_price,
                data=data,
            )

            if signal and signal.strength >= self.min_signal_strength:
                signals.append(signal)

                logger.debug(
                    "momentum.signal",
                    symbol=symbol,
                    type=signal.signal_type.value,
                    strength=signal.strength,
                    momentum=momentum_score,
                )

        return signals

    def _calculate_momentum_score(
        self, data: pd.DataFrame, features: FeatureSet, current_volume: float
    ) -> float:
        """
        Calculate composite momentum score.

        Args:
            data: Price data
            features: Technical indicators
            current_volume: Current volume

        Returns:
            Momentum score between -1 and 1
        """
        score = 0.0
        weights = 0.0

        # RSI component (0.3 weight)
        if features.rsi is not None:
            rsi = features.rsi
            if rsi < self.rsi_oversold:
                rsi_score = (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                rsi_score = -(rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            else:
                rsi_score = (rsi - 50) / 50 * 0.5

            score += rsi_score * 0.3
            weights += 0.3

        # MACD component (0.3 weight)
        if features.macd_line is not None and features.macd_signal is not None:
            macd_diff = features.macd_line - features.macd_signal
            macd_score = np.clip(macd_diff / abs(features.macd_signal + 1e-10), -1, 1)
            score += macd_score * 0.3
            weights += 0.3

        # Price momentum (0.2 weight)
        if len(data) >= 20:
            returns_5d = (data["close"].iloc[-1] / data["close"].iloc[-5] - 1) * 100
            returns_20d = (data["close"].iloc[-1] / data["close"].iloc[-20] - 1) * 100
            price_momentum = (returns_5d * 0.7 + returns_20d * 0.3) / 10
            price_momentum = np.clip(price_momentum, -1, 1)
            score += price_momentum * 0.2
            weights += 0.2

        # Volume confirmation (0.2 weight)
        if "volume" in data.columns and len(data) >= self.volume_ma_period:
            volume_ma = data["volume"].rolling(self.volume_ma_period).mean().iloc[-1]
            if volume_ma > 0:
                volume_ratio = current_volume / volume_ma
                if volume_ratio > self.volume_threshold:
                    volume_score = min((volume_ratio - 1) / 2, 1)
                else:
                    volume_score = (volume_ratio - 1) * 0.5
                score += volume_score * 0.2
                weights += 0.2

        # Normalize score
        if weights > 0:
            score = score / weights

        return np.clip(score, -1, 1)

    def _evaluate_momentum_signal(
        self,
        symbol: str,
        momentum_score: float,
        feature_set: FeatureSet,
        current_price: float,
        data: pd.DataFrame,
    ) -> Optional[Signal]:
        """
        Evaluate if momentum warrants a trading signal.

        Args:
            symbol: Trading symbol
            momentum_score: Calculated momentum score
            feature_set: Technical indicators
            current_price: Current price
            data: Price history

        Returns:
            Signal if conditions met, None otherwise
        """
        # Check if we have a position
        has_position = symbol in self.state.positions

        # Strong bullish momentum
        if momentum_score > 0.4 and not has_position:
            # Calculate stops and targets using ATR
            atr = feature_set.atr if feature_set.atr else current_price * 0.02

            stop_loss = current_price - (atr * self.atr_multiplier)
            take_profit = current_price + (
                atr * self.atr_multiplier * self.risk_reward_min
            )

            # Check Bollinger Bands for entry timing
            entry_quality = 1.0
            if feature_set.bb_upper and feature_set.bb_lower:
                bb_position = (current_price - feature_set.bb_lower) / (
                    feature_set.bb_upper - feature_set.bb_lower
                )
                if bb_position > 0.8:  # Near upper band
                    entry_quality *= 0.7
                elif bb_position < 0.2:  # Near lower band
                    entry_quality *= 1.3

            signal_strength = min(abs(momentum_score) * entry_quality, 1.0)

            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=signal_strength,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=self.risk_reward_min,
                rationale=f"Strong momentum: {momentum_score:.2f}",
                metadata={
                    "momentum_score": momentum_score,
                    "rsi": feature_set.rsi,
                    "macd_histogram": feature_set.macd_histogram,
                    "volume_spike": self._check_volume_spike(data),
                },
            )

        # Strong bearish momentum (close position)
        elif momentum_score < -0.3 and has_position:
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=abs(momentum_score),
                entry_price=current_price,
                rationale=f"Momentum reversal: {momentum_score:.2f}",
                metadata={"momentum_score": momentum_score, "rsi": feature_set.rsi},
            )

        # Weak momentum - potential scale out
        elif has_position and -0.3 <= momentum_score <= 0.2:
            # Check if we should scale out
            position = self.state.positions[symbol]
            if "entry_momentum" in position:
                momentum_decay = position["entry_momentum"] - momentum_score
                if momentum_decay > 0.5:
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.SCALE_OUT,
                        strength=0.5,
                        entry_price=current_price,
                        rationale=f"Momentum decay: {momentum_decay:.2f}",
                        metadata={"momentum_score": momentum_score},
                    )

        return None

    def _check_volume_spike(self, data: pd.DataFrame) -> bool:
        """Check if there's a volume spike."""
        if "volume" not in data.columns or len(data) < self.volume_ma_period:
            return False

        current_volume = data["volume"].iloc[-1]
        volume_ma = data["volume"].rolling(self.volume_ma_period).mean().iloc[-1]

        return current_volume > volume_ma * self.volume_threshold

    def update_position(self, symbol: str, position: Optional[Dict[str, Any]]) -> None:
        """Update position tracking with momentum data."""
        super().update_position(symbol, position)

        if position and symbol in self.momentum_scores:
            # Store entry momentum for later comparison
            if "entry_momentum" not in position:
                position["entry_momentum"] = self.momentum_scores[symbol]
