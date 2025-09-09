"""
Mean reversion trading strategy.

This strategy identifies oversold/overbought conditions
and trades the reversion to the mean.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ..features.engine import FeatureSet
from ..logger import get_logger
from ..ml.model_registry import ModelRegistry
from ..ml.model_selector import ModelSelector
from .framework import Signal, SignalType, Strategy

logger = get_logger(__name__)


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI.

    Features:
    - Bollinger Bands for deviation from mean
    - RSI for extreme conditions
    - Z-score for statistical significance
    - Volume confirmation
    - Dynamic position sizing based on deviation
    """

    def __init__(
        self,
        symbols: List[str],
        # Bollinger Bands
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_squeeze_threshold: float = 0.05,
        # RSI
        rsi_period: int = 14,
        rsi_oversold: float = 25,
        rsi_overbought: float = 75,
        # Z-score
        zscore_period: int = 20,
        zscore_threshold: float = 2.0,
        # Mean reversion
        reversion_target: float = 0.5,  # Target % of move back to mean
        max_holding_periods: int = 20,
        # Risk
        max_deviation_entry: float = 3.0,  # Max std devs for entry
        stop_loss_multiplier: float = 1.5,
        # ML Enhancement
        use_ml_enhancement: bool = True,
        ml_confidence_threshold: float = 0.6,
        ml_model_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize mean reversion strategy.

        Args:
            symbols: Symbols to trade
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviations
            bb_squeeze_threshold: Threshold for volatility squeeze
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold level
            rsi_overbought: RSI overbought level
            zscore_period: Z-score lookback period
            zscore_threshold: Z-score entry threshold
            reversion_target: Target reversion percentage
            max_holding_periods: Maximum bars to hold position
            max_deviation_entry: Maximum std devs for entry
            stop_loss_multiplier: Stop loss distance multiplier
        """
        super().__init__(
            name="MeanReversion",
            symbols=symbols,
            lookback_period=max(bb_period, zscore_period) * 2,
            **kwargs,
        )

        # Bollinger Bands parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_squeeze_threshold = bb_squeeze_threshold

        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Z-score parameters
        self.zscore_period = zscore_period
        self.zscore_threshold = zscore_threshold

        # Mean reversion parameters
        self.reversion_target = reversion_target
        self.max_holding_periods = max_holding_periods
        self.max_deviation_entry = max_deviation_entry
        self.stop_loss_multiplier = stop_loss_multiplier

        # Track mean reversion opportunities
        self.reversion_scores: Dict[str, float] = {}
        self.entry_deviations: Dict[str, float] = {}
        self.holding_periods: Dict[str, int] = {}

        # ML Enhancement
        self.use_ml_enhancement = use_ml_enhancement
        self.ml_confidence_threshold = ml_confidence_threshold
        self.ml_model = None
        self.model_registry = ModelRegistry()
        self.model_selector = ModelSelector()

        # Load ML model if specified
        if use_ml_enhancement and ml_model_path:
            self._load_ml_model(ml_model_path)

    async def _initialize(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Initialize strategy with historical data."""
        logger.info(
            "mean_reversion.initializing",
            symbols=len(historical_data),
            bb_period=self.bb_period,
            ml_enabled=self.use_ml_enhancement,
        )

        for symbol in historical_data:
            self.reversion_scores[symbol] = 0.0
            self.holding_periods[symbol] = 0

        # Train ML model if needed and no pre-trained model
        if self.use_ml_enhancement and self.ml_model is None:
            await self._train_ml_model(historical_data)

    async def _generate_signals(
        self, market_data: Dict[str, pd.DataFrame], features: Dict[str, FeatureSet]
    ) -> List[Signal]:
        """Generate mean reversion trading signals."""
        signals = []

        for symbol in self.symbols:
            if symbol not in market_data or symbol not in features:
                continue

            data = market_data[symbol]
            feature_set = features[symbol]

            if len(data) < self.min_data_points:
                continue

            # Update holding period if in position
            if symbol in self.state.positions:
                self.holding_periods[symbol] += 1
            else:
                self.holding_periods[symbol] = 0

            # Calculate mean reversion opportunity
            reversion_score = self._calculate_reversion_score(data, feature_set)
            self.reversion_scores[symbol] = reversion_score

            # Generate signal with ML enhancement
            if self.use_ml_enhancement and self.ml_model:
                signal = await self._evaluate_ml_enhanced_signal(
                    symbol=symbol,
                    reversion_score=reversion_score,
                    feature_set=feature_set,
                    data=data,
                )
            else:
                signal = self._evaluate_reversion_signal(
                    symbol=symbol,
                    reversion_score=reversion_score,
                    feature_set=feature_set,
                    data=data,
                )

            if signal:
                signals.append(signal)

                logger.debug(
                    "mean_reversion.signal",
                    symbol=symbol,
                    type=signal.signal_type.value,
                    reversion_score=reversion_score,
                    strength=signal.strength,
                )

        return signals

    def _calculate_reversion_score(self, data: pd.DataFrame, features: FeatureSet) -> float:
        """
        Calculate mean reversion opportunity score.

        Args:
            data: Price data
            features: Technical indicators

        Returns:
            Reversion score between -1 (oversold) and 1 (overbought)
        """
        score = 0.0
        weights = 0.0

        # Bollinger Bands position (40% weight)
        if features.bb_upper and features.bb_lower and features.bb_middle:
            current_price = data["close"].iloc[-1]
            bb_width = features.bb_upper - features.bb_lower

            if bb_width > 0:
                # Position within bands (-1 to 1)
                bb_position = (current_price - features.bb_middle) / (bb_width / 2)
                bb_score = np.clip(bb_position, -1, 1)

                # Enhance score if at extremes
                if current_price <= features.bb_lower:
                    bb_score = -1.0 - (features.bb_lower - current_price) / features.bb_lower * 0.5
                elif current_price >= features.bb_upper:
                    bb_score = 1.0 + (current_price - features.bb_upper) / features.bb_upper * 0.5

                score += bb_score * 0.4
                weights += 0.4

                # Check for Bollinger squeeze
                typical_width = features.bb_middle * self.bb_std * 2 / 100
                if bb_width < typical_width * self.bb_squeeze_threshold:
                    # Reduce signal during squeeze
                    score *= 0.5

        # RSI extremes (30% weight)
        if features.rsi is not None:
            if features.rsi <= self.rsi_oversold:
                rsi_score = -1.0 * (self.rsi_oversold - features.rsi) / self.rsi_oversold
            elif features.rsi >= self.rsi_overbought:
                rsi_score = 1.0 * (features.rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            else:
                rsi_score = (features.rsi - 50) / 50 * 0.3

            score += rsi_score * 0.3
            weights += 0.3

        # Z-score calculation (30% weight)
        if len(data) >= self.zscore_period:
            closes = data["close"].tail(self.zscore_period)
            zscore = (closes.iloc[-1] - closes.mean()) / (closes.std() + 1e-10)
            zscore_norm = np.clip(zscore / self.zscore_threshold, -1, 1)

            score += zscore_norm * 0.3
            weights += 0.3

        # Normalize
        if weights > 0:
            score = score / weights

        return np.clip(score, -2, 2)  # Allow for extreme readings

    def _evaluate_reversion_signal(
        self,
        symbol: str,
        reversion_score: float,
        feature_set: FeatureSet,
        data: pd.DataFrame,
    ) -> Optional[Signal]:
        """
        Evaluate if conditions warrant a mean reversion trade.

        Args:
            symbol: Trading symbol
            reversion_score: Calculated reversion score
            feature_set: Technical indicators
            data: Price history

        Returns:
            Signal if conditions met
        """
        current_price = data["close"].iloc[-1]
        has_position = symbol in self.state.positions

        # Entry signals - looking for extremes
        if not has_position:
            # Oversold - potential long entry
            if reversion_score < -self.zscore_threshold / 2:
                return self._create_reversion_entry(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    current_price=current_price,
                    feature_set=feature_set,
                    reversion_score=reversion_score,
                    target_mean=feature_set.bb_middle
                    or data["close"].rolling(self.bb_period).mean().iloc[-1],
                )

            # Overbought - potential short entry (if enabled)
            elif reversion_score > self.zscore_threshold / 2 and self.enable_shorts:
                return self._create_reversion_entry(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    current_price=current_price,
                    feature_set=feature_set,
                    reversion_score=reversion_score,
                    target_mean=feature_set.bb_middle
                    or data["close"].rolling(self.bb_period).mean().iloc[-1],
                )

        # Exit signals for existing positions
        else:
            position = self.state.positions[symbol]
            holding_period = self.holding_periods[symbol]

            # Check if reversion target hit
            if feature_set.bb_middle:
                entry_deviation = self.entry_deviations.get(symbol, 0)
                current_deviation = (current_price - feature_set.bb_middle) / feature_set.bb_middle

                # Long position exit conditions
                if position.get("side") == "long":
                    if (
                        current_deviation >= 0  # Reached mean
                        or reversion_score > 0.5  # Becoming overbought
                        or holding_period >= self.max_holding_periods
                    ):  # Max holding time
                        return Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            strength=0.8,
                            entry_price=current_price,
                            rationale=f"Mean reversion target reached or timeout",
                            metadata={
                                "reversion_score": reversion_score,
                                "holding_period": holding_period,
                                "profit_pct": (
                                    current_price / position.get("entry_price", current_price) - 1
                                )
                                * 100,
                            },
                        )

                # Short position exit conditions
                elif position.get("side") == "short":
                    if (
                        current_deviation <= 0  # Reached mean
                        or reversion_score < -0.5  # Becoming oversold
                        or holding_period >= self.max_holding_periods
                    ):
                        return Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            signal_type=SignalType.CLOSE,
                            strength=0.8,
                            entry_price=current_price,
                            rationale=f"Mean reversion target reached or timeout",
                            metadata={
                                "reversion_score": reversion_score,
                                "holding_period": holding_period,
                            },
                        )

        return None

    def _create_reversion_entry(
        self,
        symbol: str,
        signal_type: SignalType,
        current_price: float,
        feature_set: FeatureSet,
        reversion_score: float,
        target_mean: float,
    ) -> Signal:
        """Create mean reversion entry signal."""
        # Calculate ATR for stops
        atr = feature_set.atr if feature_set.atr else current_price * 0.02

        if signal_type == SignalType.BUY:
            # Long entry
            stop_loss = current_price - (atr * self.stop_loss_multiplier)
            take_profit = target_mean * (1 - self.reversion_target * 0.01)
            deviation = (current_price - target_mean) / target_mean
        else:
            # Short entry
            stop_loss = current_price + (atr * self.stop_loss_multiplier)
            take_profit = target_mean * (1 + self.reversion_target * 0.01)
            deviation = (current_price - target_mean) / target_mean

        # Calculate risk:reward
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = reward / risk if risk > 0 else 0

        # Signal strength based on deviation extremity
        signal_strength = min(abs(reversion_score) / self.zscore_threshold, 1.0)

        # Store entry deviation
        self.entry_deviations[symbol] = deviation

        return Signal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_type,
            strength=signal_strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            rationale=f"Mean reversion opportunity: {reversion_score:.2f} std devs",
            metadata={
                "reversion_score": reversion_score,
                "target_mean": target_mean,
                "entry_deviation": deviation,
                "bb_width": (
                    (feature_set.bb_upper - feature_set.bb_lower) if feature_set.bb_upper else None
                ),
                "rsi": feature_set.rsi,
            },
        )

    def update_position(self, symbol: str, position: Optional[Dict[str, Any]]) -> None:
        """Update position tracking."""
        super().update_position(symbol, position)

        if not position:
            # Position closed, reset tracking
            self.holding_periods[symbol] = 0
            self.entry_deviations.pop(symbol, None)
        elif position and "side" not in position:
            # Store position side
            if position.get("quantity", 0) > 0:
                position["side"] = "long"
            else:
                position["side"] = "short"

    def _load_ml_model(self, model_path: str) -> None:
        """Load pre-trained ML model."""
        try:
            path = Path(model_path)
            if path.exists():
                self.ml_model = joblib.load(path)
                logger.info("mean_reversion.ml_model_loaded", path=model_path)
            else:
                logger.warning("mean_reversion.ml_model_not_found", path=model_path)
        except Exception as e:
            logger.error("mean_reversion.ml_model_load_error", error=str(e))
            self.ml_model = None

    async def _train_ml_model(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Train ML model for mean reversion prediction."""
        try:
            # Prepare training data
            X_train = []
            y_train = []

            for symbol, data in historical_data.items():
                if len(data) < self.lookback_period:
                    continue

                # Extract features for ML training
                features = self._extract_ml_features(data)
                labels = self._generate_ml_labels(data)

                if features and labels:
                    X_train.extend(features)
                    y_train.extend(labels)

            if len(X_train) > 100:  # Minimum samples for training
                # Use model selector to find best model
                import numpy as np

                X_train = np.array(X_train)
                y_train = np.array(y_train)

                # Train ensemble model
                from sklearn.ensemble import RandomForestClassifier

                self.ml_model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )
                self.ml_model.fit(X_train, y_train)

                logger.info(
                    "mean_reversion.ml_model_trained",
                    samples=len(X_train),
                    features=X_train.shape[1],
                )
        except Exception as e:
            logger.error("mean_reversion.ml_training_error", error=str(e))
            self.ml_model = None

    def _extract_ml_features(self, data: pd.DataFrame) -> List[List[float]]:
        """Extract ML features from price data."""
        features = []

        if len(data) < self.lookback_period:
            return features

        # Calculate rolling features
        for i in range(self.lookback_period, len(data)):
            window = data.iloc[i - self.lookback_period : i]

            # Price-based features
            returns = window["close"].pct_change().dropna()

            feature_vector = [
                # Statistical moments
                returns.mean(),
                returns.std(),
                returns.skew(),
                returns.kurtosis(),
                # Price levels
                (window["close"].iloc[-1] - window["close"].mean()) / window["close"].std(),
                (window["high"].iloc[-1] - window["low"].iloc[-1]) / window["close"].iloc[-1],
                # Volume features
                window["volume"].iloc[-1] / window["volume"].mean() if "volume" in window else 1.0,
                # Trend features
                (window["close"].iloc[-1] / window["close"].iloc[0]) - 1,
                (window["close"].iloc[-1] / window["close"].iloc[-5]) - 1
                if len(window) >= 5
                else 0,
            ]

            features.append(feature_vector)

        return features

    def _generate_ml_labels(self, data: pd.DataFrame, forward_periods: int = 5) -> List[int]:
        """Generate labels for ML training (1 for successful reversion, 0 otherwise)."""
        labels = []

        if len(data) < self.lookback_period + forward_periods:
            return labels

        for i in range(self.lookback_period, len(data) - forward_periods):
            # Check if price reverts to mean in next N periods
            current_price = data["close"].iloc[i]
            mean_price = data["close"].iloc[i - self.bb_period : i].mean()
            future_prices = data["close"].iloc[i + 1 : i + forward_periods + 1]

            # Label as 1 if price reverts toward mean
            if current_price > mean_price:
                # Price above mean, check if it goes down
                reverted = any(price <= mean_price for price in future_prices)
            else:
                # Price below mean, check if it goes up
                reverted = any(price >= mean_price for price in future_prices)

            labels.append(1 if reverted else 0)

        return labels

    async def _evaluate_ml_enhanced_signal(
        self,
        symbol: str,
        reversion_score: float,
        feature_set: FeatureSet,
        data: pd.DataFrame,
    ) -> Optional[Signal]:
        """Evaluate signal with ML enhancement."""
        # Get base signal
        base_signal = self._evaluate_reversion_signal(symbol, reversion_score, feature_set, data)

        if not base_signal or not self.ml_model:
            return base_signal

        try:
            # Extract current features
            current_features = self._extract_ml_features(data)
            if not current_features:
                return base_signal

            # Get ML prediction
            import numpy as np

            X = np.array([current_features[-1]])
            ml_probability = self.ml_model.predict_proba(X)[0, 1]

            # Enhance signal with ML confidence
            if ml_probability >= self.ml_confidence_threshold:
                # Boost signal strength
                base_signal.strength = min(base_signal.strength * (1 + ml_probability * 0.5), 1.0)
                base_signal.metadata["ml_confidence"] = ml_probability
                base_signal.metadata["ml_enhanced"] = True

                logger.debug(
                    "mean_reversion.ml_enhancement",
                    symbol=symbol,
                    ml_confidence=ml_probability,
                    enhanced_strength=base_signal.strength,
                )
            elif ml_probability < 0.3:
                # ML suggests avoiding trade
                logger.debug(
                    "mean_reversion.ml_rejection", symbol=symbol, ml_confidence=ml_probability
                )
                return None

        except Exception as e:
            logger.error("mean_reversion.ml_evaluation_error", error=str(e))

        return base_signal
