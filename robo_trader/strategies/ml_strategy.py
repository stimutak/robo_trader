"""ML-Driven Trading Strategy for RoboTrader.

This module implements a sophisticated ML-driven strategy that:
- Uses predictions from trained ML models
- Implements multi-timeframe analysis
- Incorporates regime detection for adaptive behavior
- Manages position sizing based on confidence and regime
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from ..features.feature_pipeline import FeaturePipeline
from ..ml.model_selector import ModelSelector
from ..ml.model_trainer import ModelType, PredictionType
from .framework import Signal, SignalType
from .framework import Strategy as BaseStrategy
from .framework import StrategyState

logger = structlog.get_logger(__name__)


class TimeFrame(Enum):
    """Trading timeframes for multi-timeframe analysis."""

    SCALP = "1min"  # 1-minute bars for scalping
    INTRADAY = "5min"  # 5-minute bars for intraday
    SWING = "30min"  # 30-minute bars for swing trades
    POSITION = "1day"  # Daily bars for position trades


@dataclass
class MLSignal(Signal):
    """Extended signal with ML-specific metadata."""

    ml_confidence: float = 0.0  # Model prediction confidence
    regime: str = "neutral"  # Market regime
    timeframe: TimeFrame = TimeFrame.INTRADAY
    feature_importance: Dict[str, float] = None
    ensemble_votes: Dict[str, float] = None


class MLStrategy(BaseStrategy):
    """ML-driven trading strategy with multi-timeframe analysis."""

    def __init__(
        self,
        model_selector: ModelSelector,
        feature_pipeline: FeaturePipeline,
        confidence_threshold: float = 0.65,
        ensemble_agreement: float = 0.6,
        use_regime_filter: bool = True,
        timeframes: List[TimeFrame] = None,
        position_size_method: str = "kelly",
        max_position_pct: float = 0.1,
        risk_per_trade: float = 0.02,
        symbols: List[str] = None,
        name: str = "MLStrategy",
    ):
        """Initialize ML strategy.

        Args:
            model_selector: Model selection and validation system
            feature_pipeline: Feature engineering pipeline
            confidence_threshold: Minimum confidence for signals
            ensemble_agreement: Minimum agreement among models
            use_regime_filter: Whether to use regime detection
            timeframes: List of timeframes to analyze
            position_size_method: Method for position sizing
            max_position_pct: Maximum position as % of portfolio
            risk_per_trade: Risk per trade as % of portfolio
            symbols: List of symbols to trade
            name: Strategy name
        """
        # Initialize base strategy
        super().__init__(
            name=name,
            symbols=symbols or [],
            lookback_period=100,
            min_data_points=50,
            position_sizing=position_size_method,
        )

        self.model_selector = model_selector
        self.feature_pipeline = feature_pipeline
        self.confidence_threshold = confidence_threshold
        self.ensemble_agreement = ensemble_agreement
        self.use_regime_filter = use_regime_filter
        self.timeframes = timeframes or [TimeFrame.INTRADAY, TimeFrame.SWING]
        self.position_size_method = position_size_method
        self.max_position_pct = max_position_pct
        self.risk_per_trade = risk_per_trade

        # Cache for ML predictions
        self.prediction_cache: Dict[str, Dict] = {}
        self.regime_cache: Dict[str, str] = {}
        self.feature_cache: Dict[str, pd.DataFrame] = {}

        # Performance metrics
        self.performance = {"total_pnl": 0, "win_rate": 0, "sharpe": 0}

    async def _initialize(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Strategy-specific initialization.

        Args:
            historical_data: Historical OHLCV data by symbol
        """
        try:
            # Load available models
            await self.model_selector.load_available_models()

            # Select best performing model
            best_model = await self.model_selector.select_best_model()
            if best_model:
                logger.info(
                    "Selected ML model",
                    model_type=best_model.get("model_type"),
                    test_score=best_model.get("metrics", {}).get("test_score"),
                )
            else:
                logger.warning("No suitable ML model found")

            self._is_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize ML strategy: {e}")
            self._is_initialized = False

    async def _generate_signals(
        self, market_data: Dict[str, pd.DataFrame], features: Dict[str, Any]
    ) -> List[Signal]:
        """Generate ML-based trading signals.

        Args:
            market_data: Current market data by symbol
            features: Calculated features by symbol

        Returns:
            List of trading signals
        """
        signals = []

        for symbol in self.symbols:
            if symbol not in market_data:
                continue

            # Prepare market data dict for the symbol
            symbol_market_data = {
                "symbol": symbol,
                "data": market_data[symbol],
                "price": (
                    market_data[symbol]["close"].iloc[-1] if len(market_data[symbol]) > 0 else 0
                ),
                "portfolio_value": 100000,  # Default, should be passed in
                "atr": (
                    market_data[symbol]["close"].rolling(14).std().iloc[-1]
                    if len(market_data[symbol]) > 14
                    else 2
                ),
            }

            # Generate signal for this symbol
            signal = await self.generate_signal(symbol, symbol_market_data)
            if signal:
                signals.append(signal)

        return signals

    async def generate_features(
        self, symbol: str, data: pd.DataFrame, timeframe: TimeFrame
    ) -> pd.DataFrame:
        """Generate features for ML prediction.

        Args:
            symbol: Trading symbol
            data: Price/volume data
            timeframe: Timeframe for analysis

        Returns:
            DataFrame with engineered features
        """
        cache_key = f"{symbol}_{timeframe.value}"

        # Check cache
        if cache_key in self.feature_cache:
            cached_time = (
                self.feature_cache[cache_key].index[-1]
                if len(self.feature_cache[cache_key]) > 0
                else None
            )
            if cached_time and cached_time >= data.index[-1] - timedelta(minutes=1):
                return self.feature_cache[cache_key]

        # Generate features directly here for the improved model
        # The improved model uses specific features that we need to calculate
        features = pd.DataFrame(index=data.index)

        # Ensure column names are lowercase
        if "Close" in data.columns:
            data = data.rename(
                columns={
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Open": "open",
                    "Volume": "volume",
                }
            )

        # Calculate the features the improved model expects
        if "close" in data.columns:
            # Returns
            features["returns"] = data["close"].pct_change()
            features["log_returns"] = np.log(data["close"] / data["close"].shift(1))

            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f"return_{period}d"] = data["close"].pct_change(period)
                features[f"sma_{period}"] = data["close"].rolling(period).mean() / data["close"]

            # Volatility
            features["volatility_20"] = features["returns"].rolling(20).std()
            features["volatility_5"] = features["returns"].rolling(5).std()

            # Volume features
            if "volume" in data.columns:
                features["volume_ratio"] = data["volume"] / data["volume"].rolling(20).mean()
                features["dollar_volume"] = data["close"] * data["volume"]

            # Time features
            features["day_of_week"] = data.index.dayofweek
            features["month"] = data.index.month

            # Microstructure
            if "high" in data.columns and "low" in data.columns:
                features["high_low_ratio"] = data["high"] / data["low"]
                features["close_to_high"] = data["close"] / data["high"]
                features["close_to_low"] = data["close"] / data["low"]

            # RSI
            delta = data["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = data["close"].ewm(span=12).mean()
            ema_26 = data["close"].ewm(span=26).mean()
            features["macd"] = (ema_12 - ema_26) / data["close"]

            # Bollinger Bands
            sma_20 = data["close"].rolling(20).mean()
            std_20 = data["close"].rolling(20).std()
            features["bb_position"] = (data["close"] - sma_20) / (2 * std_20)

        # Fill NaN values
        features = features.ffill().fillna(0)

        # Cache features
        self.feature_cache[cache_key] = features

        return features

    async def get_ml_predictions(self, symbol: str, features: pd.DataFrame) -> Dict[str, Any]:
        """Get ML model predictions with confidence filtering.

        Args:
            symbol: Trading symbol
            features: Feature DataFrame

        Returns:
            Dictionary with predictions and metadata
        """
        try:
            # First try to use the improved model
            improved_model = self.model_selector.available_models.get("improved_model")
            if improved_model:
                model_obj = improved_model["model"]
                scaler = improved_model.get("scaler")
                feature_cols = improved_model.get(
                    "features", improved_model.get("feature_columns", [])
                )
                confidence_threshold = improved_model.get("confidence_threshold", 0.6)

                # Prepare features
                if feature_cols and len(feature_cols) > 0:
                    # Use only the features the model was trained on
                    available_features = [f for f in feature_cols if f in features.columns]
                    if len(available_features) > 0:
                        X = features[available_features].iloc[-1:].fillna(0)
                    else:
                        X = features.iloc[-1:].fillna(0)
                else:
                    X = features.iloc[-1:].fillna(0)

                # Scale if scaler available
                if scaler:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X

                # Get prediction with probability
                if hasattr(model_obj, "predict_proba"):
                    probabilities = model_obj.predict_proba(X_scaled)[0]
                    prediction = model_obj.predict(X_scaled)[0]
                    confidence = max(probabilities)

                    # Apply confidence threshold
                    if confidence < confidence_threshold:
                        logger.info(
                            f"Low confidence prediction for {symbol}: {confidence:.3f} < {confidence_threshold}"
                        )
                        return {"prediction": 0, "confidence": confidence, "filtered": True}

                    # Convert to trading signal
                    signal = 1 if prediction == 1 else -1

                    logger.info(
                        f"High confidence prediction for {symbol}: signal={signal}, confidence={confidence:.3f}"
                    )

                    return {
                        "prediction": signal,
                        "confidence": confidence,
                        "agreement": confidence,  # Use confidence as agreement
                        "model_type": "improved_model",
                        "filtered": False,
                    }

            # Fallback to ensemble if improved model not available
            model = self.model_selector.selected_model
            if not model:
                return {"prediction": 0, "confidence": 0}

            # Get predictions from ensemble
            predictions = {}
            for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
                try:
                    pred = await self.model_selector.get_prediction(
                        model_type, features.iloc[-1:], prediction_type=PredictionType.DIRECTION
                    )
                    predictions[model_type.value] = pred
                except Exception as e:
                    logger.debug(f"Model {model_type} prediction failed: {e}")

            if not predictions:
                return {"prediction": 0, "confidence": 0}

            # Calculate ensemble prediction
            pred_values = [p["prediction"] for p in predictions.values()]
            pred_confidences = [p.get("confidence", 0.5) for p in predictions.values()]

            # Weighted average by confidence
            weights = np.array(pred_confidences)
            weights = weights / weights.sum()
            ensemble_pred = np.average(pred_values, weights=weights)

            # Calculate agreement
            direction_votes = [1 if p > 0 else -1 for p in pred_values]
            agreement = abs(sum(direction_votes)) / len(direction_votes)

            return {
                "prediction": ensemble_pred,
                "confidence": np.mean(pred_confidences),
                "agreement": agreement,
                "ensemble_votes": predictions,
                "model_type": "ensemble",
            }

        except Exception as e:
            logger.error(f"ML prediction failed for {symbol}: {e}")
            return {"prediction": 0, "confidence": 0}

    async def analyze_timeframe(
        self, symbol: str, data: pd.DataFrame, timeframe: TimeFrame
    ) -> Dict[str, Any]:
        """Analyze a specific timeframe.

        Args:
            symbol: Trading symbol
            data: Price data for timeframe
            timeframe: Timeframe to analyze

        Returns:
            Analysis results for timeframe
        """
        # Generate features
        features = await self.generate_features(symbol, data, timeframe)

        # Get ML predictions
        ml_pred = await self.get_ml_predictions(symbol, features)

        # Calculate technical confirmation
        tech_signals = self.calculate_technical_signals(data)

        return {
            "timeframe": timeframe,
            "ml_prediction": ml_pred["prediction"],
            "ml_confidence": ml_pred["confidence"],
            "ml_agreement": ml_pred.get("agreement", 0),
            "technical_score": tech_signals["score"],
            "trend": tech_signals["trend"],
            "momentum": tech_signals["momentum"],
            "volume_profile": self.analyze_volume_profile(data),
        }

    def calculate_technical_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicator signals.

        Args:
            data: Price data

        Returns:
            Technical signal scores
        """
        signals = {}

        # Trend signals
        sma_20 = data["close"].rolling(20).mean()
        sma_50 = data["close"].rolling(50).mean()
        ema_9 = data["close"].ewm(span=9).mean()

        trend_score = 0
        if len(data) >= 50:
            if data["close"].iloc[-1] > sma_20.iloc[-1]:
                trend_score += 0.33
            if data["close"].iloc[-1] > sma_50.iloc[-1]:
                trend_score += 0.33
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                trend_score += 0.34

        # Momentum signals
        rsi = self.calculate_rsi(data["close"])
        macd_line, signal_line = self.calculate_macd(data["close"])

        momentum_score = 0
        if rsi.iloc[-1] > 30 and rsi.iloc[-1] < 70:
            momentum_score += 0.5
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            momentum_score += 0.5

        # Overall score
        overall_score = (trend_score + momentum_score) / 2

        return {
            "score": overall_score,
            "trend": (
                "bullish" if trend_score > 0.5 else "bearish" if trend_score < 0.5 else "neutral"
            ),
            "momentum": (
                "strong" if momentum_score > 0.7 else "weak" if momentum_score < 0.3 else "moderate"
            ),
            "rsi": rsi.iloc[-1] if len(rsi) > 0 else 50,
            "macd_histogram": (macd_line - signal_line).iloc[-1] if len(macd_line) > 0 else 0,
        }

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    def analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """Analyze volume profile."""
        if "volume" not in data.columns:
            return "unknown"

        avg_volume = data["volume"].rolling(20).mean()
        current_volume = data["volume"].iloc[-1]

        if current_volume > avg_volume.iloc[-1] * 1.5:
            return "high"
        elif current_volume < avg_volume.iloc[-1] * 0.5:
            return "low"
        else:
            return "normal"

    async def multi_timeframe_analysis(
        self, symbol: str, data_dict: Dict[TimeFrame, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Perform multi-timeframe analysis.

        Args:
            symbol: Trading symbol
            data_dict: Dictionary of data for each timeframe

        Returns:
            Combined analysis across timeframes
        """
        analyses = {}

        # Analyze each timeframe
        for timeframe in self.timeframes:
            if timeframe in data_dict:
                analysis = await self.analyze_timeframe(symbol, data_dict[timeframe], timeframe)
                analyses[timeframe] = analysis

        # Combine analyses
        if not analyses:
            return {"signal": SignalType.HOLD, "confidence": 0}

        # Weight by timeframe (longer timeframes get more weight)
        weights = {
            TimeFrame.SCALP: 0.1,
            TimeFrame.INTRADAY: 0.3,
            TimeFrame.SWING: 0.4,
            TimeFrame.POSITION: 0.2,
        }

        total_score = 0
        total_confidence = 0
        total_weight = 0

        for tf, analysis in analyses.items():
            weight = weights.get(tf, 0.25)
            score = analysis["ml_prediction"]
            confidence = analysis["ml_confidence"]

            # Apply technical confirmation
            if analysis["technical_score"] < 0.3:
                confidence *= 0.7  # Reduce confidence if technicals disagree

            total_score += score * weight * confidence
            total_confidence += confidence * weight
            total_weight += weight

        if total_weight > 0:
            final_score = total_score / total_weight
            final_confidence = total_confidence / total_weight
        else:
            final_score = 0
            final_confidence = 0

        # Determine signal
        if final_confidence < self.confidence_threshold:
            signal = SignalType.HOLD
        elif final_score > 0.3:
            signal = SignalType.BUY
        elif final_score < -0.3:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        return {
            "signal": signal,
            "score": final_score,
            "confidence": final_confidence,
            "analyses": analyses,
        }

    def calculate_position_size(
        self,
        signal_strength: float,
        confidence: float,
        regime: str,
        portfolio_value: float,
        current_price: float,
    ) -> int:
        """Calculate position size based on ML confidence and regime.

        Args:
            signal_strength: Strength of the signal (-1 to 1)
            confidence: ML model confidence (0 to 1)
            regime: Current market regime
            portfolio_value: Total portfolio value
            current_price: Current asset price

        Returns:
            Number of shares to trade
        """
        # Base position from risk management
        risk_amount = portfolio_value * self.risk_per_trade
        max_position = portfolio_value * self.max_position_pct

        if self.position_size_method == "kelly":
            # Kelly criterion with ML confidence
            win_prob = (confidence + 1) / 2  # Convert to probability
            loss_prob = 1 - win_prob

            # Expected win/loss ratio (simplified)
            win_loss_ratio = 2.0  # Assume 2:1 reward/risk

            # Kelly fraction
            kelly_f = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
            kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25%

            position_value = portfolio_value * kelly_f * abs(signal_strength)

        elif self.position_size_method == "fixed_risk":
            # Fixed risk with confidence adjustment
            position_value = risk_amount * confidence

        else:  # fixed_pct
            position_value = max_position * confidence * abs(signal_strength)

        # Adjust for regime
        regime_multipliers = {"bull": 1.2, "bear": 0.8, "volatile": 0.6, "neutral": 1.0}
        regime_mult = regime_multipliers.get(regime, 1.0)
        position_value *= regime_mult

        # Convert to shares
        shares = int(position_value / current_price)
        max_shares = int(max_position / current_price)

        return min(shares, max_shares)

    async def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[MLSignal]:
        """Generate trading signal using ML predictions.

        Args:
            symbol: Trading symbol
            market_data: Current market data

        Returns:
            ML-driven trading signal or None
        """
        try:
            # Extract data for different timeframes
            data_dict = {}
            for timeframe in self.timeframes:
                if f"data_{timeframe.value}" in market_data:
                    data_dict[timeframe] = market_data[f"data_{timeframe.value}"]

            if not data_dict:
                # Use default data if no timeframe-specific data
                data_dict[TimeFrame.INTRADAY] = market_data.get("data", pd.DataFrame())

            # Perform multi-timeframe analysis
            analysis = await self.multi_timeframe_analysis(symbol, data_dict)

            if analysis["signal"] == SignalType.HOLD:
                return None

            # Get current regime (will be implemented in regime_detector.py)
            regime = self.regime_cache.get(symbol, "neutral")

            # Create ML signal
            current_price = market_data.get("price", 0)
            portfolio_value = market_data.get("portfolio_value", 100000)

            # Calculate position size
            position_size = self.calculate_position_size(
                analysis["score"], analysis["confidence"], regime, portfolio_value, current_price
            )

            # Calculate stop loss and take profit
            atr = market_data.get("atr", current_price * 0.02)

            if analysis["signal"] == SignalType.BUY:
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)
            else:  # SELL
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)

            # Create signal
            signal = MLSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=analysis["signal"],
                strength=abs(analysis["score"]),
                quantity=position_size,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=1.5,
                expected_value=analysis["score"] * analysis["confidence"],
                ml_confidence=analysis["confidence"],
                regime=regime,
                timeframe=TimeFrame.INTRADAY,
                feature_importance=analysis.get("feature_importance"),
                ensemble_votes=analysis.get("ensemble_votes"),
            )

            # Log signal generation
            logger.info(
                "ML signal generated",
                symbol=symbol,
                signal_type=signal.signal_type.value,
                confidence=signal.ml_confidence,
                regime=regime,
                position_size=position_size,
                analyses=analysis.get("analyses", {}),
            )

            # Update metrics
            if hasattr(self.metrics, "signals_generated"):
                self.metrics.signals_generated += 1
            if hasattr(self.metrics, "last_signal_time"):
                self.metrics.last_signal_time = datetime.now()

            return signal

        except Exception as e:
            logger.error(f"Failed to generate ML signal for {symbol}: {e}")
            return None

    def update_regime(self, symbol: str, regime: str) -> None:
        """Update market regime for symbol.

        Args:
            symbol: Trading symbol
            regime: Market regime
        """
        self.regime_cache[symbol] = regime
        logger.debug(f"Updated regime for {symbol}: {regime}")

    async def update_state(self, market_data: Dict[str, Any]) -> None:
        """Update strategy state based on market data.

        Args:
            market_data: Current market data
        """
        # Update performance metrics
        if "pnl" in market_data:
            self.performance["total_pnl"] = market_data["pnl"]

        # Clear old cache entries
        cache_ttl = timedelta(minutes=5)
        current_time = datetime.now()

        for cache in [self.prediction_cache, self.feature_cache]:
            old_keys = []
            for key in cache:
                if isinstance(cache[key], pd.DataFrame):
                    if cache[key].index[-1] < current_time - cache_ttl:
                        old_keys.append(key)
            for key in old_keys:
                del cache[key]

    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status.

        Returns:
            Status dictionary
        """
        return {
            "state": self.state.state if hasattr(self.state, "state") else "ready",
            "metrics": self.metrics.__dict__ if hasattr(self.metrics, "__dict__") else {},
            "performance": self.performance,
            "model_count": len(self.model_selector.available_models),
            "cached_predictions": len(self.prediction_cache),
            "cached_features": len(self.feature_cache),
            "active_regimes": self.regime_cache,
        }
