"""Market Regime Detection for Adaptive Trading.

This module implements sophisticated regime detection that identifies:
- Market trends (bull, bear, sideways)
- Volatility regimes (low, normal, high, extreme)
- Market microstructure regimes
- Sector rotation patterns
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import structlog

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    
    # Trend regimes
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    
    # Volatility regimes
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"
    
    # Special regimes
    CRASH = "crash"
    SQUEEZE = "squeeze"
    BREAKOUT = "breakout"
    RANGE_BOUND = "range_bound"


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    
    timestamp: datetime
    trend_regime: MarketRegime
    volatility_regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    transition_probability: float = 0.0
    expected_duration: int = 0  # Expected duration in periods
    historical_performance: Dict[str, float] = None


class RegimeDetector:
    """Advanced market regime detection system."""
    
    def __init__(
        self,
        lookback_period: int = 100,
        vol_lookback: int = 30,
        regime_threshold: float = 0.6,
        use_ml_detection: bool = True,
        min_regime_duration: int = 5
    ):
        """Initialize regime detector.
        
        Args:
            lookback_period: Periods for trend analysis
            vol_lookback: Periods for volatility analysis
            regime_threshold: Confidence threshold for regime change
            use_ml_detection: Use ML-based regime detection
            min_regime_duration: Minimum periods for regime confirmation
        """
        self.lookback_period = lookback_period
        self.vol_lookback = vol_lookback
        self.regime_threshold = regime_threshold
        self.use_ml_detection = use_ml_detection
        self.min_regime_duration = min_regime_duration
        
        # Regime history
        self.regime_history: Dict[str, List[RegimeState]] = {}
        self.current_regimes: Dict[str, RegimeState] = {}
        
        # ML models for regime detection
        self.gmm_model = None
        self.regime_features: List[str] = []
        
    async def initialize(self) -> None:
        """Initialize regime detection models."""
        if self.use_ml_detection:
            # Initialize Gaussian Mixture Model for regime clustering
            self.gmm_model = GaussianMixture(
                n_components=4,  # 4 main regimes
                covariance_type='full',
                random_state=42
            )
            
            # Define features for regime detection
            self.regime_features = [
                'returns_mean', 'returns_std', 'returns_skew', 'returns_kurt',
                'volume_ratio', 'price_momentum', 'volatility_ratio',
                'trend_strength', 'market_efficiency'
            ]
            
        logger.info("Regime detector initialized")
        
    def calculate_trend_regime(self, data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Calculate trend-based market regime.
        
        Args:
            data: Price data
            
        Returns:
            Tuple of (regime, confidence)
        """
        if len(data) < self.lookback_period:
            return MarketRegime.NEUTRAL, 0.5
            
        # Calculate trend indicators
        prices = data['close'].values
        returns = pd.Series(prices).pct_change().dropna()
        
        # Moving averages
        sma_20 = pd.Series(prices).rolling(20).mean()
        sma_50 = pd.Series(prices).rolling(50).mean()
        sma_200 = pd.Series(prices).rolling(200).mean() if len(prices) > 200 else sma_50
        
        current_price = prices[-1]
        
        # Trend strength calculation
        trend_score = 0
        confidence = 0
        
        # Price relative to moving averages
        if current_price > sma_20.iloc[-1]:
            trend_score += 0.25
            confidence += 0.2
        if current_price > sma_50.iloc[-1]:
            trend_score += 0.25
            confidence += 0.2
        if len(prices) > 200 and current_price > sma_200.iloc[-1]:
            trend_score += 0.25
            confidence += 0.2
            
        # Moving average alignment
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend_score += 0.25
            confidence += 0.2
            
        # Return momentum
        returns_1m = returns[-20:].mean() if len(returns) > 20 else 0
        returns_3m = returns[-60:].mean() if len(returns) > 60 else returns_1m
        
        if returns_1m > 0.001:  # Positive monthly returns
            trend_score += 0.2
        if returns_3m > 0.002:  # Positive quarterly returns
            trend_score += 0.2
            
        # Linear regression trend
        if len(prices) >= 20:
            x = np.arange(20)
            y = prices[-20:]
            slope, _, r_value, _, _ = stats.linregress(x, y)
            
            trend_strength = abs(r_value)
            confidence = min(confidence + trend_strength * 0.2, 1.0)
            
            if slope > 0:
                trend_score += 0.3 * trend_strength
            else:
                trend_score -= 0.3 * trend_strength
                
        # Determine regime
        if trend_score > 0.7:
            regime = MarketRegime.STRONG_BULL
        elif trend_score > 0.4:
            regime = MarketRegime.BULL
        elif trend_score < -0.7:
            regime = MarketRegime.STRONG_BEAR
        elif trend_score < -0.4:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.NEUTRAL
            
        return regime, confidence
        
    def calculate_volatility_regime(self, data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Calculate volatility-based market regime.
        
        Args:
            data: Price data
            
        Returns:
            Tuple of (regime, confidence)
        """
        if len(data) < self.vol_lookback:
            return MarketRegime.NORMAL_VOL, 0.5
            
        prices = data['close'].values
        returns = pd.Series(prices).pct_change().dropna()
        
        # Calculate volatility metrics
        current_vol = returns[-self.vol_lookback:].std()
        
        # Historical volatility percentiles
        if len(returns) > 252:  # 1 year of data
            vol_series = returns.rolling(self.vol_lookback).std().dropna()
            vol_percentile = stats.percentileofscore(vol_series, current_vol)
        else:
            vol_percentile = 50
            
        # GARCH-style volatility clustering
        squared_returns = returns ** 2
        vol_autocorr = squared_returns.autocorr(lag=1) if len(squared_returns) > 2 else 0
        
        # Realized volatility
        if 'high' in data.columns and 'low' in data.columns:
            daily_range = (data['high'] - data['low']) / data['close']
            realized_vol = daily_range[-self.vol_lookback:].mean()
        else:
            realized_vol = current_vol
            
        # Determine volatility regime
        confidence = 0.5 + abs(vol_percentile - 50) / 100
        
        if vol_percentile < 20:
            regime = MarketRegime.LOW_VOL
        elif vol_percentile < 40:
            regime = MarketRegime.NORMAL_VOL
        elif vol_percentile < 75:
            regime = MarketRegime.HIGH_VOL
        else:
            regime = MarketRegime.EXTREME_VOL
            
        # Adjust confidence based on volatility clustering
        confidence = min(confidence + abs(vol_autocorr) * 0.2, 1.0)
        
        return regime, confidence
        
    def detect_special_regimes(self, data: pd.DataFrame) -> Optional[MarketRegime]:
        """Detect special market conditions.
        
        Args:
            data: Price data
            
        Returns:
            Special regime if detected, None otherwise
        """
        if len(data) < 20:
            return None
            
        prices = data['close'].values
        returns = pd.Series(prices).pct_change().dropna()
        
        # Crash detection
        if len(returns) >= 5:
            recent_return = returns[-5:].sum()
            if recent_return < -0.1:  # 10% drop in 5 days
                return MarketRegime.CRASH
                
        # Squeeze detection (low volatility before breakout)
        if len(returns) >= 20:
            recent_vol = returns[-10:].std()
            historical_vol = returns[-60:].std() if len(returns) > 60 else returns.std()
            
            if recent_vol < historical_vol * 0.5:
                # Check for bollinger band squeeze
                sma = pd.Series(prices).rolling(20).mean()
                std = pd.Series(prices).rolling(20).std()
                
                if std.iloc[-1] < std.mean() * 0.7:
                    return MarketRegime.SQUEEZE
                    
        # Breakout detection
        if len(prices) >= 50:
            resistance = prices[-50:].max()
            support = prices[-50:].min()
            current = prices[-1]
            
            if current > resistance * 0.98:  # Near resistance
                if data['volume'].iloc[-1] > data['volume'][-20:].mean() * 1.5:
                    return MarketRegime.BREAKOUT
                    
        # Range-bound detection
        if len(prices) >= 30:
            price_range = prices[-30:].max() - prices[-30:].min()
            avg_price = prices[-30:].mean()
            
            if price_range / avg_price < 0.05:  # Less than 5% range
                return MarketRegime.RANGE_BOUND
                
        return None
        
    async def extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for ML-based regime detection.
        
        Args:
            data: Price and volume data
            
        Returns:
            Feature array for ML model
        """
        features = []
        
        prices = data['close'].values
        returns = pd.Series(prices).pct_change().dropna()
        
        # Return statistics
        features.append(returns.mean())
        features.append(returns.std())
        features.append(returns.skew())
        features.append(returns.kurtosis())
        
        # Volume ratio
        if 'volume' in data.columns:
            vol_ratio = data['volume'].iloc[-1] / data['volume'].mean()
            features.append(vol_ratio)
        else:
            features.append(1.0)
            
        # Price momentum
        if len(prices) > 20:
            momentum = (prices[-1] - prices[-20]) / prices[-20]
            features.append(momentum)
        else:
            features.append(0.0)
            
        # Volatility ratio
        if len(returns) > 60:
            recent_vol = returns[-20:].std()
            hist_vol = returns[-60:].std()
            vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0
            features.append(vol_ratio)
        else:
            features.append(1.0)
            
        # Trend strength (R-squared of linear regression)
        if len(prices) >= 20:
            x = np.arange(20)
            y = prices[-20:]
            _, _, r_value, _, _ = stats.linregress(x, y)
            features.append(r_value ** 2)
        else:
            features.append(0.0)
            
        # Market efficiency (Hurst exponent approximation)
        if len(returns) > 100:
            # Simplified Hurst calculation
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            features.append(hurst)
        else:
            features.append(0.5)
            
        return np.array(features).reshape(1, -1)
        
    async def detect_ml_regime(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Tuple[MarketRegime, float]:
        """Detect regime using machine learning.
        
        Args:
            symbol: Trading symbol
            data: Price and volume data
            
        Returns:
            Tuple of (regime, confidence)
        """
        if not self.use_ml_detection or self.gmm_model is None:
            return MarketRegime.NEUTRAL, 0.5
            
        try:
            # Extract features
            features = await self.extract_regime_features(data)
            
            # Predict regime cluster
            cluster = self.gmm_model.predict(features)[0]
            probabilities = self.gmm_model.predict_proba(features)[0]
            confidence = probabilities.max()
            
            # Map clusters to regimes (based on training)
            cluster_regime_map = {
                0: MarketRegime.BULL,
                1: MarketRegime.BEAR,
                2: MarketRegime.HIGH_VOL,
                3: MarketRegime.RANGE_BOUND
            }
            
            regime = cluster_regime_map.get(cluster, MarketRegime.NEUTRAL)
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"ML regime detection failed for {symbol}: {e}")
            return MarketRegime.NEUTRAL, 0.5
            
    def calculate_transition_probability(
        self,
        symbol: str,
        current_regime: MarketRegime,
        data: pd.DataFrame
    ) -> float:
        """Calculate probability of regime transition.
        
        Args:
            symbol: Trading symbol
            current_regime: Current market regime
            data: Price data
            
        Returns:
            Probability of regime change
        """
        if symbol not in self.regime_history or len(self.regime_history[symbol]) < 2:
            return 0.1  # Default low probability
            
        history = self.regime_history[symbol]
        
        # Calculate average regime duration
        regime_durations = []
        current_duration = 1
        
        for i in range(1, len(history)):
            if history[i].trend_regime == history[i-1].trend_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_duration = 1
                
        if regime_durations:
            avg_duration = np.mean(regime_durations)
            current_regime_duration = sum(
                1 for r in reversed(history)
                if r.trend_regime == current_regime
            )
            
            # Probability increases with duration
            transition_prob = min(current_regime_duration / avg_duration, 1.0)
        else:
            transition_prob = 0.1
            
        # Adjust based on recent volatility
        returns = pd.Series(data['close'].values).pct_change().dropna()
        if len(returns) > 10:
            recent_vol = returns[-10:].std()
            if recent_vol > returns.std() * 1.5:
                transition_prob = min(transition_prob * 1.5, 1.0)
                
        return transition_prob
        
    async def detect_regime(
        self,
        symbol: str,
        data: pd.DataFrame,
        use_ensemble: bool = True
    ) -> RegimeState:
        """Detect current market regime for symbol.
        
        Args:
            symbol: Trading symbol
            data: Market data
            use_ensemble: Use ensemble of detection methods
            
        Returns:
            Current regime state
        """
        # Detect trend regime
        trend_regime, trend_conf = self.calculate_trend_regime(data)
        
        # Detect volatility regime
        vol_regime, vol_conf = self.calculate_volatility_regime(data)
        
        # Check for special regimes
        special_regime = self.detect_special_regimes(data)
        
        # ML-based detection if enabled
        ml_regime = None
        ml_conf = 0
        if use_ensemble and self.use_ml_detection:
            ml_regime, ml_conf = await self.detect_ml_regime(symbol, data)
            
        # Combine regimes with ensemble voting
        if special_regime:
            final_trend_regime = special_regime
            confidence = 0.9  # High confidence for special regimes
        elif use_ensemble and ml_regime:
            # Weight ML prediction with rule-based
            if ml_conf > 0.7:
                final_trend_regime = ml_regime
            else:
                final_trend_regime = trend_regime
            confidence = (trend_conf + ml_conf) / 2
        else:
            final_trend_regime = trend_regime
            confidence = trend_conf
            
        # Calculate additional indicators
        prices = data['close'].values
        returns = pd.Series(prices).pct_change().dropna()
        
        indicators = {
            'trend_strength': trend_conf,
            'volatility': returns.std() if len(returns) > 0 else 0,
            'momentum': returns[-20:].mean() if len(returns) > 20 else 0,
            'volume_profile': data['volume'].iloc[-1] / data['volume'].mean() if 'volume' in data.columns else 1.0
        }
        
        # Calculate transition probability
        transition_prob = self.calculate_transition_probability(
            symbol, final_trend_regime, data
        )
        
        # Create regime state
        regime_state = RegimeState(
            timestamp=datetime.now(),
            trend_regime=final_trend_regime,
            volatility_regime=vol_regime,
            confidence=confidence,
            indicators=indicators,
            transition_probability=transition_prob,
            expected_duration=self.min_regime_duration
        )
        
        # Update history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        self.regime_history[symbol].append(regime_state)
        
        # Keep only recent history
        max_history = 1000
        if len(self.regime_history[symbol]) > max_history:
            self.regime_history[symbol] = self.regime_history[symbol][-max_history:]
            
        # Update current regime
        self.current_regimes[symbol] = regime_state
        
        logger.info(
            "Regime detected",
            symbol=symbol,
            trend_regime=final_trend_regime.value,
            volatility_regime=vol_regime.value,
            confidence=confidence,
            transition_probability=transition_prob
        )
        
        return regime_state
        
    def get_regime_recommendations(self, regime: RegimeState) -> Dict[str, Any]:
        """Get trading recommendations based on regime.
        
        Args:
            regime: Current regime state
            
        Returns:
            Trading recommendations
        """
        recommendations = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'preferred_strategies': [],
            'avoid_strategies': [],
            'risk_level': 'normal'
        }
        
        # Trend regime recommendations
        if regime.trend_regime == MarketRegime.STRONG_BULL:
            recommendations['position_size_multiplier'] = 1.2
            recommendations['preferred_strategies'] = ['momentum', 'breakout']
            recommendations['risk_level'] = 'aggressive'
            
        elif regime.trend_regime == MarketRegime.BULL:
            recommendations['position_size_multiplier'] = 1.0
            recommendations['preferred_strategies'] = ['momentum', 'trend_following']
            
        elif regime.trend_regime in [MarketRegime.BEAR, MarketRegime.STRONG_BEAR]:
            recommendations['position_size_multiplier'] = 0.7
            recommendations['stop_loss_multiplier'] = 0.8
            recommendations['preferred_strategies'] = ['short', 'mean_reversion']
            recommendations['risk_level'] = 'conservative'
            
        elif regime.trend_regime == MarketRegime.RANGE_BOUND:
            recommendations['preferred_strategies'] = ['mean_reversion', 'range_trading']
            recommendations['avoid_strategies'] = ['breakout', 'momentum']
            
        # Volatility adjustments
        if regime.volatility_regime == MarketRegime.LOW_VOL:
            recommendations['position_size_multiplier'] *= 1.1
            recommendations['stop_loss_multiplier'] *= 1.2
            
        elif regime.volatility_regime == MarketRegime.HIGH_VOL:
            recommendations['position_size_multiplier'] *= 0.8
            recommendations['stop_loss_multiplier'] *= 0.9
            recommendations['take_profit_multiplier'] *= 1.2
            
        elif regime.volatility_regime == MarketRegime.EXTREME_VOL:
            recommendations['position_size_multiplier'] *= 0.5
            recommendations['stop_loss_multiplier'] *= 0.8
            recommendations['risk_level'] = 'very_conservative'
            
        # Special regime handling
        if regime.trend_regime == MarketRegime.CRASH:
            recommendations['position_size_multiplier'] = 0.0  # No new positions
            recommendations['risk_level'] = 'exit_all'
            
        elif regime.trend_regime == MarketRegime.SQUEEZE:
            recommendations['position_size_multiplier'] *= 0.5  # Reduce size before breakout
            recommendations['preferred_strategies'] = ['wait_for_breakout']
            
        return recommendations
        
    def get_regime_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get historical regime statistics for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Regime statistics
        """
        if symbol not in self.regime_history:
            return {}
            
        history = self.regime_history[symbol]
        
        # Count regime occurrences
        regime_counts = {}
        for state in history:
            regime = state.trend_regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        # Calculate average confidence per regime
        regime_confidence = {}
        for regime in regime_counts:
            confidences = [s.confidence for s in history if s.trend_regime.value == regime]
            regime_confidence[regime] = np.mean(confidences)
            
        # Calculate regime transitions
        transitions = {}
        for i in range(1, len(history)):
            from_regime = history[i-1].trend_regime.value
            to_regime = history[i].trend_regime.value
            
            if from_regime != to_regime:
                key = f"{from_regime}_to_{to_regime}"
                transitions[key] = transitions.get(key, 0) + 1
                
        return {
            'regime_counts': regime_counts,
            'regime_confidence': regime_confidence,
            'transitions': transitions,
            'current_regime': self.current_regimes.get(symbol),
            'history_length': len(history)
        }