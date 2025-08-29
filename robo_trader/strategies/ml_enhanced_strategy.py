"""
Enhanced ML-Driven Strategy with Regime Detection and Multi-Timeframe Analysis.

This strategy combines:
- ML predictions with confidence filtering
- Market regime detection for adaptive behavior
- Multi-timeframe signal confirmation
- Dynamic position sizing based on regime and confidence
- Risk-aware entry and exit management
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import structlog

from .framework import StrategyFramework, Signal
from .regime_detector import RegimeDetector, MarketRegime, RegimeState
from ..ml.model_selector import ModelSelector
from ..features.feature_pipeline import FeaturePipeline
from ..config import Config


logger = structlog.get_logger(__name__)


@dataclass
class MultiTimeframeSignal:
    """Signal from multiple timeframes."""
    timeframe_1m: Optional[Signal] = None
    timeframe_5m: Optional[Signal] = None
    timeframe_15m: Optional[Signal] = None
    timeframe_1h: Optional[Signal] = None
    timeframe_1d: Optional[Signal] = None
    combined_signal: Optional[Signal] = None
    alignment_score: float = 0.0  # How well timeframes agree
    

class MLEnhancedStrategy(StrategyFramework):
    """
    Advanced ML strategy with regime adaptation and multi-timeframe analysis.
    
    This strategy:
    1. Uses ML predictions as primary signal
    2. Confirms with multi-timeframe analysis
    3. Adapts to market regimes
    4. Dynamically sizes positions
    5. Manages risk based on regime
    """
    
    def __init__(self, config: Config):
        """Initialize ML enhanced strategy."""
        super().__init__(config)
        
        # ML components
        self.model_selector = ModelSelector()
        self.feature_pipeline = FeaturePipeline(config)
        self.regime_detector = RegimeDetector(lookback_window=100)
        
        # Strategy parameters
        self.base_confidence_threshold = 0.6
        self.min_alignment_score = 0.5
        self.max_correlation_positions = 3
        
        # Multi-timeframe settings
        self.timeframes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '1d': 390  # Trading minutes in a day
        }
        
        # Position sizing parameters
        self.base_position_size = 0.02  # 2% of capital base
        self.max_position_size = 0.05   # 5% max
        self.regime_size_adjustments = {
            MarketRegime.STRONG_BULL: 1.5,
            MarketRegime.BULL: 1.2,
            MarketRegime.NEUTRAL: 1.0,
            MarketRegime.BEAR: 0.7,
            MarketRegime.STRONG_BEAR: 0.5,
            MarketRegime.HIGH_VOL: 0.6,
            MarketRegime.EXTREME_VOL: 0.3,
            MarketRegime.CRASH: 0.0  # No new positions in crash
        }
        
        # Risk parameters by regime
        self.regime_risk_params = {
            MarketRegime.STRONG_BULL: {'stop_loss': 0.03, 'take_profit': 0.10},
            MarketRegime.BULL: {'stop_loss': 0.025, 'take_profit': 0.07},
            MarketRegime.NEUTRAL: {'stop_loss': 0.02, 'take_profit': 0.04},
            MarketRegime.BEAR: {'stop_loss': 0.015, 'take_profit': 0.03},
            MarketRegime.STRONG_BEAR: {'stop_loss': 0.01, 'take_profit': 0.02},
            MarketRegime.HIGH_VOL: {'stop_loss': 0.04, 'take_profit': 0.08},
            MarketRegime.EXTREME_VOL: {'stop_loss': 0.05, 'take_profit': 0.10},
        }
        
        # State tracking
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []
        self.mtf_signals: Dict[str, MultiTimeframeSignal] = {}
        self.position_correlations: Dict[str, float] = {}
        
        logger.info("Initialized ML Enhanced Strategy", 
                   confidence_threshold=self.base_confidence_threshold,
                   timeframes=list(self.timeframes.keys()))
    
    async def _initialize(self) -> None:
        """Initialize strategy components."""
        await self.feature_pipeline.start()
        self.model_selector.load_available_models()
        
        if not self.model_selector.available_models:
            logger.warning("No ML models available")
        else:
            logger.info(f"Loaded {len(self.model_selector.available_models)} ML models")
    
    async def analyze(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Analyze symbol with ML predictions and multi-timeframe confirmation.
        
        Args:
            symbol: Stock symbol
            data: Price data (OHLCV)
            
        Returns:
            Trading signal or None
        """
        if data is None or len(data) < 200:
            return None
        
        try:
            # 1. Detect market regime
            regime = await self._detect_regime(symbol, data)
            self.current_regime = regime
            
            # Skip if in crash regime
            if regime.primary_regime == MarketRegime.CRASH:
                logger.warning(f"Crash regime detected for {symbol}, skipping")
                return None
            
            # 2. Get ML predictions
            ml_signal = await self._get_ml_prediction(symbol, data)
            if ml_signal is None:
                return None
            
            # 3. Multi-timeframe analysis
            mtf_signal = await self._analyze_multi_timeframe(symbol, data)
            self.mtf_signals[symbol] = mtf_signal
            
            # 4. Check signal alignment
            if mtf_signal.alignment_score < self.min_alignment_score:
                logger.debug(f"Poor timeframe alignment for {symbol}: {mtf_signal.alignment_score:.2f}")
                return None
            
            # 5. Combine signals with regime awareness
            final_signal = self._combine_signals(ml_signal, mtf_signal, regime)
            
            # 6. Apply position sizing
            if final_signal:
                final_signal = self._apply_position_sizing(final_signal, regime)
                
            # 7. Check correlation limits
            if final_signal and not self._check_correlation_limits(symbol, final_signal):
                logger.debug(f"Correlation limits exceeded for {symbol}")
                return None
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    async def _detect_regime(self, symbol: str, data: pd.DataFrame) -> RegimeState:
        """Detect current market regime."""
        # Get correlation data if available
        correlations = None  # Would fetch from correlation manager
        
        # Detect regime
        regime = self.regime_detector.detect_regime(data, correlations)
        
        # Log regime changes
        if self.regime_history:
            last_regime = self.regime_history[-1]
            if regime.primary_regime != last_regime.primary_regime:
                logger.info(f"Regime change for {symbol}: {last_regime.primary_regime.value} -> {regime.primary_regime.value}")
        
        self.regime_history.append(regime)
        
        # Trim history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        return regime
    
    async def _get_ml_prediction(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """Get ML model prediction."""
        # Calculate features
        features_df = await self.feature_pipeline.calculate_features_timeseries(
            symbol=symbol,
            price_data=data,
            lookback_window=20
        )
        
        if features_df.empty:
            return None
        
        # Get latest features
        latest_features = features_df.iloc[-1:].copy()
        
        # Select best model
        best_model = self.model_selector.select_best_model(
            features=latest_features,
            market_conditions={'volatility': self.current_regime.volatility_percentile if self.current_regime else 50}
        )
        
        if not best_model:
            return None
        
        # Make prediction
        model_data = self.model_selector.available_models[best_model]
        model = model_data['model']
        
        # Prepare features
        feature_names = model_data['features']
        X = latest_features[feature_names].values
        
        # Get prediction and confidence
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = np.max(probabilities)
        
        # Apply confidence threshold (adjusted for regime)
        confidence_threshold = self._get_adjusted_confidence_threshold()
        
        if confidence < confidence_threshold:
            return None
        
        # Create signal
        if prediction == 1:  # Buy signal
            signal = Signal(
                symbol=symbol,
                action='BUY',
                confidence=confidence,
                features={'ml_prediction': 1, 'ml_confidence': confidence}
            )
        elif prediction == -1:  # Sell signal
            signal = Signal(
                symbol=symbol,
                action='SELL',
                confidence=confidence,
                features={'ml_prediction': -1, 'ml_confidence': confidence}
            )
        else:
            signal = None
        
        return signal
    
    async def _analyze_multi_timeframe(self, symbol: str, data: pd.DataFrame) -> MultiTimeframeSignal:
        """Analyze multiple timeframes for signal confirmation."""
        mtf = MultiTimeframeSignal()
        signals = {}
        
        # Analyze each timeframe
        for tf_name, tf_minutes in self.timeframes.items():
            # Resample data to timeframe
            tf_data = self._resample_data(data, tf_minutes)
            
            if len(tf_data) < 50:
                continue
            
            # Calculate timeframe-specific indicators
            tf_signal = self._analyze_timeframe(tf_data)
            signals[tf_name] = tf_signal
            
            # Store in MTF signal
            setattr(mtf, f'timeframe_{tf_name}', tf_signal)
        
        # Calculate alignment score
        if signals:
            directions = [s.action for s in signals.values() if s]
            if directions:
                buy_count = sum(1 for d in directions if d == 'BUY')
                sell_count = sum(1 for d in directions if d == 'SELL')
                total = len(directions)
                
                # Alignment score: how much timeframes agree
                if buy_count > sell_count:
                    mtf.alignment_score = buy_count / total
                    mtf.combined_signal = Signal(
                        symbol=symbol,
                        action='BUY',
                        confidence=mtf.alignment_score,
                        features={'mtf_agreement': buy_count}
                    )
                elif sell_count > buy_count:
                    mtf.alignment_score = sell_count / total
                    mtf.combined_signal = Signal(
                        symbol=symbol,
                        action='SELL',
                        confidence=mtf.alignment_score,
                        features={'mtf_agreement': sell_count}
                    )
                else:
                    mtf.alignment_score = 0.0
        
        return mtf
    
    def _resample_data(self, data: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Resample data to specified timeframe."""
        if minutes == 1:
            return data
        
        # Resample OHLCV data
        resampled = pd.DataFrame()
        resampled['open'] = data['open'].resample(f'{minutes}min').first()
        resampled['high'] = data['high'].resample(f'{minutes}min').max()
        resampled['low'] = data['low'].resample(f'{minutes}min').min()
        resampled['close'] = data['close'].resample(f'{minutes}min').last()
        resampled['volume'] = data['volume'].resample(f'{minutes}min').sum()
        
        return resampled.dropna()
    
    def _analyze_timeframe(self, data: pd.DataFrame) -> Optional[Signal]:
        """Analyze a single timeframe for signals."""
        if len(data) < 20:
            return None
        
        close = data['close'].values
        
        # Simple trend analysis
        sma_fast = pd.Series(close).rolling(10).mean().iloc[-1]
        sma_slow = pd.Series(close).rolling(20).mean().iloc[-1]
        
        current_price = close[-1]
        
        # Generate signal based on MA crossover and price position
        if current_price > sma_fast > sma_slow:
            return Signal(
                symbol='',
                action='BUY',
                confidence=0.6,
                features={'trend': 'up'}
            )
        elif current_price < sma_fast < sma_slow:
            return Signal(
                symbol='',
                action='SELL',
                confidence=0.6,
                features={'trend': 'down'}
            )
        
        return None
    
    def _combine_signals(self, 
                        ml_signal: Signal,
                        mtf_signal: MultiTimeframeSignal,
                        regime: RegimeState) -> Optional[Signal]:
        """Combine ML and multi-timeframe signals with regime awareness."""
        # Check if signals agree
        if ml_signal.action != mtf_signal.combined_signal.action:
            # Signals disagree - use ML if high confidence
            if ml_signal.confidence > 0.8:
                final_signal = ml_signal
            else:
                return None
        else:
            # Signals agree - combine confidence
            combined_confidence = (ml_signal.confidence + mtf_signal.alignment_score) / 2
            
            final_signal = Signal(
                symbol=ml_signal.symbol,
                action=ml_signal.action,
                confidence=combined_confidence,
                features={
                    'ml_confidence': ml_signal.confidence,
                    'mtf_alignment': mtf_signal.alignment_score,
                    'regime': regime.primary_regime.value,
                    'regime_confidence': regime.confidence
                }
            )
        
        # Adjust for regime
        regime_params = self.regime_detector.get_regime_parameters(regime)
        
        # Filter based on regime entry threshold
        if final_signal.confidence < regime_params['entry_threshold']:
            return None
        
        # Add risk parameters
        risk_params = self.regime_risk_params.get(
            regime.primary_regime,
            {'stop_loss': 0.02, 'take_profit': 0.05}
        )
        
        final_signal.features.update({
            'stop_loss': risk_params['stop_loss'],
            'take_profit': risk_params['take_profit'],
            'max_holding_period': self._get_max_holding_period(regime)
        })
        
        return final_signal
    
    def _apply_position_sizing(self, signal: Signal, regime: RegimeState) -> Signal:
        """Apply dynamic position sizing based on regime and confidence."""
        # Base position size
        position_size = self.base_position_size
        
        # Adjust for regime
        regime_mult = self.regime_size_adjustments.get(regime.primary_regime, 1.0)
        position_size *= regime_mult
        
        # Adjust for confidence
        confidence_mult = signal.confidence
        position_size *= confidence_mult
        
        # Adjust for volatility
        if regime.volatility_percentile > 80:
            position_size *= 0.5
        elif regime.volatility_percentile < 20:
            position_size *= 1.2
        
        # Cap at maximum
        position_size = min(position_size, self.max_position_size)
        
        # Add to signal
        signal.features['position_size'] = position_size
        signal.features['position_value'] = position_size  # Will be multiplied by account value
        
        return signal
    
    def _check_correlation_limits(self, symbol: str, signal: Signal) -> bool:
        """Check if adding position would exceed correlation limits."""
        # This would integrate with correlation manager
        # For now, simple limit on number of positions
        current_positions = len([s for s in self.signals.values() if s])
        
        if current_positions >= self.max_correlation_positions:
            # Check if this is a better opportunity
            min_confidence = min(s.confidence for s in self.signals.values() if s)
            if signal.confidence > min_confidence * 1.2:
                return True  # Allow if significantly better
            return False
        
        return True
    
    def _get_adjusted_confidence_threshold(self) -> float:
        """Get confidence threshold adjusted for current regime."""
        if not self.current_regime:
            return self.base_confidence_threshold
        
        # Lower threshold in strong trends
        if self.current_regime.primary_regime in [MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR]:
            return self.base_confidence_threshold * 0.9
        
        # Higher threshold in volatile regimes
        if self.current_regime.primary_regime in [MarketRegime.HIGH_VOL, MarketRegime.EXTREME_VOL]:
            return self.base_confidence_threshold * 1.1
        
        return self.base_confidence_threshold
    
    def _get_max_holding_period(self, regime: RegimeState) -> int:
        """Get maximum holding period based on regime."""
        # Shorter holding in volatile markets
        if regime.primary_regime in [MarketRegime.HIGH_VOL, MarketRegime.EXTREME_VOL]:
            return 30  # 30 minutes
        
        # Longer holding in trends
        if regime.primary_regime in [MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR]:
            return 240  # 4 hours
        
        # Default
        return 120  # 2 hours
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy-specific metrics."""
        metrics = super().get_metrics()
        
        # Add ML-specific metrics
        if self.current_regime:
            metrics.update({
                'current_regime': self.current_regime.primary_regime.value,
                'regime_confidence': self.current_regime.confidence,
                'volatility_percentile': self.current_regime.volatility_percentile,
                'trend_strength': self.current_regime.trend_strength,
            })
        
        # Add MTF metrics
        if self.mtf_signals:
            avg_alignment = np.mean([s.alignment_score for s in self.mtf_signals.values()])
            metrics['avg_timeframe_alignment'] = avg_alignment
        
        # Add model metrics
        if self.model_selector.performance_history:
            recent_accuracy = np.mean([p['accuracy'] for p in self.model_selector.performance_history[-10:]])
            metrics['recent_model_accuracy'] = recent_accuracy
        
        return metrics