"""
Integrated feature pipeline combining technical, ML, and microstructure features.

This module implements:
- Unified feature calculation
- Feature store integration
- Real-time and batch processing
- Feature selection and filtering
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from ..config import Config
from ..logger import get_logger
from .engine import FeatureEngine, FeatureSet
from .feature_store import FeatureMetadata, FeatureStore
from .ml_features import MLFeatureEngine


class FeaturePipeline:
    """Unified feature pipeline for ML trading."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("features.pipeline")
        
        # Initialize components
        self.feature_engine = FeatureEngine(config)
        self.ml_engine = MLFeatureEngine(lookback_window=100)
        self.feature_store = FeatureStore(
            db_path="feature_store.db",
            max_versions=10
        )
        
        # Feature configuration
        self.enabled_features: Set[str] = set()
        self.feature_groups = {
            'technical': True,
            'ml': True,
            'microstructure': True,
            'cross_asset': True,
            'sentiment': True
        }
        
        # Performance tracking
        self.metrics = {
            'features_calculated': 0,
            'pipeline_latency_ms': [],
            'feature_count': 0
        }
        
        self._initialize_feature_config()
    
    def _initialize_feature_config(self) -> None:
        """Initialize feature configuration."""
        # Technical features (from FeatureSet)
        if self.feature_groups['technical']:
            self.enabled_features.update([
                'returns_1m', 'returns_5m', 'returns_15m', 'returns_1h',
                'log_returns', 'rsi', 'macd_line', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth',
                'atr', 'vwap', 'obv', 'volume_ratio',
                'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
                'momentum_1d', 'momentum_5d', 'roc',
                'historical_volatility', 'volatility_ratio',
                'trend_strength', 'mean_reversion_signal', 'breakout_signal'
            ])
        
        # ML features
        if self.feature_groups['ml']:
            self.enabled_features.update([
                'regime_trending_up', 'regime_trending_down', 'regime_ranging',
                'regime_volatile', 'regime_confidence', 'regime_change_probability',
                'volatility_percentile', 'trend_strength'
            ])
        
        # Cross-asset features
        if self.feature_groups['cross_asset']:
            self.enabled_features.update([
                'correlation_spy', 'avg_correlation', 'max_correlation',
                'min_correlation', 'beta', 'pca_loading_1', 'pca_loading_2',
                'pca_variance_ratio_1'
            ])
        
        # Microstructure features
        if self.feature_groups['microstructure']:
            self.enabled_features.update([
                'spread_bps', 'bid_ask_imbalance', 'tick_direction',
                'order_flow_imbalance', 'buy_ratio', 'avg_trade_size',
                'large_trade_ratio', 'quote_imbalance', 'effective_spread',
                'price_impact'
            ])
        
        # Sentiment features
        if self.feature_groups['sentiment']:
            self.enabled_features.update([
                'return_skewness', 'return_kurtosis', 'volume_return_correlation',
                'adl_trend', 'adl_strength', 'price_position', 'momentum_divergence'
            ])
    
    async def calculate_features(
        self,
        symbol: str,
        price_data: Optional[pd.DataFrame] = None,
        cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
        trades: Optional[List[Dict]] = None,
        quotes: Optional[List[Dict]] = None,
        store_features: bool = True
    ) -> pd.DataFrame:
        """Calculate all enabled features for a symbol."""
        start_time = datetime.now()
        
        try:
            all_features = {}
            
            # Get technical features from feature engine
            if self.feature_groups['technical']:
                technical_features = await self.feature_engine.calculate_features(symbol)
                if technical_features:
                    tech_dict = technical_features.get_non_null_features()
                    all_features.update(tech_dict)
            
            # Calculate ML features if we have price data
            if price_data is not None and len(price_data) > 0:
                ml_features = self.ml_engine.calculate_all_ml_features(
                    symbol=symbol,
                    df=price_data,
                    price_data=cross_asset_data,
                    trades=trades,
                    quotes=quotes
                )
                all_features.update(ml_features)
            
            # Filter to enabled features only
            filtered_features = {
                k: v for k, v in all_features.items()
                if k in self.enabled_features
            }
            
            # Convert to DataFrame
            features_df = pd.DataFrame([filtered_features])
            features_df['symbol'] = symbol
            features_df['timestamp'] = datetime.now()
            
            # Store in feature store
            if store_features and len(features_df) > 0:
                calc_time = (datetime.now() - start_time).total_seconds() * 1000
                metadata = FeatureMetadata(
                    feature_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    timestamp=datetime.now(),
                    version=1,
                    num_features=len(filtered_features),
                    feature_names=list(filtered_features.keys()),
                    calculation_time_ms=calc_time,
                    data_quality_score=self._calculate_quality_score(filtered_features)
                )
                
                await self.feature_store.store_features(
                    symbol=symbol,
                    features=features_df,
                    metadata=metadata
                )
            
            # Update metrics
            self.metrics['features_calculated'] += 1
            self.metrics['feature_count'] = len(filtered_features)
            pipeline_latency = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics['pipeline_latency_ms'].append(pipeline_latency)
            if len(self.metrics['pipeline_latency_ms']) > 100:
                self.metrics['pipeline_latency_ms'] = self.metrics['pipeline_latency_ms'][-100:]
            
            self.logger.debug(
                f"Calculated {len(filtered_features)} features for {symbol} "
                f"in {pipeline_latency:.2f}ms"
            )
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature calculation error for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_features_for_training(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get historical features for model training."""
        all_features = []
        
        for symbol in symbols:
            features = await self.feature_store.get_feature_history(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                feature_names=feature_names
            )
            if not features.empty:
                features['symbol'] = symbol
                all_features.append(features)
        
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame()
    
    async def get_top_features(
        self,
        n: int = 50,
        model_name: Optional[str] = None
    ) -> List[str]:
        """Get top N most important features."""
        top_features = await self.feature_store.get_top_features(n, model_name)
        return [f[0] for f in top_features]
    
    async def update_feature_importance(
        self,
        importance_scores: Dict[str, float],
        model_name: str
    ) -> None:
        """Update feature importance from ML model."""
        await self.feature_store.update_feature_importance(
            importance_scores=importance_scores,
            model_name=model_name
        )
        
        # Optionally update enabled features based on importance
        if len(importance_scores) > 0:
            # Get top 80% of features by importance
            sorted_features = sorted(
                importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            cumulative_importance = 0
            total_importance = sum(importance_scores.values())
            
            selected_features = []
            for feature, importance in sorted_features:
                cumulative_importance += importance
                selected_features.append(feature)
                if cumulative_importance / total_importance >= 0.8:
                    break
            
            self.logger.info(
                f"Selected {len(selected_features)} features covering 80% importance"
            )
    
    def _calculate_quality_score(self, features: Dict[str, Any]) -> float:
        """Calculate data quality score."""
        if not features:
            return 0.0
        
        # Count non-null features
        non_null_count = sum(1 for v in features.values() if v is not None)
        
        # Check for invalid values
        invalid_count = 0
        for value in features.values():
            if value is not None:
                try:
                    if pd.isna(value) or pd.isinf(value):
                        invalid_count += 1
                except:
                    pass
        
        # Calculate score
        completeness = non_null_count / len(features) if features else 0
        validity = 1 - (invalid_count / len(features)) if features else 0
        
        return (completeness + validity) / 2
    
    def enable_feature_group(self, group: str, enabled: bool = True) -> None:
        """Enable or disable a feature group."""
        if group in self.feature_groups:
            self.feature_groups[group] = enabled
            self._initialize_feature_config()
            self.logger.info(f"Feature group '{group}' {'enabled' if enabled else 'disabled'}")
    
    def set_enabled_features(self, features: List[str]) -> None:
        """Set specific features to be calculated."""
        self.enabled_features = set(features)
        self.logger.info(f"Enabled {len(features)} specific features")
    
    async def start(self) -> None:
        """Start the feature pipeline."""
        await self.feature_engine.start()
        self.logger.info("Feature pipeline started")
    
    async def stop(self) -> None:
        """Stop the feature pipeline."""
        await self.feature_engine.stop()
        self.logger.info("Feature pipeline stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        import numpy as np
        
        avg_latency = np.mean(self.metrics['pipeline_latency_ms']) \
                     if self.metrics['pipeline_latency_ms'] else 0
        
        return {
            'features_calculated': self.metrics['features_calculated'],
            'avg_pipeline_latency_ms': avg_latency,
            'feature_count': self.metrics['feature_count'],
            'enabled_features': len(self.enabled_features),
            'feature_groups': self.feature_groups,
            'feature_engine_metrics': self.feature_engine.get_metrics(),
            'feature_store_metrics': self.feature_store.get_metrics()
        }