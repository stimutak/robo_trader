"""
Feature engineering module for RoboTrader.
Provides technical indicators and feature calculation infrastructure.
"""

from robo_trader.features.base import (
    BaseFeatureCalculator,
    CompositeFeatureCalculator,
    FeatureSet,
    FeatureMetadata,
    FeatureValue,
    FeatureType,
    TimeFrame
)

from robo_trader.features.technical_indicators import (
    MomentumIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators
)

from robo_trader.features.feature_engine import FeatureEngine

__all__ = [
    # Base classes
    'BaseFeatureCalculator',
    'CompositeFeatureCalculator',
    'FeatureSet',
    'FeatureMetadata',
    'FeatureValue',
    'FeatureType',
    'TimeFrame',
    
    # Indicators
    'MomentumIndicators',
    'TrendIndicators',
    'VolatilityIndicators',
    'VolumeIndicators',
    
    # Engine
    'FeatureEngine'
]