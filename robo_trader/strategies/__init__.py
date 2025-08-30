"""
Strategy Framework for Robo Trader.

This package provides:
- Base strategy interface
- Common strategy implementations
- Signal generation and validation
- Strategy combination and ensemble methods
"""

# Import legacy function for backward compatibility
from ..legacy_strategies import sma_crossover_signals
from .framework import Signal, SignalType, Strategy, StrategyMetrics, StrategyState

# Lazy import ML strategy to avoid TensorFlow crashes
try:
    from .ml_strategy import MLStrategy

    ML_STRATEGY_AVAILABLE = True
except Exception:
    MLStrategy = None
    ML_STRATEGY_AVAILABLE = False

__all__ = [
    "Strategy",
    "Signal",
    "SignalType",
    "StrategyState",
    "StrategyMetrics",
    "ML_STRATEGY_AVAILABLE",
    # Legacy exports
    "sma_crossover_signals",
]

# Add MLStrategy to __all__ only if available
if ML_STRATEGY_AVAILABLE:
    __all__.append("MLStrategy")
