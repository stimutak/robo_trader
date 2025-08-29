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
from .framework import (Signal, SignalType, Strategy, StrategyMetrics,
                        StrategyState)
from .ml_strategy import MLStrategy

__all__ = [
    "Strategy",
    "Signal",
    "SignalType",
    "StrategyState",
    "StrategyMetrics",
    "MLStrategy",
    # Legacy exports
    "sma_crossover_signals",
]
