"""
Strategy Framework for Robo Trader.

This package provides:
- Base strategy interface
- Common strategy implementations
- Signal generation and validation
- Strategy combination and ensemble methods
"""

from .framework import Strategy, Signal, SignalType, StrategyState, StrategyMetrics

# Import legacy function for backward compatibility
from ..legacy_strategies import sma_crossover_signals

__all__ = [
    "Strategy",
    "Signal",
    "SignalType",
    "StrategyState",
    "StrategyMetrics",
    # Legacy exports
    "sma_crossover_signals",
]
