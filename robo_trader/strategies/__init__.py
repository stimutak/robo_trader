"""
Strategy Framework for Robo Trader.

This package provides:
- Base strategy interface
- Common strategy implementations
- Signal generation and validation
- Strategy combination and ensemble methods
"""

from .framework import (
    Strategy,
    Signal,
    SignalType,
    StrategyState,
    StrategyMetrics
)

__all__ = [
    "Strategy",
    "Signal", 
    "SignalType",
    "StrategyState",
    "StrategyMetrics"
]