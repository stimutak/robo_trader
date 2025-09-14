"""Backtesting framework for strategy validation and optimization."""

from .engine import BacktestEngine
from .execution_simulator import ExecutionSimulator, MarketImpactModel
from .walk_forward_optimization import WalkForwardOptimizer

__all__ = [
    "BacktestEngine",
    "ExecutionSimulator",
    "MarketImpactModel",
    "WalkForwardOptimizer",
]
