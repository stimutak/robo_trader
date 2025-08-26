"""
Backtesting framework for strategy validation.

This package provides:
- Event-driven backtesting engine
- Historical data replay
- Performance metrics calculation
- Transaction cost modeling
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics import PerformanceMetrics, calculate_metrics

__all__ = [
    "BacktestEngine",
    "BacktestConfig", 
    "BacktestResult",
    "PerformanceMetrics",
    "calculate_metrics"
]