"""Analytics module for RoboTrader.

This module provides comprehensive performance analytics and metrics
for trading strategies and portfolio performance.
"""

from .performance import PerformanceAnalyzer
from .risk_metrics import RiskAnalyzer
from .attribution import AttributionAnalyzer

__all__ = [
    "PerformanceAnalyzer",
    "RiskAnalyzer", 
    "AttributionAnalyzer",
]