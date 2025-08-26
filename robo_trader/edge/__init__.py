"""
Edge calculation and trade quality assessment.

This package provides:
- Expected Value (EV) calculation
- Risk:reward ratio analysis
- Win probability estimation
- Trade quality scoring
"""

from .calculator import EdgeCalculator, EdgeMetrics, TradeEdge
from .gating import EdgeGatingFilter, GatingConfig

__all__ = [
    "EdgeCalculator",
    "TradeEdge",
    "EdgeMetrics",
    "EdgeGatingFilter",
    "GatingConfig",
]
