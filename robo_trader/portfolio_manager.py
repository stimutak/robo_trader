"""Compatibility re-export for portfolio manager.

Some tests import from `robo_trader.portfolio_manager` while others use
`robo_trader.portfolio.portfolio_manager`. This module re-exports the public
API to support both import paths.
"""

from .portfolio_manager.portfolio_manager import (
    AllocationMethod,
    MultiStrategyPortfolioManager,
    PortfolioMetrics,
    StrategyAllocation,
)

__all__ = [
    "AllocationMethod",
    "StrategyAllocation",
    "PortfolioMetrics",
    "MultiStrategyPortfolioManager",
]
