"""
Multi-portfolio / multiuser support for RoboTrader.

This package provides:
- PortfolioConfig: Per-portfolio configuration with risk overrides
- Database migration: Adds portfolio_id partitioning to existing tables
- Portfolio registry: Load/save portfolio definitions
"""

from .portfolio_config import PortfolioConfig, load_portfolio_configs
from .migration import MultiuserMigration

__all__ = [
    "PortfolioConfig",
    "load_portfolio_configs",
    "MultiuserMigration",
]
