"""
Multi-portfolio / multiuser support for RoboTrader.

This package provides:
- PortfolioConfig: Per-portfolio configuration with risk overrides
- Database migration: Adds portfolio_id partitioning to existing tables
- PortfolioScopedDB: Proxy that auto-injects portfolio_id into DB calls
- Portfolio registry: Load/save portfolio definitions
"""

from .db_proxy import PortfolioScopedDB
from .migration import MultiuserMigration
from .portfolio_config import PortfolioConfig, load_portfolio_configs

__all__ = [
    "PortfolioConfig",
    "load_portfolio_configs",
    "MultiuserMigration",
    "PortfolioScopedDB",
]
