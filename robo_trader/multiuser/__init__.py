"""
Multi-portfolio / multiuser support for RoboTrader.

This package provides:
- PortfolioConfig: Per-portfolio configuration with risk overrides
- Legacy migration inspection with mutation entrypoints quarantined
- PortfolioScopedDB: Proxy that auto-injects portfolio_id into DB calls
- Portfolio registry: Load/save portfolio definitions
"""

from .db_proxy import PortfolioScopedDB
from .migration import LegacyMultiuserMigrationDisabled, MultiuserMigration
from .portfolio_config import PortfolioConfig, load_portfolio_configs

__all__ = [
    "PortfolioConfig",
    "load_portfolio_configs",
    "MultiuserMigration",
    "LegacyMultiuserMigrationDisabled",
    "PortfolioScopedDB",
]
