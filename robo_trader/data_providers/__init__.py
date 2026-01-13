"""
Data providers module for market data abstraction.

This module provides a unified interface for fetching market data from
different providers (Polygon.io, IBKR, etc.) allowing the trading system
to switch between data sources without changing the core logic.
"""

from robo_trader.data_providers.base import DataProvider
from robo_trader.data_providers.polygon_provider import PolygonDataProvider

__all__ = ["DataProvider", "PolygonDataProvider"]
