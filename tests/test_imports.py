"""Test that all required imports work correctly across platforms."""

import sys
import platform


def test_core_imports():
    """Test core package imports."""
    import robo_trader
    from robo_trader.config import load_config
    from robo_trader.database_async import AsyncTradingDatabase
    from robo_trader.execution import PaperExecutor
    from robo_trader.clients.async_ibkr_client import AsyncIBKRClient
    from robo_trader.portfolio import Portfolio
    from robo_trader.risk import RiskManager
    from robo_trader.legacy_strategies import sma_crossover_signals


def test_async_imports():
    """Test async module imports."""
    from robo_trader.clients import AsyncIBKRClient, ConnectionConfig
    from robo_trader.database_async import AsyncTradingDatabase
    from robo_trader.runner_async import AsyncRunner


def test_monitoring_imports():
    """Test monitoring module imports."""
    from robo_trader.monitoring.performance import PerformanceMonitor, Timer


def test_third_party_imports():
    """Test third-party library imports."""
    import asyncio
    import aiosqlite
    import nest_asyncio
    import pandas
    import numpy
    import pydantic
    import tenacity
    from ib_insync import IB


def test_platform_info():
    """Print platform information for debugging."""
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")


if __name__ == "__main__":
    test_core_imports()
    test_async_imports()
    test_monitoring_imports()
    test_third_party_imports()
    test_platform_info()
    print("All imports successful!")