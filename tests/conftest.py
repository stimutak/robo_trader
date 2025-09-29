"""Pytest fixtures and configuration for the test suite."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# Configure asyncio for testing
@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the asyncio event loop policy."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Connection test fixtures
@pytest.fixture
def host():
    """Host for connection tests."""
    return "127.0.0.1"


@pytest.fixture
def port():
    """Port for connection tests."""
    return 7497


@pytest.fixture
def client_id():
    """Client ID for connection tests."""
    return 999


# Correlation tracker fixtures
@pytest.fixture
def tracker():
    """Mock correlation tracker for testing."""
    from robo_trader.ml.correlation_tracking import CorrelationTracker

    tracker = CorrelationTracker(
        symbols=["AAPL", "MSFT", "GOOGL", "META"], lookback_window=30, update_interval=300
    )

    # Add mock data
    tracker.correlation_matrix = pd.DataFrame(
        np.array(
            [
                [1.00, 0.75, 0.60, 0.85],
                [0.75, 1.00, 0.55, 0.70],
                [0.60, 0.55, 1.00, 0.65],
                [0.85, 0.70, 0.65, 1.00],
            ]
        ),
        index=["AAPL", "MSFT", "GOOGL", "META"],
        columns=["AAPL", "MSFT", "GOOGL", "META"],
    )

    return tracker


@pytest.fixture
def sizer():
    """Mock position sizer for testing."""
    from robo_trader.ml.correlation_tracking import CorrelationAwarePositionSizer

    sizer = CorrelationAwarePositionSizer(
        base_position_size=100,
        max_correlation_threshold=0.7,
        scale_factor=0.5,
        min_position_size=10,
        max_portfolio_correlation=2.5,
    )

    return sizer


# Mock market data fixtures
@pytest.fixture
def mock_market_data():
    """Generate mock market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")

    data = pd.DataFrame(
        {
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(101, 201, 100),
            "low": np.random.uniform(99, 199, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    # Ensure high >= low and high >= open/close
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


@pytest.fixture
def mock_portfolio():
    """Mock portfolio for testing."""
    portfolio = MagicMock()
    portfolio.cash = Decimal("100000")
    portfolio.positions = {}
    portfolio.get_total_value = MagicMock(return_value=Decimal("100000"))
    portfolio.get_position = MagicMock(return_value=None)
    portfolio.update_position = MagicMock()

    return portfolio


@pytest.fixture
def mock_executor():
    """Mock executor for testing."""
    executor = MagicMock()
    executor.place_order = MagicMock(return_value=MagicMock(ok=True))
    executor.cancel_order = MagicMock(return_value=True)
    executor.get_order_status = MagicMock(return_value={"status": "filled"})

    return executor


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    from robo_trader.config import Config

    config = Config()
    config.symbols = ["AAPL", "MSFT", "GOOGL"]
    config.cash = 100000
    config.max_positions = 10
    config.position_size = 0.1
    config.stop_loss_pct = 0.02
    config.take_profit_pct = 0.05

    return config


# Async helper fixtures
@pytest.fixture
async def async_client():
    """Mock async client for testing."""
    client = MagicMock()
    client.connect = MagicMock(return_value=asyncio.Future())
    client.connect.return_value.set_result(True)
    client.disconnect = MagicMock(return_value=asyncio.Future())
    client.disconnect.return_value.set_result(True)
    client.is_connected = MagicMock(return_value=True)

    return client


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files(request):
    """Clean up test files after tests."""
    yield
    # Cleanup code here if needed
    pass


# Skip markers for certain tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_connection: mark test as requiring live connection"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
