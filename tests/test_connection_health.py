"""Tests for ConnectionHealth - centralized IBKR connection state gate.

Per the 2026-05-16 design spec, ConnectionHealth is the single decision
point for 'is the connection usable?', replacing health logic scattered
across subprocess_ibkr_client, connection_manager, and runner_async.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.connection_health import ConnectionHealth, HealthStatus


def make_fake_ib_client():
    """Return a fake IB client matching the SubprocessIBKRClient surface
    that ConnectionHealth needs: ping() -> bool, isConnected() -> bool."""
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.isConnected = MagicMock(return_value=True)
    return client


def test_initial_status_is_healthy():
    health = ConnectionHealth(ib_client=make_fake_ib_client())
    assert health.status is HealthStatus.HEALTHY
