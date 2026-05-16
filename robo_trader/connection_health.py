"""Centralized IBKR connection health gate.

Single decision point for 'is the connection usable?'. Replaces ad-hoc
health checks scattered across subprocess_ibkr_client.py,
connection_manager.py, and runner_async._monitor_subprocess_health.

Per 2026-05-16 design spec (docs/superpowers/specs/).
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class HealthStatus(Enum):
    HEALTHY = "healthy"          # consecutive_failures < max_consecutive_failures
    UNHEALTHY = "unhealthy"      # consecutive_failures >= max_consecutive_failures
    RECOVERING = "recovering"    # recover_connection() is mid-flight


class ConnectionHealth:
    def __init__(
        self,
        ib_client: Any,
        ping_interval_seconds: int = 30,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._ib_client = ib_client
        self._ping_interval = ping_interval_seconds
        self._max_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._status: HealthStatus = HealthStatus.HEALTHY

    @property
    def status(self) -> HealthStatus:
        return self._status
