"""Centralized IBKR connection health gate.

Single decision point for 'is the connection usable?'. Replaces ad-hoc
health checks scattered across subprocess_ibkr_client.py,
connection_manager.py, and runner_async._monitor_subprocess_health.

Per 2026-05-16 design spec (docs/superpowers/specs/).
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


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

    def record_failure(self, error: BaseException, context: str) -> None:
        """Report a transient failure (e.g., Errno 54 in a cycle).

        Increments the consecutive-failure counter. If the counter crosses
        max_consecutive_failures, status transitions to UNHEALTHY and
        background monitor (when wired) will trigger recovery.
        """
        self._consecutive_failures += 1
        prev_status = self._status
        if self._consecutive_failures >= self._max_failures:
            self._status = HealthStatus.UNHEALTHY
        if prev_status is not self._status:
            logger.info(
                "event=connection_state_change from=%s to=%s reason=%r context=%s",
                prev_status.value,
                self._status.value,
                str(error),
                context,
            )

    def record_success(self) -> None:
        """Report a successful API exchange. Resets failure counter."""
        if self._consecutive_failures == 0 and self._status is HealthStatus.HEALTHY:
            return
        prev_status = self._status
        self._consecutive_failures = 0
        if self._status in (HealthStatus.UNHEALTHY, HealthStatus.RECOVERING):
            self._status = HealthStatus.HEALTHY
        if prev_status is not self._status:
            logger.info(
                "event=connection_state_change from=%s to=%s reason=record_success",
                prev_status.value,
                self._status.value,
            )
