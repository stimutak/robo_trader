"""Centralized IBKR connection health gate.

Single decision point for 'is the connection usable?'. Replaces ad-hoc
health checks scattered across subprocess_ibkr_client.py,
connection_manager.py, and runner_async._monitor_subprocess_health.

Per 2026-05-16 design spec (docs/superpowers/specs/).
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class IBKRClientProtocol(Protocol):
    """Minimal interface ConnectionHealth requires from its IB client.

    The canonical conforming implementation is
    ``robo_trader.clients.subprocess_ibkr_client.SubprocessIBKRClient``.
    Documenting the surface explicitly (rather than ``Any``) prevents
    drift between the client and its only consumer here.

    ``runtime_checkable`` is set so callers and tests can validate
    conformance via ``isinstance(client, IBKRClientProtocol)``.
    """

    @property
    def is_connected(self) -> bool:
        """True iff the underlying API session is currently usable."""

    async def ping(self) -> bool:
        """Active probe — return True iff the API responds healthy."""


# H2: process-wide mutex that serializes Gateway recovery across all
# AsyncRunner instances (multi-portfolio mode runs N runners sharing one
# IBKR Gateway on port 4002). Without this lock, Portfolio A's
# recover_connection() can call _safe_disconnect() on the shared connection
# while Portfolio B is mid-recovery — producing the IBKR throttle cascade
# documented for 2026-05-13.
#
# Acquisition order (must not be violated to avoid deadlock):
#   1. AsyncRunner._recovery_lock  (per-instance, prevents one runner from
#      re-entering its own recovery while one is in flight)
#   2. _GATEWAY_RECOVERY_LOCK      (process-wide, prevents concurrent
#      recovery across runners)
#
# This is a no-op in single-portfolio mode (only one runner ever acquires
# it), so it's safe to leave on unconditionally.
_GATEWAY_RECOVERY_LOCK: asyncio.Lock = asyncio.Lock()


def get_gateway_recovery_lock() -> asyncio.Lock:
    """Return the process-wide Gateway recovery mutex.

    Exposed as a function (not a direct import) so tests can patch it
    via ``patch("robo_trader.connection_health._GATEWAY_RECOVERY_LOCK", ...)``
    and so callers always pick up the current module-level binding rather
    than a stale snapshot taken at import time.
    """
    return _GATEWAY_RECOVERY_LOCK


class HealthStatus(Enum):
    HEALTHY = "healthy"  # consecutive_failures < max_consecutive_failures
    UNHEALTHY = "unhealthy"  # consecutive_failures >= max_consecutive_failures
    RECOVERING = "recovering"  # recover_connection() is mid-flight


class ConnectionHealth:
    def __init__(
        self,
        ib_client: IBKRClientProtocol,
        ping_interval_seconds: float = 30,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._ib_client = ib_client
        self._ping_interval = ping_interval_seconds
        self._max_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._status: HealthStatus = HealthStatus.HEALTHY
        self._monitor_task: Optional[asyncio.Task] = None
        self._stopped = False
        self._on_unhealthy: Optional[Callable[[str], Awaitable[None]]] = None

    @property
    def status(self) -> HealthStatus:
        return self._status

    @property
    def consecutive_failures(self) -> int:
        """Read-only view of the consecutive-failure counter.

        Public, stable surface for tests and callers that need to observe
        recovery state without poking the private ``_consecutive_failures``
        attribute.
        """
        return self._consecutive_failures

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

    async def perform_check(self) -> HealthStatus:
        """Active probe: subprocess ping + ib.is_connected.

        Returns the current status after the probe. Mutates internal state
        via record_success() or record_failure().
        """
        try:
            is_connected = self._ib_client.is_connected
        except Exception as e:
            self.record_failure(e, "perform_check:is_connected")
            return self._status

        if not is_connected:
            self.record_failure(
                RuntimeError("ib.is_connected returned False"),
                "perform_check:not_connected",
            )
            return self._status

        try:
            ping_ok = await self._ib_client.ping()
        except Exception as e:
            self.record_failure(e, "perform_check:ping_exception")
            return self._status

        if ping_ok:
            self.record_success()
        else:
            self.record_failure(
                RuntimeError("ping returned False"),
                "perform_check:ping_falsy",
            )
        return self._status

    async def start_monitoring(
        self,
        on_unhealthy: Callable[[str], Awaitable[None]],
    ) -> None:
        """Spawn background task that calls perform_check() every
        ping_interval_seconds. On transition to UNHEALTHY, awaits
        on_unhealthy(reason)."""
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        self._on_unhealthy = on_unhealthy
        self._stopped = False
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Cancel the background monitor task. Idempotent."""
        self._stopped = True
        if self._monitor_task is None:
            return
        if not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """Survives perform_check exceptions — fails safe to UNHEALTHY,
        keeps trying. Never silently dies."""
        while not self._stopped:
            try:
                prev_status = self._status
                await self.perform_check()
                # C3: also skip firing on_unhealthy when we're already in
                # RECOVERING. Without this, a transient ping failure during
                # an in-progress recovery would queue a *second* recovery
                # task after the first releases _recovery_lock. The
                # RECOVERING state is set by recover_connection() at the top
                # of its critical section and cleared by record_success()
                # on successful re-init.
                if (
                    prev_status is not HealthStatus.UNHEALTHY
                    and prev_status is not HealthStatus.RECOVERING
                    and self._status is HealthStatus.UNHEALTHY
                    and self._on_unhealthy is not None
                ):
                    try:
                        await self._on_unhealthy(
                            f"perform_check transitioned to UNHEALTHY after "
                            f"{self._consecutive_failures} consecutive failures"
                        )
                    except Exception:
                        logger.exception("on_unhealthy callback raised; continuing monitor loop")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # C3: route the fail-safe through record_failure() so the
                # counter increments symmetrically (3 strikes to UNHEALTHY)
                # instead of jumping straight to UNHEALTHY on any crash.
                # The asymmetric direct-write previously could queue an
                # extra on_unhealthy callback during recovery (the prev/
                # current status guard above closes the other window).
                logger.exception("Health monitor iteration crashed - recording as failure")
                self.record_failure(e, "monitor_loop_crash")
            try:
                await asyncio.sleep(self._ping_interval)
            except asyncio.CancelledError:
                break
