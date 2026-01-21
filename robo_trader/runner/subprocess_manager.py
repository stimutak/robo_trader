"""
Subprocess Manager Module - IBKR subprocess worker lifecycle management.

This module handles:
- Health monitoring of the IBKR subprocess worker
- Automatic restart on failure
- Lock file cleanup

Extracted from runner_async.py to improve modularity.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import TYPE_CHECKING, Any, Optional

from ..logger import get_logger

if TYPE_CHECKING:
    from ..config import TradingConfig

logger = get_logger(__name__)


class SubprocessManager:
    """Manages the IBKR subprocess worker lifecycle."""

    def __init__(
        self,
        config: TradingConfig,
        max_failures: int = 3,
        health_check_interval: int = 60,
        lock_file_path: Optional[str] = None,
    ):
        """
        Initialize the SubprocessManager.

        Args:
            config: Trading configuration with IBKR settings
            max_failures: Consecutive failures before restart
            health_check_interval: Seconds between health checks
            lock_file_path: Path to connection lock file
        """
        self.cfg = config
        self.max_failures = max_failures
        self.health_check_interval = health_check_interval
        self.lock_file_path = lock_file_path or os.environ.get(
            "IBKR_LOCK_FILE_PATH", "/tmp/ibkr_connect.lock"  # nosec B108
        )
        self.ib_client: Optional[Any] = None
        self._health_task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0

    def set_client(self, client: Any) -> None:
        """Set the IBKR client to monitor."""
        self.ib_client = client

    def get_client(self) -> Optional[Any]:
        """Get the current IBKR client."""
        return self.ib_client

    async def start_health_monitoring(self) -> asyncio.Task:
        """Start the background health monitoring task."""
        if self._health_task and not self._health_task.done():
            logger.warning("Health monitoring already running")
            return self._health_task

        self._health_task = asyncio.create_task(self._monitor_health())
        logger.info(
            f"Started subprocess health monitoring "
            f"({self.health_check_interval}s interval, {self.max_failures} failures to restart)"
        )
        return self._health_task

    async def stop_health_monitoring(self) -> None:
        """Stop the background health monitoring task."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            logger.info("Subprocess health monitoring stopped")

    async def _monitor_health(self) -> None:
        """
        Background task to monitor subprocess health.

        Pings the subprocess every interval and automatically restarts
        if the subprocess becomes unresponsive after multiple failures.
        """
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Only monitor if using subprocess client
                if not self.ib_client or not hasattr(self.ib_client, "ping"):
                    logger.debug("Not using subprocess client, skipping health check")
                    continue

                # Ping subprocess
                logger.debug("Pinging subprocess for health check...")
                is_healthy = await self.ib_client.ping()

                if not is_healthy:
                    self._consecutive_failures += 1
                    logger.warning(
                        f"Subprocess health check failed "
                        f"({self._consecutive_failures}/{self.max_failures})"
                    )

                    if self._consecutive_failures >= self.max_failures:
                        logger.error(
                            f"Subprocess unresponsive after {self.max_failures} "
                            "consecutive failures - restarting"
                        )
                        await self.restart_subprocess()
                        self._consecutive_failures = 0
                else:
                    if self._consecutive_failures > 0:
                        logger.info("Subprocess health check recovered")
                    self._consecutive_failures = 0
                    logger.debug("Subprocess health check passed")

            except asyncio.CancelledError:
                logger.info("Subprocess health monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in subprocess health monitoring: {e}")
                self._consecutive_failures += 1
                continue

    async def restart_subprocess(self) -> bool:
        """
        Restart the IBKR subprocess after a crash or health check failure.

        Returns:
            True if restart was successful, False otherwise
        """
        logger.warning("⚠️ Restarting IBKR subprocess due to health check failure")

        try:
            # Clean up stale lock files first
            self._cleanup_lock_file()

            # Stop the old subprocess
            if self.ib_client and hasattr(self.ib_client, "stop"):
                try:
                    await asyncio.wait_for(self.ib_client.stop(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Subprocess stop timed out, continuing with restart")
                except Exception as e:
                    logger.warning(f"Error stopping subprocess: {e}")

            # Kill any orphaned worker processes
            self._kill_orphaned_workers()
            await asyncio.sleep(1)  # Give time for cleanup

            # Reconnect using robust connection
            from ..utils.robust_connection import CircuitBreakerConfig, connect_ibkr_robust

            circuit_config = CircuitBreakerConfig(
                failure_threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
                recovery_timeout=float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "300")),
                success_threshold=2,
            )

            # Reconnect
            host = self.cfg.ibkr.host
            port = self.cfg.ibkr.port
            logger.info(f"Reconnecting to IBKR at {host}:{port}")

            self.ib_client = await connect_ibkr_robust(
                host=host,
                port=port,
                client_id=self.cfg.ibkr.client_id,
                readonly=self.cfg.ibkr.readonly,
                timeout=self.cfg.ibkr.timeout,
                max_retries=3,
                circuit_breaker_config=circuit_config,
                ssl_mode=self.cfg.ibkr.ssl_mode,
            )

            logger.info("✅ Subprocess restarted successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to restart subprocess: {e}")
            logger.warning("Will retry on next health check cycle")
            return False

    def _cleanup_lock_file(self) -> None:
        """Remove stale lock file if it exists."""
        try:
            if os.path.exists(self.lock_file_path):
                os.remove(self.lock_file_path)
                logger.info(f"Removed stale lock file: {self.lock_file_path}")
        except Exception as e:
            logger.warning(f"Could not remove lock file: {e}")

    def _kill_orphaned_workers(self) -> None:
        """Kill any orphaned IBKR subprocess workers."""
        try:
            subprocess.run(
                ["pkill", "-9", "-f", "ibkr_subprocess_worker"],
                capture_output=True,
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"Error killing orphaned workers: {e}")

    @property
    def consecutive_failures(self) -> int:
        """Get the current consecutive failure count."""
        return self._consecutive_failures
