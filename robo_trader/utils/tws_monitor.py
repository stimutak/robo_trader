"""
TWS Recovery Monitor.

This module handles TWS stuck state detection and recovery coordination.
Since TWS requires manual login, this provides:
1. Early detection of stuck TWS API
2. Clear alerts to user
3. Graceful trading system pause
4. Monitoring mode to auto-resume when TWS is manually restarted
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Callable, Literal, Optional

from ..logger import get_logger
from .tws_health import check_tws_api_health, diagnose_tws_connection

logger = get_logger(__name__)


class TWSState(Enum):
    """TWS connection states."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    STUCK = "stuck"  # API not responding to handshakes
    OFFLINE = "offline"  # Port not listening


class TWSMonitor:
    """
    Monitor TWS health and coordinate recovery.

    Since TWS requires manual restart (login credentials), this monitor:
    - Detects stuck TWS state BEFORE creating zombie connections
    - Alerts user with clear instructions
    - Optionally waits in monitoring mode for manual TWS restart
    - Auto-resumes trading when TWS recovers
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        health_check_interval: float = 30.0,
        recovery_check_interval: float = 10.0,
        alert_callback: Optional[Callable[[str, str], None]] = None,
        ssl_mode: Literal["auto", "require", "disabled"] = "auto",
    ):
        """
        Initialize TWS monitor.

        Args:
            host: TWS host
            port: TWS port
            health_check_interval: Seconds between health checks when healthy
            recovery_check_interval: Seconds between checks when waiting for recovery
            alert_callback: Function to call for alerts: alert_callback(level, message)
                          Levels: "critical", "warning", "info"
            ssl_mode: Socket transport to use when probing the API (aligned with trading config)
        """
        self.host = host
        self.port = port
        self.health_check_interval = health_check_interval
        self.recovery_check_interval = recovery_check_interval
        self.alert_callback = alert_callback or self._default_alert
        self.ssl_mode = ssl_mode

        self.state = TWSState.UNKNOWN
        self.last_check_time: Optional[datetime] = None
        self.consecutive_failures = 0
        self.stuck_since: Optional[datetime] = None

        self._monitor_task: Optional[asyncio.Task] = None
        self._recovery_mode = False

    def _default_alert(self, level: str, message: str):
        """Default alert handler - just log."""
        if level == "critical":
            logger.error(f"🚨 TWS ALERT: {message}")
        elif level == "warning":
            logger.warning(f"⚠️ TWS WARNING: {message}")
        else:
            logger.info(f"ℹ️ TWS INFO: {message}")

    async def check_health(self) -> bool:
        """
        Check TWS health once.

        Returns:
            True if healthy, False otherwise
        """
        self.last_check_time = datetime.now()

        # Get diagnosis
        diagnosis = await diagnose_tws_connection(self.host, self.port, self.ssl_mode)

        if not diagnosis["port_listening"]:
            self._update_state(TWSState.OFFLINE)
            return False

        if diagnosis["api_healthy"]:
            self._update_state(TWSState.HEALTHY)
            return True
        else:
            # API not healthy - check if it's stuck (handshake timeout)
            if "timeout" in diagnosis["status_message"].lower():
                self._update_state(TWSState.STUCK)
            else:
                self._update_state(TWSState.OFFLINE)
            return False

    def _update_state(self, new_state: TWSState):
        """Update state and trigger alerts if needed."""
        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state

        logger.info(f"TWS state change: {old_state.value} → {new_state.value}")

        # Handle state transitions
        if new_state == TWSState.STUCK:
            self.consecutive_failures += 1
            if self.stuck_since is None:
                self.stuck_since = datetime.now()

            self.alert_callback(
                "critical",
                f"TWS API STUCK - Handshake timeout detected!\n"
                f"Action required: Please restart TWS manually (requires login)\n"
                f"Trading system will pause until TWS recovers.\n"
                f"Port {self.port} is listening but API not responding.",
            )

        elif new_state == TWSState.OFFLINE:
            self.consecutive_failures += 1
            self.alert_callback(
                "critical",
                f"TWS OFFLINE - Port {self.port} not responding!\n"
                f"Action required: Start TWS or TWS Gateway\n"
                f"Trading system paused.",
            )

        elif new_state == TWSState.HEALTHY:
            if old_state in [TWSState.STUCK, TWSState.OFFLINE]:
                self.alert_callback(
                    "info",
                    f"✅ TWS RECOVERED - API responding normally!\n" f"Trading system can resume.",
                )
            self.consecutive_failures = 0
            self.stuck_since = None

    async def wait_for_recovery(
        self, timeout: Optional[float] = None, max_checks: Optional[int] = None
    ) -> bool:
        """
        Wait for TWS to recover (monitoring mode).

        This enters a monitoring loop, checking TWS health periodically
        until it recovers or timeout/max_checks reached.

        Args:
            timeout: Maximum seconds to wait (None = wait indefinitely)
            max_checks: Maximum number of checks (None = unlimited)

        Returns:
            True if TWS recovered, False if timeout/max_checks reached
        """
        self._recovery_mode = True
        start_time = datetime.now()
        check_count = 0

        logger.info(
            f"🔄 Entering TWS recovery monitoring mode "
            f"(timeout={timeout}s, max_checks={max_checks})"
        )

        try:
            while True:
                # Check health
                is_healthy = await self.check_health()

                if is_healthy:
                    logger.info("✅ TWS recovered! Exiting recovery mode.")
                    return True

                check_count += 1

                # Check limits
                if max_checks and check_count >= max_checks:
                    logger.warning(f"Max checks ({max_checks}) reached in recovery mode")
                    return False

                if timeout:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= timeout:
                        logger.warning(f"Timeout ({timeout}s) reached in recovery mode")
                        return False

                # Wait before next check
                logger.info(
                    f"TWS still {self.state.value} - waiting {self.recovery_check_interval}s "
                    f"for manual restart (check {check_count})"
                )
                await asyncio.sleep(self.recovery_check_interval)

        finally:
            self._recovery_mode = False

    def start_monitoring(self):
        """Start background health monitoring task."""
        if self._monitor_task and not self._monitor_task.done():
            logger.warning("Monitoring already running")
            return

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started TWS health monitoring")

    def stop_monitoring(self):
        """Stop background health monitoring."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            logger.info("Stopped TWS health monitoring")

    async def _monitor_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await self.check_health()

                # If stuck/offline, enter recovery mode
                if self.state in [TWSState.STUCK, TWSState.OFFLINE]:
                    logger.warning(
                        f"TWS {self.state.value} - pausing monitoring, "
                        "waiting for manual recovery"
                    )
                    # Wait for recovery before resuming monitoring
                    recovered = await self.wait_for_recovery(timeout=300)  # 5 min max wait
                    if not recovered:
                        logger.error("TWS did not recover within timeout")

                # Wait before next check
                interval = (
                    self.recovery_check_interval
                    if self._recovery_mode
                    else self.health_check_interval
                )
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TWS monitoring loop: {e}")
                await asyncio.sleep(5)

    def get_status(self) -> dict:
        """Get current monitor status."""
        return {
            "state": self.state.value,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "consecutive_failures": self.consecutive_failures,
            "stuck_since": self.stuck_since.isoformat() if self.stuck_since else None,
            "recovery_mode": self._recovery_mode,
            "monitoring_active": self._monitor_task is not None and not self._monitor_task.done(),
        }


async def quick_tws_check_before_connect(
    host: str = "127.0.0.1",
    port: int = 7497,
    ssl_mode: Literal["auto", "require", "disabled"] = "auto",
) -> tuple[bool, str]:
    """
    Quick pre-flight check before attempting connection.

    Use this BEFORE entering retry loops to prevent zombie accumulation.

    Returns:
        Tuple of (should_proceed, reason)
        If should_proceed is False, DO NOT attempt connection.

    Example:
        ```python
        should_connect, reason = await quick_tws_check_before_connect()
        if not should_connect:
            raise ConnectionError(f"TWS not ready: {reason}")

        # Only proceed if health check passed
        ib = await connect_ibkr_robust(max_retries=2)  # Reduced retries
        ```
    """
    healthy, message = await check_tws_api_health(host, port, timeout=2.0, ssl_mode=ssl_mode)

    if healthy:
        return True, "TWS API healthy"

    # Not healthy - diagnose why
    if "timeout" in message.lower():
        return False, (
            f"TWS API STUCK (handshake timeout). "
            f"Restart TWS manually to clear stuck state. "
            f"Port {port} is listening but API not responding."
        )
    elif "refused" in message.lower():
        return False, f"TWS not running. Start TWS or Gateway on port {port}."
    else:
        return False, f"TWS not ready: {message}"


# Example usage
async def main():
    """Test TWS monitoring."""
    print("=" * 70)
    print("TWS Health Monitor Test")
    print("=" * 70)

    # Custom alert handler
    def my_alert_handler(level: str, message: str):
        """Custom alert handler."""
        print(f"\n{'=' * 70}")
        print(f"ALERT [{level.upper()}]:")
        print(message)
        print("=" * 70)

    # Create monitor
    monitor = TWSMonitor(alert_callback=my_alert_handler)

    # Check health once
    print("\n1. Single health check:")
    is_healthy = await monitor.check_health()
    print(f"   Result: {'✅ HEALTHY' if is_healthy else '❌ UNHEALTHY'}")
    print(f"   Status: {monitor.get_status()}")

    # Test quick pre-flight check
    print("\n2. Quick pre-flight check (use before connection attempts):")
    should_proceed, reason = await quick_tws_check_before_connect()
    print(f"   Should connect: {should_proceed}")
    print(f"   Reason: {reason}")

    if not should_proceed:
        print("\n   ⚠️ DO NOT attempt connection - would create zombie connections!")
        print("   User must restart TWS manually first.")


if __name__ == "__main__":
    asyncio.run(main())
