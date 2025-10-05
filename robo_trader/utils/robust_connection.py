"""
Robust IBKR connection utilities with exponential backoff and circuit breaker.

This module provides enhanced connection resilience for IBKR connections:
- Exponential backoff retry with jitter
- Circuit breaker pattern to prevent connection storms
- Connection health monitoring
- Automatic reconnection on failure
"""

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from ..logger import get_logger

logger = get_logger(__name__)


def kill_zombie_connections(port: int = 7497) -> None:
    """Kill zombie CLOSE_WAIT connections on specified port.

    This clears leftover connections from failed TWS API handshake attempts.
    """
    try:
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "kill_zombies.sh")
        if os.path.exists(script_path):
            result = subprocess.run([script_path], capture_output=True, text=True, timeout=5)
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    logger.info(f"Zombie killer: {line}")
        else:
            # Fallback: Check manually
            result = subprocess.run(["netstat", "-an"], capture_output=True, text=True, timeout=5)
            zombie_count = len(
                [
                    line
                    for line in result.stdout.split("\n")
                    if str(port) in line and "CLOSE_WAIT" in line
                ]
            )
            if zombie_count > 0:
                logger.warning(f"Found {zombie_count} zombie connections on port {port}")
    except Exception as e:
        logger.debug(f"Could not check zombie connections: {e}")


T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, allowing connections
    OPEN = "open"  # Too many failures, blocking connections
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 2  # Successes in half-open before closing


@dataclass
class CircuitBreaker:
    """Circuit breaker for connection management."""

    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)

    def record_success(self) -> None:
        """Record a successful connection."""
        logger.debug(f"Circuit breaker: Recording success (state={self.state.value})")

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset on success

    def record_failure(self) -> None:
        """Record a failed connection."""
        logger.debug(f"Circuit breaker: Recording failure (state={self.state.value})")

        self.last_failure_time = datetime.now()
        self.failure_count += 1

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            self._open_circuit()

    def can_attempt_connection(self) -> bool:
        """Check if connection attempt is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.recovery_timeout:
                    self._half_open_circuit()
                    return True
            return False

        # HALF_OPEN state
        return True

    def _open_circuit(self) -> None:
        """Open the circuit breaker."""
        logger.warning(
            f"Circuit breaker OPENED after {self.failure_count} failures. "
            f"Blocking connections for {self.config.recovery_timeout}s"
        )
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.now()
        self.success_count = 0

    def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        logger.info("Circuit breaker CLOSED - Normal operation resumed")
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.failure_count = 0
        self.success_count = 0

    def _half_open_circuit(self) -> None:
        """Set circuit to half-open for testing."""
        logger.info("Circuit breaker HALF-OPEN - Testing connection recovery")
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.now()
        self.success_count = 0

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
        }


def kill_tws_zombie_connections(port: int = 7497) -> tuple[bool, str]:
    """
    Detect and kill CLOSE_WAIT zombie connections on TWS port.

    This clears kernel-level TCP state that prevents new connections.

    Returns:
        tuple: (success, message)
    """
    try:
        # Step 1: Find processes with CLOSE_WAIT connections on this port
        lsof_result = subprocess.run(
            ["lsof", "-i", f"tcp:{port}", "-sTCP:CLOSE_WAIT"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if not lsof_result.stdout:
            logger.info(f"✅ No CLOSE_WAIT zombies found on port {port}")
            return True, "No zombies detected"

        # Step 2: Extract PIDs from lsof output
        pids = set()
        for line in lsof_result.stdout.splitlines()[1:]:  # Skip header
            parts = line.split()
            if len(parts) > 1:
                try:
                    pid = int(parts[1])
                    pids.add(pid)
                except ValueError:
                    continue

        if not pids:
            logger.warning("Found CLOSE_WAIT connections but could not extract PIDs")
            return False, "Could not identify zombie processes"

        # Step 3: Kill the zombie processes
        killed_count = 0
        for pid in pids:
            try:
                subprocess.run(["kill", "-9", str(pid)], timeout=2)
                killed_count += 1
                logger.info(f"Killed zombie process {pid}")
            except Exception as e:
                logger.warning(f"Could not kill process {pid}: {e}")

        # Step 4: Verify cleanup
        time.sleep(0.5)  # Give kernel time to clean up
        verify_result = subprocess.run(
            ["netstat", "-an"], capture_output=True, text=True, timeout=5
        )

        remaining_zombies = 0
        for line in verify_result.stdout.splitlines():
            if str(port) in line and "CLOSE_WAIT" in line:
                remaining_zombies += 1

        if remaining_zombies == 0:
            msg = f"✅ Successfully killed {killed_count} zombie process(es). Port {port} is clean."
            logger.info(msg)
            return True, msg
        else:
            msg = f"⚠️ Killed {killed_count} processes but {remaining_zombies} zombies remain"
            logger.warning(msg)
            return False, msg

    except FileNotFoundError:
        logger.warning("lsof command not found - cannot kill zombies automatically")
        return False, "lsof not available"
    except Exception as e:
        logger.error(f"Error killing zombie connections: {e}")
        return False, f"Error: {e}"


def check_tws_zombie_connections(port: int = 7497) -> tuple[int, str]:
    """
    Check for CLOSE_WAIT zombie connections on TWS port.

    Returns:
        tuple: (zombie_count, error_message)
               zombie_count = 0 means clean state
               zombie_count > 0 means zombies detected, error_message explains
    """
    try:
        result = subprocess.run(["netstat", "-an"], capture_output=True, text=True, timeout=5)

        # Count CLOSE_WAIT connections on TWS port
        zombie_count = 0
        for line in result.stdout.splitlines():
            if str(port) in line and "CLOSE_WAIT" in line:
                zombie_count += 1

        if zombie_count > 0:
            error_msg = (
                f"❌ DETECTED {zombie_count} CLOSE_WAIT zombie connection(s) on port {port}.\n"
                f"Attempting automatic cleanup..."
            )
            return zombie_count, error_msg

        return 0, ""

    except Exception as e:
        logger.warning(f"Could not check for zombie connections: {e}")
        return 0, ""  # Fail open - allow connection attempt


class RobustConnectionManager:
    """
    Enhanced connection manager with exponential backoff and circuit breaker.

    Features:
    - Exponential backoff with jitter to prevent thundering herd
    - Circuit breaker to prevent connection storms
    - Automatic reconnection with health monitoring
    - Connection pooling and reuse
    """

    def __init__(
        self,
        connect_func: Callable,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize robust connection manager.

        Args:
            connect_func: Async function that establishes connection
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            jitter: Add random jitter to prevent thundering herd
            circuit_breaker_config: Circuit breaker configuration
        """
        self.connect_func = connect_func
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

        # Circuit breaker
        config = circuit_breaker_config or CircuitBreakerConfig()
        self.circuit_breaker = CircuitBreaker(config)

        # Connection state
        self.connection: Optional[Any] = None
        self.last_connect_time: Optional[datetime] = None
        self.consecutive_failures: int = 0

        # Connection health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._reconnect_lock = asyncio.Lock()

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = min(self.base_delay * (2**attempt), self.max_delay)

        if self.jitter:
            # Add random jitter (0-25% of delay)
            import random

            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay

    async def connect(self) -> Any:
        """
        Establish connection with robust retry logic.

        Returns:
            Connection object

        Raises:
            ConnectionError: If all retry attempts fail or circuit is open
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt_connection():
            status = self.circuit_breaker.get_status()
            raise ConnectionError(f"Circuit breaker is OPEN. Too many failures. Status: {status}")

        # Use existing connection if available
        if self.connection:
            try:
                # Verify connection is still alive (implement based on connection type)
                if await self._verify_connection():
                    return self.connection
            except Exception:
                logger.warning("Existing connection is dead, reconnecting...")
                await self.disconnect()

        # Prevent concurrent connection attempts
        async with self._reconnect_lock:
            # Double-check after acquiring lock
            if self.connection and await self._verify_connection():
                return self.connection

            last_exception = None

            for attempt in range(self.max_retries):
                try:
                    # Kill zombie connections before each attempt
                    if attempt > 0:  # Skip first attempt, kill before retries
                        await asyncio.to_thread(kill_zombie_connections)

                    logger.info(
                        f"Connection attempt {attempt + 1}/{self.max_retries} "
                        f"(circuit: {self.circuit_breaker.state.value})"
                    )

                    # Attempt connection
                    self.connection = await asyncio.wait_for(self.connect_func(), timeout=30.0)

                    # Verify connection is functional
                    if not await self._verify_connection():
                        raise ConnectionError("Connection verification failed")

                    # Success!
                    self.circuit_breaker.record_success()
                    self.consecutive_failures = 0
                    self.last_connect_time = datetime.now()

                    logger.info("✅ Connection established successfully")

                    # Start health monitoring
                    self._start_health_monitor()

                    return self.connection

                except asyncio.TimeoutError as e:
                    last_exception = e
                    logger.warning(f"Connection timeout on attempt {attempt + 1}")
                    self.circuit_breaker.record_failure()
                    self.consecutive_failures += 1

                except Exception as e:
                    last_exception = e
                    logger.warning(f"Connection failed on attempt {attempt + 1}: {e}")
                    self.circuit_breaker.record_failure()
                    self.consecutive_failures += 1

                # Calculate backoff delay
                if attempt < self.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)

                # Clean up failed connection
                await self._cleanup_connection()

            # All retries exhausted
            error_msg = (
                f"Failed to connect after {self.max_retries} attempts. "
                f"Last error: {last_exception}"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg)

    async def disconnect(self) -> None:
        """Gracefully disconnect and clean up."""
        self._stop_health_monitor()
        await self._cleanup_connection()
        self.connection = None

    async def _cleanup_connection(self) -> None:
        """Clean up connection resources."""
        if self.connection:
            try:
                # Implement cleanup based on connection type
                if hasattr(self.connection, "disconnect"):
                    await self.connection.disconnect()
                elif hasattr(self.connection, "close"):
                    await self.connection.close()
            except Exception as e:
                logger.warning(f"Error during connection cleanup: {e}")

    async def _verify_connection(self) -> bool:
        """
        Verify connection is alive and functional.

        Override this method for specific connection types.
        """
        if not self.connection:
            return False

        # Default implementation - check for common connection methods
        if hasattr(self.connection, "isConnected"):
            return self.connection.isConnected()
        elif hasattr(self.connection, "is_connected"):
            return self.connection.is_connected()

        # Assume connected if no verification method available
        return True

    def _start_health_monitor(self) -> None:
        """Start background health monitoring task."""
        if not self._health_check_task or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_monitor())

    def _stop_health_monitor(self) -> None:
        """Stop health monitoring task."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()

    async def _health_monitor(self) -> None:
        """Monitor connection health and reconnect if needed."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not await self._verify_connection():
                    logger.warning("Health check failed, attempting reconnection...")
                    await self.connect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    def get_status(self) -> dict:
        """Get connection manager status."""
        return {
            "connected": self.connection is not None,
            "last_connect_time": (
                self.last_connect_time.isoformat() if self.last_connect_time else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "circuit_breaker": self.circuit_breaker.get_status(),
        }


async def connect_ibkr_robust(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    max_retries: int = 2,  # Reduced from 5 to prevent zombie accumulation
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
) -> Any:
    """
    Connect to IBKR with robust retry and circuit breaker logic.

    Args:
        host: IBKR Gateway/TWS host
        port: IBKR Gateway/TWS port
        client_id: Client ID for connection
        max_retries: Maximum retry attempts
        circuit_breaker_config: Circuit breaker configuration

    Returns:
        IB connection object

    Example:
        ```python
        # Basic usage
        ib = await connect_ibkr_robust()

        # With custom circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=1
        )
        ib = await connect_ibkr_robust(circuit_breaker_config=config)
        ```
    """
    from ib_async import IB

    attempt_count = 0

    async def _connect():
        """Internal connection function."""
        nonlocal attempt_count

        # Use fixed client_id on first attempt (TWS remembers, no dialog)
        # Use unique client_id on retries (prevent TWS confusion from quick reconnects)
        if attempt_count == 0:
            use_client_id = client_id
        else:
            import random

            use_client_id = client_id + random.randint(1, 99)

        attempt_count += 1

        ib = IB()
        try:
            await ib.connectAsync(
                host=host, port=port, clientId=use_client_id, timeout=10.0, readonly=True
            )

            # Verify connection
            accounts = ib.managedAccounts()
            if not accounts:
                await asyncio.sleep(1)
                accounts = ib.managedAccounts()

            if not accounts:
                raise ConnectionError("No managed accounts - connection invalid")

            logger.info(
                f"Connected to IBKR at {host}:{port} with client_id={use_client_id}"
                + (f" (base={client_id}, retry)" if use_client_id != client_id else "")
            )
            return ib
        except Exception as e:
            # CRITICAL: ALWAYS disconnect, even if isConnected() returns False
            # When handshake times out, isConnected() is False but socket is still open on TWS side
            # This is THE fix for zombie connection accumulation
            try:
                ib.disconnect()  # Always call, regardless of connection state
                await asyncio.sleep(0.5)  # Give TWS time to process disconnect
                logger.debug(f"Disconnected failed IB connection after error: {e}")
            except Exception as cleanup_err:
                logger.debug(f"Error during disconnect (non-critical): {cleanup_err}")
            raise

    # Create robust connection manager
    manager = RobustConnectionManager(
        connect_func=_connect,
        max_retries=max_retries,
        circuit_breaker_config=circuit_breaker_config,
    )

    return await manager.connect()


# Example usage and testing
async def example_usage():
    """Example of using robust connection."""
    try:
        # Connect with circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=3,  # Open circuit after 3 failures
            recovery_timeout=30,  # Wait 30s before retry
            success_threshold=1,  # Need 1 success to close circuit
        )

        ib = await connect_ibkr_robust(circuit_breaker_config=config)

        # Use connection
        print(f"Connected: {ib.isConnected()}")
        print(f"Accounts: {ib.managedAccounts()}")

        # Disconnect when done
        ib.disconnect()

    except ConnectionError as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())
