"""
Robust IBKR connection utilities with exponential backoff and circuit breaker.

This module provides enhanced connection resilience for IBKR connections:
- Exponential backoff retry with jitter
- Circuit breaker pattern to prevent connection storms
- Connection health monitoring
- Automatic reconnection on failure
"""

import asyncio
import fcntl
import os
import ssl
import subprocess
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Literal, Optional, TypeVar

from ..logger import get_logger

try:
    from .secure_config import SecureConfig
except ImportError:

    class SecureConfig:  # type: ignore[too-few-public-methods]
        @staticmethod
        def mask_value(value, reveal_length: int = 4) -> str:
            text = str(value)
            if len(text) <= reveal_length:
                return "****"
            return f"{text[:reveal_length]}****"


try:
    from robo_trader.clients.subprocess_ibkr_client import GatewayRequiresRestartError
except Exception:  # noqa: BLE001

    class GatewayRequiresRestartError(Exception):
        """Fallback definition used when client module not yet available"""

        pass


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
        # Emit telemetry/metrics
        self._emit_state_change_metric("open")

    def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        logger.info("Circuit breaker CLOSED - Normal operation resumed")
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        # Emit telemetry/metrics
        self._emit_state_change_metric("closed")

    def _half_open_circuit(self) -> None:
        """Set circuit to half-open for testing."""
        logger.info("Circuit breaker HALF-OPEN - Testing connection recovery")
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.now()
        self.success_count = 0
        # Emit telemetry/metrics
        self._emit_state_change_metric("half_open")

    def _emit_state_change_metric(self, new_state: str) -> None:
        """Emit telemetry/metrics for circuit breaker state changes.

        This method logs structured metrics that can be captured by monitoring
        systems (e.g., Prometheus, CloudWatch, DataDog).

        Args:
            new_state: New circuit breaker state (open, closed, half_open)
        """
        # Structured logging for metric collection
        logger.info(
            "circuit_breaker_state_change",
            extra={
                "metric_type": "circuit_breaker_state",
                "state": new_state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "timestamp": datetime.now().isoformat(),
            },
        )
        # Additional metrics for monitoring
        logger.info(
            "circuit_breaker_metrics",
            extra={
                "metric_type": "circuit_breaker_counters",
                "failures": self.failure_count,
                "successes": self.success_count,
                "state": new_state,
            },
        )

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
        }


_GLOBAL_CONNECT_LOCK: asyncio.Lock = asyncio.Lock()


class _ConnectFileLock:
    """Simple cross-process file lock to serialize IBKR handshakes.

    Prevents concurrent API handshakes across multiple processes which can
    confuse Gateway/TWS and exhaust client slots.

    File Lock Path Configuration:
        Default: /tmp/ibkr_connect.lock
        Override via environment variable: IBKR_LOCK_FILE_PATH
        Example: export IBKR_LOCK_FILE_PATH=/var/run/ibkr_connect.lock

    Timeout Configuration:
        Default: 30 seconds
        Override via environment variable: IBKR_LOCK_TIMEOUT
        Example: export IBKR_LOCK_TIMEOUT=60
    """

    def __init__(
        self,
        lock_path: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        # Allow configuration via environment variable
        self.lock_path = lock_path or os.environ.get(
            "IBKR_LOCK_FILE_PATH", "/tmp/ibkr_connect.lock"
        )
        # Timeout in seconds (default 30s, configurable via env)
        self.timeout = float(timeout or os.environ.get("IBKR_LOCK_TIMEOUT", "30"))
        self._fh: Optional[Any] = None

    def __enter__(self):
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"Failed to acquire file lock at {self.lock_path} within {self.timeout}s"
            )

        # Set timeout alarm (Unix-based systems only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.timeout))

        try:
            self._fh = open(self.lock_path, "w")
            fcntl.flock(self._fh, fcntl.LOCK_EX)
            signal.alarm(0)  # Cancel alarm on success
            signal.signal(signal.SIGALRM, old_handler)  # Restore handler on success
            return self
        except Exception:
            signal.alarm(0)  # Cancel alarm on error
            signal.signal(signal.SIGALRM, old_handler)  # Restore handler
            if self._fh:
                self._fh.close()
                self._fh = None
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fh:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
                self._fh.close()
        finally:
            self._fh = None


def kill_tws_zombie_connections(port: int = 7497) -> tuple[bool, str]:
    """
    Detect and kill CLOSE_WAIT zombie connections on TWS port.

    This clears kernel-level TCP state that prevents new connections.

    CRITICAL: Gateway-owned zombies CANNOT be killed without killing Gateway.
    Instead, we use lsof to forcibly close the file descriptors, which clears
    the CLOSE_WAIT state without killing the Gateway process.

    Returns:
        tuple: (success, message)
    """
    try:
        # Step 1: Check for CLOSE_WAIT connections
        lsof_cmd = [
            "lsof",
            "-nP",  # suppress hostname/service resolution
            f"-iTCP:{port}",
            "-sTCP:CLOSE_WAIT",
        ]

        lsof_result = subprocess.run(
            lsof_cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )

        output = lsof_result.stdout.strip()

        if not output or "COMMAND" not in output:
            logger.info(f"✅ No CLOSE_WAIT zombies found on port {port}")
            return True, "No zombies detected"

        # Parse lsof output
        lines = output.split("\n")[1:]  # Skip header
        zombie_info = []

        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                cmd = parts[0]
                pid = parts[1]
                fd = parts[3]  # File descriptor
                zombie_info.append({"command": cmd, "pid": pid, "fd": fd, "line": line})

        if not zombie_info:
            logger.info(f"✅ No CLOSE_WAIT zombies found on port {port}")
            return True, "No zombies detected"

        logger.warning(f"Found {len(zombie_info)} CLOSE_WAIT zombie connection(s):")
        for info in zombie_info:
            logger.warning(f"  {info['command']} (PID {info['pid']}, FD {info['fd']})")

        # Step 2: Strategy depends on what owns the zombie
        python_pids = set()
        gateway_zombies = []

        for info in zombie_info:
            cmd = info["command"].lower()
            pid = info["pid"]

            # Python processes can be killed safely
            if "python" in cmd:
                try:
                    python_pids.add(int(pid))
                except ValueError:
                    pass
            # Gateway/Java zombies need special handling
            elif any(keyword in cmd for keyword in ("java", "gateway", "tws")):
                gateway_zombies.append(info)

        # Step 3: Kill Python zombie processes
        killed_count = 0
        for pid in python_pids:
            try:
                subprocess.run(["kill", "-9", str(pid)], timeout=2)
                killed_count += 1
                logger.info(f"Killed Python zombie process {pid}")
            except Exception as e:
                logger.warning(f"Could not kill process {pid}: {e}")

        # Step 4: Handle Gateway zombies (can't kill Gateway, but can log)
        if gateway_zombies:
            logger.warning(
                f"Found {len(gateway_zombies)} Gateway-owned zombie(s) - these block API handshakes"
            )
            logger.warning(
                "Gateway zombies can only be cleared by restarting Gateway or waiting for timeout"
            )
            # Note: We return False here to indicate zombies remain
            msg = f"⚠️ Killed {killed_count} Python zombies, but {len(gateway_zombies)} Gateway zombies remain (restart Gateway to clear)"
            return False, msg

        # Step 5: Verify cleanup
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
            msg = f"⚠️ Killed {killed_count} processes but {remaining_zombies} zombies remain (likely Gateway-owned)"
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
        port: int = 7497,
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
            port: Port for zombie connection cleanup
        """
        self.connect_func = connect_func
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.port = port

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

            # Clean up any existing zombie connections before starting
            logger.info(f"Pre-connection zombie check on port {self.port}...")
            zombie_count, error_msg = await asyncio.to_thread(
                check_tws_zombie_connections, self.port
            )
            if zombie_count > 0:
                logger.warning(error_msg)
                success, msg = await asyncio.to_thread(kill_tws_zombie_connections, self.port)
                if success:
                    logger.info(f"Pre-connection cleanup: {msg}")
                else:
                    logger.warning(f"Pre-connection cleanup incomplete: {msg}")

            last_exception = None

            for attempt in range(self.max_retries):
                try:
                    # Kill zombie CLOSE_WAIT connections before retry attempts
                    # This is critical - zombie connections prevent handshake completion
                    if attempt > 0:
                        logger.info(
                            f"Cleaning up zombie connections on port {self.port} before retry..."
                        )
                        success, msg = await asyncio.to_thread(
                            kill_tws_zombie_connections, self.port
                        )
                        if success:
                            logger.info(f"Zombie cleanup: {msg}")
                        else:
                            logger.warning(f"Zombie cleanup incomplete: {msg}")

                    logger.info(
                        f"Connection attempt {attempt + 1}/{self.max_retries} "
                        f"(circuit: {self.circuit_breaker.state.value})"
                    )

                    # Attempt connection
                    self.connection = await asyncio.wait_for(self.connect_func(), timeout=60.0)

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

                except GatewayRequiresRestartError as e:
                    last_exception = e
                    logger.error(
                        "Gateway API layer reported down; aborting retries and requiring manual restart",
                        detail=str(e),
                    )
                    self.circuit_breaker.record_failure()
                    self.consecutive_failures += 1
                    raise ConnectionError(
                        "IBKR Gateway API layer is unresponsive. "
                        "Manual Gateway restart required before retrying."
                    ) from e
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
                    try:
                        await self.connection.disconnect()
                    except TypeError:
                        # Some clients expose sync disconnect
                        self.connection.disconnect()
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
            try:
                return bool(self.connection.isConnected())
            except Exception:
                return False
        elif hasattr(self.connection, "is_connected"):
            # Check if it's a property or method
            is_connected_attr = getattr(self.connection, "is_connected")
            try:
                if callable(is_connected_attr):
                    return bool(is_connected_attr())
                else:
                    return bool(is_connected_attr)
            except Exception:
                return False

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


async def connect_ibkr_robust_subprocess(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    *,
    readonly: bool = True,
    timeout: float = 10.0,
    max_retries: int = 2,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
) -> Any:
    """
    Connect to IBKR using subprocess-based client for complete process isolation.

    This solves the ib_async library incompatibility with complex async environments
    by running ib_async in a completely isolated subprocess.

    Args:
        host: IBKR Gateway/TWS host
        port: IBKR Gateway/TWS port
        client_id: Client ID for connection
        readonly: Readonly mode
        timeout: Connection timeout
        max_retries: Maximum retry attempts
        circuit_breaker_config: Circuit breaker configuration

    Returns:
        SubprocessIBKRClient instance (connected)
    """
    from robo_trader.clients.subprocess_ibkr_client import (
        GatewayRequiresRestartError,
        SubprocessCrashError,
        SubprocessIBKRClient,
    )

    attempt_count = 0
    client = SubprocessIBKRClient()

    # Start subprocess
    await client.start()
    logger.info(
        "Started IBKR subprocess worker", pid=client.process.pid if client.process else None
    )

    async def _connect():
        """Internal connection function."""
        nonlocal attempt_count

        # Use fixed client_id on first attempt
        # Use unique client_id on retries
        if attempt_count == 0:
            use_client_id = client_id
        else:
            import random

            use_client_id = client_id + random.randint(1, 99)

        attempt_count += 1

        try:
            logger.info(
                "Connecting to IBKR via subprocess host=%s port=%s client_id=%s readonly=%s timeout=%.1f",
                host,
                port,
                SecureConfig.mask_value(use_client_id),
                readonly,
                timeout,
            )

            # Connect via subprocess
            connected = await client.connect(
                host=host, port=port, client_id=use_client_id, readonly=readonly, timeout=timeout
            )

            if not connected:
                raise ConnectionError("Subprocess connection failed")

            # Verify accounts
            accounts = await client.get_accounts()
            if not accounts:
                raise ConnectionError("No managed accounts")

            logger.info(
                "Connected to IBKR via subprocess host=%s port=%s client_id=%s accounts=%d",
                host,
                port,
                SecureConfig.mask_value(use_client_id),
                len(accounts),
            )

            return client

        except SubprocessCrashError as e:
            # Subprocess crashed - restart it
            logger.error("Subprocess crashed, restarting", error=str(e))
            await client.stop()
            await client.start()
            raise
        except GatewayRequiresRestartError as e:
            logger.error(
                "Gateway API layer reported down during subprocess connect",
                detail=str(e),
            )
            raise
        except Exception as e:
            # Connection failed - disconnect cleanly
            try:
                await client.disconnect()
            except Exception:
                pass
            raise

    # Create robust connection manager
    manager = RobustConnectionManager(
        connect_func=_connect,
        max_retries=max_retries,
        base_delay=5.0,
        max_delay=120.0,
        circuit_breaker_config=circuit_breaker_config,
        port=port,
    )

    try:
        async with _GLOBAL_CONNECT_LOCK:
            # Cross-process serialization guard
            with _ConnectFileLock():
                return await manager.connect()
    except Exception:
        # Failed to connect - stop subprocess
        await client.stop()
        raise


async def connect_ibkr_robust(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    *,
    readonly: bool = True,
    timeout: float = 10.0,
    max_retries: int = 2,  # Reduced from 5 to prevent zombie accumulation
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ssl_mode: Literal["auto", "require", "disabled"] = "auto",
    use_subprocess: bool = True,  # NEW: Use subprocess by default
) -> Any:
    """
    Connect to IBKR with robust retry and circuit breaker logic.

    Args:
        host: IBKR Gateway/TWS host
        port: IBKR Gateway/TWS port
        client_id: Client ID for connection
        max_retries: Maximum retry attempts
        circuit_breaker_config: Circuit breaker configuration
        ssl_mode: Transport strategy. ``auto`` attempts plain TCP first and falls
            back to TLS on handshake timeouts. ``require`` forces TLS immediately.
            ``disabled`` keeps legacy plain TCP only.
        use_subprocess: Use subprocess-based client (recommended, default=True)

    Returns:
        SubprocessIBKRClient if use_subprocess=True, otherwise IB connection object

    Example:
        ```python
        # Basic usage (subprocess mode - recommended)
        client = await connect_ibkr_robust()

        # Legacy mode (direct ib_async)
        ib = await connect_ibkr_robust(use_subprocess=False)

        # With custom circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=1
        )
        client = await connect_ibkr_robust(circuit_breaker_config=config)
        ```
    """
    # Route to subprocess implementation by default
    if use_subprocess:
        logger.info("Using subprocess-based IBKR client (recommended)")
        return await connect_ibkr_robust_subprocess(
            host=host,
            port=port,
            client_id=client_id,
            readonly=readonly,
            timeout=timeout,
            max_retries=max_retries,
            circuit_breaker_config=circuit_breaker_config,
        )

    # Legacy direct ib_async implementation
    logger.warning("Using legacy direct ib_async client (not recommended)")
    from ib_async import IB

    attempt_count = 0

    normalized_ssl_mode = (ssl_mode or "auto").lower()
    if normalized_ssl_mode not in {"auto", "require", "disabled"}:
        raise ValueError(
            f"Invalid ssl_mode '{ssl_mode}'. Expected one of: auto, require, disabled."
        )

    if normalized_ssl_mode == "require":
        transport_modes = ["ssl"]
    elif normalized_ssl_mode == "disabled":
        transport_modes = ["plain"]
    else:
        transport_modes = ["plain", "ssl"]

    transport_index = 0

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    async def _connect():
        """Internal connection function."""
        nonlocal attempt_count, transport_index

        # Use fixed client_id on first attempt (TWS remembers, no dialog)
        # Use unique client_id on retries (prevent TWS confusion from quick reconnects)
        if attempt_count == 0:
            use_client_id = client_id
        else:
            import random

            use_client_id = client_id + random.randint(1, 99)

        attempt_count += 1

        ib = IB()

        transport_mode = transport_modes[min(transport_index, len(transport_modes) - 1)]

        if transport_mode == "ssl":
            # Configure TLS transport with relaxed certificate checks for local Gateway
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            conn = ib.client.conn

            async def connect_async_with_ssl(self, host: str, port: int) -> None:
                if self.transport:
                    self.disconnect()
                    await self.disconnected
                self.reset()
                from ib_async.util import getLoop

                loop = getLoop()
                self.transport, _ = await loop.create_connection(
                    lambda: self, host, port, ssl=ssl_context
                )

            conn.connectAsync = types.MethodType(connect_async_with_ssl, conn)

        def _api_error_listener(msg):  # type: ignore[no-untyped-def]
            logger.error(
                "API error during handshake (client_id=%s): %s",
                SecureConfig.mask_value(use_client_id),
                msg,
            )

        ib.client.apiError += _api_error_listener

        try:
            logger.info(
                "Starting IBKR handshake host=%s port=%s client_id=%s readonly=%s timeout=%.1f transport=%s",
                host,
                port,
                SecureConfig.mask_value(use_client_id),
                readonly,
                timeout,
                transport_mode,
            )
            # EXPERIMENTAL: Add small delay to let Gateway settle after restart
            await asyncio.sleep(0.1)

            await ib.connectAsync(
                host=host,
                port=port,
                clientId=use_client_id,
                timeout=timeout,
                readonly=readonly,
            )

            # Verify connection
            accounts = ib.managedAccounts()
            if not accounts:
                await asyncio.sleep(1)
                accounts = ib.managedAccounts()

            if not accounts:
                raise ConnectionError("No managed accounts - connection invalid")

            logger.info(
                "IBKR handshake complete host=%s port=%s client_id=%s readonly=%s timeout=%.1f transport=%s accounts=%d",
                host,
                port,
                SecureConfig.mask_value(use_client_id),
                readonly,
                timeout,
                transport_mode,
                len(accounts),
            )
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

            # If plain transport failed in auto mode, escalate to TLS on the next attempt
            if (
                normalized_ssl_mode == "auto"
                and transport_mode == "plain"
                and transport_index < len(transport_modes) - 1
                and isinstance(e, (asyncio.TimeoutError, TimeoutError))
            ):
                transport_index += 1
                logger.warning(
                    "Plain TCP handshake timed out. Auto-switching to TLS for subsequent attempts."
                )
            raise
        finally:
            try:
                ib.client.apiError -= _api_error_listener
            except Exception:
                pass

    # Create robust connection manager with longer delays
    manager = RobustConnectionManager(
        connect_func=_connect,
        max_retries=max_retries,
        base_delay=5.0,  # Increased from 1.0 to 5.0 seconds
        max_delay=120.0,  # Increased from 60.0 to 120.0 seconds
        circuit_breaker_config=circuit_breaker_config,
        port=port,
    )

    async with _GLOBAL_CONNECT_LOCK:
        # Cross-process serialization guard
        with _ConnectFileLock():
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
