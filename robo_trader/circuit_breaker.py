"""
Circuit Breaker System for Fault Tolerance

Implements the circuit breaker pattern to prevent cascading failures
and allow automatic recovery after errors are resolved.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from robo_trader.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    times_opened: int = 0
    times_recovered: int = 0
    current_state: CircuitState = CircuitState.CLOSED
    state_changed_at: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        recovery_timeout: int = None,
        half_open_requests: int = 1,
        failure_rate_threshold: float = 0.5,
        min_requests: int = 10,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
    ):
        self.name = name

        # Load from environment or use defaults
        self.failure_threshold = failure_threshold or int(
            os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")
        )
        self.recovery_timeout = recovery_timeout or int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "300"))
        self.half_open_requests = half_open_requests
        self.failure_rate_threshold = failure_rate_threshold
        self.min_requests = min_requests

        # Callbacks
        self.on_open = on_open
        self.on_close = on_close

        # State
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.half_open_count = 0
        self.state_lock = asyncio.Lock()

        # Failure tracking window (sliding window for failure rate)
        self.recent_calls: List[Tuple[datetime, bool]] = []  # (timestamp, success)
        self.window_duration = 60  # seconds

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self.state == CircuitState.HALF_OPEN

    async def can_proceed(self) -> bool:
        """Check if a request can proceed through the circuit."""
        async with self.state_lock:
            # Update state if needed
            await self._check_state_transition()

            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if await self._should_attempt_recovery():
                    await self._transition_to_half_open()
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self.half_open_count < self.half_open_requests:
                    self.half_open_count += 1
                    return True
                return False

        return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self.state_lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self.stats.last_success_time = datetime.now()

            # Track in sliding window
            self._add_to_window(True)

            # Handle state transitions
            if self.state == CircuitState.HALF_OPEN:
                # Successful call in half-open state
                if self.stats.consecutive_successes >= self.half_open_requests:
                    await self._transition_to_closed()

            logger.debug(f"Circuit breaker '{self.name}': Success recorded (state: {self.state})")

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call."""
        async with self.state_lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = datetime.now()

            # Track in sliding window
            self._add_to_window(False)

            logger.warning(
                f"Circuit breaker '{self.name}': Failure recorded "
                f"({self.stats.consecutive_failures}/{self.failure_threshold})"
                f"{f' - Error: {error}' if error else ''}"
            )

            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                if await self._should_open_circuit():
                    await self._transition_to_open()

            elif self.state == CircuitState.HALF_OPEN:
                # Failure in half-open state immediately opens circuit
                await self._transition_to_open()

    async def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on failures."""
        # Check consecutive failures
        if self.stats.consecutive_failures >= self.failure_threshold:
            return True

        # Check failure rate
        failure_rate = self._calculate_failure_rate()
        if failure_rate is not None and failure_rate > self.failure_rate_threshold:
            recent_count = len(self.recent_calls)
            if recent_count >= self.min_requests:
                logger.info(
                    f"Circuit breaker '{self.name}': "
                    f"Failure rate {failure_rate:.1%} exceeds threshold {self.failure_rate_threshold:.1%}"
                )
                return True

        return False

    async def _should_attempt_recovery(self) -> bool:
        """Check if we should attempt recovery (transition to half-open)."""
        if self.state != CircuitState.OPEN:
            return False

        if not self.stats.last_failure_time:
            return True

        time_since_failure = (datetime.now() - self.stats.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout

    async def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        previous_state = self.state
        self.state = CircuitState.OPEN
        self.stats.current_state = CircuitState.OPEN
        self.stats.state_changed_at = datetime.now()
        self.stats.times_opened += 1
        self.half_open_count = 0

        logger.critical(
            f"Circuit breaker '{self.name}' OPENED "
            f"(failures: {self.stats.consecutive_failures}, "
            f"rate: {self._calculate_failure_rate():.1%} if self._calculate_failure_rate() else 'N/A')"
        )

        # Execute callback
        if self.on_open and previous_state != CircuitState.OPEN:
            try:
                if asyncio.iscoroutinefunction(self.on_open):
                    await self.on_open()
                else:
                    self.on_open()
            except Exception as e:
                logger.error(f"Error in on_open callback: {e}")

    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        previous_state = self.state
        self.state = CircuitState.CLOSED
        self.stats.current_state = CircuitState.CLOSED
        self.stats.state_changed_at = datetime.now()
        self.stats.consecutive_failures = 0
        self.half_open_count = 0

        if previous_state == CircuitState.OPEN:
            self.stats.times_recovered += 1

        logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")

        # Execute callback
        if self.on_close and previous_state != CircuitState.CLOSED:
            try:
                if asyncio.iscoroutinefunction(self.on_close):
                    await self.on_close()
                else:
                    self.on_close()
            except Exception as e:
                logger.error(f"Error in on_close callback: {e}")

    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.stats.current_state = CircuitState.HALF_OPEN
        self.stats.state_changed_at = datetime.now()
        self.half_open_count = 0
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = 0

        logger.info(f"Circuit breaker '{self.name}' HALF-OPEN (testing recovery)")

    async def _check_state_transition(self) -> None:
        """Check if state should transition based on current conditions."""
        if self.state == CircuitState.OPEN:
            if await self._should_attempt_recovery():
                await self._transition_to_half_open()

    def _add_to_window(self, success: bool) -> None:
        """Add a call result to the sliding window."""
        now = datetime.now()
        self.recent_calls.append((now, success))

        # Remove old entries outside window
        cutoff = now - timedelta(seconds=self.window_duration)
        self.recent_calls = [(ts, result) for ts, result in self.recent_calls if ts > cutoff]

    def _calculate_failure_rate(self) -> Optional[float]:
        """Calculate current failure rate within the sliding window."""
        if not self.recent_calls:
            return None

        failures = sum(1 for _, success in self.recent_calls if not success)
        total = len(self.recent_calls)

        if total == 0:
            return None

        return failures / total

    async def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        async with self.state_lock:
            self.state = CircuitState.CLOSED
            self.stats = CircuitStats()
            self.half_open_count = 0
            self.recent_calls.clear()
            logger.info(f"Circuit breaker '{self.name}' reset")

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        stats_dict = {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "consecutive_failures": self.stats.consecutive_failures,
            "consecutive_successes": self.stats.consecutive_successes,
            "times_opened": self.stats.times_opened,
            "times_recovered": self.stats.times_recovered,
            "failure_rate": self._calculate_failure_rate(),
            "state_duration": (datetime.now() - self.stats.state_changed_at).total_seconds(),
        }

        if self.stats.last_failure_time:
            stats_dict["last_failure_ago"] = (
                datetime.now() - self.stats.last_failure_time
            ).total_seconds()

        if self.stats.last_success_time:
            stats_dict["last_success_ago"] = (
                datetime.now() - self.stats.last_success_time
            ).total_seconds()

        return stats_dict


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services."""

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.global_emergency = False

    def create_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        if name in self.breakers:
            logger.warning(f"Circuit breaker '{name}' already exists")
            return self.breakers[name]

        breaker = CircuitBreaker(name, **kwargs)
        self.breakers[name] = breaker
        logger.info(f"Created circuit breaker: {name}")
        return breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.breakers.get(name)

    async def emergency_open_all(self) -> None:
        """Open all circuit breakers (emergency shutdown)."""
        self.global_emergency = True
        logger.critical("EMERGENCY: Opening all circuit breakers")

        for breaker in self.breakers.values():
            async with breaker.state_lock:
                await breaker._transition_to_open()

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        self.global_emergency = False
        logger.info("Resetting all circuit breakers")

        for breaker in self.breakers.values():
            await breaker.reset()

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_statistics() for name, breaker in self.breakers.items()}

    def get_open_breakers(self) -> List[str]:
        """Get names of all open circuit breakers."""
        return [name for name, breaker in self.breakers.items() if breaker.is_open]

    def is_any_open(self) -> bool:
        """Check if any circuit breaker is open."""
        return any(breaker.is_open for breaker in self.breakers.values())


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    breaker = circuit_manager.get_breaker(name)
    if not breaker:
        breaker = circuit_manager.create_breaker(name, **kwargs)
    return breaker
