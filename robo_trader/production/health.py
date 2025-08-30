"""
Health monitoring and system status checks.

Provides health check endpoints, system monitoring,
and circuit breaker functionality.
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

from ..database_async import AsyncTradingDatabase as TradingDatabase
from ..logger import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentStatus(Enum):
    """Individual component status."""

    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """System health metrics."""

    timestamp: datetime = field(default_factory=datetime.now)

    # System metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    open_file_descriptors: int = 0
    thread_count: int = 0

    # Trading metrics
    active_positions: int = 0
    pending_orders: int = 0
    daily_trades: int = 0
    daily_pnl: float = 0.0

    # Performance metrics
    order_latency_ms: float = 0.0
    data_latency_ms: float = 0.0
    api_request_rate: float = 0.0
    error_rate: float = 0.0

    # Connection status
    ibkr_connected: bool = False
    database_connected: bool = False
    market_data_streaming: bool = False


@dataclass
class ComponentHealth:
    """Health status of a system component."""

    name: str
    status: ComponentStatus
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "metrics": self.metrics,
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for fault tolerance."""

    is_open: bool = False
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0

    # Thresholds
    failure_threshold: int = 5
    timeout_seconds: int = 60

    def record_success(self) -> None:
        """Record successful operation."""
        self.last_success = datetime.now()
        self.consecutive_failures = 0
        self.is_open = False

    def record_failure(self) -> None:
        """Record failed operation."""
        self.last_failure = datetime.now()
        self.failure_count += 1
        self.consecutive_failures += 1

        if self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened after {self.consecutive_failures} failures")

    def should_attempt(self) -> bool:
        """Check if operation should be attempted."""
        if not self.is_open:
            return True

        # Check if timeout has passed
        if self.last_failure:
            elapsed = (datetime.now() - self.last_failure).total_seconds()
            if elapsed > self.timeout_seconds:
                # Try half-open state
                logger.info("Circuit breaker attempting half-open state")
                return True

        return False


class HealthMonitor:
    """
    System health monitoring and status reporting.

    Features:
    - Component health checks
    - System metrics collection
    - Circuit breaker patterns
    - Health endpoints for monitoring
    """

    def __init__(self, check_interval: int = 30):
        """
        Initialize health monitor.

        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.metrics_history: deque = deque(maxlen=100)
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.health_checks: Dict[str, Callable] = {}
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Register default health checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health check functions."""
        self.register_health_check("system", self._check_system_health)
        self.register_health_check("database", self._check_database_health)
        self.register_health_check("trading", self._check_trading_health)

    def register_health_check(self, name: str, check_func: Callable) -> None:
        """
        Register a health check function.

        Args:
            name: Component name
            check_func: Function that returns ComponentHealth
        """
        self.health_checks[name] = check_func
        self.circuit_breakers[name] = CircuitBreakerState()
        logger.info(f"Registered health check: {name}")

    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._running:
            logger.warning("Health monitoring already running")
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Started health monitoring")

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped health monitoring")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Run all health checks
                self.run_health_checks()

                # Collect metrics
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(5)  # Brief pause on error

    def run_health_checks(self) -> Dict[str, ComponentHealth]:
        """Run all registered health checks."""
        results = {}

        for name, check_func in self.health_checks.items():
            circuit_breaker = self.circuit_breakers.get(name)

            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.should_attempt():
                results[name] = ComponentHealth(
                    name=name,
                    status=ComponentStatus.DOWN,
                    message="Circuit breaker open",
                )
                continue

            try:
                # Run health check
                health = check_func()
                results[name] = health
                self.components[name] = health

                # Update circuit breaker
                if circuit_breaker:
                    if health.status in [ComponentStatus.UP, ComponentStatus.DEGRADED]:
                        circuit_breaker.record_success()
                    else:
                        circuit_breaker.record_failure()

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")

                results[name] = ComponentHealth(
                    name=name, status=ComponentStatus.UNKNOWN, message=str(e)
                )

                if circuit_breaker:
                    circuit_breaker.record_failure()

        return results

    def collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        metrics = HealthMetrics()

        try:
            # System metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=1)
            metrics.memory_percent = psutil.virtual_memory().percent
            metrics.disk_percent = psutil.disk_usage("/").percent

            process = psutil.Process()
            metrics.open_file_descriptors = len(process.open_files())
            metrics.thread_count = process.num_threads()

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

        return metrics

    def _check_system_health(self) -> ComponentHealth:
        """Check system resource health."""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent

            # Determine status
            if cpu > 90 or memory > 90 or disk > 95:
                status = ComponentStatus.DEGRADED
                message = f"High resource usage - CPU: {cpu}%, Memory: {memory}%, Disk: {disk}%"
            else:
                status = ComponentStatus.UP
                message = "System resources normal"

            return ComponentHealth(
                name="system",
                status=status,
                message=message,
                metrics={
                    "cpu_percent": cpu,
                    "memory_percent": memory,
                    "disk_percent": disk,
                },
            )

        except Exception as e:
            return ComponentHealth(name="system", status=ComponentStatus.UNKNOWN, message=str(e))

    def _check_database_health(self) -> ComponentHealth:
        """Check database connectivity and performance."""
        try:
            # Try to execute a simple query
            start_time = time.time()

            # This would use the actual database connection
            # For now, simulate
            db_connected = True
            query_time = (time.time() - start_time) * 1000

            if db_connected:
                status = ComponentStatus.UP
                message = f"Database responding (latency: {query_time:.1f}ms)"
            else:
                status = ComponentStatus.DOWN
                message = "Database connection failed"

            return ComponentHealth(
                name="database",
                status=status,
                message=message,
                metrics={"query_latency_ms": query_time},
            )

        except Exception as e:
            return ComponentHealth(name="database", status=ComponentStatus.DOWN, message=str(e))

    def _check_trading_health(self) -> ComponentHealth:
        """Check trading system health."""
        try:
            # Check IBKR connection
            # This would use actual IBKR client status
            ibkr_connected = False  # Placeholder

            if ibkr_connected:
                status = ComponentStatus.UP
                message = "Trading system operational"
            else:
                status = ComponentStatus.DEGRADED
                message = "IBKR disconnected"

            return ComponentHealth(
                name="trading",
                status=status,
                message=message,
                metrics={"ibkr_connected": ibkr_connected, "active_positions": 0},
            )

        except Exception as e:
            return ComponentHealth(name="trading", status=ComponentStatus.UNKNOWN, message=str(e))

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.components:
            return HealthStatus.UNKNOWN

        # Count component statuses
        status_counts = {
            ComponentStatus.UP: 0,
            ComponentStatus.DEGRADED: 0,
            ComponentStatus.DOWN: 0,
            ComponentStatus.UNKNOWN: 0,
        }

        for component in self.components.values():
            status_counts[component.status] += 1

        # Determine overall health
        total = len(self.components)

        if status_counts[ComponentStatus.DOWN] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[ComponentStatus.DEGRADED] > total / 2:
            return HealthStatus.DEGRADED
        elif status_counts[ComponentStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        overall = self.get_overall_health()

        # Get latest metrics
        latest_metrics = None
        if self.metrics_history:
            latest_metrics = asdict(self.metrics_history[-1])

        return {
            "status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "metrics": latest_metrics,
            "circuit_breakers": {
                name: {
                    "is_open": cb.is_open,
                    "failure_count": cb.failure_count,
                    "consecutive_failures": cb.consecutive_failures,
                }
                for name, cb in self.circuit_breakers.items()
            },
        }

    def get_readiness_status(self) -> bool:
        """Check if system is ready to accept traffic."""
        # System is ready if health is not critical
        overall = self.get_overall_health()
        return overall != HealthStatus.CRITICAL

    def get_liveness_status(self) -> bool:
        """Check if system is alive (for k8s liveness probe)."""
        # System is alive if monitoring is running
        return self._running


class HealthEndpoint:
    """
    Health check HTTP endpoint for monitoring systems.

    Provides standard health check endpoints:
    - /health - Overall health status
    - /health/live - Liveness probe
    - /health/ready - Readiness probe
    - /metrics - Prometheus metrics
    """

    def __init__(self, monitor: HealthMonitor):
        """
        Initialize health endpoint.

        Args:
            monitor: Health monitor instance
        """
        self.monitor = monitor

    async def health_check(self) -> Dict[str, Any]:
        """Main health check endpoint."""
        report = self.monitor.get_health_report()

        # Add HTTP status code hint
        overall = self.monitor.get_overall_health()
        if overall == HealthStatus.HEALTHY:
            report["http_status"] = 200
        elif overall == HealthStatus.DEGRADED:
            report["http_status"] = 200  # Still operational
        else:
            report["http_status"] = 503  # Service unavailable

        return report

    async def liveness_check(self) -> Dict[str, Any]:
        """Kubernetes liveness probe endpoint."""
        is_alive = self.monitor.get_liveness_status()

        return {
            "status": "ok" if is_alive else "dead",
            "timestamp": datetime.now().isoformat(),
            "http_status": 200 if is_alive else 503,
        }

    async def readiness_check(self) -> Dict[str, Any]:
        """Kubernetes readiness probe endpoint."""
        is_ready = self.monitor.get_readiness_status()

        return {
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.now().isoformat(),
            "http_status": 200 if is_ready else 503,
        }

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        lines = []

        # Get latest metrics
        if self.monitor.metrics_history:
            metrics = self.monitor.metrics_history[-1]

            # System metrics
            lines.append(f"# HELP system_cpu_percent CPU usage percentage")
            lines.append(f"# TYPE system_cpu_percent gauge")
            lines.append(f"system_cpu_percent {metrics.cpu_percent}")

            lines.append(f"# HELP system_memory_percent Memory usage percentage")
            lines.append(f"# TYPE system_memory_percent gauge")
            lines.append(f"system_memory_percent {metrics.memory_percent}")

            # Trading metrics
            lines.append(f"# HELP trading_active_positions Number of active positions")
            lines.append(f"# TYPE trading_active_positions gauge")
            lines.append(f"trading_active_positions {metrics.active_positions}")

            lines.append(f"# HELP trading_daily_pnl Daily profit and loss")
            lines.append(f"# TYPE trading_daily_pnl gauge")
            lines.append(f"trading_daily_pnl {metrics.daily_pnl}")

        # Component status
        for name, component in self.monitor.components.items():
            status_value = 1 if component.status == ComponentStatus.UP else 0
            lines.append(f"# HELP component_{name}_up Component status (1=up, 0=down)")
            lines.append(f"# TYPE component_{name}_up gauge")
            lines.append(f"component_{name}_up {status_value}")

        return "\n".join(lines)
