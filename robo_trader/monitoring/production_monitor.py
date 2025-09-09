"""
Production Monitoring Stack for RoboTrader.

Real-time system monitoring with automated alerts, performance metrics,
and comprehensive dashboards for production trading operations.
"""

import asyncio
import json
import logging
import os
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Handle optional dependencies
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. System metrics will be limited.")

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    warnings.warn("aiohttp not available. Webhook alerts will be disabled.")


class MetricType(Enum):
    """Types of metrics to monitor."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    PNL = "pnl"
    POSITION_COUNT = "position_count"
    ORDER_COUNT = "order_count"
    API_CALLS = "api_calls"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point."""

    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert configuration and state."""

    name: str
    metric_type: MetricType
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    severity: AlertSeverity
    cooldown_minutes: int = 5
    message_template: str = ""
    last_triggered: Optional[datetime] = None
    triggered_count: int = 0
    active: bool = True


@dataclass
class SystemHealth:
    """System health snapshot."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    open_connections: int
    process_count: int
    thread_count: int
    is_healthy: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collects and aggregates system and trading metrics."""

    def __init__(
        self,
        retention_hours: int = 24,
        aggregation_interval: int = 60,  # seconds
    ):
        self.retention_hours = retention_hours
        self.aggregation_interval = aggregation_interval

        # Metric storage (deques for efficient rotation)
        self.metrics: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=retention_hours * 3600 // aggregation_interval)
            for metric_type in MetricType
        }

        # Aggregation buffers
        self.buffers: Dict[MetricType, List[float]] = {
            metric_type: [] for metric_type in MetricType
        }

        # Performance counters
        self.counters = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "total_api_calls": 0,
            "api_errors": 0,
            "total_trades": 0,
            "profitable_trades": 0,
        }

        # Timing metrics
        self.timings: Dict[str, deque] = {}

    def record_metric(
        self, metric_type: MetricType, value: float, tags: Optional[Dict] = None
    ) -> None:
        """Record a single metric value."""
        point = MetricPoint(metric_type, value, tags=tags or {})
        self.buffers[metric_type].append(value)

    def record_timing(self, operation: str, duration_ms: float) -> None:
        """Record operation timing."""
        if operation not in self.timings:
            self.timings[operation] = deque(maxlen=1000)
        self.timings[operation].append(duration_ms)

    def increment_counter(self, counter: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        if counter in self.counters:
            self.counters[counter] += amount

    def aggregate_metrics(self) -> Dict[MetricType, Dict[str, float]]:
        """Aggregate buffered metrics."""
        aggregated = {}

        for metric_type, values in self.buffers.items():
            if not values:
                continue

            aggregated[metric_type] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "sum": sum(values),
            }

            # Store aggregated point
            self.metrics[metric_type].append(aggregated[metric_type]["mean"])

            # Clear buffer
            self.buffers[metric_type] = []

        return aggregated

    def get_percentile(self, metric_type: MetricType, percentile: float) -> Optional[float]:
        """Get percentile value for a metric."""
        if metric_type not in self.metrics or not self.metrics[metric_type]:
            return None

        values = sorted(self.metrics[metric_type])
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]

    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        total = self.counters["successful_orders"] + self.counters["failed_orders"]
        if total == 0:
            return 1.0
        return self.counters["successful_orders"] / total

    def get_api_error_rate(self) -> float:
        """Calculate API error rate."""
        if self.counters["total_api_calls"] == 0:
            return 0.0
        return self.counters["api_errors"] / self.counters["total_api_calls"]

    def get_win_rate(self) -> float:
        """Calculate trading win rate."""
        if self.counters["total_trades"] == 0:
            return 0.0
        return self.counters["profitable_trades"] / self.counters["total_trades"]


class HealthChecker:
    """System health monitoring and diagnostics."""

    def __init__(
        self,
        warning_thresholds: Optional[Dict] = None,
        critical_thresholds: Optional[Dict] = None,
    ):
        self.warning_thresholds = warning_thresholds or {
            "cpu_percent": 70,
            "memory_percent": 80,
            "disk_usage_percent": 85,
            "open_connections": 100,
        }

        self.critical_thresholds = critical_thresholds or {
            "cpu_percent": 90,
            "memory_percent": 95,
            "disk_usage_percent": 95,
            "open_connections": 200,
        }

        self.last_health_check: Optional[SystemHealth] = None
        self.health_history: deque = deque(maxlen=1440)  # 24 hours at 1 min intervals

    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check."""
        health = SystemHealth(
            timestamp=datetime.now(),
            cpu_percent=0,
            memory_percent=0,
            disk_usage_percent=0,
            network_bytes_sent=0,
            network_bytes_recv=0,
            open_connections=0,
            process_count=0,
            thread_count=0,
            is_healthy=True,
        )

        if PSUTIL_AVAILABLE:
            try:
                # CPU and Memory
                health.cpu_percent = psutil.cpu_percent(interval=0.1)
                health.memory_percent = psutil.virtual_memory().percent

                # Disk usage
                disk = psutil.disk_usage("/")
                health.disk_usage_percent = disk.percent

                # Network
                net_io = psutil.net_io_counters()
                health.network_bytes_sent = net_io.bytes_sent
                health.network_bytes_recv = net_io.bytes_recv

                # Connections
                health.open_connections = len(psutil.net_connections())

                # Processes
                health.process_count = len(psutil.pids())

                # Current process threads
                current_process = psutil.Process()
                health.thread_count = current_process.num_threads()

            except Exception as e:
                health.errors.append(f"psutil error: {e}")

        # Check thresholds
        self._check_thresholds(health)

        # Store health check
        self.last_health_check = health
        self.health_history.append(health)

        return health

    def _check_thresholds(self, health: SystemHealth) -> None:
        """Check health against thresholds."""
        # Check critical thresholds
        for metric, threshold in self.critical_thresholds.items():
            value = getattr(health, metric, 0)
            if value > threshold:
                health.errors.append(f"{metric} critical: {value:.1f}% > {threshold}%")
                health.is_healthy = False

        # Check warning thresholds
        for metric, threshold in self.warning_thresholds.items():
            value = getattr(health, metric, 0)
            if value > threshold and metric not in [e.split()[0] for e in health.errors]:
                health.warnings.append(f"{metric} warning: {value:.1f}% > {threshold}%")

    def get_health_trend(self, hours: int = 1) -> Dict[str, List[float]]:
        """Get health metrics trend over time."""
        cutoff = datetime.now() - timedelta(hours=hours)

        trends = {
            "timestamps": [],
            "cpu_percent": [],
            "memory_percent": [],
            "disk_usage_percent": [],
        }

        for health in self.health_history:
            if health.timestamp > cutoff:
                trends["timestamps"].append(health.timestamp.isoformat())
                trends["cpu_percent"].append(health.cpu_percent)
                trends["memory_percent"].append(health.memory_percent)
                trends["disk_usage_percent"].append(health.disk_usage_percent)

        return trends


class AlertManager:
    """Manages alert rules and notifications."""

    def __init__(
        self,
        alert_config_file: Optional[Path] = None,
        webhook_url: Optional[str] = None,
        email_config: Optional[Dict] = None,
    ):
        self.alert_config_file = alert_config_file
        self.webhook_url = webhook_url
        self.email_config = email_config

        # Alert rules
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)

        # Load alert configuration
        if alert_config_file and alert_config_file.exists():
            self.load_alerts(alert_config_file)
        else:
            self._create_default_alerts()

    def _create_default_alerts(self) -> None:
        """Create default alert rules."""
        default_alerts = [
            Alert(
                name="high_cpu",
                metric_type=MetricType.CPU_USAGE,
                threshold=90,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                message_template="CPU usage is {value:.1f}%",
            ),
            Alert(
                name="high_memory",
                metric_type=MetricType.MEMORY_USAGE,
                threshold=90,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                message_template="Memory usage is {value:.1f}%",
            ),
            Alert(
                name="high_error_rate",
                metric_type=MetricType.ERROR_RATE,
                threshold=0.05,
                comparison="gt",
                severity=AlertSeverity.ERROR,
                message_template="Error rate is {value:.2%}",
            ),
            Alert(
                name="low_success_rate",
                metric_type=MetricType.SUCCESS_RATE,
                threshold=0.95,
                comparison="lt",
                severity=AlertSeverity.WARNING,
                message_template="Success rate dropped to {value:.2%}",
            ),
        ]

        for alert in default_alerts:
            self.alerts[alert.name] = alert

    def check_alerts(self, metrics: Dict[MetricType, float]) -> List[Alert]:
        """Check if any alerts should be triggered."""
        triggered = []

        for alert in self.alerts.values():
            if not alert.active:
                continue

            if alert.metric_type not in metrics:
                continue

            value = metrics[alert.metric_type]

            # Check threshold
            should_trigger = False
            if alert.comparison == "gt" and value > alert.threshold:
                should_trigger = True
            elif alert.comparison == "lt" and value < alert.threshold:
                should_trigger = True
            elif alert.comparison == "eq" and value == alert.threshold:
                should_trigger = True

            # Check cooldown
            if should_trigger:
                if alert.last_triggered:
                    elapsed = datetime.now() - alert.last_triggered
                    if elapsed < timedelta(minutes=alert.cooldown_minutes):
                        continue

                # Trigger alert
                alert.last_triggered = datetime.now()
                alert.triggered_count += 1
                triggered.append(alert)

                # Send notifications
                self._send_alert(alert, value)

        return triggered

    async def _send_alert(self, alert: Alert, value: float) -> None:
        """Send alert notifications."""
        message = alert.message_template.format(value=value)

        # Log alert
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "alert": alert.name,
            "severity": alert.severity.value,
            "message": message,
            "value": value,
        }
        self.alert_history.append(log_entry)

        # Send webhook notification
        if self.webhook_url and AIOHTTP_AVAILABLE:
            await self._send_webhook(alert, message, value)

        # Log to console
        print(f"🚨 {alert.severity.value.upper()}: {alert.name} - {message}")

    async def _send_webhook(self, alert: Alert, message: str, value: float) -> None:
        """Send webhook notification."""
        if not AIOHTTP_AVAILABLE:
            return

        payload = {
            "alert": alert.name,
            "severity": alert.severity.value,
            "message": message,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        print(f"Webhook failed: {response.status}")
        except Exception as e:
            print(f"Webhook error: {e}")

    def add_alert(self, alert: Alert) -> None:
        """Add or update an alert rule."""
        self.alerts[alert.name] = alert

    def remove_alert(self, name: str) -> None:
        """Remove an alert rule."""
        if name in self.alerts:
            del self.alerts[name]

    def save_alerts(self, filepath: Path) -> None:
        """Save alert configuration to file."""
        config = {
            name: {
                "metric_type": alert.metric_type.value,
                "threshold": alert.threshold,
                "comparison": alert.comparison,
                "severity": alert.severity.value,
                "cooldown_minutes": alert.cooldown_minutes,
                "message_template": alert.message_template,
                "active": alert.active,
            }
            for name, alert in self.alerts.items()
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    def load_alerts(self, filepath: Path) -> None:
        """Load alert configuration from file."""
        with open(filepath, "r") as f:
            config = json.load(f)

        self.alerts = {}
        for name, alert_config in config.items():
            self.alerts[name] = Alert(
                name=name,
                metric_type=MetricType(alert_config["metric_type"]),
                threshold=alert_config["threshold"],
                comparison=alert_config["comparison"],
                severity=AlertSeverity(alert_config["severity"]),
                cooldown_minutes=alert_config.get("cooldown_minutes", 5),
                message_template=alert_config.get("message_template", ""),
                active=alert_config.get("active", True),
            )


class SystemMonitor:
    """Alias for ProductionMonitor for backward compatibility."""

    pass


class ProductionMonitor:
    """Main production monitoring orchestrator."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        log_dir: Optional[Path] = None,
        enable_alerts: bool = True,
        enable_health_checks: bool = True,
    ):
        self.config = config or {}
        self.log_dir = log_dir or Path("logs")
        self.enable_alerts = enable_alerts
        self.enable_health_checks = enable_health_checks

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker() if enable_health_checks else None
        self.alert_manager = (
            AlertManager(
                webhook_url=config.get("webhook_url"), email_config=config.get("email_config")
            )
            if enable_alerts
            else None
        )

        # Monitoring state
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.start_time = datetime.now()

        # Performance tracking
        self.performance_log: deque = deque(maxlen=10000)

    async def start(self, interval: int = 60) -> None:
        """Start monitoring loop."""
        if self.is_running:
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        print(f"✅ Production monitoring started (interval: {interval}s)")

    async def stop(self) -> None:
        """Stop monitoring loop."""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            await asyncio.gather(self.monitoring_task, return_exceptions=True)
        print("⏹️ Production monitoring stopped")

    async def _monitoring_loop(self, interval: int) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()

                # Check system health
                if self.health_checker:
                    health = self.health_checker.check_system_health()
                    metrics[MetricType.CPU_USAGE] = health.cpu_percent
                    metrics[MetricType.MEMORY_USAGE] = health.memory_percent

                # Check alerts
                if self.alert_manager:
                    triggered_alerts = self.alert_manager.check_alerts(metrics)
                    if triggered_alerts:
                        await self._handle_alerts(triggered_alerts)

                # Log metrics
                await self._log_metrics(metrics)

                # Generate dashboard data
                dashboard = self.get_dashboard()
                await self._save_dashboard(dashboard)

            except Exception as e:
                print(f"Monitoring error: {e}")

            await asyncio.sleep(interval)

    async def _collect_metrics(self) -> Dict[MetricType, float]:
        """Collect current metrics."""
        # Aggregate buffered metrics
        aggregated = self.metrics_collector.aggregate_metrics()

        # Add calculated metrics
        metrics = {}
        metrics[MetricType.SUCCESS_RATE] = self.metrics_collector.get_success_rate()
        metrics[MetricType.ERROR_RATE] = self.metrics_collector.get_api_error_rate()

        # Add aggregated metrics
        for metric_type, values in aggregated.items():
            metrics[metric_type] = values["mean"]

        return metrics

    async def _handle_alerts(self, alerts: List[Alert]) -> None:
        """Handle triggered alerts."""
        for alert in alerts:
            # Log alert
            self.performance_log.append(
                {
                    "type": "alert",
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Take action based on severity
            if alert.severity == AlertSeverity.CRITICAL:
                print(f"🚨 CRITICAL ALERT: {alert.name} - Taking emergency action")
                # Could trigger kill switch or other emergency actions

    async def _log_metrics(self, metrics: Dict[MetricType, float]) -> None:
        """Log metrics to file."""
        log_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {k.value: v for k, v in metrics.items()},
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        uptime = datetime.now() - self.start_time

        dashboard = {
            "status": {
                "is_running": self.is_running,
                "uptime_hours": uptime.total_seconds() / 3600,
                "start_time": self.start_time.isoformat(),
            },
            "metrics": {
                "success_rate": self.metrics_collector.get_success_rate(),
                "error_rate": self.metrics_collector.get_api_error_rate(),
                "win_rate": self.metrics_collector.get_win_rate(),
                "total_orders": self.metrics_collector.counters["total_orders"],
                "total_trades": self.metrics_collector.counters["total_trades"],
            },
            "health": None,
            "alerts": {
                "active": len([a for a in self.alert_manager.alerts.values() if a.active])
                if self.alert_manager
                else 0,
                "triggered_today": len(
                    [
                        a
                        for a in self.alert_manager.alert_history
                        if datetime.fromisoformat(a["timestamp"]).date() == datetime.now().date()
                    ]
                )
                if self.alert_manager
                else 0,
                "recent": list(self.alert_manager.alert_history)[-10:]
                if self.alert_manager
                else [],
            },
            "performance": {
                "latency_p50": self.metrics_collector.get_percentile(MetricType.LATENCY, 50),
                "latency_p95": self.metrics_collector.get_percentile(MetricType.LATENCY, 95),
                "latency_p99": self.metrics_collector.get_percentile(MetricType.LATENCY, 99),
            },
        }

        # Add health data
        if self.health_checker and self.health_checker.last_health_check:
            health = self.health_checker.last_health_check
            dashboard["health"] = {
                "is_healthy": health.is_healthy,
                "cpu_percent": health.cpu_percent,
                "memory_percent": health.memory_percent,
                "disk_usage_percent": health.disk_usage_percent,
                "warnings": health.warnings,
                "errors": health.errors,
            }

        return dashboard

    async def _save_dashboard(self, dashboard: Dict) -> None:
        """Save dashboard snapshot to file."""
        dashboard_file = self.log_dir / "dashboard.json"

        with open(dashboard_file, "w") as f:
            json.dump(dashboard, f, indent=2, default=str)

    def record_trade(self, symbol: str, pnl: float, success: bool) -> None:
        """Record trade execution."""
        self.metrics_collector.increment_counter("total_trades")
        if pnl > 0:
            self.metrics_collector.increment_counter("profitable_trades")

        self.metrics_collector.record_metric(MetricType.PNL, pnl, {"symbol": symbol})

    def record_order(self, symbol: str, success: bool, latency_ms: float) -> None:
        """Record order execution."""
        self.metrics_collector.increment_counter("total_orders")
        if success:
            self.metrics_collector.increment_counter("successful_orders")
        else:
            self.metrics_collector.increment_counter("failed_orders")

        self.metrics_collector.record_metric(MetricType.LATENCY, latency_ms, {"symbol": symbol})
        self.metrics_collector.record_timing("order_execution", latency_ms)

    def record_api_call(self, endpoint: str, success: bool, latency_ms: float) -> None:
        """Record API call."""
        self.metrics_collector.increment_counter("total_api_calls")
        if not success:
            self.metrics_collector.increment_counter("api_errors")

        self.metrics_collector.record_timing(f"api_{endpoint}", latency_ms)
