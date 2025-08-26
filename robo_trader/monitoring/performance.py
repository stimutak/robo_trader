"""
Performance monitoring module for the trading system.

This implements Phase 1 F5: Add Basic Performance Monitoring
- Tracks latency and throughput metrics
- Provides real-time visibility into system performance
- Integrates with the dashboard
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from robo_trader.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Latency metrics (in milliseconds)
    data_fetch_latency: float = 0.0
    signal_generation_latency: float = 0.0
    order_execution_latency: float = 0.0
    database_write_latency: float = 0.0
    total_processing_latency: float = 0.0

    # Throughput metrics
    symbols_per_second: float = 0.0
    orders_per_minute: int = 0
    data_points_per_second: int = 0

    # System metrics
    active_connections: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Trading metrics
    total_symbols_processed: int = 0
    total_orders_placed: int = 0
    total_trades_executed: int = 0
    success_rate: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Monitors and tracks system performance metrics."""

    def __init__(self, window_size: int = 100, history_minutes: int = 60):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of samples to keep for moving averages
            history_minutes: Minutes of history to maintain
        """
        self.window_size = window_size
        self.history_minutes = history_minutes

        # Latency tracking (using deque for efficient sliding windows)
        self.data_fetch_samples = deque(maxlen=window_size)
        self.signal_gen_samples = deque(maxlen=window_size)
        self.order_exec_samples = deque(maxlen=window_size)
        self.db_write_samples = deque(maxlen=window_size)

        # Throughput tracking
        self.symbol_timestamps = deque(maxlen=1000)
        self.order_timestamps = deque(maxlen=1000)
        self.data_point_timestamps = deque(maxlen=10000)

        # Historical metrics
        self.history: List[PerformanceMetrics] = []
        self.max_history_size = history_minutes * 60  # One sample per second

        # Counters
        self.total_symbols_processed = 0
        self.total_orders_placed = 0
        self.total_trades_executed = 0
        self.successful_operations = 0
        self.failed_operations = 0

        # Timing contexts
        self._timers: Dict[str, float] = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self._timers[operation] = time.perf_counter()

    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and return duration in milliseconds.

        Args:
            operation: Name of the operation

        Returns:
            Duration in milliseconds
        """
        if operation not in self._timers:
            logger.warning(f"Timer for {operation} was not started")
            return 0.0

        duration_ms = (time.perf_counter() - self._timers[operation]) * 1000
        del self._timers[operation]

        # Record the sample based on operation type
        if operation == "data_fetch":
            self.data_fetch_samples.append(duration_ms)
        elif operation == "signal_generation":
            self.signal_gen_samples.append(duration_ms)
        elif operation == "order_execution":
            self.order_exec_samples.append(duration_ms)
        elif operation == "database_write":
            self.db_write_samples.append(duration_ms)

        return duration_ms

    async def record_symbol_processed(self, symbol: str, success: bool = True) -> None:
        """Record that a symbol was processed."""
        self.total_symbols_processed += 1
        self.symbol_timestamps.append(datetime.now())

        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        logger.debug(f"Processed {symbol} - Total: {self.total_symbols_processed}")

    async def record_order_placed(self, symbol: str, quantity: int) -> None:
        """Record that an order was placed."""
        self.total_orders_placed += 1
        self.order_timestamps.append(datetime.now())
        logger.debug(f"Order placed for {symbol} qty={quantity}")

    async def record_trade_executed(self, symbol: str, side: str, quantity: int) -> None:
        """Record that a trade was executed."""
        self.total_trades_executed += 1
        logger.debug(f"Trade executed: {side} {quantity} {symbol}")

    async def record_data_points(self, count: int) -> None:
        """Record that data points were processed."""
        for _ in range(count):
            self.data_point_timestamps.append(datetime.now())

    def calculate_throughput(self) -> Dict[str, float]:
        """Calculate current throughput metrics."""
        now = datetime.now()

        # Symbols per second (last 10 seconds)
        cutoff = now - timedelta(seconds=10)
        recent_symbols = sum(1 for ts in self.symbol_timestamps if ts > cutoff)
        symbols_per_second = recent_symbols / 10.0 if recent_symbols > 0 else 0.0

        # Orders per minute
        cutoff = now - timedelta(minutes=1)
        recent_orders = sum(1 for ts in self.order_timestamps if ts > cutoff)

        # Data points per second (last 10 seconds)
        cutoff = now - timedelta(seconds=10)
        recent_data_points = sum(1 for ts in self.data_point_timestamps if ts > cutoff)
        data_points_per_second = recent_data_points / 10.0 if recent_data_points > 0 else 0.0

        return {
            "symbols_per_second": round(symbols_per_second, 2),
            "orders_per_minute": recent_orders,
            "data_points_per_second": round(data_points_per_second, 0),
        }

    def get_average_latencies(self) -> Dict[str, float]:
        """Get average latencies for all operations."""
        return {
            "data_fetch": (
                sum(self.data_fetch_samples) / len(self.data_fetch_samples)
                if self.data_fetch_samples
                else 0.0
            ),
            "signal_generation": (
                sum(self.signal_gen_samples) / len(self.signal_gen_samples)
                if self.signal_gen_samples
                else 0.0
            ),
            "order_execution": (
                sum(self.order_exec_samples) / len(self.order_exec_samples)
                if self.order_exec_samples
                else 0.0
            ),
            "database_write": (
                sum(self.db_write_samples) / len(self.db_write_samples)
                if self.db_write_samples
                else 0.0
            ),
        }

    def get_system_metrics(self) -> Dict[str, float]:
        """Get system resource metrics."""
        try:
            import psutil

            process = psutil.Process()
            return {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_usage_percent": process.cpu_percent(interval=0.1),
                "num_threads": process.num_threads(),
            }
        except ImportError:
            logger.debug("psutil not installed, system metrics unavailable")
            return {"memory_usage_mb": 0.0, "cpu_usage_percent": 0.0, "num_threads": 0}

    async def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot."""
        latencies = self.get_average_latencies()
        throughput = self.calculate_throughput()
        system = self.get_system_metrics()

        success_rate = (
            self.successful_operations
            / (self.successful_operations + self.failed_operations)
            if (self.successful_operations + self.failed_operations) > 0
            else 1.0
        )

        metrics = PerformanceMetrics(
            # Latencies
            data_fetch_latency=round(latencies["data_fetch"], 2),
            signal_generation_latency=round(latencies["signal_generation"], 2),
            order_execution_latency=round(latencies["order_execution"], 2),
            database_write_latency=round(latencies["database_write"], 2),
            total_processing_latency=round(sum(latencies.values()), 2),
            # Throughput
            symbols_per_second=throughput["symbols_per_second"],
            orders_per_minute=throughput["orders_per_minute"],
            data_points_per_second=int(throughput["data_points_per_second"]),
            # System
            memory_usage_mb=round(system["memory_usage_mb"], 2),
            cpu_usage_percent=round(system["cpu_usage_percent"], 2),
            # Trading
            total_symbols_processed=self.total_symbols_processed,
            total_orders_placed=self.total_orders_placed,
            total_trades_executed=self.total_trades_executed,
            success_rate=round(success_rate, 4),
        )

        # Add to history
        self.history.append(metrics)
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

        return metrics

    async def log_performance_summary(self) -> None:
        """Log a performance summary."""
        metrics = await self.get_current_metrics()

        logger.info(
            f"Performance Summary: "
            f"Throughput: {metrics.symbols_per_second} sym/s, "
            f"{metrics.orders_per_minute} orders/min | "
            f"Latency: {metrics.total_processing_latency:.1f}ms total | "
            f"Success rate: {metrics.success_rate:.1%} | "
            f"Memory: {metrics.memory_usage_mb:.1f}MB"
        )

    def get_performance_report(self) -> Dict:
        """Generate a comprehensive performance report."""
        if not self.history:
            return {}

        recent_metrics = self.history[-min(60, len(self.history)) :]  # Last minute

        return {
            "current": self.history[-1].__dict__ if self.history else {},
            "averages": {
                "avg_data_fetch_latency": sum(m.data_fetch_latency for m in recent_metrics)
                / len(recent_metrics),
                "avg_total_latency": sum(m.total_processing_latency for m in recent_metrics)
                / len(recent_metrics),
                "avg_throughput": sum(m.symbols_per_second for m in recent_metrics)
                / len(recent_metrics),
            },
            "peaks": {
                "max_latency": max(m.total_processing_latency for m in recent_metrics),
                "max_throughput": max(m.symbols_per_second for m in recent_metrics),
                "max_memory": max(m.memory_usage_mb for m in recent_metrics),
            },
            "totals": {
                "total_symbols": self.total_symbols_processed,
                "total_orders": self.total_orders_placed,
                "total_trades": self.total_trades_executed,
            },
        }


# Global instance for easy access
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


# Context manager for timing operations
class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation: str, monitor: Optional[PerformanceMonitor] = None):
        self.operation = operation
        self.monitor = monitor or get_monitor()

    def __enter__(self):
        self.monitor.start_timer(self.operation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.monitor.end_timer(self.operation)
        if duration > 1000:  # Log slow operations (> 1 second)
            logger.warning(f"Slow operation: {self.operation} took {duration:.1f}ms")