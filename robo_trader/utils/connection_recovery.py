"""
Connection Recovery and State Management Utilities - Fix for Critical Bugs #2, #8, #9

Provides state recovery after connection loss, network heartbeat monitoring,
and comprehensive connection failure handling.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, Optional, Set

from ..logger import get_logger
from .market_time import get_market_time

logger = get_logger(__name__)


@dataclass
class PositionState:
    """Represents a position state for recovery."""

    symbol: str
    quantity: int
    entry_price: Decimal
    current_value: Decimal
    last_updated: datetime


class StateRecoveryManager:
    """Handles state recovery after connection loss."""

    def __init__(self):
        self.positions: Dict[str, PositionState] = {}
        self.pending_orders: Set[str] = set()
        self.last_successful_sync: Optional[datetime] = None

    async def resync_state_with_broker(self, connection_manager) -> bool:
        """Resync internal state with broker after reconnection."""
        try:
            logger.info("Starting state resynchronization with broker...")

            if not (
                connection_manager and hasattr(connection_manager, "ib") and connection_manager.ib
            ):
                logger.error("No valid broker connection available for state resync")
                return False

            broker_positions = await connection_manager.ib.positionsAsync()
            broker_orders = await connection_manager.ib.ordersAsync()

            # Build fresh state without mutating existing caches until successful
            fresh_positions: Dict[str, PositionState] = {}
            fresh_pending: Set[str] = set()

            for pos in broker_positions:
                if pos.position != 0:  # Only track actual positions
                    position_state = PositionState(
                        symbol=pos.contract.symbol,
                        quantity=int(pos.position),
                        entry_price=Decimal(str(pos.avgCost)),
                        current_value=Decimal(str(pos.position * pos.avgCost)),
                        last_updated=get_market_time(),
                    )
                    fresh_positions[pos.contract.symbol] = position_state

            for order in broker_orders:
                if order.orderStatus.status in ["PreSubmitted", "Submitted", "PendingSubmit"]:
                    fresh_pending.add(str(order.orderId))

            old_position_count = len(self.positions)

            # Atomically swap state now that we have valid data
            self.positions.clear()
            self.positions.update(fresh_positions)
            self.pending_orders.clear()
            self.pending_orders.update(fresh_pending)
            self.last_successful_sync = get_market_time()

            logger.info(
                f"State resync complete: {len(self.positions)} positions, "
                f"{len(self.pending_orders)} pending orders "
                f"(was {old_position_count} positions)"
            )
            return True

        except Exception as e:
            logger.error(f"State resync failed: {e}")
            return False

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of current position state."""
        total_value = sum(pos.current_value for pos in self.positions.values())

        return {
            "total_positions": len(self.positions),
            "pending_orders": len(self.pending_orders),
            "total_value": float(total_value),
            "last_sync": self.last_successful_sync.isoformat()
            if self.last_successful_sync
            else None,
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "entry_price": float(pos.entry_price),
                    "current_value": float(pos.current_value),
                }
                for symbol, pos in self.positions.items()
            },
        }

    def update_position(self, symbol: str, quantity: int, price: Decimal) -> None:
        """Update position state manually."""
        if quantity == 0:
            # Position closed
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            # Position opened or modified
            self.positions[symbol] = PositionState(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_value=Decimal(str(quantity)) * price,
                last_updated=get_market_time(),
            )


class NetworkHeartbeatMonitor:
    """Monitors network connectivity with periodic heartbeats."""

    def __init__(self, heartbeat_interval: int = 30, timeout: int = 5):
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self.running = False
        self.last_heartbeat: Optional[datetime] = None
        self.failure_count = 0
        self.failure_callbacks: list[Callable] = []

    def register_failure_callback(self, callback: Callable) -> None:
        """Register callback to be called on heartbeat failure."""
        self.failure_callbacks.append(callback)

    async def start_monitoring(self, connection_manager) -> None:
        """Start heartbeat monitoring."""
        self.running = True
        logger.info(f"Starting network heartbeat monitor (interval: {self.heartbeat_interval}s)")

        while self.running:
            try:
                await self._perform_heartbeat(connection_manager)
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                logger.info("Heartbeat monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(self.heartbeat_interval)

        logger.info("Heartbeat monitor stopped")

    async def _perform_heartbeat(self, connection_manager) -> None:
        """Perform a single heartbeat check."""
        try:
            if connection_manager and hasattr(connection_manager, "ib") and connection_manager.ib:
                # Use reqCurrentTime as heartbeat - lightweight operation
                start_time = time.time()
                await asyncio.wait_for(
                    connection_manager.ib.reqCurrentTimeAsync(), timeout=self.timeout
                )

                # Heartbeat successful
                latency = time.time() - start_time
                self.last_heartbeat = get_market_time()
                self.failure_count = 0

                if latency > 2.0:  # Warn on high latency
                    logger.warning(f"High heartbeat latency: {latency:.2f}s")
                else:
                    logger.debug(f"Heartbeat OK (latency: {latency:.3f}s)")

            else:
                raise Exception("No valid broker connection available")

        except asyncio.TimeoutError:
            self.failure_count += 1
            logger.error(f"Heartbeat timeout (failure #{self.failure_count})")
            await self._handle_heartbeat_failure("timeout")

        except Exception as e:
            self.failure_count += 1
            logger.error(f"Heartbeat failed: {e} (failure #{self.failure_count})")
            await self._handle_heartbeat_failure(str(e))

    async def _handle_heartbeat_failure(self, reason: str) -> None:
        """Handle heartbeat failure."""
        if self.failure_count >= 3:  # 3 consecutive failures
            logger.critical(f"Multiple heartbeat failures detected - connection may be dead")

            # Trigger failure callbacks
            for callback in self.failure_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(reason)
                    else:
                        callback(reason)
                except Exception as e:
                    logger.error(f"Error in heartbeat failure callback: {e}")

    def stop(self) -> None:
        """Stop heartbeat monitoring."""
        self.running = False

    def get_status(self) -> Dict[str, Any]:
        """Get heartbeat monitor status."""
        return {
            "running": self.running,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "failure_count": self.failure_count,
            "heartbeat_interval": self.heartbeat_interval,
            "timeout": self.timeout,
        }


class OrderRateLimiter:
    """Rate limiter to prevent hitting IB's order rate limits."""

    def __init__(self, max_per_second: int = 2, max_per_minute: int = 50):
        self.max_per_second = max_per_second
        self.max_per_minute = max_per_minute
        self.second_times: list[float] = []
        self.minute_times: list[float] = []

    async def acquire(self) -> None:
        """Acquire permission to send an order - blocks if rate limited."""
        now = time.time()

        # Clean old timestamps
        self.second_times = [t for t in self.second_times if now - t < 1.0]
        self.minute_times = [t for t in self.minute_times if now - t < 60.0]

        # Check second rate limit
        if len(self.second_times) >= self.max_per_second:
            wait_time = 1.0 - (now - self.second_times[0])
            if wait_time > 0:
                logger.info(f"Rate limited: waiting {wait_time:.2f}s (second limit)")
                await asyncio.sleep(wait_time)

        # Check minute rate limit
        if len(self.minute_times) >= self.max_per_minute:
            wait_time = 60.0 - (now - self.minute_times[0])
            if wait_time > 0:
                logger.warning(f"Rate limited: waiting {wait_time:.1f}s (minute limit)")
                await asyncio.sleep(wait_time)

        # Record the order
        current_time = time.time()
        self.second_times.append(current_time)
        self.minute_times.append(current_time)

    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        now = time.time()
        recent_second = len([t for t in self.second_times if now - t < 1.0])
        recent_minute = len([t for t in self.minute_times if now - t < 60.0])

        return {
            "orders_last_second": recent_second,
            "orders_last_minute": recent_minute,
            "second_limit": self.max_per_second,
            "minute_limit": self.max_per_minute,
            "second_remaining": max(0, self.max_per_second - recent_second),
            "minute_remaining": max(0, self.max_per_minute - recent_minute),
        }


class TaskManager:
    """Manages background tasks with proper cleanup."""

    def __init__(self):
        self._background_tasks: Set[asyncio.Task] = set()

    def create_background_task(self, coro, name: str = None) -> asyncio.Task:
        """Create a background task and track it for cleanup."""
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        # Log task completion/failure
        def log_completion(completed_task):
            if completed_task.cancelled():
                logger.debug(f"Background task cancelled: {name or 'unnamed'}")
            elif completed_task.exception():
                logger.error(
                    f"Background task failed: {name or 'unnamed'} - {completed_task.exception()}"
                )
            else:
                logger.debug(f"Background task completed: {name or 'unnamed'}")

        task.add_done_callback(log_completion)
        return task

    async def cleanup_all_tasks(self) -> None:
        """Cancel and cleanup all background tasks."""
        if not self._background_tasks:
            return

        logger.info(f"Cleaning up {len(self._background_tasks)} background tasks...")

        # Cancel all tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for all to finish with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True), timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some background tasks did not finish within timeout")

        self._background_tasks.clear()
        logger.info("Background task cleanup completed")

    def get_status(self) -> Dict[str, Any]:
        """Get task manager status."""
        running_tasks = [t for t in self._background_tasks if not t.done()]
        completed_tasks = [t for t in self._background_tasks if t.done()]

        return {
            "total_tasks": len(self._background_tasks),
            "running_tasks": len(running_tasks),
            "completed_tasks": len(completed_tasks),
            "task_names": [t.get_name() for t in running_tasks if t.get_name() != "Task-*"],
        }


# Convenience instances for global use
state_recovery_manager = StateRecoveryManager()
heartbeat_monitor = NetworkHeartbeatMonitor()
rate_limiter = OrderRateLimiter()
task_manager = TaskManager()


# Export main classes and instances
__all__ = [
    "StateRecoveryManager",
    "NetworkHeartbeatMonitor",
    "OrderRateLimiter",
    "TaskManager",
    "PositionState",
    "state_recovery_manager",
    "heartbeat_monitor",
    "rate_limiter",
    "task_manager",
]
