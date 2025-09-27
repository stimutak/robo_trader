"""
Market Data Manager - Fix for Critical Bug #10: Memory Leaks in Subscriptions

Provides centralized market data subscription management with automatic cleanup,
prevents memory leaks from abandoned subscriptions and duplicate feeds.
"""

import asyncio
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set
from weakref import WeakSet

import structlog

from .market_time import get_market_time

logger = structlog.get_logger(__name__)


@dataclass
class Subscription:
    """Market data subscription tracking."""

    id: str
    symbol: str
    data_type: str  # 'tick', 'bar', 'depth', 'trades'
    callback: Callable
    created_at: datetime = field(default_factory=get_market_time)
    last_data_time: Optional[datetime] = None
    data_count: int = 0
    is_active: bool = True
    subscriber_ref: Optional[weakref.ReferenceType] = None


class MarketDataManager:
    """
    Centralized market data subscription manager.

    Prevents memory leaks by:
    - Tracking all subscriptions centrally
    - Automatic cleanup of dead subscribers
    - Preventing duplicate subscriptions
    - Monitoring stale subscriptions
    - Batching unsubscription requests
    """

    def __init__(
        self,
        max_subscriptions_per_symbol: int = 50,
        cleanup_interval: int = 300,  # 5 minutes
        stale_threshold: int = 1800,  # 30 minutes
        batch_size: int = 10,
    ):
        self.max_subscriptions_per_symbol = max_subscriptions_per_symbol
        self.cleanup_interval = cleanup_interval
        self.stale_threshold = stale_threshold
        self.batch_size = batch_size

        # Subscription tracking
        self.subscriptions: Dict[str, Subscription] = {}
        self.symbol_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.subscriber_subscriptions: Dict[int, Set[str]] = defaultdict(set)

        # Connection tracking
        self.active_connections: WeakSet = WeakSet()
        self.connection_subscribers: Dict[object, Set[str]] = defaultdict(set)

        # Cleanup tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Statistics
        self.stats = {
            "total_subscriptions": 0,
            "active_subscriptions": 0,
            "cleaned_subscriptions": 0,
            "duplicate_requests": 0,
            "failed_subscriptions": 0,
            "data_messages_processed": 0,
        }

    async def start(self) -> None:
        """Start the market data manager."""
        if self.is_running:
            return

        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Market data manager started")

    async def stop(self) -> None:
        """Stop the market data manager and cleanup all subscriptions."""
        self.is_running = False

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Cleanup all subscriptions
        await self._cleanup_all_subscriptions()
        logger.info("Market data manager stopped")

    async def subscribe(
        self,
        symbol: str,
        data_type: str,
        callback: Callable,
        subscriber: Optional[object] = None,
        connection: Optional[object] = None,
    ) -> Optional[str]:
        """
        Subscribe to market data for a symbol.

        Args:
            symbol: Trading symbol
            data_type: Type of data ('tick', 'bar', 'depth', 'trades')
            callback: Function to call with data
            subscriber: Object requesting subscription (for cleanup tracking)
            connection: Connection object (for connection-based cleanup)

        Returns:
            Subscription ID if successful, None if failed
        """
        # Check subscription limits
        if len(self.symbol_subscriptions[symbol]) >= self.max_subscriptions_per_symbol:
            logger.warning(f"Max subscriptions reached for {symbol}")
            self.stats["failed_subscriptions"] += 1
            return None

        # Generate subscription ID
        sub_id = f"{symbol}_{data_type}_{get_market_time().isoformat()}_{id(callback)}"

        # Check for duplicate subscription
        existing_key = f"{symbol}_{data_type}_{id(callback)}"
        for existing_id, sub in self.subscriptions.items():
            if f"{sub.symbol}_{sub.data_type}_{id(sub.callback)}" == existing_key:
                logger.debug(f"Duplicate subscription request for {symbol} {data_type}")
                self.stats["duplicate_requests"] += 1
                return existing_id

        # Create subscription
        subscription = Subscription(
            id=sub_id,
            symbol=symbol,
            data_type=data_type,
            callback=callback,
            subscriber_ref=weakref.ref(subscriber) if subscriber else None,
        )

        # Register subscription
        self.subscriptions[sub_id] = subscription
        self.symbol_subscriptions[symbol].add(sub_id)

        if subscriber:
            self.subscriber_subscriptions[id(subscriber)].add(sub_id)

        if connection:
            self.active_connections.add(connection)
            self.connection_subscribers[connection].add(sub_id)

        # Update statistics
        self.stats["total_subscriptions"] += 1
        self.stats["active_subscriptions"] += 1

        logger.debug(f"Subscribed to {symbol} {data_type}, ID: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from market data.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if successfully unsubscribed
        """
        if subscription_id not in self.subscriptions:
            logger.warning(f"Subscription {subscription_id} not found")
            return False

        subscription = self.subscriptions[subscription_id]

        # Mark as inactive
        subscription.is_active = False

        # Remove from tracking
        symbol = subscription.symbol
        self.symbol_subscriptions[symbol].discard(subscription_id)

        # Remove from subscriber tracking
        if subscription.subscriber_ref:
            subscriber = subscription.subscriber_ref()
            if subscriber:
                self.subscriber_subscriptions[id(subscriber)].discard(subscription_id)

        # Remove from connection tracking
        for conn, subs in self.connection_subscribers.items():
            if subscription_id in subs:
                subs.discard(subscription_id)

        # Remove subscription
        del self.subscriptions[subscription_id]

        # Update statistics
        self.stats["active_subscriptions"] -= 1
        self.stats["cleaned_subscriptions"] += 1

        logger.debug(f"Unsubscribed from {subscription_id}")
        return True

    async def unsubscribe_all(self, subscriber: object) -> int:
        """
        Unsubscribe all subscriptions for a subscriber.

        Args:
            subscriber: Object to unsubscribe

        Returns:
            Number of subscriptions removed
        """
        subscriber_id = id(subscriber)

        if subscriber_id not in self.subscriber_subscriptions:
            return 0

        subscription_ids = list(self.subscriber_subscriptions[subscriber_id])
        count = 0

        for sub_id in subscription_ids:
            if await self.unsubscribe(sub_id):
                count += 1

        logger.debug(f"Unsubscribed {count} subscriptions for subscriber {subscriber_id}")
        return count

    async def process_data(self, symbol: str, data_type: str, data: dict) -> int:
        """
        Process incoming market data and distribute to subscribers.

        Args:
            symbol: Trading symbol
            data_type: Type of data
            data: Market data payload

        Returns:
            Number of callbacks executed
        """
        callbacks_executed = 0
        current_time = get_market_time()

        # Find matching subscriptions
        matching_subs = []
        for sub_id in self.symbol_subscriptions.get(symbol, set()):
            if sub_id in self.subscriptions:
                sub = self.subscriptions[sub_id]
                if sub.is_active and sub.data_type == data_type:
                    matching_subs.append(sub)

        # Execute callbacks
        for subscription in matching_subs:
            try:
                # Check if subscriber still exists
                if subscription.subscriber_ref:
                    subscriber = subscription.subscriber_ref()
                    if subscriber is None:
                        # Subscriber has been garbage collected
                        await self.unsubscribe(subscription.id)
                        continue

                # Execute callback
                if asyncio.iscoroutinefunction(subscription.callback):
                    await subscription.callback(symbol, data_type, data)
                else:
                    subscription.callback(symbol, data_type, data)

                # Update subscription stats
                subscription.last_data_time = current_time
                subscription.data_count += 1
                callbacks_executed += 1

            except Exception as e:
                logger.error(f"Error processing data for subscription {subscription.id}: {e}")
                # Consider unsubscribing on repeated errors

        self.stats["data_messages_processed"] += 1
        return callbacks_executed

    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_subscriptions()
                await self._cleanup_dead_subscribers()
                await self._cleanup_dead_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_stale_subscriptions(self) -> None:
        """Remove subscriptions that haven't received data recently."""
        current_time = get_market_time()
        stale_threshold = timedelta(seconds=self.stale_threshold)

        stale_subscriptions = []

        for sub_id, subscription in self.subscriptions.items():
            if subscription.last_data_time:
                time_since_data = current_time - subscription.last_data_time
                if time_since_data > stale_threshold:
                    stale_subscriptions.append(sub_id)
            else:
                # No data received since creation
                time_since_creation = current_time - subscription.created_at
                if time_since_creation > stale_threshold:
                    stale_subscriptions.append(sub_id)

        # Batch cleanup
        for i in range(0, len(stale_subscriptions), self.batch_size):
            batch = stale_subscriptions[i : i + self.batch_size]
            for sub_id in batch:
                await self.unsubscribe(sub_id)

        if stale_subscriptions:
            logger.info(f"Cleaned up {len(stale_subscriptions)} stale subscriptions")

    async def _cleanup_dead_subscribers(self) -> None:
        """Remove subscriptions for garbage collected subscribers."""
        dead_subscriptions = []

        for sub_id, subscription in self.subscriptions.items():
            if subscription.subscriber_ref:
                subscriber = subscription.subscriber_ref()
                if subscriber is None:
                    dead_subscriptions.append(sub_id)

        # Batch cleanup
        for i in range(0, len(dead_subscriptions), self.batch_size):
            batch = dead_subscriptions[i : i + self.batch_size]
            for sub_id in batch:
                await self.unsubscribe(sub_id)

        if dead_subscriptions:
            logger.info(f"Cleaned up {len(dead_subscriptions)} dead subscriber subscriptions")

    async def _cleanup_dead_connections(self) -> None:
        """Remove subscriptions for dead connections."""
        dead_connections = []

        # Find dead connections (removed from WeakSet)
        for connection in list(self.connection_subscribers.keys()):
            if connection not in self.active_connections:
                dead_connections.append(connection)

        # Cleanup subscriptions for dead connections
        for connection in dead_connections:
            subscription_ids = list(self.connection_subscribers[connection])

            for sub_id in subscription_ids:
                await self.unsubscribe(sub_id)

            del self.connection_subscribers[connection]

        if dead_connections:
            logger.info(f"Cleaned up subscriptions for {len(dead_connections)} dead connections")

    async def _cleanup_all_subscriptions(self) -> None:
        """Remove all subscriptions."""
        subscription_ids = list(self.subscriptions.keys())

        for sub_id in subscription_ids:
            await self.unsubscribe(sub_id)

        logger.info(f"Cleaned up all {len(subscription_ids)} subscriptions")

    def get_statistics(self) -> Dict[str, int]:
        """Get manager statistics."""
        self.stats["active_subscriptions"] = len(self.subscriptions)
        return self.stats.copy()

    def get_subscriptions_by_symbol(self, symbol: str) -> List[str]:
        """Get all subscription IDs for a symbol."""
        return list(self.symbol_subscriptions.get(symbol, set()))

    def get_subscription_details(self, subscription_id: str) -> Optional[Dict]:
        """Get detailed information about a subscription."""
        if subscription_id not in self.subscriptions:
            return None

        sub = self.subscriptions[subscription_id]
        return {
            "id": sub.id,
            "symbol": sub.symbol,
            "data_type": sub.data_type,
            "created_at": sub.created_at.isoformat(),
            "last_data_time": sub.last_data_time.isoformat() if sub.last_data_time else None,
            "data_count": sub.data_count,
            "is_active": sub.is_active,
            "has_subscriber": sub.subscriber_ref is not None and sub.subscriber_ref() is not None,
        }


# Global instance for easy access
market_data_manager = MarketDataManager()


# Convenience functions
async def subscribe_to_data(
    symbol: str, data_type: str, callback: Callable, subscriber: Optional[object] = None
) -> Optional[str]:
    """Subscribe to market data using global manager."""
    return await market_data_manager.subscribe(symbol, data_type, callback, subscriber)


async def unsubscribe_from_data(subscription_id: str) -> bool:
    """Unsubscribe from market data using global manager."""
    return await market_data_manager.unsubscribe(subscription_id)


async def cleanup_subscriber(subscriber: object) -> int:
    """Cleanup all subscriptions for a subscriber."""
    return await market_data_manager.unsubscribe_all(subscriber)


# Export main classes and functions
__all__ = [
    "MarketDataManager",
    "Subscription",
    "market_data_manager",
    "subscribe_to_data",
    "unsubscribe_from_data",
    "cleanup_subscriber",
]
