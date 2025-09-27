"""
Thread-Safe Order ID Generator - Fix for Critical Bug #9: Race Conditions

Provides atomic order ID generation with collision prevention,
ensures unique IDs even under high-frequency concurrent order placement.
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

import structlog

from .market_time import get_market_time

logger = structlog.get_logger(__name__)


@dataclass
class OrderIDStats:
    """Statistics for order ID generation."""

    total_generated: int = 0
    collisions_detected: int = 0
    sequence_resets: int = 0
    prefix_changes: int = 0


class ThreadSafeOrderIDGenerator:
    """
    Thread-safe order ID generator with collision prevention.

    Features:
    - Atomic sequence number generation
    - Timestamp-based prefixes for uniqueness
    - Collision detection and prevention
    - Automatic sequence reset on time boundaries
    - Memory-efficient collision tracking
    """

    def __init__(
        self,
        prefix_format: str = "OID",
        sequence_bits: int = 16,  # Max 65535 orders per timestamp
        collision_tracking_size: int = 10000,
        reset_interval_seconds: int = 300,  # 5 minutes
    ):
        self.prefix_format = prefix_format
        self.sequence_bits = sequence_bits
        self.max_sequence = (1 << sequence_bits) - 1
        self.collision_tracking_size = collision_tracking_size
        self.reset_interval_seconds = reset_interval_seconds

        # Thread-safe state
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._async_lock = asyncio.Lock()  # Async lock for coroutines

        # ID generation state
        self._current_sequence = 0
        self._last_timestamp = 0
        self._current_prefix = ""

        # Collision tracking
        self._generated_ids: Set[str] = set()
        self._last_reset_time = time.time()

        # Statistics
        self.stats = OrderIDStats()

    def _generate_timestamp_prefix(self) -> str:
        """Generate timestamp-based prefix for ID uniqueness."""
        market_time = get_market_time()
        # Use microseconds for high precision
        timestamp_ms = int(market_time.timestamp() * 1000000)
        return f"{self.prefix_format}_{timestamp_ms:016x}"

    def _should_reset_sequence(self, new_prefix: str) -> bool:
        """Determine if sequence should be reset."""
        current_time = time.time()

        # Reset if prefix changed (time boundary crossed)
        if new_prefix != self._current_prefix:
            return True

        # Reset if interval elapsed
        if current_time - self._last_reset_time > self.reset_interval_seconds:
            return True

        # Reset if sequence approaching limit
        if self._current_sequence >= self.max_sequence * 0.9:
            return True

        return False

    def _reset_sequence(self, new_prefix: str) -> None:
        """Reset sequence counter and collision tracking."""
        self._current_sequence = 0
        self._current_prefix = new_prefix
        self._last_reset_time = time.time()

        # Clear collision tracking (keep small set of recent IDs)
        if len(self._generated_ids) > self.collision_tracking_size:
            # Keep only recent IDs (simple LRU-like behavior)
            recent_ids = list(self._generated_ids)[-self.collision_tracking_size // 2 :]
            self._generated_ids = set(recent_ids)

        self.stats.sequence_resets += 1
        if new_prefix != self._current_prefix:
            self.stats.prefix_changes += 1

    def generate_sync(self) -> str:
        """
        Generate order ID synchronously (thread-safe).

        Returns:
            Unique order ID string
        """
        with self._lock:
            # Generate timestamp prefix
            new_prefix = self._generate_timestamp_prefix()

            # Check if sequence reset needed
            if self._should_reset_sequence(new_prefix):
                self._reset_sequence(new_prefix)

            # Generate ID with collision checking
            max_attempts = 100  # Prevent infinite loops
            attempts = 0

            while attempts < max_attempts:
                # Create candidate ID
                order_id = (
                    f"{self._current_prefix}_{self._current_sequence:0{self.sequence_bits//4}x}"
                )

                # Check for collision
                if order_id not in self._generated_ids:
                    # Success - register and return
                    self._generated_ids.add(order_id)
                    self.stats.total_generated += 1

                    # Increment sequence for next call
                    self._current_sequence += 1

                    logger.debug(f"Generated order ID: {order_id}")
                    return order_id

                # Collision detected
                self.stats.collisions_detected += 1
                self._current_sequence += 1
                attempts += 1

                # Check sequence overflow
                if self._current_sequence > self.max_sequence:
                    # Force new prefix to avoid sequence overflow
                    time.sleep(0.001)  # Ensure timestamp advances
                    new_prefix = self._generate_timestamp_prefix()
                    self._reset_sequence(new_prefix)

            # Fallback: use timestamp + thread ID + attempt count
            fallback_id = f"{self.prefix_format}_FALLBACK_{int(time.time()*1000000):016x}_{threading.get_ident():08x}_{attempts:04x}"
            logger.warning(f"Using fallback order ID after {max_attempts} attempts: {fallback_id}")

            self._generated_ids.add(fallback_id)
            self.stats.total_generated += 1
            return fallback_id

    async def generate_async(self) -> str:
        """
        Generate order ID asynchronously (async-safe).

        Returns:
            Unique order ID string
        """
        async with self._async_lock:
            # Use sync generation with proper locking
            return self.generate_sync()

    def is_valid_id(self, order_id: str) -> bool:
        """
        Check if an order ID was generated by this instance.

        Args:
            order_id: Order ID to validate

        Returns:
            True if ID was generated by this instance
        """
        with self._lock:
            return order_id in self._generated_ids

    def mark_id_used(self, order_id: str) -> bool:
        """
        Mark an external order ID as used to prevent collisions.

        Args:
            order_id: External order ID to register

        Returns:
            True if ID was successfully registered (not already used)
        """
        with self._lock:
            if order_id in self._generated_ids:
                return False

            self._generated_ids.add(order_id)
            return True

    def get_statistics(self) -> Dict[str, int]:
        """Get generation statistics."""
        with self._lock:
            return {
                "total_generated": self.stats.total_generated,
                "collisions_detected": self.stats.collisions_detected,
                "sequence_resets": self.stats.sequence_resets,
                "prefix_changes": self.stats.prefix_changes,
                "current_sequence": self._current_sequence,
                "tracked_ids": len(self._generated_ids),
                "collision_rate": (
                    self.stats.collisions_detected / max(1, self.stats.total_generated) * 100
                ),
            }

    def cleanup_old_ids(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old tracked IDs to prevent memory growth.

        Args:
            max_age_seconds: Maximum age of IDs to keep

        Returns:
            Number of IDs removed
        """
        with self._lock:
            current_time = int(time.time() * 1000000)
            cutoff_time = current_time - (max_age_seconds * 1000000)

            # Extract timestamp from tracked IDs and filter
            old_ids = []
            for order_id in self._generated_ids:
                try:
                    # Parse timestamp from ID format: PREFIX_TIMESTAMP_SEQUENCE
                    parts = order_id.split("_")
                    if len(parts) >= 2 and parts[0] == self.prefix_format:
                        timestamp_hex = parts[1]
                        timestamp = int(timestamp_hex, 16)
                        if timestamp < cutoff_time:
                            old_ids.append(order_id)
                except (ValueError, IndexError):
                    # Invalid format, keep for safety
                    continue

            # Remove old IDs
            for old_id in old_ids:
                self._generated_ids.discard(old_id)

            if old_ids:
                logger.info(f"Cleaned up {len(old_ids)} old order IDs")

            return len(old_ids)


class OrderIDManager:
    """
    High-level order ID management with multiple generators.

    Supports different ID formats for different order types or systems.
    """

    def __init__(self):
        self._generators: Dict[str, ThreadSafeOrderIDGenerator] = {}
        self._default_generator = ThreadSafeOrderIDGenerator()

    def get_generator(self, prefix: str = "OID") -> ThreadSafeOrderIDGenerator:
        """
        Get or create a generator for a specific prefix.

        Args:
            prefix: ID prefix (e.g., "BUY", "SELL", "STP")

        Returns:
            Thread-safe generator instance
        """
        if prefix not in self._generators:
            self._generators[prefix] = ThreadSafeOrderIDGenerator(prefix_format=prefix)

        return self._generators[prefix]

    def generate_id(self, prefix: str = "OID") -> str:
        """
        Generate order ID with specific prefix.

        Args:
            prefix: ID prefix

        Returns:
            Unique order ID
        """
        generator = self.get_generator(prefix)
        return generator.generate_sync()

    async def generate_id_async(self, prefix: str = "OID") -> str:
        """
        Generate order ID asynchronously with specific prefix.

        Args:
            prefix: ID prefix

        Returns:
            Unique order ID
        """
        generator = self.get_generator(prefix)
        return await generator.generate_async()

    def get_all_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all generators."""
        stats = {}

        # Default generator
        stats["default"] = self._default_generator.get_statistics()

        # Named generators
        for prefix, generator in self._generators.items():
            stats[prefix] = generator.get_statistics()

        return stats

    def cleanup_all(self, max_age_seconds: int = 3600) -> int:
        """Clean up old IDs from all generators."""
        total_cleaned = 0

        # Default generator
        total_cleaned += self._default_generator.cleanup_old_ids(max_age_seconds)

        # Named generators
        for generator in self._generators.values():
            total_cleaned += generator.cleanup_old_ids(max_age_seconds)

        return total_cleaned


# Global instances for easy access
order_id_generator = ThreadSafeOrderIDGenerator()
order_id_manager = OrderIDManager()


# Convenience functions
def generate_order_id(prefix: str = "OID") -> str:
    """Generate unique order ID synchronously."""
    if prefix == "OID":
        return order_id_generator.generate_sync()
    else:
        return order_id_manager.generate_id(prefix)


async def generate_order_id_async(prefix: str = "OID") -> str:
    """Generate unique order ID asynchronously."""
    if prefix == "OID":
        return await order_id_generator.generate_async()
    else:
        return await order_id_manager.generate_id_async(prefix)


def validate_order_id(order_id: str) -> bool:
    """Validate if order ID was generated by this system."""
    return order_id_generator.is_valid_id(order_id)


# Export main classes and functions
__all__ = [
    "ThreadSafeOrderIDGenerator",
    "OrderIDManager",
    "OrderIDStats",
    "order_id_generator",
    "order_id_manager",
    "generate_order_id",
    "generate_order_id_async",
    "validate_order_id",
]
