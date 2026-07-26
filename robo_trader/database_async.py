"""
Async database integration for trading data persistence.

This implements Phase 1 F3: Async Database Operations
- Converts all SQLite operations to async using aiosqlite
- Implements connection pooling for database access
- Ensures no event loop blocking
- Multi-portfolio support: all user-scoped tables partitioned by portfolio_id
"""

import asyncio
import hashlib
import hmac
import json
import os
import stat
import threading
import uuid
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiosqlite

from robo_trader.database_validator import DatabaseValidator, ValidationError
from robo_trader.logger import get_logger
from robo_trader.safety.sqlite_identity import (
    SQLiteDescriptorIdentity,
    SQLiteIdentityError,
    SQLitePathBinding,
    lexical_path_preserving_leaf,
    sqlite_connection_file_identity,
)

logger = get_logger(__name__)

_EXACT_AIOSQLITE_CLOSE = aiosqlite.Connection.close

DB_PATH = lexical_path_preserving_leaf(Path(os.getenv("RT_DB_PATH", "trading_data.db")))

# Default portfolio ID for backward compatibility
DEFAULT_PORTFOLIO_ID = "default"


class SafetyAllocationSnapshotError(ValidationError):
    """Stored allocation state cannot form authoritative safety evidence."""


class SafetyDatabasePoolError(SafetyAllocationSnapshotError):
    """The allocation-ledger connection pool cannot be used safely."""


@dataclass(frozen=True, slots=True)
class _PoolRecoveryFailure:
    """Latched reason shared with callers already waiting on the pool."""

    reason: str


@dataclass(slots=True)
class _PoolGenerationState:
    """Broadcast failure state for exactly one connection-pool generation."""

    failure_event: asyncio.Event = field(default_factory=asyncio.Event)
    failure: Optional[_PoolRecoveryFailure] = None


@dataclass(frozen=True, slots=True)
class SafetyPortfolioAllocation:
    """One validated portfolio allocation from a coherent database snapshot."""

    portfolio_id: str
    symbol: str
    quantity: Decimal
    updated_at: Optional[datetime]


@dataclass(frozen=True)
class SafetyAllocationSnapshot:
    """Immutable cross-portfolio allocation truth for one symbol.

    The current ledger schema has no authoritative broker contract identifier.
    This object therefore must be bound to a separately qualified broker
    contract before it can contribute to order authorization.
    """

    snapshot_id: str
    observed_at: datetime
    symbol: str
    allocations: Tuple[SafetyPortfolioAllocation, ...]
    aggregate_allocated_quantity: Decimal
    has_offsetting_allocations: bool
    complete: bool
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    _producer_marker: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._producer_marker is not _ALLOCATION_SNAPSHOT_PRODUCER_MARKER:
            raise SafetyAllocationSnapshotError(
                "allocation snapshot was not created by the trusted ledger producer"
            )
        path = Path(self.database_path)
        if (
            not path.is_absolute()
            or str(path) != self.database_path
            or lexical_path_preserving_leaf(path) != path
        ):
            raise SafetyAllocationSnapshotError(
                "allocation snapshot database path must be absolute and canonical"
            )
        if (
            not isinstance(self.database_identity, str)
            or not self.database_identity
            or self.database_identity != self.database_identity.strip()
            or len(self.database_identity) > 256
            or any(ord(character) < 32 for character in self.database_identity)
        ):
            raise SafetyAllocationSnapshotError(
                "allocation snapshot database identity is malformed"
            )
        for field_name in ("database_device", "database_inode"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise SafetyAllocationSnapshotError(
                    f"allocation snapshot {field_name} is malformed"
                )


_ALLOCATION_SNAPSHOT_PRODUCER_MARKER = object()
_ALLOCATION_SNAPSHOT_REGISTRY_LOCK = threading.Lock()
_ALLOCATION_SNAPSHOT_REGISTRY: Dict[
    int, Tuple["weakref.ReferenceType[SafetyAllocationSnapshot]", str]
] = {}


def _allocation_snapshot_digest(snapshot: SafetyAllocationSnapshot) -> str:
    payload = {
        "snapshot_id": snapshot.snapshot_id,
        "observed_at": snapshot.observed_at.isoformat(),
        "symbol": snapshot.symbol,
        "allocations": [
            {
                "portfolio_id": allocation.portfolio_id,
                "symbol": allocation.symbol,
                "quantity": str(allocation.quantity),
                "updated_at": (
                    allocation.updated_at.isoformat() if allocation.updated_at is not None else None
                ),
            }
            for allocation in snapshot.allocations
        ],
        "aggregate_allocated_quantity": str(snapshot.aggregate_allocated_quantity),
        "has_offsetting_allocations": snapshot.has_offsetting_allocations,
        "complete": snapshot.complete,
        "database_path": snapshot.database_path,
        "database_identity": snapshot.database_identity,
        "database_device": snapshot.database_device,
        "database_inode": snapshot.database_inode,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _produce_safety_allocation_snapshot(**fields: object) -> SafetyAllocationSnapshot:
    snapshot = SafetyAllocationSnapshot(
        **fields,
        _producer_marker=_ALLOCATION_SNAPSHOT_PRODUCER_MARKER,
    )
    snapshot_id = id(snapshot)

    def discard(reference: "weakref.ReferenceType[SafetyAllocationSnapshot]") -> None:
        with _ALLOCATION_SNAPSHOT_REGISTRY_LOCK:
            registered = _ALLOCATION_SNAPSHOT_REGISTRY.get(snapshot_id)
            if registered is not None and registered[0] is reference:
                _ALLOCATION_SNAPSHOT_REGISTRY.pop(snapshot_id, None)

    reference = weakref.ref(snapshot, discard)
    with _ALLOCATION_SNAPSHOT_REGISTRY_LOCK:
        _ALLOCATION_SNAPSHOT_REGISTRY[snapshot_id] = (
            reference,
            _allocation_snapshot_digest(snapshot),
        )
    return snapshot


def assert_producer_owned_safety_allocation_snapshot(
    snapshot: SafetyAllocationSnapshot,
) -> None:
    """Reject copied, reconstructed, or altered allocation evidence."""

    if type(snapshot) is not SafetyAllocationSnapshot:
        raise SafetyAllocationSnapshotError(
            "allocation snapshot must be the exact trusted producer type"
        )
    with _ALLOCATION_SNAPSHOT_REGISTRY_LOCK:
        registered = _ALLOCATION_SNAPSHOT_REGISTRY.get(id(snapshot))
    if (
        registered is None
        or registered[0]() is not snapshot
        or not hmac.compare_digest(registered[1], _allocation_snapshot_digest(snapshot))
    ):
        raise SafetyAllocationSnapshotError(
            "allocation snapshot is not registered producer-owned evidence"
        )


class AsyncTradingDatabase:
    """Async database manager for trading data persistence."""

    def __init__(self, db_path: Path = DB_PATH, pool_size: int = 5):
        """Initialize async database with connection pooling."""
        self.db_path = lexical_path_preserving_leaf(db_path)
        self.pool_size = pool_size
        self._pool: List[aiosqlite.Connection] = []
        self._leased_connections: List[aiosqlite.Connection] = []
        self._quarantined_connections: List[aiosqlite.Connection] = []
        # Object-identity evidence only: aiosqlite 0.19 marks its private
        # fields closed even when the underlying close raises, so those fields
        # cannot prove revocation.  Membership is recorded solely after the
        # exact captured close coroutine completes successfully.
        self._proven_closed_connections: weakref.WeakSet = weakref.WeakSet()
        self._available: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._pool_generation = _PoolGenerationState()
        self._initialized = False
        self._lock = asyncio.Lock()
        self._ensure_lock = asyncio.Lock()
        self._closed = False
        self._lifecycle_revision = 0
        self._pool_recovery_failure: Optional[_PoolRecoveryFailure] = None
        self._expected_database_file_identity = self._existing_database_file_identity()

    def _existing_database_file_identity(self) -> Optional[Tuple[int, int]]:
        """Capture a pre-existing regular ledger without following its leaf."""

        try:
            metadata = os.lstat(self.db_path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SafetyAllocationSnapshotError(
                "configured allocation ledger identity cannot be inspected"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SafetyAllocationSnapshotError(
                "configured allocation ledger must be a non-symlink regular file"
            )
        return metadata.st_dev, metadata.st_ino

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize database and connection pool."""
        async with self._lock:
            await self._initialize_locked()

    async def _initialize_locked(self) -> None:
        """Initialize while the lifecycle lock is held by the caller."""

        if self._quarantined_connections:
            raise SafetyDatabasePoolError(
                "allocation ledger pool cannot initialize while quarantined "
                "connections remain unresolved"
            )
        if self._initialized:
            return

        self._closed = False
        binding: Optional[SQLitePathBinding] = None
        try:
            expected = self._expected_database_file_identity
            binding = SQLitePathBinding.open_for_initialization(
                self.db_path,
                create=expected is None,
            )
            if expected is not None and (binding.device, binding.inode) != expected:
                raise SafetyAllocationSnapshotError(
                    "configured allocation ledger was replaced before initialization"
                )
            self._expected_database_file_identity = (binding.device, binding.inode)
            binding.assert_path_identity()

            # Schema and pool PRAGMAs are permitted only after SQLite's own
            # descriptor is tied to the pre-mutation guardian.
            await self._init_database(binding)
            binding.assert_path_identity()

            for _ in range(self.pool_size):
                conn = await aiosqlite.connect(self.db_path)
                self._quarantine_pool_connection(conn)
                try:
                    pool_binding = binding.bind_sqlite_connection(
                        await self._sqlite_descriptor_identity(conn)
                    )
                    pool_binding.assert_connection_identity(
                        await self._sqlite_descriptor_identity(conn)
                    )
                    await conn.execute("PRAGMA journal_mode=WAL")
                    await conn.execute("PRAGMA busy_timeout=5000")
                    pool_binding.assert_connection_identity(
                        await self._sqlite_descriptor_identity(conn)
                    )
                except BaseException:
                    # The outer initialization cleanup closes every tracked
                    # partial connection through the exact close primitive.
                    raise
                self._pool.append(conn)
                self._forget_quarantined_connection(conn)
                # The queue is sized to the pool and this initializer is its
                # sole producer.  Avoid introducing a cancellation point
                # after the connection has joined the authoritative pool.
                self._available.put_nowait(conn)

            binding.assert_path_identity()
            self._pool_recovery_failure = None
            self._initialized = True
            self._lifecycle_revision += 1
            logger.info(
                f"Async database initialized at {self.db_path} "
                f"with {self.pool_size} connections"
            )
        except SafetyAllocationSnapshotError:
            await self._close_partial_pool()
            raise
        except (SQLiteIdentityError, OSError, aiosqlite.Error) as exc:
            await self._close_partial_pool()
            raise SafetyAllocationSnapshotError(
                "allocation ledger identity cannot be proven during initialization"
            ) from exc
        except BaseException:
            # Cancellation and interpreter-level aborts must not strand
            # partially opened SQLite connections or descriptors.
            await self._close_partial_pool()
            raise
        finally:
            if binding is not None:
                binding.close()

    def _retire_pool_generation(self, reason: str) -> List[aiosqlite.Connection]:
        """Synchronously fail one generation before any cleanup can suspend."""

        generation = self._pool_generation
        failure = generation.failure or _PoolRecoveryFailure(reason)
        generation.failure = failure
        generation.failure_event.set()
        # Retirement is terminal for the entire generation. Checked-out
        # handles are force-closed too; their stale context finalizers only
        # release ownership and never touch a later generation.
        for connection in self._pool:
            self._quarantine_pool_connection(connection)
        connections = list(self._quarantined_connections)
        self._pool.clear()
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._available = asyncio.Queue(maxsize=self.pool_size)
        self._pool_generation = _PoolGenerationState()
        self._pool_recovery_failure = None
        return connections

    async def _close_partial_pool(self) -> None:
        """Close connections opened by an initialization attempt."""

        self._retire_pool_generation("database initialization failed")
        await self._close_quarantined_connections(
            "initialization cleanup could not close every connection"
        )

    async def close(self):
        """Close all connections in the pool."""
        async with self._lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        """Close every pool state while the lifecycle lock is held."""

        self._retire_pool_generation("database was closed")
        self._initialized = False
        self._closed = True
        self._lifecycle_revision += 1
        await self._close_quarantined_connections(
            "database close could not resolve every quarantined connection",
            rollback=True,
        )
        logger.info("Closed all database connections")

    async def _close_quarantined_connections(
        self,
        message: str,
        *,
        rollback: bool = False,
    ) -> None:
        """Attempt every quarantine close and fail if any remain unresolved."""

        first_error: Optional[BaseException] = None
        cancellation: Optional[asyncio.CancelledError] = None
        for connection in list(self._quarantined_connections):
            try:
                if connection in self._proven_closed_connections:
                    self._forget_quarantined_connection(connection)
                    continue
                if rollback:
                    try:
                        in_transaction = getattr(connection, "in_transaction", False)
                    except BaseException:
                        # A connection already closed by an old borrower has no
                        # active transaction property; close remains the proof.
                        in_transaction = False
                    if in_transaction:
                        try:
                            await connection.rollback()
                        except BaseException as rollback_error:
                            if (
                                isinstance(rollback_error, asyncio.CancelledError)
                                and cancellation is None
                            ):
                                cancellation = rollback_error
                            logger.debug(
                                "Quarantined connection rollback failed; "
                                f"attempting close: {rollback_error}"
                            )
                close_task = asyncio.create_task(_EXACT_AIOSQLITE_CLOSE(connection))
                cancellation = await self._await_owned_cleanup_task(
                    close_task,
                    cancellation,
                )
                self._proven_closed_connections.add(connection)
                self._forget_quarantined_connection(connection)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                if isinstance(error, asyncio.CancelledError) and cancellation is None:
                    cancellation = error
                logger.debug(f"Quarantined connection cleanup failed: {error}")
        if self._quarantined_connections:
            cleanup_error = SafetyDatabasePoolError(message)
            if first_error is not None:
                raise cleanup_error from first_error
            raise cleanup_error
        if cancellation is not None:
            raise cancellation

    async def _close_owned_connection(
        self,
        connection: aiosqlite.Connection,
        message: str,
        cancellation: Optional[asyncio.CancelledError] = None,
    ) -> None:
        """Prove one temporary connection closed before releasing ownership."""

        self._quarantine_pool_connection(connection)
        if connection in self._proven_closed_connections:
            # A concurrent lifecycle close can revoke a temporary initializer
            # or snapshot handle before its owner's finally block runs.
            # aiosqlite 0.19 rejects a second exact close, so consume only the
            # prior successful-close identity proof; a raised/unproven close
            # never enters this branch.
            self._forget_quarantined_connection(connection)
            if cancellation is not None:
                raise cancellation
            return
        close_task = asyncio.create_task(_EXACT_AIOSQLITE_CLOSE(connection))
        try:
            cancellation = await self._await_owned_cleanup_task(
                close_task,
                cancellation,
            )
        except BaseException as error:
            raise SafetyDatabasePoolError(message) from error
        self._proven_closed_connections.add(connection)
        self._forget_quarantined_connection(connection)
        if cancellation is not None:
            raise cancellation

    @staticmethod
    async def _await_owned_cleanup_task(
        task: asyncio.Task,
        cancellation: Optional[asyncio.CancelledError],
    ) -> Optional[asyncio.CancelledError]:
        """Await one cleanup task through any number of outer cancellations."""

        while True:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if task.cancelled():
                    raise SafetyDatabasePoolError(
                        "owned allocation-ledger close task was cancelled"
                    ) from error
                if cancellation is None:
                    cancellation = error
                if not task.done():
                    continue
                task.result()
            return cancellation

    async def health_check(self) -> bool:
        """Check if database connections are healthy."""
        if not self._initialized or self._closed:
            return False

        try:
            async with self.get_connection() as conn:
                await conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    async def ensure_connection(self):
        """Ensure database is connected and healthy."""
        async with self._ensure_lock:
            observed_revision = self._lifecycle_revision
            if not await self.health_check():
                logger.info("Database unhealthy, reinitializing...")
                async with self._lock:
                    # A concurrent explicit lifecycle operation wins. This
                    # prevents ensure_connection from resurrecting a database
                    # after close() advanced the revision.
                    if self._lifecycle_revision != observed_revision:
                        return
                    await self._close_locked()
                    await self._initialize_locked()

    def _latch_pool_recovery_failure(
        self,
        error: BaseException,
        expected_generation: _PoolGenerationState,
        operation: str,
    ) -> Tuple[Optional[_PoolRecoveryFailure], List[aiosqlite.Connection]]:
        """Poison one pool generation and wake every queued borrower.

        This method intentionally performs the state transition without an
        ``await``. Even cancellation while opening the replacement therefore
        cannot leave the pool initialized-but-empty or strand queue waiters.
        """

        # A close/reinitialize transition owns the newer generation. A stale
        # borrower must never poison or drain that replacement pool.
        if self._closed or expected_generation is not self._pool_generation:
            return None, []

        existing = self._pool_recovery_failure
        if existing is not None:
            return existing, []

        if isinstance(error, asyncio.CancelledError):
            reason = f"{operation} was cancelled"
        else:
            reason = f"{operation} failed ({type(error).__name__})"
        failure = _PoolRecoveryFailure(reason=reason)
        self._pool_recovery_failure = failure
        generation = expected_generation
        generation.failure = failure

        connections = list(self._pool)
        for connection in connections:
            self._quarantine_pool_connection(connection)
        self._pool.clear()
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Event.set() broadcasts to every current waiter. Unlike a queue
        # sentinel, no waiter can consume the only wakeup while recovery swaps
        # in a new pool generation.
        generation.failure_event.set()
        return failure, connections

    async def _poison_connection_pool(
        self,
        error: BaseException,
        expected_generation: _PoolGenerationState,
        operation: str = "replacement connection opening",
    ) -> None:
        """Latch an unusable pool and best-effort close its connections."""

        self._latch_pool_recovery_failure(
            error,
            expected_generation,
            operation,
        )
        if self._quarantined_connections:
            await self._close_quarantined_connections(
                "poisoned pool cleanup could not close every connection"
            )

    def _detach_pool_connection(self, connection: aiosqlite.Connection) -> None:
        """Synchronously remove a connection before fallible finalization."""

        try:
            self._pool.remove(connection)
        except ValueError:
            pass
        self._release_connection_lease(connection)
        self._quarantine_pool_connection(connection)

    def _lease_pool_connection(self, connection: aiosqlite.Connection) -> None:
        if self._is_connection_leased(connection):
            raise SafetyDatabasePoolError("pooled connection is already leased")
        self._leased_connections.append(connection)

    def _release_connection_lease(self, connection: aiosqlite.Connection) -> None:
        self._leased_connections = [
            item for item in self._leased_connections if item is not connection
        ]

    def _is_connection_leased(self, connection: aiosqlite.Connection) -> bool:
        return any(item is connection for item in self._leased_connections)

    def _quarantine_pool_connection(self, connection: aiosqlite.Connection) -> None:
        if connection in self._proven_closed_connections:
            return
        if not any(item is connection for item in self._quarantined_connections):
            self._quarantined_connections.append(connection)

    def _forget_quarantined_connection(self, connection: aiosqlite.Connection) -> None:
        self._quarantined_connections = [
            item for item in self._quarantined_connections if item is not connection
        ]

    @staticmethod
    def _pool_failure_error(failure: _PoolRecoveryFailure) -> SafetyDatabasePoolError:
        return SafetyDatabasePoolError(
            "allocation ledger connection pool is poisoned: "
            f"{failure.reason}; call ensure_connection() before retrying"
        )

    async def _wait_for_pool_connection(
        self,
        borrowed_queue: asyncio.Queue,
        generation: _PoolGenerationState,
    ) -> aiosqlite.Connection:
        """Wait for a connection or broadcast poison without losing a slot."""

        connection_wait = asyncio.create_task(borrowed_queue.get())
        failure_wait = asyncio.create_task(generation.failure_event.wait())
        queue_result_handled = False
        connection_to_return: Optional[aiosqlite.Connection] = None
        cancellation: Optional[asyncio.CancelledError] = None
        try:
            done, _ = await asyncio.wait(
                (connection_wait, failure_wait),
                timeout=10.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise asyncio.TimeoutError()
            if failure_wait in done:
                failure = generation.failure or _PoolRecoveryFailure(
                    "pool generation failed without a recovery reason"
                )
                if connection_wait in done and not connection_wait.cancelled():
                    connection = connection_wait.result()
                    queue_result_handled = True
                    self._quarantine_pool_connection(connection)
                    await self._poison_connection_pool(
                        self._pool_failure_error(failure),
                        generation,
                        "orphaned checkout close",
                    )
                raise self._pool_failure_error(failure)

            connection = connection_wait.result()
            queue_result_handled = True
            connection_to_return = connection
            return connection
        except asyncio.CancelledError as error:
            # Preserve the first cancellation, but do not let it or any later
            # cancellation interrupt ownership recovery for a queue result.
            cancellation = error
        finally:
            cleanup_task = asyncio.create_task(
                self._cleanup_pool_waiters(
                    connection_wait,
                    failure_wait,
                    borrowed_queue,
                    generation,
                    queue_result_handled,
                )
            )
            cancellation = await self._await_owned_cleanup_task(
                cleanup_task,
                cancellation,
            )
            if cancellation is not None and connection_to_return is not None:
                # A successful dequeue is not transferred to the caller until
                # this finally block completes.  If cancellation wins during
                # waiter cleanup, restore/retire that pending result first.
                salvage_task = asyncio.create_task(
                    self._cleanup_pool_waiters(
                        connection_wait,
                        failure_wait,
                        borrowed_queue,
                        generation,
                        False,
                    )
                )
                cancellation = await self._await_owned_cleanup_task(
                    salvage_task,
                    cancellation,
                )
            if cancellation is not None:
                raise cancellation

    async def _cleanup_pool_waiters(
        self,
        connection_wait: asyncio.Task,
        failure_wait: asyncio.Task,
        borrowed_queue: asyncio.Queue,
        generation: _PoolGenerationState,
        queue_result_handled: bool,
    ) -> None:
        """Drain waiter tasks and recover any connection they already claimed."""

        try:
            for waiter in (connection_wait, failure_wait):
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(connection_wait, failure_wait, return_exceptions=True)

            # Cancellation or timeout can race with queue delivery. Return the
            # claimed slot only to its exact healthy generation; otherwise
            # close it so a rebuilt pool cannot inherit a stale connection.
            if (
                not queue_result_handled
                and connection_wait.done()
                and not connection_wait.cancelled()
            ):
                try:
                    orphaned_connection = connection_wait.result()
                except BaseException:
                    pass
                else:
                    if (
                        generation is self._pool_generation
                        and generation.failure is None
                        and borrowed_queue is self._available
                        and not self._closed
                    ):
                        borrowed_queue.put_nowait(orphaned_connection)
                    else:
                        self._quarantine_pool_connection(orphaned_connection)
                        await self._poison_connection_pool(
                            SafetyDatabasePoolError("orphaned pool wait connection"),
                            generation,
                            "orphaned pool wait close",
                        )
        except asyncio.CancelledError as error:
            # This task is owned by _wait_for_pool_connection and is shielded
            # from borrower cancellation.  Internal cancellation would make
            # queue ownership unknowable, so surface it as a safety failure.
            raise SafetyDatabasePoolError(
                "allocation ledger waiter cleanup task was cancelled"
            ) from error

    async def _open_identity_bound_pool_connection(self) -> aiosqlite.Connection:
        """Open one replacement connection bound to the configured ledger inode."""

        binding: Optional[SQLitePathBinding] = None
        connection: Optional[aiosqlite.Connection] = None
        try:
            binding = SQLitePathBinding.open_for_initialization(
                self.db_path,
                create=False,
            )
            expected = self._expected_database_file_identity
            if expected is None or (binding.device, binding.inode) != expected:
                raise SafetyAllocationSnapshotError(
                    "configured allocation ledger was replaced before pool recovery"
                )
            binding.assert_path_identity()
            connection = await aiosqlite.connect(self.db_path)
            self._quarantine_pool_connection(connection)
            pool_binding = binding.bind_sqlite_connection(
                await self._sqlite_descriptor_identity(connection)
            )
            pool_binding.assert_connection_identity(
                await self._sqlite_descriptor_identity(connection)
            )
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA busy_timeout=5000")
            pool_binding.assert_connection_identity(
                await self._sqlite_descriptor_identity(connection)
            )
            binding.assert_path_identity()
            return connection
        except BaseException:
            # The replacement caller owns exact cleanup through pool poison.
            raise
        finally:
            if binding is not None:
                binding.close()

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Database is closed")

        if not self._initialized:
            await self.initialize()

        # Keep the exact queue generation. Recovery replaces ``self._available``
        # so a late return can never enter the rebuilt pool.
        borrowed_queue = self._available
        generation = self._pool_generation
        lifecycle_revision = self._lifecycle_revision
        if self._closed or not self._initialized:
            raise RuntimeError("Database is closed")
        failure = self._pool_recovery_failure
        if failure is not None:
            raise self._pool_failure_error(failure)

        conn = await self._wait_for_pool_connection(borrowed_queue, generation)
        self._lease_pool_connection(conn)
        return_connection: Optional[aiosqlite.Connection] = conn
        try:
            failure = self._pool_recovery_failure
            if (
                failure is not None
                or self._closed
                or lifecycle_revision != self._lifecycle_revision
                or borrowed_queue is not self._available
                or generation is not self._pool_generation
            ):
                return_connection = None
                self._release_connection_lease(conn)
                if failure is None:
                    failure = _PoolRecoveryFailure("pool generation changed during checkout")
                raise self._pool_failure_error(failure)
            # Test connection before use
            await conn.execute("SELECT 1")
            yield conn
        except SafetyDatabasePoolError:
            raise
        except Exception as e:
            if (
                self._closed
                or self._pool_recovery_failure is not None
                or lifecycle_revision != self._lifecycle_revision
                or borrowed_queue is not self._available
                or generation is not self._pool_generation
            ):
                return_connection = None
                self._release_connection_lease(conn)
                raise
            logger.warning(f"Connection error: {e}")
            return_connection = None
            # Detach before close: cancellation at the close await must not
            # leave this connection counted in an empty pool generation.
            self._detach_pool_connection(conn)
            try:
                await self._close_quarantined_connections(
                    "bad connection cleanup could not close every connection"
                )
            except BaseException as cleanup_error:
                await self._poison_connection_pool(
                    cleanup_error,
                    generation,
                    "bad connection cleanup",
                )
                raise
            failure = self._pool_recovery_failure
            if (
                failure is not None
                or self._closed
                or lifecycle_revision != self._lifecycle_revision
                or borrowed_queue is not self._available
                or generation is not self._pool_generation
            ):
                raise
            try:
                new_conn = await self._open_identity_bound_pool_connection()
            except BaseException as replace_error:
                logger.error(f"Failed to replace bad connection: {replace_error}")
                poison_generation = generation
                if generation is not self._pool_generation and self._quarantined_connections:
                    poison_generation = self._pool_generation
                await self._poison_connection_pool(replace_error, poison_generation)
                if not isinstance(replace_error, Exception):
                    raise
                raise e from replace_error
            if (
                self._closed
                or lifecycle_revision != self._lifecycle_revision
                or borrowed_queue is not self._available
                or generation is not self._pool_generation
            ):
                # An intervening close may already have proved and forgotten a
                # close while this old task still owned the replacement. Track
                # it again before the stale owner performs its final close.
                self._quarantine_pool_connection(new_conn)
                await self._poison_connection_pool(
                    SafetyDatabasePoolError("stale replacement generation"),
                    self._pool_generation,
                    "stale replacement close",
                )
                raise
            self._pool.append(new_conn)
            self._forget_quarantined_connection(new_conn)
            return_connection = new_conn
            raise
        except BaseException as fatal_error:
            # Cancellation or interpreter-level abort while validating/using
            # a checked-out connection makes its state ambiguous. Fail the
            # exact generation before the exception can propagate.
            return_connection = None
            if (
                self._closed
                or self._pool_recovery_failure is not None
                or lifecycle_revision != self._lifecycle_revision
                or borrowed_queue is not self._available
                or generation is not self._pool_generation
            ):
                self._release_connection_lease(conn)
                raise
            self._detach_pool_connection(conn)
            await self._poison_connection_pool(
                fatal_error,
                self._pool_generation,
                "checked-out connection use",
            )
            raise
        finally:
            checkout_retired = (
                self._closed
                or self._pool_recovery_failure is not None
                or lifecycle_revision != self._lifecycle_revision
                or borrowed_queue is not self._available
                or generation is not self._pool_generation
            )
            if checkout_retired and return_connection is not None:
                self._release_connection_lease(return_connection)
                return_connection = None

            # Ensure no transaction remains open on pooled connections
            try:
                if return_connection is not None and getattr(
                    return_connection,
                    "in_transaction",
                    False,
                ):
                    await return_connection.rollback()
            except BaseException as rollback_error:
                failed_connection = return_connection
                return_connection = None
                if failed_connection is not None:
                    if (
                        self._closed
                        or lifecycle_revision != self._lifecycle_revision
                        or borrowed_queue is not self._available
                        or generation is not self._pool_generation
                    ):
                        self._release_connection_lease(failed_connection)
                    else:
                        self._detach_pool_connection(failed_connection)
                        await self._poison_connection_pool(
                            rollback_error,
                            self._pool_generation,
                            "pooled connection rollback",
                        )
                        raise

            # Return exactly one usable connection. Never enqueue a closed
            # original alongside its replacement.
            if return_connection is not None:
                if (
                    not self._closed
                    and self._pool_recovery_failure is None
                    and borrowed_queue is self._available
                    and generation is self._pool_generation
                    and lifecycle_revision == self._lifecycle_revision
                ):
                    try:
                        self._release_connection_lease(return_connection)
                        borrowed_queue.put_nowait(return_connection)
                    except BaseException as return_error:
                        self._detach_pool_connection(return_connection)
                        await self._poison_connection_pool(
                            return_error,
                            generation,
                            "pooled connection return",
                        )
                        raise
                else:
                    self._detach_pool_connection(return_connection)
                    await self._poison_connection_pool(
                        SafetyDatabasePoolError("stale returned connection"),
                        self._pool_generation,
                        "stale connection close",
                    )

    async def _init_database(self, guardian: SQLitePathBinding) -> None:
        """Create tables if they don't exist.

        Tables with portfolio_id (user-scoped):
            positions, trades, account, equity_history, signals
        Tables without portfolio_id (global/shared):
            ticks, features, market_data
        """
        conn: Optional[aiosqlite.Connection] = None
        cancellation: Optional[asyncio.CancelledError] = None
        try:
            conn = await aiosqlite.connect(self.db_path)
            self._quarantine_pool_connection(conn)
            connection_binding = guardian.bind_sqlite_connection(
                await self._sqlite_descriptor_identity(conn)
            )
            connection_binding.assert_connection_identity(
                await self._sqlite_descriptor_identity(conn)
            )
            # Set WAL and busy timeout on the initializer connection too, to avoid rollback journal usage
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")

            # Portfolios table (multi-portfolio support)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    starting_cash REAL NOT NULL DEFAULT 100000,
                    symbols TEXT NOT NULL DEFAULT '',
                    active INTEGER NOT NULL DEFAULT 1,
                    max_position_pct REAL,
                    max_daily_loss_pct REAL,
                    max_open_positions INTEGER,
                    stop_loss_pct REAL,
                    trailing_stop_pct REAL,
                    use_trailing_stop INTEGER,
                    enabled_strategies TEXT,
                    min_confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert default portfolio if not exists
            await conn.execute("""
                INSERT OR IGNORE INTO portfolios (id, name, starting_cash)
                VALUES ('default', 'Default Portfolio', 100000)
            """)

            # Schema migrations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Positions table (portfolio-scoped)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    market_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(portfolio_id, symbol)
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_portfolio
                ON positions (portfolio_id)
            """)

            # Tick data table (global - shared across portfolios)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    bid REAL,
                    ask REAL,
                    last REAL,
                    bid_size INTEGER,
                    ask_size INTEGER,
                    last_size INTEGER,
                    volume INTEGER,
                    PRIMARY KEY (timestamp, symbol)
                )
            """)

            # Create index for efficient queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticks_symbol
                ON ticks (symbol, timestamp DESC)
            """)

            # Features table (global - shared across portfolios)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    rsi REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    atr REAL,
                    vwap REAL,
                    obv REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    sma_200 REAL,
                    volume_ratio REAL,
                    spread_bps REAL,
                    trend_strength REAL,
                    mean_reversion_signal REAL,
                    breakout_signal REAL,
                    PRIMARY KEY (timestamp, symbol)
                )
            """)

            # Create index for efficient queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_symbol
                ON features (symbol, timestamp DESC)
            """)

            # Trades table (portfolio-scoped)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    notional REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    pnl REAL DEFAULT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_portfolio
                ON trades (portfolio_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_portfolio_symbol
                ON trades (portfolio_id, symbol, timestamp DESC)
            """)

            # Migrations for existing tables
            migrations = [
                "ALTER TABLE trades ADD COLUMN pnl REAL DEFAULT NULL",
                "ALTER TABLE trades ADD COLUMN notional REAL DEFAULT 0",
                "ALTER TABLE trades ADD COLUMN portfolio_id TEXT DEFAULT 'default'",
                "ALTER TABLE positions ADD COLUMN portfolio_id TEXT DEFAULT 'default'",
            ]
            for migration in migrations:
                try:
                    await conn.execute(migration)
                except Exception:
                    pass  # Column already exists

            # Account table (portfolio-scoped, keyed by portfolio_id)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS account (
                    portfolio_id TEXT PRIMARY KEY DEFAULT 'default',
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Equity history table for portfolio value over time (portfolio-scoped)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    date TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL DEFAULT 0,
                    positions_value REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(portfolio_id, date)
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_equity_history_portfolio
                ON equity_history (portfolio_id, date)
            """)

            # Market data table (global - shared across portfolios)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER,
                    timestamp DATETIME NOT NULL,
                    UNIQUE(symbol, timestamp)
                )
            """)

            # Strategy signals table (portfolio-scoped)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_portfolio
                ON signals (portfolio_id)
            """)

            # Insert default account if not exists
            await conn.execute("""
                INSERT OR IGNORE INTO account (portfolio_id, cash, equity)
                VALUES ('default', 100000, 100000)
            """)

            await conn.commit()
            connection_binding.assert_connection_identity(
                await self._sqlite_descriptor_identity(conn)
            )
        except asyncio.CancelledError as error:
            cancellation = error
        finally:
            if conn is not None:
                await self._close_owned_connection(
                    conn,
                    "initializer connection could not be closed",
                    cancellation,
                )
        if cancellation is not None:
            raise cancellation

    async def update_position(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float,
        market_price: Optional[float] = None,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Update or insert a position asynchronously."""
        # Validate inputs
        try:
            symbol = DatabaseValidator.validate_symbol(symbol)
            quantity = DatabaseValidator.validate_quantity(quantity, allow_negative=True)
            # Skip price validation when closing position (quantity=0)
            if quantity != 0:
                avg_cost = DatabaseValidator.validate_price(avg_cost, field_name="avg_cost")
                if market_price is not None:
                    market_price = DatabaseValidator.validate_price(
                        market_price, field_name="market_price"
                    )
            portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        except ValidationError as e:
            logger.error(f"Position update validation failed: {e}")
            raise

        async with self.get_connection() as conn:
            if quantity == 0:
                # Close position - delete from database
                await conn.execute(
                    "DELETE FROM positions WHERE portfolio_id = ? AND symbol = ?",
                    (portfolio_id, symbol),
                )
                logger.info(f"Closed position for {symbol} (portfolio={portfolio_id})")
            else:
                # Update or insert position
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (portfolio_id, symbol, quantity, avg_cost, market_price)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (portfolio_id, symbol, quantity, avg_cost, market_price),
                )

            await conn.commit()
            logger.debug(
                f"Updated position: {symbol} qty={quantity} avg={avg_cost} (portfolio={portfolio_id})"
            )

    async def _calculate_fifo_pnl(
        self,
        conn,
        symbol: str,
        sell_quantity: int,
        sell_price: float,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> float:
        """
        Calculate realized P&L for a SELL trade using weighted average cost.

        Note: Despite the name, this uses weighted average cost basis (not strict
        FIFO lot matching) for simplicity. The average cost is calculated across
        all BUY trades for the symbol within this portfolio.

        Args:
            conn: Database connection
            symbol: Stock symbol
            sell_quantity: Number of shares being sold
            sell_price: Price per share for the sell
            portfolio_id: Portfolio to scope the calculation to

        Returns:
            Realized P&L (positive = profit, negative = loss)
        """
        # Get all BUY trades for this symbol in this portfolio, ordered by timestamp (FIFO)
        cursor = await conn.execute(
            """
            SELECT id, quantity, price FROM trades
            WHERE portfolio_id = ? AND symbol = ? AND side = 'BUY'
            ORDER BY timestamp ASC
            """,
            (portfolio_id, symbol),
        )
        buy_trades = await cursor.fetchall()

        if not buy_trades:
            # No BUY trades found - use position's avg_cost if available
            cursor = await conn.execute(
                "SELECT avg_cost FROM positions WHERE portfolio_id = ? AND symbol = ?",
                (portfolio_id, symbol),
            )
            pos = await cursor.fetchone()
            if pos and pos[0]:
                avg_cost = pos[0]
                return (sell_price - avg_cost) * sell_quantity
            # No cost basis - return 0
            logger.warning(
                f"No cost basis found for {symbol} SELL trade (portfolio={portfolio_id})"
            )
            return 0.0

        # Calculate weighted average cost from BUY trades
        # For simplicity, use weighted average rather than strict FIFO lot matching
        total_shares = sum(t[1] for t in buy_trades)
        total_cost = sum(t[1] * t[2] for t in buy_trades)

        if total_shares > 0:
            avg_cost = total_cost / total_shares
            realized_pnl = (sell_price - avg_cost) * sell_quantity
            logger.debug(
                f"FIFO P&L for {symbol}: sell ${sell_price:.2f} - avg cost ${avg_cost:.2f} "
                f"x {sell_quantity} = ${realized_pnl:.2f}"
            )
            return realized_pnl

        return 0.0

    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        slippage: float = 0.0,
        commission: float = 0.0,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Record a trade asynchronously with P&L calculation for SELL trades."""
        # Validate inputs
        try:
            symbol = DatabaseValidator.validate_symbol(symbol)
            side = DatabaseValidator.validate_order_side(side)
            quantity = DatabaseValidator.validate_quantity(quantity)
            price = DatabaseValidator.validate_price(price)
            slippage = DatabaseValidator._validate_numeric(
                slippage, "slippage", min_val=0, max_val=1000
            )
            commission = DatabaseValidator._validate_numeric(
                commission, "commission", min_val=0, max_val=1000
            )
            portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        except ValidationError as e:
            logger.error(f"Trade record validation failed: {e}")
            raise

        async with self.get_connection() as conn:
            # Calculate P&L for SELL trades (realized profit/loss)
            pnl = None
            if side.upper() in ("SELL", "BUY_TO_COVER"):
                pnl = await self._calculate_fifo_pnl(conn, symbol, quantity, price, portfolio_id)

            # Ensure consistent float type for SQLite storage
            notional = float(quantity) * float(price)
            await conn.execute(
                """
                INSERT INTO trades (portfolio_id, symbol, side, quantity, price, notional, slippage, commission, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (portfolio_id, symbol, side, quantity, price, notional, slippage, commission, pnl),
            )
            await conn.commit()

            if pnl is not None:
                logger.info(
                    f"Recorded trade: {side} {quantity} {symbol} @ {price} "
                    f"(P&L: ${pnl:,.2f}, portfolio={portfolio_id})"
                )
            else:
                logger.info(
                    f"Recorded trade: {side} {quantity} {symbol} @ {price} (portfolio={portfolio_id})"
                )

    async def update_account(
        self,
        cash: float,
        equity: float,
        daily_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Update account values asynchronously."""
        # Validate inputs
        try:
            portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
            account_data = {
                "cash": cash,
                "equity": equity,
                "daily_pnl": daily_pnl,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
            }
            validated_data = DatabaseValidator.validate_account_data(account_data)
            cash = validated_data.get("cash", cash)
            equity = validated_data.get("equity", equity)
            daily_pnl = validated_data.get("daily_pnl", daily_pnl)
            realized_pnl = validated_data.get("realized_pnl", realized_pnl)
            unrealized_pnl = validated_data.get("unrealized_pnl", unrealized_pnl)
        except ValidationError as e:
            logger.error(f"Account update validation failed: {e}")
            raise

        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO account
                    (portfolio_id, cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (portfolio_id, cash, equity, daily_pnl, realized_pnl, unrealized_pnl),
            )
            await conn.commit()
            logger.debug(
                f"Updated account: cash={cash:.2f} equity={equity:.2f} (portfolio={portfolio_id})"
            )

    async def record_signal(
        self,
        symbol: str,
        strategy: str,
        signal_type: str,
        strength: float = 0.0,
        metadata: str = "",
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Record a strategy signal asynchronously."""
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        symbol = DatabaseValidator.validate_symbol(symbol)
        if strategy is not None:
            strategy = DatabaseValidator._validate_string(strategy, "strategy", max_length=64)
        if signal_type is not None:
            signal_type = DatabaseValidator._validate_string(
                signal_type, "signal_type", max_length=32
            )
        if metadata is not None and metadata != "":
            metadata = DatabaseValidator._validate_string(
                metadata, "metadata", max_length=4096, allow_empty=True
            )
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO signals (portfolio_id, symbol, strategy, signal_type, strength, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (portfolio_id, symbol, strategy, signal_type, strength, metadata),
            )
            await conn.commit()
            logger.debug(
                f"Recorded signal: {strategy} {signal_type} for {symbol} (portfolio={portfolio_id})"
            )

    async def store_market_data(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """Store market data bar asynchronously."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO market_data
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (symbol, timestamp, open_price, high, low, close, volume),
            )
            await conn.commit()

    async def batch_store_market_data(self, data: List[Dict]) -> None:
        """Store multiple market data bars in a batch for efficiency.

        D-16: validate dict keys before executemany so a caller-supplied row
        missing a bind parameter raises a clear ValueError instead of an
        opaque sqlite ProgrammingError (or, worse, silently binding NULL).
        """
        if not data:
            return

        required_keys = {
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(
                    f"batch_store_market_data row[{idx}] must be a dict, "
                    f"got {type(row).__name__}"
                )
            missing = required_keys - row.keys()
            if missing:
                raise ValueError(
                    f"batch_store_market_data row[{idx}] missing keys: {sorted(missing)}"
                )

        async with self.get_connection() as conn:
            await conn.executemany(
                """
                INSERT OR REPLACE INTO market_data
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
            """,
                data,
            )
            await conn.commit()
            logger.debug(f"Stored {len(data)} market data bars")

    async def get_position(
        self, symbol: str, portfolio_id: str = DEFAULT_PORTFOLIO_ID
    ) -> Optional[Dict]:
        """Get position for a specific symbol in a portfolio."""
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT symbol, quantity, avg_cost, market_price
                FROM positions
                WHERE portfolio_id = ? AND symbol = ? AND quantity != 0
            """,
                (portfolio_id, symbol),
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "symbol": row[0],
                    "quantity": row[1],
                    "avg_cost": row[2],
                    "market_price": row[3],
                }
            return None

    async def get_positions(self, portfolio_id: str = DEFAULT_PORTFOLIO_ID) -> List[Dict]:
        """Get all current positions for a portfolio."""
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT symbol, quantity, avg_cost, market_price
                FROM positions
                WHERE portfolio_id = ? AND quantity != 0
            """,
                (portfolio_id,),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "symbol": row[0],
                    "quantity": row[1],
                    "avg_cost": row[2],
                    "market_price": row[3],
                }
                for row in rows
            ]

    async def get_all_positions(self) -> List[Dict]:
        """Get all current positions across ALL portfolios."""
        async with self.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT portfolio_id, symbol, quantity, avg_cost, market_price
                FROM positions
                WHERE quantity != 0
            """)
            rows = await cursor.fetchall()
            return [
                {
                    "portfolio_id": row[0],
                    "symbol": row[1],
                    "quantity": row[2],
                    "avg_cost": row[3],
                    "market_price": row[4],
                }
                for row in rows
            ]

    def _expected_safety_database(
        self,
        *,
        runtime_contract: Optional[object],
    ) -> Tuple[Path, str]:
        """Resolve the ledger only from the exact validated runtime contract."""

        # Import lazily so the shared SQLite identity helper remains a
        # stdlib-only primitive and database module import stays inert.
        from robo_trader.config import RuntimeContract

        if type(runtime_contract) is not RuntimeContract:
            raise SafetyAllocationSnapshotError(
                "allocation snapshot requires an exact RuntimeContract"
            )
        expected_path = lexical_path_preserving_leaf(runtime_contract.database_path)
        identity = runtime_contract.database_identity
        namespace = runtime_contract.state_namespace
        if namespace not in {"paper", "backtest"} or namespace != runtime_contract.execution_mode:
            raise SafetyAllocationSnapshotError(
                "runtime contract database namespace is not a validated execution mode"
            )
        digest = hashlib.sha256(
            str(expected_path.resolve(strict=False)).encode("utf-8")
        ).hexdigest()[:12]
        if identity != f"{namespace}:{digest}":
            raise SafetyAllocationSnapshotError(
                "runtime contract database identity is inconsistent with its path"
            )

        if expected_path != self.db_path:
            raise SafetyAllocationSnapshotError(
                "configured allocation ledger path differs from the expected database path"
            )
        if (
            not isinstance(identity, str)
            or not identity
            or identity != identity.strip()
            or len(identity) > 256
            or any(ord(character) < 32 for character in identity)
        ):
            raise SafetyAllocationSnapshotError("expected database identity is malformed")
        return expected_path, identity

    @staticmethod
    async def _sqlite_descriptor_identity(
        connection: aiosqlite.Connection,
    ) -> SQLiteDescriptorIdentity:
        """Inspect the raw connection on its owning aiosqlite worker thread."""

        try:
            return await connection._execute(  # type: ignore[attr-defined]
                sqlite_connection_file_identity,
                connection._conn,  # type: ignore[attr-defined]
            )
        except SQLiteIdentityError:
            raise
        except Exception as exc:
            raise SQLiteIdentityError(
                "SQLite connection descriptor identity cannot be inspected"
            ) from exc

    async def get_safety_allocation_snapshot(
        self,
        symbol: str,
        *,
        runtime_contract: Optional[object] = None,
    ) -> SafetyAllocationSnapshot:
        """Read authoritative allocation truth for ``symbol`` without mutation.

        This boundary intentionally does not use ``PortfolioScopedDB``.  It
        derives the portfolio universe from the underlying ``portfolios`` table
        and reads all matching ``positions`` rows in one SELECT inside one
        read-only SQLite transaction.  Missing position rows are represented as
        exact zero allocations; orphaned, duplicate, future-dated, or malformed
        rows fail closed instead of being silently omitted.

        The returned snapshot binds the caller's validated runtime identity to
        the exact configured lexical path and the common device/inode proven by
        an ``O_NOFOLLOW`` guardian plus SQLite's own unix-VFS descriptor. The
        binding is checked before and after the one read-only transaction.

        SQLite read transactions do not mutate user rows, the main database,
        or WAL contents. SQLite may update shared-memory lock bytes in ``-shm``
        while coordinating a WAL reader; callers must not interpret literal
        filesystem immutability as part of this evidence contract.

        ``positions.timestamp`` is validated as UTC, non-future provenance but
        is not required to be recent: the existing schema records mutation
        time, not a heartbeat.  The current schema also has no broker conId, so
        this symbol-only allocation snapshot cannot establish broker identity.
        Runtime must bind it to independently qualified contract/conId evidence.
        """
        try:
            symbol = DatabaseValidator.validate_symbol(symbol)
        except ValidationError as exc:
            raise SafetyAllocationSnapshotError("snapshot symbol is invalid") from exc

        _, database_identity = self._expected_safety_database(
            runtime_contract=runtime_contract,
        )

        query = """
            SELECT
                definitions.id AS registry_portfolio_id,
                positions.id AS position_id,
                positions.portfolio_id AS stored_portfolio_id,
                positions.symbol AS stored_symbol,
                positions.quantity AS stored_quantity,
                typeof(positions.quantity) AS quantity_storage_type,
                positions.timestamp AS stored_timestamp,
                typeof(positions.timestamp) AS timestamp_storage_type
            FROM portfolios AS definitions
            LEFT JOIN positions
              ON lower(trim(positions.portfolio_id)) = lower(trim(definitions.id))
             AND upper(trim(positions.symbol)) = ?

            UNION ALL

            SELECT
                NULL AS registry_portfolio_id,
                positions.id AS position_id,
                positions.portfolio_id AS stored_portfolio_id,
                positions.symbol AS stored_symbol,
                positions.quantity AS stored_quantity,
                typeof(positions.quantity) AS quantity_storage_type,
                positions.timestamp AS stored_timestamp,
                typeof(positions.timestamp) AS timestamp_storage_type
            FROM positions
            WHERE upper(trim(positions.symbol)) = ?
              AND NOT EXISTS (
                    SELECT 1
                    FROM portfolios AS definitions
                    WHERE lower(trim(definitions.id)) =
                          lower(trim(positions.portfolio_id))
              )

            ORDER BY registry_portfolio_id, stored_portfolio_id, position_id
        """

        binding: Optional[SQLitePathBinding] = None
        try:
            binding = SQLitePathBinding.open_readonly(self.db_path)
            expected_file_identity = self._expected_database_file_identity
            if expected_file_identity is None:
                raise SafetyAllocationSnapshotError(
                    "allocation ledger has no previously bound file identity"
                )
            if (binding.device, binding.inode) != expected_file_identity:
                raise SafetyAllocationSnapshotError(
                    "configured allocation ledger was replaced before snapshot collection"
                )
            database_uri = binding.path.as_uri() + "?mode=ro"
            conn: Optional[aiosqlite.Connection] = None
            connection_cancellation: Optional[asyncio.CancelledError] = None
            try:
                conn = await aiosqlite.connect(database_uri, uri=True)
                self._quarantine_pool_connection(conn)
                connection_identity = await self._sqlite_descriptor_identity(conn)
                binding = binding.bind_sqlite_connection(connection_identity)

                await conn.execute("PRAGMA query_only = ON")
                query_only = await conn.execute("PRAGMA query_only")
                if await query_only.fetchone() != (1,):
                    raise SafetyAllocationSnapshotError(
                        "allocation ledger query-only mode could not be proven"
                    )
                binding.assert_connection_identity(await self._sqlite_descriptor_identity(conn))

                await conn.execute("BEGIN")
                try:
                    # Bind freshness conservatively before the SELECT establishes
                    # its SQLite snapshot. If the query waits, the evidence can
                    # only appear older, never newer than the rows it observed.
                    observed_at = datetime.now(timezone.utc)
                    cursor = await conn.execute(query, (symbol, symbol))
                    rows = await cursor.fetchall()
                    row_timestamp_upper_bound = datetime.now(timezone.utc)
                    binding.assert_connection_identity(await self._sqlite_descriptor_identity(conn))
                finally:
                    # Explicitly end the only read transaction. The URI and
                    # query_only proof prevent mutation even after refactoring.
                    await conn.rollback()

                binding.assert_connection_identity(await self._sqlite_descriptor_identity(conn))
                snapshot = self._build_safety_allocation_snapshot(
                    symbol=symbol,
                    rows=rows,
                    observed_at=observed_at,
                    row_timestamp_upper_bound=row_timestamp_upper_bound,
                    database_identity=database_identity,
                    database_device=binding.device,
                    database_inode=binding.inode,
                )
                # Keep the connection and guardian alive through validation so
                # replacement after the read cannot escape the final proof.
                binding.assert_connection_identity(await self._sqlite_descriptor_identity(conn))
                return snapshot
            except asyncio.CancelledError as error:
                connection_cancellation = error
            finally:
                if conn is not None:
                    await self._close_owned_connection(
                        conn,
                        "allocation snapshot connection could not be closed",
                        connection_cancellation,
                    )
            if connection_cancellation is not None:
                raise connection_cancellation
        except SafetyAllocationSnapshotError:
            raise
        except (SQLiteIdentityError, OSError, aiosqlite.Error) as exc:
            raise SafetyAllocationSnapshotError(
                "allocation snapshot database identity cannot be proven"
            ) from exc
        finally:
            if binding is not None:
                try:
                    binding.close()
                except SQLiteIdentityError as exc:
                    raise SafetyAllocationSnapshotError(
                        "allocation snapshot guardian descriptor could not be closed"
                    ) from exc

    def _build_safety_allocation_snapshot(
        self,
        *,
        symbol: str,
        rows: List[Tuple],
        observed_at: datetime,
        row_timestamp_upper_bound: datetime,
        database_identity: str,
        database_device: int,
        database_inode: int,
    ) -> SafetyAllocationSnapshot:
        """Validate rows and construct immutable evidence while binding is held."""

        if not rows:
            raise SafetyAllocationSnapshotError(
                "allocation snapshot is incomplete: no portfolio definitions"
            )

        allocations: List[SafetyPortfolioAllocation] = []
        seen_portfolio_ids = set()
        integer_quantities: List[int] = []

        for row in rows:
            (
                registry_portfolio_id,
                position_id,
                stored_portfolio_id,
                stored_symbol,
                stored_quantity,
                quantity_storage_type,
                stored_timestamp,
                timestamp_storage_type,
            ) = row

            raw_portfolio_id = (
                stored_portfolio_id if position_id is not None else registry_portfolio_id
            )
            try:
                portfolio_id = DatabaseValidator.validate_portfolio_id(raw_portfolio_id)
            except ValidationError as exc:
                raise SafetyAllocationSnapshotError(
                    "allocation snapshot contains an invalid portfolio_id"
                ) from exc

            try:
                registry_id = DatabaseValidator.validate_portfolio_id(registry_portfolio_id)
            except ValidationError as exc:
                raise SafetyAllocationSnapshotError(
                    f"allocation for portfolio {portfolio_id!r} is orphaned"
                ) from exc

            if registry_id != portfolio_id:
                raise SafetyAllocationSnapshotError(
                    f"allocation for portfolio {portfolio_id!r} is orphaned"
                )
            if portfolio_id in seen_portfolio_ids:
                raise SafetyAllocationSnapshotError(
                    f"duplicate allocation for portfolio {portfolio_id!r}"
                )
            seen_portfolio_ids.add(portfolio_id)

            if position_id is None:
                quantity_int = 0
                updated_at = None
            else:
                try:
                    stored_symbol_normalized = DatabaseValidator.validate_symbol(stored_symbol)
                except ValidationError as exc:
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} has an invalid symbol"
                    ) from exc
                if stored_symbol_normalized != symbol or stored_symbol != stored_symbol_normalized:
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} has a "
                        "noncanonical or mismatched symbol"
                    )

                if quantity_storage_type != "integer" or type(stored_quantity) is not int:
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} is not stored " "as an integer"
                    )
                try:
                    quantity_int = DatabaseValidator.validate_quantity(
                        stored_quantity, allow_negative=True
                    )
                except ValidationError as exc:
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} has an " "invalid quantity"
                    ) from exc

                if (
                    timestamp_storage_type != "text"
                    or not isinstance(stored_timestamp, str)
                    or len(stored_timestamp) < 19
                ):
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} has an " "invalid timestamp"
                    )
                try:
                    parsed_timestamp = datetime.fromisoformat(
                        stored_timestamp.replace("Z", "+00:00")
                    )
                except ValueError as exc:
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} has an " "invalid timestamp"
                    ) from exc
                if parsed_timestamp.tzinfo is None:
                    updated_at = parsed_timestamp.replace(tzinfo=timezone.utc)
                elif parsed_timestamp.utcoffset() == timedelta(0):
                    updated_at = parsed_timestamp.astimezone(timezone.utc)
                else:
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} timestamp " "is not UTC"
                    )

                age = row_timestamp_upper_bound - updated_at
                if age < timedelta(0):
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} has a " "future timestamp"
                    )
                if updated_at < datetime(2000, 1, 1, tzinfo=timezone.utc):
                    raise SafetyAllocationSnapshotError(
                        f"allocation for portfolio {portfolio_id!r} has an "
                        "implausibly old timestamp"
                    )

            integer_quantities.append(quantity_int)
            allocations.append(
                SafetyPortfolioAllocation(
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    quantity=Decimal(quantity_int),
                    updated_at=updated_at,
                )
            )

        # Sum integers first, then cross the exact-Decimal boundary.  Decimal
        # addition is context-sensitive for very large values; this ordering
        # cannot round a valid aggregate.
        aggregate = Decimal(sum(integer_quantities))
        has_positive = any(quantity > 0 for quantity in integer_quantities)
        has_negative = any(quantity < 0 for quantity in integer_quantities)

        # Safety boundary text rejects strings that merely resemble raw broker
        # account numbers (for example a random ``f`` followed by digits).
        # Encode UUID nibbles with a letters-only alphabet so a random snapshot
        # identifier can never trip that fail-closed secret detector.
        snapshot_nonce = uuid.uuid4().hex.translate(
            str.maketrans("0123456789abcdef", "ghjkmnpqrstvwxyz")
        )
        return _produce_safety_allocation_snapshot(
            snapshot_id=f"allocation-db-{snapshot_nonce}",
            observed_at=observed_at,
            symbol=symbol,
            allocations=tuple(sorted(allocations, key=lambda allocation: allocation.portfolio_id)),
            aggregate_allocated_quantity=aggregate,
            has_offsetting_allocations=has_positive and has_negative,
            complete=True,
            database_path=str(self.db_path),
            database_identity=database_identity,
            database_device=database_device,
            database_inode=database_inode,
        )

    async def has_recent_buy_trade(
        self,
        symbol: str,
        seconds: int = 60,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> bool:
        """
        Check if a BUY trade for the symbol exists within the last N seconds.

        Used to prevent duplicate BUY trades across strategy systems (main + pairs).

        Args:
            symbol: Stock symbol to check (validated against symbol format)
            seconds: Time window in seconds (default 60, must be 1-86400)
            portfolio_id: Portfolio to scope the check to

        Returns:
            True if a BUY trade exists within the time window

        Raises:
            ValidationError: If symbol format is invalid
            ValueError: If seconds is not a positive integer in valid range
        """
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        # Validate symbol for consistency with other methods
        symbol = DatabaseValidator.validate_symbol(symbol)

        # Validate seconds parameter - must be positive int in reasonable range
        if not isinstance(seconds, int):
            raise ValueError(f"seconds must be int, got {type(seconds).__name__}")
        if seconds <= 0 or seconds > 86400:  # Max 24 hours
            raise ValueError(f"seconds must be between 1 and 86400, got {seconds}")

        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM trades
                WHERE portfolio_id = ?
                AND symbol = ?
                AND side = 'BUY'
                AND timestamp > datetime('now', ? || ' seconds')
                """,
                (portfolio_id, symbol, f"-{seconds}"),
            )
            row = await cursor.fetchone()
            return row[0] > 0 if row else False

    async def has_recent_sell_trade(
        self,
        symbol: str,
        seconds: int = 60,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> bool:
        """
        Check if a SELL trade for the symbol exists within the last N seconds.

        Used to prevent duplicate SELL trades in pairs trading strategy.

        Args:
            symbol: Stock symbol to check (validated against symbol format)
            seconds: Time window in seconds (default 60, must be 1-86400)
            portfolio_id: Portfolio to scope the check to

        Returns:
            True if a SELL trade exists within the time window

        Raises:
            ValidationError: If symbol format is invalid
            ValueError: If seconds is not a positive integer in valid range
        """
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        # Validate symbol for consistency with other methods
        symbol = DatabaseValidator.validate_symbol(symbol)

        # Validate seconds parameter - must be positive int in reasonable range
        if not isinstance(seconds, int):
            raise ValueError(f"seconds must be int, got {type(seconds).__name__}")
        if seconds <= 0 or seconds > 86400:  # Max 24 hours
            raise ValueError(f"seconds must be between 1 and 86400, got {seconds}")

        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM trades
                WHERE portfolio_id = ?
                AND symbol = ?
                AND side = 'SELL'
                AND timestamp > datetime('now', ? || ' seconds')
                """,
                (portfolio_id, symbol, f"-{seconds}"),
            )
            row = await cursor.fetchone()
            return row[0] > 0 if row else False

    async def get_recent_trades(
        self,
        limit: int = 100,
        symbol: Optional[str] = None,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> List[Dict]:
        """Get recent trades, optionally filtered by symbol, scoped to portfolio."""
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        # Validate symbol if provided
        if symbol:
            symbol = DatabaseValidator.validate_symbol(symbol)

        async with self.get_connection() as conn:
            if symbol:
                cursor = await conn.execute(
                    """
                    SELECT symbol, side, quantity, price, notional, slippage, commission, pnl, timestamp
                    FROM trades
                    WHERE portfolio_id = ? AND symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (portfolio_id, symbol, limit),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT symbol, side, quantity, price, notional, slippage, commission, pnl, timestamp
                    FROM trades
                    WHERE portfolio_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (portfolio_id, limit),
                )

            rows = await cursor.fetchall()
            return [
                {
                    "symbol": row[0],
                    "side": row[1],  # API uses 'side' for backward compatibility
                    "quantity": row[2],
                    "price": row[3],
                    "notional": row[4],
                    "slippage": row[5],
                    "commission": row[6],
                    "pnl": row[7],
                    "timestamp": row[8],
                }
                for row in rows
            ]

    async def get_account_info(self, portfolio_id: str = DEFAULT_PORTFOLIO_ID) -> Dict:
        """Get current account information for a portfolio."""
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp
                FROM account
                WHERE portfolio_id = ?
            """,
                (portfolio_id,),
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "cash": row[0],
                    "equity": row[1],
                    "daily_pnl": row[2],
                    "realized_pnl": row[3],
                    "unrealized_pnl": row[4],
                    "timestamp": row[5],
                }
            return {}

    async def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest market data for a symbol."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (symbol, limit),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "timestamp": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                }
                for row in rows
            ]

    async def save_equity_snapshot(
        self,
        equity: float,
        cash: float = 0.0,
        positions_value: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        snapshot_date: Optional[str] = None,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> None:
        """Save a daily equity snapshot for portfolio value tracking.

        This is the industry standard approach for tracking portfolio value over time.
        Called at end of each trading day or when account summary is updated.
        """
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)

        if snapshot_date is None:
            snapshot_date = datetime.now().strftime("%Y-%m-%d")

        async with self.get_connection() as conn:
            # Use INSERT OR REPLACE to update if date already exists for this portfolio
            await conn.execute(
                """
                INSERT OR REPLACE INTO equity_history
                (portfolio_id, date, equity, cash, positions_value, realized_pnl, unrealized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    portfolio_id,
                    snapshot_date,
                    equity,
                    cash,
                    positions_value,
                    realized_pnl,
                    unrealized_pnl,
                ),
            )
            await conn.commit()
            logger.debug(
                f"Saved equity snapshot: {snapshot_date} equity={equity:.2f} (portfolio={portfolio_id})"
            )

    async def get_equity_history(
        self,
        days: int = 365,
        portfolio_id: str = DEFAULT_PORTFOLIO_ID,
    ) -> List[Dict]:
        """Get equity history for the specified number of days.

        Returns list of daily snapshots ordered by date ascending (oldest first).
        """
        portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT date, equity, cash, positions_value, realized_pnl, unrealized_pnl, timestamp
                FROM equity_history
                WHERE portfolio_id = ?
                ORDER BY date ASC
                LIMIT ?
            """,
                (portfolio_id, days),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "date": row[0],
                    "equity": row[1],
                    "cash": row[2],
                    "positions_value": row[3],
                    "realized_pnl": row[4],
                    "unrealized_pnl": row[5],
                    "timestamp": row[6],
                }
                for row in rows
            ]

    async def portfolio_exists(self, portfolio_id: str) -> bool:
        """Check if a portfolio has any data in the database.

        Checks the account table first (every active portfolio should have an
        account row), then falls back to the portfolios definition table.

        Args:
            portfolio_id: The portfolio ID to check

        Returns:
            True if the portfolio exists (has account data or a portfolio definition)
        """
        try:
            portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
        except ValidationError:
            return False
        async with self.get_connection() as conn:
            # Check account table (every portfolio should have an account row)
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM account WHERE portfolio_id = ?",
                (portfolio_id,),
            )
            count = (await cursor.fetchone())[0]
            if count > 0:
                return True

            # Fall back to portfolios definition table
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM portfolios WHERE id = ?",
                (portfolio_id,),
            )
            count = (await cursor.fetchone())[0]
            return count > 0

    # ── Portfolio management methods ──

    async def get_portfolios(self) -> List[Dict]:
        """Get all portfolio definitions."""
        async with self.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT id, name, starting_cash, symbols, active,
                       max_position_pct, max_daily_loss_pct, max_open_positions,
                       stop_loss_pct, trailing_stop_pct, use_trailing_stop,
                       enabled_strategies, min_confidence,
                       created_at, updated_at
                FROM portfolios
                ORDER BY created_at ASC
            """)
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "starting_cash": row[2],
                    "symbols": row[3],
                    "active": bool(row[4]),
                    "max_position_pct": row[5],
                    "max_daily_loss_pct": row[6],
                    "max_open_positions": row[7],
                    "stop_loss_pct": row[8],
                    "trailing_stop_pct": row[9],
                    "use_trailing_stop": bool(row[10]) if row[10] is not None else None,
                    "enabled_strategies": row[11],
                    "min_confidence": row[12],
                    "created_at": row[13],
                    "updated_at": row[14],
                }
                for row in rows
            ]

    async def upsert_portfolio(self, portfolio_data: Dict) -> None:
        """Insert or update a portfolio definition."""
        portfolio_data["id"] = DatabaseValidator.validate_portfolio_id(portfolio_data.get("id"))
        if "name" in portfolio_data and portfolio_data["name"] is not None:
            portfolio_data["name"] = DatabaseValidator._validate_string(
                portfolio_data["name"], "name", max_length=128
            )
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO portfolios
                    (id, name, starting_cash, symbols, active,
                     max_position_pct, max_daily_loss_pct, max_open_positions,
                     stop_loss_pct, trailing_stop_pct, use_trailing_stop,
                     enabled_strategies, min_confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    portfolio_data["id"],
                    portfolio_data.get("name", portfolio_data["id"]),
                    portfolio_data.get("starting_cash", 100000),
                    portfolio_data.get("symbols", ""),
                    1 if portfolio_data.get("active", True) else 0,
                    portfolio_data.get("max_position_pct"),
                    portfolio_data.get("max_daily_loss_pct"),
                    portfolio_data.get("max_open_positions"),
                    portfolio_data.get("stop_loss_pct"),
                    portfolio_data.get("trailing_stop_pct"),
                    (
                        1
                        if portfolio_data.get("use_trailing_stop")
                        else (0 if portfolio_data.get("use_trailing_stop") is False else None)
                    ),
                    portfolio_data.get("enabled_strategies"),
                    portfolio_data.get("min_confidence"),
                ),
            )
            await conn.commit()
            logger.info(f"Upserted portfolio: {portfolio_data['id']}")

    async def cleanup_old_data(
        self, days_to_keep: int = 30, portfolio_id: Optional[str] = None
    ) -> None:
        """Clean up old data from the database.

        Always cleans truly-global tables (market_data, ticks). The signals table
        is portfolio-scoped, so it is only cleaned when an explicit portfolio_id
        is provided — never blanketly across all tenants.
        """
        async with self.get_connection() as conn:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 86400)
            cutoff_dt = datetime.fromtimestamp(cutoff_date)

            # Clean up old market data (global)
            await conn.execute(
                "DELETE FROM market_data WHERE timestamp < ?",
                (cutoff_dt,),
            )

            # Clean up old ticks (global)
            await conn.execute(
                "DELETE FROM ticks WHERE timestamp < ?",
                (cutoff_dt,),
            )

            # Signals are portfolio-scoped; only clean if a specific portfolio is named.
            if portfolio_id is not None:
                portfolio_id = DatabaseValidator.validate_portfolio_id(portfolio_id)
                await conn.execute(
                    "DELETE FROM signals WHERE portfolio_id = ? AND timestamp < ?",
                    (portfolio_id, cutoff_dt),
                )

            await conn.commit()
            logger.info(
                f"Cleaned up data older than {days_to_keep} days "
                f"(portfolio={portfolio_id or 'global-only'})"
            )


# Backward compatibility wrapper
def create_async_database(db_path: Path = DB_PATH) -> AsyncTradingDatabase:
    """Create an async database instance."""
    return AsyncTradingDatabase(db_path)
