"""
Subprocess-Based IBKR Client

Manages a subprocess running ibkr_subprocess_worker.py to completely isolate
ib_async from the main trading system's complex async environment.

This solves the ib_async library incompatibility issue where API handshakes
timeout in complex async environments despite successful TCP connections.

CRITICAL FIX: Uses threading for subprocess I/O instead of asyncio.create_subprocess_exec
to avoid event loop starvation in busy async environments.
"""

import asyncio
import hmac
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar, cast

import structlog

from robo_trader.broker_account_identity import (
    is_supported_paper_account_identifier,
    normalize_synthetic_paper_account_environment,
)
from robo_trader.broker_safety_evidence import (
    BrokerContractSafetySnapshot,
    BrokerSafetyContract,
    BrokerSafetyOpenOrder,
    BrokerSafetyPosition,
    BrokerSafetySnapshot,
    _BrokerContractSnapshotCapability,
    _BrokerSnapshotCapability,
    _issue_broker_contract_snapshot_capability,
    _issue_broker_snapshot_capability,
    _produce_broker_contract_safety_snapshot,
    _produce_broker_safety_snapshot,
    assert_producer_owned_broker_contract_safety_snapshot,
    assert_producer_owned_broker_safety_snapshot,
)
from robo_trader.reconciliation.identity import (
    RuntimeSafetyContext,
    assert_validated_runtime_safety_context,
    mask_account_identifier,
    validate_ibc_safety_config,
)

logger = structlog.get_logger(__name__)
_BrokerCallbackResult = TypeVar("_BrokerCallbackResult")


def _find_project_root(start: Path) -> Path:
    """
    NEW-IB-M1.1: Locate the project root by walking up from `start` looking
    for a marker file (`pyproject.toml` or `START_TRADER.sh`). This is more
    robust than `Path(__file__).parents[2]` because it survives the file
    being relocated within the package and doesn't silently break if the
    package layout changes.

    Falls back to ``parents[2]`` only if no marker is found (preserves
    pre-fix behavior so we never raise during import).
    """
    cur = start.resolve()
    # Walk up at most 8 levels to bound the search.
    for _ in range(8):
        if (cur / "pyproject.toml").exists() or (cur / "START_TRADER.sh").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: legacy probe.
    return start.resolve().parents[2]


# NEW-IB-M1.1: Allowlist of prefixes the worker interpreter may resolve to.
# A symlink whose realpath escapes both the project root AND these system
# locations is rejected. Order matters only for log clarity.
_INTERPRETER_PREFIX_ALLOWLIST: tuple[Path, ...] = (
    Path("/usr/bin"),
    Path("/usr/local/bin"),
    Path("/opt/homebrew"),
    Path("/Library/Frameworks/Python.framework"),
)

# The diagnostic worker receives broker connection parameters over its
# request-correlated stdin transport. It does not need the dashboard, model,
# database, IBC, or shell environment. Keep only non-secret locale settings
# required for predictable text/time handling.
_WORKER_ENV_ALLOWLIST = frozenset(
    {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        # Required by CPython on Windows; harmless and non-secret elsewhere.
        "SYSTEMROOT",
        "WINDIR",
    }
)
_WORKER_SYNTHETIC_ACCOUNT_ENVIRONMENT = "ROBOTRADER_WORKER_SYNTHETIC_ACCOUNT_ENVIRONMENT"
_LSOF_EXECUTABLE_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/sbin/lsof"),
    Path("/usr/bin/lsof"),
)
_WORKER_BOOTSTRAP = (
    "import runpy,sys;"
    "root=sys.argv[1];worker=sys.argv[2];"
    "sys.path.insert(0,root);"
    "runpy.run_path(worker,run_name='__main__')"
)


def _trusted_lsof_executable() -> Path:
    """Return one fixed, non-symlinked system lsof executable or fail closed."""
    for candidate in _LSOF_EXECUTABLE_CANDIDATES:
        if not candidate.is_absolute() or candidate.is_symlink():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved != candidate or not resolved.is_file() or not os.access(resolved, os.X_OK):
            continue
        return resolved
    raise IBKRError("Trusted absolute lsof executable is unavailable")


def _is_interpreter_path_safe(resolved: Path, project_root: Path) -> bool:
    """Return True if a resolved interpreter path is acceptable for exec."""
    try:
        resolved.relative_to(project_root)
        return True
    except ValueError:
        pass
    for prefix in _INTERPRETER_PREFIX_ALLOWLIST:
        try:
            resolved.relative_to(prefix)
            return True
        except ValueError:
            continue
    return False


class SubprocessCrashError(Exception):
    """Raised when subprocess crashes or becomes unresponsive"""

    pass


class IBKRError(Exception):
    """Raised when IBKR operation fails"""

    pass


class IBKRTimeoutError(SubprocessCrashError, IBKRError):
    """Raised when the subprocess transport times out and must be replaced."""

    pass


class IBKRTransportPoisonedError(SubprocessCrashError, IBKRError):
    """Raised when request/response integrity for a worker generation is lost."""

    pass


class IBKRDisconnectedError(IBKRError):
    """Raised when the worker reports that its broker session is disconnected."""

    pass


class IBKRConnectionConflictError(IBKRError):
    """Raised when a connected worker is reused with different parameters."""

    pass


class BrokerSnapshotAccountMismatchError(IBKRTransportPoisonedError):
    """Raised without exposing raw IDs when snapshot account identity is wrong."""

    pass


class GatewayRequiresRestartError(IBKRError):
    """Raised when the worker detects the Gateway API layer has crashed"""

    pass


class IBKRTimeoutRequiresGatewayRestartError(IBKRTimeoutError, GatewayRequiresRestartError):
    """Worker timeout that requires both worker replacement and Gateway recovery."""

    pass


_TRANSPORT_PROTOCOL_VERSION = 1
_INTRADAY_BAR_SIZES = {
    "1 secs",
    "5 secs",
    "10 secs",
    "15 secs",
    "30 secs",
    "1 min",
    "2 mins",
    "3 mins",
    "5 mins",
    "10 mins",
    "15 mins",
    "20 mins",
    "30 mins",
    "1 hour",
    "2 hours",
    "3 hours",
    "4 hours",
    "8 hours",
}
_BROKER_SNAPSHOT_SCHEMA_VERSION = 1
_BROKER_SAFETY_SNAPSHOT_SCHEMA_VERSION = 1
_BROKER_CONTRACT_SAFETY_SNAPSHOT_SCHEMA_VERSION = 1
_BROKER_SNAPSHOT_BALANCE_TAGS = {
    "NetLiquidation",
    "TotalCashValue",
    "SettledCash",
    "GrossPositionValue",
    "RealizedPnL",
    "UnrealizedPnL",
}
_BROKER_SNAPSHOT_REQUIRED_BALANCE_TAGS = {"NetLiquidation", "TotalCashValue"}
_BROKER_SNAPSHOT_MAX_AGE_SECONDS = 300.0
_BROKER_SNAPSHOT_MAX_WINDOW_SECONDS = 60.0
_BROKER_SNAPSHOT_MAX_CLOCK_SKEW_SECONDS = 120.0
_BROKER_EXECUTION_LOOKBACK_SECONDS = 24 * 60 * 60
_CONTRACT_TEXT_RE = re.compile(r"^[A-Z0-9][A-Z0-9._:/-]{0,63}$")
_BROKER_SAFETY_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_OPAQUE_ACCOUNT_SCOPE_RE = re.compile(r"^acct_v1_[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class QualifiedStockContractLineage:
    """Immutable IBKR identity and timing for one historical-data response."""

    con_id: int
    symbol: str
    local_symbol: str
    security_type: str
    currency: str
    exchange: str
    primary_exchange: str
    trading_class: str
    broker_timestamp: datetime
    retrieval_timestamp: datetime
    transport_generation: str

    def __post_init__(self) -> None:
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValueError("qualified contract con_id must be a positive integer")
        for field_name in (
            "symbol",
            "local_symbol",
            "primary_exchange",
            "trading_class",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _CONTRACT_TEXT_RE.fullmatch(value):
                raise ValueError(f"qualified contract {field_name} is malformed")
        if self.local_symbol != self.symbol:
            raise ValueError("qualified contract local_symbol must match symbol")
        if self.security_type != "STK":
            raise ValueError("qualified contract security_type must be STK")
        if self.currency != "USD":
            raise ValueError("qualified contract currency must be USD")
        if self.exchange != "SMART":
            raise ValueError("qualified contract exchange must be SMART")
        for field_name in ("broker_timestamp", "retrieval_timestamp"):
            value = getattr(self, field_name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"qualified contract {field_name} must be timezone-aware")
            object.__setattr__(self, field_name, value.astimezone(timezone.utc))
        if abs((self.broker_timestamp - self.retrieval_timestamp).total_seconds()) > (
            _BROKER_SNAPSHOT_MAX_CLOCK_SKEW_SECONDS
        ):
            raise ValueError(
                "qualified contract broker/retrieval timestamps exceed clock-skew tolerance"
            )
        if (
            not isinstance(self.transport_generation, str)
            or not self.transport_generation
            or self.transport_generation != self.transport_generation.strip()
            or len(self.transport_generation) > 128
        ):
            raise ValueError("qualified contract transport_generation is malformed")


@dataclass
class _PendingResponse:
    command: str
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


@dataclass
class _WorkerGeneration:
    """All response-routing state owned by one worker process generation."""

    generation_id: str
    process: subprocess.Popen
    pending: dict[str, _PendingResponse] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    completed_order: deque[str] = field(default_factory=deque)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    poisoned_reason: Optional[str] = None
    intentional_stop: bool = False
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    debug_log_file: Any = None
    debug_log_path: Optional[str] = None


class SubprocessIBKRClient:
    """
    Async client that manages IBKR connection via subprocess.

    Provides complete process isolation for ib_async library to avoid
    async environment conflicts.

    Uses threading for subprocess I/O to avoid asyncio event loop starvation.
    """

    # Worker timeout constants (must match ibkr_subprocess_worker.py)
    WORKER_HANDSHAKE_TIMEOUT = 15.0  # max_handshake_wait in worker
    WORKER_STABILIZATION_DELAY = 0.5  # stabilization sleep in worker
    WORKER_ACCOUNT_TIMEOUT = 10.0  # max_account_wait in worker
    WORKER_MAX_WAIT = (
        WORKER_HANDSHAKE_TIMEOUT + WORKER_STABILIZATION_DELAY + WORKER_ACCOUNT_TIMEOUT
    )  # 25.5s

    def __init__(self, *, worker_runtime_environment: object = None):
        self.process: Optional[subprocess.Popen] = None
        self.lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._connection_state_lock = threading.Lock()
        self._debug_log_cleanup_lock = threading.Lock()
        self._connected = False
        self._connection_identity: Optional[tuple[str, int, int, bool]] = None
        self._connection_generation_id: Optional[str] = None
        self._generation: Optional[_WorkerGeneration] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_reader_thread: Optional[threading.Thread] = None
        self._last_activity: Optional[datetime] = None
        self._connection_start_time: Optional[datetime] = None
        self._gateway_api_down_detail: Optional[str] = None
        self._gateway_failure_generation_id: Optional[str] = None
        self._debug_log_file = None  # For capturing worker stderr to file
        self._debug_log_path: Optional[str] = None  # D-3: randomized tempfile path
        self._zombies_detected_before_connect = (
            False  # Track if zombies were present before connect
        )
        self._historical_lineage_lock = threading.Lock()
        self._historical_lineage_generation_id: Optional[str] = None
        self._historical_lineage_by_symbol: dict[str, QualifiedStockContractLineage] = {}
        # The worker receives a fresh, secret-free environment rather than the
        # parent's ambient process environment. Preserve only the validated
        # dev/test classification needed by deterministic synthetic accounts;
        # every other value is production-like and therefore omitted.
        self._worker_synthetic_account_environment = (
            normalize_synthetic_paper_account_environment(worker_runtime_environment) or ""
        )

    def _invalidate_historical_lineage(self) -> None:
        """Forget broker identity evidence whenever its transport is no longer current."""
        with self._historical_lineage_lock:
            self._historical_lineage_generation_id = None
            self._historical_lineage_by_symbol.clear()

    def _cache_historical_lineage(
        self,
        generation: _WorkerGeneration,
        lineage: QualifiedStockContractLineage,
    ) -> None:
        """Publish lineage only while its exact worker generation is current."""
        with generation.state_lock:
            with self._connection_state_lock:
                current = (
                    self._generation is generation
                    and generation.poisoned_reason is None
                    and self._connected
                    and self._connection_generation_id == generation.generation_id
                )
                if current:
                    with self._historical_lineage_lock:
                        if self._historical_lineage_generation_id != generation.generation_id:
                            self._historical_lineage_by_symbol.clear()
                            self._historical_lineage_generation_id = generation.generation_id
                        self._historical_lineage_by_symbol[lineage.symbol] = lineage
        if not current:
            self._invalidate_historical_lineage()
            raise IBKRTransportPoisonedError(
                "Historical contract lineage belongs to a stale worker generation"
            )

    def get_cached_historical_lineage(self, symbol: str) -> QualifiedStockContractLineage:
        """Return current-generation lineage or reject stale/missing evidence."""
        requested = self._normalize_historical_symbol(symbol)
        generation = self._generation
        if generation is None:
            self._invalidate_historical_lineage()
            raise IBKRTransportPoisonedError(
                "Historical contract lineage has no current worker generation"
            )
        with generation.state_lock:
            with self._connection_state_lock:
                generation_current = (
                    self._generation is generation
                    and generation.poisoned_reason is None
                    and self._connected
                    and self._connection_generation_id == generation.generation_id
                )
                if generation_current:
                    with self._historical_lineage_lock:
                        cache_current = self._historical_lineage_generation_id in (
                            None,
                            generation.generation_id,
                        )
                        lineage = (
                            self._historical_lineage_by_symbol.get(requested)
                            if cache_current
                            else None
                        )
                        lineage_current = (
                            lineage is not None
                            and lineage.transport_generation == generation.generation_id
                        )
                else:
                    cache_current = False
                    lineage = None
                    lineage_current = False
        if not generation_current or not cache_current:
            self._invalidate_historical_lineage()
            raise IBKRTransportPoisonedError(
                "Historical contract lineage belongs to a stale worker generation"
            )
        if lineage is None:
            raise IBKRError(f"No current qualified-contract lineage for {requested}")
        if not lineage_current:
            self._invalidate_historical_lineage()
            raise IBKRTransportPoisonedError(
                "Historical contract lineage generation is inconsistent"
            )
        return lineage

    def _clear_connection_tuple_locked(self) -> None:
        """Clear connected-session fields while holding the state lock."""
        self._connected = False
        self._connection_identity = None
        self._connection_generation_id = None
        self._connection_start_time = None
        self._last_activity = None
        self._invalidate_historical_lineage()

    def _clear_cached_connection_state(
        self,
        *,
        generation: Optional[_WorkerGeneration] = None,
        clear_gateway_detail: bool = True,
    ) -> bool:
        """Atomically forget state, optionally only for its bound generation."""
        with self._connection_state_lock:
            if generation is not None:
                generation_id = generation.generation_id
                if self._connection_generation_id not in (None, generation_id) or (
                    self._connection_generation_id is None and self._generation is not generation
                ):
                    return False
            else:
                generation_id = None

            self._clear_connection_tuple_locked()
            if clear_gateway_detail and (
                generation is None or self._gateway_failure_generation_id in (None, generation_id)
            ):
                self._gateway_api_down_detail = None
                self._gateway_failure_generation_id = None
            return True

    def _connection_state_snapshot(
        self,
    ) -> tuple[
        bool,
        Optional[tuple[str, int, int, bool]],
        Optional[str],
        Optional[datetime],
        Optional[datetime],
        Optional[str],
    ]:
        """Return one internally consistent cached-state snapshot."""
        with self._connection_state_lock:
            return (
                self._connected,
                self._connection_identity,
                self._connection_generation_id,
                self._connection_start_time,
                self._last_activity,
                self._gateway_api_down_detail,
            )

    def _cleanup_worker_debug_log(
        self,
        generation: Optional[_WorkerGeneration],
        *,
        required: bool = False,
    ) -> None:
        """Close and unlink one generation's temporary stderr capture.

        References are cleared only for the operation that actually succeeded.
        A later stop or cleanup call can therefore retry a transient close or
        unlink failure without losing the remaining handle or path.
        """
        failures: list[tuple[str, Exception]] = []
        with self._debug_log_cleanup_lock:
            debug_log_file = (
                generation.debug_log_file if generation is not None else self._debug_log_file
            )
            debug_log_path = (
                generation.debug_log_path if generation is not None else self._debug_log_path
            )

            close_succeeded = debug_log_file is None
            if debug_log_file is not None:
                try:
                    debug_log_file.close()
                    close_succeeded = True
                except Exception as exc:
                    failures.append(("close", exc))

            unlink_succeeded = not debug_log_path
            if debug_log_path:
                try:
                    os.unlink(debug_log_path)
                    unlink_succeeded = True
                except FileNotFoundError:
                    unlink_succeeded = True
                except OSError as exc:
                    failures.append(("remove", exc))

            if close_succeeded:
                if generation is not None and generation.debug_log_file is debug_log_file:
                    generation.debug_log_file = None
                if self._debug_log_file is debug_log_file:
                    self._debug_log_file = None
            if unlink_succeeded:
                if generation is not None and generation.debug_log_path == debug_log_path:
                    generation.debug_log_path = None
                if self._debug_log_path == debug_log_path:
                    self._debug_log_path = None

        for operation, failure_exc in failures:
            logger.warning(
                f"Could not {operation} worker debug log",
                error=str(failure_exc),
            )
        if failures and required:
            operations = " and ".join(operation for operation, _ in failures)
            raise SubprocessCrashError(
                f"Could not complete diagnostic worker debug log cleanup ({operations})"
            ) from failures[0][1]

    def _cleanup_worker_debug_log_with_retry(
        self,
        generation: Optional[_WorkerGeneration],
        *,
        attempts: int = 2,
    ) -> None:
        """Require verified cleanup, retrying transient close/unlink failures."""
        last_error: Optional[SubprocessCrashError] = None
        for _ in range(attempts):
            try:
                self._cleanup_worker_debug_log(generation, required=True)
                return
            except SubprocessCrashError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    def _record_gateway_failure(
        self,
        generation: _WorkerGeneration,
        detail: str,
    ) -> bool:
        """Bind a Gateway-down diagnosis to the exact responding generation."""
        with generation.state_lock:
            if generation.poisoned_reason is not None:
                return False
            with self._connection_state_lock:
                if self._generation is not generation or self._connection_generation_id not in (
                    None,
                    generation.generation_id,
                ):
                    return False
                self._clear_connection_tuple_locked()
                self._gateway_api_down_detail = detail
                self._gateway_failure_generation_id = generation.generation_id
                return True

    def _accept_ping_response(
        self,
        data: dict,
        generation: _WorkerGeneration,
    ) -> bool:
        """Update cached state from a lifecycle-serialized worker ping."""
        with generation.state_lock:
            if generation.poisoned_reason is not None:
                self._clear_cached_connection_state(generation=generation)
                return False
            with self._connection_state_lock:
                # A stale response must never mutate replacement state.
                if (
                    self._generation is not generation
                    or self._connection_generation_id != generation.generation_id
                ):
                    return False

                if data.get("gateway_api_down"):
                    detail = data.get("detail") or "Gateway API layer reported down by worker ping"
                    self._clear_connection_tuple_locked()
                    self._gateway_api_down_detail = detail
                    self._gateway_failure_generation_id = generation.generation_id
                    logger.error("Worker ping reports Gateway API down", detail=detail)
                    return False

                if data.get("pong") is not True or data.get("connected") is not True:
                    self._clear_connection_tuple_locked()
                    self._gateway_api_down_detail = None
                    self._gateway_failure_generation_id = None
                    logger.warning(
                        "Worker is responsive but broker session is disconnected",
                        pong=data.get("pong"),
                        connected=data.get("connected"),
                    )
                    return False

                if self._connection_identity is None:
                    self._clear_connection_tuple_locked()
                    logger.warning(
                        "Worker reports a broker session without validated "
                        "connection identity; explicit reconnect required"
                    )
                    return False

                self._connected = True
                self._last_activity = datetime.now(timezone.utc)
                self._gateway_api_down_detail = None
                self._gateway_failure_generation_id = None
                return True

    async def start(self) -> None:
        """Serialize worker creation against commands and shutdown."""
        async with self._lifecycle_lock:
            await self._start_unlocked()

    async def _start_unlocked(self) -> None:
        """Start the subprocess worker with threading-based I/O."""
        current_generation = self._generation
        current_poison = None
        if current_generation:
            with current_generation.state_lock:
                current_poison = current_generation.poisoned_reason
        if current_poison:
            await self._stop_unlocked()
        if self.process and self.process.poll() is None:
            logger.warning("Subprocess already running")
            return
        if self._generation:
            await self._stop_unlocked()

        # Start subprocess using the same Python interpreter.
        #
        # SECURITY (IB-M1 + NEW-IB-M1.1):
        # - IB-M1: Only honor VIRTUAL_ENV if it points inside the project root.
        #   A local attacker who controls the user's shell environment could
        #   otherwise set VIRTUAL_ENV to a malicious venv (e.g. via a poisoned
        #   .envrc / direnv / asdf shim) and execute arbitrary code with the
        #   worker's IBKR credentials.
        # - NEW-IB-M1.1: Validating the venv directory alone is not enough —
        #   the candidate interpreter path is often a symlink. We resolve the
        #   final realpath and require it to live under the project root or
        #   one of a small set of trusted system prefixes
        #   (/usr/bin, /usr/local/bin, /opt/homebrew,
        #   /Library/Frameworks/Python.framework). A symlink that resolves
        #   outside the allowlist is rejected.
        # - NEW-IB-M1.1: Project root is now found via marker-file probing
        #   (pyproject.toml / START_TRADER.sh) instead of fragile
        #   parents[2] indexing — this stays correct if the file moves.
        project_root = _find_project_root(Path(__file__)).resolve()
        worker_script = (
            project_root / "robo_trader" / "clients" / "ibkr_subprocess_worker.py"
        ).resolve(strict=True)
        try:
            worker_script.relative_to(project_root)
        except ValueError as exc:
            raise RuntimeError("IBKR worker script resolves outside the project root") from exc
        if not worker_script.is_file():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")

        logger.info("Starting IBKR subprocess worker", script=str(worker_script))

        project_venv_python = project_root / ".venv" / "bin" / "python3"

        python_exe = None
        ve = os.environ.get("VIRTUAL_ENV")
        if ve:
            ve_path = Path(ve).resolve()
            try:
                ve_path.relative_to(project_root)
            except ValueError:
                logger.warning(
                    "Ignoring VIRTUAL_ENV=%s: outside project root, refusing for security.",
                    ve,
                )
            else:
                import platform

                if platform.system() == "Windows":
                    candidate = ve_path / "Scripts" / "python.exe"
                else:
                    candidate = ve_path / "bin" / "python3"
                if candidate.exists():
                    # NEW-IB-M1.1: resolve symlink and validate realpath.
                    candidate_resolved = candidate.resolve()
                    if _is_interpreter_path_safe(candidate_resolved, project_root):
                        python_exe = str(candidate)
                        logger.debug(
                            "Using VIRTUAL_ENV Python",
                            python_exe=python_exe,
                            resolved=str(candidate_resolved),
                        )
                    else:
                        logger.warning(
                            "Refusing VIRTUAL_ENV interpreter: realpath outside allowlist.",
                            candidate=str(candidate),
                            resolved=str(candidate_resolved),
                        )

        # Preferred: project's own .venv python
        if python_exe is None and project_venv_python.exists():
            project_venv_resolved = project_venv_python.resolve()
            if _is_interpreter_path_safe(project_venv_resolved, project_root):
                python_exe = str(project_venv_python)
                logger.debug(
                    "Using project venv Python",
                    python_exe=python_exe,
                    resolved=str(project_venv_resolved),
                )
            else:
                logger.warning(
                    "Project venv Python resolves outside allowlist - skipping.",
                    candidate=str(project_venv_python),
                    resolved=str(project_venv_resolved),
                )

        # Last resort: sys.executable (still validated against allowlist)
        if python_exe is None:
            sys_exe_resolved = Path(sys.executable).resolve()
            if _is_interpreter_path_safe(sys_exe_resolved, project_root):
                python_exe = sys.executable
                logger.debug(
                    "Using sys.executable Python",
                    python_exe=python_exe,
                    resolved=str(sys_exe_resolved),
                )
            else:
                # No safe interpreter available — refuse rather than exec
                # an interpreter from an untrusted location.
                raise RuntimeError(
                    "No safe Python interpreter found. sys.executable "
                    f"({sys.executable}) resolves to {sys_exe_resolved}, "
                    "which is outside both the project root and the trusted "
                    "system prefix allowlist (/usr/bin, /usr/local/bin, "
                    "/opt/homebrew, /Library/Frameworks/Python.framework). "
                    "This blocks IBKR worker startup as a defense against "
                    "PATH/symlink hijacking (NEW-IB-M1.1)."
                )

        # DEBUGGING FIX: Create debug log file for worker stderr capture.
        # D-3: Use tempfile to get an unpredictable path with 0o600 perms (mkstemp
        # creates the file atomically with O_EXCL, so a pre-existing symlink at
        # the path cannot be followed). The previous deterministic path
        # `/tmp/worker_debug.log` was a symlink-attack vector on multi-user hosts.
        # Use try/finally to ensure cleanup even if subprocess/thread startup fails.
        debug_log_path: Optional[str] = None
        debug_log_file = None
        try:
            fd, debug_log_path = tempfile.mkstemp(prefix="worker_debug_", suffix=".log")
            debug_log_file = os.fdopen(fd, "w")
            self._debug_log_file = debug_log_file
            self._debug_log_path = debug_log_path
            logger.info("Worker debug output will be captured", debug_log=debug_log_path)
        except Exception as e:
            logger.warning("Could not create debug log file", error=str(e))
            debug_log_file = None

        # Wrap subprocess and thread startup in try block to ensure cleanup
        generation: Optional[_WorkerGeneration] = None
        try:
            generation_id = uuid.uuid4().hex
            worker_env = {
                name: os.environ[name] for name in _WORKER_ENV_ALLOWLIST if name in os.environ
            }
            worker_env.update(
                {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONSAFEPATH": "1",
                    "PYTHONUNBUFFERED": "1",
                    "ROBOTRADER_WORKER_GENERATION_ID": generation_id,
                }
            )
            if self._worker_synthetic_account_environment:
                worker_env[_WORKER_SYNTHETIC_ACCOUNT_ENVIRONMENT] = (
                    self._worker_synthetic_account_environment
                )
            # CRITICAL FIX: Use regular subprocess.Popen with threading instead of
            # asyncio.create_subprocess_exec to avoid event loop starvation in
            # busy async environments
            # Launch the exact resolved project worker under isolated mode.
            # ``-I`` ignores PYTHONPATH, the user site, and the inherited current
            # directory. The fixed bootstrap admits only the verified project
            # root before executing the verified script path.
            self.process = subprocess.Popen(
                [
                    python_exe,
                    "-I",
                    "-c",
                    _WORKER_BOOTSTRAP,
                    str(project_root),
                    str(worker_script),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr for logging
                text=True,
                bufsize=1,  # Line buffered
                close_fds=True,  # Don't inherit file descriptors
                env=worker_env,
                cwd=str(project_root),
            )
            generation = _WorkerGeneration(generation_id, self.process)
            generation.debug_log_file = debug_log_file
            generation.debug_log_path = debug_log_path
            with self._connection_state_lock:
                # Installing a worker is an authoritative disconnected state.
                # Synchronizing the pointer swap with cached state prevents a
                # delayed poison from the previous worker from clearing state
                # subsequently bound to this generation.
                self._clear_connection_tuple_locked()
                self._gateway_api_down_detail = None
                self._gateway_failure_generation_id = None
                self._generation = generation
            self._invalidate_historical_lineage()

            logger.info(
                "IBKR subprocess worker started",
                pid=self.process.pid,
                generation_id=generation_id,
            )

            # Start reader threads to avoid blocking
            self._reader_thread = threading.Thread(
                target=self._read_loop,
                args=(generation,),
                daemon=True,
                name="IBKRSubprocessReader",
            )
            generation.stdout_thread = self._reader_thread
            self._reader_thread.start()

            # Start stderr reader thread (this thread will manage debug_log_file lifecycle)
            self._stderr_reader_thread = threading.Thread(
                target=self._stderr_read_loop,
                args=(generation,),
                daemon=True,
                name="IBKRSubprocessStderrReader",
            )
            generation.stderr_thread = self._stderr_reader_thread
            self._stderr_reader_thread.start()

        except Exception:
            if generation:
                generation.intentional_stop = True
                generation.stop_event.set()
            process = generation.process if generation else self.process
            if process:
                try:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=1.0)
                except Exception:
                    pass
                for stream in (
                    getattr(process, "stdout", None),
                    getattr(process, "stderr", None),
                ):
                    try:
                        if stream:
                            stream.close()
                    except Exception:
                        pass
            if generation:
                for thread in (generation.stdout_thread, generation.stderr_thread):
                    if thread and thread.is_alive():
                        thread.join(timeout=1.0)
            with self._connection_state_lock:
                if generation is not None and self._generation is generation:
                    self._generation = None
                    self._clear_connection_tuple_locked()
                    if self._gateway_failure_generation_id in (
                        None,
                        generation.generation_id,
                    ):
                        self._gateway_api_down_detail = None
                        self._gateway_failure_generation_id = None
            if self.process is process:
                self.process = None
            self._cleanup_worker_debug_log_with_retry(generation)
            raise

    def _poison_generation(self, generation: _WorkerGeneration, reason: str) -> None:
        """Fail closed after transport ambiguity; a poisoned worker is never reused."""
        with generation.state_lock:
            if generation.poisoned_reason is not None:
                return
            generation.poisoned_reason = reason
            pending = list(generation.pending.values())
            generation.pending.clear()
            # Poisoning is an authoritative fail-closed transition. Clear the
            # generation-bound session state before exposing the poison or
            # notifying callbacks, using the global lock order
            # generation.state_lock -> _connection_state_lock.
            self._clear_cached_connection_state(generation=generation)

        self._invalidate_historical_lineage()
        error = IBKRTransportPoisonedError(f"IBKR worker generation poisoned: {reason}")
        for request in pending:

            def fail_pending(
                future: asyncio.Future = request.future,
                pending_error: Exception = error,
            ) -> None:
                if not future.done():
                    future.set_exception(pending_error)

            try:
                request.loop.call_soon_threadsafe(fail_pending)
            except RuntimeError:
                # The owning loop is already closed; transport teardown must
                # still continue and reap the ambiguous worker generation.
                pass

        logger.error(
            "Poisoning IBKR worker generation",
            generation_id=generation.generation_id,
            reason=reason,
        )

        def reap_poisoned_worker() -> None:
            try:
                if generation.process.poll() is not None:
                    return
                generation.process.terminate()
                try:
                    generation.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    generation.process.kill()
                    generation.process.wait(timeout=1.0)
            except Exception as exc:
                logger.warning("Failed to reap poisoned worker", error=str(exc))

        threading.Thread(
            target=reap_poisoned_worker,
            daemon=True,
            name=f"IBKRPoisonReaper-{generation.generation_id[:8]}",
        ).start()

    def _read_loop(self, generation: _WorkerGeneration) -> None:
        """Read and route responses for exactly one worker generation."""
        try:
            while not generation.stop_event.is_set():
                line = generation.process.stdout.readline()
                if not line:
                    if not generation.intentional_stop:
                        self._poison_generation(generation, "worker stdout closed")
                    break

                line_stripped = line.strip()
                if not line_stripped:
                    continue

                if line_stripped.startswith('{"timestamp":'):
                    logger.debug("ib_async_stdout", message=line_stripped)
                    continue
                if not line_stripped.startswith("{"):
                    self._poison_generation(generation, "malformed non-JSON worker response")
                    break

                try:
                    response = json.loads(line_stripped)
                except (json.JSONDecodeError, TypeError) as exc:
                    self._poison_generation(generation, f"malformed JSON response: {exc}")
                    break

                required = {
                    "protocol_version",
                    "generation_id",
                    "request_id",
                    "command",
                    "status",
                }
                if not isinstance(response, dict) or not required.issubset(response):
                    self._poison_generation(generation, "malformed response envelope")
                    break
                if response["protocol_version"] != _TRANSPORT_PROTOCOL_VERSION:
                    self._poison_generation(generation, "response protocol version mismatch")
                    break
                if any(
                    not isinstance(response.get(field), str) or not response[field]
                    for field in ("generation_id", "request_id", "command", "status")
                ):
                    self._poison_generation(generation, "invalid response identity types")
                    break
                if response["generation_id"] != generation.generation_id:
                    self._poison_generation(generation, "stale worker generation response")
                    break

                request_id = response["request_id"]
                with generation.state_lock:
                    if request_id in generation.completed:
                        duplicate = True
                        pending = None
                    else:
                        duplicate = False
                        pending = generation.pending.get(request_id)

                if duplicate:
                    self._poison_generation(generation, "duplicate response")
                    break
                if pending is None:
                    self._poison_generation(generation, "unknown response request ID")
                    break
                matched_pending = pending
                if response["command"] != matched_pending.command:
                    self._poison_generation(generation, "response command mismatch")
                    break
                with generation.state_lock:
                    generation.pending.pop(request_id, None)
                    generation.completed.add(request_id)
                    generation.completed_order.append(request_id)
                    while len(generation.completed_order) > 1024:
                        expired = generation.completed_order.popleft()
                        generation.completed.discard(expired)

                def deliver_response(
                    future: asyncio.Future = matched_pending.future,
                    value: dict = response,
                ) -> None:
                    if not future.done():
                        future.set_result(value)

                matched_pending.loop.call_soon_threadsafe(deliver_response)
        except Exception as e:
            logger.error("Reader thread error", error=str(e))
            if not generation.intentional_stop:
                self._poison_generation(generation, f"reader failure: {e}")
        finally:
            logger.debug("Reader thread exiting", generation_id=generation.generation_id)

    def _stderr_read_loop(self, generation: _WorkerGeneration) -> None:
        """Thread function to read and log subprocess stderr"""
        try:
            while not generation.stop_event.is_set():
                line = generation.process.stderr.readline()
                if not line:
                    break

                # Write to debug file for detailed analysis
                if generation.debug_log_file:
                    try:
                        generation.debug_log_file.write(
                            f"{datetime.now(timezone.utc).isoformat()}: {line}"
                        )
                        generation.debug_log_file.flush()
                    except Exception:
                        pass  # Don't let debug logging break the main flow

                # Log stderr output with appropriate level
                line_stripped = line.strip()
                if line_stripped:  # Only log non-empty lines
                    if "DEBUG:" in line_stripped:
                        logger.debug("subprocess_stderr", message=line_stripped)
                    elif "ERROR:" in line_stripped or "Exception" in line_stripped:
                        logger.error("subprocess_stderr", message=line_stripped)
                    else:
                        logger.warning("subprocess_stderr", message=line_stripped)
        except Exception as e:
            logger.error("Stderr reader thread error", error=str(e))
            if not generation.intentional_stop:
                self._poison_generation(generation, f"stderr reader failure: {e}")
        finally:
            logger.debug("Stderr reader thread exiting")
            self._cleanup_worker_debug_log(generation)

    async def stop(self) -> None:
        """Serialize shutdown against worker creation and commands."""
        async with self._lifecycle_lock:
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        """Stop the subprocess worker with graceful shutdown."""
        generation = self._generation
        process = generation.process if generation else self.process
        self._invalidate_historical_lineage()
        if not process:
            self._clear_cached_connection_state(generation=generation)
            self._cleanup_worker_debug_log_with_retry(generation)
            return

        logger.info("Stopping IBKR subprocess worker", pid=process.pid)

        try:
            connected, _, bound_generation_id, _, _, _ = self._connection_state_snapshot()
            if connected and (
                generation is None or bound_generation_id == generation.generation_id
            ):
                logger.debug("Sending disconnect command to worker")
                await self._execute_command_unlocked({"command": "disconnect"}, timeout=5.0)
        except Exception:
            logger.warning("Diagnostic broker disconnect failed during worker shutdown")
        finally:
            self._clear_cached_connection_state(generation=generation)

        # Signal reader threads to stop
        if generation:
            generation.intentional_stop = True
            generation.stop_event.set()

        # Terminate process with graceful escalation (use run_in_executor to avoid blocking)
        def terminate_process():
            try:
                # Send SIGTERM for graceful shutdown
                logger.debug("Sending SIGTERM to worker process")
                process.terminate()

                # Wait up to 3 seconds for graceful shutdown
                process.wait(timeout=3.0)
                logger.info("Worker terminated gracefully via SIGTERM")

            except subprocess.TimeoutExpired:
                # Worker didn't respond to SIGTERM, escalate to SIGKILL
                logger.warning(
                    "Worker did not respond to SIGTERM within 3s, sending SIGKILL",
                    pid=process.pid,
                )
                process.kill()
                process.wait()
                logger.warning("Worker killed via SIGKILL")

            except Exception:
                logger.error("Error during diagnostic worker process termination")
                # Ensure process is dead
                try:
                    process.kill()
                    process.wait()
                except Exception:
                    pass

        await asyncio.get_event_loop().run_in_executor(None, terminate_process)
        for stream in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

        # Wait for reader threads to finish
        stdout_thread = generation.stdout_thread if generation else self._reader_thread
        stderr_thread = generation.stderr_thread if generation else self._stderr_reader_thread
        if stdout_thread and stdout_thread.is_alive():
            logger.debug("Waiting for reader thread to finish")
            await asyncio.to_thread(stdout_thread.join, 2.0)

        if stderr_thread and stderr_thread.is_alive():
            logger.debug("Waiting for stderr reader thread to finish")
            await asyncio.to_thread(stderr_thread.join, 2.0)
        if (stdout_thread and stdout_thread.is_alive()) or (
            stderr_thread and stderr_thread.is_alive()
        ):
            self._cleanup_worker_debug_log_with_retry(generation)
            raise SubprocessCrashError(
                "Worker reader thread did not stop; refusing generation replacement"
            )

        if self.process is process:
            self.process = None
        with self._connection_state_lock:
            if self._generation is generation:
                self._generation = None
                self._clear_connection_tuple_locked()
                if generation is None or self._gateway_failure_generation_id in (
                    None,
                    generation.generation_id,
                ):
                    self._gateway_api_down_detail = None
                    self._gateway_failure_generation_id = None

        # Ensure the temporary stderr capture never survives its worker.
        self._cleanup_worker_debug_log_with_retry(generation)

        logger.info("IBKR subprocess worker stopped cleanly")

    async def _execute_command(self, command: dict, timeout: float = 30.0) -> dict:
        """Serialize a command against generation start and stop."""
        async with self._lifecycle_lock:
            return await self._execute_command_unlocked(command, timeout)

    async def _execute_command_unlocked(self, command: dict, timeout: float = 30.0) -> dict:
        """
        Send command to subprocess and wait for response.

        Args:
            command: Command dict to send
            timeout: Timeout in seconds

        Returns:
            Response data dict

        Raises:
            SubprocessCrashError: If subprocess is not running
            IBKRTimeoutError: If command times out
            IBKRError: If command fails
        """
        async with self.lock:
            generation = self._generation
            if not generation:
                raise SubprocessCrashError("Subprocess not running")
            with generation.state_lock:
                poison = generation.poisoned_reason
            if poison:
                raise IBKRTransportPoisonedError(f"IBKR worker generation poisoned: {poison}")
            if (
                not self.process
                or generation.process is not self.process
                or self.process.poll() is not None
            ):
                raise SubprocessCrashError("Subprocess not running")

            command_name = command.get("command")
            if not isinstance(command_name, str) or not command_name:
                self._poison_generation(generation, "malformed command name")
                raise IBKRTransportPoisonedError("Malformed command name")
            request_id = uuid.uuid4().hex
            loop = asyncio.get_running_loop()
            pending = _PendingResponse(command_name, loop, loop.create_future())
            envelope = {
                "protocol_version": _TRANSPORT_PROTOCOL_VERSION,
                "generation_id": generation.generation_id,
                "request_id": request_id,
                "command": command_name,
                "params": command.get("params", {}),
            }
            try:
                command_json = json.dumps(envelope) + "\n"
            except (TypeError, ValueError) as exc:
                raise IBKRError(f"Command is not JSON serializable: {exc}") from exc
            with generation.state_lock:
                if generation.poisoned_reason:
                    raise IBKRTransportPoisonedError(
                        "IBKR worker generation poisoned: " f"{generation.poisoned_reason}"
                    )
                generation.pending[request_id] = pending

            # Send command - write directly to stdin (no executor to avoid event loop starvation)
            logger.debug(
                "Sending command to subprocess",
                command=command_name,
                request_id=request_id,
                generation_id=generation.generation_id,
            )

            try:
                generation.process.stdin.write(command_json)
                generation.process.stdin.flush()
            except Exception as e:
                # This caller raises synchronously below, so it will never
                # await its response Future. Remove and cancel that Future
                # before poisoning the remaining generation requests to avoid
                # an unobserved "Future exception was never retrieved".
                with generation.state_lock:
                    generation.pending.pop(request_id, None)
                if not pending.future.done():
                    pending.future.cancel()
                self._poison_generation(generation, f"command write failure: {e}")
                raise IBKRTransportPoisonedError(f"Failed to send command: {e}")

            # Log write confirmation
            logger.debug(
                "Command sent to subprocess",
                command=command_name,
                request_id=request_id,
                bytes_written=len(command_json),
            )

            # Await the request-correlated Future. No executor-backed queue read is
            # used, so cancellation cannot leave an orphan response consumer.
            try:
                response = await asyncio.wait_for(pending.future, timeout=timeout)

            except asyncio.TimeoutError:
                self._poison_generation(
                    generation, f"command timeout: {command_name} ({request_id})"
                )
                logger.error("Command timeout", command=command_name, timeout=timeout)
                raise IBKRTimeoutError(f"Command timeout after {timeout}s")
            except asyncio.CancelledError:
                self._poison_generation(
                    generation, f"command cancelled: {command_name} ({request_id})"
                )
                raise

            if isinstance(response, Exception):
                raise response
            if not isinstance(response, dict):
                self._poison_generation(generation, "non-object routed response")
                raise IBKRTransportPoisonedError("Invalid routed response")

            # Check response status
            if response.get("status") == "error":
                error_msg = response.get("error", "Unknown error")
                error_type = response.get("error_type", "IBKRError")
                detail = response.get("detail") or ""
                requires_restart = bool(response.get("requires_restart"))
                if command_name in {"connect", "disconnect"}:
                    # Treat the worker response as untrusted. Broker connection
                    # libraries may embed account identifiers or credentials in
                    # exception text, so neither logs nor raised exceptions may
                    # reuse the worker's free-form fields.
                    if command_name == "connect":
                        allowed_error_types = {
                            "TimeoutError",
                            "ConnectionError",
                            "NotConnectedError",
                            "GatewayRequiresRestartError",
                        }
                        if error_type not in allowed_error_types:
                            error_type = "BrokerConnectionError"
                        error_msg = (
                            "Diagnostic broker connection timed out"
                            if error_type == "TimeoutError"
                            else "Diagnostic broker connection failed"
                        )
                        detail = (
                            "Gateway restart required after diagnostic connection failure"
                            if requires_restart
                            else ""
                        )
                    else:
                        error_type = "BrokerDisconnectionError"
                        error_msg = "Diagnostic broker disconnect failed"
                        detail = ""
                        requires_restart = False

                logger.error(
                    "Command failed",
                    command=command_name,
                    error=error_msg,
                    error_type=error_type,
                    requires_restart=requires_restart,
                )

                # A timeout reported by the worker is as ambiguous as a timeout
                # observed by the parent: the underlying broker request may
                # still complete after the response is emitted. Never reuse
                # that exact worker generation for another command.
                if error_type == "TimeoutError":
                    diagnostic = f"{error_type}: {error_msg}"
                    if detail and detail != error_msg:
                        diagnostic = f"{diagnostic} ({detail})"
                    self._poison_generation(
                        generation,
                        (
                            "worker-reported broker timeout: "
                            f"{command_name} ({request_id}): {diagnostic}"
                        ),
                    )
                    if requires_restart:
                        raise IBKRTimeoutRequiresGatewayRestartError(diagnostic)
                    raise IBKRTimeoutError(diagnostic)

                if error_type == "BrokerSnapshotAccountMismatchError":
                    self._poison_generation(
                        generation,
                        "worker-reported broker snapshot account mismatch",
                    )
                    raise BrokerSnapshotAccountMismatchError("Broker snapshot account mismatch")

                if requires_restart or error_type == "GatewayRequiresRestartError":
                    message = detail or error_msg
                    gateway_detail = message or "Gateway API layer reported down"
                    self._record_gateway_failure(generation, gateway_detail)
                    raise GatewayRequiresRestartError(message)
                if error_type in {"ConnectionError", "NotConnectedError"}:
                    raise IBKRDisconnectedError(error_msg)

                raise IBKRError(f"{error_type}: {error_msg}")
            if response.get("status") != "success":
                self._poison_generation(generation, "unknown response status")
                raise IBKRTransportPoisonedError("Unknown worker response status")
            if not isinstance(response.get("data"), dict):
                self._poison_generation(generation, "malformed success response data")
                raise IBKRTransportPoisonedError("Worker success response data must be an object")

            logger.debug("Command succeeded", command=command_name, request_id=request_id)
            return response["data"]

    async def connect(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        readonly: bool = True,
        timeout: float = 30.0,
    ) -> bool:
        """
        Connect to IBKR via subprocess.

        Args:
            host: IBKR host
            port: IBKR port
            client_id: Client ID
            readonly: Readonly mode
            timeout: Connection timeout

        Returns:
            True if connected successfully

        Raises:
            SubprocessCrashError: If subprocess crashes
            IBKRError: If connection fails
        """
        if isinstance(port, bool) or port != 4002:
            raise ValueError("Diagnostic client requires IBKR paper port 4002")
        if readonly is not True:
            raise ValueError("Diagnostic client requires readonly exactly true")
        if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
            raise ValueError("Diagnostic client requires a non-negative client ID")
        command = {
            "command": "connect",
            "params": {
                "host": host,
                "port": port,
                "client_id": client_id,
                "readonly": readonly,
                "timeout": timeout,
            },
        }

        logger.info(
            "Connecting diagnostic IBKR subprocess",
            host=host,
            port=port,
            client_id_alias="configured-client-id",
        )

        # ZOMBIE CONNECTION CHECK: Detect zombies before connection attempt
        # Gateway-owned zombies will block API handshakes
        # The caller (robust_connection.py) should handle Gateway restart if connection fails
        logger.debug("Pre-connection zombie check...")
        zombie_count, zombie_msg = await self._check_zombie_connections(port)
        self._zombies_detected_before_connect = zombie_count > 0
        if zombie_count > 0:
            logger.warning(
                "Gateway zombies detected - connection may fail due to blocked handshake",
                zombie_count=zombie_count,
                message=zombie_msg,
            )
            # Note: We proceed here but the caller should restart Gateway if connection fails

        # Record connection attempt timing for debugging
        connection_start = time.time()
        requested_identity = (host, port, client_id, readonly)

        try:
            # Calculate timeout: base TCP timeout + worker's internal sequence + buffer
            # Worker sequence: handshake (15s) + stabilization (0.5s) + accounts (10s) = 25.5s
            # Add 5s buffer for network/processing overhead
            # Total: timeout (TCP) + 25.5s (worker) + 5s (buffer)
            extended_timeout = timeout + self.WORKER_MAX_WAIT + 5.0
            logger.debug(
                "Sending connect command with extended timeout",
                base_timeout=timeout,
                worker_max_wait=self.WORKER_MAX_WAIT,
                extended_timeout=extended_timeout,
            )

            async with self._lifecycle_lock:
                generation = self._generation
                if generation is None:
                    raise SubprocessCrashError("Subprocess generation is unavailable")
                connected_state, _, _, _, _, _ = self._connection_state_snapshot()
                if connected_state:
                    with generation.state_lock:
                        poison = generation.poisoned_reason
                    if poison is not None or generation.process.poll() is not None:
                        raise SubprocessCrashError("Connected worker generation is not healthy")
                    ping_data = await self._execute_command_unlocked(
                        {"command": "ping"}, timeout=5.0
                    )
                    if self._accept_ping_response(ping_data, generation):
                        _, cached_identity, _, _, _, _ = self._connection_state_snapshot()
                        if cached_identity != requested_identity:
                            raise IBKRConnectionConflictError(
                                "Worker is already connected with different "
                                "host/port/client_id/readonly parameters; call stop() "
                                "and start() before connecting with a new identity"
                            )
                        logger.info(
                            "Reusing matching IBKR subprocess connection",
                            host=host,
                            port=port,
                            client_id_alias="configured-client-id",
                            readonly=readonly,
                        )
                        return True
                    logger.warning(
                        "Cached broker connection was stale; reconnecting worker session",
                        host=host,
                        port=port,
                        client_id_alias="configured-client-id",
                    )
                data = await self._execute_command_unlocked(command, timeout=extended_timeout)
                connected = data.get("connected", False)
                with generation.state_lock:
                    poison = generation.poisoned_reason
                    if poison:
                        raise IBKRTransportPoisonedError(
                            f"IBKR worker generation poisoned: {poison}"
                        )
                    with self._connection_state_lock:
                        if self._generation is not generation:
                            raise IBKRTransportPoisonedError(
                                "IBKR worker generation changed during connect"
                            )
                        if connected:
                            self._connected = True
                            self._connection_identity = requested_identity
                            self._connection_generation_id = generation.generation_id
                            self._connection_start_time = datetime.now(timezone.utc)
                            self._last_activity = datetime.now(timezone.utc)
                            self._gateway_api_down_detail = None
                            self._gateway_failure_generation_id = None
                        else:
                            self._clear_connection_tuple_locked()
                            self._gateway_api_down_detail = None
                            self._gateway_failure_generation_id = None

                connection_duration = time.time() - connection_start
                logger.info(
                    "Connect command completed",
                    duration_seconds=f"{connection_duration:.2f}",
                )

        except GatewayRequiresRestartError:
            connection_duration = time.time() - connection_start
            logger.error(
                "Connection failed - Gateway restart required",
                duration_seconds=f"{connection_duration:.2f}",
            )
            raise

        accounts = data.get("accounts", [])
        server_version = data.get("server_version")

        connected_state, _, _, _, _, _ = self._connection_state_snapshot()
        logger.info(
            "Connected to IBKR via subprocess",
            connected=connected_state,
            managed_account_count=len(accounts),
            server_version=server_version,
            duration_seconds=f"{time.time() - connection_start:.2f}",
        )

        # Defense-in-depth: log the readonly assumption. ib_async does not
        # expose a post-handshake "is readonly" flag, so we cannot verify
        # programmatically here. The authoritative read-only enforcement is
        # IBC's ReadOnlyApi=yes (verified at startup by START_TRADER.sh and
        # START_TRADER.sh). This client always passes readonly=True.
        if connected_state:
            logger.info(
                "IBKR client connection: readonly flag was requested",
                readonly_requested=readonly,
                note=(
                    "Gateway-side ReadOnlyApi=yes is the authoritative "
                    "order-placement safeguard."
                ),
            )

        return connected_state

    async def _check_zombie_connections(self, port: int) -> tuple[int, str]:
        """
        Check for zombie CLOSE_WAIT connections that block API handshakes.

        Returns:
            tuple: (zombie_count, error_message)
        """
        try:
            lsof_executable = _trusted_lsof_executable()
            lsof_env = {
                name: os.environ[name] for name in _WORKER_ENV_ALLOWLIST if name in os.environ
            }

            # Use one approved absolute lsof binary with no inherited secrets,
            # PATH lookup, working directory, or file descriptors.
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        str(lsof_executable),
                        "-nP",
                        f"-iTCP:{port}",
                        "-sTCP:CLOSE_WAIT",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    close_fds=True,
                    cwd="/",
                    env=lsof_env,
                ),
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if result.returncode == 1 and not stdout and not stderr:
                return 0, "No zombies detected"
            if result.returncode != 0 or stderr:
                raise IBKRError("Trusted lsof zombie check failed")
            if not stdout:
                return 0, "No zombies detected"

            # Count zombie connections (skip header line)
            lines = [
                line
                for line in result.stdout.split("\n")
                if line.strip() and not line.startswith("COMMAND")
            ]
            zombie_count = len(lines)

            if zombie_count > 0:
                error_msg = f"Found {zombie_count} CLOSE_WAIT zombie connection(s) on port {port}"
                logger.warning("Zombie connections detected", count=zombie_count, port=port)
                return zombie_count, error_msg

            return 0, "No zombies detected"

        except IBKRError:
            raise
        except (OSError, subprocess.SubprocessError) as exc:
            raise IBKRError("Trusted lsof zombie check failed") from exc

    @property
    def zombies_detected_before_connect(self) -> bool:
        """Returns True if zombie connections were detected before the last connect attempt."""
        return self._zombies_detected_before_connect

    async def get_accounts(self) -> list[str]:
        """Get managed accounts"""
        data = await self._execute_command({"command": "get_accounts"})
        return data.get("accounts", [])

    async def get_positions(self) -> list[dict]:
        """Get current positions"""
        data = await self._execute_command({"command": "get_positions"})
        return data.get("positions", [])

    async def get_account_summary(self) -> dict:
        """Get account summary"""
        data = await self._execute_command({"command": "get_account_summary"})
        return data.get("summary", {})

    @staticmethod
    def _masked_account(account: object) -> str:
        normalized = str(account or "").strip()
        return "****" if len(normalized) <= 4 else f"***{normalized[-4:]}"

    @staticmethod
    def _strict_decimal(value: Any, field: str) -> Decimal:
        if not isinstance(value, str) or value != value.strip() or not value:
            raise ValueError(f"{field} must be a canonical decimal string")
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"{field} must be a canonical decimal string") from exc
        if not parsed.is_finite():
            raise ValueError(f"{field} must be finite")
        canonical = "0" if parsed == 0 else format(parsed.normalize(), "f")
        if value != canonical:
            raise ValueError(f"{field} is not canonical")
        return parsed

    @staticmethod
    def _strict_timestamp(value: Any, field: str) -> datetime:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{field} must be timezone-aware")
        return parsed

    @staticmethod
    def _strict_identifier(
        value: Any,
        field: str,
        *,
        optional: bool = False,
        allow_zero: bool = False,
    ) -> Optional[int]:
        if optional and value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (value == 0 and not allow_zero)
        ):
            raise ValueError(f"{field} must be a positive integer")
        return value

    def _snapshot_fail(
        self,
        generation: Optional[_WorkerGeneration],
        reason: str,
    ) -> None:
        if generation:
            self._poison_generation(generation, reason)
        raise IBKRTransportPoisonedError(reason)

    def _require_snapshot_generation_binding(
        self,
        generation: _WorkerGeneration,
        *,
        expected_identity: Optional[tuple[str, int, int, bool]] = None,
        poison_on_mismatch: bool,
    ) -> tuple[str, int, int, bool]:
        """Require one unpoisoned diagnostic session at a snapshot boundary.

        The caller holds ``_lifecycle_lock``.  The nested locks match the
        transport's global state order so a reader-thread poison and its
        connection-state clear are observed atomically.
        """
        with generation.state_lock:
            poisoned = generation.poisoned_reason is not None
            with self._connection_state_lock:
                identity = self._connection_identity
                bound = (
                    self._generation is generation
                    and self._connected
                    and identity is not None
                    and identity[1] == 4002
                    and identity[2] > 0
                    and identity[3] is True
                    and self._connection_generation_id == generation.generation_id
                    and (expected_identity is None or identity == expected_identity)
                )

        if poisoned:
            raise IBKRTransportPoisonedError("Broker snapshot worker generation is poisoned")
        if not bound:
            if poison_on_mismatch:
                self._snapshot_fail(
                    generation,
                    "worker generation changed during broker snapshot",
                )
            raise IBKRDisconnectedError(
                "Broker snapshot requires the validated diagnostic connection"
            )
        return cast(tuple[str, int, int, bool], identity)

    def _validate_snapshot_contract(self, value: Any) -> tuple[int, str]:
        expected_keys = {
            "con_id",
            "symbol",
            "local_symbol",
            "security_type",
            "currency",
            "exchange",
            "primary_exchange",
            "trading_class",
        }
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError("broker snapshot contract schema is invalid")
        con_id = self._strict_identifier(value["con_id"], "contract con_id")
        symbol = value["symbol"]
        local_symbol = value["local_symbol"]
        if (
            not isinstance(symbol, str)
            or not symbol
            or symbol != symbol.strip().upper()
            or local_symbol != symbol
        ):
            raise ValueError("broker snapshot contract alias is unsupported")
        if value["security_type"] != "STK":
            raise ValueError("broker snapshot contract is not STK")
        if value["currency"] != "USD":
            raise ValueError("broker snapshot contract is not USD")
        if value["exchange"] != "SMART":
            raise ValueError("broker snapshot contract exchange is not SMART")
        for field_name in ("primary_exchange", "trading_class"):
            if not isinstance(value[field_name], str) or not value[field_name].strip():
                raise ValueError("broker snapshot contract identity is incomplete")
        return cast(int, con_id), symbol

    @staticmethod
    def _validate_unavailable(
        record: dict,
        optional_fields: set[str],
    ) -> dict[str, str]:
        unavailable = record.get("unavailable")
        if not isinstance(unavailable, dict) or not set(unavailable).issubset(optional_fields):
            raise ValueError("broker snapshot unavailable-field schema is invalid")
        if any(not isinstance(reason, str) or not reason for reason in unavailable.values()):
            raise ValueError("broker snapshot unavailable reason is invalid")
        for field_name in optional_fields:
            if record[field_name] is None and field_name not in unavailable:
                raise ValueError("broker snapshot silently omitted optional evidence")
            if record[field_name] is not None and field_name in unavailable:
                raise ValueError("broker snapshot optional evidence is contradictory")
        return cast(dict[str, str], unavailable)

    def _validate_broker_snapshot(
        self,
        expected_account: str,
        data: dict,
        generation: Optional[_WorkerGeneration],
        max_age_seconds: float,
    ) -> dict:
        top_keys = {
            "snapshot_schema_version",
            "account",
            "broker_time_before",
            "broker_time_after",
            "retrieved_at",
            "positions",
            "balances",
            "open_orders",
            "executions",
            "execution_scope",
        }
        try:
            if set(data) != top_keys:
                raise ValueError("broker snapshot top-level schema is invalid")
            if (
                isinstance(data["snapshot_schema_version"], bool)
                or data["snapshot_schema_version"] != _BROKER_SNAPSHOT_SCHEMA_VERSION
            ):
                raise ValueError("broker snapshot schema version is unsupported")

            account = data["account"]
            if not isinstance(account, str) or not account:
                raise ValueError("broker snapshot account identity is malformed")
            if account != expected_account:
                reason = "broker snapshot account identity mismatch"
                if generation:
                    self._poison_generation(generation, reason)
                raise BrokerSnapshotAccountMismatchError(
                    "Broker snapshot account mismatch "
                    f"(expected={self._masked_account(expected_account)}, "
                    f"connected={self._masked_account(account)})"
                )

            broker_before = self._strict_timestamp(data["broker_time_before"], "broker_time_before")
            broker_after = self._strict_timestamp(data["broker_time_after"], "broker_time_after")
            retrieved_at = self._strict_timestamp(data["retrieved_at"], "retrieved_at")
            if broker_after < broker_before:
                raise ValueError("broker snapshot broker times are reversed")
            broker_window = (broker_after - broker_before).total_seconds()
            if (
                not math.isfinite(broker_window)
                or broker_window > _BROKER_SNAPSHOT_MAX_WINDOW_SECONDS
            ):
                raise ValueError("broker snapshot collection window is unbounded")
            age = (
                datetime.now(timezone.utc) - retrieved_at.astimezone(timezone.utc)
            ).total_seconds()
            if age < -5 or age > max_age_seconds:
                raise ValueError("broker snapshot retrieval time is outside freshness bounds")
            if any(
                abs((broker_time - retrieved_at).total_seconds())
                > _BROKER_SNAPSHOT_MAX_CLOCK_SKEW_SECONDS
                for broker_time in (broker_before, broker_after)
            ):
                raise ValueError("broker snapshot clock skew exceeds safety bound")

            positions = data["positions"]
            if not isinstance(positions, list):
                raise ValueError("broker snapshot positions are not a list")
            seen_contract_ids: set[int] = set()
            seen_symbols: set[str] = set()
            position_keys = {"account", "contract", "quantity", "avg_cost"}
            for position in positions:
                if not isinstance(position, dict) or set(position) != position_keys:
                    raise ValueError("broker snapshot position schema is invalid")
                if position["account"] != expected_account:
                    raise ValueError("broker snapshot position account is inconsistent")
                con_id, symbol = self._validate_snapshot_contract(position["contract"])
                if con_id in seen_contract_ids or symbol in seen_symbols:
                    raise ValueError("broker snapshot position identity is duplicated")
                seen_contract_ids.add(con_id)
                seen_symbols.add(symbol)
                if self._strict_decimal(position["quantity"], "position quantity") == 0:
                    raise ValueError("broker snapshot position quantity is zero")
                if self._strict_decimal(position["avg_cost"], "position avg_cost") < 0:
                    raise ValueError("broker snapshot position average cost is negative")

            balances = data["balances"]
            if not isinstance(balances, list):
                raise ValueError("broker snapshot balances are not a list")
            seen_balances: set[tuple[str, str]] = set()
            present_tags: set[str] = set()
            for balance in balances:
                if not isinstance(balance, dict) or set(balance) != {
                    "tag",
                    "currency",
                    "value",
                }:
                    raise ValueError("broker snapshot balance schema is invalid")
                tag = balance["tag"]
                currency = balance["currency"]
                if tag not in _BROKER_SNAPSHOT_BALANCE_TAGS:
                    raise ValueError("broker snapshot balance tag is not allowed")
                if not isinstance(currency, str) or not currency:
                    raise ValueError("broker snapshot balance currency is invalid")
                identity = (tag, currency)
                if identity in seen_balances:
                    raise ValueError("broker snapshot balance identity is duplicated")
                seen_balances.add(identity)
                present_tags.add(tag)
                self._strict_decimal(balance["value"], "balance value")
            if not _BROKER_SNAPSHOT_REQUIRED_BALANCE_TAGS.issubset(present_tags):
                raise ValueError("broker snapshot required balances are missing")

            self._validate_snapshot_orders(data["open_orders"], expected_account)
            execution_scope = data["execution_scope"]
            if not isinstance(execution_scope, dict) or set(execution_scope) != {
                "kind",
                "start_at",
                "end_at",
            }:
                raise ValueError("broker snapshot execution scope schema is invalid")
            if execution_scope["kind"] != "bounded_execution_filter":
                raise ValueError("broker snapshot execution scope is unsupported")
            execution_start = self._strict_timestamp(
                execution_scope["start_at"], "execution scope start"
            )
            execution_end = self._strict_timestamp(execution_scope["end_at"], "execution scope end")
            expected_execution_start = broker_before.replace(microsecond=0) - timedelta(
                seconds=_BROKER_EXECUTION_LOOKBACK_SECONDS
            )
            if execution_start != expected_execution_start:
                raise ValueError("broker snapshot execution scope does not match the wire filter")
            if not broker_before <= execution_end <= broker_after:
                raise ValueError(
                    "broker snapshot execution scope end is outside broker collection bounds"
                )
            self._validate_snapshot_executions(
                data["executions"],
                expected_account,
                execution_start,
                execution_end,
            )
            return data
        except BrokerSnapshotAccountMismatchError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            self._snapshot_fail(generation, str(exc))
        raise AssertionError("unreachable")

    def _validate_snapshot_orders(self, orders: Any, account: str) -> None:
        if not isinstance(orders, list):
            raise ValueError("broker snapshot open orders are not a list")
        keys = {
            "account",
            "broker_order_id",
            "permanent_id",
            "client_id",
            "contract",
            "side",
            "status",
            "order_type",
            "time_in_force",
            "total_quantity",
            "filled_quantity",
            "remaining_quantity",
            "limit_price",
            "stop_price",
            "avg_fill_price",
            "last_status_at",
            "unavailable",
        }
        optional = {
            "permanent_id",
            "limit_price",
            "stop_price",
            "avg_fill_price",
            "last_status_at",
        }
        seen: set[tuple[int, int]] = set()
        for order in orders:
            if not isinstance(order, dict) or set(order) != keys:
                raise ValueError("broker snapshot open-order schema is invalid")
            self._validate_unavailable(order, optional)
            if order["account"] != account:
                raise ValueError("broker snapshot order account is inconsistent")
            order_id = self._strict_identifier(order["broker_order_id"], "broker order ID")
            client_id = self._strict_identifier(order["client_id"], "client ID", allow_zero=True)
            order_identity = (cast(int, client_id), cast(int, order_id))
            if order_identity in seen:
                raise ValueError("broker snapshot order identity is duplicated")
            seen.add(order_identity)
            self._strict_identifier(order["permanent_id"], "permanent ID", optional=True)
            self._validate_snapshot_contract(order["contract"])
            if order["side"] not in {"BUY", "SELL"}:
                raise ValueError("broker snapshot order side is invalid")
            for field_name in ("status", "order_type", "time_in_force"):
                if not isinstance(order[field_name], str) or not order[field_name]:
                    raise ValueError("broker snapshot order text evidence is missing")
            total = self._strict_decimal(order["total_quantity"], "order total quantity")
            filled = self._strict_decimal(order["filled_quantity"], "order filled quantity")
            remaining = self._strict_decimal(
                order["remaining_quantity"], "order remaining quantity"
            )
            if total <= 0 or filled < 0 or remaining < 0:
                raise ValueError("broker snapshot order quantities are invalid")
            if filled + remaining != total:
                raise ValueError("broker snapshot order quantities are inconsistent")
            for field_name in ("limit_price", "stop_price", "avg_fill_price"):
                if order[field_name] is not None:
                    if self._strict_decimal(order[field_name], field_name) <= 0:
                        raise ValueError("broker snapshot order price evidence is not positive")
            if order["last_status_at"] is not None:
                self._strict_timestamp(order["last_status_at"], "last_status_at")

    def _validate_snapshot_executions(
        self,
        executions: Any,
        account: str,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        if not isinstance(executions, list):
            raise ValueError("broker snapshot executions are not a list")
        keys = {
            "account",
            "execution_id",
            "broker_order_id",
            "permanent_id",
            "client_id",
            "contract",
            "side",
            "quantity",
            "price",
            "average_price",
            "executed_at",
            "execution_exchange",
            "commission",
            "commission_currency",
            "realized_pnl",
            "unavailable",
        }
        optional = {
            "broker_order_id",
            "permanent_id",
            "commission",
            "commission_currency",
            "realized_pnl",
        }
        seen: set[str] = set()
        for execution in executions:
            if not isinstance(execution, dict) or set(execution) != keys:
                raise ValueError("broker snapshot execution schema is invalid")
            self._validate_unavailable(execution, optional)
            if execution["account"] != account:
                raise ValueError("broker snapshot execution account is inconsistent")
            execution_id = execution["execution_id"]
            if not isinstance(execution_id, str) or not execution_id or execution_id in seen:
                raise ValueError("broker snapshot execution identity is invalid")
            seen.add(execution_id)
            self._strict_identifier(
                execution["broker_order_id"], "execution order ID", optional=True
            )
            self._strict_identifier(
                execution["permanent_id"], "execution permanent ID", optional=True
            )
            self._strict_identifier(execution["client_id"], "execution client ID", allow_zero=True)
            self._validate_snapshot_contract(execution["contract"])
            if execution["side"] not in {"BUY", "SELL"}:
                raise ValueError("broker snapshot execution side is invalid")
            for field_name in ("quantity", "price", "average_price"):
                if self._strict_decimal(execution[field_name], field_name) <= 0:
                    raise ValueError("broker snapshot execution numeric evidence is invalid")
            executed_at = self._strict_timestamp(execution["executed_at"], "executed_at")
            if executed_at < window_start or executed_at > window_end:
                raise ValueError("broker snapshot execution timestamp is outside requested window")
            if (
                not isinstance(execution["execution_exchange"], str)
                or not execution["execution_exchange"]
            ):
                raise ValueError("broker snapshot execution exchange is missing")
            for field_name in ("commission", "realized_pnl"):
                if execution[field_name] is not None:
                    self._strict_decimal(execution[field_name], field_name)
            if execution["commission_currency"] is not None and (
                not isinstance(execution["commission_currency"], str)
                or not execution["commission_currency"]
            ):
                raise ValueError("broker snapshot commission currency is invalid")

    async def get_broker_snapshot(
        self,
        expected_account: str,
        *,
        max_age_seconds: float = 30.0,
    ) -> dict:
        """Return one strictly validated, read-only broker evidence snapshot."""
        normalized_account = str(expected_account or "").strip()
        if not normalized_account:
            raise ValueError("Expected broker account is required")
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(float(max_age_seconds))
            or max_age_seconds <= 0
            or max_age_seconds > _BROKER_SNAPSHOT_MAX_AGE_SECONDS
        ):
            raise ValueError("Snapshot freshness bound must be finite and at most 300 seconds")
        async with self._lifecycle_lock:
            generation = self._generation
            if generation is None:
                raise SubprocessCrashError("Subprocess generation is unavailable")
            try:
                identity = self._require_snapshot_generation_binding(
                    generation,
                    poison_on_mismatch=False,
                )
                data = await self._execute_command_unlocked(
                    {
                        "command": "get_broker_snapshot",
                        "params": {"expected_account": normalized_account},
                    },
                    timeout=30.0,
                )
                self._require_snapshot_generation_binding(
                    generation,
                    expected_identity=identity,
                    poison_on_mismatch=True,
                )
                validated = self._validate_broker_snapshot(
                    normalized_account,
                    data,
                    generation,
                    float(max_age_seconds),
                )
                self._require_snapshot_generation_binding(
                    generation,
                    expected_identity=identity,
                    poison_on_mismatch=True,
                )
                return validated
            finally:
                # Broker evidence must not leave diagnostic stderr artifacts on
                # disk, whether validation succeeds or fails closed.
                self._cleanup_worker_debug_log(generation, required=True)

    def _broker_safety_contract(self, value: Any) -> BrokerSafetyContract:
        self._validate_snapshot_contract(value)
        return BrokerSafetyContract(
            con_id=value["con_id"],
            symbol=value["symbol"],
            local_symbol=value["local_symbol"],
            security_type=value["security_type"],
            currency=value["currency"],
            exchange=value["exchange"],
            primary_exchange=value["primary_exchange"],
            trading_class=value["trading_class"],
        )

    def _broker_safety_open_orders(
        self,
        value: Any,
        expected_account: str,
    ) -> tuple[BrokerSafetyOpenOrder, ...]:
        self._validate_snapshot_orders(value, expected_account)
        return tuple(
            BrokerSafetyOpenOrder(
                broker_order_id=order["broker_order_id"],
                permanent_id=order["permanent_id"],
                client_id=order["client_id"],
                contract=self._broker_safety_contract(order["contract"]),
                side=order["side"],
                status=order["status"],
                order_type=order["order_type"],
                time_in_force=order["time_in_force"],
                total_quantity=self._strict_decimal(
                    order["total_quantity"], "order total quantity"
                ),
                filled_quantity=self._strict_decimal(
                    order["filled_quantity"], "order filled quantity"
                ),
                remaining_quantity=self._strict_decimal(
                    order["remaining_quantity"], "order remaining quantity"
                ),
                limit_price=(
                    self._strict_decimal(order["limit_price"], "limit_price")
                    if order["limit_price"] is not None
                    else None
                ),
                stop_price=(
                    self._strict_decimal(order["stop_price"], "stop_price")
                    if order["stop_price"] is not None
                    else None
                ),
                average_fill_price=(
                    self._strict_decimal(order["avg_fill_price"], "avg_fill_price")
                    if order["avg_fill_price"] is not None
                    else None
                ),
                last_status_at=(
                    self._strict_timestamp(order["last_status_at"], "last_status_at")
                    if order["last_status_at"] is not None
                    else None
                ),
            )
            for order in value
        )

    @staticmethod
    def _validated_broker_runtime_context(
        runtime_context: object,
    ) -> tuple[RuntimeSafetyContext, str, str, str, str]:
        """Revalidate the one registered runtime/IBC/account binding."""

        context = assert_validated_runtime_safety_context(runtime_context)
        expected_account = context.expected_account_for_provider
        if (
            not isinstance(expected_account, str)
            or expected_account != expected_account.strip()
            or not is_supported_paper_account_identifier(
                expected_account,
                environment=context.runtime_contract.environment,
            )
        ):
            raise ValueError("validated runtime does not identify a paper account")
        contract = context.runtime_contract
        account_scope = getattr(contract, "safety_account_scope", None)
        if not isinstance(account_scope, str) or not _OPAQUE_ACCOUNT_SCOPE_RE.fullmatch(
            account_scope
        ):
            raise ValueError("validated runtime lacks an opaque safety account scope")
        if len(set(account_scope.removeprefix("acct_v1_"))) == 1:
            raise ValueError("validated runtime safety account scope is a placeholder")
        connection = context.diagnostic_connection
        if (
            getattr(contract, "execution_mode", None) != "paper"
            or getattr(contract, "ibkr_host", None) != connection.host
            or getattr(contract, "ibkr_port", None) != connection.port
            or getattr(contract, "ibkr_readonly", None) is not True
            or connection.port != 4002
            or connection.readonly is not True
            or getattr(contract, "account_alias", None) != mask_account_identifier(expected_account)
        ):
            raise ValueError("validated runtime paper/read-only identity is inconsistent")
        current_ibc_hash = validate_ibc_safety_config(
            context.project_root / "config" / "ibc" / "config.ini"
        )
        if not hmac.compare_digest(current_ibc_hash, context.ibc_config_hash):
            raise ValueError("validated IBC safety configuration changed")
        runtime_fingerprint = contract.fingerprint
        if not isinstance(runtime_fingerprint, str) or not runtime_fingerprint:
            raise ValueError("validated runtime fingerprint is unavailable")
        return (
            context,
            expected_account,
            account_scope,
            current_ibc_hash,
            runtime_fingerprint,
        )

    def _validate_broker_safety_snapshot(
        self,
        *,
        expected_account: str,
        requested_symbol: str,
        capability: _BrokerSnapshotCapability,
        data: dict,
        generation: _WorkerGeneration,
        connection_identity: tuple[str, int, int, bool],
        max_age_seconds: float,
    ) -> BrokerSafetySnapshot:
        top_keys = {
            "safety_snapshot_schema_version",
            "account",
            "requested_symbol",
            "broker_time_before",
            "broker_time_after",
            "retrieved_at",
            "positions",
            "open_orders",
            "positions_complete",
            "open_orders_complete",
            "open_orders_all_clients",
            "open_orders_stable",
            "unknown_order_count",
        }
        try:
            if not isinstance(data, dict) or set(data) != top_keys:
                raise ValueError("broker safety snapshot top-level schema is invalid")
            if (
                type(data["safety_snapshot_schema_version"]) is not int
                or data["safety_snapshot_schema_version"] != _BROKER_SAFETY_SNAPSHOT_SCHEMA_VERSION
            ):
                raise ValueError("broker safety snapshot schema version is unsupported")
            account = data["account"]
            if not isinstance(account, str) or not account:
                raise ValueError("broker safety snapshot account identity is malformed")
            if account != expected_account:
                reason = "broker safety snapshot account identity mismatch"
                self._poison_generation(generation, reason)
                raise BrokerSnapshotAccountMismatchError(
                    "Broker safety snapshot account mismatch "
                    f"(expected={self._masked_account(expected_account)}, "
                    f"connected={self._masked_account(account)})"
                )
            if data["requested_symbol"] != requested_symbol:
                raise ValueError("broker safety snapshot requested symbol mismatch")
            if any(
                data[field_name] is not True
                for field_name in (
                    "positions_complete",
                    "open_orders_complete",
                    "open_orders_all_clients",
                    "open_orders_stable",
                )
            ):
                raise ValueError("broker safety snapshot completeness is unproven")
            if type(data["unknown_order_count"]) is not int or data["unknown_order_count"] != 0:
                raise ValueError("broker safety snapshot contains unknown orders")

            broker_before = self._strict_timestamp(data["broker_time_before"], "broker_time_before")
            broker_after = self._strict_timestamp(data["broker_time_after"], "broker_time_after")
            retrieved_at = self._strict_timestamp(data["retrieved_at"], "retrieved_at")
            if broker_after < broker_before:
                raise ValueError("broker safety snapshot broker times are reversed")
            broker_window = (broker_after - broker_before).total_seconds()
            if (
                not math.isfinite(broker_window)
                or broker_window > _BROKER_SNAPSHOT_MAX_WINDOW_SECONDS
            ):
                raise ValueError("broker safety snapshot collection window is unbounded")
            age = (
                datetime.now(timezone.utc) - retrieved_at.astimezone(timezone.utc)
            ).total_seconds()
            if age < -5 or age > max_age_seconds:
                raise ValueError("broker safety snapshot is outside freshness bounds")
            if any(
                abs((broker_time - retrieved_at).total_seconds())
                > _BROKER_SNAPSHOT_MAX_CLOCK_SKEW_SECONDS
                for broker_time in (broker_before, broker_after)
            ):
                raise ValueError("broker safety snapshot clock skew exceeds safety bound")

            raw_positions = data["positions"]
            if not isinstance(raw_positions, list):
                raise ValueError("broker safety snapshot positions are not a list")
            position_keys = {"account", "contract", "quantity"}
            positions = []
            seen_contract_ids: set[int] = set()
            seen_symbols: set[str] = set()
            for position in raw_positions:
                if not isinstance(position, dict) or set(position) != position_keys:
                    raise ValueError("broker safety snapshot position schema is invalid")
                if position["account"] != expected_account:
                    raise ValueError("broker safety snapshot position account is inconsistent")
                contract = self._broker_safety_contract(position["contract"])
                if contract.con_id in seen_contract_ids or contract.symbol in seen_symbols:
                    raise ValueError("broker safety snapshot position identity is duplicated")
                seen_contract_ids.add(contract.con_id)
                seen_symbols.add(contract.symbol)
                quantity = self._strict_decimal(position["quantity"], "position quantity")
                if quantity == 0:
                    raise ValueError("broker safety snapshot position quantity is zero")
                positions.append(BrokerSafetyPosition(contract=contract, quantity=quantity))

            matching = [
                position for position in positions if position.contract.symbol == requested_symbol
            ]
            if len(matching) != 1:
                raise ValueError(
                    "broker safety snapshot requested symbol is not one exact held position"
                )
            open_orders = self._broker_safety_open_orders(data["open_orders"], expected_account)
            snapshot = _produce_broker_safety_snapshot(
                capability=capability,
                observed_at=retrieved_at,
                broker_time_before=broker_before,
                broker_time_after=broker_after,
                snapshot_id=f"broker-safety-v1-{uuid.uuid4().hex}",
                source="ibkr-subprocess-safety-v1",
                requested_contract=matching[0].contract,
                positions=tuple(positions),
                open_orders=open_orders,
            )
            assert_producer_owned_broker_safety_snapshot(snapshot)
            return snapshot
        except BrokerSnapshotAccountMismatchError:
            raise
        except Exception as exc:
            self._snapshot_fail(generation, str(exc))
        raise AssertionError("unreachable")

    async def _get_broker_safety_snapshot_unlocked(
        self,
        runtime_context: RuntimeSafetyContext,
        normalized_symbol: str,
        *,
        max_age_seconds: float,
    ) -> tuple[
        BrokerSafetySnapshot,
        _WorkerGeneration,
        tuple[str, int, int, bool],
        tuple[RuntimeSafetyContext, str, str, str, str],
    ]:
        """Produce one snapshot while the caller holds ``_lifecycle_lock``."""

        generation = self._generation
        if generation is None:
            raise SubprocessCrashError("Subprocess generation is unavailable")
        try:
            try:
                validated_context = self._validated_broker_runtime_context(runtime_context)
                (
                    context,
                    normalized_account,
                    account_scope,
                    ibc_config_hash,
                    runtime_fingerprint,
                ) = validated_context
            except Exception as exc:
                try:
                    self._snapshot_fail(
                        generation,
                        "validated broker runtime context is unavailable",
                    )
                except IBKRTransportPoisonedError as poisoned:
                    raise poisoned from exc
            identity = self._require_snapshot_generation_binding(
                generation,
                poison_on_mismatch=False,
            )
            expected_identity = (
                context.diagnostic_connection.host,
                context.diagnostic_connection.port,
                context.diagnostic_connection.client_id,
                context.diagnostic_connection.readonly,
            )
            if identity != expected_identity:
                self._snapshot_fail(
                    generation,
                    "broker transport differs from validated runtime context",
                )
            data = await self._execute_command_unlocked(
                {
                    "command": "get_broker_safety_snapshot",
                    "params": {
                        "expected_account": normalized_account,
                        "requested_symbol": normalized_symbol,
                    },
                },
                timeout=30.0,
            )
            self._require_snapshot_generation_binding(
                generation,
                expected_identity=expected_identity,
                poison_on_mismatch=True,
            )
            try:
                revalidated = self._validated_broker_runtime_context(runtime_context)
            except Exception as exc:
                try:
                    self._snapshot_fail(
                        generation,
                        "validated broker runtime context changed during snapshot",
                    )
                except IBKRTransportPoisonedError as poisoned:
                    raise poisoned from exc
            if revalidated[1:] != validated_context[1:]:
                self._snapshot_fail(
                    generation,
                    "validated broker runtime context changed during snapshot",
                )
            try:
                capability = _issue_broker_snapshot_capability(
                    context,
                    connection_identity=identity,
                    transport_generation=generation.generation_id,
                    requested_symbol=normalized_symbol,
                )
            except Exception as exc:
                try:
                    self._snapshot_fail(
                        generation,
                        "broker snapshot capability issuance failed",
                    )
                except IBKRTransportPoisonedError as poisoned:
                    raise poisoned from exc
            snapshot = self._validate_broker_safety_snapshot(
                expected_account=normalized_account,
                requested_symbol=normalized_symbol,
                capability=capability,
                data=data,
                generation=generation,
                connection_identity=identity,
                max_age_seconds=float(max_age_seconds),
            )
            self._require_snapshot_generation_binding(
                generation,
                expected_identity=expected_identity,
                poison_on_mismatch=True,
            )
            try:
                final_context = self._validated_broker_runtime_context(runtime_context)
            except Exception as exc:
                try:
                    self._snapshot_fail(
                        generation,
                        "validated broker runtime context changed after snapshot",
                    )
                except IBKRTransportPoisonedError as poisoned:
                    raise poisoned from exc
            if final_context[1:] != revalidated[1:]:
                self._snapshot_fail(
                    generation,
                    "validated broker runtime context changed after snapshot",
                )
            assert_producer_owned_broker_safety_snapshot(snapshot)
            return snapshot, generation, expected_identity, final_context
        except Exception:
            raise

    async def run_with_locked_broker_safety_snapshot(
        self,
        runtime_context: RuntimeSafetyContext,
        requested_symbol: str,
        async_callback: Callable[[BrokerSafetySnapshot], Awaitable[_BrokerCallbackResult]],
        *,
        max_age_seconds: float = 30.0,
    ) -> _BrokerCallbackResult:
        """Run one async finalization callback under the broker lifecycle lock."""

        if not isinstance(requested_symbol, str):
            raise ValueError("Broker safety symbol must be a string")
        normalized_symbol = requested_symbol.strip().upper()
        if not _BROKER_SAFETY_SYMBOL_RE.fullmatch(normalized_symbol):
            raise ValueError("Broker safety symbol is malformed")
        if not callable(async_callback):
            raise TypeError("Broker safety callback must be callable")
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(float(max_age_seconds))
            or max_age_seconds <= 0
            or max_age_seconds > 30
        ):
            raise ValueError("Broker safety freshness bound must be finite and at most 30 seconds")

        async with self._lifecycle_lock:
            generation = self._generation
            if generation is None:
                raise SubprocessCrashError("Subprocess generation is unavailable")
            snapshot: Optional[BrokerSafetySnapshot] = None
            expected_identity: Optional[tuple[str, int, int, bool]] = None
            final_context: Optional[tuple[RuntimeSafetyContext, str, str, str, str]] = None
            try:
                (
                    snapshot,
                    generation,
                    expected_identity,
                    final_context,
                ) = await self._get_broker_safety_snapshot_unlocked(
                    runtime_context,
                    normalized_symbol,
                    max_age_seconds=float(max_age_seconds),
                )
                callback_result = async_callback(snapshot)
                if not inspect.isawaitable(callback_result):
                    raise TypeError("Broker safety callback must return an awaitable")
                return await callback_result
            finally:
                if snapshot is not None and expected_identity is not None:
                    self._require_snapshot_generation_binding(
                        generation,
                        expected_identity=expected_identity,
                        poison_on_mismatch=True,
                    )
                    assert_producer_owned_broker_safety_snapshot(snapshot)
                    try:
                        current_context = self._validated_broker_runtime_context(runtime_context)
                    except Exception as exc:
                        try:
                            self._snapshot_fail(
                                generation,
                                "validated broker runtime context changed during finalization",
                            )
                        except IBKRTransportPoisonedError as poisoned:
                            raise poisoned from exc
                    if final_context is None or current_context[1:] != final_context[1:]:
                        self._snapshot_fail(
                            generation,
                            "validated broker runtime context changed during finalization",
                        )
                self._cleanup_worker_debug_log(generation, required=True)

    async def get_broker_safety_snapshot(
        self,
        runtime_context: RuntimeSafetyContext,
        requested_symbol: str,
        *,
        max_age_seconds: float = 30.0,
    ) -> BrokerSafetySnapshot:
        """Return one snapshot through the lifecycle-held finalization boundary."""

        async def return_snapshot(snapshot: BrokerSafetySnapshot) -> BrokerSafetySnapshot:
            return snapshot

        return await self.run_with_locked_broker_safety_snapshot(
            runtime_context,
            requested_symbol,
            return_snapshot,
            max_age_seconds=max_age_seconds,
        )

    def _validate_broker_contract_safety_snapshot(
        self,
        *,
        expected_account: str,
        requested_symbol: str,
        capability: _BrokerContractSnapshotCapability,
        data: dict,
        generation: _WorkerGeneration,
        max_age_seconds: float,
    ) -> BrokerContractSafetySnapshot:
        expected_keys = {
            "contract_safety_snapshot_schema_version",
            "account",
            "requested_symbol",
            "broker_time_before",
            "broker_time_after",
            "retrieved_at",
            "qualified_contract",
        }
        try:
            if not isinstance(data, dict) or set(data) != expected_keys:
                raise ValueError("broker contract safety snapshot schema is invalid")
            if (
                type(data["contract_safety_snapshot_schema_version"]) is not int
                or data["contract_safety_snapshot_schema_version"]
                != _BROKER_CONTRACT_SAFETY_SNAPSHOT_SCHEMA_VERSION
            ):
                raise ValueError("broker contract safety snapshot version is unsupported")
            account = data["account"]
            if not isinstance(account, str) or not account:
                raise ValueError("broker contract safety account identity is malformed")
            if account != expected_account:
                reason = "broker contract safety account identity mismatch"
                self._poison_generation(generation, reason)
                raise BrokerSnapshotAccountMismatchError(
                    "Broker contract safety snapshot account mismatch "
                    f"(expected={self._masked_account(expected_account)}, "
                    f"connected={self._masked_account(account)})"
                )
            if data["requested_symbol"] != requested_symbol:
                raise ValueError("broker contract safety requested symbol mismatch")
            broker_before = self._strict_timestamp(
                data["broker_time_before"],
                "broker_time_before",
            )
            broker_after = self._strict_timestamp(
                data["broker_time_after"],
                "broker_time_after",
            )
            retrieved_at = self._strict_timestamp(data["retrieved_at"], "retrieved_at")
            if broker_after < broker_before:
                raise ValueError("broker contract safety times are reversed")
            broker_window = (broker_after - broker_before).total_seconds()
            if (
                not math.isfinite(broker_window)
                or broker_window > _BROKER_SNAPSHOT_MAX_WINDOW_SECONDS
            ):
                raise ValueError("broker contract safety collection window is unbounded")
            age = (
                datetime.now(timezone.utc) - retrieved_at.astimezone(timezone.utc)
            ).total_seconds()
            if age < -5 or age > max_age_seconds:
                raise ValueError("broker contract safety snapshot is outside freshness bounds")
            if any(
                abs((broker_time - retrieved_at).total_seconds())
                > _BROKER_SNAPSHOT_MAX_CLOCK_SKEW_SECONDS
                for broker_time in (broker_before, broker_after)
            ):
                raise ValueError("broker contract safety clock skew exceeds safety bound")
            qualified_contract = self._broker_safety_contract(data["qualified_contract"])
            if qualified_contract.symbol != requested_symbol:
                raise ValueError("qualified broker contract differs from requested symbol")
            snapshot = _produce_broker_contract_safety_snapshot(
                capability=capability,
                broker_time_before=broker_before,
                broker_time_after=broker_after,
                retrieved_at=retrieved_at,
                snapshot_id=f"broker-contract-safety-v1-{uuid.uuid4().hex}",
                source="ibkr-subprocess-contract-safety-v1",
                qualified_contract=qualified_contract,
            )
            assert_producer_owned_broker_contract_safety_snapshot(snapshot)
            return snapshot
        except BrokerSnapshotAccountMismatchError:
            raise
        except Exception as exc:
            self._snapshot_fail(generation, str(exc))
        raise AssertionError("unreachable")

    async def _get_broker_contract_safety_snapshot_unlocked(
        self,
        runtime_context: RuntimeSafetyContext,
        normalized_symbol: str,
        *,
        max_age_seconds: float,
    ) -> tuple[
        BrokerContractSafetySnapshot,
        _WorkerGeneration,
        tuple[str, int, int, bool],
        tuple[RuntimeSafetyContext, str, str, str, str],
    ]:
        """Produce contract-only proof while the caller holds the lifecycle lock."""

        generation = self._generation
        if generation is None:
            raise SubprocessCrashError("Subprocess generation is unavailable")
        try:
            validated_context = self._validated_broker_runtime_context(runtime_context)
            context, normalized_account, _, _, _ = validated_context
        except Exception as exc:
            try:
                self._snapshot_fail(
                    generation,
                    "validated broker runtime context is unavailable",
                )
            except IBKRTransportPoisonedError as poisoned:
                raise poisoned from exc
        identity = self._require_snapshot_generation_binding(
            generation,
            poison_on_mismatch=False,
        )
        expected_identity = (
            context.diagnostic_connection.host,
            context.diagnostic_connection.port,
            context.diagnostic_connection.client_id,
            context.diagnostic_connection.readonly,
        )
        if identity != expected_identity:
            self._snapshot_fail(
                generation,
                "broker transport differs from validated runtime context",
            )
        data = await self._execute_command_unlocked(
            {
                "command": "get_broker_contract_safety_snapshot",
                "params": {
                    "expected_account": normalized_account,
                    "requested_symbol": normalized_symbol,
                },
            },
            timeout=30.0,
        )
        self._require_snapshot_generation_binding(
            generation,
            expected_identity=expected_identity,
            poison_on_mismatch=True,
        )
        try:
            revalidated = self._validated_broker_runtime_context(runtime_context)
        except Exception as exc:
            try:
                self._snapshot_fail(
                    generation,
                    "validated broker runtime context changed during contract snapshot",
                )
            except IBKRTransportPoisonedError as poisoned:
                raise poisoned from exc
        if revalidated[1:] != validated_context[1:]:
            self._snapshot_fail(
                generation,
                "validated broker runtime context changed during contract snapshot",
            )
        try:
            capability = _issue_broker_contract_snapshot_capability(
                context,
                connection_identity=identity,
                transport_generation=generation.generation_id,
                requested_symbol=normalized_symbol,
            )
        except Exception as exc:
            try:
                self._snapshot_fail(
                    generation,
                    "broker contract snapshot capability issuance failed",
                )
            except IBKRTransportPoisonedError as poisoned:
                raise poisoned from exc
        snapshot = self._validate_broker_contract_safety_snapshot(
            expected_account=normalized_account,
            requested_symbol=normalized_symbol,
            capability=capability,
            data=data,
            generation=generation,
            max_age_seconds=max_age_seconds,
        )
        self._require_snapshot_generation_binding(
            generation,
            expected_identity=expected_identity,
            poison_on_mismatch=True,
        )
        try:
            final_context = self._validated_broker_runtime_context(runtime_context)
        except Exception as exc:
            try:
                self._snapshot_fail(
                    generation,
                    "validated broker runtime context changed after contract snapshot",
                )
            except IBKRTransportPoisonedError as poisoned:
                raise poisoned from exc
        if final_context[1:] != revalidated[1:]:
            self._snapshot_fail(
                generation,
                "validated broker runtime context changed after contract snapshot",
            )
        assert_producer_owned_broker_contract_safety_snapshot(snapshot)
        return snapshot, generation, expected_identity, final_context

    async def run_with_locked_broker_contract_safety_snapshot(
        self,
        runtime_context: RuntimeSafetyContext,
        requested_symbol: str,
        async_callback: Callable[
            [BrokerContractSafetySnapshot],
            Awaitable[_BrokerCallbackResult],
        ],
        *,
        max_age_seconds: float = 30.0,
    ) -> _BrokerCallbackResult:
        """Run final local paper dispatch under current contract/transport proof."""

        if not isinstance(requested_symbol, str):
            raise ValueError("Broker contract safety symbol must be a string")
        normalized_symbol = requested_symbol.strip().upper()
        if not _BROKER_SAFETY_SYMBOL_RE.fullmatch(normalized_symbol):
            raise ValueError("Broker contract safety symbol is malformed")
        if not callable(async_callback):
            raise TypeError("Broker contract safety callback must be callable")
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(float(max_age_seconds))
            or max_age_seconds <= 0
            or max_age_seconds > 30
        ):
            raise ValueError(
                "Broker contract safety freshness bound must be finite and at most 30 seconds"
            )

        async with self._lifecycle_lock:
            generation = self._generation
            if generation is None:
                raise SubprocessCrashError("Subprocess generation is unavailable")
            snapshot: Optional[BrokerContractSafetySnapshot] = None
            expected_identity: Optional[tuple[str, int, int, bool]] = None
            final_context: Optional[tuple[RuntimeSafetyContext, str, str, str, str]] = None
            try:
                (
                    snapshot,
                    generation,
                    expected_identity,
                    final_context,
                ) = await self._get_broker_contract_safety_snapshot_unlocked(
                    runtime_context,
                    normalized_symbol,
                    max_age_seconds=float(max_age_seconds),
                )
                callback_result = async_callback(snapshot)
                if not inspect.isawaitable(callback_result):
                    raise TypeError("Broker contract safety callback must return an awaitable")
                return await callback_result
            finally:
                if snapshot is not None and expected_identity is not None:
                    self._require_snapshot_generation_binding(
                        generation,
                        expected_identity=expected_identity,
                        poison_on_mismatch=True,
                    )
                    assert_producer_owned_broker_contract_safety_snapshot(snapshot)
                    try:
                        current_context = self._validated_broker_runtime_context(runtime_context)
                    except Exception as exc:
                        try:
                            self._snapshot_fail(
                                generation,
                                "validated broker runtime context changed during finalization",
                            )
                        except IBKRTransportPoisonedError as poisoned:
                            raise poisoned from exc
                    if final_context is None or current_context[1:] != final_context[1:]:
                        self._snapshot_fail(
                            generation,
                            "validated broker runtime context changed during finalization",
                        )
                self._cleanup_worker_debug_log(generation, required=True)

    async def get_broker_contract_safety_snapshot(
        self,
        runtime_context: RuntimeSafetyContext,
        requested_symbol: str,
        *,
        max_age_seconds: float = 30.0,
    ) -> BrokerContractSafetySnapshot:
        """Return contract-only proof through the lifecycle-held boundary."""

        async def return_snapshot(
            snapshot: BrokerContractSafetySnapshot,
        ) -> BrokerContractSafetySnapshot:
            return snapshot

        return await self.run_with_locked_broker_contract_safety_snapshot(
            runtime_context,
            requested_symbol,
            return_snapshot,
            max_age_seconds=max_age_seconds,
        )

    @staticmethod
    def _normalize_historical_symbol(symbol: Any) -> str:
        if not isinstance(symbol, str):
            raise ValueError("Historical symbol must be a string")
        normalized = symbol.strip().upper()
        if not _CONTRACT_TEXT_RE.fullmatch(normalized):
            raise ValueError("Historical symbol is malformed")
        return normalized

    def _historical_lineage_from_response(
        self,
        requested: str,
        data: dict,
        generation: _WorkerGeneration,
    ) -> QualifiedStockContractLineage:
        expected_contract_keys = {
            "con_id",
            "symbol",
            "local_symbol",
            "security_type",
            "currency",
            "exchange",
            "primary_exchange",
            "trading_class",
        }
        echoed = data.get("requested_symbol")
        contract = data.get("qualified_contract")
        if not isinstance(echoed, str) or echoed.strip().upper() != requested:
            raise ValueError("historical response requested symbol mismatch")
        if not isinstance(contract, dict) or set(contract) != expected_contract_keys:
            raise ValueError("historical response missing or malformed qualified contract identity")
        if contract.get("symbol") != requested:
            raise ValueError("historical response qualified contract symbol mismatch")
        if contract.get("local_symbol") != requested:
            raise ValueError("historical response local symbol alias is not explicitly allowed")
        for field_name in ("primary_exchange", "trading_class"):
            value = contract.get(field_name)
            if not isinstance(value, str) or not _CONTRACT_TEXT_RE.fullmatch(value):
                raise ValueError("historical response has incomplete contract identity")
        try:
            broker_timestamp = self._strict_timestamp(
                data.get("broker_timestamp"),
                "historical response broker_timestamp",
            )
            retrieval_timestamp = self._strict_timestamp(
                data.get("retrieval_timestamp"),
                "historical response retrieval_timestamp",
            )
            if retrieval_timestamp > datetime.now(timezone.utc) + timedelta(
                seconds=_BROKER_SNAPSHOT_MAX_CLOCK_SKEW_SECONDS
            ):
                raise ValueError("retrieval_timestamp exceeds allowed clock skew")
            return QualifiedStockContractLineage(
                con_id=contract.get("con_id"),
                symbol=contract.get("symbol"),
                local_symbol=contract.get("local_symbol"),
                security_type=contract.get("security_type"),
                currency=contract.get("currency"),
                exchange=contract.get("exchange"),
                primary_exchange=contract.get("primary_exchange"),
                trading_class=contract.get("trading_class"),
                broker_timestamp=broker_timestamp,
                retrieval_timestamp=retrieval_timestamp,
                transport_generation=generation.generation_id,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"historical response contract lineage is invalid: {exc}") from exc

    def _validate_historical_response(
        self,
        symbol: str,
        data: dict,
        generation: Optional[_WorkerGeneration],
    ) -> list[dict]:
        """Validate and, on ambiguity, poison the exact responding generation."""
        try:
            requested = self._normalize_historical_symbol(symbol)
        except ValueError as exc:
            raise IBKRTransportPoisonedError(str(exc)) from exc
        integrity_error: Optional[str] = None
        lineage: Optional[QualifiedStockContractLineage] = None
        if generation is None:
            integrity_error = "historical response has no worker generation"
        else:
            with generation.state_lock:
                generation_current = (
                    self._generation is generation and generation.poisoned_reason is None
                )
            if not generation_current:
                integrity_error = "historical response belongs to a stale worker generation"
            else:
                try:
                    lineage = self._historical_lineage_from_response(
                        requested,
                        data,
                        generation,
                    )
                except ValueError as exc:
                    integrity_error = str(exc)
        bars = data.get("bars")
        if integrity_error is None and (
            not isinstance(bars, list) or any(not isinstance(bar, dict) for bar in bars)
        ):
            integrity_error = "historical response bars are malformed"

        if integrity_error:
            self._invalidate_historical_lineage()
            if generation:
                self._poison_generation(generation, integrity_error)
            raise IBKRTransportPoisonedError(integrity_error)
        assert generation is not None
        assert lineage is not None
        self._cache_historical_lineage(generation, lineage)
        return cast(list[dict], bars)

    async def get_historical_bars(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> list[dict]:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Stock symbol
            duration: IB duration string (e.g., "2 D", "1 W")
            bar_size: IB bar size (e.g., "5 mins", "1 hour")
            what_to_show: Data type (e.g., "TRADES", "MIDPOINT")
            use_rth: Use regular trading hours only

        Returns:
            List of bar dictionaries with date, open, high, low, close, volume
        """
        normalized_symbol = self._normalize_historical_symbol(symbol)
        if bar_size not in _INTRADAY_BAR_SIZES:
            raise ValueError(
                "Subprocess transport supports only intraday datetime bars; "
                f"unsupported bar_size={bar_size!r}"
            )
        async with self._lifecycle_lock:
            generation = self._generation
            data = await self._execute_command_unlocked(
                {
                    "command": "get_historical_bars",
                    "params": {
                        "symbol": normalized_symbol,
                        "duration": duration,
                        "bar_size": bar_size,
                        "what_to_show": what_to_show,
                        "use_rth": use_rth,
                    },
                },
                timeout=60.0,  # Historical data can take longer
            )
            return self._validate_historical_response(normalized_symbol, data, generation)

    async def disconnect(self) -> None:
        """Disconnect from IBKR"""
        async with self._lifecycle_lock:
            generation = self._generation
            connected, _, bound_generation_id, _, _, _ = self._connection_state_snapshot()
            if not connected or (
                generation is not None and bound_generation_id != generation.generation_id
            ):
                self._clear_cached_connection_state(generation=generation)
                return
            self._clear_cached_connection_state(generation=generation)
            logger.info("Disconnecting from IBKR via subprocess")
            try:
                await self._execute_command_unlocked({"command": "disconnect"}, timeout=5.0)
            except Exception:
                logger.warning("Diagnostic broker disconnect failed; terminating worker subprocess")
            await self._stop_unlocked()
        logger.info("Disconnected from IBKR")

    async def ping(self) -> bool:
        """
        Check if subprocess is alive and responsive.

        Returns:
            True if subprocess responds to ping
        """
        try:
            async with self._lifecycle_lock:
                generation = self._generation
                if generation is None:
                    return False
                data = await self._execute_command_unlocked({"command": "ping"}, timeout=5.0)
                return self._accept_ping_response(data, generation)
        except Exception as e:
            logger.warning("Ping failed", error=str(e))
            return False

    async def health_check(self) -> bool:
        """
        Check if connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        # Check subprocess is alive
        if not self.process or self.process.poll() is not None:
            logger.error("Health check failed: Worker process is dead")
            return False

        # Ping worker to verify responsiveness
        try:
            pong = await self.ping()
            if pong:
                logger.debug("Health check passed: Worker is responsive")
                return True
            else:
                _, _, _, _, _, gateway_detail = self._connection_state_snapshot()
                if gateway_detail:
                    logger.error(
                        "Health check failed: Gateway API reported down",
                        detail=gateway_detail,
                    )
                logger.warning("Health check failed: Ping returned false")
                return False
        except Exception as e:
            logger.error("Health check failed with exception", error=str(e))
            return False

    async def ensure_healthy(self) -> None:
        """
        Ensure connection is healthy, reconnect if needed.

        Raises:
            SubprocessCrashError: If reconnection fails

        NEW-IB-L1 — Reconnect idempotency assumption:
            We assume that an IB reconnect cannot cause duplicate orders
            because this client always connects in ``readonly=True`` mode and
            the Gateway side enforces ``ReadOnlyApi=yes`` (verified at
            startup by ``START_TRADER.sh``,
            see also ``connect()`` below). If this client is ever extended
            to place orders, the reconnect path MUST be revisited — repeating
            an order command after a transient disconnect could double-fill.
        """
        generation = self._generation
        poison_reason = None
        if generation:
            with generation.state_lock:
                poison_reason = generation.poisoned_reason
        if poison_reason:
            raise IBKRTransportPoisonedError(
                "Refusing automatic retry of poisoned worker generation: " f"{poison_reason}"
            )
        if not await self.health_check():
            generation = self._generation
            poison_reason = None
            if generation:
                with generation.state_lock:
                    poison_reason = generation.poisoned_reason
            if poison_reason:
                raise IBKRTransportPoisonedError(
                    "Refusing automatic retry of poisoned worker generation: " f"{poison_reason}"
                )
            _, _, _, _, _, gateway_detail = self._connection_state_snapshot()
            if gateway_detail:
                raise GatewayRequiresRestartError(gateway_detail)
            logger.warning("Connection unhealthy, attempting reconnection...")
            # NEW-IB-L1: Idempotent because this client is read-only end-to-end.
            await self.stop()
            await self.start()
            # Note: Caller needs to call connect() with appropriate params

    @property
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        connected, _, _, _, _, _ = self._connection_state_snapshot()
        return connected

    @property
    def gateway_failure_detail(self) -> Optional[str]:
        """Return the last Gateway failure detail, if any."""
        _, _, _, _, _, detail = self._connection_state_snapshot()
        return detail

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
