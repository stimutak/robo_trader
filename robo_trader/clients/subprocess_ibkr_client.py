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
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, cast

import structlog

logger = structlog.get_logger(__name__)


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


class GatewayRequiresRestartError(IBKRError):
    """Raised when the worker detects the Gateway API layer has crashed"""

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

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._connection_state_lock = threading.Lock()
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

    def _clear_connection_tuple_locked(self) -> None:
        """Clear connected-session fields while holding the state lock."""
        self._connected = False
        self._connection_identity = None
        self._connection_generation_id = None
        self._connection_start_time = None
        self._last_activity = None

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
                self._last_activity = datetime.now()
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

        # Find worker script
        worker_script = Path(__file__).parent / "ibkr_subprocess_worker.py"
        if not worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")

        logger.info("Starting IBKR subprocess worker", script=str(worker_script))

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
        project_root = _find_project_root(Path(__file__))
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
            logger.info("Worker debug output will be captured", debug_log=debug_log_path)
        except Exception as e:
            logger.warning("Could not create debug log file", error=str(e))
            debug_log_file = None

        # Wrap subprocess and thread startup in try block to ensure cleanup
        generation: Optional[_WorkerGeneration] = None
        try:
            generation_id = uuid.uuid4().hex
            worker_env = os.environ.copy()
            worker_env["ROBOTRADER_WORKER_GENERATION_ID"] = generation_id
            # CRITICAL FIX: Use regular subprocess.Popen with threading instead of
            # asyncio.create_subprocess_exec to avoid event loop starvation in
            # busy async environments
            # CRITICAL FIX 2: Launch as module with -m to ensure robo_trader/__init__.py
            self.process = subprocess.Popen(
                [python_exe, "-m", "robo_trader.clients.ibkr_subprocess_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr for logging
                text=True,
                bufsize=1,  # Line buffered
                close_fds=True,  # Don't inherit file descriptors
                env=worker_env,
            )
            generation = _WorkerGeneration(generation_id, self.process)
            with self._connection_state_lock:
                # Installing a worker is an authoritative disconnected state.
                # Synchronizing the pointer swap with cached state prevents a
                # delayed poison from the previous worker from clearing state
                # subsequently bound to this generation.
                self._clear_connection_tuple_locked()
                self._gateway_api_down_detail = None
                self._gateway_failure_generation_id = None
                self._generation = generation

            logger.info(
                "IBKR subprocess worker started",
                pid=self.process.pid,
                generation_id=generation_id,
            )

            # Store debug log file only after successful subprocess start
            generation.debug_log_file = debug_log_file
            generation.debug_log_path = debug_log_path
            self._debug_log_file = debug_log_file
            self._debug_log_path = debug_log_path

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
            if debug_log_file:
                try:
                    debug_log_file.close()
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
            self._debug_log_file = None
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
                        generation.debug_log_file.write(f"{datetime.now().isoformat()}: {line}")
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
            if generation.debug_log_file:
                try:
                    generation.debug_log_file.close()
                except Exception:
                    pass

    async def stop(self) -> None:
        """Serialize shutdown against worker creation and commands."""
        async with self._lifecycle_lock:
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        """Stop the subprocess worker with graceful shutdown."""
        generation = self._generation
        process = generation.process if generation else self.process
        if not process:
            self._clear_cached_connection_state(generation=generation)
            return

        logger.info("Stopping IBKR subprocess worker", pid=process.pid)

        try:
            connected, _, bound_generation_id, _, _, _ = self._connection_state_snapshot()
            if connected and (
                generation is None or bound_generation_id == generation.generation_id
            ):
                logger.debug("Sending disconnect command to worker")
                await self._execute_command_unlocked({"command": "disconnect"}, timeout=5.0)
        except Exception as e:
            logger.warning("Disconnect command failed during shutdown", error=str(e))
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

            except Exception as e:
                logger.error("Error during process termination", error=str(e))
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

        # Ensure debug log file is closed (belt-and-suspenders cleanup)
        debug_log_file = generation.debug_log_file if generation else self._debug_log_file
        if debug_log_file:
            try:
                debug_log_file.close()
            except Exception:
                pass
            if generation:
                generation.debug_log_file = None
            self._debug_log_file = None

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

                logger.error(
                    "Command failed",
                    command=command_name,
                    error=error_msg,
                    error_type=error_type,
                    requires_restart=requires_restart,
                )

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

        logger.info("Connecting to IBKR via subprocess", host=host, port=port, client_id=client_id)

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
                            client_id=client_id,
                            readonly=readonly,
                        )
                        return True
                    logger.warning(
                        "Cached broker connection was stale; reconnecting worker session",
                        host=host,
                        port=port,
                        client_id=client_id,
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
                            self._connection_start_time = datetime.now()
                            self._last_activity = datetime.now()
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
            accounts=accounts,
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
            import subprocess as sp

            # Use lsof to check for CLOSE_WAIT connections on the port
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sp.run(
                    ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:CLOSE_WAIT"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ),
            )

            if not result.stdout.strip():
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
                for line in lines:
                    logger.warning("Zombie connection", connection=line.strip())
                return zombie_count, error_msg

            return 0, "No zombies detected"

        except FileNotFoundError:
            logger.warning("lsof command not available - cannot check for zombies")
            return 0, "lsof not available"
        except Exception as e:
            logger.warning("Error checking for zombie connections", error=str(e))
            return 0, f"Error checking zombies: {e}"

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

    def _validate_historical_response(
        self,
        symbol: str,
        data: dict,
        generation: Optional[_WorkerGeneration],
    ) -> list[dict]:
        """Validate and, on ambiguity, poison the exact responding generation."""
        requested = symbol.strip().upper()
        echoed = data.get("requested_symbol")
        contract = data.get("qualified_contract")

        integrity_error: Optional[str] = None
        if not isinstance(echoed, str) or echoed.strip().upper() != requested:
            integrity_error = "historical response requested symbol mismatch"
        elif not isinstance(contract, dict):
            integrity_error = "historical response missing qualified contract identity"
        elif str(contract.get("symbol", "")).strip().upper() != requested:
            integrity_error = "historical response qualified contract symbol mismatch"
        elif (
            not isinstance(contract.get("con_id"), int)
            or isinstance(contract["con_id"], bool)
            or contract["con_id"] <= 0
        ):
            integrity_error = "historical response has invalid qualified contract ID"
        elif str(contract.get("local_symbol", "")).strip().upper() != requested:
            integrity_error = "historical response local symbol alias is not explicitly allowed"
        elif contract.get("security_type") != "STK":
            integrity_error = "historical response security type is not STK"
        elif contract.get("currency") != "USD":
            integrity_error = "historical response currency is not USD"
        elif contract.get("exchange") != "SMART":
            integrity_error = "historical response exchange is not SMART"
        elif any(
            not isinstance(contract.get(field), str) or not contract[field]
            for field in ("primary_exchange", "trading_class")
        ):
            integrity_error = "historical response has incomplete contract identity"

        bars = data.get("bars")
        if integrity_error is None:
            for timestamp_field in ("broker_timestamp", "retrieval_timestamp"):
                timestamp = data.get(timestamp_field)
                try:
                    parsed = (
                        datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else None
                    )
                except ValueError:
                    parsed = None
                if parsed is None or parsed.tzinfo is None or parsed.utcoffset() is None:
                    integrity_error = f"historical response {timestamp_field} is not timezone-aware"
                    break

        if integrity_error is None and (
            not isinstance(bars, list) or any(not isinstance(bar, dict) for bar in bars)
        ):
            integrity_error = "historical response bars are malformed"

        if integrity_error:
            if generation:
                self._poison_generation(generation, integrity_error)
            raise IBKRTransportPoisonedError(integrity_error)
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
                        "symbol": symbol,
                        "duration": duration,
                        "bar_size": bar_size,
                        "what_to_show": what_to_show,
                        "use_rth": use_rth,
                    },
                },
                timeout=60.0,  # Historical data can take longer
            )
            return self._validate_historical_response(symbol, data, generation)

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
            except Exception as e:
                logger.warning(f"Disconnect command failed, will terminate subprocess: {e}")
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
