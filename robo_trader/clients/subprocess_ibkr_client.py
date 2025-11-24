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
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class SubprocessCrashError(Exception):
    """Raised when subprocess crashes or becomes unresponsive"""

    pass


class IBKRError(Exception):
    """Raised when IBKR operation fails"""

    pass


class IBKRTimeoutError(IBKRError):
    """Raised when IBKR operation times out"""

    pass


class GatewayRequiresRestartError(IBKRError):
    """Raised when the worker detects the Gateway API layer has crashed"""

    pass


class SubprocessIBKRClient:
    """
    Async client that manages IBKR connection via subprocess.

    Provides complete process isolation for ib_async library to avoid
    async environment conflicts.

    Uses threading for subprocess I/O to avoid asyncio event loop starvation.
    """

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.lock = asyncio.Lock()
        self._connected = False
        self._response_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_reader_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()
        self._last_activity: Optional[datetime] = None
        self._connection_start_time: Optional[datetime] = None
        self._gateway_api_down_detail: Optional[str] = None
        self._debug_log_file = None  # For capturing worker stderr to file

    async def start(self) -> None:
        """Start the subprocess worker with threading-based I/O"""
        if self.process and self.process.poll() is None:
            logger.warning("Subprocess already running")
            return

        # Find worker script
        worker_script = Path(__file__).parent / "ibkr_subprocess_worker.py"
        if not worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")

        logger.info("Starting IBKR subprocess worker", script=str(worker_script))

        # Start subprocess using the same Python interpreter
        # CRITICAL FIX: sys.executable might not be venv Python if runner was started
        # via shebang or other means. Check for venv and use it if available.
        venv_python = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python3"
        if venv_python.exists():
            python_exe = str(venv_python)
            logger.debug("Using venv Python", python_exe=python_exe)
        else:
            python_exe = sys.executable
            logger.debug("Using sys.executable Python", python_exe=python_exe)

        # DEBUGGING FIX: Create debug log file for worker stderr capture
        debug_log_path = "/tmp/worker_debug.log"
        try:
            self._debug_log_file = open(debug_log_path, "w")
            logger.info("Worker debug output will be captured", debug_log=debug_log_path)
        except Exception as e:
            logger.warning("Could not create debug log file", error=str(e))
            self._debug_log_file = None

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
        )

        logger.info("IBKR subprocess worker started", pid=self.process.pid)

        # Start reader threads to avoid blocking
        self._stop_reader.clear()
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="IBKRSubprocessReader"
        )
        self._reader_thread.start()

        # Start stderr reader thread
        self._stderr_reader_thread = threading.Thread(
            target=self._stderr_read_loop, daemon=True, name="IBKRSubprocessStderrReader"
        )
        self._stderr_reader_thread.start()

    def _read_loop(self) -> None:
        """Thread function to read subprocess stdout"""
        try:
            while not self._stop_reader.is_set() and self.process:
                line = self.process.stdout.readline()
                if not line:
                    break

                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # CRITICAL FIX: Filter out ib_async log messages that pollute stdout
                # Only queue lines that look like JSON responses from our worker
                if line_stripped.startswith('{"status":') or line_stripped.startswith('{"timestamp":'):
                    # ib_async logs start with {"timestamp": - log these but don't queue
                    if line_stripped.startswith('{"timestamp":'):
                        logger.debug("ib_async_stdout", message=line_stripped)
                        continue

                    # Worker responses start with {"status": - queue these
                    self._response_queue.put(line_stripped)
                else:
                    # Unknown format - log for debugging
                    logger.warning("unexpected_stdout", message=line_stripped)
        except Exception as e:
            logger.error("Reader thread error", error=str(e))
        finally:
            logger.debug("Reader thread exiting")

    def _stderr_read_loop(self) -> None:
        """Thread function to read and log subprocess stderr"""
        try:
            while not self._stop_reader.is_set() and self.process:
                line = self.process.stderr.readline()
                if not line:
                    break

                # Write to debug file for detailed analysis
                if self._debug_log_file:
                    try:
                        self._debug_log_file.write(f"{datetime.now().isoformat()}: {line}")
                        self._debug_log_file.flush()
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
        finally:
            logger.debug("Stderr reader thread exiting")
            if self._debug_log_file:
                try:
                    self._debug_log_file.close()
                except Exception:
                    pass

    async def stop(self) -> None:
        """Stop the subprocess worker with graceful shutdown"""
        if not self.process:
            return

        logger.info("Stopping IBKR subprocess worker", pid=self.process.pid)

        try:
            # Try graceful disconnect first (sends disconnect command to worker)
            if self._connected:
                logger.debug("Sending disconnect command to worker")
                await self.disconnect()
        except Exception as e:
            logger.warning("Disconnect command failed during shutdown", error=str(e))

        # Signal reader threads to stop
        self._stop_reader.set()

        # Terminate process with graceful escalation (use run_in_executor to avoid blocking)
        def terminate_process():
            try:
                # Send SIGTERM for graceful shutdown
                logger.debug("Sending SIGTERM to worker process")
                self.process.terminate()

                # Wait up to 3 seconds for graceful shutdown
                self.process.wait(timeout=3.0)
                logger.info("Worker terminated gracefully via SIGTERM")

            except subprocess.TimeoutExpired:
                # Worker didn't respond to SIGTERM, escalate to SIGKILL
                logger.warning(
                    "Worker did not respond to SIGTERM within 3s, sending SIGKILL",
                    pid=self.process.pid,
                )
                self.process.kill()
                self.process.wait()
                logger.warning("Worker killed via SIGKILL")

            except Exception as e:
                logger.error("Error during process termination", error=str(e))
                # Ensure process is dead
                try:
                    self.process.kill()
                    self.process.wait()
                except Exception:
                    pass

        await asyncio.get_event_loop().run_in_executor(None, terminate_process)

        # Wait for reader threads to finish
        if self._reader_thread and self._reader_thread.is_alive():
            logger.debug("Waiting for reader thread to finish")
            await asyncio.get_event_loop().run_in_executor(None, self._reader_thread.join, 2.0)

        if self._stderr_reader_thread and self._stderr_reader_thread.is_alive():
            logger.debug("Waiting for stderr reader thread to finish")
            await asyncio.get_event_loop().run_in_executor(
                None, self._stderr_reader_thread.join, 2.0
            )

        self.process = None
        self._connected = False
        logger.info("IBKR subprocess worker stopped cleanly")

    async def _execute_command(self, command: dict, timeout: float = 30.0) -> dict:
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
            # Check subprocess is running
            if not self.process or self.process.poll() is not None:
                raise SubprocessCrashError("Subprocess not running")

            # Send command (use run_in_executor to avoid blocking)
            command_json = json.dumps(command) + "\n"
            logger.debug("Sending command to subprocess", command=command.get("command"))

            def write_command():
                try:
                    self.process.stdin.write(command_json)
                    self.process.stdin.flush()
                except Exception as e:
                    raise SubprocessCrashError(f"Failed to send command: {e}")

            await asyncio.get_event_loop().run_in_executor(None, write_command)

            # Read response from queue with timeout
            try:
                # Use run_in_executor to wait on queue without blocking event loop
                def read_response():
                    return self._response_queue.get(timeout=timeout)

                line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, read_response),
                    timeout=timeout + 1.0,  # Add buffer to asyncio timeout
                )

                if not line:
                    raise SubprocessCrashError("Subprocess closed stdout")

                response = json.loads(line)

            except (asyncio.TimeoutError, queue.Empty):
                logger.error("Command timeout", command=command.get("command"), timeout=timeout)
                raise IBKRTimeoutError(f"Command timeout after {timeout}s")
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON response", error=str(e), line=line)
                raise SubprocessCrashError(f"Invalid JSON response: {e}")

            # Check response status
            if response.get("status") == "error":
                error_msg = response.get("error", "Unknown error")
                error_type = response.get("error_type", "IBKRError")
                detail = response.get("detail") or ""
                requires_restart = bool(response.get("requires_restart"))

                logger.error(
                    "Command failed",
                    command=command.get("command"),
                    error=error_msg,
                    error_type=error_type,
                    requires_restart=requires_restart,
                )

                if requires_restart or error_type == "GatewayRequiresRestartError":
                    message = detail or error_msg
                    self._gateway_api_down_detail = message or "Gateway API layer reported down"
                    raise GatewayRequiresRestartError(message)

                raise IBKRError(f"{error_type}: {error_msg}")

            logger.debug("Command succeeded", command=command.get("command"))
            return response.get("data", {})

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

        # ZOMBIE CONNECTION PREVENTION: Check for Gateway zombies before attempting connection
        # This prevents wasting time on connections that will fail due to zombie blocking
        logger.debug("Pre-connection zombie check...")
        zombie_count, zombie_msg = await self._check_zombie_connections(port)
        if zombie_count > 0:
            logger.error("Zombie connections detected - aborting connection attempt",
                        zombie_count=zombie_count, message=zombie_msg)
            self._gateway_api_down_detail = (
                f"Gateway has {zombie_count} zombie connection(s) blocking API handshakes. "
                "Restart Gateway (File→Exit, relaunch with 2FA) before retrying."
            )
            raise GatewayRequiresRestartError(self._gateway_api_down_detail)

        # Record connection attempt timing for debugging
        connection_start = time.time()

        try:
            # Increase timeout to account for new synchronization wait in worker
            extended_timeout = timeout + 15  # Extra buffer for handshake + account data wait
            logger.debug("Sending connect command with extended timeout", timeout=extended_timeout)

            data = await self._execute_command(command, timeout=extended_timeout)

            connection_duration = time.time() - connection_start
            logger.info("Connect command completed", duration_seconds=f"{connection_duration:.2f}")

        except GatewayRequiresRestartError:
            self._connected = False
            connection_duration = time.time() - connection_start
            logger.error("Connection failed - Gateway restart required", duration_seconds=f"{connection_duration:.2f}")
            raise

        self._connected = data.get("connected", False)
        accounts = data.get("accounts", [])

        # Track connection timing for health monitoring
        if self._connected:
            self._connection_start_time = datetime.now()
            self._last_activity = datetime.now()
            self._gateway_api_down_detail = None

        logger.info(
            "Connected to IBKR via subprocess", connected=self._connected, accounts=accounts
        )

        return self._connected

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
                )
            )

            if not result.stdout.strip():
                return 0, "No zombies detected"

            # Count zombie connections (skip header line)
            lines = [line for line in result.stdout.split("\n") if line.strip() and not line.startswith("COMMAND")]
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
        data = await self._execute_command(
            {
                "command": "get_historical_bars",
                "symbol": symbol,
                "duration": duration,
                "bar_size": bar_size,
                "what_to_show": what_to_show,
                "use_rth": use_rth,
            },
            timeout=60.0,  # Historical data can take longer
        )
        return data.get("bars", [])

    async def disconnect(self) -> None:
        """Disconnect from IBKR"""
        if not self._connected:
            return

        logger.info("Disconnecting from IBKR via subprocess")
        await self._execute_command({"command": "disconnect"})
        self._connected = False
        logger.info("Disconnected from IBKR")

    async def ping(self) -> bool:
        """
        Check if subprocess is alive and responsive.

        Returns:
            True if subprocess responds to ping
        """
        try:
            data = await self._execute_command({"command": "ping"}, timeout=5.0)
            pong = data.get("pong", False)
            if data.get("gateway_api_down"):
                detail = data.get("detail") or "Gateway API layer reported down by worker ping"
                self._gateway_api_down_detail = detail
                logger.error("Worker ping reports Gateway API down", detail=detail)
                return False
            if pong:
                self._last_activity = datetime.now()
            return pong
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
                if self._gateway_api_down_detail:
                    logger.error(
                        "Health check failed: Gateway API reported down",
                        detail=self._gateway_api_down_detail,
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
        """
        if not await self.health_check():
            if self._gateway_api_down_detail:
                raise GatewayRequiresRestartError(self._gateway_api_down_detail)
            logger.warning("Connection unhealthy, attempting reconnection...")
            await self.stop()
            await self.start()
            # Note: Caller needs to call connect() with appropriate params

    @property
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self._connected

    @property
    def gateway_failure_detail(self) -> Optional[str]:
        """Return the last Gateway failure detail, if any."""
        return self._gateway_api_down_detail

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
