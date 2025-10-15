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
        self._stop_reader = threading.Event()

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

        # CRITICAL FIX: Use regular subprocess.Popen with threading instead of
        # asyncio.create_subprocess_exec to avoid event loop starvation in
        # busy async environments
        # TEMP DEBUG: Redirect stderr to file
        import tempfile

        stderr_log = tempfile.NamedTemporaryFile(
            mode="w", delete=False, prefix="ibkr_worker_stderr_", suffix=".log"
        )
        logger.info("subprocess_stderr_log", path=stderr_log.name)

        self.process = subprocess.Popen(
            [python_exe, str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_log,
            text=True,
            bufsize=1,  # Line buffered
            close_fds=True,  # Don't inherit file descriptors
        )

        logger.info("IBKR subprocess worker started", pid=self.process.pid)

        # Start reader thread to avoid blocking
        self._stop_reader.clear()
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="IBKRSubprocessReader"
        )
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Thread function to read subprocess stdout"""
        try:
            while not self._stop_reader.is_set() and self.process:
                line = self.process.stdout.readline()
                if not line:
                    break
                self._response_queue.put(line.strip())
        except Exception as e:
            logger.error("Reader thread error", error=str(e))
        finally:
            logger.debug("Reader thread exiting")

    async def stop(self) -> None:
        """Stop the subprocess worker"""
        if not self.process:
            return

        logger.info("Stopping IBKR subprocess worker", pid=self.process.pid)

        try:
            # Try graceful disconnect first
            if self._connected:
                await self.disconnect()
        except Exception:
            pass

        # Signal reader thread to stop
        self._stop_reader.set()

        # Terminate process (use run_in_executor to avoid blocking)
        def terminate_process():
            try:
                self.process.terminate()
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess did not terminate, killing", pid=self.process.pid)
                self.process.kill()
                self.process.wait()

        await asyncio.get_event_loop().run_in_executor(None, terminate_process)

        # Wait for reader thread to finish
        if self._reader_thread and self._reader_thread.is_alive():
            await asyncio.get_event_loop().run_in_executor(None, self._reader_thread.join, 2.0)

        self.process = None
        self._connected = False
        logger.info("IBKR subprocess worker stopped")

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
                logger.error(
                    "Command failed",
                    command=command.get("command"),
                    error=error_msg,
                    error_type=error_type,
                )
                raise IBKRError(f"{error_type}: {error_msg}")

            logger.debug("Command succeeded", command=command.get("command"))
            return response.get("data", {})

    async def connect(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        readonly: bool = True,
        timeout: float = 15.0,
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
            "host": host,
            "port": port,
            "client_id": client_id,
            "readonly": readonly,
            "timeout": timeout,
        }

        logger.info("Connecting to IBKR via subprocess", host=host, port=port, client_id=client_id)

        data = await self._execute_command(command, timeout=timeout + 10)

        self._connected = data.get("connected", False)
        accounts = data.get("accounts", [])

        logger.info(
            "Connected to IBKR via subprocess", connected=self._connected, accounts=accounts
        )

        return self._connected

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
            return data.get("pong", False)
        except Exception as e:
            logger.warning("Ping failed", error=str(e))
            return False

    @property
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self._connected

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
