"""
Subprocess-Based IBKR Client

Manages a subprocess running ibkr_subprocess_worker.py to completely isolate
ib_async from the main trading system's complex async environment.

This solves the ib_async library incompatibility issue where API handshakes
timeout in complex async environments despite successful TCP connections.
"""
import asyncio
import json
import sys
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
    """

    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.lock = asyncio.Lock()
        self._connected = False

    async def start(self) -> None:
        """Start the subprocess worker"""
        if self.process and self.process.returncode is None:
            logger.warning("Subprocess already running")
            return

        # Find worker script
        worker_script = Path(__file__).parent / "ibkr_subprocess_worker.py"
        if not worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")

        logger.info("Starting IBKR subprocess worker", script=str(worker_script))

        # Start subprocess using the same Python interpreter
        # CRITICAL: Use sys.executable to ensure we use the venv Python
        python_exe = sys.executable
        logger.debug("Using Python executable", python_exe=python_exe)

        self.process = await asyncio.create_subprocess_exec(
            python_exe,
            str(worker_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        logger.info("IBKR subprocess worker started", pid=self.process.pid)

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

        # Terminate process
        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Subprocess did not terminate, killing", pid=self.process.pid)
            self.process.kill()
            await self.process.wait()

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
            if not self.process or self.process.returncode is not None:
                raise SubprocessCrashError("Subprocess not running")

            # Send command
            command_json = json.dumps(command) + "\n"
            logger.debug("Sending command to subprocess", command=command.get("command"))

            try:
                self.process.stdin.write(command_json.encode())
                await self.process.stdin.drain()
            except Exception as e:
                raise SubprocessCrashError(f"Failed to send command: {e}")

            # Read response with timeout
            try:
                line = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)

                if not line:
                    raise SubprocessCrashError("Subprocess closed stdout")

                response = json.loads(line.decode().strip())

            except asyncio.TimeoutError:
                logger.error("Command timeout", command=command.get("command"), timeout=timeout)
                raise IBKRTimeoutError(f"Command timeout after {timeout}s")
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON response", error=str(e), line=line.decode())
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
