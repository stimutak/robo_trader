"""Database monitoring and cleanup service for RoboTrader.

This module provides:
1. Periodic database health monitoring
2. Automatic cleanup of stale connections
3. Lock detection and resolution
4. Connection pool management
"""

import asyncio
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


class DatabaseMonitor:
    """Monitor database health and manage connections."""

    def __init__(self, db_path: str = "trading_data.db", check_interval: int = 60):
        self.db_path = Path(db_path)
        self.check_interval = check_interval
        self.running = False
        self.last_check = None
        self.lock_warnings = 0
        self.max_lock_warnings = 5

    async def start(self):
        """Start the database monitoring service."""
        if self.running:
            return

        self.running = True
        logger.info("Starting database monitor")

        # Start monitoring task
        asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop the database monitoring service."""
        self.running = False
        logger.info("Stopped database monitor")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._check_database_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Database monitor error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_database_health(self):
        """Check database health and resolve issues."""
        self.last_check = datetime.now()

        # Check if database file exists
        if not self.db_path.exists():
            logger.warning(f"Database file {self.db_path} does not exist")
            return

        # Check for lock holders
        lock_holders = await self._get_lock_holders()

        if lock_holders:
            self.lock_warnings += 1
            logger.warning(
                f"Database has {len(lock_holders)} lock holders "
                f"(warning {self.lock_warnings}/{self.max_lock_warnings})"
            )

            # Log lock holder details
            for holder in lock_holders:
                logger.info(f"Lock holder: PID {holder['pid']} ({holder['command']})")

            # If too many warnings, attempt cleanup
            if self.lock_warnings >= self.max_lock_warnings:
                logger.warning("Too many lock warnings, attempting cleanup")
                await self._cleanup_stale_locks(lock_holders)
                self.lock_warnings = 0
        else:
            # Reset warning counter if no locks
            if self.lock_warnings > 0:
                logger.info("Database locks cleared")
                self.lock_warnings = 0

        # Test database access
        accessible = await self._test_database_access()
        if not accessible:
            logger.error("Database is not accessible")

    async def _get_lock_holders(self) -> List[Dict]:
        """Get list of processes holding database locks."""
        try:
            # Run lsof command to check for file locks
            proc = await asyncio.create_subprocess_exec(
                "lsof",
                str(self.db_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode == 0:
                lines = stdout.decode().strip().split("\n")
                lock_holders = []

                for line in lines[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 2:
                        lock_holders.append(
                            {
                                "command": parts[0],
                                "pid": parts[1],
                                "user": parts[2] if len(parts) > 2 else "unknown",
                                "fd": parts[3] if len(parts) > 3 else "unknown",
                            }
                        )

                return lock_holders
            else:
                return []

        except asyncio.TimeoutError:
            logger.warning("lsof command timed out")
            return []
        except FileNotFoundError:
            logger.debug("lsof command not available")
            return []
        except Exception as e:
            logger.debug(f"Error checking lock holders: {e}")
            return []

    async def _test_database_access(self) -> bool:
        """Test if database can be accessed."""
        try:
            import sqlite3

            # Try to connect with a short timeout
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=2.0)

            # Run a simple query
            conn.execute("SELECT 1")
            conn.close()

            return True

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.debug("Database access test: locked")
            else:
                logger.debug(f"Database access test error: {e}")
            return False
        except Exception as e:
            logger.debug(f"Database access test unexpected error: {e}")
            return False

    async def _cleanup_stale_locks(self, lock_holders: List[Dict]):
        """Attempt to cleanup stale database locks."""
        logger.info("Attempting to cleanup stale database locks")

        for holder in lock_holders:
            pid = holder["pid"]
            command = holder["command"]

            # Check if process is a RoboTrader process
            if any(
                keyword in command.lower() for keyword in ["python", "runner_async", "robo_trader"]
            ):
                logger.info(f"Found potential stale RoboTrader process: PID {pid} ({command})")

                # Get process details
                process_info = await self._get_process_info(pid)

                if process_info and "command" in process_info:
                    full_command = process_info["command"]

                    # Check if it's definitely a RoboTrader process
                    if (
                        "robo_trader" in full_command.lower()
                        or "runner_async" in full_command.lower()
                    ):
                        logger.warning(f"Terminating stale RoboTrader process: PID {pid}")

                        # Try graceful termination first
                        success = await self._terminate_process(pid, graceful=True)

                        if not success:
                            logger.warning(f"Graceful termination failed, force killing PID {pid}")
                            await self._terminate_process(pid, graceful=False)
                    else:
                        logger.info(f"Process PID {pid} doesn't appear to be RoboTrader, skipping")
                else:
                    logger.warning(f"Could not get details for PID {pid}, skipping")
            else:
                logger.info(f"Process PID {pid} ({command}) is not a Python process, skipping")

    async def _get_process_info(self, pid: str) -> Optional[Dict]:
        """Get detailed process information."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ps",
                "-p",
                pid,
                "-o",
                "pid,ppid,user,command",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                lines = stdout.decode().strip().split("\n")
                if len(lines) > 1:
                    parts = lines[1].split(None, 3)  # Split into max 4 parts
                    return {
                        "pid": parts[0] if len(parts) > 0 else pid,
                        "ppid": parts[1] if len(parts) > 1 else "unknown",
                        "user": parts[2] if len(parts) > 2 else "unknown",
                        "command": parts[3] if len(parts) > 3 else "unknown",
                    }

            return None

        except Exception as e:
            logger.debug(f"Error getting process info for PID {pid}: {e}")
            return None

    async def _terminate_process(self, pid: str, graceful: bool = True) -> bool:
        """Terminate a process."""
        try:
            signal_type = "TERM" if graceful else "KILL"

            proc = await asyncio.create_subprocess_exec(
                "kill",
                f"-{signal_type}",
                pid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                logger.info(f"Successfully sent SIG{signal_type} to PID {pid}")
                return True
            else:
                logger.warning(f"Failed to send SIG{signal_type} to PID {pid}: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Error terminating process PID {pid}: {e}")
            return False

    def get_status(self) -> Dict:
        """Get current monitor status."""
        return {
            "running": self.running,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "lock_warnings": self.lock_warnings,
            "max_lock_warnings": self.max_lock_warnings,
            "check_interval": self.check_interval,
            "database_path": str(self.db_path),
        }


# Global monitor instance
_monitor_instance: Optional[DatabaseMonitor] = None


async def start_database_monitor(db_path: str = "trading_data.db", check_interval: int = 60):
    """Start the global database monitor."""
    global _monitor_instance

    if _monitor_instance is None:
        _monitor_instance = DatabaseMonitor(db_path, check_interval)

    await _monitor_instance.start()
    return _monitor_instance


async def stop_database_monitor():
    """Stop the global database monitor."""
    global _monitor_instance

    if _monitor_instance:
        await _monitor_instance.stop()
        _monitor_instance = None


def get_database_monitor() -> Optional[DatabaseMonitor]:
    """Get the global database monitor instance."""
    return _monitor_instance
