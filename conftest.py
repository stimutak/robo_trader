"""
Test configuration for pytest.
Handles async event loop setup for ib_insync and other async tests.

CRITICAL: Isolates tests from production paths (log file, database).
Without this, a test teardown that closes its DB connections will tear down
the live trader's pool too, and any log line a test writes will land in the
production robo_trader.log. This caused a 2.5-day outage on 2026-05-24 when
pytest's database teardown crashed the running trader.
"""

import asyncio
import os
import platform
import sys
import tempfile
from pathlib import Path

import pytest

# --- production-path isolation: runs before pytest collects test files ---
# Test files import robo_trader modules (logger, database_async) which read
# these env vars at import time, so they MUST be set before collection. This
# module body executes during pytest's conftest-loading phase, BEFORE any test
# file is imported, so the ordering holds.
_TEST_ARTIFACTS = Path(__file__).parent / ".test_artifacts"
_TEST_ARTIFACTS.mkdir(exist_ok=True)
os.environ["LOG_FILE"] = str(_TEST_ARTIFACTS / f"pytest_{os.getpid()}.log")
_test_db_fd, _test_db_path = tempfile.mkstemp(
    suffix=".db", prefix=f"rt_test_{os.getpid()}_", dir=str(_TEST_ARTIFACTS)
)
os.close(_test_db_fd)
os.environ["RT_DB_PATH"] = _test_db_path
os.environ["RT_TEST_MODE"] = "1"
# -------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    # Set event loop policy for Windows
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Create new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    yield loop

    # Clean up
    try:
        # Cancel all pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # Wait for all tasks to complete cancellation
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        loop.close()
    except Exception:
        pass


def pytest_configure(config):
    """Configure pytest with async support."""
    # Ensure we have an event loop for ib_insync imports
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in current thread, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def pytest_sessionstart(session):
    """Set up event loop at session start."""
    # Fix for ib_insync event loop issues
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
