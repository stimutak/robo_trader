"""
Test configuration for pytest.
Handles async event loop setup for ib_insync and other async tests.
"""

import asyncio
import platform
import sys

import pytest


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
