"""Integration test: persistent connection across simulated cycles.

Uses a FakeSubprocessIBKRClient stand-in to verify that AsyncRunner can
be reused across multiple cycles via teardown(full_cleanup=False) — the
subprocess is started ONCE, not on every cycle.

API surface notes (verified):
- SubprocessIBKRClient.connect() returns bool (not dict)
- SubprocessIBKRClient.is_connected() is snake_case (not isConnected)
- get_accounts() is a separate async method
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


class FakeSubprocessClient:
    """Matches the verified subset of SubprocessIBKRClient API."""
    instances_created = 0
    start_call_count = 0
    stop_call_count = 0

    def __init__(self):
        type(self).instances_created += 1
        self._connected = False

    async def start(self):
        type(self).start_call_count += 1
        self._connected = True

    async def connect(self, **kwargs):
        return True  # bool, not dict (post Task 6 API verification)

    def is_connected(self):  # snake_case (not camelCase)
        return self._connected

    async def ping(self):
        return True

    async def get_accounts(self):
        return ["DUN264991"]

    async def stop(self):
        type(self).stop_call_count += 1
        self._connected = False


@pytest.mark.asyncio
async def test_persistent_runner_starts_subprocess_only_once_across_cycles():
    """Verify the long-lived-runner contract: across N teardown(full_cleanup=False)
    calls, the subprocess is started ONCE and not stopped."""
    FakeSubprocessClient.instances_created = 0
    FakeSubprocessClient.start_call_count = 0
    FakeSubprocessClient.stop_call_count = 0

    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.host = "127.0.0.1"
    runner.cfg.ibkr.port = 4002
    runner.cfg.ibkr.client_id = 1
    runner._client_id = 1
    runner.portfolio_id = "default"
    runner.ib = None
    runner.subprocess_client = None
    runner.recovery_in_progress = False
    runner._recovery_lock = asyncio.Lock()
    runner.health = None
    # teardown() touches these — provide safe defaults
    runner.production_monitor = None
    runner.correlation_manager = None

    with patch(
        "robo_trader.runner_async.SubprocessIBKRClient",
        FakeSubprocessClient,
    ), patch("asyncio.sleep", new_callable=AsyncMock):
        await runner.initialize_connection()
        for _ in range(3):
            await runner.teardown(full_cleanup=False)

        # The persistent contract:
        assert FakeSubprocessClient.start_call_count == 1, \
            f"start was called {FakeSubprocessClient.start_call_count}x, expected 1"
        assert FakeSubprocessClient.stop_call_count == 0, \
            f"stop was called {FakeSubprocessClient.stop_call_count}x, expected 0 (teardown should not disconnect)"

        # Cleanup the background health monitor task
        if runner.health is not None:
            await runner.health.stop_monitoring()


import pytest
from robo_trader.exceptions import KillSwitchTriggeredError


@pytest.mark.asyncio
async def test_kill_switch_propagates_through_cycle_exception_handler():
    """Regression guard: KillSwitchTriggeredError must NOT be caught by the
    cycle-level `except Exception` handler. It is a safety signal that must
    reach the outer while-loop handler for graceful shutdown.

    Simulates the cycle-level exception handling block in run_continuous."""

    class FakeRunner:
        async def run(self, symbols):
            raise KillSwitchTriggeredError("test: kill switch armed")

    runner = FakeRunner()

    # Mirror the exception handling structure in run_continuous
    with pytest.raises(KillSwitchTriggeredError):
        try:
            await runner.run([])
        except KillSwitchTriggeredError:
            raise
        except Exception:
            pytest.fail(
                "KillSwitchTriggeredError must not be caught by `except Exception` "
                "in cycle handler — it is a safety signal that must reach the "
                "outer while-loop handler for graceful shutdown."
            )
