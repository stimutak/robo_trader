# Persistent IBKR Connection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fresh-AsyncRunner-each-cycle pattern with a long-lived IBKR connection that reconnects only on detected failure, eliminating the IBKR-throttle cascade root-caused on 2026-05-13 and 2026-05-15.

**Architecture:** Add a centralized `ConnectionHealth` module that owns "is the connection usable?". Move IBKR connection lifecycle from per-cycle scope to `run_continuous()` scope. New `AsyncRunner.recover_connection()` with exponential backoff `[15, 30, 60, 120, 300]` and Gateway restart on attempt 3+. Watchdog stays as the outer safety net.

**Tech Stack:** Python 3.12, `ib_async`, `pytest`, `pytest-asyncio`, `asyncio.Lock`. Project venv at `/Users/oliver/Projects/robo_trader/.venv/bin/python3`.

**Branch:** `feature/persistent-ibkr-connection` (already created from `main` @ `1a68a0a`). All commits land here.

**Spec:** `docs/superpowers/specs/2026-05-16-persistent-ibkr-connection-design.md` — read it first if you haven't.

**Critical rules carved in blood (from prior outages, do NOT violate):**
1. NEVER use `socket.connect_ex()` for port checks — use `lsof` (2026-01-05 handoff)
2. NEVER call `ib.disconnect()` on a failed connection — crashes Gateway API layer (2025-11-20)
3. NEVER reuse a dead `SubprocessIBKRClient` instance — always instantiate fresh (2025-12-24)
4. ALWAYS run 2.0s stabilization wait + `isConnected()` poll after Gateway handshake (2025-11-24)
5. ALWAYS commit changes immediately after passing tests; do NOT batch commits across tasks

---

## Pre-flight verification

Run these once before starting. If anything fails, stop and tell the user — do not start the plan in a broken state.

- [ ] **Pre-flight 1: Confirm branch and clean tree**

```bash
cd /Users/oliver/Projects/robo_trader
git branch --show-current
# Expected: feature/persistent-ibkr-connection

git diff --stat
# Expected: empty (no uncommitted code changes other than possibly robo_trader.log.1 type change)
```

- [ ] **Pre-flight 2: Confirm Python venv works**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -c "from robo_trader.runner_async import AsyncRunner; print('OK')"
# Expected: OK (after some startup warnings about feedparser etc)
```

- [ ] **Pre-flight 3: Confirm baseline test suite is green**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/ -q --tb=no 2>&1 | tail -3
# Expected: "638 passed, 3 skipped" (or close to it). NO failures.
```

If any test fails on `main` baseline before this plan starts, fix it FIRST or pause — we don't want to confuse new failures with pre-existing ones.

---

## Task 1: Scaffold `ConnectionHealth` with initial state test

**Files:**
- Create: `robo_trader/connection_health.py`
- Test: `tests/test_connection_health.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_connection_health.py`:

```python
"""Tests for ConnectionHealth - centralized IBKR connection state gate.

Per the 2026-05-16 design spec, ConnectionHealth is the single decision
point for 'is the connection usable?', replacing health logic scattered
across subprocess_ibkr_client, connection_manager, and runner_async.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.connection_health import ConnectionHealth, HealthStatus


def make_fake_ib_client():
    """Return a fake IB client matching the SubprocessIBKRClient surface
    that ConnectionHealth needs: ping() -> bool, isConnected() -> bool."""
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.isConnected = MagicMock(return_value=True)
    return client


def test_initial_status_is_healthy():
    health = ConnectionHealth(ib_client=make_fake_ib_client())
    assert health.status is HealthStatus.HEALTHY
```

- [ ] **Step 2: Run test, verify it fails**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_connection_health.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `robo_trader.connection_health`.

- [ ] **Step 3: Implement the minimal class**

Create `robo_trader/connection_health.py`:

```python
"""Centralized IBKR connection health gate.

Single decision point for 'is the connection usable?'. Replaces ad-hoc
health checks scattered across subprocess_ibkr_client.py,
connection_manager.py, and runner_async._monitor_subprocess_health.

Per 2026-05-16 design spec (docs/superpowers/specs/).
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class HealthStatus(Enum):
    HEALTHY = "healthy"          # consecutive_failures < max_consecutive_failures
    UNHEALTHY = "unhealthy"      # consecutive_failures >= max_consecutive_failures
    RECOVERING = "recovering"    # recover_connection() is mid-flight


class ConnectionHealth:
    def __init__(
        self,
        ib_client: Any,
        ping_interval_seconds: int = 30,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._ib_client = ib_client
        self._ping_interval = ping_interval_seconds
        self._max_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._status: HealthStatus = HealthStatus.HEALTHY

    @property
    def status(self) -> HealthStatus:
        return self._status
```

- [ ] **Step 4: Run test, verify it passes**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_connection_health.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add robo_trader/connection_health.py tests/test_connection_health.py
git commit -m "feat(connection-health): scaffold ConnectionHealth with HEALTHY initial state

First TDD increment of the persistent-IBKR-connection design.
HealthStatus enum is 3-valued: HEALTHY, UNHEALTHY, RECOVERING.
Defaults: ping_interval=30s, max_consecutive_failures=3."
```

---

## Task 2: `record_failure` / `record_success` state transitions

**Files:**
- Modify: `robo_trader/connection_health.py`
- Modify: `tests/test_connection_health.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_connection_health.py`:

```python
def test_record_failure_below_threshold_stays_healthy():
    health = ConnectionHealth(ib_client=make_fake_ib_client(), max_consecutive_failures=3)
    health.record_failure(RuntimeError("transient"), context="cycle:AAPL")
    health.record_failure(RuntimeError("transient"), context="cycle:NVDA")
    assert health.status is HealthStatus.HEALTHY


def test_record_failure_at_threshold_transitions_unhealthy():
    health = ConnectionHealth(ib_client=make_fake_ib_client(), max_consecutive_failures=3)
    health.record_failure(RuntimeError("transient"), context="cycle:AAPL")
    health.record_failure(RuntimeError("transient"), context="cycle:NVDA")
    health.record_failure(RuntimeError("transient"), context="cycle:TSLA")
    assert health.status is HealthStatus.UNHEALTHY


def test_record_success_resets_counter():
    health = ConnectionHealth(ib_client=make_fake_ib_client(), max_consecutive_failures=3)
    health.record_failure(RuntimeError("transient"), context="ping")
    health.record_failure(RuntimeError("transient"), context="ping")
    health.record_success()
    health.record_failure(RuntimeError("transient"), context="ping")
    # After reset, this is failure 1, not failure 3
    assert health.status is HealthStatus.HEALTHY
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_connection_health.py -v
```

Expected: 3 failures with `AttributeError: ... has no attribute 'record_failure'`.

- [ ] **Step 3: Implement `record_failure` and `record_success`**

Append to `robo_trader/connection_health.py`:

```python
import logging

logger = logging.getLogger(__name__)


# ... inside class ConnectionHealth:

    def record_failure(self, error: BaseException, context: str) -> None:
        """Report a transient failure (e.g., Errno 54 in a cycle).

        Increments the consecutive-failure counter. If the counter crosses
        max_consecutive_failures, status transitions to UNHEALTHY and
        background monitor (when wired) will trigger recovery.
        """
        self._consecutive_failures += 1
        prev_status = self._status
        if self._consecutive_failures >= self._max_failures:
            self._status = HealthStatus.UNHEALTHY
        if prev_status is not self._status:
            logger.info(
                "event=connection_state_change from=%s to=%s reason=%r context=%s",
                prev_status.value,
                self._status.value,
                str(error),
                context,
            )

    def record_success(self) -> None:
        """Report a successful API exchange. Resets failure counter."""
        if self._consecutive_failures == 0 and self._status is HealthStatus.HEALTHY:
            return
        prev_status = self._status
        self._consecutive_failures = 0
        if self._status is HealthStatus.UNHEALTHY:
            self._status = HealthStatus.HEALTHY
        if prev_status is not self._status:
            logger.info(
                "event=connection_state_change from=%s to=%s reason=record_success",
                prev_status.value,
                self._status.value,
            )
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_connection_health.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add robo_trader/connection_health.py tests/test_connection_health.py
git commit -m "feat(connection-health): record_failure / record_success state transitions

Failures past threshold transition HEALTHY -> UNHEALTHY.
Success resets counter and recovers to HEALTHY if previously UNHEALTHY.
Both transitions log a structured event=connection_state_change line."
```

---

## Task 3: `perform_check` active probe

**Files:**
- Modify: `robo_trader/connection_health.py`
- Modify: `tests/test_connection_health.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_connection_health.py`:

```python
@pytest.mark.asyncio
async def test_perform_check_calls_subprocess_ping():
    fake = make_fake_ib_client()
    health = ConnectionHealth(ib_client=fake)
    result = await health.perform_check()
    fake.ping.assert_awaited_once()
    assert result is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_perform_check_failure_increments_counter():
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(return_value=False)
    health = ConnectionHealth(ib_client=fake, max_consecutive_failures=3)
    await health.perform_check()
    await health.perform_check()
    assert health.status is HealthStatus.HEALTHY  # 2/3 failures
    await health.perform_check()
    assert health.status is HealthStatus.UNHEALTHY  # 3/3 -> threshold


@pytest.mark.asyncio
async def test_perform_check_success_resets_counter():
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(side_effect=[False, False, True])
    health = ConnectionHealth(ib_client=fake, max_consecutive_failures=3)
    await health.perform_check()
    await health.perform_check()
    await health.perform_check()
    assert health.status is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_perform_check_respects_ib_not_connected():
    fake = make_fake_ib_client()
    fake.isConnected = MagicMock(return_value=False)
    health = ConnectionHealth(ib_client=fake, max_consecutive_failures=3)
    # Even if ping would succeed, isConnected()==False is a hard failure.
    result = await health.perform_check()
    assert result is HealthStatus.HEALTHY  # 1/3, still healthy by status
    assert health._consecutive_failures == 1
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_connection_health.py -v
```

Expected: 4 failures with `AttributeError: ... has no attribute 'perform_check'`.

- [ ] **Step 3: Implement `perform_check`**

Append to `robo_trader/connection_health.py`:

```python
# ... inside class ConnectionHealth:

    async def perform_check(self) -> HealthStatus:
        """Active probe: subprocess ping + ib.isConnected().

        Returns the current status after the probe. Mutates internal state
        via record_success() or record_failure().
        """
        try:
            is_connected = self._ib_client.isConnected()
        except Exception as e:
            self.record_failure(e, "perform_check:isConnected")
            return self._status

        if not is_connected:
            self.record_failure(
                RuntimeError("ib.isConnected() returned False"),
                "perform_check:not_connected",
            )
            return self._status

        try:
            ping_ok = await self._ib_client.ping()
        except Exception as e:
            self.record_failure(e, "perform_check:ping_exception")
            return self._status

        if ping_ok:
            self.record_success()
        else:
            self.record_failure(
                RuntimeError("ping returned False"),
                "perform_check:ping_falsy",
            )
        return self._status
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_connection_health.py -v
```

Expected: `8 passed` (4 from earlier + 4 new).

- [ ] **Step 5: Commit**

```bash
git add robo_trader/connection_health.py tests/test_connection_health.py
git commit -m "feat(connection-health): perform_check active probe

Combines ib.isConnected() AND subprocess ping(). Both must succeed for
record_success(); any failure mode (exception, falsy result, not connected)
calls record_failure() with a distinct context string for diagnostics."
```

---

## Task 4: Background monitoring loop

**Files:**
- Modify: `robo_trader/connection_health.py`
- Modify: `tests/test_connection_health.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_connection_health.py`:

```python
import asyncio


@pytest.mark.asyncio
async def test_start_monitoring_calls_on_unhealthy_at_threshold():
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(return_value=False)
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=0.01,  # fast for test
        max_consecutive_failures=2,
    )
    await health.start_monitoring(on_unhealthy=on_unhealthy)
    # Give the monitor loop time to fire 2 checks
    await asyncio.sleep(0.05)
    await health.stop_monitoring()
    on_unhealthy.assert_awaited()
    # Reason argument should describe the failure
    call_args = on_unhealthy.await_args
    assert "perform_check" in str(call_args) or "ping" in str(call_args)


@pytest.mark.asyncio
async def test_monitor_loop_survives_perform_check_exception():
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(side_effect=[RuntimeError("boom"), True, True])
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=0.01,
        max_consecutive_failures=5,  # high so exception doesn't trip it
    )
    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await asyncio.sleep(0.05)
    await health.stop_monitoring()
    # Despite the first ping raising, subsequent checks ran (no silent death)
    assert fake.ping.await_count >= 2


@pytest.mark.asyncio
async def test_stop_monitoring_cancels_task_cleanly():
    fake = make_fake_ib_client()
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=10,  # very long so we can stop before it fires
    )
    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await health.stop_monitoring()
    # stop_monitoring should be idempotent
    await health.stop_monitoring()
    # No assertion needed — just that no exception/hang
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_connection_health.py -v
```

Expected: 3 failures, `AttributeError: ... start_monitoring`.

- [ ] **Step 3: Implement `start_monitoring`, `stop_monitoring`, and `_monitor_loop`**

Append to `robo_trader/connection_health.py`:

```python
import asyncio
from typing import Awaitable, Callable, Optional


# ... inside class ConnectionHealth.__init__, ADD:
#     self._monitor_task: Optional[asyncio.Task] = None
#     self._stopped = False
#     self._on_unhealthy: Optional[Callable[[str], Awaitable[None]]] = None
#
# Then add the methods:

    async def start_monitoring(
        self,
        on_unhealthy: Callable[[str], Awaitable[None]],
    ) -> None:
        """Spawn background task that calls perform_check() every
        ping_interval_seconds. On transition to UNHEALTHY, awaits
        on_unhealthy(reason)."""
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        self._on_unhealthy = on_unhealthy
        self._stopped = False
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Cancel the background monitor task. Idempotent."""
        self._stopped = True
        if self._monitor_task is None:
            return
        if not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """Survives perform_check exceptions — fails safe to UNHEALTHY,
        keeps trying. Never silently dies."""
        while not self._stopped:
            try:
                prev_status = self._status
                await self.perform_check()
                if (
                    prev_status is not HealthStatus.UNHEALTHY
                    and self._status is HealthStatus.UNHEALTHY
                    and self._on_unhealthy is not None
                ):
                    try:
                        await self._on_unhealthy(
                            f"perform_check transitioned to UNHEALTHY after "
                            f"{self._consecutive_failures} consecutive failures"
                        )
                    except Exception:
                        logger.exception(
                            "on_unhealthy callback raised; continuing monitor loop"
                        )
            except Exception:
                logger.exception(
                    "Health monitor iteration crashed - failing safe to UNHEALTHY"
                )
                self._status = HealthStatus.UNHEALTHY
            try:
                await asyncio.sleep(self._ping_interval)
            except asyncio.CancelledError:
                break
```

You also need to update the `__init__` signature to initialize the new attributes:

```python
    def __init__(
        self,
        ib_client: Any,
        ping_interval_seconds: int = 30,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._ib_client = ib_client
        self._ping_interval = ping_interval_seconds
        self._max_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._status: HealthStatus = HealthStatus.HEALTHY
        self._monitor_task: Optional[asyncio.Task] = None
        self._stopped = False
        self._on_unhealthy: Optional[Callable[[str], Awaitable[None]]] = None
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_connection_health.py -v
```

Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add robo_trader/connection_health.py tests/test_connection_health.py
git commit -m "feat(connection-health): background monitor loop

start_monitoring spawns an asyncio task that calls perform_check every
ping_interval. On transition to UNHEALTHY, awaits the user-supplied
on_unhealthy(reason) callback. Loop survives its own exceptions, falling
safe to UNHEALTHY rather than silently dying. stop_monitoring is idempotent."
```

---

## Task 5: `_safe_disconnect` helper on `AsyncRunner`

**Files:**
- Modify: `robo_trader/runner_async.py` (add method to `AsyncRunner`)
- Create: `tests/test_runner_safe_disconnect.py`

This addresses Edge 1 from the spec (2025-11-20 handoff): calling `ib.disconnect()` on a failed connection crashes Gateway.

- [ ] **Step 1: Write failing test**

Create `tests/test_runner_safe_disconnect.py`:

```python
"""Tests for AsyncRunner._safe_disconnect.

Per 2025-11-20 handoff: calling ib.disconnect() on a FAILED connection
crashes the Gateway API layer. _safe_disconnect must check isConnected()
first and skip the disconnect call when the connection is already gone.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.runner_async import AsyncRunner


def make_runner_with_fake_ib(is_connected: bool):
    """Build an AsyncRunner with a stubbed-out IB client.
    Skips heavy __init__ side effects."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.ib = MagicMock()
    runner.ib.isConnected = MagicMock(return_value=is_connected)
    runner.ib.disconnectAsync = AsyncMock()
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_safe_disconnect_skips_disconnect_when_not_connected():
    runner = make_runner_with_fake_ib(is_connected=False)
    await runner._safe_disconnect()
    runner.ib.disconnectAsync.assert_not_awaited()


@pytest.mark.asyncio
async def test_safe_disconnect_calls_disconnect_when_connected():
    runner = make_runner_with_fake_ib(is_connected=True)
    await runner._safe_disconnect()
    runner.ib.disconnectAsync.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_stops_subprocess_regardless_of_connection_state():
    for connected in (True, False):
        runner = make_runner_with_fake_ib(is_connected=connected)
        await runner._safe_disconnect()
        runner.subprocess_client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_swallows_disconnect_timeout():
    import asyncio
    runner = make_runner_with_fake_ib(is_connected=True)
    runner.ib.disconnectAsync = AsyncMock(side_effect=asyncio.TimeoutError())
    # Should not raise — we're already past hope, don't make Gateway crash matter
    await runner._safe_disconnect()
    runner.subprocess_client.stop.assert_awaited_once()
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_runner_safe_disconnect.py -v
```

Expected: 4 failures, `AttributeError: 'AsyncRunner' object has no attribute '_safe_disconnect'`.

- [ ] **Step 3: Implement `_safe_disconnect`**

Add the following method inside `class AsyncRunner` in `robo_trader/runner_async.py`. Place it AFTER the existing `cleanup()` method (around line 4140, after the existing cleanup completes) so related code is co-located:

```python
    async def _safe_disconnect(self) -> None:
        """Disconnect IBKR cleanly when possible; never crash Gateway.

        Per 2025-11-20 handoff: calling ib.disconnect() on a FAILED
        connection crashes the Gateway API layer. Skip disconnect when
        isConnected() is False. ALWAYS stop the subprocess.
        """
        import asyncio

        if self.ib is not None:
            try:
                if self.ib.isConnected():
                    try:
                        await asyncio.wait_for(
                            self.ib.disconnectAsync(), timeout=5.0
                        )
                    except (asyncio.TimeoutError, Exception):
                        # Connection already broken — don't make Gateway crash matter
                        pass
            except Exception:
                # isConnected() itself can throw on a dead client
                pass

        if getattr(self, "subprocess_client", None) is not None:
            try:
                await self.subprocess_client.stop()
            except Exception:
                logger.warning(
                    "subprocess_client.stop() raised during _safe_disconnect; ignoring",
                    exc_info=True,
                )
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_runner_safe_disconnect.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add robo_trader/runner_async.py tests/test_runner_safe_disconnect.py
git commit -m "feat(runner): _safe_disconnect helper to prevent Gateway crash regression

Per 2025-11-20 handoff, calling ib.disconnect() on a failed connection
crashes the Gateway API layer. _safe_disconnect checks isConnected()
first and only attempts the polite disconnect on live connections. The
subprocess is always stopped regardless. Exceptions during teardown are
swallowed deliberately - we're already past hope and adding errors on
top makes diagnosis harder."
```

---

## Task 6: Extract `initialize_connection` from `AsyncRunner.run()`

**Files:**
- Modify: `robo_trader/runner_async.py`
- Create: `tests/test_runner_initialize_connection.py`

This is a refactor task. We need to extract the IBKR-connection-setup portion of `run()` into a separately-callable method so `recover_connection()` can invoke it.

- [ ] **Step 1: Read the current connection-setup code**

```bash
grep -n "Initializing robust connection\|Pre-connection zombie\|Connecting to IBKR via subprocess\|isConnected.*poll" /Users/oliver/Projects/robo_trader/robo_trader/runner_async.py | head -20
```

Note line numbers for orientation. The current connection setup is inline within `setup()` (around line 888) and `run()` (around line 3221). For Task 6 we extract the portion responsible for: (1) zombie pre-flight, (2) subprocess client start, (3) ib.connect with stabilization wait. We do NOT yet move portfolio sync (Task 9 handles that).

- [ ] **Step 2: Write failing test**

Create `tests/test_runner_initialize_connection.py`:

```python
"""Tests for AsyncRunner.initialize_connection() — the extraction of
IBKR connection setup from run() into a separately-callable method.

This enables recover_connection() to call the same setup path during
runtime recovery without going through full setup()/run() startup.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


@pytest.mark.asyncio
async def test_initialize_connection_starts_subprocess_and_connects(monkeypatch):
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.host = "127.0.0.1"
    runner.cfg.ibkr.port = 4002
    runner.cfg.ibkr.client_id = 1
    runner._client_id = 1
    runner.portfolio_id = "default"
    runner.ib = None
    runner.subprocess_client = None

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value={"connected": True, "accounts": ["DUN264991"]})
    fake_client.isConnected = MagicMock(return_value=True)
    fake_client.ping = AsyncMock(return_value=True)

    with patch(
        "robo_trader.runner_async.SubprocessIBKRClient",
        return_value=fake_client,
    ):
        await runner.initialize_connection()

    fake_client.start.assert_awaited_once()
    fake_client.connect.assert_awaited_once()
    # After init, runner.ib should be set
    assert runner.ib is fake_client


@pytest.mark.asyncio
async def test_initialize_connection_raises_on_connect_failure(monkeypatch):
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.host = "127.0.0.1"
    runner.cfg.ibkr.port = 4002
    runner.cfg.ibkr.client_id = 1
    runner._client_id = 1
    runner.portfolio_id = "default"
    runner.ib = None
    runner.subprocess_client = None

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(
        return_value={"connected": False, "error": "Errno 54"}
    )
    fake_client.isConnected = MagicMock(return_value=False)
    fake_client.stop = AsyncMock()

    with patch(
        "robo_trader.runner_async.SubprocessIBKRClient",
        return_value=fake_client,
    ):
        with pytest.raises(ConnectionError):
            await runner.initialize_connection()
```

- [ ] **Step 3: Run test, verify it fails**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_runner_initialize_connection.py -v
```

Expected: 2 failures with `AttributeError: ... initialize_connection`.

- [ ] **Step 4: Implement `initialize_connection`**

Add to `class AsyncRunner` in `robo_trader/runner_async.py`, near the existing `setup()` method:

```python
    async def initialize_connection(self) -> None:
        """Establish IBKR connection: start subprocess + connect + stabilize.

        Extracted from run()/setup() so recover_connection() can call the
        same path during runtime recovery. Raises ConnectionError if the
        subprocess fails to connect.

        Per 2025-11-24 handoff: includes 2.0s stabilization wait + isConnected()
        poll AFTER handshake to ensure Gateway has fully published nextValidId.
        """
        import asyncio

        from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient

        client_id = getattr(self, "_client_id", self.cfg.ibkr.client_id)

        client = SubprocessIBKRClient()
        await client.start()
        try:
            result = await client.connect(
                host=self.cfg.ibkr.host,
                port=self.cfg.ibkr.port,
                client_id=client_id,
                readonly=True,
                timeout=10.0,
            )
        except Exception:
            await client.stop()
            raise

        if not result.get("connected"):
            await client.stop()
            raise ConnectionError(
                f"Subprocess connect failed: {result.get('error', 'unknown')}"
            )

        # 2.0s stabilization wait per 2025-11-24 handoff
        await asyncio.sleep(2.0)

        # Poll isConnected() to verify the Gateway-side state is stable
        for _ in range(10):
            if client.isConnected():
                break
            await asyncio.sleep(0.2)
        else:
            await client.stop()
            raise ConnectionError(
                "ib.isConnected() never returned True after 2s+ stabilization"
            )

        self.subprocess_client = client
        self.ib = client
        logger.info(
            "event=connection_initialized client_id=%s accounts=%s",
            client_id,
            result.get("accounts", []),
        )
```

- [ ] **Step 5: Run tests, verify they pass**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_runner_initialize_connection.py -v
```

Expected: `2 passed`.

Also re-run the full suite to make sure we haven't broken anything:

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: same baseline as pre-flight (638+ passed, 3 skipped, **0 failures**).

- [ ] **Step 6: Commit**

```bash
git add robo_trader/runner_async.py tests/test_runner_initialize_connection.py
git commit -m "refactor(runner): extract initialize_connection from run()

Pulls the IBKR-subprocess-start + ib.connect + stabilization-wait
sequence into a separately-callable method. recover_connection (next
task) reuses this. No behavior change vs current run() path - same
2.0s wait, same isConnected() poll, same exception semantics."
```

---

## Task 7: `recover_connection` with exponential backoff

**Files:**
- Modify: `robo_trader/runner_async.py`
- Create: `tests/test_recover_connection.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_recover_connection.py`:

```python
"""Tests for AsyncRunner.recover_connection.

Per 2026-05-16 design spec: exponential backoff [15, 30, 60, 120, 300],
Gateway restart on attempt >=3, returns bool, mutex via _recovery_lock.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


def make_runner_for_recovery(initialize_succeeds_on=None):
    """Build AsyncRunner with stubs.
    initialize_succeeds_on: int N — initialize_connection fails on attempts
    1..N-1 and succeeds on attempt N. None = always succeed."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.recovery_in_progress = False
    runner._recovery_lock = asyncio.Lock()
    runner.ib = MagicMock()
    runner.ib.isConnected = MagicMock(return_value=False)
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()

    runner._safe_disconnect = AsyncMock()

    if initialize_succeeds_on is None:
        runner.initialize_connection = AsyncMock()
    else:
        call_count = {"n": 0}

        async def fail_then_succeed():
            call_count["n"] += 1
            if call_count["n"] < initialize_succeeds_on:
                raise ConnectionError(f"attempt {call_count['n']} fails")

        runner.initialize_connection = fail_then_succeed
    return runner


@pytest.mark.asyncio
async def test_returns_true_on_first_attempt_success():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    # Speed up: patch asyncio.sleep so backoff doesn't take 15s in tests
    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test-reason")
    assert result is True
    assert runner.recovery_in_progress is False


@pytest.mark.asyncio
async def test_first_attempt_does_not_restart_gateway():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies",
        return_value=True,
    ) as gm_restart:
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            await runner.recover_connection("test")
        gm_restart.assert_not_called()


@pytest.mark.asyncio
async def test_third_attempt_restarts_gateway():
    runner = make_runner_for_recovery(initialize_succeeds_on=3)
    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies",
        return_value=True,
    ) as gm_restart:
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            result = await runner.recover_connection("test")
        assert result is True
        gm_restart.assert_called()  # >=1 call on attempt 3


@pytest.mark.asyncio
async def test_returns_false_after_exhausted_attempts():
    runner = make_runner_for_recovery(initialize_succeeds_on=999)  # never succeeds
    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies",
        return_value=True,
    ) as gm_restart:
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            result = await runner.recover_connection("test")
    assert result is False


@pytest.mark.asyncio
async def test_backoff_schedule_is_15_30_60_120_300():
    runner = make_runner_for_recovery(initialize_succeeds_on=999)
    sleeps = []

    async def record_sleep(seconds):
        sleeps.append(seconds)

    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies",
        return_value=True,
    ) as gm_restart:
        with patch(
            "robo_trader.runner_async.asyncio.sleep", side_effect=record_sleep
        ):
            await runner.recover_connection("test")
    assert sleeps == [15, 30, 60, 120, 300]


@pytest.mark.asyncio
async def test_concurrent_invocations_serialize_via_lock():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    call_order = []

    original_init = runner.initialize_connection

    async def slow_init():
        call_order.append("start")
        await asyncio.sleep(0)  # yield to event loop
        call_order.append("end")
        return await original_init()

    runner.initialize_connection = slow_init

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await asyncio.gather(
            runner.recover_connection("first"),
            runner.recover_connection("second"),
        )
    # The two runs must serialize: start, end, start, end (not interleaved)
    assert call_order == ["start", "end", "start", "end"]


@pytest.mark.asyncio
async def test_recovery_in_progress_flag_set_and_cleared():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    flag_during = []

    original_init = runner.initialize_connection

    async def check_flag():
        flag_during.append(runner.recovery_in_progress)
        return await original_init()

    runner.initialize_connection = check_flag

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await runner.recover_connection("test")

    assert flag_during == [True]  # set during init
    assert runner.recovery_in_progress is False  # cleared after
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_recover_connection.py -v
```

Expected: 7 failures, all with `AttributeError: ... recover_connection`.

- [ ] **Step 3: Implement `recover_connection`**

For the Gateway-restart step we reuse the existing helper at
`robo_trader/utils/robust_connection.py:460` (`restart_gateway_for_zombies`),
which already does the subprocess call to `scripts/gateway_manager.py restart`.
This avoids reinventing process-management logic that's been hardened by
prior outages.

Update the test in Step 2 to patch this function instead of `gateway_manager`.
Replace the `with patch("robo_trader.runner_async.gateway_manager") as gm:`
blocks with:

```python
with patch(
    "robo_trader.runner_async.restart_gateway_for_zombies",
    new=AsyncMock(return_value=True),
) as gm_restart:
    # ... test body
    # And assert: gm_restart.assert_not_awaited() / gm_restart.assert_awaited()
```

NOTE: `restart_gateway_for_zombies` is currently SYNCHRONOUS in
`robust_connection.py`. Wrap calls in `asyncio.to_thread()` to avoid
blocking the event loop. The test patches can use `MagicMock` instead of
`AsyncMock` to match.

Then implement:

```python
    async def recover_connection(self, reason: str) -> bool:
        """Re-establish IBKR connection after ConnectionHealth reports unhealthy.

        Returns True if recovered, False if all 5 backoff attempts failed.
        On False, the caller should exit run_continuous and let the watchdog
        do process-level restart.

        Per 2026-05-16 design spec:
        - Backoff: [15, 30, 60, 120, 300] seconds
        - Gateway restart on attempt >= 3
        - Mutex via _recovery_lock (no concurrent recovery)
        - Sets recovery_in_progress flag for cycle-skip logic
        """
        import asyncio

        backoff_schedule = [15, 30, 60, 120, 300]
        gateway_restart_attempt_threshold = 3

        # Module-level import to add near the top of runner_async.py:
        #   from robo_trader.utils.robust_connection import restart_gateway_for_zombies

        async with self._recovery_lock:
            self.recovery_in_progress = True
            try:
                for attempt_idx, delay in enumerate(backoff_schedule):
                    attempt = attempt_idx + 1
                    logger.warning(
                        "event=recovery_started attempt=%d backoff_seconds=%d reason=%r",
                        attempt,
                        delay,
                        reason,
                    )

                    await self._safe_disconnect()
                    await asyncio.sleep(delay)

                    if attempt >= gateway_restart_attempt_threshold:
                        try:
                            # restart_gateway_for_zombies is sync — run off-loop
                            ok = await asyncio.to_thread(
                                restart_gateway_for_zombies,
                                self.cfg.ibkr.port,
                                180,  # timeout seconds
                            )
                            logger.info(
                                "event=recovery_gateway_restart_done attempt=%d success=%s",
                                attempt,
                                ok,
                            )
                        except Exception as e:
                            logger.warning(
                                "event=recovery_gateway_restart_failed attempt=%d error=%r",
                                attempt,
                                e,
                            )

                    try:
                        await self.initialize_connection()
                        logger.info(
                            "event=recovery_succeeded attempt=%d", attempt
                        )
                        return True
                    except Exception as e:
                        logger.warning(
                            "event=recovery_attempt_failed attempt=%d error=%r",
                            attempt,
                            e,
                        )

                logger.error(
                    "event=recovery_exhausted attempts=%d reason=%r",
                    len(backoff_schedule),
                    reason,
                )
                return False
            finally:
                self.recovery_in_progress = False
```

You also need to initialize `_recovery_lock` and `recovery_in_progress` in `AsyncRunner.__init__`. Around line 333 (after `self.cleanup_task = None`):

```python
        self.recovery_in_progress = False
        self._recovery_lock = asyncio.Lock()
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_recover_connection.py -v
```

Expected: `7 passed`.

If the `gateway_manager` import path in the test (`patch("robo_trader.runner_async.gateway_manager")`) doesn't match where your impl imports it, adjust the test patch target or use a top-level import in `runner_async.py` so both tests and impl agree on the symbol.

- [ ] **Step 5: Full-suite regression check**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: still 0 failures.

- [ ] **Step 6: Commit**

```bash
git add robo_trader/runner_async.py tests/test_recover_connection.py
git commit -m "feat(runner): recover_connection with exponential backoff

Backoff schedule [15, 30, 60, 120, 300] seconds. Gateway restart kicks
in on attempt 3+ (saves ~2 min on common transient blips). Returns True
on first success, False after all 5 attempts exhausted. Mutex via
_recovery_lock prevents concurrent recovery. recovery_in_progress flag
allows the cycle loop to skip cycles while recovery is mid-flight.

On exhausted recovery, returns False - caller exits run_continuous and
the watchdog handles process-level restart (Layer 5+6)."
```

---

## Task 8: Wire `ConnectionHealth` into `AsyncRunner`

**Files:**
- Modify: `robo_trader/runner_async.py`
- Create: `tests/test_runner_health_integration.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_runner_health_integration.py`:

```python
"""Tests for ConnectionHealth integration with AsyncRunner.

Verifies that initialize_connection wires up health monitoring and that
the on_unhealthy callback triggers recover_connection."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner
from robo_trader.connection_health import ConnectionHealth, HealthStatus


@pytest.mark.asyncio
async def test_initialize_connection_creates_health_module():
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

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value={"connected": True, "accounts": []})
    fake_client.isConnected = MagicMock(return_value=True)
    fake_client.ping = AsyncMock(return_value=True)

    with patch(
        "robo_trader.runner_async.SubprocessIBKRClient",
        return_value=fake_client,
    ):
        await runner.initialize_connection()

    assert isinstance(runner.health, ConnectionHealth)
    assert runner.health.status is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_unhealthy_callback_invokes_recover_connection():
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.recover_connection = AsyncMock(return_value=True)

    # Manually call the callback that initialize_connection would have wired
    await runner._on_connection_unhealthy("test reason from health monitor")

    runner.recover_connection.assert_awaited_once()
    call_arg = runner.recover_connection.await_args.args[0]
    assert "test reason" in call_arg
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_runner_health_integration.py -v
```

Expected: 2 failures (missing `runner.health` attribute and missing `_on_connection_unhealthy` method).

- [ ] **Step 3: Wire `ConnectionHealth` into `initialize_connection` and add callback**

In `robo_trader/runner_async.py`:

**3a.** Add this method to `AsyncRunner`:

```python
    async def _on_connection_unhealthy(self, reason: str) -> None:
        """Callback fired by ConnectionHealth when threshold hit.

        Invokes recover_connection. If recovery exhausted, the runner exits
        run_continuous - watchdog handles process-level restart."""
        logger.warning(
            "event=health_triggered_recovery reason=%r", reason
        )
        await self.recover_connection(f"health monitor: {reason}")
```

**3b.** At the END of `initialize_connection()`, after `self.ib = client` and the logger.info line, add:

```python
        from robo_trader.connection_health import ConnectionHealth

        # Replace any existing health module from a prior connection
        if getattr(self, "health", None) is not None:
            await self.health.stop_monitoring()

        self.health = ConnectionHealth(
            ib_client=client,
            ping_interval_seconds=30,
            max_consecutive_failures=3,
        )
        await self.health.start_monitoring(on_unhealthy=self._on_connection_unhealthy)
```

**3c.** In `AsyncRunner.__init__`, near line 333 where you added `_recovery_lock`, also add:

```python
        self.health: "ConnectionHealth | None" = None
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_runner_health_integration.py tests/test_runner_initialize_connection.py -v
```

Expected: `4 passed`.

Then full-suite regression:

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: still 0 failures.

- [ ] **Step 5: Commit**

```bash
git add robo_trader/runner_async.py tests/test_runner_health_integration.py
git commit -m "feat(runner): wire ConnectionHealth into initialize_connection

Health monitor starts automatically after a successful connection.
On UNHEALTHY transition, _on_connection_unhealthy callback fires,
which invokes recover_connection. Pre-existing health module from
a prior connection is stopped before a new one starts (idempotent
re-initialization for the recovery-then-reconnect path)."
```

---

## Task 9: Restructure `run_continuous` to use long-lived runner

**Files:**
- Modify: `robo_trader/runner_async.py` (the top-level `run_continuous` function around line 4174-4360)
- Create: `tests/test_run_continuous_persistent.py`

This is the headline behavior change. Before this task, all the pieces are in place but the per-cycle pattern still runs. After this task, the IBKR connection is long-lived.

- [ ] **Step 1: Read current `run_continuous` for the exact replacement scope**

```bash
sed -n '4170,4360p' /Users/oliver/Projects/robo_trader/robo_trader/runner_async.py
```

Note especially:
- The `while not shutdown_flag` outer loop
- The `for portfolio_cfg in active_portfolios` inner loop (multi-portfolio)
- The `runner = AsyncRunner(...)` creation INSIDE the inner loop (this moves OUT)
- The `await runner.cleanup()` call INSIDE the inner loop (this moves OUT)
- The `await asyncio.sleep(interval_seconds)` (this stays)

- [ ] **Step 2: Write failing integration test**

Create `tests/test_run_continuous_persistent.py`:

```python
"""Integration test: run_continuous with persistent connection.

Uses a FakeSubprocessIBKRClient stand-in to verify the AsyncRunner is
created ONCE per portfolio (not per cycle) and the connection persists
across cycles."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


class FakeSubprocessClient:
    """Matches the subset of SubprocessIBKRClient that AsyncRunner uses."""
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
        return {"connected": True, "accounts": ["DUN264991"]}

    def isConnected(self):
        return self._connected

    async def ping(self):
        return True

    async def stop(self):
        type(self).stop_call_count += 1
        self._connected = False

    async def disconnectAsync(self):
        pass


@pytest.mark.asyncio
async def test_persistent_runner_starts_subprocess_only_once_across_cycles():
    """Verify the long-lived-runner property: across N cycles, subprocess
    is started ONCE, not N times."""
    FakeSubprocessClient.instances_created = 0
    FakeSubprocessClient.start_call_count = 0

    # The actual run_continuous + cycle loop is heavy; we test the
    # initialize_connection + (simulated) cycle pattern directly.
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

    with patch(
        "robo_trader.runner_async.SubprocessIBKRClient",
        FakeSubprocessClient,
    ):
        await runner.initialize_connection()
        # Simulate 3 cycles that each call teardown(full_cleanup=False)
        for _ in range(3):
            await runner.teardown(full_cleanup=False)

        # The persistent contract: subprocess was started ONCE only
        assert FakeSubprocessClient.start_call_count == 1
        # And NOT stopped between cycles
        assert FakeSubprocessClient.stop_call_count == 0
        # Cleanup at end
        await runner.health.stop_monitoring()
```

- [ ] **Step 3: Run test, verify it fails or passes (depends on initial state)**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_run_continuous_persistent.py -v
```

This test should already PASS if Task 8 wired things up correctly — it's verifying that `teardown(full_cleanup=False)` doesn't kill the subprocess. If it fails, debug before proceeding.

Expected on this run: 1 passed.

- [ ] **Step 4: Replace `run_continuous` to keep runner alive across cycles**

In `robo_trader/runner_async.py`, find `async def run_continuous(` (around line 4174). Replace its body. The structure changes from:

```python
# OLD (current):
while not shutdown_flag:
    for portfolio_cfg in active_portfolios:
        runner = AsyncRunner(...)         # FRESH every cycle
        await runner.run(portfolio_symbols)
        await runner.cleanup()            # FRESH disconnect every cycle
        runner = None
    await asyncio.sleep(interval_seconds)
```

to:

```python
# NEW:
runners: dict[str, AsyncRunner] = {}  # portfolio_id -> long-lived runner
try:
    while not shutdown_flag:
        for portfolio_cfg in active_portfolios:
            portfolio_id = portfolio_cfg.id

            if portfolio_id not in runners:
                # First cycle for this portfolio - create long-lived runner
                runner = AsyncRunner(
                    duration=duration,
                    bar_size=bar_size,
                    sma_fast=sma_fast,
                    sma_slow=sma_slow,
                    slippage_bps=slippage_bps,
                    max_order_notional=max_order_notional,
                    max_daily_notional=max_daily_notional,
                    default_cash=portfolio_cfg.starting_cash,
                    max_concurrent_symbols=max_concurrent,
                    use_correlation_sizing=True,
                    use_ml_strategy=use_ml_strategy,
                    use_smart_execution=use_smart_execution,
                    portfolio_id=portfolio_id,
                )
                await runner.setup()
                await runner.initialize_connection()
                runners[portfolio_id] = runner
            else:
                runner = runners[portfolio_id]

            # Recovery in progress? Skip this cycle for this portfolio.
            if runner.recovery_in_progress:
                logger.info(
                    "event=cycle_skipped_recovery_in_progress portfolio=%s",
                    portfolio_id,
                )
                continue

            # Cycle health gate
            from robo_trader.connection_health import HealthStatus
            if runner.health is not None and runner.health.status is not HealthStatus.HEALTHY:
                logger.warning(
                    "event=cycle_skipped_unhealthy portfolio=%s status=%s",
                    portfolio_id,
                    runner.health.status.value,
                )
                continue

            symbols = portfolio_cfg.symbols if portfolio_cfg.symbols else symbols
            try:
                await runner.run(symbols)
            except Exception as e:
                logger.exception(
                    "event=cycle_error portfolio=%s error=%r",
                    portfolio_id,
                    e,
                )
                # Health monitor will catch persistent issues; let recovery
                # decide whether to escalate.
                if runner.health is not None:
                    runner.health.record_failure(e, f"cycle:{portfolio_id}")

            await runner.teardown(full_cleanup=False)

        if not shutdown_flag and is_trading_allowed():
            logger.info(
                "Waiting %.1f minutes before next iteration...",
                interval_seconds / 60,
            )
            await asyncio.sleep(interval_seconds)
finally:
    # Final cleanup: ALL portfolios get full disconnect on process exit
    for portfolio_id, runner in runners.items():
        try:
            await runner.cleanup()
        except Exception:
            logger.exception(
                "event=final_cleanup_failed portfolio=%s", portfolio_id
            )
```

- [ ] **Step 5: Run new integration test plus regression**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/test_run_continuous_persistent.py -v
```

Expected: passed.

Then full suite:

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

Expected: still 0 failures, total count may have increased slightly with new tests.

If any existing test breaks here, **STOP** and read the failure carefully. This is the most likely task to introduce regressions because it changes behavior in the critical path. Common pitfall: a test that creates an `AsyncRunner` directly and assumed certain init-time setup is now in `initialize_connection()` — those tests need `initialize_connection()` calls or adjustment.

- [ ] **Step 6: Commit**

```bash
git add robo_trader/runner_async.py tests/test_run_continuous_persistent.py
git commit -m "feat(runner): persistent connection across cycles via run_continuous

The headline behavior change. AsyncRunner is now created ONCE per
portfolio at the run_continuous scope, NOT once per cycle. The IBKR
connection persists across all trading cycles. teardown(full_cleanup=False)
between cycles stops monitors but keeps subprocess + ib.connect alive.

Cycles skip themselves cleanly if:
- recovery_in_progress is set (mutex with recover_connection)
- health.status is not HEALTHY (gives recovery time to work)

Cycle exceptions get recorded via health.record_failure() and the
background ConnectionHealth monitor decides whether to escalate to
recover_connection.

On process exit (shutdown_flag, exception), all portfolios get final
cleanup() via the try/finally."
```

---

## Task 10: Delete `_monitor_subprocess_health` (replaced by `ConnectionHealth`)

**Files:**
- Modify: `robo_trader/runner_async.py`

- [ ] **Step 1: Confirm `_monitor_subprocess_health` is no longer referenced**

```bash
grep -n "_monitor_subprocess_health\|_restart_subprocess" /Users/oliver/Projects/robo_trader/robo_trader/runner_async.py | head -10
```

Note all callers. They should be:
- The method definition itself
- Possibly a `create_task` invocation in `setup()` or similar

- [ ] **Step 2: Remove the method and any `create_task` invocations**

Delete `async def _monitor_subprocess_health(self)` (around line 1630, ~50 LOC) AND any line that creates a task from it (look for `create_task(self._monitor_subprocess_health` in `setup()`).

Also delete the helper `_restart_subprocess` if it's no longer called from anywhere else.

- [ ] **Step 3: Run full suite**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

Expected: still 0 failures.

- [ ] **Step 4: Commit**

```bash
git add robo_trader/runner_async.py
git commit -m "refactor(runner): remove _monitor_subprocess_health (replaced by ConnectionHealth)

The previous in-runner health monitor (60s interval, 3-failure threshold,
restart-subprocess-only response) is replaced by ConnectionHealth which:
- Centralizes health logic in one module with explicit tests
- Has the SAME 30s ping interval and 3-failure threshold
- Escalates correctly to recover_connection() (not just subprocess restart)

_restart_subprocess removed if unreferenced - if any other caller exists
it's untouched."
```

---

## Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the new Common Mistake entry**

In `CLAUDE.md`, find the table under `### Trading Logic Errors` (a giant Common Mistakes table near the bottom). Add this row at the bottom:

```markdown
| Adding per-cycle IBKR disconnect/reconnect in run_continuous | Connection is long-lived under run_continuous; cycles must reuse via teardown(full_cleanup=False). Re-introducing per-cycle disconnect causes 2026-05-13-style IBKR-throttle cascade | 2026-05-16 |
```

Also update the `Current Issues Status` section (near the top, around line 200), changing entry 14+ to include:

```markdown
14. ✅ Persistent IBKR Connection - IMPLEMENTED (2026-05-16, prevents IBKR-throttle cascade)
```

(If there is no entry 14 yet, add it as the next number after the existing last entry.)

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude-md): document persistent-connection invariant

Adds a Common Mistakes entry warning future contributors not to
re-introduce per-cycle disconnect logic, which would resurrect the
2026-05-13 IBKR-throttle cascade. Marks persistent-connection feature
in the Current Issues Status list."
```

---

## Task 12: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite green**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: 638+ tests passed, 3 skipped, **0 failures**. New tests from this plan should add ~30+ tests.

- [ ] **Step 2: CLAUDE.md-pinned regression tests specifically**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/security/test_web.py -v 2>&1 | tail -10
```

Expected: 42 passed including:
- `test_ws_request_headers_shim_handles_v15_api`
- `test_ws_auth_end_to_end_against_real_library`

- [ ] **Step 3: Black on changed files only (skip pre-existing drift)**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m black --check robo_trader/connection_health.py tests/test_connection_health.py tests/test_recover_connection.py tests/test_runner_safe_disconnect.py tests/test_runner_initialize_connection.py tests/test_runner_health_integration.py tests/test_run_continuous_persistent.py 2>&1 | tail -5
```

Expected: "All done" or "would be left unchanged" on all listed files.

If `runner_async.py` shows new drift in our new code (vs pre-existing drift), apply black to ONLY the new code:

```bash
# DO NOT run black on the entire runner_async.py — it has pre-existing drift
# that we agreed not to clean up in this branch.
```

- [ ] **Step 4: Flake8 critical errors on new code**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m flake8 robo_trader/connection_health.py --count --select=E9,F63,F7,F82 --show-source --statistics 2>&1 | tail -3
```

Expected: `0`.

- [ ] **Step 5: Import smoke test**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -c "
from robo_trader.connection_health import ConnectionHealth, HealthStatus
from robo_trader.runner_async import AsyncRunner
r = AsyncRunner.__new__(AsyncRunner)
assert hasattr(r, '_safe_disconnect') is False or callable(getattr(AsyncRunner, '_safe_disconnect'))
assert callable(getattr(AsyncRunner, 'initialize_connection'))
assert callable(getattr(AsyncRunner, 'recover_connection'))
assert callable(getattr(AsyncRunner, '_on_connection_unhealthy'))
print('Smoke OK')
"
```

Expected: `Smoke OK`.

---

## Task 13: Manual canary — 30-minute idle test

**Files:** none (operator task)

This is **not** code. It's the canary plan from the spec, executed by you (the operator) on this branch before any merge.

- [ ] **Step 1: Pre-canary system state**

```bash
launchctl unload ~/Library/LaunchAgents/com.robotrader.watchdog.plist  # Disable watchdog during canary
rm -f /Users/oliver/Projects/robo_trader/.watchdog_failures             # Clear failure state
pkill -9 -f "python.*runner_async" 2>/dev/null
pkill -9 -f "python.*app.py" 2>/dev/null
pkill -9 -f "python.*websocket_server" 2>/dev/null
sleep 3
ps aux | grep -E "runner_async|app.py|websocket_server" | grep -v grep || echo "All processes clean"
```

- [ ] **Step 2: Start trader on this branch**

```bash
cd /Users/oliver/Projects/robo_trader
git status  # Confirm clean tree on feature/persistent-ibkr-connection
./START_TRADER.sh
```

When Gateway comes up and IBKR connects, **leave it running for 30 minutes**.

- [ ] **Step 3: While running, verify persistent-connection behavior**

In a separate terminal:

```bash
# Count Disconnecting events in the last 5 minutes - should be ~0 with persistent connection
tail -10000 /Users/oliver/Projects/robo_trader/robo_trader.log | grep -c "Disconnecting from IBKR"
# Expected: 0 (vs ~25 with old per-cycle architecture in same window)

# Count connection_state_change events - should be 1 initial transition only
grep "connection_state_change" /Users/oliver/Projects/robo_trader/robo_trader.log | tail -20
# Expected: 1 initial "to=HEALTHY", no further state transitions

# Count Trading cycle complete events - should be ~150 in 30 min (12s cycles)
tail -10000 /Users/oliver/Projects/robo_trader/robo_trader.log | grep -c "Trading cycle complete"
# Expected: ~150
```

If any of the above shows the **old** behavior (many Disconnecting events, multiple state transitions), the persistent-connection wiring is broken. Stop and debug before proceeding.

- [ ] **Step 4: Tear down canary cleanly**

```bash
pkill -9 -f "python.*runner_async"
pkill -9 -f "python.*app.py"
pkill -9 -f "python.*websocket_server"
launchctl load ~/Library/LaunchAgents/com.robotrader.watchdog.plist  # Restore watchdog
```

- [ ] **Step 5: Note results in the design doc**

If canary passed, append a `## 10. Canary Results` section to `docs/superpowers/specs/2026-05-16-persistent-ibkr-connection-design.md` with the actual numbers observed (cycles completed, disconnect-events, state changes).

Commit:

```bash
git add docs/superpowers/specs/2026-05-16-persistent-ibkr-connection-design.md
git commit -m "docs(spec): record 30-minute canary results"
```

---

## Task 14: Manual canary — forced-failure test

**Files:** none (operator task)

- [ ] **Step 1: Start fresh, with watchdog DISABLED during test**

```bash
launchctl unload ~/Library/LaunchAgents/com.robotrader.watchdog.plist
rm -f /Users/oliver/Projects/robo_trader/.watchdog_failures
./START_TRADER.sh
# Wait for trader to be connected and trading (5-10 min)
```

- [ ] **Step 2: Verify trader is healthy**

```bash
ps aux | grep "python.*runner_async" | grep -v grep
grep "Trading cycle complete" /Users/oliver/Projects/robo_trader/robo_trader.log | tail -1
# Should show a recent cycle within the last 30 seconds
```

- [ ] **Step 3: Kill Gateway, watch recovery**

```bash
# Force Gateway down WITHOUT killing runner
pkill -f "IB Gateway"

# Now watch the runner's response
tail -f /Users/oliver/Projects/robo_trader/robo_trader.log | grep -E "recovery_started|recovery_succeeded|recovery_attempt_failed|recovery_exhausted|connection_state_change"
```

Expected within 60-90 seconds:
- `event=connection_state_change from=HEALTHY to=UNHEALTHY`
- `event=recovery_started attempt=1`
- After backoff: `event=recovery_attempt_failed attempt=1` OR `event=recovery_succeeded attempt=1`
- If attempts go to 3+: `event=recovery_gateway_restart_done`
- Eventually: `event=recovery_succeeded` AND `event=connection_state_change from=RECOVERING to=HEALTHY`

Critical check: **runner_async process must NOT die**. Run in another terminal during recovery:

```bash
watch -n 5 'ps aux | grep "python.*runner_async" | grep -v grep'
# PID should be stable throughout the recovery
```

- [ ] **Step 4: Tear down**

```bash
pkill -9 -f "python.*runner_async"
pkill -9 -f "python.*app.py"
pkill -9 -f "python.*websocket_server"
launchctl load ~/Library/LaunchAgents/com.robotrader.watchdog.plist
```

- [ ] **Step 5: Record results, commit**

If the runner survived the Gateway kill and recovered the connection without process restart, the design is working as intended.

```bash
git commit --allow-empty -m "canary: forced-failure recovery test passed

Manually killed IB Gateway with pkill -f 'IB Gateway' while runner was
trading. Recovery completed within X attempts (Y seconds total). The
runner_async process PID stayed stable throughout - no watchdog restart
was needed. This validates the persistent-connection + in-runner-recovery
architecture vs the old fresh-runner-each-cycle pattern."
```

(Replace X, Y with actual numbers observed.)

---

## Task 15: 24-hour soak (Sunday afternoon -> Monday 4 AM ET)

**Files:** none (operator task)

This is the final pre-merge gate. Run it on a Sunday before a Monday market open.

- [ ] **Step 1: Start clean Sunday afternoon**

```bash
# Around Sunday 14:00-16:00 ET
launchctl load ~/Library/LaunchAgents/com.robotrader.watchdog.plist  # Watchdog ON for soak
rm -f /Users/oliver/Projects/robo_trader/.watchdog_failures
./START_TRADER.sh
```

- [ ] **Step 2: Sunday 23:30 ET — pre-Gateway-auto-restart check**

```bash
date
ps -p $(pgrep -f "python.*runner_async") -o pid,etime
grep -c "Trading cycle complete" /Users/oliver/Projects/robo_trader/robo_trader.log
grep "connection_state_change" /Users/oliver/Projects/robo_trader/robo_trader.log | tail -5
```

Note: runner uptime should be ~8 hours by now. State changes should be 1 (initial HEALTHY).

- [ ] **Step 3: Sunday 23:50 ET — watch the 23:45 ET Gateway auto-restart**

The IBC config has `AutoRestartTime=11:45 PM`. Watch the runner's response:

```bash
tail -f /Users/oliver/Projects/robo_trader/robo_trader.log | grep -E "recovery_|connection_state_change"
```

Expected around 23:45-23:50: ONE recovery cycle, succeeds in <2 min.

- [ ] **Step 4: Monday 04:15 ET — extended hours start check**

```bash
date
# Verify runner_async is STILL the same PID it was Sunday afternoon
ps -p $(pgrep -f "python.*runner_async") -o pid,etime
# Should show ~12 hours of uptime, including the 23:45 recovery

# Trading cycles should have resumed already at 04:00 ET extended-hours start
grep "Trading cycle complete" /Users/oliver/Projects/robo_trader/robo_trader.log | tail -1
```

- [ ] **Step 5: Decision point**

If all checks pass:
- Runner survived the soak
- Recovery cycle at 23:45 worked
- Extended hours resumed cleanly
- **Ready to merge to main and let Monday's open happen on this branch**

If anything failed:
- Tear down, stay on main, investigate before re-trying

- [ ] **Step 6: Record results**

```bash
git commit --allow-empty -m "canary: 24-hour soak passed

Ran from Sunday DDDD to Monday DDDD. Stats:
- Runner uptime: NN hours
- Trading cycles completed: NNNN
- connection_state_change events: NN (expected: ~3 = init, scheduled restart, post-restart)
- recovery_started events: NN (expected: 1, the 23:45 scheduled restart)
- recovery_exhausted events: 0
- Watchdog restart count: 0

Branch ready for merge to main."
```

---

## Task 16: Final pre-merge checklist

**Files:** none (verification only)

- [ ] **Step 1: Branch clean**

```bash
cd /Users/oliver/Projects/robo_trader
git status
git branch --show-current  # feature/persistent-ibkr-connection
git log --oneline main..HEAD  # Should show all task commits
```

- [ ] **Step 2: All three canaries committed**

```bash
git log --oneline | grep -E "canary:" | head -5
# Expect: 30-min, forced-failure, 24h-soak commits all present
```

- [ ] **Step 3: Full test suite one more time**

```bash
/Users/oliver/Projects/robo_trader/.venv/bin/python3 -m pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: still 0 failures.

- [ ] **Step 4: Hand to user for merge decision**

Do NOT merge automatically. Tell the user:

> "All 16 tasks complete. 30-min canary, forced-failure canary, and 24-hour soak all green. Full test suite (X passed, 3 skipped, 0 failed). Ready for your call on merging `feature/persistent-ibkr-connection` to `main`."

---

## Rollback plan (if anything goes wrong post-merge)

If the persistent-connection architecture causes problems in production:

```bash
# Revert the merge commit (creates a new commit on main)
git checkout main
git revert -m 1 <merge-commit-sha>
git push origin main

# Or, if not yet pushed, simple branch reset
git reset --hard <pre-merge-commit-sha>
```

Then restart the trader:

```bash
./START_TRADER.sh
```

The previous fresh-runner-each-cycle architecture is now back. Reproduce the issue, file a postmortem, plan a fix on a new branch.

---

## Appendix: Why each file was chosen

- `robo_trader/connection_health.py` — single responsibility (connection health), small enough to hold in context, testable without IBKR
- `robo_trader/runner_async.py` — already the home of `AsyncRunner` and `run_continuous`. Splitting these out would be a bigger refactor (>2000 LOC file) than this change should attempt.
- `tests/test_*.py` — one test file per new module/concept. Avoids dumping into existing test files which have their own scope.
- `docs/superpowers/specs/...` — design spec lives next to plan, both versioned
- `CLAUDE.md` — only the Common Mistakes section is touched; targeted, low-risk

## Appendix: Critical rules reference

1. **NEVER use `socket.connect_ex()` for port checks** — use `lsof` (2026-01-05 handoff)
2. **NEVER call `ib.disconnect()` on a failed connection** — crashes Gateway API layer (2025-11-20 handoff)
3. **NEVER reuse a dead `SubprocessIBKRClient` instance** — always instantiate fresh (2025-12-24 handoff)
4. **ALWAYS 2.0s stabilization wait + `isConnected()` poll** after Gateway handshake (2025-11-24 handoff)
5. **ALWAYS commit per task** — don't batch across tasks; per-task commits are the rollback granularity
