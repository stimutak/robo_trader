# Persistent IBKR Connection — Design

**Status:** Draft — pending implementation
**Branch:** `feature/persistent-ibkr-connection`
**Date:** 2026-05-16
**Supersedes:** The 2025-12-10 persistent-connection plan referenced in `handoff/HANDOFF_2026-02-03_*` (proposed but never executed)

---

## 1. Why this exists

### Incident that surfaced it
On 2026-05-13 12:01:27 the runner exited cleanly with `IBKR Gateway API layer is unresponsive. Automatic restart failed. Please restart Gateway manually`. The trigger was a server-side TCP reset (`Errno 54 Connection reset by peer`) 1.1 seconds after a fresh connect. The watchdog then fired ~330 restart attempts over the next 22 hours; every single one timed out at `START_TRADER.sh:140`'s 120-second wait. The 2026-05-15 outage repeated the same pattern (8 failures during Friday trading hours before market close).

### Root cause established by investigation
- Today's runner architecture creates a **fresh `AsyncRunner` every 12 seconds** (`runner_async.py:4304-4326`) — see comment: *"Create fresh runner each cycle for stability (disconnect between cycles). This avoids connection timeouts and subprocess health check issues."*
- Each cycle: subprocess spawn → `ib.connect()` → trade → `ib.disconnect()` → 12s sleep → repeat
- Steady-state: ~7,200 logins per day, ~50% of wall-clock time in a "disconnected" state
- When IBKR's auth backend throttled this account on 2026-05-13, every reconnect attempt fed the throttle further. Gateway's "Connecting to server..." dialog (IBC log, line 13246+) showed 6+ minute waits in failed cycles vs. ~0s in successful cycles
- `START_TRADER.sh`'s 120-second hard timeout meant Gateway was repeatedly SIGTERM'd mid-handshake by the next watchdog cycle's `pkill -f "IB Gateway"`. IBC exit status 143 appears in every failed cycle's IBC log.

### Why the workaround pattern is no longer needed
The 2025-11-24 handoff explains the original `AsyncRunner`-per-cycle decision: async-context conflicts when reusing the same `ib_async` session across cycles. **Subprocess isolation** (`SubprocessIBKRClient`, also 2025-11-24) eliminated those conflicts. The fresh-runner pattern is now a stability workaround for problems that no longer exist.

It also directly caused the 2026-01-26 $5M duplicate-buy disaster: `_pending_orders` in-memory state reset every 12 seconds. A persistent runner removes that entire failure class.

### What this design changes
Move the IBKR connection lifecycle from per-cycle scope to `run_continuous()` scope. Reconnect only on detected failure, with bounded exponential backoff and Gateway restart escalation after attempt 3+. Add a `ConnectionHealth` module as the single decision point for "is the connection usable?", replacing health logic currently scattered across `subprocess_ibkr_client.py`, `connection_manager.py`, `runner_async._monitor_subprocess_health`, and `START_TRADER.sh`.

### Non-goals
- Bypassing IBKR's session-level rate limits (out of our control)
- Replacing `START_TRADER.sh` (used for initial bootstrap and as watchdog last-resort recovery — keeps existing surface)
- Touching the subprocess worker (`ibkr_subprocess_worker.py`) — the isolation pattern is what makes persistent connections safe
- Removing watchdog layers 1-6 — the outer safety net stays, just gets invoked far less frequently
- Changing the multi-portfolio scoping model

---

## 2. Architecture

### Before
```
run_continuous() loop                        outer, infinite
    every 12s:
         AsyncRunner()                       FRESH instance every cycle
         |- subprocess_ibkr_client.start()   FRESH subprocess every cycle
         |- ib.connect()                     FRESH IBKR session every cycle
         |- run trading cycle
         '- cleanup() -> ib.disconnect()     FRESH disconnect every cycle
```

### After
```
run_continuous() loop                        outer, infinite
    AsyncRunner()                            created ONCE, long-lived
    |- initialize_connection()
    |  |- subprocess_ibkr_client.start()
    |  |- ib.connect() + 2.0s stabilization wait
    |  |- ib.isConnected() poll
    |  '- portfolio sync from DB
    |
    |- ConnectionHealth.start_monitoring()   background task, 30s ping interval
    |
    |- every 12s:
    |  |- check health.status; if UNHEALTHY raise ConnectionUnhealthy
    |  |- run(symbols)                       reuses connection
    |  '- teardown(full_cleanup=False)       stops cycle monitors only
    |
    |- on ConnectionUnhealthy:
    |  |- recover_connection(reason)
    |  |  |- attempt 1: cleanup + sleep 15s + reconnect
    |  |  |- attempt 2: cleanup + sleep 30s + reconnect
    |  |  |- attempt 3: cleanup + sleep 60s + gateway_manager.restart() + reconnect
    |  |  |- attempt 4: cleanup + sleep 120s + restart + reconnect
    |  |  '- attempt 5: cleanup + sleep 300s + restart + reconnect
    |  |
    |  '- on exhausted (returns False): break loop, exit process
    |
    '- watchdog Layer 5+6 handles process-level restart (existing, unchanged)
```

### Three architectural decisions worth flagging

1. **The IBKR connection lives at the `run_continuous` scope, not the per-cycle scope.** `AsyncRunner` is no longer destroyed and recreated every 12 seconds.
2. **`teardown(full_cleanup=False)` is finally used as documented.** Between cycles it stops monitors and leaves the connection alone. This was the half-implemented hook proposed in the 2025-12-10 plan.
3. **A new `ConnectionHealth` boundary** decides "still healthy?" vs "needs recovery?" — currently scattered across four files.

### Multi-portfolio
Each portfolio gets its own *long-lived* `AsyncRunner`. They share the IBKR subprocess connection (single `client_id=1`) by serializing access — only one portfolio's `run(symbols)` runs at a time (already serial today via the outer `for portfolio_cfg in active_portfolios` loop). `ConnectionHealth` is per-`AsyncRunner`, matching existing scoping.

---

## 3. Components

### New files

#### `robo_trader/connection_health.py` (~150 LOC)

Centralized health gate. Replaces ad-hoc checks across `subprocess_ibkr_client.py`, `connection_manager.py`, and `runner_async._monitor_subprocess_health`.

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"        # consecutive_failures < max_consecutive_failures
    UNHEALTHY = "unhealthy"    # consecutive_failures >= max_consecutive_failures
    RECOVERING = "recovering"  # recover_connection() is mid-flight

class ConnectionHealth:
    def __init__(
        self,
        ib_client,
        ping_interval_seconds: int = 30,
        max_consecutive_failures: int = 3,
    ): ...

    async def perform_check(self) -> HealthStatus:
        """Active probe: subprocess ping + ib.isConnected()."""

    def record_failure(self, error: Exception, context: str) -> None:
        """External callers (cycles) report transient errors here.
        Doesn't immediately transition state; next perform_check decides."""

    def record_success(self) -> None:
        """Reset failure counter."""

    @property
    def status(self) -> HealthStatus: ...

    async def start_monitoring(self, on_unhealthy: Callable[[str], Awaitable]) -> None:
        """Background loop. Calls on_unhealthy(reason) when threshold hit.
        Survives perform_check exceptions (fails safe to UNHEALTHY, keeps trying)."""

    async def stop_monitoring(self) -> None: ...
```

### Modified files

#### `robo_trader/runner_async.py`

| Change | Location | Notes |
|---|---|---|
| `run_continuous()` restructured | ~lines 4250-4360 | Single long-lived `AsyncRunner` per portfolio; cycles reuse connection; `try/except ConnectionUnhealthy` wraps cycle body |
| `AsyncRunner.__init__` | ~line 330 | Instantiate `self.health = ConnectionHealth(...)`; add `self.recovery_in_progress = False`; add `self._recovery_lock = asyncio.Lock()` |
| `AsyncRunner.initialize_connection()` | NEW method | Extract subprocess + ib.connect + stabilization wait from current `run()` startup. Callable from both initial start and recovery |
| `AsyncRunner.recover_connection(reason)` | NEW method, ~80 LOC | Exponential backoff `[15, 30, 60, 120, 300]`. Gateway restart on attempt >=3. Returns bool. Mutex via `_recovery_lock`. |
| `AsyncRunner._safe_disconnect()` | NEW helper, ~20 LOC | Only calls `ib.disconnect()` if `ib.isConnected()` — prevents the Gateway-crash regression (2025-11-20) |
| `AsyncRunner._monitor_subprocess_health()` | DELETED, ~line 1630 | Replaced by `ConnectionHealth` |
| `AsyncRunner.teardown()` | ~line 1610 | Docstring updated to reflect that `full_cleanup=False` now has a real caller (corrects misleading docstring from commit `1a68a0a`) |

#### `CLAUDE.md`
Add entry under `Common Mistakes`:

> "Adding per-cycle IBKR disconnect/reconnect logic | The connection is long-lived under run_continuous(); cycles must reuse it via teardown(full_cleanup=False). Adding disconnect-in-cycle re-introduces the throttle cascade from 2026-05-13. | 2026-05-16"

### Unchanged (deliberately)

| File | Why |
|---|---|
| `robo_trader/clients/subprocess_ibkr_client.py` | Subprocess isolation is what makes persistent connections safe (per 2025-11-24 handoff). Reuse its `ping()` method. |
| `robo_trader/clients/ibkr_subprocess_worker.py` | Worker process logic — no changes needed |
| `robo_trader/utils/robust_connection.py` | Zombie cleanup, lsof-based checks (NEVER touch — per 2026-01-05 handoff) |
| `scripts/watchdog.sh` | Layer 1-6 outer safety net stays. The runner exits cleanly on exhausted recovery; watchdog handles process-level restart. |
| `START_TRADER.sh` | Used for initial bootstrap and watchdog last-resort recovery only. Still has the 120s race bug, but becomes ~once-per-outage instead of every-12-seconds. Out of scope for this change. |
| `robo_trader/circuit_breaker.py` | Already wraps connect attempts. Keep. |
| Database-level duplicate-trade protection | Defense in depth — keep even though the racing reason for it is removed |

### Lines-changed estimate

| File | Type | Lines |
|---|---|---|
| `robo_trader/connection_health.py` | NEW | ~150 |
| `robo_trader/runner_async.py` | MODIFIED | ~120 changed |
| `tests/test_connection_health.py` | NEW | ~250 |
| `tests/test_recover_connection.py` | NEW | ~200 |
| `tests/test_run_continuous_persistent.py` | NEW | ~250 |
| `CLAUDE.md` | MODIFIED | +1 row in Common Mistakes |
| `docs/superpowers/specs/2026-05-16-persistent-ibkr-connection-design.md` | NEW (this doc) | — |

---

## 4. Data flow and failure recovery

### Happy path (cycle N -> cycle N+1)
- `run_continuous` starts: `initialize_connection()` -> `health.start_monitoring()` -> portfolio sync
- Every 12s: read `health.status`; if `HEALTHY`, call `runner.run(symbols)`; on completion `runner.teardown(full_cleanup=False)`. If `UNHEALTHY` or `RECOVERING`, skip the cycle and let recovery run.
- `ConnectionHealth` pings every 30s in parallel, independent of cycles

### Failure classes

#### Class 1 — Transient error inside a cycle (recoverable, no reconnect)
A single `reqHistoricalData` raises `Errno 54`.
- `health.record_failure(err, "historical_data")` increments counter (e.g., 1/3)
- Cycle aborts that symbol, continues with next
- Next 30s health ping succeeds -> counter resets

**No reconnect.** Today this triggers a full disconnect-reconnect; in the new design it doesn't.

#### Class 2 — Connection genuinely unhealthy
- `health.perform_check()` ping fails 3 consecutive times
- `health.status = UNHEALTHY`
- `on_unhealthy` callback fires -> raises `ConnectionUnhealthy` at next cycle boundary
- `recover_connection("3 consecutive ping failures")` runs:
  - attempt 1: cleanup + sleep 15s + reconnect
  - attempt 2: cleanup + sleep 30s + reconnect
  - attempt 3: cleanup + sleep 60s + `gateway_manager.restart()` + reconnect
  - attempts 4-5: same with 120s and 300s backoff
- On success: resume `run_continuous` loop with fresh connection

#### Class 3 — Recovery exhausted
- All 5 backoff attempts fail
- `recover_connection` returns False
- `RUNNER_EXIT_EVENT(reason="connection_recovery_exhausted")` logged
- `_fire_runner_exit_alert(...)` fires
- `run_continuous` loop breaks; process exits cleanly
- Watchdog detects stale log -> Layer 5 process restart -> Layer 6 notification if also failing at process level

### Error taxonomy

| Type | Source | Detection | Response |
|---|---|---|---|
| Cycle-local error | One symbol's call raises | Call site catches | Skip symbol; cycle continues |
| Subprocess pipe error | `BrokenPipeError`, `EOFError` | Client raises into cycle | `health.record_failure()` + raise `ConnectionUnhealthy` |
| Errno 54 / Connection reset | TCP layer | Wrapped client exception | `health.record_failure()`; multiple in one cycle -> `ConnectionUnhealthy` |
| API handshake timeout (during recovery) | `ib.connect()` | Timeout exception | Recovery attempt fails -> next backoff iteration |
| Gateway scheduled restart (11:45 PM daily) | All API calls fail simultaneously | Health ping fails | Standard recovery path; succeeds in ~30s |
| Gateway crash (kill -9 outside) | Process gone, port not listening | Health ping fails AND lsof shows no listener | Recovery escalates to `gateway_manager.restart()` |
| Account locked by IBKR | Server-side error code | IBKR error callback | Recovery exhausted -> exit, alert user |
| macOS sleep/wake | Wall-clock jump | Detect in health loop | Force reconnect on resume |

### Edge cases from prior handoffs

**Edge 1 (2025-11-20):** Calling `ib.disconnect()` on a failed connection crashes Gateway. New `_safe_disconnect()` only calls `disconnect` when `ib.isConnected()` is True.

**Edge 2 (2026-01-05):** Use `lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT` for port/zombie checks. NEVER `socket.connect_ex()`. Recovery clears Python-owned zombies before reconnect; Gateway-owned zombies require Gateway restart (attempt 3+).

**Edge 3 (2025-12-24):** Subprocess pipe race on reconnect. Use a fresh `SubprocessIBKRClient` instance — don't reuse the dead one. Dedicated stdin reader thread pattern (not `run_in_executor(timeout=...)`).

**Edge 4:** ConnectionHealth itself crashes -> log with stack trace, set status to UNHEALTHY, keep trying. Never silently die.

### Idempotency

| Operation | Idempotent? | Mitigation |
|---|---|---|
| `recover_connection()` | Yes, expensive | `_recovery_lock` mutex |
| `health.record_failure()` | Yes (counter, capped) | — |
| `gateway_manager.restart()` | Yes | Only attempt 3+ |
| New cycle during recovery | Must NOT | `if recovery_in_progress: skip` |
| Stop-loss orders persist | Yes (in DB) | `load_existing_positions()` re-runs after `initialize_connection`, recreates stop-loss monitors (per 2026-02-03 fix) |

### Logging

Each connection state transition logs at INFO with structured fields:

```
event=connection_state_change from=HEALTHY to=UNHEALTHY reason="3 consecutive ping failures"
event=recovery_started attempt=1 backoff_seconds=15
event=recovery_attempt_failed attempt=1 error="TimeoutError"
event=recovery_succeeded attempt=2 elapsed_seconds=47
event=connection_state_change from=RECOVERING to=HEALTHY
```

### Failures that should kill the process (not recover)

Per CLAUDE.md / prior runner-exit-alert work — these go through `_fire_runner_exit_alert()` and let the watchdog do process-level restart:

- Recovery exhausted (all 5 backoff attempts failed)
- Kill switch triggered (existing `KillSwitchTriggeredError`)
- Database corruption / unreadable state
- Unhandled exception in `run_continuous` itself

---

## 5. Testing strategy

### Test pyramid

- **Unit (~25 tests):** `ConnectionHealth`, `recover_connection`, `_safe_disconnect` — pure-Python, no IBKR, no subprocess
- **Integration (~10 tests):** Full `run_continuous` loop with `FakeSubprocessIBKRClient`
- **Regression:** Every test in `tests/security/` must stay green (especially the WS-shim tests called out in CLAUDE.md)
- **Manual canary:** 30-min idle, forced-failure, 24-hour soak before Monday open

### Unit tests — `tests/test_connection_health.py` (new, ~250 LOC)

State machine:
- `test_initial_status_is_healthy`
- `test_record_failure_below_threshold_stays_healthy`
- `test_record_failure_at_threshold_transitions_unhealthy`
- `test_record_success_resets_counter`
- `test_status_transitions_emit_log_events`

Heartbeat:
- `test_perform_check_calls_subprocess_ping`
- `test_perform_check_failure_increments_counter`
- `test_perform_check_success_resets_counter`
- `test_perform_check_respects_ib_not_connected`

Background monitor:
- `test_start_monitoring_calls_on_unhealthy_at_threshold`
- `test_monitor_loop_survives_perform_check_exception`
- `test_stop_monitoring_cancels_task_cleanly`

Concurrency:
- `test_record_failure_thread_safe_with_perform_check`

### Unit tests — `tests/test_recover_connection.py` (new, ~200 LOC)

- `test_first_attempt_does_not_restart_gateway`
- `test_third_attempt_restarts_gateway`
- `test_backoff_schedule_15_30_60_120_300`
- `test_returns_true_on_success`
- `test_returns_false_after_exhausted_attempts`
- `test_concurrent_invocations_serialize_via_lock`
- `test_recovery_in_progress_flag_set_and_cleared`
- `test_safe_disconnect_skips_disconnect_when_not_connected` (Edge 1 regression)
- `test_recovery_kills_python_zombies_before_reconnect` (Edge 2 regression)
- `test_exhausted_recovery_fires_runner_exit_alert`

### Integration tests — `tests/test_run_continuous_persistent.py` (new, ~250 LOC)

Using a `FakeSubprocessIBKRClient` test double matching real client API surface:

- `test_single_runner_reused_across_cycles`
- `test_connection_persists_when_cycles_succeed`
- `test_teardown_full_cleanup_false_keeps_subprocess_alive`
- `test_cycle_error_does_not_trigger_recovery` (Class 1)
- `test_three_consecutive_ping_failures_trigger_recovery` (Class 2)
- `test_cycles_pause_during_recovery`
- `test_cycles_resume_after_successful_recovery`
- `test_exhausted_recovery_exits_run_continuous_cleanly` (Class 3)
- `test_multi_portfolio_each_has_own_persistent_runner`

### Regression suite (non-negotiable)

- `tests/security/test_web.py::test_ws_request_headers_shim_handles_v15_api`
- `tests/security/test_web.py::test_ws_auth_end_to_end_against_real_library`
- Full `tests/security/` suite (currently 42 tests, all passing)
- Anything in `tests/integration/` that touches runner restart logic

If any of these break, the change does not ship.

### Manual canary plan

**30-minute idle canary** on `feature/persistent-ibkr-connection`:
- Start trader; watch `event=connection_state_change` log lines
- Should see: 1x initial HEALTHY, then nothing else during normal operation
- Confirm `Trading cycle complete` keeps logging without "Disconnecting from IBKR" between cycles

**Forced-failure canary** (~10 min):
- With trader running, manually `pkill -f "IB Gateway"`
- Watch for `event=recovery_started attempt=1` -> `attempt=2` -> `recovery_succeeded`
- Verify trading resumes within ~5 min, no `runner_async` process restart

**24-hour soak before Monday open**:
- Run Sunday afternoon -> Monday 4 AM ET
- At Sunday 11:45 PM: confirm scheduled Gateway auto-restart triggers ONE recovery cycle, succeeds, resumes
- Monday 4 AM: extended hours start, system should resume trading on existing persistent connection

### Out-of-scope tests (explicit)

| | Why |
|---|---|
| Real IBKR backend rate-limit behavior | Can't reproduce in CI; this change prevents hitting it but we can't unit-test that |
| Multi-day state rot in `ib_async` | Untestable in CI; canary + production monitoring catches it |
| macOS sleep/wake | Manual test only |
| IBKR auth server unavailability | Out of our control |

### Metrics to capture post-deploy

```python
metrics.counter("connection_state_changes", labels=["from", "to"])
metrics.counter("recovery_attempts", labels=["outcome", "attempt"])
metrics.histogram("recovery_duration_seconds")
metrics.gauge("connection_uptime_seconds")
```

If `connection_uptime_seconds` doesn't reliably reach hours in production, the design is failing and we need a postmortem. Expected steady-state: hours-to-days uptime, with one recovery cycle ~24h apart for the scheduled Gateway restart.

---

## 6. Open questions

These are deferred decisions, not blockers:

1. **`recover_connection` location** — method on `AsyncRunner` or top-level helper? Currently method (state access). Revisitable.
2. **Backoff schedule `[15, 30, 60, 120, 300]`** — IBKR's actual throttle clear time is unknown. Empirically tune from production metrics.
3. **Gateway restart on attempt 1 vs 3** — currently 3 to save ~2 min on common transient blips. If recovery turns out to be flaky, move to attempt 1.
4. **`ConnectionHealth.perform_check`** — should it consume `nextValidId` callbacks from IB subprocess directly (richer heartbeat) or just `ping()`? `ping()` to start; deeper integration is a follow-up.

---

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Long-running state rot in `ib_async` | Unknown — undocumented | Daily Gateway auto-restart (11:45 PM) forces recovery cycle, bounding rot to <24h |
| `ConnectionHealth` has bugs (new code) | Medium | 25+ unit tests, 10 integration tests; canary plan catches behavioral issues |
| Recovery loop itself hangs or races | Medium | `asyncio.Lock` mutex; bounded budget (5 attempts max); watchdog as outer safety |
| Reconnection breaks something subtle (DB sync, stop-loss recreate) | Medium | Reuse existing `load_existing_positions()` path; integration test asserts stop-losses recreated post-recovery |
| Plan looks fine, fails on Monday open | Low (Sunday soak should catch) | Manual canary mandatory before live |

---

## 8. Implementation sequence

This document is the design. The implementation plan (sequenced TDD tasks) is the responsibility of the next skill in the pipeline (`superpowers:writing-plans`). High-level expected sequence:

1. `ConnectionHealth` module + its unit tests (TDD)
2. `_safe_disconnect` + `recover_connection` + their unit tests (TDD)
3. `AsyncRunner.initialize_connection()` extraction + tests
4. `run_continuous()` restructure + integration tests
5. `_monitor_subprocess_health` deletion + verify replacement coverage
6. CLAUDE.md update
7. Pre-deploy: full test suite green, manual canary 30 min + 24 hour soak
8. Deploy Monday 4 AM ET (extended hours start)

---

## 9. References

- Today's session root-cause investigation (this conversation, 2026-05-14 -> 2026-05-16)
- `handoff/2025-11-20_zombie_connection_analysis.md` (zombie + disconnect crash rules)
- `handoff/2025-11-24_1300_subprocess_connection_fix_complete.md` (subprocess isolation)
- `handoff/2025-11-24_1430_gateway_api_handshake_timeout_comprehensive.md` (API handshake timing)
- `handoff/HANDOFF_2025-12-06_ZOMBIE_FIX.md` (lsof not socket.connect_ex)
- `handoff/HANDOFF_2025-12-24_subprocess_pipe_fix_complete.md` (subprocess pipe race)
- `handoff/HANDOFF_2026-02-03_*` (referenced 2025-12-10 persistent-connection plan, never executed)
- `docs/SUBPROCESS_WORKER_CONNECTION_FIX.md`
- `docs/ZOMBIE_CONNECTION_CLEANUP.md`
- Commit `9412479` (watchdog 5-layer defense, 2026-05-11 outage)
- Commit `1a68a0a` (watchdog Layer 6 + teardown docstring — this branch's parent)
