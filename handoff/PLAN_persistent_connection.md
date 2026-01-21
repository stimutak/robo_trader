# Plan: Persistent IBKR Connection Between Trading Cycles

**Date:** 2025-12-10
**Status:** PROPOSED - Ready for implementation
**Branch:** feature/persistent-connection (to be created)

## Current Behavior

The `run_continuous()` function in `runner_async.py` currently:
1. Creates a **new AsyncRunner** for each trading cycle (line 2461)
2. Calls `runner.cleanup()` after each cycle (line 2480) - this **disconnects from IBKR**
3. Reconnects on the next cycle

This means:
- Connection is torn down and re-established every cycle (default 30 min intervals)
- Dashboard shows "Waiting for cycle" most of the time
- Overhead from reconnection handshake each cycle
- Unnecessary load on Gateway

## Proposed Change

Keep the IBKR connection alive between trading cycles:
1. Create the AsyncRunner **once** before the loop
2. Use `teardown(full_cleanup=False)` between cycles - keeps connection alive
3. Only call `cleanup()` on final shutdown

## Why This Is Now Safe

The per-cycle disconnect was a workaround for issues that are now resolved:

| Original Issue | Current Solution |
|----------------|------------------|
| Async context conflicts (`patchAsyncio()`) | Subprocess isolation - worker has its own event loop |
| Connection staleness (Gateway stops responding) | Subprocess ping/health checks detect this |
| Resource leaks over long sessions | Subprocess can be restarted independently |
| Zombie connections on disconnect | `safe_disconnect()` with `IBKR_FORCE_DISCONNECT=1` |

## Implementation Steps

### Step 1: Modify `run_continuous()` in `runner_async.py`

```python
# BEFORE (lines 2458-2481):
try:
    logger.info(f"Starting trading cycle at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Create fresh runner each cycle for stability (disconnect between cycles)
    logger.info("Creating fresh runner for this cycle...")
    runner = AsyncRunner(...)

    await runner.run(symbols)

    # Clean up connection after each cycle
    logger.info("Cleaning up runner after cycle...")
    await runner.cleanup()
    runner = None

# AFTER:
# Create runner ONCE before the loop (move outside while loop)
runner = AsyncRunner(...)
await runner.setup()  # Connect once

try:
    while not shutdown_flag:
        # ... market hours check ...

        logger.info(f"Starting trading cycle at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        await runner.run(symbols)

        # Keep connection alive between cycles
        logger.info("Cycle complete - keeping connection alive")
        await runner.teardown(full_cleanup=False)

        # Wait before next iteration
        if not shutdown_flag and is_market_open():
            logger.info(f"Waiting {interval_seconds/60:.1f} minutes before next iteration...")
            await asyncio.sleep(interval_seconds)

finally:
    # Full cleanup only on shutdown
    if runner:
        logger.info("Shutting down - disconnecting from IBKR...")
        await runner.cleanup()
```

### Step 2: Add Connection Health Check Between Cycles

```python
# Before starting next cycle, verify connection is still healthy
if runner and hasattr(runner.ib, 'ping'):
    is_healthy = await runner.ib.ping()
    if not is_healthy:
        logger.warning("Connection unhealthy, reconnecting...")
        await runner.cleanup()
        runner = AsyncRunner(...)
        await runner.setup()
```

### Step 3: Handle Market Close/Open Transitions

When market closes:
- Keep connection alive (no disconnect)
- Sleep until market opens
- Verify connection health before resuming

### Step 4: Update Dashboard Status Logic

The dashboard will now show:
- "API Connected" continuously during market hours
- "API Connected (Market Closed)" when connected but market closed
- "Waiting for cycle" only during the brief sleep between cycles

## Files to Modify

1. `robo_trader/runner_async.py`
   - Refactor `run_continuous()` to create runner once
   - Add health check between cycles
   - Update logging messages

2. `app.py` (optional)
   - Update status messages for persistent connection mode

## Testing Plan

1. Start system with `./START_TRADER.sh`
2. Verify initial connection shows "API Connected"
3. Wait for one cycle to complete (~30 min or reduce for testing)
4. Verify connection stays established (check `lsof -nP -iTCP:4002 -sTCP:ESTABLISHED`)
5. Verify dashboard shows "API Connected" continuously
6. Test graceful shutdown (Ctrl+C) - verify clean disconnect
7. Test connection failure recovery (kill Gateway, verify reconnect)

## Rollback Plan

If issues arise, revert to per-cycle connection by:
1. Reverting the `run_continuous()` changes
2. Or setting an env var `IBKR_PERSISTENT_CONNECTION=false` to toggle behavior

## Expected Benefits

- Reduced reconnection overhead
- More accurate dashboard status ("API Connected" vs "Waiting for cycle")
- Lower Gateway load
- Faster cycle start times (no handshake delay)
- Better connection monitoring (continuous health checks)

## Risks

- Long-running connections may accumulate issues (mitigated by health checks)
- Gateway updates may require restart (same as before)
- Memory growth over time (monitor and add periodic restart if needed)

---

## Quick Start Implementation

```bash
# Create branch
git checkout -b feature/persistent-connection

# Make changes to runner_async.py
# Test locally
./START_TRADER.sh

# If working, commit and create PR
git add robo_trader/runner_async.py
git commit -m "feat: keep IBKR connection alive between trading cycles"
git push -u origin feature/persistent-connection
gh pr create --title "Keep IBKR connection alive between cycles" --body "..."
```
