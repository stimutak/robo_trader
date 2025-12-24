# Handoff: Subprocess Pipe Fix Complete

**Date:** 2025-12-24
**Status:** FIXED AND VERIFIED
**Priority:** Critical bug resolved

## Summary

Fixed a critical race condition in the subprocess-based IBKR client that caused data fetch commands to be lost after the first few commands. The system now correctly receives all commands including `get_historical_bars` for market data.

## Root Causes Found

### Issue 1: Parent Side - Event Loop Starvation (Minor)
**File:** `robo_trader/clients/subprocess_ibkr_client.py`

The `run_in_executor` wrapper for stdin writes could cause event loop starvation in busy async environments.

**Fix:** Direct stdin write without executor:
```python
# Before (problematic):
await asyncio.get_event_loop().run_in_executor(None, write_command)

# After (fixed):
self.process.stdin.write(command_json)
self.process.stdin.flush()
await asyncio.sleep(0.001)  # Small yield
```

### Issue 2: Worker Side - Thread Pool Race Condition (THE MAIN BUG)
**File:** `robo_trader/clients/ibkr_subprocess_worker.py`

The worker used `run_in_executor` with a 1-second timeout for `sys.stdin.readline()`. This caused a race condition:

1. First `readline()` submitted to thread pool
2. After 1s timeout, asyncio cancels the future BUT the thread continues blocking
3. Next iteration submits ANOTHER `readline()` to the pool
4. When data arrives, the orphaned first thread consumes it
5. Result never returned because its future was cancelled

**Fix:** Dedicated stdin reader thread with queue:
```python
# New dedicated reader thread
def _stdin_reader():
    while not shutdown_requested:
        line = sys.stdin.readline()
        stdin_queue.put(line)

# Main loop reads from queue (no race condition)
line = stdin_queue.get(timeout=1.0)
```

## Files Modified

1. **`robo_trader/clients/subprocess_ibkr_client.py`**
   - Removed `run_in_executor` from stdin write
   - Direct write with flush
   - Added small async yield after write

2. **`robo_trader/clients/ibkr_subprocess_worker.py`**
   - Added `queue` and `threading` imports
   - Added `stdin_queue` and `_stdin_reader()` function
   - Modified `main()` to start dedicated stdin reader thread
   - Changed main loop to read from queue instead of run_in_executor

## Verification

Tested and confirmed working:
- `connect` command: ✅
- `get_accounts` command: ✅
- `get_historical_bars` for multiple symbols: ✅
- AI signal generation: ✅
- Trading cycle completion: ✅

Sample output showing fix working:
```
Fetched 124 bars for SPY
Fetched 124 bars for QQQ
Fetched 124 bars for TSLA
Fetched 124 bars for AAPL
Fetched 124 bars for MSFT
AI BUY signal for MSFT: Positive market momentum...
Trading cycle complete. Equity: $100,000.00
```

## Additional Changes

1. **Reduced cycle interval from 5 minutes to 1 minute** in `runner_async.py`
   - Line 2534: `interval_seconds: int = 60`

2. **Handoff and plan documents created:**
   - `handoff/HANDOFF_2025-12-24_subprocess_pipe_fix.md` - Initial bug analysis
   - `handoff/PLAN_subprocess_pipe_fix.md` - Fix plan

## Next Steps

1. **Phase 1 Quick Win (planned):** Reduce cycle to 15s, use 1-min bars
2. **Phase 2 Streaming (future):** True real-time streaming with `reqMktData()`

## Technical Details

### Why the Race Condition Occurred

Python's `concurrent.futures.ThreadPoolExecutor` (used by `run_in_executor(None, ...)`) manages a pool of worker threads. When you submit a blocking operation like `readline()`:

1. A thread from the pool executes `readline()` and blocks waiting for input
2. If `asyncio.wait_for` times out, it cancels the asyncio Future
3. BUT the underlying thread is still blocked on `readline()`
4. The thread pool doesn't kill threads - it waits for them to complete
5. On next iteration, a NEW thread is spawned for the next `readline()`
6. Now you have multiple threads waiting on the same stdin
7. When data arrives, the FIRST (orphaned) thread gets it
8. That data is lost because its Future was already cancelled

The fix uses a single dedicated thread that continuously reads stdin and puts lines into a queue. The main async loop reads from the queue, which can be safely timed out without losing data.

### Connection Flow (After Fix)

```
Parent                          Worker
  |                               |
  |-- {"command": "connect"} ---> |
  |                               |-- stdin_queue.put(line)
  |                               |-- main() gets from queue
  |                               |-- handle_connect()
  |<-- {"status": "success"} ---- |
  |                               |
  |-- {"command": "get_bars"} --> |
  |                               |-- stdin_queue.put(line)  <-- NOW WORKS!
  |                               |-- main() gets from queue
  |                               |-- handle_get_historical_bars()
  |<-- {"status": "success"} ---- |
```
