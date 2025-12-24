# Handoff: Subprocess Pipe Blocking Bug

**Date:** 2025-12-24
**Status:** Bug Identified, Fix Required
**Priority:** Critical - Blocks all trading signals

## Executive Summary

The trading system's subprocess-based IBKR client has a critical bug where the stdin pipe becomes blocked after the first few commands (connect, get_accounts). Subsequent commands like `get_historical_bars` are sent by the parent process but never received by the worker subprocess. This prevents all data fetching and signal generation.

## Problem Description

### Symptoms
1. IBKR connection establishes successfully (serverVersion=178, accounts=['DUN264991'])
2. First commands work: `connect`, `get_accounts`
3. Data fetch commands (`get_historical_bars`) timeout after 60 seconds
4. Worker debug log shows NO receipt of data fetch commands
5. All signals show `Signal=0, Price=$0.00, No data available`

### Evidence from Logs

**Worker Debug Log (`/tmp/worker_debug.log`):**
```
2025-12-24T12:27:26: DEBUG: Received command: connect
2025-12-24T12:27:26: DEBUG: Connection fully established in 0.50s
2025-12-24T12:27:27: DEBUG: Received command: get_accounts
# NO MORE COMMANDS RECEIVED - pipe is blocked
```

**Runner Log:**
```
12:27:29 - AI OPPORTUNITY: TUBI - Confidence: 80%
12:27:29 - Processing 7 symbols with max 8 concurrent
12:28:29 - ERROR: Command timeout after 60.0s (get_historical_bars for SPY)
```

**Connection State:**
```
java  55362 - TCP *:4002 (LISTEN)
java  55362 - TCP 127.0.0.1:4002->127.0.0.1:59987 (ESTABLISHED)
Python 90825 - TCP 127.0.0.1:59987->127.0.0.1:4002 (ESTABLISHED)
```

The TCP connection is ESTABLISHED but subprocess stdin is blocked.

## Root Cause Analysis

### Suspected Causes

1. **Asyncio Event Loop Starvation**
   - The `run_in_executor` calls for stdin.write may not be executing
   - The asyncio lock (`async with self.lock`) may be causing contention
   - Heavy async operations (AI calls, ML model loading) may starve the executor

2. **Pipe Buffer Deadlock**
   - Python subprocess pipes have limited buffer sizes (typically 64KB)
   - If stdout buffer fills before parent reads, pipe can deadlock
   - The reader thread may not be draining stdout fast enough

3. **Threading/Async Interaction Issues**
   - The subprocess client uses threads for I/O with asyncio coordination
   - `run_in_executor` wraps blocking I/O but may have race conditions
   - The response queue may have synchronization issues

### Key Code Locations

**Parent Side (subprocess_ibkr_client.py):**
- Line 150-157: `subprocess.Popen` creation with `bufsize=1`
- Line 331-400: `_execute_command` - sends command, waits for response
- Line 356-361: `write_command` - writes to stdin in executor
- Line 180-220: `_stdout_read_loop` - reader thread for responses

**Worker Side (ibkr_subprocess_worker.py):**
- Line 506-540: `main()` - reads stdin, handles commands, writes stdout
- Line 512-517: `sys.stdin.readline()` with 1-second timeout

## Proposed Fix Plan

### Phase 1: Diagnostics (30 min)

1. **Add detailed logging to stdin write**
   ```python
   # In _execute_command, after write:
   logger.debug("Command written to stdin", command=command.get("command"), bytes=len(command_json))
   ```

2. **Add pipe buffer monitoring**
   ```python
   # Check if stdin buffer is full
   import select
   ready = select.select([], [self.process.stdin], [], 0)
   logger.debug("stdin writable", ready=bool(ready[1]))
   ```

3. **Add worker heartbeat logging**
   ```python
   # In worker main loop, log every iteration
   logger.debug("Worker waiting for command", iteration=i)
   ```

### Phase 2: Fix Options (Choose One)

#### Option A: Remove Executor for Writes (Recommended)
The `run_in_executor` may be causing the issue. Try direct write with non-blocking mode:

```python
# In _execute_command
async with self.lock:
    # Write directly without executor
    self.process.stdin.write(command_json)
    self.process.stdin.flush()

    # Read response from queue
    ...
```

#### Option B: Use asyncio.subprocess Instead
Replace `subprocess.Popen` with `asyncio.create_subprocess_exec`:

```python
self.process = await asyncio.create_subprocess_exec(
    python_exe, "-m", "robo_trader.clients.ibkr_subprocess_worker",
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)

# Write becomes:
self.process.stdin.write(command_json.encode())
await self.process.stdin.drain()

# Read becomes:
line = await self.process.stdout.readline()
```

#### Option C: Use Separate Process with Queue
Replace pipes with multiprocessing.Queue for more reliable IPC:

```python
from multiprocessing import Process, Queue

class SubprocessIBKRClient:
    def __init__(self):
        self.cmd_queue = Queue()
        self.response_queue = Queue()
        self.worker = Process(target=worker_main, args=(self.cmd_queue, self.response_queue))
```

### Phase 3: Testing

1. **Unit Test for Pipe Communication**
   ```python
   async def test_subprocess_multiple_commands():
       client = SubprocessIBKRClient()
       await client.start()

       # Send multiple commands rapidly
       for i in range(10):
           result = await client.execute({"command": "ping", "id": i})
           assert result["id"] == i
   ```

2. **Integration Test with IBKR**
   - Connect to Gateway
   - Fetch data for 5 symbols concurrently
   - Verify all responses received

3. **Stress Test**
   - Run for 30 minutes continuously
   - Monitor for pipe blocking or timeouts

## Temporary Workarounds

### Workaround 1: Direct ib_async Mode
Set environment variable to bypass subprocess:
```bash
export IBKR_USE_SUBPROCESS=false
```

**Note:** This was attempted but the environment variable wasn't being read correctly. Need to add proper env var handling in `robust_connection.py`.

### Workaround 2: Restart Worker Between Commands
Add worker restart logic after each data fetch cycle to prevent pipe buildup.

## Files to Modify

1. **robo_trader/clients/subprocess_ibkr_client.py**
   - Primary fix location
   - Modify `_execute_command` method
   - Improve pipe handling

2. **robo_trader/clients/ibkr_subprocess_worker.py**
   - Add heartbeat/debug logging
   - Ensure proper stdin reading

3. **robo_trader/utils/robust_connection.py**
   - Add `IBKR_USE_SUBPROCESS` env var support
   - Line 1063: Check env var before using subprocess

## Success Criteria

1. Worker receives ALL commands sent by parent
2. Data fetch for 7+ symbols completes without timeout
3. System runs stable for 1+ hour without pipe blocking
4. Signals are generated with actual prices (not $0.00)

## Additional Notes

- Today is Christmas Eve, market closed at 1:00 PM EST
- AI Analyst IS working - found TUBI (80%), LULU (70%) opportunities
- IBKR Gateway connection IS working
- Only the subprocess pipe communication is broken

## References

- Worker debug log: `/tmp/worker_debug.log`
- Runner log: `/tmp/runner.log`
- Subprocess client: `robo_trader/clients/subprocess_ibkr_client.py`
- Worker: `robo_trader/clients/ibkr_subprocess_worker.py`
