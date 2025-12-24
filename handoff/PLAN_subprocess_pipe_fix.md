# Plan: Fix Subprocess Pipe Blocking

## The Bug
Subprocess stdin pipe blocks after first 2 commands. Data fetch commands never reach worker.

## Quick Fix Steps

### Step 1: Add Direct Write (No Executor)
**File:** `robo_trader/clients/subprocess_ibkr_client.py`
**Line:** ~356-363

```python
# BEFORE (broken):
def write_command():
    try:
        self.process.stdin.write(command_json)
        self.process.stdin.flush()
    except Exception as e:
        raise SubprocessCrashError(f"Failed to send command: {e}")

await asyncio.get_event_loop().run_in_executor(None, write_command)

# AFTER (fix):
try:
    self.process.stdin.write(command_json)
    self.process.stdin.flush()
except Exception as e:
    raise SubprocessCrashError(f"Failed to send command: {e}")

# Small yield to let reader thread process
await asyncio.sleep(0.001)
```

### Step 2: Add Write Confirmation Logging
**File:** `robo_trader/clients/subprocess_ibkr_client.py`
**After the write:**

```python
logger.debug("Command sent to subprocess",
             command=command.get("command"),
             bytes_written=len(command_json))
```

### Step 3: Add Worker Receipt Logging
**File:** `robo_trader/clients/ibkr_subprocess_worker.py`
**In main() loop, line ~525:**

```python
# After parsing command, log receipt
cmd = command.get("command", "unknown")
debug_log(f"Processing command: {cmd}")
```

### Step 4: Test
```bash
# Kill existing
pkill -9 -f "runner_async"

# Clear debug log
rm -f /tmp/worker_debug.log

# Start fresh
.venv/bin/python -m robo_trader.runner_async > /tmp/runner.log 2>&1 &

# Watch for data fetches
tail -f /tmp/runner.log | grep -E "(fetch|Signal|timeout)"
```

## Alternative: Use asyncio.subprocess

If direct write doesn't work, replace subprocess.Popen entirely:

```python
# In start() method:
self.process = await asyncio.create_subprocess_exec(
    python_exe, "-m", "robo_trader.clients.ibkr_subprocess_worker",
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)

# In _execute_command:
self.process.stdin.write(command_json.encode())
await self.process.stdin.drain()

line = await asyncio.wait_for(
    self.process.stdout.readline(),
    timeout=timeout
)
response = json.loads(line.decode())
```

## Fallback: Enable Direct Mode

Add env var support to bypass subprocess entirely:

**File:** `robo_trader/utils/robust_connection.py`
**Line ~1063:**

```python
# Check env var
use_subprocess = os.getenv("IBKR_USE_SUBPROCESS", "true").lower() == "true"

if use_subprocess:
    logger.info("Using subprocess-based IBKR client (recommended)")
    return await connect_ibkr_robust_subprocess(...)
else:
    logger.warning("Using legacy direct ib_async client")
    return await _connect_ibkr_direct(...)
```

## Verification Checklist

- [ ] Worker debug log shows `get_historical_bars` commands received
- [ ] Runner log shows data fetched with actual prices
- [ ] Signals show non-zero values (BUY=1, SELL=-1, HOLD=0)
- [ ] No timeout errors for data fetch
- [ ] System stable for 30+ minutes

## Time Estimate
- Quick fix (Step 1-4): 15 minutes
- Full asyncio.subprocess rewrite: 1-2 hours
- Testing and verification: 30 minutes
