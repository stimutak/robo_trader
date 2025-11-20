# IBKR Gateway Zombie Connection Analysis & Remediation Plan
**Date:** 2025-11-20
**Status:** BLOCKED - Need to fix subprocess worker account data retrieval

## Executive Summary

The trading system cannot connect to IBKR Gateway due to a zombie connection bug. Every connection attempt creates a CLOSE_WAIT zombie that blocks subsequent connections, forcing Gateway restart. Root cause: subprocess worker fails to receive account data after successful API handshake, then disconnects improperly creating zombie.

## Problem Statement

### The Catch-22
We're stuck between two bad options:
1. **Call `ib.disconnect()` on failed connection** → Gateway API layer crashes (goes RED), requires restart
2. **DON'T call `ib.disconnect()`** → Python closes socket improperly, creates zombie CLOSE_WAIT connection that blocks all future API handshakes

Both paths lead to failure. The only working path is: **successful connection that gets account data and stays connected**.

### Symptoms
- System cannot start - runner exits during setup
- Gateway shows one active connection in CLOSE_WAIT state
- Connection attempts log: "Subprocess connection failed" / "No managed accounts found"
- Zombie persists until Gateway restart (with 2FA)

### Zombie Details
```bash
# Zombie detection
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT

# Example output
COMMAND     PID   USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
JavaAppli 13551 oliver   43u  IPv6 0x9e24b80a29537f7a      0t0  TCP 127.0.0.1:4002->127.0.0.1:62636 (CLOSE_WAIT)
```

## What Works vs What Fails

### ✅ Working: START_TRADER.sh Test Script (Step 4)
**File:** `START_TRADER.sh` lines 74-148
**Test Script:** `/tmp/test_gateway.py` (generated at runtime)

```python
# This successfully gets account data
import asyncio
from ib_async import IB

ib = IB()
await asyncio.wait_for(
    ib.connectAsync("127.0.0.1", 4002, clientId=999, readonly=True, timeout=15.0),
    timeout=20.0
)

# Wait for accounts to arrive (asynchronous)
accounts = []
for attempt in range(20):
    accounts = ib.managedAccounts()
    if accounts:
        break
    try:
        ib.waitOnUpdate(timeout=0.5)  # Process incoming IB messages
    except:
        await asyncio.sleep(0.5)

# Result: Gets account data successfully!
print(f"✅ Gateway connection successful! Account: {accounts[0]}")
ib.disconnect()  # This creates a zombie (even with IBKR_FORCE_DISCONNECT=1)
```

**But:** When this test disconnects, it creates a zombie that blocks the subsequent runner connection.

### ❌ Failing: Subprocess Worker
**File:** `robo_trader/clients/ibkr_subprocess_worker.py` lines 51-172

```python
# Subprocess worker approach - NO account data received
ib = IB()

# Connect using sync method wrapped in executor
loop = asyncio.get_event_loop()
await loop.run_in_executor(
    None,
    lambda: ib.connect(host=host, port=port, clientId=client_id, readonly=readonly, timeout=timeout),
)

# Wait for accounts - BUT THEY NEVER ARRIVE
accounts = []
for attempt in range(20):  # 10 seconds total
    accounts = ib.managedAccounts()
    if accounts:
        break
    try:
        await loop.run_in_executor(
            None,
            lambda: ib.waitOnUpdate(timeout=0.5)
        )
    except:
        pass

# After 10 seconds: No accounts!
if not accounts:
    raise ConnectionError("No managed accounts found after 10s wait")
    # Exception handler sets ib = None (NO disconnect call)
    # Python closes socket when subprocess exits
    # Gateway left with zombie CLOSE_WAIT connection
```

## Root Cause Analysis

### Why Subprocess Worker Fails to Get Account Data

**Hypothesis 1: Async/Sync Mismatch (MOST LIKELY)**
- Test script uses native async: `ib.connectAsync()` - messages process naturally
- Subprocess worker uses sync wrapped in executor: `ib.connect()` in `run_in_executor()`
- `ib_async` library expects its event loop to run in main thread to process incoming messages
- When wrapped in executor, incoming account data messages cannot be processed
- Result: Connection succeeds but account data never arrives

**Hypothesis 2: ibkr_safe Import Interference**
- Subprocess worker imports `robo_trader.utils.ibkr_safe` (line 21)
- This patches `ib.disconnect()` to be a no-op by default
- Patch might interfere with internal ib_async message handling
- Test script doesn't import this module

**Hypothesis 3: Message Loop Not Running**
- After `ib.connect()` returns in executor, ib_async expects its internal event loop to process messages
- But the worker is in an async context with its own event loop
- Messages from Gateway arrive but there's no active processor to handle them

### Why Zombies Get Created

When subprocess worker fails (no accounts), the exception handler does this:
```python
except Exception as e:
    # NOTE: Do NOT call ib.disconnect() here! It crashes IBKR Gateway's API layer.
    ib = None  # Just abandon the connection
```

Then when the subprocess exits:
1. Python's garbage collector cleans up the `ib` object
2. This closes the TCP socket (FIN packet sent)
3. But no proper API disconnect message was sent to Gateway
4. Gateway sees socket close without API disconnect
5. Gateway leaves connection in CLOSE_WAIT state (zombie)
6. Zombie blocks all future API handshakes until Gateway restart

## Technical Details

### File Locations
- **Subprocess worker:** `robo_trader/clients/ibkr_subprocess_worker.py`
- **Subprocess client:** `robo_trader/clients/subprocess_ibkr_client.py`
- **Robust connection:** `robo_trader/utils/robust_connection.py`
- **Safe disconnect:** `robo_trader/utils/ibkr_safe.py`
- **Runner:** `robo_trader/runner_async.py` (setup at line 457)
- **Startup script:** `START_TRADER.sh`

### Connection Flow
1. Runner calls `connect_ibkr_robust()` in `robust_connection.py:912`
2. This calls `connect_ibkr_robust_subprocess()` at line 771
3. Subprocess worker started: `SubprocessIBKRClient.start()` in `subprocess_ibkr_client.py:86`
4. Worker script: `ibkr_subprocess_worker.py` runs as separate process
5. Worker receives "connect" command via stdin
6. Worker calls `handle_connect()` at line 51
7. Worker attempts connection but fails to get accounts
8. Worker exits, creates zombie

### Zombie Detection Commands
```bash
# Check for zombies on Gateway port
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT

# Check Gateway process
ps aux | grep -i "gateway\|tws" | grep -i java | grep -v grep

# Monitor connections
netstat -an | grep 4002
```

## Recommended Solution Path

### Phase 1: Fix Subprocess Worker Account Data Retrieval (PRIORITY)

**Option A: Use connectAsync (Recommended)**
Modify subprocess worker to use native async connection like the test script:

```python
# In ibkr_subprocess_worker.py handle_connect()
# REMOVE executor wrapping, use native async

# Current (broken):
await loop.run_in_executor(None, lambda: ib.connect(...))

# Change to (like test script):
await ib.connectAsync(host=host, port=port, clientId=client_id, readonly=readonly, timeout=timeout)

# For account waiting loop:
for attempt in range(20):
    accounts = ib.managedAccounts()
    if accounts:
        break
    # REMOVE executor wrapping here too:
    try:
        ib.waitOnUpdate(timeout=0.5)  # Direct call, not in executor
    except:
        await asyncio.sleep(0.5)
```

**Rationale:** This matches the working test script approach exactly. The test script proves that `connectAsync` + direct `waitOnUpdate` successfully receives account data.

**Option B: Remove ibkr_safe Import**
The subprocess worker doesn't need the disconnect patch since it runs in isolation:

```python
# In ibkr_subprocess_worker.py
# REMOVE: from robo_trader.utils import ibkr_safe as _ibkr_safe

# This removes the disconnect() monkey patch
# Let the worker use original ib.disconnect()
# If it crashes Gateway, it only affects this subprocess
```

**Option C: Increase Wait Time / Better Logging**
Maybe account data just takes longer to arrive:
- Increase wait from 10s to 30s
- Add extensive DEBUG logging to see exactly when messages arrive
- Capture subprocess stderr to file for analysis

### Phase 2: Handle Disconnect Properly

**If Connection Succeeds:**
- Keep connection alive permanently (never disconnect)
- This avoids zombie creation entirely
- Connection maintained for lifetime of runner

**If Connection Fails:**
- Let subprocess exit without calling disconnect
- Accept that this creates a zombie
- Detect zombie immediately after failed attempt
- Show clear error: "Gateway zombie detected - restart Gateway required"
- Provide restart instructions with 2FA note

### Phase 3: Eliminate Test Script Zombie

The START_TRADER.sh connection test creates zombies even when successful. Options:

**Option A: Skip Test Entirely**
- Remove step 4 (connectivity test) from START_TRADER.sh
- Let runner make first connection
- Faster startup, no test-created zombies

**Option B: Keep Test Alive**
- Don't disconnect test connection
- Keep it alive as "keepalive"
- Runner uses different client ID for actual work

**Option C: Better Disconnect**
- Research proper way to disconnect without creating zombie
- May require specific IB API sequence we're missing

## Fallback: No-Subprocess Approach

If subprocess continues to fail, revert to simpler architecture:

**Direct Connection in Runner:**
```python
# In runner_async.py setup()
# DON'T use subprocess worker
# Connect directly like test script does

self.ib = IB()
await self.ib.connectAsync(
    host=self.cfg.ibkr.host,
    port=self.cfg.ibkr.port,
    clientId=self.cfg.ibkr.client_id,
    readonly=True,
    timeout=30.0
)

# Wait for accounts
accounts = []
for i in range(20):
    accounts = self.ib.managedAccounts()
    if accounts:
        break
    try:
        self.ib.waitOnUpdate(timeout=0.5)
    except:
        await asyncio.sleep(0.5)
```

**Pros:**
- Uses proven working approach from test script
- No subprocess complexity
- Direct access to ib_async in runner

**Cons:**
- Back in main process async environment (original reason for subprocess)
- Risk of async conflicts with trading system's event loop
- But test script proves it CAN work

## Implementation Steps

### Step 1: Try connectAsync in Subprocess Worker (1-2 hours)
1. Backup current `ibkr_subprocess_worker.py`
2. Modify `handle_connect()` to use `connectAsync` instead of executor-wrapped `connect()`
3. Remove executor wrapping from `waitOnUpdate()` loop
4. Test with fresh Gateway (no zombies)
5. Check if account data arrives

### Step 2: Enhanced Logging (30 mins)
1. Add timestamps to all DEBUG prints in worker
2. Log every step of connection and account retrieval
3. Capture subprocess stderr to `/tmp/worker_debug.log`
4. Run test and analyze exact timing

### Step 3: Test & Document (1 hour)
1. Kill all existing processes
2. Restart Gateway (get new PID)
3. Verify no zombies: `lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT`
4. Start runner directly (not via START_TRADER.sh)
5. Monitor for zombies after connection attempt
6. Document results

### Step 4: If Still Failing - Remove ibkr_safe Import (30 mins)
1. Comment out `from robo_trader.utils import ibkr_safe` in worker
2. Let worker use original disconnect
3. Test if account data arrives
4. Accept that failed disconnect might crash Gateway (isolated to subprocess)

### Step 5: Last Resort - Direct Connection (2-3 hours)
1. Create new connection method in `runner_async.py`
2. Use `connectAsync` directly in runner (no subprocess)
3. Copy working code from test script
4. Test thoroughly
5. Update documentation

## Success Criteria

**Phase 1 Success:**
- Subprocess worker successfully connects to Gateway
- Worker receives account data (not empty)
- No zombie created after connection
- Runner proceeds past setup phase

**Phase 2 Success:**
- Runner maintains persistent connection
- All trading operations work normally
- System runs for hours without issues
- Gateway restart not needed

**Complete Success:**
- `./START_TRADER.sh` starts all components
- No manual Gateway intervention required
- System resilient to connection issues
- Clear error messages guide recovery

## Next Session TODO

1. **Immediate:** Try connectAsync modification in subprocess worker
2. **Backup:** If fails, try removing ibkr_safe import
3. **Fallback:** If still fails, implement direct connection in runner
4. **Documentation:** Update START_TRADER.sh with findings
5. **Testing:** Create test script that validates zombie-free connection

## References

- **Previous session:** Handoff document trail
- **Original bug:** Gateway API timeout remediation plan
- **Subprocess approach:** PR #48 - Fix IBKR Gateway API timeout blocker
- **Safe disconnect:** `robo_trader/utils/ibkr_safe.py` implementation
- **Test script:** START_TRADER.sh lines 74-148 (working connection test)

---

**Status:** Awaiting implementation of connectAsync approach
**Blocker:** Subprocess worker cannot receive account data
**Risk Level:** HIGH - System cannot start without fix
**Estimated Fix Time:** 2-4 hours (best case) to 1 day (if need fallback)
