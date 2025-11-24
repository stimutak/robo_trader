# IBKR Subprocess Worker Connection Failure - Handoff Document
**Date:** 2025-11-24
**Status:** BLOCKED - Subprocess worker timing issue causing connection failures
**Severity:** CRITICAL - Trading system cannot start

## Executive Summary

The IBKR subprocess worker (`ibkr_subprocess_worker.py`) successfully establishes TCP connections to Gateway but fails to retrieve account data before timing out. Additionally, every connection attempt creates zombie CLOSE_WAIT connections that block subsequent connections, requiring Gateway restart.

**Key Finding:** Worker responds with "connected": false, "accounts": [] approximately 163ms after connection initiation, but IB Gateway logs show "Connected" message arrives ~130ms AFTER the response was already sent back to the parent process. This indicates a critical timing/synchronization issue.

## Problem Statement

### Primary Issue: Account Data Never Arrives
The subprocess worker connects to IBKR Gateway using `ib_async.connectAsync()` but returns empty results before account data arrives:

```python
# Worker response (sent at T+163ms):
{"connected": false, "accounts": []}

# But ib_async logs show connection succeeded at T+293ms:
"Connecting to 127.0.0.1:4002 with clientId 1..."  # T+0ms
# Worker already responded here ↓
"Connected"  # T+293ms (too late!)
```

### Secondary Issue: Zombie Connections
Every connection attempt (successful or failed) creates a Gateway-owned zombie in CLOSE_WAIT state that blocks all future API handshakes until Gateway restart (requires 2FA).

```bash
# Zombie example:
JavaAppli (PID 34611, FD 43u) TCP 127.0.0.1:4002->127.0.0.1:52767 (CLOSE_WAIT)
```

## Architecture Context

### Subprocess Worker Design
The system uses a subprocess-based architecture to isolate `ib_async` from the main trading system's complex async environment:

1. **Parent Process** (`robo_trader.runner_async.py`)
   - Main trading system with complex async event loops
   - Cannot use `ib_async` directly (library conflicts with async patches)

2. **Subprocess Worker** (`robo_trader/clients/ibkr_subprocess_worker.py`)
   - Runs in isolated process
   - Communicates via JSON over stdin/stdout
   - Uses `ib_async` library to connect to Gateway

3. **Subprocess Client** (`robo_trader/clients/subprocess_ibkr_client.py`)
   - Manages worker subprocess
   - Sends commands, receives responses
   - Timeout: 30 seconds for connect command

### Working Test vs Failing Production

**✅ Direct Worker Test (WORKS):**
```bash
echo '{"command": "connect", "params": {...}}' | python3 -m robo_trader.clients.ibkr_subprocess_worker

# DEBUG output shows:
DEBUG: Connecting to 127.0.0.1:4002 client_id=999 timeout=30.0
DEBUG: Connection initiated, waiting for handshake...
DEBUG: Connected successfully!
DEBUG: Waiting for account data to arrive...
DEBUG: Received accounts after 0.5s: ['DUN264991']

# Response:
{"status": "success", "data": {"connected": true, "accounts": ["DUN264991"], "client_id": 999}}
```

**❌ Production System (FAILS):**
```bash
./START_TRADER.sh

# No DEBUG output visible
# Response comes back in ~163ms:
{"connected": false, "accounts": []}

# But ib_async logs show connection succeeded AFTER response sent:
11:57:47.778 - "Connecting to 127.0.0.1:4002 with clientId 1..."
11:57:47.942 - {"connected": false, "accounts": []}  ← Response sent
11:57:47.942 - "Connected"  ← Connection succeeds (too late!)
```

## Root Cause Analysis

### Hypothesis 1: Subprocess Communication Timing Issue (MOST LIKELY)
The subprocess client (`subprocess_ibkr_client.py`) appears to be receiving/reading the response before the worker actually completes its work:

**Evidence:**
- Worker's DEBUG prints (to stderr) are NOT captured in logs, suggesting stderr is not properly redirected
- Response arrives in ~163ms, which is too fast for a full connection handshake + account data retrieval
- `ib_async` logs "Connected" AFTER the parent already received empty response
- Direct test shows account data takes ~500ms to arrive after connection initiated

**Possible Causes:**
1. **Premature stdout read**: Subprocess client reads from worker's stdout before worker writes complete response
2. **Race condition**: Worker writes partial response, subprocess client reads it immediately, connection completes later
3. **Buffering issue**: stdout not properly flushed, or parent reads before flush completes
4. **Event loop desynchronization**: Worker's async event loop not properly synchronized with subprocess communication

### Hypothesis 2: Zombie Connection Blocks Handshake
The pre-existing zombie connection might be interfering with the API handshake, causing:
- TCP connection to succeed (SYN/ACK completes)
- API protocol handshake to stall (Gateway not responding to API messages)
- Worker times out waiting for account data
- Worker responds with empty results

**Evidence:**
- Fresh Gateway (no zombies) required for any connection attempts
- Logs show "Found 1 Gateway-owned zombie(s) - these block API handshakes"
- Gateway PID 34611 has zombie on FD 43u even after user claims restart

**Counter-evidence:**
- Direct worker test succeeded and retrieved account data (proving worker code is correct)
- TCP connection logs show "Connected" message, indicating handshake progressed

### Hypothesis 3: Missing ibkr_safe Patch Causing Issues
Removed `from robo_trader.utils import ibkr_safe` from worker to allow proper disconnect. This might have unintended side effects on connection establishment (not just disconnection).

**Evidence:**
- Issue persists after removing ibkr_safe import
- No other patches appear to interfere with connection logic

**Counter-evidence:**
- Direct worker test succeeded WITHOUT ibkr_safe import
- Patch only affects `disconnect()`, not `connectAsync()`

## Fixes Attempted

### Fix #1: Use connectAsync Instead of Executor-Wrapped connect() ✅ IMPLEMENTED
**Rationale:** Original handoff document identified that executor-wrapped sync `connect()` blocked ib_async's message processing loop

**Changes:**
```python
# BEFORE (broken):
await loop.run_in_executor(
    None,
    lambda: ib.connect(host, port, clientId, readonly, timeout)
)

# AFTER (fixed):
await ib.connectAsync(host, port, clientId, readonly, timeout)
await asyncio.sleep(0.5)  # Wait for API handshake to complete
```

**Result:** Direct test proved this works correctly. Production system still fails due to different issue (timing/communication problem).

### Fix #2: Remove Executor Wrapping from waitOnUpdate() ✅ IMPLEMENTED
**Changes:**
```python
# BEFORE (broken):
await loop.run_in_executor(None, lambda: ib.waitOnUpdate(timeout=0.5))

# AFTER (fixed):
ib.waitOnUpdate(timeout=poll_interval)
```

**Result:** Same as Fix #1 - works in direct test, fails in production.

### Fix #3: Remove ibkr_safe Import from Worker ✅ IMPLEMENTED
**Rationale:** Worker runs in isolation, doesn't need the disconnect patch that might interfere

**Changes:**
```python
# REMOVED:
from robo_trader.utils import ibkr_safe as _ibkr_safe
```

**Result:** No improvement. Issue persists.

### Fix #4: Clear Python Bytecode Cache ✅ IMPLEMENTED
**Rationale:** Ensure updated code is being used, not cached .pyc files

**Changes:**
```bash
rm -rf robo_trader/clients/__pycache__
rm -rf robo_trader/utils/__pycache__
```

**Result:** No improvement. Fresh code is being used but issue persists.

## Current State

### Files Modified
1. **robo_trader/clients/ibkr_subprocess_worker.py**
   - Line 82-95: Changed to `await ib.connectAsync()` with 0.5s handshake wait
   - Line 121-127: Removed executor wrapping from `waitOnUpdate()`
   - Line 21-22: Removed `ibkr_safe` import

2. **START_TRADER.sh**
   - Line 71-75: Commented out connectivity test that creates zombies

3. **robo_trader/clients/ibkr_subprocess_worker.py.backup**
   - Created backup of original code

### Zombie Status
- Gateway PID: 34611 (claimed "restarted" but has zombie on FD 43u)
- Zombie: 127.0.0.1:4002->127.0.0.1:52767 (CLOSE_WAIT)
- Impact: Blocks all new API handshake attempts

### System Status
- Trading system: CANNOT START
- Dashboard: Running (PID 34686)
- WebSocket server: Running (PID 34680)
- Runner: Exits after connection failure

## Recommended Next Steps

### Phase 1: Fix Subprocess Communication Timing (HIGHEST PRIORITY)

**Option A: Add Explicit Wait After Connection**
```python
# In ibkr_subprocess_worker.py handle_connect()
await ib.connectAsync(...)

# Current 0.5s wait is too short - increase it
await asyncio.sleep(2.0)  # Wait longer for API handshake + account data

# Then wait for accounts as before
for attempt in range(20):
    accounts = ib.managedAccounts()
    if accounts:
        break
    ib.waitOnUpdate(timeout=0.5)
```

**Option B: Investigate Subprocess Client Timeout**
```python
# In subprocess_ibkr_client.py - check how response is read from worker
# Possible issue: Reading stdout before worker finishes writing
# Need to examine subprocess communication code
```

**Option C: Capture and Analyze Worker DEBUG Output**
```python
# In subprocess_ibkr_client.py start()
# Redirect worker stderr to a file for analysis
stderr_file = open('/tmp/worker_debug.log', 'w')
self.process = subprocess.Popen(
    [...],
    stderr=stderr_file  # Instead of stderr=subprocess.PIPE
)

# Check /tmp/worker_debug.log to see what worker is actually doing
```

**Option D: Synchronization Fix - Wait for "Connected" Before Proceeding**
```python
# In ibkr_subprocess_worker.py handle_connect()
# Add explicit check for connection status before proceeding

await ib.connectAsync(...)

# Wait for connection to be fully established
max_wait = 10  # seconds
start = time.time()
while not ib.isConnected():
    if time.time() - start > max_wait:
        raise TimeoutError("Connection handshake timeout")
    await asyncio.sleep(0.1)

# Now connection is fully established, proceed with account retrieval
```

### Phase 2: Address Zombie Connection Problem

**Option A: Accept Zombies, Detect and Abort Early**
Current approach - system detects zombies and warns, but proceeds anyway. Could abort immediately instead:
```python
# In robust_connection.py connect()
if gateway_zombies > 0:
    raise ConnectionError(
        "Cannot connect: Gateway has zombie connections. "
        "Restart Gateway (File→Exit, relaunch with 2FA) before retrying."
    )
```

**Option B: Implement Proper Disconnect Sequence**
Research IB API documentation for correct disconnect procedure that doesn't create zombies:
```python
# Hypothesis: Need specific sequence
ib.reqIds(-1)  # Cancel all requests?
await asyncio.sleep(0.5)
ib.disconnect()
```

**Option C: Keep Connections Alive Permanently**
Never disconnect - maintain persistent connection for runner lifetime:
```python
# Modify architecture:
# - Worker subprocess stays alive entire session
# - Handles multiple commands over single connection
# - Only disconnects on final shutdown
# - Accept that shutdown will create zombie (unavoidable)
```

### Phase 3: Verify Gateway Actually Restarted

**Diagnostic Commands:**
```bash
# Check Gateway PID and start time
ps aux | grep -i "gateway\|tws" | grep -i java | grep -v grep

# Expected: New PID, recent start time
# If PID 34611 persists, Gateway was NOT restarted

# Check ALL connections (not just zombies)
lsof -nP -iTCP:4002

# Force kill Gateway if needed (USER MUST DO THIS, NOT AUTOMATED):
# 1. Cmd+Q or File→Exit in Gateway UI
# 2. If unresponsive: kill -9 <GATEWAY_PID>
# 3. Relaunch Gateway, login with 2FA
# 4. Verify port 4002 has NO existing connections before starting trader
```

### Phase 4: Debug Subprocess Communication (IF TIMING ISSUE PERSISTS)

**Investigate subprocess_ibkr_client.py:**
1. How is stdout being read from worker?
2. Is there a timeout on the read?
3. Is read() blocking or non-blocking?
4. Is response being parsed before it's completely written?

**Add extensive logging:**
```python
# In subprocess_ibkr_client.py connect()
logger.info(f"Sending connect command to worker...")
self.process.stdin.write(json.dumps(cmd))
self.process.stdin.flush()
logger.info(f"Command sent, waiting for response...")

response_line = self.process.stdout.readline()
logger.info(f"Received response line: {response_line!r}")

response = json.loads(response_line)
logger.info(f"Parsed response: {response}")
```

## Testing Protocol

### Prerequisites
1. Gateway completely exited (Cmd+Q or File→Exit)
2. Gateway relaunched with 2FA login
3. Verify no zombies: `lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT` returns nothing
4. Verify no established connections: `lsof -nP -iTCP:4002` shows only Gateway listening

### Test Procedure
1. Clear Python cache: `rm -rf robo_trader/**/__pycache__`
2. Kill all trader processes: `pkill -9 -f "runner_async"`
3. Run direct worker test to confirm basic functionality:
   ```bash
   echo '{"command": "connect", "params": {"host": "127.0.0.1", "port": 4002, "client_id": 999, "readonly": true, "timeout": 30.0}}' | python3 -m robo_trader.clients.ibkr_subprocess_worker
   ```
4. Check for new zombies immediately after test
5. If zombie created, restart Gateway before proceeding
6. Run full system: `./START_TRADER.sh AAPL`
7. Monitor logs in real-time: `tail -f robo_trader.log`
8. Check for zombies after startup attempt
9. Document timing between "Connecting...", response sent, and "Connected" logs

### Success Criteria
1. Worker returns `{"status": "success", "connected": true, "accounts": ["DUN..."]}`
2. No zombie created after connection
3. Runner proceeds past setup phase
4. System runs normally

## Technical Details

### Relevant Code Locations

**Subprocess Worker:**
- File: `robo_trader/clients/ibkr_subprocess_worker.py`
- Function: `handle_connect()` (line 51-172)
- Connection: Line 82-95 (connectAsync)
- Account wait: Line 106-139 (waitOnUpdate loop)

**Subprocess Client:**
- File: `robo_trader/clients/subprocess_ibkr_client.py`
- Function: `start()` (line 86-113) - Starts worker subprocess
- Function: `connect()` (line 345-362) - Sends connect command, receives response
- Communication: stdin/stdout pipes

**Runner Setup:**
- File: `robo_trader/runner_async.py`
- Function: `setup()` (line 457-560) - Initializes IBKR connection
- Connection call: Line 520 (calls robust_connection)

**Robust Connection:**
- File: `robo_trader/utils/robust_connection.py`
- Function: `connect_ibkr_robust()` (line 912) - Entry point
- Function: `connect_ibkr_robust_subprocess()` (line 771) - Subprocess approach
- Function: `SubprocessConnection.connect()` (line 548-640) - Actual connection logic

### Environment
- OS: macOS (Darwin 25.1.0)
- Python: 3.11.14
- ib_async: 2.0.1 (community fork of archived ib_insync)
- Gateway: IB Gateway 10.41
- Port: 4002 (paper trading)

### Logs Location
- Main log: `/Users/oliver/robo_trader/robo_trader.log`
- Worker stderr: Not currently captured (THIS IS A PROBLEM)
- Startup output: Can be captured with `./START_TRADER.sh 2>&1 | tee /tmp/startup.log`

## Known Issues and Constraints

1. **Gateway Restart Requires 2FA**: Cannot automate full Gateway restart
2. **Zombies Persist**: Even "successful" disconnects create zombies
3. **No Worker DEBUG in Logs**: subprocess stderr not captured, making debugging difficult
4. **Timing Inconsistency**: Direct worker test works, production system fails with identical code
5. **Subprocess Communication Black Box**: Unclear how subprocess_ibkr_client.py reads worker responses

## Questions for Next Developer

1. **Where is subprocess worker stderr going?** Why are DEBUG prints not in logs?
2. **How does subprocess_ibkr_client.py read stdout?** Is there a timeout or buffer issue?
3. **Why does direct test succeed but production fail?** What's different about how they invoke the worker?
4. **Is there a better IPC mechanism?** Should we use sockets/pipes instead of stdin/stdout?
5. **Can we keep worker alive between commands?** Avoid repeated connect/disconnect cycles?

## References

- **Original Handoff:** `handoff/2025-11-20_zombie_connection_analysis.md`
- **Previous Session:** Fixed executor-wrapped connect, but timing issue revealed
- **Subprocess Architecture:** Introduced in PR #48 to solve ib_async async conflicts
- **Safe Disconnect Patch:** `robo_trader/utils/ibkr_safe.py` (removed from worker)

---

**Status:** BLOCKED - Need to fix subprocess communication timing or investigate zombie prevention
**Blocker:** Worker responds before connection completes; zombies block retries
**Risk Level:** CRITICAL - Trading system cannot start
**Estimated Fix Time:** 4-8 hours (subprocess communication debugging) OR 1-2 days (architectural refactor)

## Next Session Priorities

1. **IMMEDIATE:** Investigate subprocess_ibkr_client.py response reading logic
2. **HIGH:** Capture worker stderr to see DEBUG output in production
3. **HIGH:** Increase wait time after connectAsync to allow handshake completion
4. **MEDIUM:** Verify Gateway actually restarted (check PID, no zombies)
5. **LOW:** Consider architectural alternatives (persistent worker, different IPC)
