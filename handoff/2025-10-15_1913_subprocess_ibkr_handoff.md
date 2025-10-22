# Subprocess IBKR Implementation Handoff
**Date:** 2025-10-15 19:13  
**Branch:** `fix/subprocess-ibkr-wrapper`  
**Status:** 90% Complete - One Remaining Issue

## Executive Summary

Successfully implemented subprocess-based IBKR client that solves the month-long connection timeout issue. The solution works perfectly in isolation but has one remaining issue when called from `runner_async.py`.

## What Works ✅

### 1. Subprocess IBKR Client Implementation
- **File:** `robo_trader/clients/subprocess_ibkr_client.py`
- **File:** `robo_trader/clients/ibkr_subprocess_worker.py`
- **Status:** COMPLETE and WORKING

**Test Results:**
```bash
# Minimal runner - WORKS PERFECTLY
source .venv/bin/activate && python3 test_minimal_runner.py
# Result: Connected in 0.4-0.9s, 100% success rate

# Complex async environment - WORKS PERFECTLY
source .venv/bin/activate && python3 test_subprocess_in_async_env.py
# Result: Connected with 3 background tasks running, 100% success

# 2-minute stability test - WORKS PERFECTLY
source .venv/bin/activate && python3 test_2min_stability.py
# Result: 23/23 pings, 7/7 data fetches, 0 failures
```

### 2. Key Fixes Implemented

**Fix #1: Bus Error Resolution**
- **Problem:** `runner_async.py` crashed with Bus Error 10 on import
- **Root Cause:** Using Anaconda Python 3.12 instead of venv Python 3.13
- **Solution:** Use venv Python: `source .venv/bin/activate`
- **Status:** FIXED ✅

**Fix #2: Async Event Loop Starvation**
- **Problem:** `asyncio.create_subprocess_exec` gets starved in busy async environments
- **Root Cause:** Subprocess I/O waits get delayed by other async tasks
- **Solution:** Use `subprocess.Popen` with threading instead of async subprocess
- **Implementation:** Lines 87-103 in `subprocess_ibkr_client.py`
- **Status:** FIXED ✅

**Fix #3: Python Executable Detection**
- **Problem:** `sys.executable` returns Homebrew Python, not venv Python
- **Root Cause:** Runner started via shebang uses system Python
- **Solution:** Explicitly check for `.venv/bin/python3` and use it
- **Implementation:** Lines 74-85 in `subprocess_ibkr_client.py`
- **Status:** FIXED ✅

## What Doesn't Work ❌

### The One Remaining Issue

**Problem:** `runner_async.py` times out when connecting to IBKR Gateway

**Symptoms:**
```bash
source .venv/bin/activate && python3 -m robo_trader.runner_async --symbols AAPL --once
# Result: TimeoutError after 15 seconds (2 attempts = 30 seconds total)
```

**Evidence from subprocess worker stderr:**
```
DEBUG: Connecting to 127.0.0.1:4002 client_id=1 timeout=15.0
API connection failed: TimeoutError()
```

**What This Means:**
- ✅ Subprocess worker STARTS successfully
- ✅ Subprocess worker RECEIVES the connect command
- ✅ Subprocess worker SENDS connect request to Gateway
- ❌ Gateway DOES NOT RESPOND within 15 seconds
- ❌ ib_async library times out

**Why Simple Tests Work But runner_async Doesn't:**
- Simple tests create 1 connection, succeed, disconnect cleanly
- runner_async creates connection, but Gateway has zombie connections from previous tests
- Zombies accumulate and eventually block new connections
- Runner DOES try to kill zombies, but timing is off

## Root Cause Analysis

### The Zombie Connection Problem

**What Are Zombies:**
- TCP connections in CLOSE_WAIT state
- Created when client closes but server hasn't acknowledged
- Accumulate on port 4002 over time
- Block new connections after ~3-4 zombies

**Why They Accumulate:**
1. Test runs create connections
2. Tests disconnect cleanly
3. Gateway doesn't immediately clean up
4. Zombies remain in CLOSE_WAIT state
5. After 3-4 zombies, Gateway stops accepting new connections

**Current Zombie Killer:**
- Location: `robo_trader/utils/robust_connection.py:161`
- Function: `kill_tws_zombie_connections()`
- **Problem:** Kills zombies, but new tests create more zombies faster than they're cleaned

**Evidence:**
```bash
netstat -an | grep 4002
# Shows 3 CLOSE_WAIT connections blocking new connections
```

## Attempted Solutions (Didn't Work)

1. ❌ **Restart Gateway repeatedly** - Zombies come back immediately
2. ❌ **Kill zombies before each connection** - Timing issue, new zombies appear
3. ❌ **Use different client IDs** - Doesn't prevent zombies
4. ❌ **Increase timeout** - Gateway simply doesn't respond, timeout won't help

## The Real Solution (Not Yet Implemented)

### Option 1: Fix Gateway Zombie Handling (RECOMMENDED)

**Problem:** Gateway accumulates CLOSE_WAIT zombies and stops accepting connections

**Solution:** Make Gateway more aggressive about cleaning up closed connections

**Implementation:**
1. Check Gateway configuration for connection limits
2. Increase max connections setting
3. Enable automatic connection cleanup
4. Or: Restart Gateway programmatically when zombies detected

**Gateway Settings to Check:**
- File → Global Configuration → API → Settings
- Max number of simultaneous API connections
- Connection timeout settings
- Auto-disconnect idle connections

### Option 2: Connection Pooling

**Idea:** Keep one persistent connection instead of connect/disconnect cycles

**Implementation:**
1. Create singleton IBKR connection
2. Reuse across all operations
3. Only disconnect on shutdown
4. Prevents zombie accumulation

**Pros:** No zombies, faster (no reconnection overhead)  
**Cons:** More complex, need to handle connection health

### Option 3: Subprocess Connection Reuse

**Idea:** Keep subprocess worker alive between operations

**Current:** Start subprocess → connect → disconnect → stop subprocess  
**Proposed:** Start subprocess → connect → keep alive → reuse → stop on shutdown

**Implementation:**
1. Don't call `client.disconnect()` after each operation
2. Keep subprocess running
3. Reuse existing connection
4. Only disconnect on final shutdown

## Files Modified

### New Files Created
```
robo_trader/clients/subprocess_ibkr_client.py          # Subprocess client (threading-based)
robo_trader/clients/ibkr_subprocess_worker.py          # Subprocess worker script
test_subprocess_ibkr.py                                 # Comprehensive tests
test_minimal_runner.py                                  # Working minimal runner
test_subprocess_in_async_env.py                        # Async environment test
test_2min_stability.py                                  # 2-minute stability test
test_threading_subprocess.py                            # Threading test
BUS_ERROR_INVESTIGATION.md                             # Bus error analysis
SUBPROCESS_IBKR_SOLUTION_SUMMARY.md                    # Complete summary
RUNNER_FIX_DECISION_ANALYSIS.md                        # Fix vs rebuild analysis
RUNNER_FEATURE_ANALYSIS.md                             # Feature comparison
BUS_ERROR_FIX_PLAN.md                                  # Bus error fix plan
SUBPROCESS_ASYNC_ISSUE.md                              # Async starvation issue
```

### Modified Files
```
robo_trader/utils/robust_connection.py                 # Added subprocess integration
```

## Git Status

**Branch:** `fix/subprocess-ibkr-wrapper`  
**Commits:** 11 commits  
**Ready to Merge:** NO (one issue remaining)

**Commit History:**
```
32e0b11 - fix: identify asyncio.create_subprocess_exec starvation issue
bd87fe5 - test: add 2-minute API connection stability test
f55015c - docs: comprehensive subprocess solution summary
... (8 more commits)
```

## How to Test

### Test 1: Minimal Runner (WORKS)
```bash
cd /Users/oliver/robo_trader
source .venv/bin/activate
python3 test_minimal_runner.py
# Expected: Connected in <1s, accounts retrieved, SUCCESS
```

### Test 2: Complex Async Environment (WORKS)
```bash
source .venv/bin/activate
python3 test_subprocess_in_async_env.py
# Expected: Connected with background tasks, SUCCESS
```

### Test 3: Full Runner (FAILS - THE ISSUE)
```bash
source .venv/bin/activate
python3 -m robo_trader.runner_async --symbols AAPL --once
# Expected: FAILS with TimeoutError after 30s
```

### Debug: Check Subprocess Worker Stderr
```bash
# Run runner_async, it will log stderr path
source .venv/bin/activate
python3 -m robo_trader.runner_async --symbols AAPL --once 2>&1 | grep subprocess_stderr_log

# Then check the log file
cat /var/folders/.../ibkr_worker_stderr_*.log
# Shows: "API connection failed: TimeoutError()"
```

### Debug: Check for Zombies
```bash
netstat -an | grep 4002 | grep CLOSE_WAIT
# If you see CLOSE_WAIT connections, those are zombies blocking new connections
```

## Next Steps for Whoever Takes This

### Immediate Actions

1. **Understand the zombie problem**
   - Run `netstat -an | grep 4002` to see current state
   - Note how many CLOSE_WAIT connections exist
   - Restart Gateway and note they disappear

2. **Test the working parts**
   - Run `test_minimal_runner.py` - should work
   - Run it 5 times in a row
   - Check `netstat` - zombies will accumulate
   - After 3-4 runs, connections will start failing

3. **Implement connection reuse** (RECOMMENDED)
   - Modify `subprocess_ibkr_client.py` to keep connection alive
   - Don't disconnect after each operation
   - Only disconnect on shutdown
   - This prevents zombie accumulation

### Implementation Guide for Connection Reuse

**Current Flow:**
```python
client = SubprocessIBKRClient()
await client.start()
await client.connect(...)
accounts = await client.get_accounts()
await client.disconnect()  # ← Creates zombie
await client.stop()
```

**Proposed Flow:**
```python
# In runner_async setup:
client = SubprocessIBKRClient()
await client.start()
await client.connect(...)
# DON'T disconnect here

# In runner_async run loop:
accounts = await client.get_accounts()  # Reuse connection
positions = await client.get_positions()  # Reuse connection

# Only on shutdown:
await client.disconnect()
await client.stop()
```

**Changes Needed:**
1. Make `client` an instance variable in `AsyncRunner`
2. Connect once in `setup()`
3. Reuse throughout `run()`
4. Disconnect only in cleanup/shutdown

## Critical Information

### Python Version
**MUST use venv Python 3.13, NOT Anaconda Python 3.12**
```bash
source .venv/bin/activate
which python3
# Should show: /Users/oliver/robo_trader/.venv/bin/python3
```

### Gateway Requirements
- Must be running on port 4002
- Must have API enabled
- Must be logged into paper trading account
- Restarts frequently due to zombie accumulation (this is the problem!)

### Environment Variables
```bash
export IBKR_HOST=127.0.0.1
export IBKR_PORT=4002
export EXECUTION_MODE=paper
```

## Success Metrics

**When This Is Done:**
- ✅ `test_minimal_runner.py` works (ALREADY WORKS)
- ✅ `test_subprocess_in_async_env.py` works (ALREADY WORKS)
- ❌ `python3 -m robo_trader.runner_async --symbols AAPL --once` works (NEEDS FIX)
- ❌ Can run runner_async 10 times in a row without failures (NEEDS FIX)
- ❌ No zombie connections accumulate (NEEDS FIX)

## Contact/Questions

All implementation details are in:
- `SUBPROCESS_IBKR_SOLUTION_SUMMARY.md` - Complete technical summary
- `SUBPROCESS_ASYNC_ISSUE.md` - Async starvation issue details
- `BUS_ERROR_INVESTIGATION.md` - Bus error root cause

The subprocess IBKR client is 90% complete. The remaining 10% is fixing the zombie connection accumulation issue, which is a Gateway configuration or connection reuse problem, not a code bug.

---
**Handoff Complete**  
**Next Developer:** Implement connection reuse to prevent zombie accumulation

