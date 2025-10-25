# Quick Start for Next Developer

## TL;DR

**Status:** Subprocess IBKR client is 90% done and works perfectly in isolation. One issue remains: zombie connections prevent `runner_async.py` from connecting.

**The Fix:** Implement connection reuse (keep connection alive instead of connect/disconnect cycles).

**Time Estimate:** 1-2 hours

## What You Need to Know

### 1. The Subprocess Client Works!

```bash
cd /Users/oliver/robo_trader
source .venv/bin/activate  # CRITICAL: Must use venv Python 3.13

# This works perfectly:
python3 test_minimal_runner.py
# Result: Connected in 0.4s, SUCCESS ✅

# This also works:
python3 test_subprocess_in_async_env.py
# Result: Connected with background tasks, SUCCESS ✅

# This fails:
python3 -m robo_trader.runner_async --symbols AAPL --once
# Result: TimeoutError after 30s ❌
```

### 2. The Problem

**Zombie connections accumulate and block new connections.**

Every time you connect and disconnect, a CLOSE_WAIT zombie is created:
```bash
netstat -an | grep 4002 | grep CLOSE_WAIT
# After 3-4 test runs, you'll see zombies
# After ~5 zombies, Gateway stops accepting new connections
```

### 3. The Solution

**Keep the connection alive instead of reconnecting every time.**

**Current (creates zombies):**
```python
# Every operation does this:
client = SubprocessIBKRClient()
await client.start()
await client.connect()
await client.get_accounts()
await client.disconnect()  # ← Creates zombie!
await client.stop()
```

**Proposed (no zombies):**
```python
# Connect once:
client = SubprocessIBKRClient()
await client.start()
await client.connect()

# Reuse many times:
await client.get_accounts()
await client.get_positions()
await client.get_account_summary()

# Disconnect only on shutdown:
await client.disconnect()
await client.stop()
```

## Implementation Steps

### Step 1: Modify runner_async.py

**File:** `robo_trader/runner_async.py`

**Change 1:** Make client an instance variable
```python
class AsyncRunner:
    def __init__(self, ...):
        self.ib_client = None  # Add this
```

**Change 2:** Connect once in setup()
```python
async def setup(self):
    # ... existing code ...
    
    # Connect to IBKR (keep connection alive)
    self.ib_client = await connect_ibkr_robust(...)
    
    # DON'T disconnect here!
```

**Change 3:** Reuse in run()
```python
async def run(self):
    while True:
        # Use self.ib_client instead of reconnecting
        positions = await self.ib_client.get_positions()
        # ... rest of logic ...
```

**Change 4:** Disconnect only on shutdown
```python
async def cleanup(self):
    if self.ib_client:
        await self.ib_client.disconnect()
        await self.ib_client.stop()
```

### Step 2: Test

```bash
source .venv/bin/activate

# Test once
python3 -m robo_trader.runner_async --symbols AAPL --once

# Test multiple times (should not accumulate zombies)
for i in {1..5}; do
    echo "Run $i"
    python3 -m robo_trader.runner_async --symbols AAPL --once
    netstat -an | grep 4002 | grep CLOSE_WAIT | wc -l
done

# Should show 0 zombies after each run
```

### Step 3: Verify

```bash
# Check no zombies
netstat -an | grep 4002 | grep CLOSE_WAIT
# Should be empty

# Run full system
python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA
# Should connect and run successfully
```

## Files to Modify

1. **`robo_trader/runner_async.py`** - Main changes here
   - Add `self.ib_client` instance variable
   - Connect once in `setup()`
   - Reuse in `run()`
   - Disconnect in `cleanup()`

2. **Optional:** `robo_trader/utils/robust_connection.py`
   - Could add a `keep_alive=True` parameter
   - But not necessary for MVP

## Testing Commands

```bash
# Activate venv (REQUIRED!)
cd /Users/oliver/robo_trader
source .venv/bin/activate

# Test minimal runner (should work)
python3 test_minimal_runner.py

# Test full runner (currently fails, should work after fix)
python3 -m robo_trader.runner_async --symbols AAPL --once

# Check for zombies
netstat -an | grep 4002 | grep CLOSE_WAIT

# Kill zombies if needed (but fix should prevent them)
lsof -ti tcp:4002 -sTCP:CLOSE_WAIT | xargs kill -9
```

## Important Notes

### Python Version
**MUST use venv Python 3.13!**
```bash
source .venv/bin/activate
which python3
# Must show: /Users/oliver/robo_trader/.venv/bin/python3
```

If you use Anaconda Python 3.12, you'll get Bus Error 10.

### Gateway
- Must be running on port 4002
- Must be logged into paper trading account
- Will need occasional restarts due to zombie accumulation (until fix is implemented)

### Branch
```bash
git checkout fix/subprocess-ibkr-wrapper
```

## Success Criteria

After implementing connection reuse:

- ✅ `python3 -m robo_trader.runner_async --symbols AAPL --once` works
- ✅ Can run 10 times in a row without failures
- ✅ No zombie CLOSE_WAIT connections accumulate
- ✅ Dashboard connects and shows data

## Documentation

Full details in:
- **`handoff/2025-10-15_1913_subprocess_ibkr_handoff.md`** - Complete handoff
- **`SUBPROCESS_IBKR_SOLUTION_SUMMARY.md`** - Technical summary
- **`SUBPROCESS_ASYNC_ISSUE.md`** - Async starvation fix details

## Questions?

All the hard work is done:
- ✅ Subprocess isolation works
- ✅ Threading prevents async starvation
- ✅ Bus error fixed
- ✅ Connection is fast and reliable

Just need to implement connection reuse to prevent zombies!

---
**Estimated Time:** 1-2 hours  
**Difficulty:** Easy (just refactoring, no new code needed)  
**Impact:** Completes the month-long IBKR connection fix!

