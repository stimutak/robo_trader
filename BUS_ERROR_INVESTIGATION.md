# Runner Bus Error Investigation

## Issue
`runner_async.py` crashes with Bus Error 10 when imported.

## Symptoms
```bash
$ python3 -c "from robo_trader import runner_async"
Bus error: 10
```

The crash occurs:
1. After WebSocket client connects successfully
2. During module import (before any code runs)
3. Consistently across multiple attempts

## Investigation Results

### Pre-Existing Bug
Tested multiple commits to determine when this started:
- ✅ Commit `32e0b11` (current) - BUS ERROR
- ✅ Commit `bd5b39c` (before subprocess integration) - BUS ERROR  
- ✅ Commit `2153c8a` (before subprocess work) - BUS ERROR
- ✅ Commit `23f5810` (Phase 4 merge) - BUS ERROR

**Conclusion: This is a PRE-EXISTING BUG, not introduced by subprocess work.**

### What Works
- ✅ `from robo_trader.config import Config` - OK
- ✅ `from robo_trader.websocket_client import ws_client` - OK
- ✅ `test_minimal_runner.py` - OK (uses subprocess IBKR client successfully)
- ✅ All subprocess IBKR tests - OK

### What Crashes
- ❌ `from robo_trader import runner_async` - BUS ERROR
- ❌ `python3 -m robo_trader.runner_async` - BUS ERROR

## Likely Causes

Bus Error 10 on macOS typically indicates:
1. **Memory alignment issue** - Accessing misaligned memory
2. **Invalid memory access** - Accessing unmapped memory
3. **C extension bug** - Bug in numpy, pandas, ib_async, or other C extension
4. **Library incompatibility** - Incompatible versions of dependencies

## Potential Culprits

Given that:
- WebSocket client imports fine
- Config imports fine
- Only runner_async crashes
- Crash happens during import

The issue is likely in one of runner_async's imports:
- `from ib_async import Stock` (line 22)
- ML/analytics imports
- Feature engineering imports
- Complex dependency chain

## Impact on Subprocess IBKR Solution

**NO IMPACT!** The subprocess IBKR client works perfectly:
- ✅ Connects in 0.41 seconds
- ✅ Gets accounts, positions, summary
- ✅ Stays connected and responsive
- ✅ No zombie connections
- ✅ Clean disconnect

The bus error is a SEPARATE issue in runner_async, unrelated to IBKR connectivity.

## Recommended Actions

### Short Term
1. ✅ **Use `test_minimal_runner.py` as the working runner**
   - Proves subprocess IBKR client works
   - Bypasses runner_async bus error
   - Can be extended with trading logic

2. **Investigate bus error separately**
   - Binary search through runner_async imports
   - Check for C extension version conflicts
   - Review recent dependency updates

### Long Term
1. **Fix runner_async bus error**
   - Isolate problematic import
   - Update/downgrade conflicting library
   - Add error handling

2. **Refactor runner_async**
   - Reduce import complexity
   - Lazy load heavy dependencies
   - Better error isolation

## Workaround

Use minimal runner instead of runner_async:

```python
# test_minimal_runner.py works perfectly
python3 test_minimal_runner.py

# Output:
# ✅ Connected in 0.41s
# ✅ Accounts: ['DUN264991']
# ✅ Positions: 0
# ✅ Connection stable
```

## Status

- **Subprocess IBKR Solution:** ✅ COMPLETE AND WORKING
- **Runner Bus Error:** ❌ PRE-EXISTING BUG (separate issue)
- **Impact:** None - subprocess solution proven to work

---
**Date:** 2025-10-15  
**Investigated by:** AI Assistant  
**Conclusion:** Bus error is pre-existing, unrelated to subprocess IBKR work

