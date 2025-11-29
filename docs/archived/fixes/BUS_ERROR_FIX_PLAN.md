# Bus Error Fix Plan

## Problem Identified

**Bus Error Location:** `robo_trader/strategies/framework.py`

**Cause:** C extension (numpy/pandas) segmentation fault when importing framework.py

**Evidence:**
- ✅ `from robo_trader.features.engine import FeatureSet` - Works
- ✅ `from robo_trader.websocket_client import ws_client` - Works  
- ✅ `from robo_trader.portfolio import Portfolio` - Works
- ❌ `from robo_trader.strategies.framework import Strategy` - Bus Error 10
- ❌ `from robo_trader.strategies import sma_crossover_signals` - Bus Error 10
- ❌ `from robo_trader import runner_async` - Bus Error 10

## Root Cause

The bus error happens when Python tries to execute code in `framework.py` that triggers a segmentation fault in a C extension (likely numpy or pandas).

**Likely culprits in framework.py:**
- Line 17: `import numpy as np`
- Line 18: `import pandas as pd`
- Or code that uses numpy/pandas in a way that triggers the segfault

## Fix Strategy

### Option 1: Quick Fix - Comment Out Problematic Code (30 min)
1. Binary search through framework.py to find exact line causing crash
2. Comment out or fix that specific code
3. Test if runner imports successfully
4. Integrate subprocess IBKR client

**Timeline:** 30 minutes - 1 hour
**Risk:** May break some strategy features

### Option 2: Library Version Fix (1-2 hours)
1. Check numpy/pandas versions
2. Try downgrading to known-good versions
3. Clear all pycache
4. Reinstall dependencies
5. Test

**Timeline:** 1-2 hours
**Risk:** May break other parts of system

### Option 3: Isolate Strategies (2-3 hours)
1. Move strategy imports to lazy loading
2. Only import when actually needed
3. Runner can start without strategies loaded
4. Load strategies on-demand

**Timeline:** 2-3 hours
**Risk:** LOW - clean solution

## Recommended Approach

**OPTION 1: Quick Fix**

Since we need to get trading working ASAP and we have a working subprocess IBKR client, let's:

1. **Find and fix the exact line causing bus error** (30 min)
2. **Test runner imports** (5 min)
3. **Integrate subprocess IBKR client** (already done, just need to test)
4. **Test full system** (30 min)

**Total: 1-1.5 hours to working system**

## Implementation Steps

### Step 1: Find Exact Line (15-30 min)
Create minimal test that imports framework.py line by line until crash

### Step 2: Fix or Comment Out (15 min)
Either fix the problematic code or comment it out temporarily

### Step 3: Test Runner (5 min)
```bash
python3 -c "from robo_trader import runner_async; print('OK')"
```

### Step 4: Test with Subprocess Client (30 min)
```bash
python3 -m robo_trader.runner_async --symbols AAPL --once
```

### Step 5: Full Integration Test (30 min)
- Test connection
- Test order execution
- Test dashboard updates
- Test all features

## Next Actions

1. Create detailed framework.py line-by-line test
2. Find exact crash location
3. Fix or work around
4. Test runner
5. Deploy with subprocess IBKR client

---
**Timeline:** 1-1.5 hours to working system with all features
**Status:** Ready to proceed

