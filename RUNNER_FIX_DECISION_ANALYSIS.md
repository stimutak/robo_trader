# Runner Fix Decision Analysis

## Problem Statement

`runner_async.py` crashes with Bus Error 10 on import, preventing the trading system from running and the dashboard from connecting.

## Analysis

### runner_async.py Complexity
- **Lines of code:** 2,345
- **Import statements:** 29+ modules
- **Dependencies:** ML, analytics, correlation, portfolio, risk, monitoring, websocket, database
- **Features:** Parallel processing, ML strategies, correlation sizing, advanced risk, production monitoring

### Bus Error Investigation
- **Symptom:** Segmentation fault (Bus Error 10) during module import
- **Location:** After WebSocket client connects, before main code runs
- **Cause:** Likely C extension incompatibility (numpy, pandas, ib_async, or other)
- **Pre-existing:** Exists in commits before subprocess work
- **Reproducible:** 100% crash rate on import

### Debugging Complexity
To fix the bus error, we would need to:
1. Binary search through 29+ imports to find the culprit
2. Check C extension versions (numpy, pandas, scipy, sklearn, etc.)
3. Test different library combinations
4. Potentially downgrade/upgrade conflicting libraries
5. Risk breaking other parts of the system
6. May uncover additional hidden issues

## Option 1: Fix runner_async.py

### Approach
1. Comment out imports one by one to isolate the crash
2. Identify the problematic C extension
3. Fix version conflicts or code issues
4. Test all features still work
5. Ensure no other bus errors lurk

### Pros
- ✅ Preserves all existing features
- ✅ Dashboard integration already exists
- ✅ ML strategies, correlation sizing, advanced risk all implemented
- ✅ Production monitoring in place

### Cons
- ❌ Time-consuming debugging (4-8+ hours, uncertain)
- ❌ Root cause unclear (could be any of 29+ imports)
- ❌ May uncover additional issues
- ❌ Risk of breaking working features
- ❌ Complex codebase (2,345 lines)
- ❌ High uncertainty in timeline

### Estimated Effort
- **Best case:** 4 hours (find issue quickly, simple fix)
- **Likely case:** 6-8 hours (multiple issues, version conflicts)
- **Worst case:** 12+ hours (deep C extension debugging)
- **Uncertainty:** HIGH

## Option 2: Create New Simple Runner

### Approach
1. Start with `test_minimal_runner.py` (already works)
2. Add essential trading logic:
   - Position tracking
   - Order execution
   - Basic risk management
3. Add WebSocket integration for dashboard
4. Keep it simple - only core features
5. Can add advanced features incrementally later

### Pros
- ✅ Fast time to production (2-4 hours, predictable)
- ✅ Proven foundation (minimal_runner works perfectly)
- ✅ No C extension issues
- ✅ Clean, maintainable codebase
- ✅ Easy to debug
- ✅ Can add features incrementally
- ✅ Lower risk

### Cons
- ❌ Need to reimplement some logic
- ❌ Won't have all features initially (ML, correlation, etc.)
- ❌ Dashboard may need minor updates
- ❌ Advanced features come later

### Estimated Effort
- **Core runner:** 2 hours
- **Dashboard integration:** 1 hour
- **Testing:** 1 hour
- **Total:** 4 hours
- **Uncertainty:** LOW

### What to Include (MVP)

**Essential (Phase 1 - 2 hours):**
- ✅ Connect to IBKR via subprocess client
- ✅ Fetch positions and account data
- ✅ Execute orders (buy/sell)
- ✅ Basic position tracking
- ✅ Simple risk checks (position limits)
- ✅ Logging and error handling

**Dashboard Integration (Phase 2 - 1 hour):**
- ✅ WebSocket connection
- ✅ Send position updates
- ✅ Send order updates
- ✅ Send connection status

**Nice to Have (Phase 3 - Later):**
- ⏳ ML strategies
- ⏳ Correlation sizing
- ⏳ Advanced risk management
- ⏳ Production monitoring
- ⏳ Parallel symbol processing

### What to Exclude (For Now)
- ML strategies (can add later)
- Correlation-based sizing (can add later)
- Advanced risk monitoring (basic risk is enough)
- Production monitoring (can add later)
- Parallel processing (single-threaded is fine for MVP)

## Comparison Matrix

| Criteria | Fix runner_async | New Simple Runner |
|----------|------------------|-------------------|
| **Time to Production** | 4-12 hours | 4 hours |
| **Uncertainty** | HIGH | LOW |
| **Risk** | HIGH | LOW |
| **Features (Initial)** | ALL | CORE |
| **Maintainability** | COMPLEX | SIMPLE |
| **Debugging** | HARD | EASY |
| **Success Probability** | 70% | 95% |

## Recommendation: **Create New Simple Runner**

### Rationale

1. **Time-Critical:** System has been down for a month. Need to get trading ASAP.

2. **Predictable Timeline:** 4 hours vs 4-12+ hours with high uncertainty.

3. **Lower Risk:** No C extension debugging, no hidden issues.

4. **Proven Foundation:** minimal_runner already works perfectly with subprocess client.

5. **Incremental Improvement:** Can add advanced features later once core is stable.

6. **Better Architecture:** Opportunity to build cleaner, more maintainable system.

### Implementation Plan

**Phase 1: Core Runner (2 hours)**
```python
# simple_runner.py
- Connect via subprocess IBKR client ✅ (already works)
- Fetch positions and account data ✅ (already works)
- Execute orders (buy/sell)
- Track positions
- Basic risk checks
- Logging
```

**Phase 2: Dashboard Integration (1 hour)**
```python
- WebSocket client connection
- Send position updates
- Send order updates
- Send connection status
```

**Phase 3: Testing (1 hour)**
```python
- Test connection stability
- Test order execution
- Test dashboard updates
- Test error handling
```

**Phase 4: Deployment (30 min)**
```python
- Update startup scripts
- Update documentation
- Deploy to production
```

**Total: 4.5 hours to production-ready system**

### Future Enhancements (Post-MVP)
Once core is stable and trading:
- Add ML strategies (1-2 days)
- Add correlation sizing (1 day)
- Add advanced risk monitoring (1 day)
- Add production monitoring (1 day)
- Add parallel processing (1 day)

## Decision

**✅ CREATE NEW SIMPLE RUNNER**

**Why:**
- Faster (4 hours vs 4-12+ hours)
- Lower risk (95% vs 70% success)
- Predictable timeline
- Proven foundation
- Can add features incrementally

**Next Steps:**
1. Create `robo_trader/simple_runner.py` based on `test_minimal_runner.py`
2. Add order execution and position tracking
3. Add WebSocket integration
4. Test thoroughly
5. Deploy

---
**Date:** 2025-10-15
**Decision:** Create new simple runner
**Timeline:** 4 hours to production
**Risk:** LOW

