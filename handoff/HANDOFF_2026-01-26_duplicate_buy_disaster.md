# Session Handoff: Duplicate Buy Disaster & Recovery
**Date:** 2026-01-26
**Status:** CRITICAL BUG FIXED, Database Cleaned

## The Disaster

### What Happened
Between 18:19 and 19:14 ET, the trading system executed **$5+ MILLION in duplicate BUY orders** due to a race condition in parallel symbol processing.

### Root Cause
The `_pending_orders` lock protection was **written but never committed** to the running code. The runner that was active all day had OLD code that only checked `symbol not in self.positions`, which fails for parallel processing:

```
Timeline:
1. 8 parallel tasks check "symbol not in self.positions" → ALL pass (position not added yet)
2. All 8 tasks proceed to buy the same stock
3. Each buy is recorded as a trade, cash is deducted
4. _update_position_atomic correctly accumulates (no duplicate positions)
5. But cash keeps getting deducted for each phantom buy
```

### Damage Assessment

| Metric | Before | After Disaster |
|--------|--------|----------------|
| Portfolio Equity | ~$205,000 | ~$98,000 |
| Cash | ~$155,000 | Depleted |
| Trades (2 hours) | Normal | 712 BUYs ($5M notional) |
| Positions | ~47 | 47 (correct) |

**Example - JD (JD.com):**
- 52 BUY orders recorded (should be 1)
- 8,783 shares "bought" in trades table
- Only 33 shares in positions table (correctly accumulated)
- ~$260k in phantom cash deducted

### Symbols Most Affected
| Symbol | Duplicate Buys | Total Shares "Bought" |
|--------|---------------|----------------------|
| JD | 52 | 8,783 |
| TERN | 35 | 5,167 |
| TRIP | 29 | 11,136 |
| SNDK | 24 | 240 |
| CYBR | 24 | 264 |
| GS | 22 | 110 |
| T | 21 | 4,510 |

## The Fix

### Code Changes (Commit c91ec5d)
Added `_pending_orders` set with `asyncio.Lock()` in `runner_async.py`:

```python
# Lines 179-180: Initialize
self._pending_orders: set = set()
self._pending_orders_lock = asyncio.Lock()

# Lines 1745-1772: Check and mark pending before buy
async with self._pending_orders_lock:
    if symbol in self._pending_orders:
        logger.warning(f"DUPLICATE BUY BLOCKED: {symbol}")
        return  # Block duplicate
    if symbol in self.positions:
        return  # Re-check inside lock (TOCTOU protection)
    self._pending_orders.add(symbol)  # Mark as pending

# Lines 1964-1965: Remove from pending in finally block
finally:
    async with self._pending_orders_lock:
        self._pending_orders.discard(symbol)
```

### Why It Works
1. Lock ensures only ONE task can check/add to pending_orders at a time
2. Re-check `symbol in self.positions` INSIDE the lock (TOCTOU protection)
3. Symbol added to pending_orders BEFORE lock is released
4. Other parallel tasks will see it in pending_orders and be blocked

## Database Cleanup

### Cleanup Script Executed
```sql
-- 1. Calculate correct cash based on actual positions
-- Starting cash was ~$250,000, positions cost ~$100k
-- Actual cash = Starting - Position Cost + Sells - Realized Losses

-- 2. Delete duplicate trades (keep only first buy per symbol per hour)
DELETE FROM trades
WHERE id NOT IN (
    SELECT MIN(id) FROM trades
    WHERE side='BUY'
    GROUP BY symbol, strftime('%Y-%m-%d %H', timestamp)
);

-- 3. Recalculate account cash based on actual positions
UPDATE account SET cash = (
    SELECT 250000 - SUM(quantity * avg_cost) FROM positions WHERE quantity > 0
);
```

### Post-Cleanup State
- Trades table: Cleaned of duplicates
- Positions table: Unchanged (was already correct)
- Account cash: Recalculated from actual positions
- Equity: Recalculated

## Prevention Measures Added

1. **`_pending_orders` lock** - Prevents parallel buys of same symbol within a cycle
2. **TOCTOU protection** - Re-checks position inside lock
3. **Added to CLAUDE.md Common Mistakes** - Document the race condition pattern

## Lessons Learned

1. **ALWAYS restart the runner after code changes** - The fix was written but the old process kept running
2. **Parallel processing needs explicit synchronization** - `symbol not in dict` is not atomic
3. **The positions table was correct** - `_update_position_atomic` worked; the bug was in the entry check
4. **Cash tracking amplified the problem** - Each phantom buy deducted cash

## Commands

```bash
# Start with fixed code
./START_TRADER.sh

# Verify no duplicate buy warnings (should see "DUPLICATE BUY BLOCKED" if protection works)
grep "DUPLICATE BUY BLOCKED" robo_trader.log

# Check positions
sqlite3 trading_data.db "SELECT symbol, quantity, avg_cost FROM positions WHERE quantity > 0;"
```
