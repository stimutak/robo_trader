# Handoff: Duplicate Trade Fix

**Date:** 2026-01-28
**Issue:** 352 duplicate BUY trade pairs across 62 symbols
**Status:** FIXED

## Problem Summary

The trading system was executing duplicate BUY trades for the same symbol, including:
- Same-second duplicates (e.g., APP IDs 2922 & 2923 at exact same timestamp)
- Cross-cycle duplicates (trades 60-300 seconds apart)

**Impact:** Excessive position accumulation, wasted capital, incorrect P&L tracking

## Root Causes Identified

### 1. Fresh AsyncRunner Each Cycle
```python
# Line 2966-2969 in runner_async.py
# Create fresh runner each cycle for stability
runner = AsyncRunner(...)
```

This resets in-memory protection mechanisms every cycle:
- `_cycle_executed_buys` → empty set
- `_pending_orders` → empty set
- `self.positions` → needs reload from DB

### 2. Pairs Trading Bypassed Duplicate Protection

The pairs trading section (lines 2550-2740) executed BUY orders WITHOUT checking:
- `_cycle_executed_buys` set
- Recent BUY trades in database
- Only checked for positions > 100 shares (not > 0)

A symbol could receive:
1. BUY from main strategy → records trade
2. BUY from pairs trading → records ANOTHER trade (same second!)

### 3. Position Check Threshold Too High

```python
# OLD (line 2551):
has_position_a = symbol_a in self.positions and self.positions[symbol_a].quantity > 100

# This allowed positions with 1-100 shares to be ignored!
```

## Fixes Applied

### Fix 1: Database-Level Duplicate Check (database_async.py)

Added `has_recent_buy_trade()` method that checks the trades table for recent BUYs:

```python
async def has_recent_buy_trade(self, symbol: str, seconds: int = 60) -> bool:
    """Check if a BUY trade for the symbol exists within the last N seconds."""
    async with self.get_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT COUNT(*) FROM trades
            WHERE symbol = ?
            AND side = 'BUY'
            AND timestamp > datetime('now', ? || ' seconds')
            """,
            (symbol, f"-{seconds}"),
        )
        row = await cursor.fetchone()
        return row[0] > 0 if row else False
```

### Fix 2: Main BUY Flow Protection (runner_async.py:1843-1857)

Added additional check for recent trades after the DB position check:

```python
# ADDITIONAL CHECK: Look for recent BUY trades (handles race conditions)
recent_buy = await self.db.has_recent_buy_trade(symbol, seconds=120)
if recent_buy:
    logger.warning(
        f"DUPLICATE BUY BLOCKED: {symbol} has recent BUY trade in last 120 seconds"
    )
    self._pending_orders.discard(symbol)
    return SymbolResult(...)
```

### Fix 3: Pairs Trading Protection (runner_async.py:2550-2579)

1. Fixed position threshold from `> 100` to `> 0`
2. Added recent trade check before executing pairs trades:

```python
# Check for existing positions (qty > 0, not > 100)
has_position_a = symbol_a in self.positions and self.positions[symbol_a].quantity > 0

# CRITICAL: Check DB for recent BUY trades
recent_buy_a = await self.db.has_recent_buy_trade(symbol_a, seconds=120)
recent_buy_b = await self.db.has_recent_buy_trade(symbol_b, seconds=120)

if recent_buy_a or recent_buy_b:
    logger.warning(
        f"DUPLICATE BLOCKED: Skipping pairs trade for {symbol_a}-{symbol_b}: "
        f"recent BUY exists (A={recent_buy_a}, B={recent_buy_b})"
    )
    continue
```

## Files Modified

1. **robo_trader/database_async.py**
   - Added `has_recent_buy_trade()` method

2. **robo_trader/runner_async.py**
   - Added recent trade check in main BUY flow (lines 1843-1857)
   - Fixed pairs trading position check threshold (line 2551)
   - Added pairs trading duplicate protection (lines 2564-2579)

## Protection Layers (Updated)

The system now has 4 layers of duplicate BUY protection:

| Layer | Location | Scope | Persists Across Cycles? |
|-------|----------|-------|------------------------|
| 1. Cycle set | `_cycle_executed_buys` | Within cycle | NO |
| 2. Pending lock | `_pending_orders_lock` | Within cycle | NO |
| 3. DB positions | `get_positions()` | Cross-cycle | YES |
| 4. Recent trades | `has_recent_buy_trade()` | Cross-cycle | YES |

Layers 3 and 4 are the critical ones since in-memory state resets each cycle.

## Verification

After fix, monitor logs for:
```
DUPLICATE BUY BLOCKED: {symbol} has recent BUY trade
DUPLICATE BLOCKED: Skipping pairs trade for {symbol_a}-{symbol_b}
```

To check for new duplicates:
```sql
SELECT t1.id, t2.id, t1.symbol, t1.timestamp, t2.timestamp
FROM trades t1
JOIN trades t2 ON t1.symbol = t2.symbol
              AND t1.side = t2.side
              AND t1.id < t2.id
              AND abs(strftime('%s', t1.timestamp) - strftime('%s', t2.timestamp)) < 300
WHERE t1.timestamp > datetime('now', '-1 day')
  AND t1.side = 'BUY';
```

## Historical Duplicates

The fix does NOT retroactively remove historical duplicates. 352 duplicate pairs exist in the database from before this fix. These affect:
- Position quantities (accumulated incorrectly)
- P&L calculations

To rebuild positions from trades if needed, use FIFO accounting as documented in `HANDOFF_2026-01-27_position_db_rebuild.md`.

## Lessons Learned

1. **Database-level checks are essential** when in-memory state doesn't persist
2. **All trade execution paths need protection** - not just the main strategy
3. **Position thresholds matter** - checking `> 100` instead of `> 0` is a bug
4. **Same-second duplicates indicate parallel execution issues** - need atomic checks
