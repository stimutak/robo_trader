# Latest Handoff: Duplicate Trade Fix

**Date:** 2026-01-28
**Session Focus:** Fix 352 duplicate BUY trades across 62 symbols

## Executive Summary

Fixed critical duplicate trade bug where same symbol received multiple BUY orders:
- Same-second duplicates (e.g., APP IDs 2922 & 2923)
- Cross-cycle duplicates (60-300 seconds apart)

**Root Causes:**
1. Fresh AsyncRunner each cycle resets in-memory protection
2. Pairs trading bypassed duplicate checks entirely
3. Position threshold `> 100` instead of `> 0`

## Fixes Applied

### 1. Database-Level Duplicate Check

Added `has_recent_buy_trade(symbol, seconds)` to `database_async.py`:
```python
async def has_recent_buy_trade(self, symbol: str, seconds: int = 60) -> bool:
    """Check if a BUY trade for the symbol exists within the last N seconds."""
    ...
    WHERE symbol = ? AND side = 'BUY'
    AND timestamp > datetime('now', ? || ' seconds')
```

### 2. Main BUY Flow Protection

Added recent trade check after DB position check (`runner_async.py:1843-1857`):
```python
recent_buy = await self.db.has_recent_buy_trade(symbol, seconds=120)
if recent_buy:
    logger.warning(f"DUPLICATE BUY BLOCKED: {symbol}")
    return SymbolResult(...)
```

### 3. Pairs Trading Protection

Fixed position threshold and added duplicate check (`runner_async.py:2550-2579`):
```python
# Fixed: > 0 instead of > 100
has_position_a = symbol_a in self.positions and self.positions[symbol_a].quantity > 0

# Added duplicate check
recent_buy_a = await self.db.has_recent_buy_trade(symbol_a, seconds=120)
if recent_buy_a or recent_buy_b:
    logger.warning(f"DUPLICATE BLOCKED: Skipping pairs trade")
    continue
```

## Files Modified

| File | Changes |
|------|---------|
| `robo_trader/database_async.py` | Added `has_recent_buy_trade()` method |
| `robo_trader/runner_async.py` | Added main flow + pairs trading duplicate protection |
| `CLAUDE.md` | Added 3 new entries to Common Mistakes table |
| `handoff/HANDOFF_2026-01-28_duplicate_trade_fix.md` | Full documentation |

## Protection Layers (Now 4 Layers)

| Layer | Location | Persists Across Cycles? |
|-------|----------|------------------------|
| 1. Cycle set | `_cycle_executed_buys` | NO |
| 2. Pending lock | `_pending_orders_lock` | NO |
| 3. DB positions | `get_positions()` | YES |
| 4. Recent trades | `has_recent_buy_trade()` | YES |

Layers 3 and 4 are critical since in-memory state resets each cycle.

## Verification

Monitor logs for:
```
DUPLICATE BUY BLOCKED: {symbol} has recent BUY trade
DUPLICATE BLOCKED: Skipping pairs trade for {symbol_a}-{symbol_b}
```

Check for new duplicates:
```sql
SELECT t1.id, t2.id, t1.symbol, t1.timestamp
FROM trades t1
JOIN trades t2 ON t1.symbol = t2.symbol
              AND t1.side = t2.side
              AND t1.id < t2.id
              AND abs(strftime('%s', t1.timestamp) - strftime('%s', t2.timestamp)) < 300
WHERE t1.timestamp > datetime('now', '-1 day') AND t1.side = 'BUY';
```

## Previous Sessions

- **2026-01-27:** Position DB rebuild, extended hours trading, dashboard fixes
- **2026-01-26:** Dashboard overview redesign, equity history tracking
- **2026-01-15:** Decimal/Float type fixes, market holidays

## Full Details

See: `handoff/HANDOFF_2026-01-28_duplicate_trade_fix.md`
