# Handoff: Duplicate BUY Race Condition Fix

**Date:** 2026-01-27
**Session Focus:** Fix duplicate BUY orders being executed in parallel

## Problem

The trading system was executing multiple BUY orders for the same symbol simultaneously:
- Example: 4 APP buys (qty=1, 7, 7, 7) within 2 seconds
- Example: 3 GME buys (qty=167 each) in the same second
- Caused position sprawl and unintended exposure

## Root Cause Analysis

1. **Fresh runner each cycle**: `AsyncRunner` is created fresh each trading cycle
2. **In-memory checks reset**: `_pending_orders` set was reset with each new runner
3. **Parallel processing race**: Multiple tasks check `symbol not in positions` simultaneously
4. **Database lag**: Position written to DB after multiple tasks already passed the check

## Solution: 3-Layer Protection

### Layer 1: Cycle-Level Set (`_cycle_executed_buys`)
```python
# In __init__:
self._cycle_executed_buys: set = set()
self._cycle_executed_buys_lock = asyncio.Lock()

# At start of BUY signal handling:
async with self._cycle_executed_buys_lock:
    if symbol in self._cycle_executed_buys:
        return  # BLOCKED
    self._cycle_executed_buys.add(symbol)  # Mark IMMEDIATELY
```

### Layer 2: Pending Orders Lock
```python
async with self._pending_orders_lock:
    if symbol in self._pending_orders:
        return  # BLOCKED
    if symbol in self.positions:
        return  # BLOCKED (TOCTOU race prevention)
    self._pending_orders.add(symbol)
```

### Layer 3: Database Check (Most Reliable)
```python
# Inside the pending orders lock:
db_positions = await self.db.get_positions()
if any(p["symbol"] == symbol and p["quantity"] > 0 for p in db_positions):
    self._pending_orders.discard(symbol)
    return  # BLOCKED - DB position exists
```

## Files Modified

1. **`robo_trader/runner_async.py`**
   - Lines 184-185: Added `_cycle_executed_buys` set and lock
   - Lines 1678-1695: Added cycle-level check at start of BUY signal
   - Lines 1798-1813: Added database position check inside pending orders lock

2. **`CLAUDE.md`**
   - Added issue #20-22 to Current Issues Status
   - Added "Duplicate BUY Race Condition Fix" documentation section
   - Updated Common Mistakes table with new patterns

3. **`app.py`**
   - Line 4172: Added `cash` to market-closed `/api/pnl` response
   - Line 4229: Added `cash` to market-open `/api/pnl` response
   - Line 4238: Added `cash` to error `/api/pnl` response

## Other Fixes This Session

1. **sklearn version mismatch warnings**: Re-serialized 82 ML models with sklearn 1.7.2
2. **Risk status API error**: Fixed `unsupported operand type(s) for +: 'int' and 'NoneType'` by adding `(t.get("pnl") or 0)` guards in `/api/risk/status`
3. **Missing `cash` in `/api/pnl` response**: Added `cash` field to all three return paths in the endpoint:
   - Market-open response (line 4229)
   - Market-closed response (line 4172)
   - Error response (line 4238)

   This enables the "Cash Available" display in the dashboard overview.

## Testing

After deploying the fix:
- New runner shows "Marked {symbol} for BUY processing this cycle" log messages
- Existing positions correctly show "Buy signal: Already have long position"
- Database check should show "DUPLICATE BUY BLOCKED: {symbol} already has DB position" for cross-cycle duplicates

## Known Limitations

- The database check adds a small latency (DB query) for each BUY attempt
- If DB is slow, there's still a tiny window for race conditions
- Consider adding a unique constraint on recent trades as future enhancement

## Equity Display Investigation

User expressed concern about equity display confidence. Investigation found:

**No calculation issues found** - All sources show consistent values:
- Database: `cash + SUM(qty * market_price)` = Correct equity
- `/api/pnl`: Uses same calculation
- `/api/status`: Uses same calculation

**Values verified consistent:**
```
Database account cash:     $144,420.92
Database account equity:   $201,581.22
Database positions value:  $57,160.30
API /api/pnl equity:       $201,581.22
API /api/status equity:    $201,581.22
```

**Issue found**: `/api/pnl` was missing `cash` field, causing "Cash Available" to show "—" in overview.

## Pending Task

Task #1 is still pending: "Make dashboard overview page live with real-time updates"
- Overview tab doesn't show real-time data updates
- Needs WebSocket integration or more frequent polling

## Commands to Verify

```bash
# Check recent trades for duplicates
sqlite3 trading_data.db "SELECT datetime(timestamp, 'localtime'), side, symbol, quantity FROM trades ORDER BY timestamp DESC LIMIT 20"

# Check for duplicate blocking logs
grep -E "DUPLICATE BUY BLOCKED|Marked.*BUY" robo_trader.log | tail -20

# Check current positions
curl -s http://localhost:5555/api/positions | python3 -c "import sys,json; [print(f'{p[\"symbol\"]:6} qty={p[\"quantity\"]}') for p in json.load(sys.stdin)['positions']]"
```
