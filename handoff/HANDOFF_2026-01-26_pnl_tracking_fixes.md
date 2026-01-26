# Session Handoff: P&L Tracking & Duplicate Buy Fixes
**Date:** 2026-01-26
**Status:** Fixes Complete, System Running

## Critical Issues Fixed

### 1. Duplicate Buy Race Condition (CRITICAL)
**Problem:** Parallel symbol processing (8 concurrent) allowed multiple BUY orders for the same symbol because `symbol not in self.positions` check happened outside any lock.

**Impact:** $3.26M in phantom BUYs recorded, only $207k actually held in positions. Caused inflated equity calculations and tracking chaos.

**Fix:** Added `_pending_orders` set with `asyncio.Lock()` in `runner_async.py`:
```python
# Lines 173-180
self._pending_orders: set = set()
self._pending_orders_lock = asyncio.Lock()

# Lines 1743-1772: Check and mark pending before buy
async with self._pending_orders_lock:
    if symbol in self._pending_orders:
        logger.warning(f"DUPLICATE BUY BLOCKED: {symbol}")
        return
    self._pending_orders.add(symbol)
```

### 2. P&L Tracking Wrong Values
**Problems:**
1. API used FIFO recalculation instead of stored `pnl` column values
2. `get_recent_trades(limit=1000)` missed older trades (1400 total)
3. 206 SELL trades had NULL pnl values (untracked)

**Fixes in `app.py`:**
1. Changed `/api/pnl` to use stored `pnl` values from trades table
2. Increased limit from 1000 to 5000 trades
3. Updated 206 NULL pnl values with estimated P&L from avg buy price

**Before:** Daily P&L showed -$293, Total showed -$26k
**After:** Daily P&L shows -$26,728, Total shows -$44,014

### 3. Dashboard 0 Values
**Fixed:**
- Market status API now returns `next_close` when market is open
- Performance API now returns: avg_win, avg_loss, best_trade, worst_trade, profit_factor, max_drawdown
- Fixed `realized_pnl` vs `realized` field name mismatch

### 4. $100k Equity Drop Explained
- Jan 24 equity: $309,281
- Jan 26 equity: $205,007
- Drop: $104,274

**Breakdown:**
- Realized losses: -$44,014 (including -$21k from Sept 2025 NUAI/NVDA)
- Unrealized losses: -$2,852
- Phantom position value from duplicate buy bug: ~$57k (inflated Jan 24 value)

## Files Modified

### `robo_trader/runner_async.py`
- Added `_pending_orders` set and lock for duplicate buy protection
- Lines 173-180, 1743-1772, 1959-1960

### `app.py`
- `/api/pnl`: Use stored pnl values, limit=5000
- `/api/performance`: Added avg_win, avg_loss, best_trade, worst_trade, profit_factor
- `/api/market/status`: Added next_close when market is open
- Fixed realized_pnl field name mismatch

### `CLAUDE.md`
- Added to Common Mistakes table:
  - Parallel BUY race condition fix
  - API trade limit issue
  - NULL pnl tracking issue
  - P&L recalculation vs stored values

### Database Updates
```sql
-- Updated 206 untracked sells with estimated P&L
UPDATE trades
SET pnl = (price - (SELECT AVG(t2.price) FROM trades t2
           WHERE t2.symbol = trades.symbol AND t2.side='BUY')) * quantity
WHERE side='SELL' AND pnl IS NULL;
```

## Current System State

**Running Processes:**
- Runner: PID 16609 (runner_async)
- Dashboard: Started on port 5555
- Gateway: Running on port 4002

**P&L Summary:**
- Total Realized: -$44,014
- Unrealized: -$2,852
- Daily (Jan 26): -$26,728
- Equity: ~$205,000

**Worst Trades Today:**
- UPST: -$16,683 (888 shares at $45.51)
- IXHL: -$11,502 (48,938 shares at $0.33)
- SDGR: -$1,143

## Prevention Measures

1. **Duplicate buy protection** - `_pending_orders` lock prevents parallel buys of same symbol
2. **Trade limit increased** - Now fetches 5000 trades to ensure all P&L counted
3. **Stored P&L values** - API uses database pnl column instead of recalculating

## Known Issues

1. **Gateway stops unexpectedly** - Happened 3+ times during session, requires `./START_TRADER.sh` to restart
2. **High position count warning** - 100 positions flagged as high

## Next Session Tasks

1. Investigate why Gateway keeps stopping
2. Consider adding daily position/buy limits as safety measure
3. Clean up duplicate positions from buying spree

## Commands

```bash
# Start everything
./START_TRADER.sh

# Check status
ps aux | grep -E "runner_async|app.py" | grep -v grep

# Check P&L
curl -s http://localhost:5555/api/pnl | python3 -m json.tool

# View logs
tail -f robo_trader.log
```
