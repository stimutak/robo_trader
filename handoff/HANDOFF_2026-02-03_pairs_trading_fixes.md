# Handoff: Pairs Trading Fixes - 2026-02-03

## Summary

Fixed critical bugs in pairs trading that caused:
1. **Duplicate BUY trades** - 138 BUY trades totaling $1.37M across 39 symbols in one day
2. **Portfolio equity miscalculation** - Pairs trades not synced to Portfolio
3. **Missing stop-loss protection** - Pairs positions had no downside protection
4. **N+1 database query pattern** - DB queried inside loop instead of once before

## Root Cause Analysis

### Duplicate BUY Trades
The pairs trading code was:
- Recording trades to `trades` table ✅
- NOT updating `positions` table ❌
- NOT updating `self.positions` in-memory ❌

Each trading cycle created a fresh `AsyncRunner` that loaded positions from DB. Since pairs trade positions weren't persisted, each cycle saw "no position exists" and bought again.

### Missing Portfolio Sync
The pairs trading code bypassed the normal order flow that goes through `_update_position_atomic()`. This method calls:
- `await self.portfolio.update_fill(symbol, side, quantity, price)`

Without this call, `portfolio.equity()` was incorrect because it uses its own internal positions dict.

### Missing Stop-Loss
Regular BUY orders create stop-loss entries via `stop_loss_monitor.add_stop_loss()`. Pairs trades didn't, leaving those positions with no downside protection.

## Changes Made

### File: `robo_trader/runner_async.py`

#### 1. Fix N+1 Query Pattern (lines ~2563-2566)
```python
# BEFORE: db.get_positions() called inside loop
for signal in pairs_signals:
    ...
    db_positions = await self.db.get_positions()  # Called N times!

# AFTER: Fetched once before loop
db_positions_list = await self.db.get_positions()  # Called 1 time
for signal in pairs_signals:
    ...
    # Uses db_positions_list instead
```

#### 2. Fix Position Persistence (lines ~2706-2745, 2795-2834)
Added after successful pairs BUY orders:
```python
# Sync to Portfolio for equity calculation
await self.portfolio.update_fill(symbol_a, "BUY", qty_a, fill_a)

# Track trade count
if hasattr(self, "trades_executed"):
    self.trades_executed += 1

# Add stop-loss for pairs position
if self.stop_loss_monitor and self.enable_stop_loss:
    try:
        new_pos = Position(
            symbol=symbol_a,
            quantity=qty_a,
            avg_price=fill_a,
            entry_time=datetime.now(),
        )
        await self.stop_loss_monitor.add_stop_loss(
            symbol=symbol_a,
            position=new_pos,
            stop_percent=self.stop_loss_percent,
            stop_type=StopType.FIXED,
        )
        logger.info(f"Stop-loss added for pairs position {symbol_a}")
    except Exception as e:
        logger.error(f"Failed to add stop-loss for pairs {symbol_a}: {e}")
```

#### 3. Earlier Fix (from earlier session): Position DB Updates
Already fixed earlier in this session:
- `self.positions[symbol] = Position(...)`
- `await self.db.update_position(...)`
- Increased `has_recent_buy_trade` timeout from 120s to 600s

### File: `CLAUDE.md`

Added new entries to Common Mistakes table:
- Pairs trading missing `portfolio.update_fill()`
- Pairs trading missing stop-loss creation
- `db.get_positions()` inside pairs loop (N+1)

## Code Review Findings (All Fixed)

| Priority | Issue | Status |
|----------|-------|--------|
| CRITICAL | Missing Portfolio sync after pairs trades | ✅ Fixed |
| CRITICAL | Missing `portfolio.update_fill()` for pairs | ✅ Fixed |
| HIGH | Missing stop-loss for pairs BUY orders | ✅ Fixed |
| HIGH | Missing `trades_executed` increment | ✅ Fixed |
| MEDIUM | N+1 query pattern in pairs loop | ✅ Fixed |

## Testing

1. Syntax verification: `python3 -m py_compile robo_trader/runner_async.py` ✅
2. Runner started successfully with `./START_TRADER.sh`
3. Monitor logs for:
   - "Stop-loss added for pairs position" messages
   - No duplicate BUY warnings
   - Correct equity calculations

## Impact

- **Duplicate BUY Prevention**: Positions now persisted correctly to both memory and DB
- **Portfolio Accuracy**: `portfolio.equity()` now includes pairs trade positions
- **Risk Management**: Pairs positions now have stop-loss protection
- **Performance**: Single DB query per cycle instead of N queries

## Files Modified

1. `robo_trader/runner_async.py` - Pairs trading fixes
2. `CLAUDE.md` - Updated common mistakes table

## Verification Commands

```bash
# Check runner is running
pgrep -f runner_async

# Check for duplicate BUYs (should show 0 or 1 per symbol)
sqlite3 trading_data.db "SELECT symbol, COUNT(*) FROM trades WHERE side='BUY' AND date(timestamp) = date('now') GROUP BY symbol HAVING COUNT(*) > 1;"

# Check positions table is being updated
sqlite3 trading_data.db "SELECT symbol, quantity FROM positions WHERE quantity > 0 ORDER BY symbol;"

# Monitor for stop-loss creation
tail -f robo_trader.log | grep -i "stop-loss"
```

## Next Steps

1. Monitor the system for 1-2 hours to verify no new duplicate BUYs
2. Verify stop-loss orders are being created for pairs trades
3. Check dashboard equity calculation is accurate
4. Consider running `/verify-trading` to validate system health
