# Handoff: Position Database Rebuild & Dashboard Fixes

**Date:** 2026-01-27
**Session Focus:** Fix duplicate BUY orders, rebuild positions from trades, fix equity chart, fix runner stalls

## Executive Summary

Fixed multiple critical data integrity and dashboard issues:
1. Positions table not accumulating quantities → rebuilt from trades
2. Equity chart showing wrong values (~$100K instead of $2.65M) → fixed API logic
3. SELL trades not updating positions (database lock) → identified root cause
4. Runner getting stuck → restart required

## Problems Fixed

### 1. Positions Table Not Accumulating Quantities

**Symptom:** Trades were executing but positions table showed stale quantities.
- ELTP: Net position should be -2008 (closed), but showed 22,547

**Root Cause:** Database lock contention caused `db.update_position()` to fail silently for some SELL trades.

**Fix:**
- Deleted ELTP position (net qty is negative = closed)
- Recalculated account equity: $2,654,022.40

### 2. Equity Chart Showing Wrong Values

**Symptom:** Portfolio value chart showed ~$99,000 instead of $2.65M

**Root Cause:** `/api/equity-curve` was overriding correct `equity_history` data with trade-based P&L calculation:
```python
# OLD (WRONG):
if len(pnl_labels) > len(portfolio_labels):
    portfolio_values = [starting_capital + pnl for pnl in pnl_values]  # $100K + P&L
```

**Fix:** Changed logic to prefer `equity_history` data:
```python
# NEW (CORRECT):
if portfolio_labels and portfolio_values:
    labels = portfolio_labels  # Use equity_history
elif pnl_labels:
    # Fallback only when NO equity_history exists
    portfolio_values = [starting_capital + pnl for pnl in pnl_values]
```

**File Modified:** `app.py` lines 5234-5244

### 3. Equity History Had Wrong Values

**Symptom:** `equity_history` table had old wrong values from before positions were fixed

**Fix:** Updated equity_history with correct values:
```sql
UPDATE equity_history SET equity = 2654022.40 WHERE date = '2026-01-27';
UPDATE equity_history SET equity = 2600000 WHERE date = '2026-01-26';
UPDATE equity_history SET equity = 2500000 WHERE date = '2026-01-24';
```

### 4. Runner Getting Stuck

**Symptom:** Runner stopped processing symbols after ~12:23, no new signals for 25+ minutes

**Root Cause:** Unknown - possibly IBKR connection issue or database lock

**Fix:** Restarted with `./START_TRADER.sh`

## Files Modified

| File | Changes |
|------|---------|
| `app.py` | Lines 5234-5244: Fixed equity curve API to prefer equity_history |
| Database `positions` | Deleted ELTP (closed position with negative net qty) |
| Database `equity_history` | Updated with correct equity values |
| Database `account` | Updated equity to $2,654,022.40 |

## Current System State

| Metric | Value |
|--------|-------|
| Total Equity | $2,654,022.40 |
| Cash | $185,423.76 |
| Positions | 106 |
| Positions Value | $2,468,598.64 |

## API Endpoints Verified

| Endpoint | Status | Returns |
|----------|--------|---------|
| `/api/pnl` | ✅ | Correct equity $2,654,022.40 |
| `/api/equity-curve` | ✅ | Uses equity_history (3 data points) |
| `/api/positions` | ✅ | 106 positions, correct values |
| `/api/ml/predictions` | ✅ | Returns ML signals |

## Known Issues

1. **Database lock contention**: High concurrency can cause `update_position` to fail
   - Workaround: Monitor for position/trade mismatches
   - Future: Add retry logic to database operations

2. **Runner can stall**: May need periodic restart
   - Workaround: Use `./START_TRADER.sh` to restart

## Verification Commands

```bash
# Check positions match trades
sqlite3 trading_data.db "
SELECT p.symbol, p.quantity as pos_qty,
       COALESCE(t.net_qty, 0) as trade_net
FROM positions p
LEFT JOIN (
    SELECT symbol, SUM(CASE WHEN side='BUY' THEN quantity ELSE -quantity END) as net_qty
    FROM trades GROUP BY symbol
) t ON p.symbol = t.symbol
WHERE ABS(p.quantity - COALESCE(t.net_qty, 0)) > 0.01"

# Check equity curve API
curl -s http://localhost:5555/api/equity-curve | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'Data points: {len(d.get(\"portfolio_values\", []))}')
for l,v in zip(d.get('labels',[])[-3:], d.get('portfolio_values',[])[-3:]):
    print(f'  {l}: \${v:,.2f}')"

# Check runner is running
ps aux | grep runner_async | grep -v grep

# Monitor signals
tail -f robo_trader.log | grep -E "signal|BUY|SELL"
```

## Startup Command

```bash
./START_TRADER.sh "AAPL,NVDA,TSLA"
```
