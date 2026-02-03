# Handoff: Trailing Stops Implementation

**Date:** 2026-02-03
**Topic:** Implemented trailing stop system to let winners run

---

## Problem Identified

System P&L analysis revealed:
- 469 total sells, 50.5% win rate
- Avg win: $444
- Avg loss: $718 (60% larger than wins!)
- Total P&L: -$60,607

**Root cause:** Fixed stop-losses cut winners early while losses ran larger.

---

## Solution Implemented

**Trailing stops** that follow price UP and lock in profits:

1. Initial stop at 5% below entry price
2. As price rises, stop rises with it (ratchets up)
3. Stop never moves down - only up
4. When price pulls back 5% from high → SELL triggered with profit locked in

### Example

```
Buy AAPL at $200 → initial stop at $190 (5% below)
Price rises to $250 → stop moves to $237.50 (5% below $250)
Price drops to $237.50 → SELL triggered, profit of $37.50/share locked in!
```

---

## Files Modified

| File | Changes |
|------|---------|
| `.env` | Added `USE_TRAILING_STOP=true`, `TRAILING_STOP_PERCENT=5.0` |
| `robo_trader/config.py` | Added `use_trailing_stop` and `trailing_stop_pct` fields to RiskConfig |
| `robo_trader/runner_async.py` | Updated all 6 stop-loss creation points to use trailing when enabled |
| `CLAUDE.md` | Documented trailing stop configuration with examples |
| `README.md` | Added trailing stops to Risk Management section and config examples |
| `STARTUP_GUIDE.md` | Added Key Risk Settings section with trailing stop config |

---

## Configuration

```bash
# Enable trailing stops (RECOMMENDED)
USE_TRAILING_STOP=true
TRAILING_STOP_PERCENT=5.0       # 5% trail

# Disable trailing (use fixed stops)
USE_TRAILING_STOP=false
STOP_LOSS_PERCENT=2.0           # Fixed 2%
```

---

## Verification

After restart, log shows:
```
TRAILING STOP enabled at 5.0% - stops follow price up, lock in profits!
Trailing stop created for existing position CVX at 5.0% (entry=$167.70, initial stop=$159.32)
Trailing stop created for existing position CYBR at 5.0% (entry=$446.20, initial stop=$423.89)
...
```

---

## Notes

- The trailing stop infrastructure already existed in `stop_loss_monitor.py`
- Only needed to enable it by passing `StopType.TRAILING_PERCENT` instead of `StopType.FIXED`
- All existing positions get trailing stops on restart
- High water mark tracking is automatic via `_update_trailing_stop()` method

---

## Future Improvements (Optional)

1. **Activation threshold** - Only start trailing after X% profit (e.g., don't trail until up 3%)
2. **ATR-based trailing** - Use Average True Range for volatility-adjusted trails
3. **Dashboard display** - Show current stop price and high water mark for each position
