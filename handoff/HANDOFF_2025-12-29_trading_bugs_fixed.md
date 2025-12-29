# Handoff: Trading Bugs Fixed

**Date:** 2025-12-29
**Status:** PARTIALLY FIXED
**Priority:** Critical

## Summary

Fixed two critical bugs that were preventing trades from executing. A third bug remains open for investigation.

## Bugs Fixed

### 1. Decimal/Float Type Mismatch
**File:** `robo_trader/runner_async.py` line 1656
**Error:** `unsupported operand type(s) for /: 'float' and 'decimal.Decimal'`

**Root Cause:** The `price` variable was a Decimal (from `PrecisePricing.to_decimal()`) but was passed to `calculate_position_size()` which performs float division.

**Fix:** Changed `current_price=price` to `current_price=price_float`:
```python
# Before
sizing_result = await self.advanced_risk.calculate_position_size(
    current_price=price,  # Decimal - causes error
    ...
)

# After
sizing_result = await self.advanced_risk.calculate_position_size(
    current_price=price_float,  # float - works correctly
    ...
)
```

### 2. Market Close Time Wrong
**File:** `robo_trader/market_hours.py` line 36
**Error:** Market showing as "open" after 4:00 PM

**Root Cause:** Market close was set to 4:30 PM instead of 4:00 PM.

**Fix:** Changed `time(16, 30)` to `time(16, 0)`:
```python
# Before
market_close = time(16, 30)  # 4:30 PM - wrong

# After
market_close = time(16, 0)   # 4:00 PM - correct
```

## Known Issue (NOT YET FIXED)

### 3. Int/Datetime Comparison Error
**Error:** `'>=' not supported between instances of 'int' and 'datetime.datetime'`
**Affects:** GM, GOLD, and possibly other symbols

**Symptoms:**
- AI opportunity signals generate successfully
- Kelly metrics calculate successfully
- Task then fails with datetime comparison error

**Investigation Notes:**
- Error occurs after Kelly metrics logging (line 1675)
- Likely in one of: correlation tracking, position validation, or stop-loss logic
- Needs deeper investigation into where an int is being compared to a datetime

## Files Modified

1. **`robo_trader/runner_async.py`** - Fixed Decimal/float mismatch
2. **`robo_trader/market_hours.py`** - Fixed market close time

## Testing

After fixes:
- Market correctly shows as closed after 4:00 PM
- No more Decimal/float errors in logs
- AAPL, NVDA, TSLA processing without errors
- GM/GOLD still failing with datetime error (open issue)

## Next Steps

1. Investigate the `int >= datetime` error for GM/GOLD
2. Fix the datetime comparison bug
3. Verify trades execute properly during market hours
