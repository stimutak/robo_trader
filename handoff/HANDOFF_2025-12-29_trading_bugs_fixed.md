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

### 3. ML Strategy Missing Await
**File:** `robo_trader/strategies/ml_enhanced_strategy.py` line 255
**Error:** `select_best_model` was called without `await` and with wrong parameters

**Root Cause:** The async method was called synchronously with incorrect keyword arguments.

**Fix:** Added `await` and corrected the parameters:
```python
# Before (wrong)
best_model = self.model_selector.select_best_model(
    features=latest_features,  # Wrong param
    market_conditions={...},   # Wrong param
)

# After (correct)
best_model = await self.model_selector.select_best_model(
    selection_criteria="test_score",
    require_recent=False,
    max_age_days=30,
)
```

### 4. Market Hours Inconsistencies
**File:** `robo_trader/market_hours.py` lines 73, 108
**Error:** `get_market_session()` and `is_extended_hours()` still used 4:30 PM

**Fix:** Changed `time(16, 30)` to `time(16, 0)` in both functions for consistency.

## Known Issue (Under Investigation)

### Int/Datetime Comparison Error
**Error:** `'>=' not supported between instances of 'int' and 'datetime.datetime'`
**Affects:** GM, GOLD, and possibly other AI-discovered symbols

**Symptoms:**
- AI opportunity signals generate successfully
- Symbols are added to processing queue
- Task fails during process_symbol with datetime comparison error

**Investigation Progress:**
- Added detailed traceback logging to identify exact location
- Error likely in ML/feature pipeline processing
- May be related to how AI-discovered symbols get different data format

**Traceback logging added to:**
- `runner_async.py` line 2031-2033 - Now logs full exception traceback

## Files Modified

1. **`robo_trader/runner_async.py`** - Fixed Decimal/float mismatch, added traceback logging
2. **`robo_trader/market_hours.py`** - Fixed market close time (all functions)
3. **`robo_trader/strategies/ml_enhanced_strategy.py`** - Fixed async model selection call

## Testing

After fixes:
- Market correctly shows as closed after 4:00 PM
- No more Decimal/float errors in logs
- AAPL, NVDA, TSLA processing without errors
- ML model selection now properly awaited
- GM/GOLD will now log full tracebacks to identify exact error location

## Next Steps

1. Run the system during market hours tomorrow morning
2. Check logs for full traceback on GM/GOLD errors
3. Fix the datetime comparison bug based on traceback
4. Verify trades execute properly
