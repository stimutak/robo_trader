# AI Trading Conviction Analysis
## Date: August 22, 2025

## Problem Summary
The AI trading system is not executing trades because conviction scores are consistently too low (20-35%) despite having aggressive risk settings. The system requires 50% conviction to trade (lowered from 75%), but the AI rarely generates signals above 35%.

## Root Causes Identified

### 1. Conservative AI Prompt (PRIMARY ISSUE)
**File:** `robo_trader/intelligence.py`, Line 55
```python
"Be conservative and data-driven. Only suggest trades with clear catalysts and asymmetric risk/reward. If uncertain, return neutral with conviction < 30."
```
This instruction explicitly tells Claude to:
- Be conservative (contradicts aggressive risk setting)
- Return <30% conviction when uncertain
- Only trade with "clear catalysts and asymmetric risk/reward"

### 2. Low Temperature Setting
**File:** `robo_trader/intelligence.py`, Line 143
```python
temperature=0.3,  # Lower temperature for consistency
```
Low temperature makes the AI more conservative and less likely to take decisive positions.

## Observed Behavior
- AI consistently returns "neutral" with 20-35% conviction
- Occasional 45% conviction (still below 50% threshold)
- Rare 65% conviction on major Fed news (would trigger trades)
- No actual trades executed during testing period

## Fixes Applied Today

### Successfully Fixed:
1. ✅ Removed non-working crypto symbols (BTC-USD, ETH-USD)
2. ✅ Lowered conviction threshold from 75% to 50%
3. ✅ Fixed SignalEvent creation with proper parameters
4. ✅ Fixed Order class parameter bug (action → side)
5. ✅ Fixed conviction scaling bug (was multiplying by 100)
6. ✅ Fixed news analysis 'reasoning' field error

### Still Needs Fixing:
1. ❌ AI prompt is too conservative for aggressive trading
2. ❌ Temperature setting may be too low
3. ❌ Company event analysis has attribute errors

## Recommended Changes

### For Aggressive Trading Mode:
```python
# Change Line 55 to:
"Given the aggressive risk profile, be decisive when you see opportunity. Aim for 60-80% conviction on clear directional moves. Only return neutral if truly ambiguous. Remember: in aggressive mode, we prefer action over inaction when reasonable edge exists."

# Change temperature to:
temperature=0.5,  # Balanced - allows more decisive positions
```

### For Moderate Trading Mode:
```python
# Current prompt but change last line to:
"Be balanced in your analysis. Suggest trades with reasonable catalysts. Return 40-60% conviction for typical setups, 60-80% for high-confidence trades."
```

### For Conservative Trading Mode:
Keep current settings.

## Alternative Solutions

1. **Dynamic Prompting Based on Risk Level**
   - Read risk level from config
   - Adjust prompt accordingly
   - Scale conviction thresholds with risk level

2. **Market Condition Awareness**
   - More aggressive in trending markets
   - More conservative in choppy/uncertain markets
   - Adjust based on VIX levels

3. **Learning from History**
   - Track which conviction levels led to winning trades
   - Adjust AI calibration based on actual results

## Testing Notes

### What Works:
- Signal generation pipeline (when conviction is high enough)
- Order creation and routing
- Event processing system
- News analysis (with fixes)

### What Doesn't Work:
- AI generating high enough conviction to trade
- Company event analysis (attribute errors)
- Options flow analysis (needs conviction >50%)

## Next Steps

1. **Immediate:** Adjust AI prompt for aggressive trading
2. **Short-term:** Fix company event analysis
3. **Medium-term:** Implement dynamic risk-based prompting
4. **Long-term:** Add learning/calibration system

## Configuration Context
- Trading Mode: Paper (safe for testing)
- Risk Level: Aggressive (per user)
- Current Threshold: 50% (lowered from 75%)
- Observed Conviction: 20-35% typical, 65% max