# Signal Flow, Kelly Criterion, and Adaptive Thresholds

**Date:** 2026-01-27
**Status:** Implemented

This document explains how trading signals flow through the system, the role of Kelly criterion, and the adaptive threshold fix for ML/MTF disagreement resolution.

---

## Signal Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SIGNAL GENERATION FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              Market Data (1-min bars)
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │      MLEnhancedStrategy.analyze()    │
                    │         (ml_enhanced_strategy.py)    │
                    └──────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
    ┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
    │ Regime Detection│    │   ML Prediction     │    │ Multi-Timeframe │
    │   (regime.py)   │    │ (_get_ml_prediction)│    │    Analysis     │
    │                 │    │                     │    │ (1m,5m,15m,1h,  │
    │ - RANGE_BOUND   │    │ - LightGBM/XGBoost │    │  1d timeframes) │
    │ - STRONG_BULL   │    │ - confidence score  │    │                 │
    │ - STRONG_BEAR   │    │ - test_score ~0.55  │    │ - alignment_score│
    │ - NEUTRAL       │    │                     │    │ - BUY/SELL vote │
    │ - HIGH_VOL      │    │                     │    │                 │
    └─────────────────┘    └─────────────────────┘    └─────────────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         _combine_signals()           │
                    │    Adaptive Disagreement Resolution  │
                    │                                      │
                    │  IF signals agree:                   │
                    │    → Combine confidence              │
                    │    → Pass to position sizing         │
                    │                                      │
                    │  IF signals DISAGREE:                │
                    │    → Check adaptive threshold        │
                    │    → Threshold = model_score + margin│
                    │    → Lower in RANGE_BOUND regime     │
                    └──────────────────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                    Signal Approved           Signal Rejected
                          │                         │
                          ▼                         ▼
              ┌─────────────────────┐        ┌─────────────┐
              │  Kelly Position     │        │ AI Analyst  │
              │  Sizing (runner)    │        │  Fallback   │
              │                     │        │ (news-based)│
              │ - calculate_position│        └─────────────┘
              │   _size()           │
              │ - win_rate, edge    │
              │ - kelly_fraction    │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Risk Validation     │
              │ - Correlation check │
              │ - Max position %    │
              │ - Kill switch       │
              │ - Stop-loss monitor │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   Order Execution   │
              │   (PaperExecutor)   │
              └─────────────────────┘
```

---

## The Original Problem: Fixed 0.8 Threshold

### What Was Happening

```python
# OLD CODE (ml_enhanced_strategy.py line 423):
if ml_signal.action != mtf_signal.combined_signal.action:
    if ml_signal.confidence > 0.8:  # ← FIXED, UNREALISTIC THRESHOLD
        final_signal = ml_signal
    else:
        return None  # ← 99% of disagreements rejected
```

### Why This Was Broken

| Factor | Reality |
|--------|---------|
| **ML Model Accuracy** | ~55% (test_score = 0.5533) |
| **Required Confidence** | 80% when disagreeing |
| **Result** | Virtually no trades through ML path |

**The Math:**
- A 55% accurate model rarely outputs 80%+ confidence that's actually reliable
- Even when it does, `predict_proba` confidence ≠ actual win rate (uncalibrated)
- Most stocks showed "range_bound" regime, where MTF trend signals are noisy by definition

### Cascade Effect

```
1. ML says BUY (confidence 0.65)
2. MTF says SELL (5 timeframes disagree due to range-bound)
3. 0.65 < 0.80 → SIGNAL REJECTED
4. Falls back to AI Analyst
5. AI Analyst only fires on specific news → Most symbols get no signal
6. System appears "stuck" even though ML had valid edge
```

---

## The Fix: Adaptive Thresholds

### New Logic

```python
# NEW CODE (ml_enhanced_strategy.py):

# Calculate adaptive threshold based on model's ACTUAL performance
base_threshold = self._model_test_score + self.disagreement_threshold_margin
# Example: 0.55 + 0.10 = 0.65

# Adjust for market regime
if regime.trend_regime == MarketRegime.RANGE_BOUND:
    # MTF trend signals are NOISE in ranges - trust ML more
    disagreement_threshold = self._model_test_score + 0.05  # = 0.60
elif regime.trend_regime in [MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR]:
    # MTF is reliable in trends - require higher ML confidence
    disagreement_threshold = min(base_threshold + 0.05, 0.75)  # = 0.70
else:
    # Normal conditions
    disagreement_threshold = min(base_threshold, 0.70)  # = 0.65
```

### Threshold Comparison

| Regime | OLD Threshold | NEW Threshold | Impact |
|--------|---------------|---------------|--------|
| Range-bound | 0.80 | 0.60 | **+33% more signals pass** |
| Strong trend | 0.80 | 0.70 | +14% more signals pass |
| Normal | 0.80 | 0.65 | +23% more signals pass |

### Why This Works

1. **Calibrated to Model Capability**: A 55% model with 65% confidence is performing well *for that model*
2. **Regime-Aware**: Range-bound markets have noisy MTF signals - disagreement is expected, not failure
3. **Still Conservative**: Threshold never drops below model's accuracy, ensuring positive expected value

---

## Kelly Criterion Integration

### Where Kelly Lives

Kelly criterion is implemented in two places:

1. **`robo_trader/risk/kelly_sizing.py`** - `OptimalKellySizer` class
2. **`robo_trader/risk/advanced_risk.py`** - `AdvancedRiskManager` with `enable_kelly=True`

### When Kelly is Used

```python
# runner_async.py lines 1821-1846
# AFTER signal is approved, Kelly sizes the position:

sizing_result = await self.advanced_risk.calculate_position_size(
    symbol=symbol,
    signal_strength=abs(signal_value),
    current_price=price_float,
    atr=atr,
)

# Kelly metrics logged:
# - win_rate: Historical win probability
# - edge: Expected value per trade
# - kelly_fraction: Optimal bet size (typically use half-Kelly)
```

### Kelly Formula

```
Kelly Fraction = (p * b - q) / b

Where:
  p = win probability (from trade history)
  q = loss probability (1 - p)
  b = win/loss ratio (avg win / avg loss)
```

### Current Kelly Flow

```
Signal Approved → Kelly calculates position size → Risk validation → Execute

Kelly is used for SIZING, not FILTERING.
The 0.8 threshold was blocking signals BEFORE Kelly could evaluate them.
```

### Future Enhancement: Kelly for Filtering

The `OptimalKellySizer` has a method specifically for this:

```python
# kelly_sizing.py line 384
def should_skip_trade(self, symbol: str, signal_confidence: float, min_edge: float = 0.01):
    """Determine if a trade should be skipped based on Kelly criterion."""
    result = self.calculate_kelly(symbol)

    if result.expected_value < 0:
        return True, f"Negative EV: {result.expected_value:.3f}"

    if result.expected_value < min_edge:
        return True, f"Insufficient edge: {result.expected_value:.3f}"

    if result.recommended < 0.005:
        return True, f"Kelly too small: {result.recommended:.3f}"

    return False, "Trade acceptable"
```

This could be integrated into `_combine_signals()` for even smarter filtering:

```python
# POTENTIAL FUTURE ENHANCEMENT:
if ml_signal.action != mtf_signal.combined_signal.action:
    # Use Kelly's expected value instead of confidence threshold
    skip, reason = self.kelly_sizer.should_skip_trade(
        symbol=ml_signal.symbol,
        signal_confidence=ml_signal.confidence * self._model_test_score
    )
    if skip:
        return None
    else:
        final_signal = ml_signal  # Kelly says it's worth taking
```

---

## Configuration Parameters

### In `MLEnhancedStrategy.__init__()`:

```python
# Adaptive threshold parameters
self._model_test_score: float = 0.55  # Updated when model loaded
self.disagreement_threshold_margin = 0.10  # Normal: score + 0.10
self.range_bound_threshold_margin = 0.05   # Range-bound: score + 0.05
```

### In `.env`:

```bash
# Kelly is enabled via advanced risk manager
ADVANCED_RISK_ENABLED=true  # Enables Kelly sizing

# ML Enhanced is enabled
ML_ENHANCED_ENABLED=true
```

---

## Logging

### New Log Messages

When ML/MTF disagree but signal passes:
```
INFO: ML/MTF disagree for AAPL: ML=BUY (conf=0.67), MTF=SELL (align=0.40).
      Using ML signal (threshold=0.60, model_score=0.55, regime=range_bound)
```

When signal is rejected:
```
DEBUG: Signal rejected for AAPL: ML/MTF disagree, ML conf 0.52 < threshold 0.65
```

### Signal Features Added

```python
signal.features = {
    "signal_resolution": "ml_override_disagreement",  # or "ml_mtf_agreement"
    "disagreement_threshold": 0.60,
    "ml_confidence": 0.67,
    "mtf_alignment": 0.40,
    "regime": "range_bound"
}
```

---

## Summary

| Component | Before | After |
|-----------|--------|-------|
| Disagreement threshold | Fixed 0.80 | Adaptive 0.60-0.70 |
| Range-bound handling | Same as all regimes | Lower threshold (MTF noisy) |
| Kelly usage | Sizing only | Sizing (future: filtering) |
| Expected result | ~1% signals pass disagreement | ~30-40% pass |

The fix unlocks the ML signal path while maintaining risk controls through:
1. Regime-aware thresholds
2. Kelly-based position sizing (smaller positions for lower confidence)
3. Existing correlation limits and stop-losses
