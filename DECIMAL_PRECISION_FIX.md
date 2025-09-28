# Decimal Precision Fix for Financial Calculations

## Problem
Float arithmetic in financial calculations causes precision errors that can lead to:
- Order rejections due to incorrect prices/quantities
- Inaccurate P&L calculations
- Portfolio value discrepancies
- Rounding errors accumulating over many trades

## Example of the Bug
```python
# Problematic float arithmetic
price = 123.456789
quantity = 0.1
total = price * quantity  # Results in 12.345678900000001 (precision loss!)

# P&L calculation error
pnl = (exit_price - entry_price) * shares  # Accumulates precision errors
```

## Solution
Implemented Decimal arithmetic using Python's `decimal` module:

```python
from decimal import Decimal, ROUND_HALF_UP
from robo_trader.utils.pricing import PrecisePricing

# Precise calculation
price = Decimal('123.456789')
quantity = Decimal('0.1')
total = price * quantity  # Results in exactly 12.3456789

# Use utility functions
total = PrecisePricing.calculate_notional(shares, price)
pnl = PrecisePricing.calculate_pnl(entry_price, exit_price, shares)
```

## Files Updated

### Core Financial Classes
1. **`robo_trader/portfolio.py`**
   - `PositionSnapshot.avg_price`: `float` → `Decimal`
   - `Portfolio.cash`: `float` → `Decimal`
   - `Portfolio.realized_pnl`: `float` → `Decimal`
   - All arithmetic operations now use `PrecisePricing` utilities

2. **`robo_trader/risk_manager.py`**
   - `Position.notional_value()`: Returns `Decimal`
   - `Position.unrealized_pnl()`: Returns `Decimal`
   - Uses `PrecisePricing` for all calculations

### Utility Module (Already Existed)
3. **`robo_trader/utils/pricing.py`**
   - Comprehensive `PrecisePricing` class with decimal arithmetic
   - Price rounding, share calculation, P&L calculation
   - Order sizing and risk calculations

## Key Benefits

1. **Elimination of Precision Errors**
   ```python
   # Before: 12.345678900000001
   # After:  12.3456789 (exact)
   ```

2. **Accurate P&L Calculations**
   ```python
   # Before: 253.08650000000057
   # After:  253.086500 (exact)
   ```

3. **Proper Order Sizing**
   - No more order rejections due to invalid prices
   - Correct position sizes and notional values

4. **Consistent Financial Reporting**
   - Portfolio values sum correctly
   - P&L aggregation is precise

## Usage Examples

### Basic Operations
```python
from robo_trader.utils.pricing import PrecisePricing

# Calculate shares for dollar amount
shares = PrecisePricing.calculate_shares(capital=10000, price="123.45")

# Calculate exact notional value
notional = PrecisePricing.calculate_notional(shares=100, price="123.45")

# Calculate precise P&L
pnl = PrecisePricing.calculate_pnl(entry_price="100.00", exit_price="105.50", shares=100)

# Round price to valid tick size
rounded_price = PrecisePricing.round_price(price="123.456789", tick_size="0.01")
```

### Portfolio Operations
```python
from robo_trader.portfolio import Portfolio

portfolio = Portfolio(starting_cash=10000.0)

# All operations now use Decimal precision
portfolio.update_fill("AAPL", "BUY", 100, 123.456789)
portfolio.update_fill("AAPL", "SELL", 50, 125.987654)

# Returns Decimal values
unrealized_pnl = portfolio.compute_unrealized({"AAPL": 130.123456})
total_equity = portfolio.equity({"AAPL": 130.123456})
```

## Testing

Run the precision tests to verify the implementation:

```bash
python3 test_decimal_simple.py
```

Expected output:
```
🎉 ALL TESTS PASSED!
✅ Decimal precision implementation working
```

## Migration Notes

1. **Backward Compatibility**: Existing code continues to work as functions accept both float and Decimal inputs
2. **Performance**: Decimal arithmetic is slightly slower than float, but the precision is critical for financial calculations
3. **Database Storage**: Consider using `DECIMAL(15,6)` or similar precise types instead of `FLOAT` for price/quantity columns

## Next Steps

Consider updating additional files that may have float arithmetic:
- `app.py` (dashboard calculations)
- `robo_trader/runner_async.py` (trading logic)
- Backtesting modules
- Database models for price storage

## Summary

This fix addresses a critical precision issue that could cause:
- ❌ Order rejections
- ❌ Incorrect position sizes
- ❌ P&L calculation errors
- ❌ Portfolio value inconsistencies

All core financial calculations now use proper Decimal arithmetic for exact results.