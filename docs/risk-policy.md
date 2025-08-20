## Risk Policy

Robo Trader prioritizes capital preservation. All orders are validated against strict limits before execution.

### Controls
- **Daily Loss Cap**: Stop trading if daily PnL <= `-MAX_DAILY_LOSS`.
- **Per-Symbol Exposure Cap**: New order notional ≤ `MAX_SYMBOL_EXPOSURE_PCT * equity`.
- **Leverage Limit**: `(sum notionals) / equity` ≤ `MAX_LEVERAGE` after the new order.

### Position Sizing
Given `cash_available` and `entry_price`:
```
notional_per_position = cash_available * MAX_POSITION_RISK_PCT
shares = floor(notional_per_position / entry_price)
```
Shares are deterministic and non-negative.

### Boundary Behavior
- At exactly the symbol exposure cap: allowed.
- At exactly the leverage cap: allowed.
- Invalid inputs (≤0 price, ≤0 quantity): rejected.

### Examples
- Equity 100,000; `MAX_SYMBOL_EXPOSURE_PCT=0.2` → per-symbol max notional = 20,000.
- Existing notional 50,000; `MAX_LEVERAGE=2.0` → max total after order = 200,000; new order passes if total ≤ 200,000.

### Testing
Unit tests cover sizing, exposure, and leverage checks, including edge cases at the limits.


