## Strategies

Strategies must be pure and deterministic: inputs → outputs with no side effects.

### SMA Crossover
Function: `sma_crossover_signals(df, fast=10, slow=20)`
- Input columns: `close`
- Output columns: `sma_fast`, `sma_slow`, `signal`
- `signal`: 1 (golden cross), -1 (death cross), 0 otherwise
- Uses `min_periods=1` for early values; crossings computed with shifted series

### Guidelines
- Keep computations transparent and documented.
- Favor simple, explainable indicators.
- Add tests for flat series, noisy data, and early-window behavior.

### Integration
- The orchestrator consumes the last signal to propose orders.
- All orders still go through risk validation before any execution.


