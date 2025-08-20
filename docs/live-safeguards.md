## Live Safeguards (Opt-in Only)

Paper mode is the default and recommended operation. Live mode requires explicit action and compliance with the following safeguards.

### Requirements
1. All tests pass in CI.
2. Risk checks identical to paper mode.
3. `TRADING_MODE=live` in environment.
4. Explicit runtime confirmation flag (e.g., `--confirm-live`).
5. Max notional per order/day configured and enforced.
6. Dry-run preview step before any session.
7. Clear rollback switch to stop trading immediately.

### Operator Checklist
- Verify environment values.
- Confirm account, margin, and symbols.
- Review throttling/pacing limits for IBKR.
- Run dry-run and inspect planned orders.
- Enable monitoring and alerts.

### Non-Goals
- No predictive guarantees; prefer explainable logic.
- No bypass of risk checks.


