## Live Trading Status: Disabled

Live order placement is intentionally unavailable during the remediation
program defined in `docs/ROBOTRADER_REMEDIATION_PLAN_2026-07-20.md`.

Current enforced boundary:

1. `EXECUTION_MODE=paper` is canonical.
2. The temporary `TRADING_MODE` alias must also be `paper`.
3. Only IBKR paper ports 4002 (Gateway) and 7497 (TWS) are accepted.
4. `IBKR_READONLY=true` is required.
5. IBC must contain `ReadOnlyApi=yes`.
6. `LiveExecutor` cannot be instantiated.
7. `--confirm-live` is retained only as a compatibility flag and is rejected.
8. The dashboard cannot start, stop, or restart the trading process.
9. `START_TRADER.sh` is the only authorized full-system launcher.

The former opt-in checklist was not a complete live-order lifecycle and must not
be restored. Live capability will be designed in PRs 11 and 12 of the
remediation plan, then qualified through failure testing and a paper soak before
the separately approved limited canary in PR 15.
