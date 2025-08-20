# TODO — Robo Trader

Review and update this file at every session start/end and on every commit.

## Next Steps (immediate)
- [ ] Data: Normalize historical bars in `IBKRClient` (columns, NaN handling, enforce RTH/TRADES)
- [ ] Tests: Mock `ib_insync` to unit-test `fetch_recent_bars` shape/behavior
- [ ] PnL: Add portfolio snapshot with realized/unrealized PnL (optional CSV export)
- [ ] Risk: Add per-order and per-day notional caps with boundary tests
- [ ] Exec: Add slippage toggle in `PaperExecutor` (off by default) and tests
- [ ] Reliability: Add retry/backoff for connect/data
- [ ] CLI: Add `--sma-fast`, `--sma-slow`, `--default-cash`
- [ ] Docs: Update `risk-policy.md`, `configuration.md` for new knobs

## Short-term (this week)
- [ ] StrategyManager: combine multiple pure strategies (vote/priority)
- [ ] Correlation guard: limit overlapping exposures across strategies
- [ ] Eventing: in-process asyncio queue publisher/consumer

## Optional/Next
- [ ] FinBERT news overlay (flagged) with caching + fixtures
- [ ] Real-time streaming adapter with pacing guards
- [ ] LiveExecutor skeleton with dry-run and explicit gating
