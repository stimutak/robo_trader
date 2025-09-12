# TODO — Robo Trader

Review and update this file at every session start/end and on every commit.

## 🚨 CRITICAL: PRODUCTION READINESS (MUST COMPLETE FIRST) 🚨
**⚠️ SYSTEM NOT READY FOR LIVE TRADING - Score: 5/10**  
**See [PRODUCTION_READINESS_PLAN.md](PRODUCTION_READINESS_PLAN.md) for mandatory action items**

### Phase 0: IMMEDIATE BLOCKERS (Complete within 3 days)
- [ ] TASK 0.1: Remove ALL hardcoded connection parameters (config.py:370-374)
- [ ] TASK 0.2: Add SQL input validation layer (create database_validator.py)
- [ ] TASK 0.3: Fix ALL exception handling (no bare except, no silent failures)
- [ ] TASK 0.4: Disable debug mode in production (test_dashboard_simple.py:117)

### Phase 1: CRITICAL SAFETY (Complete within 1 week)
- [ ] TASK 1.1: Implement active stop-loss monitoring (create stop_loss_monitor.py)
- [ ] TASK 1.2: Integrate kill switch at ALL entry points
- [ ] TASK 1.3: Fix position update race conditions (atomic transactions)

**Run `python scripts/safety_check.py` after each task to verify fixes**

---

## Next Steps (AFTER production readiness)
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

## Future Development (AFTER production readiness complete)
- [ ] FinBERT news overlay (flagged) with caching + fixtures
- [ ] Real-time streaming adapter with pacing guards
- [ ] LiveExecutor skeleton with dry-run and explicit gating
