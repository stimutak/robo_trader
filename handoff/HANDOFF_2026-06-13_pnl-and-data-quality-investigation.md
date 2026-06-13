# Handoff — P&L / Data-Quality Investigation (3 root-cause bugs)

**Created:** 2026-06-13
**Priority:** HIGH (real money-affecting logic; system is live in paper mode)
**Branch:** investigation done on `main`; **no code changed, no DB touched**
**Status:** DIAGNOSIS ONLY — awaiting coordination before any fix is implemented

---

## Why this handoff exists

User asked "are the recommendations accurate?" and "why have we lost money?".
A read-only investigation (3 parallel agents) found **three independent root
causes**. This doc exists to **coordinate with the parallel
`fix/strategy-and-test-bugs` work** before anyone edits core trading code, so we
don't collide. All findings are cited to `file:line`. Nothing here is
implemented yet.

> Numbers below were captured live (~149 trades) and drift as the trader runs.
> The *diagnosis* is stable; treat exact dollar figures as of 2026-06-13.

---

## TL;DR — the headline you need to act on

1. **The dashboard P&L is wrong and overstates profit.** The ledger shows
   ~**+$224** realized; correct running-cost accounting shows **−$215.12**
   (verified to the cent via an economic-identity check). **The system is
   actually down, not up.**
2. **The "recommendations" carry no signal.** Recorded signal `strength` is a
   **hardcoded 0.6** — the ML model's confidence never reaches the recorded
   signal. The 2,900+ signals/week are mostly a fallback path re-logging
   blocked signals every cycle.
3. **The −$419 TSLA trade was bad data, not a strategy decision.** A corrupt
   `$214.85` tick (TSLA was ~$424) fired the stop-loss because **the
   deviation-checking validator is dead code** and the on-path validator only
   checks absolute bounds.

---

## Root cause #1 — corrupt tick + unguarded stop-loss (the −$419 event)

`2026-06-04 10:54:27  TSLA SELL 2 @ $214.85  pnl −$410.93`

- The only price-deviation check in the repo lives in
  `robo_trader/data_validator.py:311-334` (`_check_anomalies`, 20% max move).
  **This class is instantiated/called nowhere in the running system — dead
  code.** (A second unused `DataValidator` also exists at
  `robo_trader/data/validation.py:205`.)
- The validator actually on the hot path,
  `robo_trader/database_validator.py:150-193` (`validate_price`, called by
  `stop_loss_monitor.py:271`), only enforces absolute bounds
  ($0.0001–$1,000,000). `$214.85` passes.
- Path: `runner_async.py:1954` pushes the raw close into
  `stop_loss_monitor.update_price` → `check_stops` (`stop_loss_monitor.py:334-387`,
  1s loop) → triggers → `execute_stop_loss` fills at `trigger_price = 214.85`
  (`stop_loss_monitor.py:418`). The stop-loss path is **exempt** from the
  "don't sell at a loss" guard, so it liquidated 2 real shares for $429.70
  instead of ~$848.
- **Likely source = cross-symbol contamination, not a random tick.** The
  subprocess IBKR client uses a **shared response queue with no request-ID /
  symbol correlation** (`subprocess_ibkr_client.py:128`, FIFO pop at
  `:499`); `get_historical_bars` returns bars without checking the symbol
  (`:749-762`), and the worker response carries no symbol field
  (`ibkr_subprocess_worker.py:501`). A late NVDA response (~$214, exactly
  NVDA's band that day) could be popped by a TSLA request.

**Proposed fix (described, not implemented):**
- **A.** Max-deviation guard in `stop_loss_monitor.py` right after line 274:
  reject a tick if `abs(price - last)/last > MAX_TICK_DEVIATION_PCT` (default
  0.20, env-configurable); don't update `last_prices`, log a warning, return.
  This neutralizes BOTH a bad tick and a contaminated price.
- **B.** Echo the symbol in the worker response and assert it client-side in
  `subprocess_ibkr_client.get_historical_bars` — closes the contamination hole
  at the source.
- **C (lower priority).** Wire `DataValidator.validate_dataframe()` into
  `runner_async.fetch_and_store_data` (~line 1810) so corrupt bars never reach
  `latest_prices` or the DB.

---

## Root cause #2 — signal flood + whipsaw + meaningless strength

- **Strength is hardcoded 0.6**, not ML confidence: `runner_async.py:2203`
  (SMA fallback) and `ml_enhanced_strategy.py:406/409` (MTF helper). DB confirms
  every ML_ENHANCED row is exactly 0.6. The model path
  (`ml_enhanced_strategy.py:295`, `np.max(probabilities)`) is being bypassed —
  likely the `len(data) < 200` gate (`ml_enhanced_strategy.py:175`) on intraday
  bars forces the fallback.
- **The flood:** `record_signal` is called every 15s cycle at
  `runner_async.py:2349-2360`, **before** any execution guard. Guards block the
  *trade* but not the *recording*. AAPL is held at a loss → emits SELL every
  cycle → blocked by the profit guard (`runner_async.py:2757`) → re-logged
  forever (≈2,324 rows/week). TSLA mirror: held position re-emits BUY, blocked
  by the "already have position" guard, re-logged.
- **The whipsaw:** no hysteresis/dwell between opposite signals; the SELL
  execution block (`runner_async.py:2731-2767`) has **no cooldown** (only BUY
  does, at `:2507/:2523`).

**Proposed fix (described, not implemented), prioritized:**
- **P1.** Don't re-record an unchanged/blocked signal: keep
  `self._last_recorded_signal[symbol]` and only write on direction/strength-band
  change (`runner_async.py:~2349`). Collapses thousands of rows → tens, kills
  the dashboard whipsaw, zero trade-logic risk.
- **P2.** Make `strength` the real ML confidence; fix why `analyze()` falls
  through to SMA (the 200-bar gate). Until then every confidence threshold is
  operating on a constant.
- **P3.** Add a symmetric opposite-signal cooldown to the SELL path / centralize
  the `has_recent_*` guards for both directions.
- **P4.** When a position is held at a loss and the stop-loss owns the exit,
  emit HOLD upstream instead of a guaranteed-blocked SELL.

---

## Root cause #3 — two incompatible P&L accounting methods

- **Ledger `trades.pnl`** is computed by `_calculate_fifo_pnl`
  (`database_async.py:478-516`) which, despite the name, is **not FIFO**: it
  averages over *all buy rows ever* without consuming sold shares. On a system
  that round-trips AAPL/TSLA/NVDA 20+ times at rising prices, this
  **systematically overstates profit** (~+$584 error on AAPL alone). Written to
  `trades` at `database_async.py:556-563`.
- **Portfolio `realized_pnl`** (`portfolio.py:82-84`) uses correct running
  weighted-average cost. This is the right method.
- **`account.realized_pnl`** is a *third*, path-dependent value: restored from
  the `account` snapshot on restart (`runner_async.py:1522-1533`), incremented
  in-memory, written back (`runner_async.py:3060-3113`). It matches neither.
- **True realized P&L = −$215.12** (running-cost recompute; economic identity:
  ledger-implied equity − $100k start − unrealized = −$215.12 exactly).
- **~$430 cash desync** = the corrupt TSLA sell notional exactly → the
  documented **TC-M8** non-transactional update window
  (`runner_async.py:867-877`): position/cash/DB writes aren't atomic, so a
  restart mid-fill drifts `portfolio.cash` from the ledger.

**Proposed fix (described, not implemented) — NEVER deletes data:**
- Replace `_calculate_fifo_pnl` with true running-average (or real FIFO
  lot-matching) so ledger == portfolio going forward; add a regression test
  (round-trip at rising prices → ledger == portfolio).
- **Append, don't overwrite:** add a `pnl_v2` column or a reconciliation
  view/table; leave original `pnl` intact for audit.
- Read-only reconciliation report script (ledger vs portfolio vs account).
- Flag (don't delete) the corrupt `2026-06-04 10:54 TSLA` row (`corrupt`
  boolean); user decides whether to back out the ~$419.
- Make `_update_position_atomic` + DB write transactional (or reconcile cash
  from the ledger on restart).

---

## ⚠️ Collision map (coordinate before editing)

| Fix | Files touched | Overlap w/ `fix/strategy-and-test-bugs`? |
|---|---|---|
| #1A deviation guard | `stop_loss_monitor.py` | Low — isolated |
| #1B subprocess assert | `subprocess_ibkr_client.py`, `ibkr_subprocess_worker.py` | Low |
| #2 signal flood/whipsaw | `runner_async.py`, `ml_enhanced_strategy.py` | **Check** — strategy logic |
| #3 accounting | `database_async.py`, `portfolio.py`, `runner_async.py` | **Check** — `runner_async.py` is shared |

The parallel branch was last seen editing `robo_trader/strategies/pairs_trading.py`
and `tests/test_mean_reversion_strategies.py`. Pairs trading is a separate path
(`runner_async.py:3269+`) from the ML_ENHANCED symptom, but **`runner_async.py`
is shared** — whoever edits it should claim it explicitly to avoid clobbering.

## Suggested division of labor

- **This session / next:** Fix #1A (deviation guard) is the highest-value,
  lowest-risk, fully-isolated change — do it on a fresh worktree off `main`.
- **Whoever owns strategy logic** (`fix/strategy-and-test-bugs`): take #2
  (signal flood/whipsaw) since it's strategy-adjacent.
- **Accounting (#3):** sequence *after* the strategy branch lands to avoid
  `runner_async.py` conflicts; start with the read-only reconciliation report.

## Hard constraints
- Trader is **live** (paper, runner PID was 87266) — don't restart without the user.
- **Never delete trading data** (CLAUDE.md rule #1). All #3 fixes are append-only.
- No edits were made during this investigation.

## Evidence / reproduce (read-only)
- `SELECT SUM(pnl), COUNT(*) FROM trades;`
- `SELECT realized_pnl, cash, equity FROM account;`
- `SELECT signal_type, COUNT(*) FROM signals WHERE timestamp>'2026-06-03' GROUP BY signal_type;`
- Per-symbol chronological replay (Python) reproduces stored `pnl` exactly under
  the buggy method; running-cost gives −$215.12.
