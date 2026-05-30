# Risk tab — backend follow-up (data plumbing)

The **Risk tab UI** (this PR) is complete and correct: it renders, is wired to
the existing `/api/risk/status`, `/api/safety/circuit-breakers`,
`/api/safety/thresholds`, and `/api/safety/data-validator` endpoints, escapes
all DB/env-derived fields, and respects the W-H3 XSS hardening.

However, **three of its four panels currently display zeros** — and they show
the *same* zeros on the live production dashboard (`:5555`), because the
underlying endpoints already returned empty data on `main` before this UI
existed. The merge made a pre-existing backend gap *visible*; it did not
introduce it.

This file tracks the backend work needed to make those panels show live values.
None of it is required for the Risk tab UI to be correct — it is follow-up.

## Verified on 2026-05-30

Identical responses from preview (worktree app.py) and live `:5555`:

| Endpoint | Live `:5555` response | Root cause |
|---|---|---|
| `/api/safety/circuit-breakers` | `{"breakers": {}, "open_count": 0}` | reads in-process singleton (see #1) |
| `/api/safety/data-validator` | `{"total_validations": 0, ...}` | `app.data_validator` unset (see #2) |
| `/api/risk/status` → `risk_metrics` | `leverage: 0, total_exposure: 0` | positions `current_price` is 0/null (see #3) |
| `/api/risk/status` → `kelly_sizing` | **real data** (AAPL 0.2174, etc.) | DB-backed — works cross-process ✅ |

## The cross-process problem (the common thread)

The **runner** (`runner_async.py`, PID A) and the **dashboard** (`app.py`,
PID B) are separate OS processes. Anything stored in a module-level singleton
or on the `app` object inside the runner is invisible to the dashboard. Only
data persisted to the **database** (or a shared state file) crosses the
boundary. That is why Kelly (DB-backed) works while breakers/validator
(memory-backed) do not.

---

## #1 — Circuit breaker stats are process-local

**Symptom:** "none registered" / `breakers: {}` on the Risk tab.

**Cause:** `app.py:get_circuit_breakers()` does
`from robo_trader.circuit_breaker import circuit_manager` and reads
`circuit_manager.get_all_statistics()`. `circuit_manager` is a module-level
singleton; in the dashboard process it is freshly constructed and empty. The
real breakers are registered/tripped inside the **runner** process.

**Fix options:**
- **(preferred)** Have the runner periodically serialize
  `circuit_manager.get_all_statistics()` + `get_open_breakers()` to a shared
  store (a row in the DB, or `data/circuit_breaker_state.json`). Change the
  endpoint to read that store instead of the in-process singleton.
- Alternatively expose the runner's breaker state over its existing
  websocket/IPC and have the dashboard subscribe.

**Acceptance:** with the runner live and at least one breaker registered, the
Risk tab "Circuit Breakers" panel lists each breaker with state
(closed/open/half_open) and call counts; the open-count badge is non-zero when
a breaker is open.

---

## #2 — Data validator stats never reach the dashboard

**Symptom:** validation panel all zeros, `pass_rate` defaults to 100%.

**Cause:** `app.py:get_data_validator_status()` does
`if hasattr(app, "data_validator"): stats = app.data_validator.get_statistics()`
— but `app.data_validator` is only ever set in the runner's `app` context, not
the dashboard's. The dashboard always hits the zero-default fallthrough.

**Fix options:**
- Persist validator counters (`total_validations`, `passed`, `failed_stale`,
  `failed_spread`, `failed_price`, `failed_volume`, `failed_anomaly`) from the
  runner to the DB or a state file on each cycle; read them in the endpoint.
- Or move the validator's running stats into a DB-backed counter table the
  validator increments directly (works regardless of which process reads).

**Acceptance:** with the runner live and having validated ≥1 bar, the Risk tab
"Data Validation" panel shows non-zero `total_validations` and a real
`pass_rate`, with per-reason failure counts.

---

## #3 — `risk_metrics.leverage` / `total_exposure` are 0

**Symptom:** leverage gauge pinned at 0; exposure 0 despite open positions.

**Cause:** `total_exposure = Σ(quantity × current_price)` over active
positions, but the positions table's `current_price` column is 0/null. (The
Kelly panel works because it reads historical *trades*, not the live
`current_price`.) Related: `daily_pnl`/`consecutive_losses`/`daily_loss` read 0
whenever there are no losing trades dated *today* — that part is expected and
correct, not a bug.

**Fix options:**
- Ensure the runner writes the latest market price into
  `positions.current_price` each cycle (the stop-loss monitor already tracks a
  live price per symbol — reuse that write path), **or**
- Have the endpoint fall back to the most recent price from `market_data` /
  the latest trade when `current_price` is null, so exposure/leverage compute
  even between price writes.

**Acceptance:** with open positions and fresh prices, leverage and
total-exposure gauges reflect real values; the leverage gauge color band
(safe/warning/danger) tracks the computed ratio.

---

## Out of scope (already correct, do not "fix")

- **Kelly sizing** — works; DB-backed.
- **Kill-switch limits** reading 0 when nothing is triggered — correct.
- **`daily_loss` = 0 with no losing trades today** — correct.
- The Risk tab markup, escaping, gauges, and tab wiring — verified working.

## Suggested sequencing

Do #3 first (smallest, reuses the stop-loss price path, lights up the most
visible gauge), then #1 and #2 together since they share the same
"persist runner state to a shared store" pattern.

---

## #4 — Pre-existing `get_risk_status` issues surfaced by the multi-agent review (2026-05-30)

The Risk-tab PR (#76) review flagged these in `get_risk_status`. They are
**pre-existing on `main`** — the PR's only change to that function was adding a
zero-capital guard — so they were intentionally left out of the UI PR and are
tracked here instead.

- **Timezone-naive daily-P&L window.** `daily_pnl` selects today's trades via
  `datetime.fromisoformat(t["timestamp"]).date() == datetime.now().date()`. If
  timestamps are stored UTC-aware, `.date()` is not adjusted to local (ET), so
  extended-hours trades between 8 PM and midnight ET land in the wrong day and
  skew the daily-loss gauge / kill-switch threshold. Fix: compare in a single
  consistent timezone (ET, or UTC on both sides).

- **`max_drawdown` is a rolling 100-trade window, not all-time.** It's computed
  over `db.get_recent_trades(limit=100)` starting from `cumulative_pnl = 0`, so
  it measures drawdown from the start of that window, not the historical equity
  peak — understating true drawdown. The UI labels it "Max Drawdown (Realized)".
  Fix: compute from the `equity_history` all-time peak, or relabel as
  "Rolling 100-trade drawdown".

- **No `portfolio_id` scoping.** `/api/risk/status` (and the three
  `/api/safety/*` endpoints) read only the default portfolio; they carry
  `@requires_auth` but not `@validate_portfolio`. The Risk tab fetches via
  `withPortfolio(...)`, but the backend ignores the `portfolio_id` query param,
  so in multi-portfolio mode the tab always shows the default portfolio. This is
  consistent across all four endpoints (not a regression). Fix: add
  `@validate_portfolio` and thread `portfolio_id` into the `SyncDatabaseReader`
  calls — do all four together for consistency.

- **Negative-capital guard is falsy-only.** The new guard
  `if not risk_state.get("current_capital"): … = 100000` catches `0`/missing but
  not a negative value (negatives are truthy), which would yield negative
  leverage. Low-likelihood data path; tighten to `<= 0` / type-check, mirroring
  the more defensive guard in `get_kelly_parameters`.

### Also noted (not backend — housekeeping)

- `app.py` does not pass `black --check` on `main` (e.g. lines ~483, ~7252),
  independent of this PR. A repo-wide `black` pass should be its own dedicated
  commit, never folded into a feature PR (it would bury real changes in a
  whole-file reformat).
