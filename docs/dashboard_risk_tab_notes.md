# Risk tab — verification notes

This branch adds a new **Risk** tab to the dashboard, cherry-picked from
the abandoned `feature/dashboard-redesign` (PR #70, Feb 2026). The
extraction skipped the parts that conflicted with security hardening
landed since then (W-H3 escape work).

The tab is wired up and ships with `escHTML()` on every DB/env-derived
field, but the backend contract was last verified against Feb 2026 main —
multi-portfolio (Feb 6) and persistent-connection (#72) have landed
since. **Before merging, confirm the four backend endpoints still
return the shapes the JS expects.**

## Backend contract expected by `loadRiskData()`

### `GET /api/risk/status?portfolio_id=<id>`
```jsonc
{
  "kill_switches": {
    "active": false,
    "limits": {
      "daily_loss":          { "current": 0.012, "limit": 0.05 },
      "consecutive_losses":  { "current": 2,     "limit": 5 },
      "max_drawdown":        { "current": 0.04,  "limit": 0.10 }
    }
  },
  "risk_metrics": {
    "leverage": 0.42,
    "total_exposure": 12500.0,
    "daily_pnl": -125.50,
    "current_capital": 100000,
    "max_drawdown": 0.04,
    "total_pnl": 1250.0
  },
  "kelly_sizing": {
    "portfolio_kelly": 0.18,
    "current_positions": {
      "NVDA": { "kelly_fraction": 0.12, "win_rate": 0.62, "edge": 0.024 }
    }
  }
}
```
Source: `app.py:get_risk_status` (around line 6269 on this branch).
Loss-prevention guard for zero `current_capital` already landed in
branch 1 (`feature/dashboard-fixes-and-overview`).

### `GET /api/safety/circuit-breakers`
```jsonc
{
  "open_count": 0,
  "breakers": {
    "<name>": {
      "state": "closed",                 // "closed" | "open" | "half_open"
      "total_calls": 1234,
      "successful_calls": 1200
    }
  }
}
```

### `GET /api/safety/thresholds`
Now includes `trailing_stop_percent` and `use_trailing_stop` (added in
branch 1). The Risk tab renders all keys generically — adding new env
vars to this endpoint is safe.

### `GET /api/safety/data-validator`
```jsonc
{
  "total_validations": 5432,
  "pass_rate": 99.2,
  "failed_stale":  3,
  "failed_spread": 1,
  "failed_price":  0,
  "failed_volume": 0
}
```

## Manual verification before merge

1. Click **Risk** tab — all four panels should populate within ~2s.
2. With multi-portfolio enabled, switch portfolios and confirm the risk
   gauges + Kelly table re-fetch per portfolio (relies on
   `withPortfolio('/api/risk/status')`).
3. Force a kill-switch trigger
   (`scripts/_set_kill_switch_state.py` if present, or hand-edit
   `data/risk_state.json`) and confirm the badge flips to **TRIGGERED**
   and turns red.
4. With Gateway down, the data validator panel should still render
   (validator stats live in DB, not in the runner).
5. Run `pytest tests/` — no dashboard tests should regress.

## Intentionally **not** included from PR #70

| Feature from redesign | Why deferred |
|---|---|
| **Sortable positions table** (F4) | Main's positions table has W-H3 escapes inline; making it sortable requires restructuring `updatePositionsTable` into `renderPositionsRows` while preserving the security hardening. Worth a follow-up PR. |
| Removal of `updateSafetyMonitoring` JS | The redesign deleted ~120 lines of "orphaned" safety monitoring code, but those functions are still wired into `window.onload` in main and may be relied on elsewhere. Risk tab is additive — it coexists with the old code rather than replacing it. A cleanup PR can remove the duplication later once both surfaces are confirmed redundant. |

## Stacked on `feature/dashboard-fixes-and-overview`

This branch builds on the fixes/overview branch (P&L by Symbol, Active
Stops, Signal Activity panels, `/api/ml/status` symlink fix, div-by-zero
guards, trailing-stop env vars, keyboard shortcuts). **Merge the
fixes/overview branch first** — the Risk tab JS calls
`/api/safety/thresholds` expecting the new `trailing_stop_percent` /
`use_trailing_stop` fields added there.
