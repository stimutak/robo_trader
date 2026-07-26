# PR 2B.3 Local Review Evidence

Date: 2026-07-26
Tracking issue: #102
Branch: `codex/pr2b3-terminal-settlement`
Implementation commit: `14f6b55`

This file preserves the exact-head local verification performed before hosted
review. It is not a GitHub approval object and does not authorize starting the
trader. The shared paper terminal-settlement readiness gate remains false.

## Scope and trust model

PR 2B.3 closes the local-paper terminal-settlement gap left deliberately open
by PR 2B.2. IBKR remains diagnostic and read-only. The only execution sink is
the local synchronous `PaperExecutor`; no broker order-write path is added.

For a terminal local-paper reduction, one SQLite `BEGIN IMMEDIATE` transaction
now commits the exact request and receipt together with:

- the trade and signed post-position;
- exact cash, realized P&L, and daily mark-to-market P&L;
- the exact day-start unrealized baseline and its canonical UTC date;
- exact position cost basis, prior/new marks, and settlement-source lineage;
- the complete canonical producer-owned protective-quote payload; and
- an idempotent terminal settlement outbox record.

Runtime position, Portfolio, stop, runner daily-risk, and advanced-risk views
must project and verify the exact committed receipt before the safety journal
can release authority. Any failure after possible submission is terminally
quarantined; the executor is never retried or replaced.

## Exact financial and quote semantics

- Durable authority uses `Decimal`; float-only compatibility updates cannot
  mint or overwrite exact account or position state.
- A filled exit first revalues the full signed pre-position from its prior exact
  mark to the authenticated protective mark. It then replaces only the filled
  shares' unrealized P&L with realized P&L at the exact fill price.
- Long `SELL` and short `BUY_TO_COVER` partial-then-full sequences converge
  across settlement, refresh, restart, and advanced-risk views, including
  fill-versus-mark slippage.
- The daily baseline value and UTC date are persisted and CAS-checked. Same-day
  restart restores them exactly. A later-day restart requires exact marks for
  every nonzero position, performs an explicit rollover, persists it, and only
  then permits admission.
- A latched stop remains bound to its immutable producer-owned quote evidence
  while that evidence is fresh, even if a newer quote arrives. Reconnect cannot
  relabel a cached historical/legacy price as live protective evidence.
- Paper fills are quantized to the executor's exact tick before producing a
  float compatibility view.

## Crash and offline recovery

The management command
`recover-exact-local-settlement --confirm RECOVER-EXACT-LOCAL-PAPER-SETTLEMENT`
is available only while the trader, Gateway, and ports 4001/4002 are stopped.
It uses the configured ledger path/device/inode and one query-only SQLite
snapshot. A unique match must prove:

- canonical request, receipt, and protective-quote fingerprints;
- one exact linked trade for a fill, or no trade mutation for a zero-fill;
- portfolio and cross-portfolio aggregate quantities;
- exact cost basis, prior/final mark, and position source lineage;
- legacy and exact cash, realized, unrealized, and daily P&L projections;
- the day-start baseline/date and account source settlement;
- canonical timestamps that cannot postdate the commit; and
- the exact journal reservation, claim, account, and execution-domain identity.

Only the terminal safety-journal event is appended. Trade, position, account,
settlement, and SQLite sidecar files are not rewritten by recovery. Missing,
forged, partial, duplicated, ambiguous, outbox-only, or tampered evidence stays
quarantined. `CANCELLED`, `REJECTED`, and `EXPIRED` zero-fill crash boundaries
are covered as well as filled outcomes.

## Two-phase review findings

Independent code-quality, trading-safety, bug, and style passes initially found
or confirmed:

1. non-atomic cash/realized/daily account state;
2. daily-risk and advanced-risk drift after restart or sequential exits;
3. missing signed short/partial-cover restart state;
4. incomplete crash-after-commit recovery;
5. offline recovery trusting a self-consistent outbox without linked
   projections;
6. zero-fill outcomes that could remain quarantined forever;
7. mutable/newer quote state invalidating a still-fresh latched stop;
8. reconnect rewarming from non-producer-owned price state;
9. terminal no-fill outcomes losing stop ownership;
10. exact/float slippage disagreement after a committed fill; and
11. recovery diagnostics or sidecars needing stronger immutability/redaction
    coverage.

Each finding was reproduced, repaired, and covered by focused adversarial
tests. The phase-two challenger then found and closed the changing-mark bridge,
baseline-date rollover, float-authority contamination, legacy-basis/source
drift, and expanded recovery-lineage interactions. Its final exact-tree verdict
was PASS with no remaining blocker in the reviewed PR 2B.3 scope.

## Final local verification

Full repository suite:

```text
2240 passed, 5 skipped, 20 warnings in 65.29s
```

Focused terminal-settlement, recovery, routing, reconnect, and trading-security
matrix:

```text
375 passed, 2 skipped, 1 warning in 4.99s
```

Additional gates:

- changed-file Black: passed, 31 files unchanged;
- changed-file isort: passed;
- full `robo_trader/` and `tests/` Flake8: zero findings;
- Python compilation: passed;
- `git diff --check`: passed;
- `pip check`: no broken requirements;
- targeted Bandit medium/high scan: zero findings;
- targeted mypy on the new settlement, runtime-settlement, quote-evidence, and
  recovery modules: no issues;
- secret-pattern scan: no matches.

The full-package mypy job remains advisory because the repository has broad
pre-existing typing and third-party-stub debt. This PR did not count that
advisory failure as a pass.

One stale pytest process from an earlier concurrently edited tree was
terminated by exact PID after it exceeded one hour. The same persistence file
then passed all 15 tests in 0.89 seconds with faulthandler enabled; no
reproducible deadlock was found, so the stale process is not counted as test
evidence.

## Remaining launch blockers

This PR does not make paper startup safe by itself:

- `PAPER_TERMINAL_SETTLEMENT_READY` remains false;
- existing ledgers need an explicit, non-destructive exact-state bootstrap;
- the repository still uses weighted-average cost rather than a strict FIFO lot
  ledger;
- PRs 3, 4, and 5 and the Gate-A subset of PR 7 remain required;
- read-only broker reconciliation and backup/restore evidence remain required;
- the current machine still has a real TSLA loss-triggered kill switch and lock,
  stale equity history, and a stopped Gateway; and
- startup requires a clean ordinary preflight, immediate operator notification,
  and explicit user confirmation.

No production database, safety state, trading history, broker credential,
Gateway, dashboard, WebSocket server, or trader process was modified or started
during this implementation and review.
