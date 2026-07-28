# PR 3 Local Review Evidence

Date: 2026-07-27
Branch: `codex/pr3-market-data-contract`

This file records local implementation and review evidence for PR 3. It is not
a GitHub approval object and does not authorize starting the trader. The shared
paper-readiness gate remains false.

## Scope and non-goals

PR 3 establishes one versioned market-data contract for historical strategy
bars and a separate broker event contract for protective monitoring. It does
not enable broker order writes, enable live trading, reconcile broker and local
financial state, migrate the legacy financial ledger, clear any kill switch,
or bypass startup preflight.

IBKR remains paper-port and read-only. Exposure-reducing paper executions keep
the PR 2B reduce-only authorization and exact-settlement boundary. Exposure-
increasing paper simulation is admitted only when current producer-owned broker
quote evidence and validated canonical strategy bars are both present.

## Canonical historical-bar contract

The subprocess broker boundary normalizes IBKR historical `TRADES` bars into a
versioned immutable batch with:

- exact symbol, contract ID, exchange, timezone, timeframe, and broker request
  semantics;
- timezone-aware broker and retrieval timestamps;
- explicit regular or extended session classification;
- exact decimal OHLC values, nonnegative volume, adjustment state, source, and
  quality flags; and
- validation of ordering, duplicates, gaps, staleness, session membership,
  finite values, and OHLC relationships.

The DataFrame is a compatibility projection for strategy calculations. It
retains the canonical batch as lineage metadata, but it is not the authority
for execution prices. A legacy or untagged DataFrame cannot reach entry
execution.

## Protective broker-event channel

Protective monitoring uses an independent read-only IBKR tick-by-tick `Last`
subscription channel. Subscriptions persist across polling intervals, are
retired when symbols leave the protected set, and expose every unseen event in
broker order. Stable event identities, contract IDs, broker timestamps,
transport generations, and exact decimal prices are validated before the stop
monitor accepts them.

Feed failure retires the transport generation and synchronously invalidates its
unlatched quote evidence. A stop crossing already latched by the monitor keeps
its immutable event evidence through the existing PR 2B terminal-settlement
flow. Dashboard status is observational; it cannot mint or revoke order
authority.

## Entry admission and paper-fill price

Historical bars determine strategy signals. A current authoritative broker
quote determines entry sizing, the paper order price, and the initial
protective stop. Immediately before exposure increase, the account-wide paper
gateway serializes admission, verifies broker connectivity, refreshes the
protective quote inside that serialization boundary, and requires a second
freshness and generation check. This closes the previous time-of-check/time-
of-use gap.

Extended-hours entries additionally require a canonical batch explicitly
requested for extended sessions and a latest bar classified in the matching
pre-market or after-hours session. The runtime never silently substitutes a
regular-session close.

## Durable history and operator visibility

Canonical rows accumulate by explicit identity instead of replacing numbered
legacy rows. Stored lineage includes request/session semantics as well as bar
values and timestamps. Database admission revalidates the complete row rather
than trusting dictionary key presence. Reads choose an explicit timeframe and
use deterministic ordering.

The dashboard and WebSocket payloads expose event time, retrieval time, source,
session, timeframe, and freshness. An empty valid result is distinct from a
database-read failure; the market-data API returns a sanitized service error
for the latter. Periodic cleanup does not delete canonical audit history, and a
portfolio-scoped signal cleanup cannot trigger global market-data deletion.

## Rollback and data implications

Rollback removes the new canonical read/write and protective-feed wiring while
leaving the pre-existing legacy market-data table untouched. This PR does not
delete, rewrite, or migrate user trades, positions, accounts, equity history,
the safety journal, or broker credentials. Canonical history is additive.

## Two-phase review findings

The first review phase found unsafe bar-close paper fills, an entry freshness
race, recovery-generation/status coupling, leaked tick subscriptions, missing
multi-event replay coverage, incomplete persisted lineage, weak database
admission, ambiguous timeframe reads, incorrect API error classification,
unbounded staleness configuration, global cleanup side effects, incomplete
extended-hours admission, and a legacy-data execution path. Each confirmed
finding must be closed by code and adversarial tests before hosted review.

The phase-two challenger result will be recorded here after reviewing the exact
post-fix tree.

## Final local verification

The implementation tree based on PR 2B.3 merge `017f43e` passed the following
local checks on 2026-07-28 before rebasing onto the PR 108 merge:

- the focused PR 3 and affected PR 1/PR 2 regression suite passed 490 tests;
- the full local suite passed 2,301 tests with 5 expected skips and 20 known
  warnings;
- Black check, isort check, Flake8, and `git diff --check` passed for every
  modified or added Python file.

After rebasing onto PR 108 merge `8fbf0f6`, the focused suite plus the PR 108
WebSocket ownership and authentication regressions passed 567 tests with one
known warning. The full post-rebase local suite passed 2,333 tests with 5
expected skips and 20 known warnings. The phase-two challenger result remains
to be recorded. No in-progress or concurrently edited test run counts as final
evidence.

## Remaining launch blockers

PR 3 does not make paper startup safe by itself:

- the shared paper-readiness flag remains false;
- PR 4 financial-state/database durability is outstanding;
- PR 5 read-only broker reconciliation is outstanding;
- the Gate A subset of PR 7, exact-state bootstrap/FIFO work, and verified
  backup restoration are outstanding;
- the current machine's loss-triggered TSLA kill switch and lock require
  explicit operator resolution; and
- startup requires a clean ordinary preflight, immediate operator
  notification, and explicit user confirmation.

No trader, dashboard, WebSocket server, Gateway, watchdog, safety state, user
trading data, or broker credential is started or modified by this review.
