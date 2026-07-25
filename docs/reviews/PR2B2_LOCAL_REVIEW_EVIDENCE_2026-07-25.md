# PR 2B.2 Local Review Evidence

Date: 2026-07-25
Tracking issue: #101
Branch: `codex/pr2b2-broker-bound-exits`
Implementation head before this evidence update:
`5ba3bde2948bb3a1469245ea1f893d346cb90d4f`

This file preserves the local verification used before hosted review. It is not
a GitHub approval object and does not authorize starting the trader.

## Scope and trust model

PR 2B.2 routes every active local-paper reduction through one account-wide
gateway. The gateway combines:

- producer-owned IBKR contract, paper-account, loopback/4002/read-only
  transport-generation, and IBC configuration evidence;
- one-transaction, exact cross-portfolio allocation evidence bound to the
  configured SQLite path, database identity, device, and inode;
- the PR 2B.1 safety coordinator and dedicated journal;
- one sealed, exact `PaperExecutor` per portfolio.

IBKR remains read-only and has no order-placement capability. The exact local
simulator ledger is therefore the allocation authority for local paper fills.
IBKR positions and open orders remain diagnostic inputs for PR 5 and cannot
automatically authorize, block, or rewrite local simulator state.

## Verification results

Final pre-documentation full suite:

```text
1995 passed, 5 skipped, 1 xfailed, 20 warnings in 63.40s
```

The single strict XFAIL is
`test_successful_fill_is_settled_and_journal_released_before_gateway_unlock`.
It proves the PR 2B.3 settlement gap and must not be removed, weakened, or
converted to a pass by disabling the assertion.

Additional checks:

- changed-file Black: passed;
- changed-file isort: passed;
- changed-file Flake8: passed;
- Python compilation: passed;
- `git diff --check`: passed;
- targeted Bandit scan of new safety boundaries: no medium or high findings;
- tracked/untracked secret scan: only documented synthetic `DU1234567` and
  fixed CI/test key fixtures were present; the ignored local `.env` was not
  changed.

Expected skips were one missing optional `feedparser` dependency, two
Docker-Compose-unavailable checks, one existing pairs-path TODO, and one
production-config fixture requiring additional environment.

## Phase-one findings and remediation

Independent code, trading, and bug-finder reviews produced these valid
findings:

1. A valid maximum-width portfolio ID could make a stop `order_ref` exceed the
   gateway's 128-character limit. The reference is now a deterministic
   fixed-width `stop:v1:` SHA-256 identity, with a maximum-boundary regression.
2. A started coordinator using the correct scopes but a different safety
   journal could be accepted. Gateway construction now proves the configured
   journal path and the exact device/inode bound at coordinator startup, with a
   same-scope/different-file rejection test.
3. SQLite pool error recovery could enqueue both a closed original and an
   unbound replacement, deadlocking a size-one queue and orphaning the new
   connection. Recovery now removes the failed connection, opens and validates
   one replacement against the configured ledger inode, and enqueues exactly
   one usable connection. The regression checks queue size, pool ownership,
   descriptor identity, and a bounded subsequent checkout.
4. The terminal-settlement readiness check originally existed only in runner
   orchestration. One shared fail-closed gate now protects runner startup,
   paper-order runtime startup, gateway startup, entry serialization, and
   reduction submission.

All repaired focused tests passed.

## Phase-two challenger verdict

The independent verification challenger re-ran the reproductions and inspected
the repaired boundaries:

- the four repaired findings were closed;
- no new merge-blocking regression was found;
- dormant PR 2B.2 code was judged suitable for hosted review and merge;
- starting the trader remains prohibited.

The challenger retained two high-severity pre-activation requirements for PR
2B.3:

1. apply every successful fill durably to the authoritative allocation ledger
   and terminally settle/release its safety-journal reservation before the
   account-wide gateway lock is released;
2. bind every authorized reduction reference price to producer-owned fresh
   protective-quote price, timestamp, and lineage, then revalidate that evidence
   immediately before submission.

The shared readiness gate must remain false until both requirements have
passing success, failure, cancellation, crash-injection, and restart-replay
coverage.

## Operational status

The trader, dashboard, WebSocket server, and Gateway were not started by this
review. No kill switch, safety journal, trading history, position, equity, or
credential state was cleared or modified. Gate A remains closed.
