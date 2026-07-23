# PR 80 Disposition and Finding Ownership

Date: 2026-07-23

Decision: close PR #80 unmerged after PR #82. Do not rebase it and do not
cherry-pick either of its commits (`dd26ad5`, `edd0288`) wholesale.

## Why PR 80 cannot merge

PR #80 branched from `103d798`, before the Phase 0 CI truth gate, runtime
stability prerequisite, and final paper-only containment design. Eight of its
13 files overlap the safety-critical scope now implemented by PR #82
(`393f533`). It also retains 11 review findings that make the branch unsafe as
a combined change.

PR #82 is the authoritative PR 1 implementation. It provides the canonical
paper/read-only runtime contract, hard-disables live execution, serializes the
launcher and Gateway lifecycle, verifies the connected account, keeps
dashboard safety controls inert, quarantines unsafe entrypoints, and makes
container and Kubernetes trader entrypoints intentionally unavailable.

## File and feature disposition

| PR 80 area | Disposition |
| --- | --- |
| `START_TRADER.sh` mode and port changes | Superseded. PR #80 can read mode too late and probe the wrong port. PR #82 enforces supervised paper port 4002. |
| `robo_trader/config.py` mode alias | Superseded by the stricter canonical runtime contract, which rejects live mode and conflicting aliases. |
| `scripts/preflight_check.py` and tests | Superseded by PR #82's fail-closed environment resolution and fixed paper topology. |
| `docs/configuration.md` live and legacy wording | Discard. It describes live-capable behavior while the remediation gate is intentionally closed. |
| Dashboard kill-switch reset | Discard. PR #82 makes reset inert. PR #80 can clear safety state before durable audit succeeds. |
| Dashboard unavailable-service responses | Preserve only as a PR 10 requirement for authoritative, timestamped unavailable-state telemetry. |
| `LiveExecutor` task handling and connection normalization | Preserve only as PR 11 requirements. PR #80 can lose the identity of a working `Submitted` broker order. |
| `runner_async.py` async execution routing | Preserve only as a PR 11 requirement and redesign across every order path, including `OrderManager`. |
| Backtesting engine and tests | Preserve only as PR 6 input. The attempted repair still mishandles return indices, zero equity, NaN OHLC values, and trigger-price fills. |
| Formatting-only Flask changes | No independent remediation value. |

## Review finding crosswalk

### Superseded by PR 1 / PR #82

- Legacy mode could be ignored.
- `.env` could be read after the decision that depended on it.
- A hardcoded or wrong Gateway port could be probed.
- Empty-mode behavior was ambiguous.

These findings are closed by the canonical paper-only runtime contract and
supervised launcher merged in PR #82.

### Assigned to PR 6 - Backtesting correctness

Problem identifier: RT-013.

- Returns and timestamps can diverge after zero equity.
- Sparse or NaN OHLC values can suppress exits.
- Intrabar stop or take-profit exits can fill at the bar close rather than the
  triggered price.

PR 6 must implement these semantics from first principles and include
deterministic regression fixtures. No PR #80 backtest code is carried forward.

### Assigned to PR 8 - Identity and privileged safety actions

Problem identifiers: RT-020 and RT-021.

- A kill-switch administrative action must require a reason, elevated
  authorization or reauthentication, and append-only durable audit.
- Safety state must never be cleared before the audit record is durably
  committed.

PR 8 owns the authorization design. The dashboard control remains inert until
that gate is complete.

### Assigned to PR 10 - Honest operational telemetry

Problem identifiers: RT-012 and RT-029.

- An unavailable safety service must never be represented as healthy.
- Telemetry must identify its authoritative source, observation timestamp, and
  stale or unavailable state.

PR 10 will reimplement this behavior as part of the operational truth model.

### Assigned to PR 11 - Broker order lifecycle

Problem identifiers: RT-001 and RT-002.

- A working `Submitted` order must retain its broker identity and remain
  tracked; it is neither a rejection nor a fill.
- Async execution must cover every order path, including the existing
  `OrderManager` path.
- Ambiguous results require broker reconciliation and must never invite blind
  resubmission.

PR 11 owns the broker-authoritative order state machine. No live executor code
from PR #80 is carried forward.

## Traceability statement

No commit from PR #80 was merged. Its safe, independent requirements are
preserved in the ordered remediation plan under PRs 6, 8, 10, and 11. Its
overlapping implementation is superseded by PR #82.
