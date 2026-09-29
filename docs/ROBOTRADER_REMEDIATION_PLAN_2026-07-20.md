# RoboTrader Remediation and Launch Plan

Document date: 2026-07-20
Last updated: 2026-07-30
Status: Active execution plan; Gate A remains closed
Source baseline: repository audit at commit `51f0e99`; execution baseline
`1ae64808ae20956e412e95f17f194776f4851478` on branch `main`
Target: safe supervised paper operation first, then remote read-only access, then an explicitly gated limited live canary

Current execution baseline (2026-07-28): `main` includes the truthful Phase 0
CI gates from PR #83 (`7f5de0a`) and runtime-stability fixes from PR #81
(`b7e5005`). PR #82 completed PR 1 and merged as `393f533`. PR #87 completed
PR 1A and merged as `4cafb782cbf43ff4397f1b89b42d5f657eceea8e`,
closing the incident-driven broker-correlation and protected-runtime gate.
PR #80 is superseded and must not be merged or cherry-picked; its independent
findings are assigned to later scoped PRs in
`docs/branch_analysis/PR80_DISPOSITION_2026-07-23.md`. PR #90 completed the
PR 1B diagnostic implementation and merged as
`0d43006561071e27b217cb6d16f3c0a245a18655`. Its first real invocation from
merged `main` failed closed before broker connection because the local runtime
does not yet define an explicit paper-account identity and allow-list. Issue
#92 tracks that local-only operator prerequisite. Protected evidence hashes
were unchanged and no correction was attempted. PR #95 then completed the
PR 1C package/import prerequisite and merged as
`dff4c8b597a54c614d8925565f28aa865f8ae676`. The package root is now inert,
active subpackages and the required dashboard template are included in built
wheels, broker safeguards remain explicit at direct `IB` users, and runtime
artifacts and secrets remain excluded. Runtime dependency metadata and a true
clean-install wheel gate remain assigned to PR 9. Gate A remains closed, the
trader remains stopped. Dormant safety-core PR 2A merged through PR #97 as
`d17b0d5b4f31ab15e2a9b138cca006c0103b7276`; it has no production imports or
runtime wiring. PR 2B.1 merged through PR #104 as
`3ecdaa05b3352ddcd4519662b0fe957751f3fdb1` from exact reviewed head
`0d5585b46f8f1b495d944e24b23df5a7c01cfc2d`. It binds the dormant paper
safety coordinator to startup replay but deliberately leaves production order
authorization and submission unwired. PR 2B.2 merged through PR #106 as
`8328c822733b1a2358a8d6f26d368ab0819cd106` from exact reviewed head
`6e2b768ea474d7e8e41037c7b3e9a3606ed00482`. Its full Python matrices,
integration, performance, lint, security, container, Cursor security, and
exact-head Codex review gates passed; the Claude workflow was unavailable
because its repository OAuth token was revoked before review. The
implementation remains deliberately dormant behind the shared
terminal-settlement readiness gate. PR 2B.3 merged through PR #107 as
`017f43e`. PR 3 merged through PR #111 as `8596920`. The PR 4 exact-state
bootstrap code slice merged through PR #112 as `bccb96b`; strict FIFO,
operator-reviewed bootstrap/application, and restore evidence remain open. PR
5 foundations merged through PR #113 (`6176931`) and PR #115 (`1ae6480`). The
PR 5 runtime-integration draft adds fail-closed startup, reconnect, and periodic
reconciliation plus sanitized dashboard status. A current operator-reviewed
broker reconciliation remains open. Gate A remains closed, the trader remains
stopped, and none of these merges or draft changes authorizes startup.

PR4A's dormant exact FIFO foundation is merged through PR #120. PR4B's
operator-reviewed legacy-opening bridge is under review as PR #123 at exact
head `318532e`; it includes sealed opening-manifest count/hash, exact trigger
body verification, and in-transaction schema revalidation. PR4C
transactionally connecting producer-owned local-paper fills to FIFO is
implemented on `codex/pr4c-fifo-settlement`, stacked on that
exact PR4B head, and is awaiting independent review. These code slices do not
apply an operational bootstrap, authorize startup, or close Gate A.

## 1. Purpose

This document turns the full codebase audit into an ordered implementation program. It is intended to remain the reference plan for future pull requests, reviews, test campaigns, launch decisions, and operational sign-off.

The ordering is deliberate. Later work must not begin merely because earlier code was merged. Every phase has evidence gates that must pass in the actual supported runtime.

The plan addresses:

- trading correctness;
- IBKR and exchange integration;
- order execution and paper/live separation;
- risk, sizing, stops, take-profit, and strategies;
- backtesting and market-data quality;
- database integrity, backups, and reconciliation;
- security, secrets, identity, authorization, and API safety;
- logging, auditability, testing, CI/CD, packaging, and deployment;
- dashboard, mobile, cloud, missing features, and unfinished features.

## 2. Non-negotiable rules

1. Keep IBKR `ReadOnlyApi=yes` until the live-enablement gate in PR 11 has been implemented and independently approved.
2. Never delete or rewrite user trading data without explicit user authorization, a verified backup, a preview of the exact impact, and a tested rollback path.
3. Do not treat a green unit-test count as launch evidence. Launch evidence includes reconciliation, restart, restore, failure-injection, alert, and soak results.
4. Broker state becomes authoritative whenever real broker orders are possible. Local database state alone must never authorize trading after startup or reconnect.
5. A kill switch must block risk-increasing orders while preserving a narrowly defined reduce-only path.
6. No remote control, mobile control, or cloud exposure is allowed before identity, authorization, TLS, WebSocket privilege separation, and durable audit controls exist.
7. Backtests cannot approve a strategy until the accounting and execution model in PR 6 is complete.
8. Every production claim must name the supported topology. Until changed by an approved architecture decision, that topology is local macOS, IBC, IB Gateway paper account, and the authoritative launcher.
9. Every PR must be independently reviewable, revertible where practical, and leave the paper system in a coherent state.
10. Incomplete strategies and alternate engines remain disabled or quarantined rather than partially integrated.

## 3. Current baseline and containment boundary

The current sanctioned runtime is materially contained:

- `START_TRADER.sh` uses paper port 4002 and validates IBC read-only configuration.
- `runner_async.py` constructs `PaperExecutor`.
- The active subprocess broker client is read-oriented and does not expose a complete order-placement interface.
- The current local dashboard is bound to loopback.
- The preflight gate, launchd watchdog, persistent connection recovery, test isolation, and persisted kill switch are valuable foundations.

This containment must be made explicit and tested in PR 1. It must not be weakened while the later live-order architecture is built.

## 4. How to use this plan

For each PR:

1. Create one tracking issue using the PR section below as its body.
2. Confirm its prerequisites are complete and evidenced.
3. Add or update tests before changing safety behavior where feasible.
4. Implement only the stated scope. Record extra discoveries in the problem register rather than silently expanding the PR.
5. Run the PR-specific verification suite plus the repository regression suite.
6. Attach evidence to the PR: commands, test output, screenshots where applicable, migration/restore results, and fault-injection results.
7. Update this document's progress register after merge.
8. Do not advance a launch gate until every required PR and operational exercise for that gate is complete.

Recommended branch naming: `codex/pr-XX-short-name` or the team's equivalent feature prefix.

## 5. Dependency sequence

The required order is:

1. PR 1 preserves the safe paper-only boundary and removes destructive hazards.
2. PR 1A closes the active cross-symbol market-data contamination path before
   any safety exit can be allowed through a kill switch.
3. PR 1B adds non-mutating broker-versus-ledger reconciliation and resolves the
   incident evidence gate without clearing safety state.
4. PR 1C makes package-root imports side-effect-free and ensures the dormant
   safety core and all active subpackages can ship in built wheels.
5. PRs 2 through 5 make active paper behavior safe and state authoritative.
6. PR 6 repairs the evidence engine used to evaluate strategies.
7. PR 7 unifies strategy and risk behavior on top of corrected data and state.
8. PRs 8 through 10 establish remote security, reproducible delivery, and honest operations.
9. PRs 11 and 12 add live order placement and broker-native protection behind a disabled gate.
10. PR 13 proves failure behavior through soak and fault injection.
11. PR 14 delivers an explicitly supported remote/cloud topology if still required.
12. PR 15 permits only a tiny, manually armed live canary.

PRs may be prepared in parallel only when their files and safety contracts do not overlap. Merge order still follows the dependency chain.

PR 7 is deliberately staged. The code foundations that satisfy the prerequisite
for preparing Gate-A-only PR 7 slices are the reviewed PR 2B.3, PR 3, PR 4
exact-state-bootstrap, and PR 5 domain/runtime-evidence merges through
`1ae64808ae20956e412e95f17f194776f4851478`. Before the remaining PR 4 and PR
5 operational work is complete, only non-authorizing slices may merge: they
must either narrow the existing local-paper surface or remain dormant. Such
slices must not widen admitted strategies or order shapes, consume backtest
results, set paper readiness true, authorize startup, admit an entry, or add
broker-write capability. Production integration that can admit an entry remains
blocked on the PR 4 FIFO/bootstrap/restore work and PR 5 runtime reconciliation
work. PR 6 remains a prerequisite for strategy readiness cards, performance or
parameter approval, selector enablement, completion of PR 7, and Gate B. This
is the sole exception to the merge order above.

# Phase A - Preserve containment and correct the active paper system

## PR 1 - Freeze live mode and quarantine destructive utilities

### Objective

Make the current paper/read-only boundary unambiguous, machine-enforced, and visible. Remove the possibility that a dormant script can destroy user data during later remediation work.

### Problems addressed

- Conflicting `EXECUTION_MODE`, `TRADING_MODE`, `TRADING_ENV`, and deployment-mode variables.
- Production manifests that claim live operation while the runner remains paper.
- Missing IBC-config branch that can allow startup checks to pass when the file is absent.
- Dashboard status that can report a hardcoded or incorrect mode.
- Destructive recovery and IB position-sync scripts.
- Shared paper/live database, logs, artifacts, or credentials.

### Step-by-step work

1. Define one canonical runtime contract containing environment, execution mode, broker port, broker account identifier, broker account type, database path, model-artifact set, and build identifier.
2. Add a startup validator that rejects every inconsistent combination, including:
   - paper mode with live port 4001;
   - live mode with paper port 4002;
   - missing IBC configuration;
   - IBC `ReadOnlyApi` not equal to `yes` during the containment phase;
   - missing or unapproved broker account identifier;
   - shared paper/live database path;
   - production mode without authentication, signing, alerting, or backup readiness.
3. Make the authoritative launcher and runner refuse live execution regardless of environment values. Use a separate compile-time or capability flag that remains disabled until PR 11.
4. Replace dashboard hardcoded mode reporting with the validated server-side runtime contract.
5. Add an unmistakable PAPER banner containing account alias, execution source, database identity, build SHA, and configuration fingerprint.
6. Inventory every alternate launcher. Disable or route `scripts/start_runner.sh`, Docker commands, dashboard start, and Kubernetes commands through the same contract.
7. Quarantine `simple_recover.py`, `recover_database.py`, and `sync_ib_positions.py` so they cannot run accidentally.
8. Design replacement maintenance commands with:
   - read-only preview;
   - portfolio-scoped diff;
   - SQLite online backup;
   - explicit typed confirmation;
   - transaction boundaries;
   - post-operation integrity and reconciliation checks.
9. Document the current supported topology and explicitly label Docker/Kubernetes/live as unsupported.

### Required tests

- Table-driven tests for every mode/port/account/database combination.
- Test that missing IBC configuration fails before connecting.
- Test that all sanctioned entrypoints invoke the same validation.
- Test that live mode remains impossible even with contradictory environment variables.
- Test that maintenance utilities default to preview and cannot delete without the authorization sequence.

### Done means

- No sanctioned command can connect to port 4001 or create a live executor.
- Dashboard and logs derive mode from the validated runtime contract.
- Paper and potential live state paths cannot collide.
- Destructive utilities cannot modify data by import or by default invocation.
- The supported-topology document matches executable behavior.

### Rollback

Revert only to the previous paper launcher. Never restore destructive utility behavior.

## PR 1A - Fail closed on broker-data correlation

### Objective

Prevent a delayed or timed-out IBKR response from being assigned to another
symbol, and prevent malformed historical timestamps from overwriting the
evidence needed to detect contamination.

This is an incident-driven prerequisite discovered on 2026-07-23. A timed-out
historical-data request left a response in an uncorrelated FIFO queue. Later
requests consumed shifted responses, a quote for one symbol was labeled as
another symbol, and the mislabeled value immediately triggered position-loss
and stop logic. The stop did not execute because the kill switch blocked all
orders. Therefore PR 2 must not enable reduce-only exits until PR 1A is merged
and verified.

### Problems addressed

- RT-010 and RT-033.
- Subprocess requests and responses have no correlation identifier.
- A timeout does not invalidate or drain the uncertain worker session.
- Market-data responses are trusted without contract-symbol or conId
  verification.
- Historical bars persist a DataFrame RangeIndex instead of normalized market
  timestamps, so later cycles overwrite history.
- Cross-symbol contamination can reach risk checks, stop-loss logic, account
  values, and the database.
- Historical bars are not a live protective feed, and no independent
  protective-price writer exists before PR 3. Existing holdings must therefore
  abort startup instead of running with empty or stale stop-monitor prices.
- Position-load and stop-registration failures can currently be logged and
  treated as empty or partially protected state.
- A fail-closed setup abort can leak resources or be restarted repeatedly by
  the watchdog unless cleanup and supervision preserve the safety decision.

### Step-by-step work

1. Version every command and response envelope with a worker-generation
   identifier, request identifier, and command name.
2. Match responses by request and worker-generation identifier; reject missing,
   duplicate, stale, malformed, or unexpected identifiers.
3. After any timeout or protocol uncertainty, terminate and recreate the worker
   session and all response state before accepting another response. Never
   automatically retry the uncertain request.
4. Include requested and returned contract identity in each response: symbol,
   conId, exchange, currency, request type, and observation timestamp.
5. Reject mismatched or incomplete identities before updating risk, stops,
   positions, account values, caches, or persistent state.
6. Normalize historical bars to validated timezone-aware timestamps from the
   broker payload. Reject RangeIndex, duplicate, non-monotonic, future, or
   session-invalid timestamps.
7. Make persistence append or upsert by the validated timestamp and contract
   identity; never overwrite another symbol or cycle silently.
8. Pass broker event time into freshness checks. Receipt time must not make a
   stale bar appear current.
9. Poison and disconnect the client generation on timeout, cancellation,
   malformed envelopes, reader failure, or identity mismatch. Reject queued and
   new commands until recovery creates an isolated generation.
10. Abort the remaining symbol cycle after protocol poison and surface the
    failure to connection health.
11. Until PR 3 supplies and confirms an independent live protective feed, fail
    setup for every nonzero existing position unless the stop monitor owns a
    matching pending stop and a fresh accepted `live_protective` event.
12. Treat database position-load and stop-registration uncertainty as fatal;
    never fall back to an empty holdings view.
13. Clean every partially initialized resource independently, exit nonzero with
    a sanitized audit reason, and make the watchdog suppress automatic restart
    for that exact terminal safety exit until an operator deliberately runs the
    authoritative launcher after protection is restored.

### Required tests

- A delayed response after timeout cannot satisfy the next request.
- Out-of-order, duplicate, missing-ID, and unknown-ID responses fail closed.
- A symbol or conId mismatch cannot reach risk or stop callbacks.
- Timeout recovery starts a clean worker session and cannot reuse queued data.
- Cancellation followed by a late response cannot affect another request.
- Missing, wrong, duplicate, stale-generation, or malformed envelopes poison
  the transport.
- A deterministic three-symbol sequence proves no response shifting.
- RangeIndex and malformed timestamps are rejected.
- Valid timezone-aware bars persist without cross-cycle overwrite.
- Cross-symbol payload duplication is detected before persistence.
- Protocol failure produces no database write, cache update, latest-price
  update, strategy call, trailing-stop adjustment, or stop execution.
- Existing holdings with no monitor-owned live event, a historical-only event,
  a missing or mismatched stop, or stale/future timestamps fail setup.
- Position-load and stop-registration failures are fatal.
- A stop-monitor cleanup failure cannot prevent IBKR and database cleanup.
- The watchdog policy suppresses the terminal unprotected-position restart and
  permits ordinary nonterminal recovery.

### Done means

- Every accepted broker-data response is bound to the exact originating
  request and contract.
- A timeout cannot contaminate a later request.
- Invalid data cannot reach trading, risk, stop, valuation, or persistence
  consumers.
- Historical rows retain auditable market timestamps.
- The incident scenario passes deterministic failure-injection tests.
- A holdings-bearing runner cannot look healthy while its stops are blind, and
  its terminal safety abort survives the process-supervisor boundary.

### Non-goals

- Do not migrate or rewrite historical rows.
- Do not correct positions, account values, kill-switch state, or lock files.
- Do not connect to the broker while running unit and failure-injection tests.
- Defer broader source lineage, session policy, and real-time feed unification
  to PR 3.

## PR 1B - Add read-only broker-ledger reconciliation

### Objective

Produce the broker evidence required to explain the incident and determine
whether a separate user-approved, backed-up state-correction action should be
designed, without placing orders or mutating local data.

### Problems addressed

- No safe read-only broker-versus-ledger reconciliation command exists.
- Preflight guidance points to a nonexistent reconciliation script.
- The old synchronization utility is intentionally inert because it previously
  deleted and replaced positions.
- Local paper-executor fills and the IBKR paper account may represent different
  systems, but the difference is not made explicit.

### Step-by-step work

1. Create a diagnostic-only command that explicitly requests an IBKR read-only
   session on paper port 4002.
2. Reuse the PR 1 runtime contract and verify the managed account exactly before
   reading account data.
3. Read and mask account identity, contract identity, positions, average cost,
   open orders, and recent executions.
4. Open SQLite in immutable or read-only mode and compute a portfolio-scoped
   diff without invoking application write paths.
5. Report source, timestamp, freshness, symbol, conId, exchange, currency,
   quantity, cost, orders, and executions for each difference.
6. Make order submission methods unavailable by construction and fail if the
   Gateway or client cannot prove read-only operation.
7. Prove that database, kill-switch, lock, bypass-log, and trading-log hashes
   remain unchanged after the command.
8. Replace the nonexistent preflight remediation reference with this command.
9. Document that output is evidence only: it cannot clear a kill switch,
   correct the ledger, or authorize startup.
10. Run the affected-symbol reconciliation and obtain explicit operator
    approval before designing any separate state-correction action.

### Required tests

- Paper port and read-only flags are mandatory and cannot be overridden.
- The managed account must match the runtime contract.
- Account identifiers and sensitive fields are masked in normal and error
  output.
- No order method is reachable.
- SQLite is opened read-only and every relevant file hash remains unchanged.
- Quantity, cost, contract, open-order, and recent-execution differences are
  deterministic.
- Missing, stale, or ambiguous broker data fails closed.
- Preflight points only to the implemented diagnostic command.

### Done means

- A broker-versus-ledger snapshot can be produced without mutation.
- The incident evidence has an explicit broker/account interpretation.
- Any proposed state correction is a separate reviewed action requiring user
  authorization and backup.

### Operational gate

The trader remains stopped after PR 1A and PR 1B. Reconciliation and operator
review are necessary evidence, but they do not authorize startup. The first
supervised paper start requires cumulative Gate A: PRs 1, 1A, 1B, 2 through 5,
the relevant PR 7 controls, reviewed reconciliation, restore evidence, and all
ordinary `./START_TRADER.sh` checks passing without deleting or bypassing
safety state.

Current operational evidence (2026-07-23): the merged command failed closed
before broker connection because the local runtime has no explicit paper
account identity or allow-list. Issue #92 tracks the local-only configuration
prerequisite. The blocked run changed none of the protected evidence files and
does not satisfy the reviewed-reconciliation requirement above.

## PR 1C - Isolate package imports and complete wheel discovery

### Objective

Remove two structural blockers discovered during the adversarial design review
for dormant safety-core issue #91: importing `robo_trader.safety` must not load
or mutate broker code, and a built wheel must not omit the safety package or
other active subpackages.

### Problems addressed

- Python executes `robo_trader/__init__.py` before any nested package import,
  and the previous initializer eagerly imported `ibkr_safe`, loaded
  `ib_async`, and globally patched `IB.disconnect`.
- `pyproject.toml` declared only the root `robo_trader` package, so source-tree
  tests could pass while built wheels omitted every nested package.
- Broad package discovery could accidentally include archived code, similarly
  named sibling packages, tests, configuration, credentials, databases, or
  logs.
- Required non-Python assets could be omitted even when their Python package
  was included.

### Step-by-step work

1. Make `robo_trader/__init__.py` metadata-only and side-effect-free.
2. Require every production module that directly constructs `ib_async.IB` to
   activate `ibkr_safe` explicitly.
3. Replace the root-only package list with precise, namespace-aware discovery
   for `robo_trader` and `robo_trader.*`.
4. Exclude archived code and similarly named sibling packages.
5. Disable implicit package-data inclusion and allow-list only the required
   bug-dashboard template.
6. Require a patched setuptools build backend and pin the supported
   development backend.
7. Add cold-import tests that reject broker imports, filesystem mutation,
   sockets, subprocesses, threads, current-directory changes, and environment
   changes.
8. Build a wheel from a copied exact tree, inspect its contents, render the
   packaged dashboard template, and import representative regular and
   namespace subpackages from outside the checkout.

### Required tests

- A cold `python -I` root import is inert and does not load broker modules.
- Explicit `ibkr_safe` activation remains idempotent and force-only.
- Every direct production `IB` user activates the disconnect guard.
- Built wheels include active regular and namespace packages plus the required
  dashboard template.
- Built wheels exclude archives, sibling packages, tests, config, `.env`, IBC
  credentials, databases, and logs.
- The full repository suite and supported Python 3.10 through 3.12 hosted
  matrices pass.

### Done means

- The package root is safe for PR 2A to add and import
  `robo_trader.safety` without activating broker behavior.
- A built wheel contains the dormant safety core and all other active
  subpackages once they are added.
- Direct broker users retain the documented disconnect safeguard.
- Package discovery does not expose secrets or runtime state.

### Explicit deferral

PR 1C proves import and distribution *structure*, not a standalone application
installation. The pre-existing `dependencies = []` metadata, separation of
runtime/dev/test/ML/operations extras, lockfiles with hashes, and a true clean
wheel install remain PR 9 scope. The current supported setup continues to
install tracked requirements before the project package.

### Operational gate

This prerequisite changes no runtime wiring, broker connection, launcher,
database, safety state, or order authority. It cannot authorize startup. Gate
A remains closed and the trader remains stopped.

## PR 2 - Implement a reduce-only safety plane

### Staging and current status

PR 2 is deliberately staged as a dormant core followed by separately reviewed
runtime-integration changes:

- **PR 2A / issue #91:** implement the immutable exact-value models, pure
  reduce-only policy, durable append-only journal, idempotency and reservation
  rules, and package-boundary dormancy tests. PR #97 merged this stage as
  `d17b0d5b4f31ab15e2a9b138cca006c0103b7276`.
- **PR 2B.1 / issue #100:** establish paper runtime identity, trusted evidence
  boundaries, and fail-closed startup journal replay without granting order
  authority. PR #104 merged this stage as
  `3ecdaa05b3352ddcd4519662b0fe957751f3fdb1`.
- **PR 2B.2 / issue #101:** route paper exits through broker-bound reduce-only
  authorization and separate hard safety blocks from entry-only soft blocks.
  PR #106 merged this stage as
  `8328c822733b1a2358a8d6f26d368ab0819cd106` from exact reviewed head
  `6e2b768ea474d7e8e41037c7b3e9a3606ed00482`. Every operational submission
  boundary remains hard-disabled.
- **PR 2B.3 / issue #102:** add exact settlement, ambiguity handling, and
  crash/restart quarantine release, and bind every reduction price to
  producer-owned fresh protective-quote price, timestamp, and lineage. PR
  #107 merged this stage as `017f43e5c31283e6cf5cc630c6fd15acd2af683b`.
  The shared readiness gate remains false.

PR 2A grants no broker connection or submission authority by itself. PR 2B.1
now imports the safety runtime in the production runner and launcher only to
bind identity and replay the journal before supervised process or Gateway
mutation. It still does not call authorization, consume a production
submission permit, or wire the executor/stop monitor to the safety coordinator.
Gate A remains closed and the trader remains stopped. PR 2B.3 code is merged;
the remaining cumulative Gate A integration and operational evidence are still
required before any supervised paper start.

### Objective

Ensure kill switches and circuit breakers stop new exposure without blocking exits that strictly reduce existing exposure.

### Problems addressed

- Kill-switch lock blocks stop-loss orders.
- All-order blocking conflates entry safety with emergency liquidation.
- Emergency shutdown cancels local stops without cancelling broker orders or flattening.
- Stop callbacks and state changes can diverge after failure.

### Step-by-step work

1. Define an `OrderIntent` contract containing account, portfolio, symbol, side, quantity, current signed position, target signed position, reason, strategy, reduce-only flag, and idempotency key.
2. Implement a pure exposure-delta validator:
   - long SELL quantity cannot exceed the existing long position;
   - short BUY-to-cover cannot exceed the existing short position;
   - a reduce-only order cannot cross through zero;
   - an absent or uncertain position rejects the order.
3. Split gates into risk-increasing and risk-reducing policies.
4. Make kill switch, daily-loss limit, max-position limit, entry circuit breaker, and closed entry window reject only risk-increasing intents.
5. Keep broker disconnect, uncertain broker position, stale account state, and failed reconciliation as hard blocks even for automated exits; escalate these to an operator rather than guessing.
6. Route stop-loss and future flatten operations through the reduce-only validator.
7. Persist the reason and authorization decision for every rejected and accepted safety order.
8. Make emergency behavior explicit:
   - cancel open entry orders;
   - preserve or replace protective exit orders;
   - never claim flattening unless broker fills prove it.
9. Correct execution failure handling, including the `.message` versus `.msg` mismatch, so rejection details are never hidden by secondary exceptions.

### Required tests

- Long reduction, full close, over-close, and reversal attempts.
- Short cover, full cover, over-cover, and reversal attempts.
- Stop execution while kill switch is active.
- Stop execution while daily limits or entry circuit breakers are active.
- Unknown broker position and stale reconciliation cases.
- Repeated identical safety intents proving idempotency.

### Done means

- A gap-triggered kill switch cannot block a valid reduce-only exit.
- No reduce-only path can create or reverse exposure.
- Every decision has a durable reason and test coverage.

## PR 3 - Establish the canonical market-data contract

### Objective

Make every trading, stop, database, strategy, and dashboard consumer use validated, timezone-aware, session-correct market data.

### Problems addressed

- RangeIndex values stored as timestamps.
- Database history overwritten on every refresh.
- Historical-bar strings not normalized.
- Regular-hours data used during extended-hours trading.
- Active runner bypasses authoritative validation.
- Stop monitor depends on periodically stale historical closes.
- Concurrent performance timers overwrite one another.

### Step-by-step work

1. Define a versioned bar schema: symbol, exchange, timezone-aware timestamp, open, high, low, close, volume, session, source, retrieval time, adjustment state, and quality flags.
2. Parse IBKR dates at the subprocess boundary and reject ambiguous or timezone-naive values.
3. Set the normalized timestamp as the DataFrame index before returning data.
4. Validate OHLC ordering, finite values, nonnegative volume, monotonic order, duplicates, gaps, staleness, and session membership.
5. Store actual timestamps and use explicit upsert rules that preserve history.
6. Make regular versus extended-hours retrieval an explicit strategy/runtime choice.
7. Add an independent quote or real-time bar channel for protective monitoring.
8. Define degraded behavior for stale or missing data:
   - block new entries;
   - retain broker-native protection;
   - alert the operator;
   - never silently substitute the last regular-session close for an extended-hours decision.
9. Make performance timer identifiers unique per symbol and operation instance.
10. Add data lineage and freshness to dashboard/API responses.

### Required tests

- DST transitions, market holidays, regular/extended session boundaries, and timezone conversions.
- Duplicate, reversed, NaN, infinite, zero-volume, and out-of-order bars.
- Database accumulation across repeated refreshes.
- Extended-hours fetch and decision tests.
- Stale protective feed behavior.
- Multi-symbol concurrent timer tests.

### Done means

- Database timestamps match broker timestamps.
- Historical refreshes accumulate rather than replace numbered rows.
- No strategy or stop consumes unvalidated bars.
- Extended-hours state is explicit and tested.

## PR 4 - Make financial state durable and database operations safe

### Objective

Create a recoverable financial ledger with constrained schema, safe pooling, safe migrations, and verified backups.

### Problems addressed

- Connection-pool replacement can deadlock and requeue closed connections.
- Financial tables lack foreign keys and value constraints.
- SQLite `REAL` is used for critical money values.
- Trades lack broker order/execution IDs and lifecycle fields.
- Formal multiuser migration is not wired; partial startup migrations swallow failures.
- Migration backup is not WAL-safe and may collapse account rows.
- Deployment and dashboard database settings disagree.
- No demonstrated restore-ready backup exists.

### Step-by-step work

1. Select one authoritative database setting and require every runner, dashboard, utility, container, backup, and migration to use it.
2. Fix pool replacement so exactly one valid connection returns to the bounded queue and the internal pool inventory remains correct.
3. Add a schema-version table and explicit ordered migrations. Remove broad exception swallowing.
4. Add database constraints for:
   - valid sides and lifecycle states;
   - positive quantities and prices where required;
   - finite, bounded risk percentages;
   - portfolio references;
   - unique idempotency and broker execution identifiers.
5. Enable foreign-key enforcement for every connection.
6. Store monetary values in lossless integer minor units or validated decimal strings according to a documented convention.
7. Add append-only tables for order intent, broker order state, fills, commissions, reconciliation snapshots, safety events, and administrator actions.
8. Make position/account projections derived from confirmed fill events or updated transactionally with those events.
9. Use SQLite's online backup API or a verified exclusive checkpoint procedure.
10. Encrypt and rotate off-host backups without exposing broker credentials.
11. Add automated backup integrity checks and a clean-room restore test.
12. Create migration dry-run, row-count, checksum, and rollback reports.

### Required tests

- Pool failure injection under concurrency.
- Constraint and foreign-key rejection tests.
- Duplicate broker callback/idempotency tests.
- WAL-active backup followed by clean restore.
- Interrupted migration at each step.
- Multiportfolio preservation tests.
- Deployment database path and volume persistence tests.

### Done means

- A database exception cannot deadlock the runner.
- An invalid financial row cannot enter through normal application paths.
- Backups include WAL state and restore to an integrity-clean database.
- Every order and fill has durable correlation identifiers.

### PR4C implementation checkpoint (2026-07-30)

The local-paper terminal settlement now appends each producer-authenticated
fill, exact USD commission, FIFO lots/matches/snapshot, compatibility
projections, terminal outbox, and immutable settlement-to-FIFO link in one
SQLite transaction. Exact replay re-authenticates the stored event and link;
missing epochs, conflicting evidence, projection divergence, and partial writes
fail closed. Stopped-system recovery also verifies the full FIFO epoch and link
from its existing query-only snapshot before releasing safety authority.
Epoch realized P&L is accumulated across every instrument through the current
event sequence. The locked settlement transaction also rejects temporary
shadows and foreign persistent triggers on all hot settlement tables before it
writes, and all such statements explicitly target the durable `main` schema.

The full local suite passed 3,055 tests with 5 expected skips and 20 known
warnings. Detailed design and evidence are recorded in
`docs/design/PR4C_FIFO_RUNTIME_SETTLEMENT.md` and
`docs/reviews/PR4C_LOCAL_REVIEW_EVIDENCE_2026-07-30.md`.

This checkpoint does not complete PR4. PR4B must merge first; the reviewed
bootstrap must then be applied only through the stopped-system operator path,
and WAL-safe backup/clean-room restore plus current reconciliation evidence
remain mandatory. The active simulator still emits complete fills only;
partial order-lifecycle support remains quarantined. Gate A remains closed and
the trader remains stopped.

## PR 5 - Add read-only broker reconciliation

### Objective

Make broker truth visible and authoritative before any broker-writing capability exists.

### Problems addressed

- Startup trusts only the local database.
- Database-load errors fail open to empty positions.
- Broker positions helper exists but is unused by the active runner.
- Open orders, completed orders, executions, cash, and commissions are not reconciled.
- Existing reconciliation documentation points to a missing script.

### Step-by-step work

1. Extend the read-only broker adapter to fetch account identity/type, cash, buying power, signed positions, open orders, completed orders, executions, and commissions.
2. Normalize broker objects into versioned domain records.
3. Implement reconciliation at:
   - startup;
   - reconnect;
   - fixed periodic intervals;
   - before arming future live capability;
   - after any ambiguous order state in later phases.
4. Compare broker state with the append-only ledger and derived projections.
5. Classify differences as expected timing lag, recoverable missing event, duplicate event, account mismatch, quantity mismatch, cash mismatch, or unknown.
6. Quarantine trading on all unknown or material mismatches.
7. Provide an operator diff that never auto-deletes data.
8. Persist reconciliation snapshots and resolution actions.
9. Expose reconciliation age and status on the dashboard.

### Required tests

- Clean startup reconciliation.
- Local missing fill recovered from broker execution.
- Duplicate callback.
- Wrong account, wrong quantity, stale cash, unknown order, and reconnect scenarios.
- Database unavailable or partially migrated.
- Operator resolution without destructive replacement.

### Runtime-integration draft evidence (2026-07-30)

- Startup, reconnect, and fixed-period triggers use one read-only diagnostic
  provider and publish a fresh, one-shot signed reconciliation result from the
  same broker transport generation. An exact provider-generation lease is held
  from snapshot collection through protective marks and publication, so health
  refresh, suspension, and close cannot splice two generations into one bundle.
- Runtime database and bootstrap-receipt readiness are verified read-only before
  broker collection; absent, partial, or unauthenticated state fails closed
  without schema repair or data mutation.
- Risk-increasing entries require a fresh eligible reconciliation. Quarantine
  preserves the existing semantic reduce-only path so protective reductions are
  not stranded.
- Every protective-mark request carries the complete unique account-wide set
  of active position symbols, so reconciliation cannot cancel another
  position's shared-worker quote subscription.
- Dashboard status exposes only sanitized reconciliation state, age, trigger,
  and quarantine/eligibility fields; signing-capability paths and broker
  evidence are not exposed. Existing owned status is replaced only through an
  atomic, fully synced inode exchange; unrelated targets are restored rather
  than overwritten, and a crash leaves a complete old or new artifact.
- Published evidence is never auto-deleted, moved, or rewritten. The runtime
  applies a non-destructive ceiling before publication (defaults: 10,000
  runtime bundles and 2 GiB, configurable with
  `RT_RECONCILIATION_EVIDENCE_MAX_BUNDLES` and
  `RT_RECONCILIATION_EVIDENCE_MAX_BYTES`). Reaching either ceiling quarantines
  new entries while portfolio cycles continue through reduce-only gates; an
  operator-reviewed archival action is required before evidence collection can
  resume. Every admission performs a fresh held-root usage scan, excluding
  only the exact device/inode binding of its own current staging directory, so
  externally added bundles cannot hide behind cached counters. A non-blocking
  POSIX lock on the held evidence-root directory serializes that final scan
  through completion-marker publication across cooperating processes. The
  receiver binds admission to the exact sealed entry identities,
  hashes, and final completion-marker bytes immediately before publication.
  Completion-marker schema v2 commits that exact artifact manifest, and the
  loader rejects any added, replaced, removed, or rewritten directory entry;
  any pre-marker raced bundle is left incomplete and preserved for inspection.
  Active/bootstrap/audit lineage remains intact.
- Adversarial regressions cover protected and raced status targets,
  suspended-provider recovery, broker-envelope replay, complete account-wide
  protective symbol scope, reduce-only continuity after artifact failures,
  non-destructive retention ceilings, and persisted eligibility expiry.
- Synthetic and mocked verification on 2026-07-30: `3082 passed, 5 skipped`
  across the complete pytest suite; Black and Flake8 passed for all changed
  Python files. No trader process, Gateway, broker session, or authoritative
  trading database was started or accessed.
- Operational evidence is still required: provision the owner-only signing and
  evidence directories, run the authoritative launcher under supervision, and
  obtain operator review of a current broker reconciliation. Those actions are
  not authorized by this draft, and Gate A remains closed.

### Done means

- Startup cannot continue trading after local-state uncertainty.
- Every mismatch produces a durable, understandable diff.
- No reconciliation operation deletes history.

# Phase B - Repair strategy evidence and execution policy

## PR 6 - Rewrite backtesting and walk-forward validation

### Objective

Create a deterministic, auditable backtest and replay system that can support strategy decisions without look-ahead or accounting distortion.

### Problems addressed

- Returns are always calculated as zero.
- Signals execute at the same bar used to generate them.
- Commissions and spread are double-counted.
- Slippage is unseeded.
- Partial fills are not implemented.
- Exceptions are swallowed and final liquidation is incomplete.
- Metrics assume an inappropriate frequency.
- Walk-forward selection contaminates out-of-sample evidence.
- Critical backtesting modules have no direct tests.

### Step-by-step work

1. Reset all engine state at the start of every run and validate sorted, nonempty input.
2. Separate observation time, decision time, submission time, and executable fill time.
3. Default to next-bar or event-realizable fills.
4. Define one cost model where spread, slippage, fees, commissions, borrowing, and market impact are counted exactly once.
5. Seed all stochastic behavior and record the seed in results.
6. Implement liquidity limits and genuine partial fills.
7. Use intrabar OHLC semantics for stop/take-profit triggers while documenting ambiguity resolution.
8. Support long and short accounting, dividends, splits, borrow costs, and delistings as needed by candidate strategies.
9. Calculate equity after fills and append final liquidation/mark-to-market economics.
10. Annualize metrics from actual sampling frequency and guard every zero denominator.
11. Fail the run on data/strategy exceptions unless an explicit, reported policy says otherwise.
12. Implement rolling or nested walk-forward validation with an untouched final holdout.
13. Produce a reproducibility manifest containing code SHA, data checksum, configuration, costs, seed, and model identifiers.

### Required tests

- Hand-calculated golden portfolios.
- Look-ahead detection fixtures.
- Commission/spread charged once.
- Partial fill and volume constraints.
- Stop gaps and intrabar ambiguity.
- Empty input, all-error input, and final liquidation.
- Deterministic reruns and untouched holdout verification.

### Done means

- Golden fixtures match exact hand calculations.
- Repeated runs with the same manifest are identical.
- No strategy receives launch approval from the old engine.

## PR 7 - Unify risk, sizing, exits, and strategy contracts

### Objective

Replace duplicate, drifting risk and strategy pathways with one authoritative decision-to-order contract.

### Problems addressed

- Hardcoded 10% advanced position cap versus configured 2% cap.
- Symbol validation mistakenly uses the sector limit.
- Daily notional resets after restart.
- Nearest-share rounding can exceed limits.
- Missing market data fails some checks open.
- Existing/short positions are missing or wrong in advanced-risk state.
- Take-profit exists as metadata but not an authoritative lifecycle.
- Pairs legs are non-atomic and use synthetic model training/placeholders.
- Extracted runner modules, `core/engine.py`, and active runner duplicate behavior.

### Step-by-step work

1. Select one active strategy interface and one active risk engine.
2. Define a versioned `Signal -> OrderIntent` contract containing confidence, expected horizon, source data version, sizing request, entry policy, stop policy, take-profit policy, and expiry.
3. Centralize all configuration loading and remove hardcoded risk values.
4. Enforce position, portfolio, sector, correlation, leverage, liquidity, buying-power, daily notional, churn, and duplicate limits in one place.
5. Floor quantities so calculated exposure never exceeds a cap.
6. Restore daily counters from the ledger on startup.
7. Synchronize long, short, and existing positions from reconciled state.
8. Define take-profit semantics as either broker bracket behavior or an explicitly tested exit policy.
9. Disable incomplete strategies by default, especially pairs, shorts, smart execution, AI discovery, and synthetic-trained selectors.
10. For future pairs support, validate both legs before submission and add combo/hedge, timeout, and compensating-unwind behavior.
11. Archive or clearly quarantine dead alternate engines and unused extracted runner modules until intentionally integrated.
12. Produce a strategy readiness card for every candidate strategy: data, parameters, risk rules, tests, backtest manifest, shadow results, and enabled environments.

### Required tests

- Every risk limit across every enabled strategy.
- Restart persistence of daily counters.
- Long and short position accounting.
- Quantity flooring at boundary values.
- Missing data and missing sector/correlation inputs fail closed.
- Two-leg failures and compensating behavior before pairs can be enabled.

### Done means

- A configured 2% position cap cannot produce or validate a larger intent.
- Only strategies with readiness cards can be enabled.
- One code path owns sizing, stops, take-profit, and risk validation.

# Phase C - Secure remote access and make delivery trustworthy

## PR 8 - Build the identity, authorization, and WebSocket security boundary

### Objective

Permit safe read-only remote use without exposing broker credentials or granting every user administrative control.

### Problems addressed

- Dashboard authentication is disabled locally and uses one shared credential when enabled.
- No active RBAC, MFA, portfolio ownership, or individual actor audit.
- Kill-switch reset/start/stop use ordinary dashboard credentials.
- Shared WebSocket token allows consumer-to-producer impersonation.
- Token is placed in HTML/query strings; null/missing origins are admitted.
- Basic auth can be exposed by direct published ports.
- Legacy unsalted SHA-256 password setup.
- Model signature enforcement uses inconsistent mode detection.

### Step-by-step work

1. Introduce individual identities with strong password hashing and MFA.
2. Define roles such as viewer, portfolio operator, safety operator, and administrator.
3. Enforce portfolio-level authorization on every HTTP and WebSocket response.
4. Use secure server-side sessions or scoped, short-lived tokens with rotation and revocation.
5. Require reauthentication, a reason, and elevated permission for start, stop, kill-switch reset, and future live arming.
6. Consider two-person approval for live kill-switch reset.
7. Write administrative actions to append-only audit storage before applying the change where safety permits.
8. Replace shared producer trust with local IPC, mTLS, or a separate producer credential unavailable to dashboard consumers.
9. Use same-origin WSS through the TLS proxy; remove query-string tokens and reject null/missing origins for remote deployment.
10. Add message schemas, size limits, rate limits, subscription authorization, and portfolio filtering.
11. Move all secrets to approved environment/secret-manager injection and implement rotation procedures.
12. Make model signing mandatory from the canonical runtime contract and deserialize the exact verified bytes.
13. Add recursive log/config redaction and stop logging broker account identifiers except approved aliases.

### Required tests

- Role and portfolio access matrix.
- Session revocation, expiration, MFA, and reauthentication.
- CSRF, CORS, origin, proxy, brute-force, and rate-limit tests.
- Consumer attempts producer impersonation.
- Query-token leakage and log-redaction tests.
- Unsigned/tampered model refusal in every production-like mode.

### Done means

- Remote users are individually attributable and least-privileged.
- Dashboard consumers cannot publish runner events.
- Safety actions have elevated controls and durable audit.
- Mobile access may begin only as read-only.

## PR 9 - Restore CI, packaging, dependency, and supply-chain truth

### Objective

Make a green build mean the installable artifact, critical tests, security checks, and supported runtime all passed.

### Problems addressed

- 551 current mypy errors and formatting/lint failures.
- False-green tests return booleans instead of asserting.
- Safety tests are ignored by default.
- Critical modules have zero coverage; `app.py` is excluded.
- Performance jobs are no-ops.
- Package metadata omits runtime dependencies.
- Security scans do not always scan installed application dependencies.
- CI actions and container images use mutable references.
- Model/runtime dependency versions drift.

### Step-by-step work

1. Choose one authoritative CI workflow and remove contradictory duplicates.
2. Preserve PR 1C's precise subpackage discovery and package-data allow-list.
3. Declare runtime dependencies and split dev, test, ML, and operations extras.
4. Create a reproducible lock with hashes for supported Python versions.
5. Build a wheel and test it in a clean environment rather than relying on checkout imports.
6. Convert return/print-based tests to assertions and remove unconditional success messages.
7. Stop ignoring the safety suite; rewrite it into deterministic tests.
8. Add direct tests for backtester, order lifecycle, reconciliation, stop safety, database failures, and deployment startup.
9. Include dashboard/API code in coverage and set risk-based coverage thresholds.
10. Resolve type errors in active safety paths first; maintain a shrinking, explicit debt budget for lower-risk legacy code.
11. Add real performance, leak, concurrency, and soak jobs.
12. Scan the resolved production dependency set and built image.
13. Pin third-party actions and images to reviewed SHAs/digests.
14. Generate SBOM and provenance, sign artifacts, and verify them during deployment.
15. Pin model-training and inference environments and reject incompatible serialized artifacts.

### Required tests

- Clean wheel install and smoke test.
- Exact locked environment test matrix.
- Coverage gates on critical modules.
- Dependency and container vulnerability policy.
- Reproducible artifact/hash comparison.
- No pending asyncio tasks or leaked processes at test completion.

### Done means

- Required CI is green with no placeholder jobs.
- A clean wheel contains every required module and dependency.
- Critical safety paths have meaningful assertion coverage.

## PR 10 - Make telemetry, alerts, and the dashboard authoritative

### Objective

Give operators a truthful, fresh, source-attributed view of the actual runner and broker state.

### Problems addressed

- Dashboard panels contain hardcoded/mock metrics.
- Missing runner data is presented as healthy zero/default state.
- Dashboard process-local objects do not reflect runner state.
- Mode/status can be stale or contradictory.
- Webhook alerts are disabled in the current environment.
- Logs are tracked in Git and may contain sensitive trading telemetry.
- Mobile layout, accessibility, and polling behavior are weak.

### Step-by-step work

1. Define versioned presentation contracts for runtime, broker, orders, fills, positions, risk, strategies, data freshness, reconciliation, protection, alerts, and build identity.
2. Include source, timestamp, age, portfolio, environment, and availability state in every response.
3. Persist runner telemetry to a shared durable store or event stream rather than accessing process-local objects.
4. Remove every hardcoded or estimated value presented as actual execution/risk state.
5. Render `Unavailable`, `Stale`, `Degraded`, `Mock`, and `Zero` distinctly.
6. Add non-dismissible mode/account/reconciliation/protection banners.
7. Expose last completed cycle, last successful broker query, last fill, data age, current kill-switch reason, and active broker protective orders.
8. Test email/SMS/pager/webhook alerts end-to-end and add a dead-man alert.
9. Add tamper-evident audit event access for authorized operators.
10. Remove tracked runtime logs from future commits and document safe history-cleaning as a separate user-approved operation if desired.
11. Use same-origin WSS, responsive breakpoints, accessible tabs, and active-tab/subscription-driven refresh.
12. Keep the initial mobile surface read-only.

### Required tests

- API contract and freshness tests.
- Runner stopped, stale, DB error, broker disconnected, alert failed, and reconciliation mismatch states.
- Browser tests for desktop/mobile layouts and keyboard navigation.
- Verified human receipt of every critical alert class.

### Done means

- No mock or unavailable value is displayed as real/healthy.
- Operators can determine mode, account, freshness, reconciliation, and protection at a glance.
- Critical alerts reach a human and are recorded.

# Phase D - Add live capability behind a disabled gate

## PR 11 - Implement one live IBKR order lifecycle adapter

### Objective

Create a complete broker order state machine while keeping the capability disabled in all normal environments.

### Problems addressed

- Dormant `LiveExecutor` schedules orders ambiguously and treats `Submitted` as filled.
- No active broker `place_order` implementation.
- No idempotent order reference or authoritative broker lifecycle.
- No partial fill, reject, cancel, replace, commission, reconnect, or unknown-state handling.
- Order state is updated from simulation rather than broker fills.

### Step-by-step work

1. Define one asynchronous broker interface. Remove or quarantine synchronous wrappers that can outlive returned results.
2. Use deterministic client order IDs/order references tied to durable order intents.
3. Model states: created, pending submit, pre-submitted, submitted, partially filled, filled, cancel pending, cancelled, rejected, inactive, unknown, and reconciliation required.
4. Persist broker order ID, permanent ID, execution ID, timestamps, quantities, prices, commissions, warnings, and rejection details.
5. Update positions, cash, risk, and dashboard only from broker-confirmed fills.
6. Make retries idempotent and ambiguity-safe. A timeout must transition to unknown/reconcile, never automatic duplicate submission.
7. Add account allowlist and contract qualification.
8. Enforce market-hours, order-type, notional, buying power, shortability, and exchange/session constraints.
9. Implement cancel and replace with reconciliation after reconnect.
10. Cancel open entry orders during safety shutdown while preserving reduce-only protection.
11. Keep the entire adapter behind a disabled capability gate and use an IBKR test/paper account for integration tests.

### Required tests

- Submit acknowledgement without fill.
- Partial fills and multiple callbacks.
- Reject, inactive, cancel, replace, timeout, reconnect, duplicate callbacks, and duplicate client IDs.
- Wrong account and contract ambiguity.
- Process crash between broker fill and local persistence.
- No position change before confirmed fill.

### Done means

- `Submitted` is never treated as a fill.
- Ambiguous outcomes cannot produce automatic duplicate orders.
- Broker fills are the only source of financial state transitions.
- Capability remains disabled outside explicit integration tests.

## PR 12 - Add broker-native protective orders and exit lifecycle

### Objective

Ensure every live position has broker-resident protection that survives local failures.

### Problems addressed

- In-memory synthetic stops disappear when process/host/Gateway fails.
- Stop prices can be stale and gap behavior is unsafe.
- Take-profit is not an authoritative execution feature.
- Emergency flattening and broker cancellation are incomplete.

### Step-by-step work

1. Define supported bracket/OCO structures for long and short positions.
2. Implement IBKR parent/child transmit sequencing so protection cannot be accidentally left unsubmitted.
3. Establish stop-market versus stop-limit policy; default gap protection must not create an unfillable stale limit.
4. Implement broker-native take-profit where strategy policy calls for it.
5. Reconcile protective orders on startup, reconnect, fill, partial fill, quantity change, and cancel/replace.
6. Implement trailing-stop modification with rate limits and broker acknowledgement.
7. Define overnight, extended-hours, outside-RTH, and corporate-action behavior.
8. Prevent entry completion from being considered safe until expected protection is acknowledged at the broker.
9. Add operator-visible protection status and emergency cancel/flatten runbooks.
10. Retain synthetic monitoring only as an independent alarm, not the primary protection.

### Required tests

- Parent fill before/after child acknowledgements.
- Partial fills and protective quantity updates.
- Gap through stop, reconnect, Gateway failure, process kill, and host restart.
- Manual broker-side cancellation detection and repair.
- OCO behavior and trailing modifications.

### Done means

- Broker UI proves protection remains active with the RoboTrader processes stopped.
- Every reconciled live position has the required protection or trading is quarantined.

# Phase E - Prove operations, then allow only a limited launch

## PR 13 - Failure injection, restoration, and multi-week paper soak

### Objective

Prove that the complete system behaves safely under realistic failures before enabling live capability.

### Step-by-step work

1. Build a broker simulator covering submit, acknowledge, partial fill, fill, reject, cancel, duplicate callback, disconnect, and reconnect.
2. Run controlled failures for:
   - stale/malformed market data;
   - Gateway restart and 2FA delay;
   - network partition;
   - process kill and host reboot;
   - database locked/corrupt/full;
   - disk full and log growth;
   - alert provider failure;
   - duplicate executions;
   - missing or cancelled protective orders;
   - second-leg pairs failure if pairs remains planned.
3. Rehearse online backup and clean-machine restore.
4. Rehearse kill switch, cancel open entries, and manual flatten using the paper broker.
5. Verify reconciliation after every failure.
6. Conduct a multi-week paper/shadow soak using the exact release artifact and operational topology.
7. Track restarts, stale data, reconciliation mismatches, duplicate order attempts, pending tasks, DB latency, alert latency, and unbounded logs.
8. Produce a signed launch-readiness evidence package.

### Done means

- Zero unexplained duplicate orders.
- Zero unresolved reconciliation drift.
- Every active paper position has expected protection behavior.
- Restore and alert drills succeed.
- Soak exits within agreed reliability and risk thresholds.

## PR 14 - Deliver the selected remote/mobile/cloud topology

### Objective

Support remote or cloud operation only after an explicit architecture decision and without creating multiple order writers.

### Step-by-step work

1. Decide whether macOS + IBC remains the only order-writing topology.
2. If cloud is not required, remove misleading Kubernetes/live deployment artifacts and publish a secure remote read-only dashboard design.
3. If cloud is required, design:
   - IB Gateway ownership and interactive 2FA;
   - single active writer with lease and fencing;
   - encrypted persistent database and backups;
   - secret-manager integration;
   - private network and NetworkPolicy;
   - immutable signed images;
   - real liveness/readiness endpoints;
   - rolling deploy prevention for the order writer;
   - rollback and disaster recovery.
4. Make Docker paper mode boot and persist correctly before any production manifest.
5. Replace placeholder deployment jobs with real staging, smoke, health, approval, production, and rollback operations.
6. Release mobile as read-only first. Add remote mutating actions only after separate threat modeling and approval.

### Done means

- Exactly one fenced order writer can exist.
- Health checks measure actual runner/broker/reconciliation readiness.
- Deployment and rollback have been exercised, not merely documented.
- Remote clients use strong identity and TLS without direct broker credential access.

## PR 15 - Limited live canary and staged expansion

### Objective

Enable the smallest reasonable real-money exposure only after all prior gates pass.

### Preconditions

- PR 1, PR 1A, PR 1B, and PRs 2 through 14 are complete as applicable.
- PR 1B reconciliation evidence was reviewed and did not itself mutate or
  authorize mutation of the ledger or safety state.
- All P0 and P1 audit findings are closed.
- Independent security and trading-safety review approves the evidence package.
- Reconciliation is clean.
- Broker-native protection is visible in IBKR.
- Human alerts, backup restore, and manual flatten drills pass.
- Multi-week paper/shadow soak passes.

### Step-by-step work

1. Create separate live account configuration, credentials, database, logs, model artifacts, and deployment identity.
2. Require a manual arming ceremony with named operator, reason, build SHA, configuration fingerprint, account confirmation, and expiry time.
3. Start with:
   - tiny symbol allowlist;
   - one open position maximum;
   - very low absolute notional and daily-loss caps;
   - long-only simple orders;
   - no pairs, shorts, smart execution, AI discovery, extended-hours entries, or automatic strategy expansion.
4. Require broker-native stop protection before the entry is considered operationally complete.
5. Monitor every order and fill in real time with human acknowledgement.
6. Automatically disable new entries on any reconciliation, data, alert, protection, or process-health degradation.
7. Review after each trade and each day. Expansion requires a new approved stage, never an automatic threshold change.
8. Maintain a tested manual cancel/flatten path and a documented return-to-disabled procedure.

### Done means

- Canary trades reconcile exactly with broker records.
- No safety or operational deviation is unexplained.
- Expansion is separately approved with evidence.

# 6. Launch gates

Launch gates are cumulative. Gate B and every later gate require Gate A,
including explicit PR 1A correlation evidence and PR 1B read-only
reconciliation evidence. A numeric PR range never implicitly omits PR 1A or PR
1B.

## Gate A - Supervised local paper readiness

Required PRs: 1, 1A, 1B, 2 through 5, plus the Gate-A parts of 7 defined below.

The Gate-A PR 7 subset requires baseline-only containment across every
sanctioned entrypoint and one exact risk contract owning entry sizing and
admission. Every applicable position, portfolio, sector, correlation, leverage,
liquidity, buying-power, daily gross-filled-notional, churn, duplicate-entry,
and maximum-open-position limit must be enforced. Quantity must be floored;
current positions and concurrent pending exposure must be reserved account-wide
under one serialization boundary. Quotes must come from the canonical
broker-bound producer. The quote, executable-price ceiling, all exposure caps,
signed-position state, and reconciliation eligibility must be fresh and
revalidated immediately before submission. Daily gross filled notional must
restore durably and every terminal fill must be ingested exactly once. Missing,
stale, malformed, replaced, unauthenticated, or quarantined evidence must block
entry. Incomplete strategies, shorts, smart execution, AI/ML discovery, and
take-profit remain disabled. Readiness remains false until the integrated
evidence is reviewed and passes.

Evidence required:

- Broker confirms paper account and read-only API.
- PR 1A failure injection proves delayed, mismatched, stale, or uncorrelated
  broker data cannot reach valuation, risk, strategy, persistence, or stops.
- PR 1B broker-versus-ledger reconciliation is current, reviewed, and proves it
  did not modify the database or safety state.
- Paper state cannot collide with any future live state.
- Reduce-only exits pass all blocking-state tests.
- Data timestamps, freshness, and session semantics are correct.
- Startup fails closed on database or reconciliation uncertainty.
- Backups restore cleanly.
- No unsafeguarded destructive utility remains.
- The operator is notified immediately before startup and gives explicit
  confirmation. Only then may the system start, and only through
  `./START_TRADER.sh`; a preflight invocation by itself is not startup consent.

## Gate B - Strategy evaluation readiness

Required PRs: 6 and 7.

Evidence required:

- Golden backtest accounting passes.
- No same-bar look-ahead.
- Costs are counted once.
- Results reproduce from a manifest.
- Candidate strategy has a readiness card and untouched holdout results.
- Risk and sizing caps are proven across all paths.

## Gate C - Remote/mobile read-only readiness

Required PRs: 8 through 10.

Evidence required:

- Identity, MFA, least privilege, portfolio authorization, TLS, and revocation tests pass.
- WebSocket consumers cannot impersonate the producer.
- Mobile is read-only.
- Dashboard never presents stale/mock/unavailable values as healthy facts.
- Alerts reach a human.

## Gate D - Live implementation readiness

Required PRs: 11 and 12.

Evidence required:

- Complete order lifecycle works in IBKR paper integration.
- Submitted is distinct from filled.
- Ambiguous timeouts reconcile without duplicate submission.
- Broker-native protection survives process and host failure.
- Broker truth drives all financial state.

## Gate E - Live canary readiness

Required PRs: Gate A through Gate D, then PRs 13 through 15.

Evidence required:

- Failure drills, backup restore, alerts, and multi-week soak pass.
- Supported deployment topology is explicit and tested.
- Independent safety/security approval is recorded.
- Canary constraints and manual arming are active.

# 7. Cross-cutting problem register

Use these identifiers in issues and PR descriptions.

- RT-001: Active runtime always uses PaperExecutor; no coherent live path.
- RT-002: Dormant LiveExecutor has ambiguous async behavior and treats Submitted as success.
- RT-003: No broker-authoritative startup/reconnect reconciliation.
- RT-004: Kill-switch lock blocks reduce-only stop exits.
- RT-005: Stops are synthetic and periodically stale.
- RT-006: No broker-native bracket/OCO protection or authoritative take-profit.
- RT-007: Position limits drift between configuration and risk implementations.
- RT-008: Daily counters and advanced-risk state do not restore correctly.
- RT-009: Pairs execution is non-atomic and selector logic contains synthetic placeholders.
- RT-010: Market-data timestamps are stored as RangeIndex values.
- RT-011: Extended-hours decisions may consume regular-hours-only data.
- RT-012: Active data validation and performance telemetry are incomplete.
- RT-013: Backtest returns, costs, execution timing, and walk-forward evidence are invalid.
- RT-014: Destructive utilities can erase databases or positions.
- RT-015: Database pool error path can deadlock.
- RT-016: Financial schema lacks broker identifiers, strong constraints, and lossless values.
- RT-017: Migration and backup behavior is not WAL-safe.
- RT-018: Database paths differ across runner, dashboard, Compose, Kubernetes, and backup tools.
- RT-019: No verified automated off-host backup and restore program.
- RT-020: Shared dashboard credential lacks authorization and individual attribution.
- RT-021: Kill-switch reset/start/stop lack elevated approval and durable audit.
- RT-022: Shared WebSocket token allows producer impersonation.
- RT-023: Model signing and runtime-version enforcement can fail open.
- RT-024: Logging and Git history expose sensitive trading telemetry.
- RT-025: Tests have false-green patterns, ignored suites, and critical coverage gaps.
- RT-026: CI, packaging, dependency scanning, and supply-chain controls are not release-grade.
- RT-027: Docker/Kubernetes paths bypass preflight and do not persist the actual ledger.
- RT-028: Deployment workflows contain placeholder deploy, smoke, and rollback steps.
- RT-029: Dashboard contains hardcoded/mock or process-local operational values.
- RT-030: Mobile/cloud transport, accessibility, and topology are incomplete.
- RT-031: Duplicate unfinished engines and runner modules create architecture drift.
- RT-032: Documentation claims conflict with executable readiness.
- RT-033: Uncorrelated or timed-out broker responses can be assigned to the
  wrong symbol and reach risk, stop, valuation, or persistence consumers.

# 8. Progress register

Update after each merge.

- Phase 0 CI truth gate: PR #83 merged on 2026-07-23 (`7f5de0a`)
- Phase 0 runtime-stability prerequisite: PR #81 merged on 2026-07-23 (`b7e5005`)
- PR 1: PR #82 merged on 2026-07-23 as `393f533`. Local evidence: 1,046
  passed, 5 skipped, 42% total coverage; Black, isort, Flake8, Bandit, pip
  integrity, shell syntax, YAML parsing, and diff checks passed. Hosted CI
  passed Python 3.10 through 3.12 tests, production unit/integration/performance
  matrices, lint, code quality, security, Trivy, Docker build, container
  structure, and Docker Compose containment. All 15 review threads were
  resolved.
  A two-phase review examined 11 initial findings: nine were confirmed and
  remediated, one dashboard `lsof` diagnostic was downgraded and remediated,
  and `RT_STATE_NAMESPACE` file-path isolation was safely deferred because
  changing the legacy paper kill-switch path could bypass the currently
  triggered state. The challenger then rejected three successive lifecycle
  designs until startup ordering, the operator-facing Gateway CLI, concurrent
  launcher/recovery races, and lock ownership were all fail-closed. The final
  design acquires one kernel advisory lock, transfers it to the launcher with
  an inherited descriptor, validates that descriptor before runtime work, and
  prevents Gateway, dashboard, or runner descendants from retaining it. The
  final independent review passed. The active runner already rejects backtest
  mode; separate non-paper safety state remains required before that mode may
  use shared risk components. The external Claude review action did not review
  code because its configured credential returned HTTP 401 with zero tokens;
  this infrastructure failure is recorded on PR #82 and was not treated as
  repository validation. PR #80 (`dd26ad5`, `edd0288`) is explicitly
  superseded: no commit from that branch was merged. Its 11 review threads are
  mapped one-to-one to PR 1 / PR #82, PR 6, or PR 11 in the branch disposition
  record. Separate branch requirements, rather than review threads, are
  retained for PRs 8 and 10.
- PR 1A: PR #87 merged on 2026-07-23 as
  `4cafb782cbf43ff4397f1b89b42d5f657eceea8e` from exact reviewed head
  `aa62b3e20dbd88096aa78a5875a8dc48e298f7ee`. Local focused validation
  passed 371 tests with 2 skipped; the local full suite passed 1,396 tests with
  5 skipped and 20 warnings. All repository-owned hosted checks were green,
  Codex exact-head review was clean, and the independent challenger returned
  PASS. The external Claude run `30047548445` provided no validation: its
  revoked OAuth credential failed with HTTP 401 after using zero tokens and
  incurring zero cost. PR 1A closes the incident-driven broker-correlation,
  event-time, transport-poisoning, stop-protection, and fail-closed lifecycle
  scope. It does not authorize a restart. Gate A remains closed, the trader
  remains stopped, and PR 1B read-only reconciliation is next.
- PR 1B: PR #90 merged on 2026-07-23 as
  `0d43006561071e27b217cb6d16f3c0a245a18655` from exact reviewed head
  `ae133c054721ea8ca656594053594e0ae43649d1`. The local full suite passed
  1,564 tests with 5 skipped and 20 warnings. The final strict whole-PR review
  returned PASS after 314 focused tests; the client-ID boundary review returned
  PASS after 151 focused tests. All repository-owned hosted checks passed,
  including Python 3.10 through 3.12 tests, production
  unit/integration/performance matrices, lint, security, Docker, import
  validation, Trivy, and SARIF upload. Earlier Codex reviews found cleanup and
  shared client-ID compatibility defects; both were fixed and all threads were
  resolved. The final-head Codex request could not run because the account
  reached its code-review usage limit. The external Claude action again
  provided no validation because its revoked OAuth credential returned HTTP
  401 with zero tokens and zero cost.

  The first real command was run from merged code against the stopped local
  paper runtime. It did not connect to the broker: runtime validation blocked
  because `.env` lacks `IBKR_ACCOUNT`, `IBKR_APPROVED_ACCOUNTS`, and
  `IBKR_ACCOUNT_TYPE`. The report stated `mutated_state=false` and
  `authorizes_startup=false`; independent before/after hashes of `.env`, the
  ledger and SQLite sidecars, kill-switch state and lock, bypass log, and
  trading log were unchanged. Issue #92 requires the operator to configure the
  exact paper account and dedicated reconciliation client ID locally without
  publishing the raw account number. Reconciliation remains incomplete, no
  data or safety state was corrected, and this merge does not authorize
  startup.
- PR 1C: PR #95 merged on 2026-07-24 as
  `dff4c8b597a54c614d8925565f28aa865f8ae676` from exact reviewed head
  `431513b9c7034ac2712ffb54acb58429e04281ba`. The focused package,
  import-isolation, and broker-boundary suite passed 76 tests. The full local
  suite passed 1,568 tests with 5 skipped and 20 known warnings. All
  repository-owned hosted checks passed, including Python 3.10 through 3.12,
  production unit/integration/performance matrices, lint, security,
  containers, build, BugBot, and Trivy. The prior missing-template review
  finding was fixed, regression-tested, and its only review thread was
  resolved.

  Exact-head two-phase review ran code-quality, bug, trading-safety, and style
  passes plus a verification challenger. Reviewers reproduced that the
  pre-existing empty runtime dependency metadata prevents a standalone clean
  wheel install; the challenger correctly retained it as medium PR 9 debt
  rather than copying the current mixed requirements into this prerequisite.
  The challenger also downgraded the offline no-build-isolation backend concern
  to optional low-priority test hardening because normal PEP 517 builds honor
  the `setuptools>=83.0.0` floor and supported CI/dev setup pins 83.0.0.
  Test-only Bandit `assert` and shell-free subprocess notices were filtered as
  false positives. Final two-phase verdict: PASS with no blocking finding.

  The external Claude action did not review code: its credential returned HTTP
  401 before inference with zero input tokens, zero output tokens, and zero
  cost. Final-head Codex and Cursor review requests reported usage limits;
  those unavailable reviews were recorded rather than counted as passes.
  Issue #94 closed and the source branch was deleted. PR 1C changes no runtime
  wiring or order authority. Gate A remains closed, the trader remains stopped,
  and PR 2A / issue #91 merged through PR #97 as
  `d17b0d5b4f31ab15e2a9b138cca006c0103b7276`. The dormant merge grants no
  startup or order authority.
- PR 2: Staged as dormant PR 2A (issue #91) followed by separately reviewed
  runtime-integration PRs 2B.1 through 2B.3. PR #97 merged PR 2A on 2026-07-25
  from exact reviewed head `5cf0e4ec48178fdecde6794e920d6af7c67b58f8`
  as `d17b0d5b4f31ab15e2a9b138cca006c0103b7276`. PR #104 merged PR 2B.1
  on 2026-07-25 from exact reviewed head
  `0d5585b46f8f1b495d944e24b23df5a7c01cfc2d` as
  `3ecdaa05b3352ddcd4519662b0fe957751f3fdb1`. PR #106 merged PR 2B.2
  on 2026-07-26 from exact reviewed head
  `6e2b768ea474d7e8e41037c7b3e9a3606ed00482` as
  `8328c822733b1a2358a8d6f26d368ab0819cd106`. PR 2B.3 merged through PR
  #107 as `017f43e` after exact-head hosted and local review. Its shared
  terminal-settlement readiness gate remains false.

  PR 2A contains strict exact-`Decimal` models, account/portfolio-aware
  reduce-only validation, zero-crossing and over-close rejection, deterministic
  idempotency, one-shot submission permits, dual-scope active reservations,
  crash/unknown-outcome quarantine, exact terminal reconciliation, and a
  dedicated append-only SQLite hash-chain journal. Journal initialization is
  explicit, owner-only, rejects unrelated databases and symlinked final paths,
  and binds the actual SQLite-owned native descriptor to the independently
  opened journal device/inode around reads and mutations. The descriptor proof
  fails closed outside supported GIL-enabled CPython 3.10 through 3.14 with a
  default Unix SQLite VFS. Tests also prove that importing
  `robo_trader.safety` does not import or wire any production runtime.

  Current PR evidence: the focused safety and package-boundary suite passes 103
  tests. The full repository suite passes 1,667 tests with 5 skipped and 20
  known warnings. Black, isort, Flake8, and Bandit pass for the new package and
  tests.
  Independent code, bug, trading-safety, style, and challenger reviews passed
  before GitHub review. GitHub Codex then identified three valid gaps: stale
  plan status, direct-model zero-crossing acceptance, and existing-symlink
  journal redirection. All three are remediated on the branch with regression
  coverage. Post-fix adversarial review then exposed same-schema
  swap-open-restore races in both read and write paths, a false-attribution
  weakness in process-wide descriptor enumeration, a callback self-deadlock,
  and unsafe CPython ABI assumptions. The final design compares the native
  descriptor owned by SQLite itself with an independent `O_NOFOLLOW` guardian,
  rejects unsupported interpreter/VFS builds before pointer access, and has
  focused substitution, decoy-descriptor, ABI-guard, reentrant-callback, and
  repeated concurrency coverage. A final independent review also found and
  verified the repair of a post-bind exception cleanup leak; the regression
  proves the binding map, SQLite connection, and guardian descriptor are all
  released.

  Exact-head hosted CI passed every repository-owned build, Docker,
  container-structure, Compose, lint, security, Trivy, code-quality,
  bug-detection, and Python 3.10 through 3.12 unit/integration/performance job.
  Exact-head Cursor security review found no medium, high, or critical finding,
  and all three earlier GitHub Codex threads were fixed, replied to, and
  resolved. A requested exact-head Codex rerun was unavailable because the
  account reached its code-review usage limit. The external Claude action did
  not review code because its revoked OAuth token returned HTTP 401 before
  inference with zero input tokens, zero output tokens, and zero cost. Those
  unavailable external reviews were recorded rather than counted as passes.

  This stage remains dormant and cannot authorize startup or order placement.
  Gate A remains closed and the trader remains stopped.

  PR 2B is split into PR 2B.1 (issue #100: paper runtime identity, trusted
  evidence, and startup replay), PR 2B.2 (issue #101: route paper exits and
  separate hard/soft gates), and PR 2B.3 (issue #102: exact settlement and
  crash/restart quarantine). PR 2B.1 is merged. It adds an identity-bound journal,
  read-only startup replay before any supervised process/Gateway mutation,
  one-transaction cross-portfolio allocation evidence, connected-generation
  qualified-contract lineage, and a typed producer-evidence assembly boundary.
  The coordinator has only a sealed in-memory fake submitter: no production
  order path is wired in PR 2B.1. Its broker snapshot wrappers are explicitly
  dormant integration-test scaffolding, not an operational trust source.
  PR 2B.2 replaces those wrappers with producer-owned snapshots emitted by the
  read-only broker client, binds them to an HMAC-derived exact paper-account
  scope, proves loopback/4002/read-only transport and IBC configuration, and
  binds local allocation evidence to the validated runtime ledger path,
  identity, device, and inode. Every active paper reduction is serialized
  account-wide, revalidates broker contract/transport and the complete
  cross-portfolio local allocation immediately before one-shot
  `PaperExecutor` submission, and cannot retry or fall back to another sink.
  Entry-only kill-switch, circuit-breaker, rate, and session gates remain
  separate from the hard evidence required for semantic reductions.

  The sanctioned executor is still the local synchronous `PaperExecutor`;
  IBKR remains diagnostic and read-only. Therefore the exact local
  cross-portfolio SQLite ledger is the allocation/exposure authority for this
  stage. IBKR paper-account positions and open orders describe a different
  system and remain diagnostic inputs for PR 5 reconciliation; they do not
  authorize or automatically correct a local simulator reduction. Broker state
  becomes authoritative before any future broker-write capability can exist.

  PR 2B.2 remains deliberately dormant after merge. A shared readiness gate
  defaults false and is enforced independently by the runner, paper-order
  runtime, gateway startup, entry serialization, and reduction submission.
  PR 2B.3 closes the previously strict-XFAIL settlement and protective-quote
  lineage gap in code, but the gate remains false through hosted review, merge,
  exact-state bootstrap design, and the remaining cumulative Gate-A work.

  Final local PR 2B.2 implementation evidence: the full repository suite passed
  2,001 tests with 5 expected skips, 1 strict settlement XFAIL, and 20 known
  warnings. Black, isort, Flake8, compilation, diff checks, targeted Bandit
  scanning, and secret review passed. Phase-one code, trading, and bug reviews
  identified an overlong stop reference, same-scope alternate-journal
  acceptance, and unsafe SQLite pool replacement; all were reproduced, fixed,
  and covered by regressions. The phase-two challenger confirmed those fixes,
  found no new blocker, and classified terminal settlement and protective-price
  lineage as high-severity PR 2B.3 pre-activation requirements. Durable detail
  is recorded in
  `docs/reviews/PR2B2_LOCAL_REVIEW_EVIDENCE_2026-07-25.md`. Exact-head hosted
  CI, security review, and Codex review later passed before PR #106 merged;
  the unavailable Claude review was recorded and was not counted as a pass.
  GitHub Codex subsequently found relative-ledger-path and persistent-reconnect
  executor-identity defects; both were reproduced and fixed end to end with
  regressions before the final 2,001-test run.
  Exact-head hosted testing also revealed that three duplicated full-suite
  workflows could remain pending indefinitely on a Linux pytest hang. Those
  jobs now fail closed with bounded job and command timeouts plus periodic
  Python thread dumps; a workflow-policy regression preserves the controls.

  PR 2B.3 exact-head local implementation evidence at `14f6b55`: every
  successful local-paper reduction commits the exact trade, signed position,
  cash, realized P&L, daily P&L, day-start baseline/date, exact cost/mark state,
  protective-quote payload, and terminal outbox receipt in one SQLite
  transaction before runtime projection and journal release. Producer-owned
  quote lineage is revalidated at submission. Long and short partial/full
  exits bridge the prior exact mark to the authenticated protective mark before
  replacing removed unrealized exposure with realized fill P&L. Float-only
  compatibility updates cannot mint or overwrite exact settlement authority.
  Restart restores exact marks and same-day risk state or performs a persisted,
  exact new-UTC-day rollover; missing, future, stale, or tampered state fails
  closed.

  Filled and zero-fill crash-after-commit outcomes have a stopped-system,
  confirmation-gated offline recovery path. It verifies the linked trade,
  position and aggregate, exact basis and marks, account state, baseline/date,
  timestamps, settlement source lineage, database identity, and quote payload
  from one read-only snapshot before appending one terminal journal event. It
  never rewrites the financial ledger. Forged, partial, ambiguous, outbox-only,
  or tampered evidence remains quarantined.

  The final full local suite passed 2,240 tests with 5 expected skips and 20
  known warnings. The final settlement/routing/security matrix passed 375 tests
  with 2 expected skips. Black, isort, full Flake8, compilation, dependency,
  diff, targeted Bandit, targeted new-module mypy, and secret-pattern checks
  passed. Two-phase code, bug, trading-safety, style, and challenger review
  reproduced and closed the daily-risk restart, offline projection,
  crash-recovery, quote-lineage, signed short/cover, zero-fill, float-authority,
  and changing-mark failures. Durable detail is recorded in
  `docs/reviews/PR2B3_LOCAL_REVIEW_EVIDENCE_2026-07-26.md`.

  This is not launch evidence. PRs 3 through 5 code slices are now merged, but
  the exact-state bootstrap for existing ledgers, strict FIFO lot accounting,
  PR 4/PR 5 production integration and operational evidence, the Gate-A subset
  of PR 7, reconciliation/restore drills, and a clean ordinary preflight remain
  required before any supervised paper start. `PAPER_TERMINAL_SETTLEMENT_READY`
  remains false and IBKR remains read-only.

  PR 2B.1 schema v2 intentionally fails closed on a schema-v1 PR 2A journal.
  No operational v1 journal is known and none exists in the inspected runtime,
  so this is not a dormant-PR blocker. Any discovered v1 file must be preserved;
  a later migration must use an explicit backup/copy-and-verify procedure and
  must never rebind or rewrite the only copy. PR 2B.3 and the PRs 3-through-5
  code slices are merged; their remaining operational evidence/integration and
  the relevant PR 7 work remain outstanding.

  Final PR 2B.1 evidence: after two late GitHub Codex findings, the configured
  journal path preserves its lexical final component so the journal can reject
  symlink substitution itself, and Python dependencies are bootstrapped before
  journal verification without importing side-effectful RoboTrader runtime
  modules. Dedicated tests prove same-identity symlink rejection, fresh and
  incomplete virtual-environment bootstrap, and fail-closed handling of a
  partially successful dependency installation. Two independent local agent
  reviews returned PASS on the repaired exact tree; their durable summaries are
  recorded in `docs/reviews/PR2B1_LOCAL_REVIEW_EVIDENCE_2026-07-25.md`. A
  separate trading-safety review also passed for the symlink boundary. The
  focused late-fix suite passed 77 tests; the final full local suite passed
  1,801 tests with 5 expected skips and 20 known warnings. Black, isort,
  Flake8, Bash syntax, and diff checks passed.

  Twenty-six exact-head checks succeeded: 24 GitHub Actions jobs, GitHub
  Advanced Security Trivy, and the external Cursor vulnerability scan. Eleven
  non-applicable jobs were skipped and Cursor Bugbot was neutral. Both late
  Codex review threads were replied to and resolved. The Claude-backed review
  workflow failed only because its revoked OAuth token returned HTTP 401 before
  inference with zero input or output tokens; it produced no code verdict and
  was not counted as a pass. The merge grants no order authority or startup
  approval. Gate A remains closed and the trader remains stopped.
- PR 3: Merged through PR #111 as `8596920`. The canonical market-data
  contract and adversarial tests are present; cumulative Gate-A operational
  evidence remains required.
- PR 4: The exact-state bootstrap code slice merged through PR #112 as
  `bccb96b`, and dormant FIFO PR4A merged through PR #120. PR4B is under review
  as PR #123 at exact head `318532e`. PR4C transactional runtime FIFO settlement
  is implemented and awaiting independent review. Stopped-system operator
  application, WAL-safe clean-room backup/restore, and current reconciliation
  evidence remain open; no PR4 slice authorizes startup.
- PR 5: Reconciliation domain and runtime evidence/service foundations merged
  through PR #113 (`6176931`) and PR #115 (`1ae6480`). Production
  startup/reconnect/periodic integration and a current reviewed read-only
  broker-versus-ledger reconciliation remain open.
- PR 6: Not started
- PR 7: Staged. Dormant entry-risk PR #114, dormant filled-notional PR #116,
  and paper-path containment PR #117 are under review. All must remain within
  the sole Gate-A staging exception above; integration and post-PR6 strategy
  readiness work remain open. PR #117 is explicitly non-authorizing for new
  exposure: Python code in one interpreter is one trust domain, so private
  attributes and closure cells are not accepted as security isolation. The
  staging slice publishes no baseline entry handle or intent, exports no BUY
  capability issuer, and terminal baseline submission fails closed. Only
  semantic paper reductions retain capability-backed execution until the
  integrated risk-admission boundary is independently enforceable.
- PR 8: Not started
- PR 9: Not started
- PR 10: Not started
- PR 11: Not started
- PR 12: Not started
- PR 13: Not started
- PR 14: Not started
- PR 15: Not started

# 9. Standard PR evidence checklist

Every safety-relevant PR must include:

- problem identifiers addressed;
- explicit non-goals;
- design or threat model where appropriate;
- migration and rollback behavior;
- unit, integration, and failure-injection tests;
- exact commands and results;
- database backup and restore implications;
- paper/live separation implications;
- security and credential implications;
- operator/dashboard implications;
- updated runbook and documentation;
- before/after screenshots for operator-facing changes;
- reviewer sign-off from trading safety and security/data integrity;
- confirmation that no user trading data was deleted.

# 10. Recommended verification commands

Commands must be adjusted as the CI contract evolves, but the starting set is:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/ -q --cov=robo_trader --cov-report=term
.venv/bin/python -m mypy robo_trader
.venv/bin/python -m black --check .
.venv/bin/python -m flake8
.venv/bin/python -m bandit -r robo_trader scripts -ll
.venv/bin/python -m pip check
python3 scripts/preflight_check.py --verbose
python3 scripts/gateway_manager.py status
lsof -nP -iTCP:4002 -sTCP:LISTEN
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT
```

Never use a test command that can access the production database unless test isolation has been verified for that exact command.

# 11. Final launch decision template

The launch approver must answer all of the following with evidence:

1. Which exact build, configuration fingerprint, model artifacts, account, database, and topology are being armed?
2. Is broker reconciliation clean now?
3. Are broker-native protective orders verified?
4. Are all risk limits loaded from the canonical configuration and displayed correctly?
5. Can reduce-only exits execute while entry trading is blocked?
6. Did backup restoration and manual cancel/flatten drills pass?
7. Are human alerts working?
8. Did the required paper/shadow soak pass without unexplained drift?
9. Are identity, authorization, WebSocket, and audit controls active?
10. Is the live canary restricted to the approved account, symbols, position count, notional, and expiry?

If any answer is no, unknown, stale, or based only on documentation rather than observed evidence, live trading remains disabled.

### September 13 isolated integration evidence

Local development integration includes PRs #122, #124, #116, and #121 plus
reviewed upgrade-binding fixes and a dormant terminal-fill replay adapter. The
combined suite passes 3501 tests (4 skipped). Detailed inputs, limitations,
restore drill, and remote-host observations are recorded in
`docs/PAPER_READINESS_EXECUTION_2026-09-13.md`. This is development evidence;
Gate A remains closed and no operational startup or bootstrap is approved.

September 13 follow-up: the dormant exact entry contract now enforces explicit
optional order notional policy, maximum occupied/pending account position slots,
duplicate-symbol rejection, and durable cooldown boundaries. Unknown admission
state fails closed, and new evidence is sealed and revalidated. Focused tests:
208 passed. Independent review: no actionable defect. Full rerun: 3536 passed,
4 skipped; a preceding migration-test outer subprocess timeout and successful
isolated rerun are retained in the execution record. Runtime evidence production,
account leverage/pending exposure, BUY settlement, and operational Gate A remain
open. No entry authority or startup gate was enabled.

September 13 account-risk follow-up: dormant entry sizing now includes exact
account leverage and explicit pending symbol/sector/portfolio/account/cash/
buying-power/daily commitments. Unknown values fail closed and all added values
are sealed. Combined contract and durable-risk suites: 408 passed. Independent
review found no actionable defect. Coherent runtime snapshots and reservations
remain unimplemented; Gate A remains closed.

September 13 ledger-evidence follow-up: added a dormant coherent read-only
account snapshot reconstructed from authenticated bootstraps and terminal
receipts, with FIFO/commission/link and mutable-state verification in one read
transaction. Persistent portfolio reads cannot be redirected by temporary
SQLite tables. Related regressions: 166 passed; broader safety/security/risk
run before final schema qualification: 1003 passed, 4 existing skips.
This partially implements the runtime evidence producer; quote valuation,
reservations, startup replay, baseline BUY settlement, and gateway wiring
remain open. Gate A and paper entry authority remain closed.

September 13 gateway valuation follow-up: portfolio entry serialization now
collects coherent account ledger evidence under its shared order lock, requests
marks for all held symbols, and validates task-owned exact equity/gross
valuation before yielding. Inactive balances remain included independently of
execution registration. Each valuation access rechecks producer quotes,
contract/generation identity and wall/monotonic freshness. Related regressions:
121 passed; independent final valuation review: 12 passed, no actionable defect.
This connects part of the runtime evidence producer; it does not yet replace
legacy sizing or implement reservations, full entry evidence, final database/
reconciliation checks, replay startup, or BUY settlement. Gate A remains closed.
Broader post-change safety/security/risk/entry-contract/runner-event-time
verification: 1117 passed, 4 existing skips; details in the execution record.

September 13 daily-risk integration follow-up: the gateway can prepare a supplied
exact daily-accounting adapter by validating complete bootstrap scope, replaying
all terminal receipts and authenticating each scope, including empty ones.
Configured terminal fills are ingested before journal release inside drained
completion; failures preserve outbox/reservation evidence and quarantine.
Task-owned daily reads revalidate valuation after awaiting independent authority.
Related regressions: 83 passed. Production verifier construction and runner
injection remain open, as do exact sizing, reservations and BUY settlement.
No entry authority or paper readiness gate was enabled.
Broader post-change safety/security/risk/runner/gateway verification: 1216 passed,
4 existing skips. Independent final failure/binding review found no actionable
defect; formatting, changed-file lint and whitespace checks pass.

September 13 history-completeness correction: successful terminal replay does
not prove pre-bootstrap daily gross executions. Owned snapshots now seal each
portfolio's authenticated bootstrap timestamp; entry daily totals require a
later New York date, one explicit read timestamp, and unchanged date after the
await. Bootstrap-day support would need authenticated historical execution
completeness evidence; none is inferred from cash or legacy float trades.
Related regressions: 111 passed; independent history review: 9 passed. Gate A
remains closed, with configuration/reservations/full admission and BUY settlement
still unimplemented.
Broader post-change verification: 1224 passed, 4 existing skips. Formatting,
changed-file lint and whitespace checks pass; operational readiness remains closed.

September 13 admission-history follow-up: gateway valuation now provides exact
account-wide symbol gross, explicit held-symbol presence and durable ten-minute
cooldown evidence from verified terminal fills, bounded conservatively by the
latest account bootstrap. Zero fills do not extend cooldown and signed holdings
do not net away gross. Related regressions: 351 passed; independent review:
4 targeted tests passed. Pending reservation evidence and final risk-contract
consumption remain open; no entry authority was enabled.
Broader post-change verification: 1228 passed, 4 existing skips; formatting,
changed-file lint and whitespace checks pass. Gate A remains closed.

September 13 configuration follow-up: Config now builds exact non-authorizing
entry limits using existing risk settings and three explicit additional
portfolio/liquidity settings. Partial/invalid explicit policies fail loading;
missing policy grants no authority. Freshness cannot exceed five seconds.
Focused verification: 349 passed; independent review: 18 passed. Broader
risk/security/entry/gateway/routing/settlement verification: 1012 passed,
4 existing skips. Formatting, changed-file lint and whitespace checks pass.
Runner overrides, reservations, final evidence/contract consumption, production
verifier injection and BUY settlement remain open. Gate A remains closed.

September 13 runner-policy follow-up: current exact limits now resolve a unique
active portfolio, its position/slot overrides, runner order/daily caps and the
stricter correlation setting without changing shared configuration. Missing or
invalid selection/overrides fail closed. Focused regressions: 328 passed;
independent review: 14 passed. Final serialized admission must consume this
resolver; pending reservations, full evidence and BUY settlement remain open.
Gate A remains closed.

Broader risk/security/entry/gateway/routing/settlement verification: 1026 passed,
4 existing skips, 4 warnings in 49.92 seconds (work/runner-entry-policy-safety.log).
Black, changed-file Flake8 and whitespace checks pass. Paper readiness remains false.

September 13 durable-capacity follow-up: a dormant risk adapter now consumes an
approved decision into a separate reservation-only journal event after a bound,
unchanged-head transaction check. It issues no permit. Replay retains capacity
through expiry/restart and verifies semantic payloads. Startup/bootstrap block
pending entries; operator status reports them; reduction-only recovery cannot
release them. Related verification: 590 passed; independent review: 19 reservation
and 4 dormancy tests passed. Pending aggregation, full head-bound evidence and
authenticated terminal release/BUY settlement remain open. Old readers reject the
new event type: retain compatible readers and preserve journal history. Gate A
remains closed.

Full post-change repository suite: 3729 passed, 4 existing skips, 19 warnings in
303.96 seconds. Formatting, changed-file lint and whitespace checks pass. Final
containment inspection confirms readiness false, BUY rejection, and no production
adapter invocation. Operational launch remains unverified and blocked by the
outstanding gate evidence; no remote services or data were changed.

September 14 pending-read follow-up: exact unresolved principal totals now feed a
task-owned gateway read model with an unchanged bound journal head. Account-wide
symbol/sector/gross/buying-power and portfolio cash/daily totals include inactive
reservations. Held/pending symbols share one slot. Reads drain on cancellation;
missing ledger coverage and unresolved reductions block. Related verification:
761 passed; independent review: 9 passed. Atomic final head comparison, complete
admission evidence, fee/price envelopes and authenticated terminal release/BUY
settlement remain open. Gate A remains closed.

Broader risk/safety/security/gateway/routing/settlement/status/bootstrap checks:
1294 passed, 4 existing skips, 1 warning in 28.52 seconds
(work/pending-capacity-safety.log). Black, changed-file Flake8 and whitespace
checks pass. This is component integration evidence; paper readiness remains false.

September 14 execution-price follow-up: a reproduced ambient-Decimal rounding
error in an authorized paper fill is fixed with shared exact rational slippage
and one half-even tick rounding. The model retains explicit zero commission.
The owning gateway context can now read current registered-executor pricing and
a conservative ceiling without modifying broker quote evidence. Related tests:
164 passed; independent review: 47 passed. Risk sizing/reservation and final
submission still must bind the estimate; no BUY issuer or launch gate changed.

Broader risk/safety/security/pricing/gateway/submitter/routing/settlement checks:
1304 passed, 4 existing skips, 1 warning in 26.92 seconds
(work/paper-cost-safety.log). Black, changed-file Flake8 and whitespace checks
pass. Paper readiness remains false; operational launch and profitability are unverified.

September 14 ceiling-consumption follow-up: the exact risk contract now requires
owned positive execution-price ceiling evidence, rejects missing/understated
values, and uses it for quantity, approved principal and all-capacity postconditions.
The reservation journal records that principal while the source quote remains
unchanged. Related regressions: 322 passed; independent review: 301 passed;
broader safety/security/risk/execution/settlement checks: 1309 passed, 4 existing
skips. Formatting, changed-file lint and whitespace checks pass. Final gateway
policy binding, complete admission and terminal release/BUY settlement remain
open. No startup or BUY authority was enabled; Gate A remains closed.


## September 14: exact sector exposure under explicit classification policy

Added a task-owned gateway sector read over the verified account ledger, current
quotes and head-pinned pending journal. It requires an immutable tuple of unique
symbol/sector classifications covering the requested symbol, every held symbol
and every pending symbol. Missing or ambiguous classifications and conflicts with
persisted pending sectors fail closed. Held gross sums absolute position values
across all portfolios, including retired portfolios, without long/short netting;
pending sector principal is account-wide. The returned policy is canonicalized
for later admission binding. These are explicit policy groupings, not claims of
an authenticated external sector taxonomy.

Validation: 37 related tests passed; independent review found no defect and
independently passed those 37 tests. Added the reviewer's suggested retired-short
regression afterward: all 8 sector tests passed. Broader risk/safety/security run:
812 passed, 4 existing skips, 1 warning in 19.92 seconds (before that final added
test). Log: work/sector-safety.log in the task workspace. Changed-file Black,
Flake8 and whitespace checks passed.

The gateway read remains non-authorizing. Final admission must bind the same
classification policy and revalidate ledger/journal state. Complete evidence
assembly, terminal BUY settlement/release and operational launch gates remain
unfinished. Readiness remains false, BUY submission disabled, and no user trading
data or running services were changed.


## September 14: canonical intraday correlation statistics

Implemented a non-authorizing canonical-bar correlation calculation for the
remaining runtime market-evidence producer. It requires explicit exact contract
coverage, one transport generation, matching timeframe/session/adjustment policy,
fresh retrievals and completed aligned return windows. The caller must supply
the window length (2..256 returns); this does not select production policy.
In-session gaps fail. Overnight/session-boundary returns and forming bars are
excluded. Missing histories, flat return series, duplicate contracts and window
mismatches fail rather than producing zero correlation or intersected samples.

Pearson arithmetic uses exact rational returns/covariance/variance and an integer
square-root bound. Absolute correlation rounds upward to 18 decimal places,
independent of the ambient Decimal context. A deterministic source-data version
binds selected return windows, contract identities and retrieval metadata.
The zero-held-account case requires explicit empty held coverage.

Validation: 41 canonical-correlation/market-data tests passed, including exact
half-correlation, negative perfect correlation, a proven irrational upper bound,
low precision, stale/mutated inputs, unfinished bars, session exclusion, gaps and
generation/policy mismatches. Independent reviewer found no numerical defect and
passed the initial 9 tests; suggested boundary tests were added afterward.
Broader risk/safety/security/canonical-data run: 854 passed, 4 existing skips,
1 warning in 20.60 seconds. Log: task-workspace work/correlation-safety.log.
Changed-file Black, Flake8 and whitespace checks pass.

This function validates structure and calculates statistics; it does not prove
batch provenance or mint risk evidence. Next integration must authenticate the
actual transport-owned batches and bind the configured window to admission.
The current transport supports intraday bars only; complete daily liquidity
history still needs an explicit producer/session-completeness design. SSH to
oliver@blackm5mbp timed out again on port 22. Entry admission, BUY settlement,
operational backup/bootstrap, actual paper-account identification and supervised
launch remain incomplete. Readiness remains false; no services/data were changed.


## September 14: transport ownership for execution bar batches

The subprocess client now retains the latest exact canonical batch, producing
worker object and full-content fingerprint per symbol. Publication follows
validated historical response and current lineage retrieval under the lifecycle
lock. Validation uses the existing generation -> connection -> lineage lock
order and requires the connected, unpoisoned producer generation, current lineage,
exact object identity and unchanged contract/bar content. A copied or modified
batch, refresh (including raw-history refresh), disconnected/poisoned generation,
or replaced worker is rejected. Malformed object fields become validation errors.

Runner execution lookup now requires this check from its exact subprocess client
before reading the batch contract. Legacy routing fixtures explicitly establish
synthetic client ownership; no production fallback accepts their former unowned
batches. A final malformed-contract regression caught and fixed an early symbol
read in the runner. Independent review found no concrete defect and passed the
initial 8 ownership tests. Related tests: 198 passed before the final runner
malformation regression. Final broad risk/safety/security/lineage/canonical-data/
routing/event-time suite: 1028 passed, 4 existing skips, 1 warning in 19.73 seconds.
Changed-file Black, Flake8 and whitespace checks pass. Task-workspace logs:
work/ownership-related.log, work/ownership-safety.log; reproduced failures in
work/canonical-ownership-red.log, work/ownership-malformed-red.log and
work/ownership-runner-malformed-red.log. An earlier fixture-incompatible routing
run was explicitly interrupted, then rerun after synthetic ownership was added.

This establishes in-process source ownership, not risk approval or freshness.
Consumers still require age, complete account coverage and window-policy checks.
The initial retention policy was one batch per symbol; the later independent-
window change below supersedes that limitation. The fingerprint still scans the
full batch under state locks.
Next work is to assemble authenticated correlation and liquidity evidence with
exact gateway account inputs and bind final admission/settlement. Paper readiness
remains false; no BUY authority, service launch or user-data mutation occurred.


## September 14: gateway correlation bound to account evidence

Added the task-owned gateway correlation read. Expected contracts come from the
owned account ledger/current marks plus every pending entry reservation; callers
cannot narrow coverage by omitting a portfolio or pending symbol. Conflicting
pending identities fail closed. Every supplied canonical batch must be the exact
unchanged current batch owned by the gateway's exact subprocess client. The read
performs no broker I/O while holding the account entry gate.

After calculation it replays the pinned journal head, reauthenticates all source
batches, revalidates held valuation and checks both historical retrieval age and
the completed return-window end against the timeframe freshness limit. The pure
correlation result now carries that window end. A clock-advance regression
reproduced history expiring during the final await despite still-fresh account
marks; the final history check fixes that boundary.

Validation: all 10 gateway integration tests passed, including complete held and
pending comparisons, missing pending history, copied batches, changed journal,
poisoned generation, cross-task use, source invalidation during an await and
history expiry. Related checks: 53 passed before the final positive-pending test.
Independent reviewer found no actionable defect and passed 38 focused tests.
Final broader risk/safety/security/lineage/canonical-data/routing/event-time run:
1038 passed, 4 existing skips, 1 warning in 20.55 seconds. Black, changed-file
Flake8 and whitespace checks pass. Task-workspace logs:
work/gateway-correlation-red.log, work/gateway-correlation-expiry-red.log,
work/gateway-correlation-safety.log.

This returns non-authorizing statistics. Final admission must bind the selected
window, source version and current journal/ledger state. Historical collection
must happen before entry serialization. The later independent-window change
below resolves the initial one-batch-per-symbol retention limitation.
The existing CORRELATION_LOOKBACK_DAYS setting is a days policy and must not be
silently treated as the new intraday return count. Collection/window policy,
complete daily liquidity evidence, final admission and BUY settlement remain
unfinished. Readiness is false; no services or user data were changed.


## September 14: retain independent authenticated history windows

Removed the one-batch-per-symbol limitation from source ownership. The client
now retains one latest batch per symbol/timeframe/session policy. Refreshing one
window supersedes only its matching TRADES batch; independent windows retain
their original source timestamps and fingerprints. A matching response that
fails canonicalization still invalidates the old matching batch. Same-timeframe
requests with different history durations still supersede each other.

The record also retains the full qualified contract identity. Any change to that
identity invalidates every window for the symbol, including trading-class changes
that are not present in the canonical bar contract. Changing the identity back
cannot revive old records. Publishing a new batch still requires the exact
current retrieval/broker timestamps; retaining another window never relabels its
age. Existing generation, connection, object-identity and mutation checks remain.

Validation: reproduced the former cross-window invalidation in two failing tests.
76 focused ownership/gateway/lineage/canonical-data tests pass, including separate
regular/extended windows, unchanged source timestamps, raw matching refresh,
canonicalization failure, identity changes and identity round trips. Independent
review found no actionable defect and passed 26 focused tests; the suggested
failed-refresh regression is included. Final broader risk/safety/security/
lineage/canonical-data/routing/event-time run: 1044 passed, 4 existing skips,
1 warning in 20.37 seconds. Changed-file Black, Flake8 and whitespace checks pass.
Task-workspace logs: work/canonical-windows-red.log and
work/canonical-windows-safety.log.

This supersedes the earlier one-batch-per-symbol limitation. It supports
multi-timeframe collection without discarding unrelated current evidence but
adds no order authority. Configured history collection/window binding, daily
liquidity evidence, final risk admission/BUY settlement and operational launch
gates remain open. Paper readiness is false; no services/data were changed.


## September 14: verified regular-session calendar and complete-session coverage

Corrected the day-after-Thanksgiving early-close calculation, the January 9,
2025 Carter closure, Juneteenth effective year and Saturday New Year handling.
Added versioned UTC regular-session bounds for 2024–2028, verified against
published NYSE schedules. All dates in those five years are compared with
independently transcribed holiday/early-close sets. Sources and scope are in
`docs/market-calendar.md`.

Added `completed_regular_sessions()`: the latest requested completed scheduled
sessions must have every aligned regular-session bar, including opening and
closing intervals. Missing days, gaps, duplicates, partial coverage, stale
retrieval, or a source predating the latest completed close fail. A current
partial session does not replace a completed one. This proves structural
coverage only; transport ownership and volume units remain separate obligations.

Validation: initial calendar regression run failed 13 tests; the missing coverage
module also failed collection before implementation. Final focused calendar and
coverage run: 31 passed. Independent review found no actionable defect and passed
26 tests before the five-year matrix was added. Broader risk/safety/security/
preflight/canonical-data/event-time run: 1172 passed, 4 skipped, 1 warning.
Changed-file Black, Flake8 and whitespace checks pass.

Full repository run reached a terminal result: 3837 passed, 4 skipped, 19 warnings,
1 failed in 312.46 seconds. Failure:
`test_real_worker_child_attests_isolated_policy_without_gateway[test-True]`
timed out awaiting the real isolated worker health command after five seconds.
The entire worker safety module rerun passed all 78 tests in 2.58 seconds without
code or timeout changes. This does not establish the cause or convert the full
run to a pass; full-suite verification remains open. The failed worker debug
file was no longer present when inspected. Evidence: task-workspace
`work/session-calendar-repository.log`, `work/session-calendar-worker-recheck.log`,
`work/session-calendar-safety.log`, `work/calendar-red.log` and
`work/complete-sessions-red.log`.

The existing SSH diagnostic terminated with port-22 timeout. No remote service,
configuration or user data was changed. The actual paper account identifier,
verified operational backup/restore and other launch gates remain pending.

IBKR's official Historical Volume Scaling documentation says historical volume
may represent shares or lots depending on a Gateway API setting. No verified
unit is currently bound to the source contract. Inspection additionally confirmed
that the worker calls `int(bar.volume)` before canonical validation, concealing
fractional values. Exact transport preservation/validation and authenticated unit
proof are required next; neither a guessed multiplier nor caller-provided unit
text is sufficient. Dollar-liquidity production, final entry admission and BUY
terminal settlement remain unfinished. Readiness stays false and no order
authority was enabled.


## September 14: reject historical-volume truncation at the worker boundary

Replaced unconditional `int(bar.volume)` with explicit finite, nonnegative,
integral numeric validation before serialization. Valid int/Decimal values retain
all digits even under low ambient decimal precision; floats use their exact
binary value rather than a rounded decimal display. Booleans, negative values,
fractions and nonfinite values return an error instead of a misleading bar.
This preserves the current integer canonical contract; fractional-volume support
and verified source units remain unfinished, not silently approximated.

Initial regression tests reproduced five failures. Independent review found a
large integral float display-rounding case, reproduced in an additional failing
test and fixed using Decimal.from_float. Final combined transport, coverage,
worker safety, lineage, source ownership and canonical-market-data verification:
217 passed, 1 existing warning in 5.57 seconds. Changed-file Black, Flake8 and
whitespace checks pass. Task-workspace logs: work/volume-validation-red.log,
work/volume-float-red.log and work/volume-validation-final.log.

No source unit is inferred from integrality. Actual Gateway unit evidence,
liquidity production, final entry admission/settlement and operational launch
requirements remain open. The prior full-suite worker timeout remains recorded;
this focused verification is not a full-suite pass. Readiness is still false.
No service, operational configuration or user data was changed; edits are local
and uncommitted.


## September 14: wire-volume precision and settings API investigation

A direct invocation of the installed ib_async 2.0.1 Decoder.historicalData with
synthetic protocol fields reproduced precision loss before the worker boundary:
wire `1000000000000000001` becomes float `1e18`, and wire
`10.0000000000000001` becomes float `10.0`. The current worker then admits those
rounded integral values. Its preceding fix correctly validates the object it
receives, but cannot claim exact wire preservation. Evidence is saved in the
task workspace at `work/historical-decoder-evidence.log`. No broker was contacted.

Next implementation must preserve decimal volume before the library float
conversion, bind that decoder to the actual worker client, test real protocol
fields through decoding/serialization/canonicalization, and support fractional
source values without converting them into shares by assumption. Do not alter
site-packages as the solution. The existing integer canonical contract must be
explicitly revised/versioned before fractional values become admitted evidence.

Official IBKR historical-volume documentation confirms that shares versus lots
is controlled by the Gateway API checkbox. New Setting Management documentation
introduces configuration requests via reqConfigProtoBuf (TWS API 10.44 family).
The installed client advertises protocol versions 157–178 and has no reqConfig
implementation. This is a potential migration/research path, not a verified
available unit attestation. Documentation says modifying settings requires
manually disabling read-only; no such change is authorized or performed. Read
availability and the precise configuration response field still need proof.
Sources:
- https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/historical-volume-scaling
- https://www.interactivebrokers.com/docs/tws-api/doc/setting-management/introduction
- https://www.interactivebrokers.com/docs/tws-api/doc/setting-management/request-configuration

A fresh full-suite verification was started after the prior run reached terminal
failure and the worker patch was validated. It remains in progress under exec
session 77821, logging to `work/volume-repository.log`; poll that handle rather
than launching another copy. No runtime/test files were edited during this run.
Paper readiness and operational gates remain unchanged.


## September 14: decoder integration and malformed-response failure path

Inspected the actual worker construction and pinned library callbacks. The worker
creates a fresh IB object before connecting, permitting a repository-owned IB/
Decoder subclass to preserve raw volume without changing site-packages. Message
17 dispatch binds historicalData during Decoder construction; replacing only an
instance method afterward would not replace the bound handler.

A second relevant finding changes the implementation plan: Decoder.interpret
logs/swallow exceptions, while reqHistoricalDataAsync catches its timeout,
clears bars and returns an empty list. Therefore malformed exact-decoder input
must explicitly fail its matching request future and clean retained results,
rather than relying on timeout to report failure. The implementation plan is in
task-workspace work/exact-volume-decoder-plan.md. Tests must exercise real
Wrapper futures and raw-message dispatch, atomic multi-bar validation, exact
request ownership and worker serialization. This is a plan based on inspected
code, not an implemented or verified decoder fix.

The full-suite session 77821 was polled and remains live. Continue polling the
same handle; no duplicate run was launched. Runtime/test files remain unchanged
while that verification runs. Readiness is still false.


## September 14: exact wire-volume decoder installed in the worker

The preceding full repository verification completed successfully: 3847 passed,
4 existing skips, 19 warnings in 438.92 seconds (work/volume-repository.log).
This supersedes the prior timeout as the latest full-suite result for the code
before this decoder installation; it is not a full-suite result for this patch.

Added repository-owned ExactHistoricalDecoder and ExactHistoricalIB. The worker
now constructs this IB subclass before connection; handshake version updates
retain the installed decoder. Message 17 preserves volume as Decimal before any
float conversion. Complete count/shape/row validation occurs before publication.
Malformed values fail the matching real wrapper future and clear retained results,
without waiting for the library's empty-result timeout. Shares/lots remain unknown;
no unit conversion is performed. Streaming historical updates are unchanged and
are not requested by this worker path.

Review found an atomicity issue with malformed dates, because the wrapper parses
dates during callbacks. Prevalidation now checks every date before publishing any
row. Independent direct reproduction confirmed zero published rows, an explicitly
failed future and cleared results for a bad second date. Real-wrapper tests cover
precision, fractional residues, nonfinite/negative values, malformed/truncated
multi-bar messages, cleanup and isolation of an unrelated request. Worker binding
is explicitly tested. A protocol-to-worker-to-canonical test preserves
1000000000000000001 exactly and rejects 10.0000000000000001 as fractional instead
of silently admitting 10 under the current integer canonical contract.

Validation: missing-module regression collection failed before installation;
original decoder precision failures remain in historical-decoder-evidence.log.
Final decoder/transport/worker-safety/canonical suite: 179 passed in 5.84 seconds.
Changed-file Black, Flake8 and whitespace checks pass. Logs in task workspace:
work/exact-decoder-red.log and work/exact-decoder-validation.log. Draft review and
standalone draft tests preceded installation while the full suite was running;
no runtime/test files were changed until that run terminated.

Fractional canonical storage/versioning and independently verified Gateway volume
units are still required before liquidity evidence. Final admission, BUY settlement
and operational launch gates remain open. Readiness is false; no service or user
data was changed. All changes remain local and uncommitted.


## September 14: fractional-volume persistence investigation and implementation plan

Confirmed that canonical_market_data accepts only schema version 1 and declares
volume INTEGER. An isolated in-memory SQLite reproduction demonstrates that even
a string-bound tiny fraction or large fractional quantity is rounded by affinity;
a TEXT column preserves the original decimal string. No user database was opened
or changed. Evidence: work/fractional-volume-storage-evidence.log.

Mapped both reader paths and found schema version is omitted from their selected
series identity (valid while only version 1 exists, insufficient for version 2).
Added docs/FRACTIONAL_VOLUME_IMPLEMENTATION.md with the end-to-end versioning,
separate-table preservation, writer conflict handling, deterministic reader
selection, analytical projection, source ownership and acceptance-test plan.
Implementation remains pending; no fractional quantity or source unit is newly
admitted. This investigation changes the next action from a serializer-only edit
to a coordinated contract/storage/reader change. Paper readiness stays false.


## September 14: opt-in exact-decimal canonical contract

Added explicit schema_version=2 canonicalization with Decimal volume, exact JSON
text storage records and unknown-only volume_unit metadata. Floats are rejected
as v2 input because their wire precision is already lost. Analytical DataFrames
use an explicit float projection while retaining the original canonical object.
Version-1 integer objects, storage record keys and default production version
remain unchanged. The source-ownership fingerprint already covers all contract
and bar dataclass fields, including the new unit field and exact Decimal value.

The representation is bounded to 128 fixed coefficient digits and exponent
-128..128. Independent review found an accepted 1e128 value serialized beyond
that limit. The input bound now includes expanded positive-exponent digits;
1e127 roundtrips and 1e128 is rejected. This prevents the serializer from emitting
a quantity that its storage validator cannot read. No context-sensitive decimal
normalization is used.

Validation: 12 initial tests failed before implementation. Initial focused run
55 passed. Broader risk/safety/security/canonical/decoder/routing/event-time run
1058 passed, 4 existing skips, 1 warning in 69.11 seconds. That run used the
pre-boundary-fix imported module; final contract/canonical verification after the
review fix passed 39 tests in 3.26 seconds. Black, Flake8 and whitespace checks
pass. Evidence in task workspace: work/fractional-contract-red.log,
work/fractional-contract-safety.log, work/fractional-contract-final.log.

This completes the contract portion only. V2 table, writer routing and both
readers remain to implement per docs/FRACTIONAL_VOLUME_IMPLEMENTATION.md. The
existing SQLite schema rejects v2 inserts; production canonicalization and worker
serialization remain v1. Verified Gateway volume units and liquidity evidence
remain missing. No migration, service launch or user-data changes occurred.
Readiness remains false and all edits are local/uncommitted.


## September 14: exact-decimal version-2 storage and atomic writer

Added canonical_market_data_v2 as a separate table with TEXT volume and explicit
unknown volume_unit. The existing version-1 table/schema remain intact. Validated
single-version batches route through fixed source-code table names and column
lists; mixed versions reject before acquiring a write transaction. Existing
immutable-event comparison and transaction-wide conflict rejection apply to both
versions. Normalized decimal strings make equivalent repeat writes idempotent.

Regression evidence: six tests failed before implementation. Storage and existing
canonical tests passed 31 checks. Independent review found no actionable defect
and independently passed six storage tests. Added a temporary-database upgrade
case that removes only its synthetic v2 table, reopens initialization, and verifies
both the v1 schema SQL and all v1 rows remain unchanged. Final contract/storage/
canonical/risk/safety run: 501 passed, 1 existing warning in 44.98 seconds.
Changed-file Black, Flake8 and whitespace checks pass. Task-workspace logs:
work/fractional-storage-red.log, work/fractional-storage-tests.log,
work/fractional-storage-safety.log.

Both production readers still select version 1; version-2 read selection and
worker rollout remain the next implementation work. No production default was
changed and no operational database was opened or migrated. Verified units,
liquidity evidence, final entry admission/settlement and launch gates remain open.
Readiness is false; all edits remain local and uncommitted.


## September 14: version-aware exact-decimal readers

Both async and synchronous canonical readers now select across v1/v2 using a
shared fixed-column relation. Selection is deterministic, preferring version 2
on an otherwise identical tie, then pins schema_version in the selected series
query. V1 integer and v2 exact-text representations cannot mix. V2 units remain
explicitly unknown. Corrupt or unsupported selected records fail validation
rather than falling back to older v1 records. The sync reader still supports a
pre-v2 database when the v2 object is genuinely absent.

Independent review found that the sync presence probe treated a same-name view
as absent. Added explicit schema-object checks in both readers and a synthetic
wrong-kind-object regression. Both now reject that malformed schema. The sync
API retains its existing generic read-unavailable error with the typed schema
error as its cause; the test checks both levels rather than changing API text.

Validation: initial reader tests reproduced two failures; 36 initial combined
reader/storage/canonical tests passed after implementation. Broader risk/safety/
security/contract/storage/reader run: 920 passed, 4 existing skips, 1 warning in
21.72 seconds before the schema-object review fix. Final affected reader/storage/
contract/canonical run after that fix: 53 passed in 2.54 seconds. Black, Flake8
and whitespace checks pass. Task-workspace evidence:
work/fractional-readers-red.log, work/fractional-readers-schema-red.log,
work/fractional-readers-safety.log, work/fractional-readers-final.log.

Worker/default production canonicalization still use version 1. Exact fractional
worker serialization, version-2 production binding and analytical consumers remain
next before rollout. Verified Gateway units and liquidity production remain open,
as do final entry admission/settlement and operational launch gates. No service
or operational database was changed. Readiness remains false; edits are local
and uncommitted.


## September 17: fractional worker integration and full-suite corrections

The subprocess worker emits exact decimal volume text with bar_schema_version=2.
The client requires that exact integer marker before accepting lineage; missing,
old, boolean or string markers poison the responding generation. Its canonical
path explicitly selects schema version 2. Legacy direct canonicalizer callers
still default to version 1. DataFetcher now uses producer-owned canonical batches,
returns a numeric analytical frame and persists original canonical rows instead
of truncating volume through the frame. Source units remain unknown.

Initial worker regression run reproduced 14 failures. Targeted integration run:
154 passed, 1 existing warning. Independent review found no actionable defect
and passed 81 focused tests. Changed-file formatting/lint/whitespace checks pass.
The data-fetcher log named fractional-fetcher-red.log is NOT failing-before
proof: its process imported the changed implementation and passed; do not report
that log as a red test. Its final integration result is included in the 154 run.

The full run then finished with 3891 passed, 3 failed, 4 skipped, 19 warnings in
143.75 seconds (work/fractional-rollout-repository.log). The direct-IB boundary
check exposed a missing explicit guard import in the new decoder module; added
that import and updated the test's expected direct importer from worker to the
new decoder. The test continues requiring every discovered direct IB importer
to import the guard.

The other two failures came from a midnight-sensitive fixture: its current-time
snapshot could include a one-minute-old execution outside its since-midnight
scope. Added six explicit midnight cases, reproduced the failure, and made the
fixture execution stay within its declared scope. Production scope validation
was not relaxed. Affected package-boundary/reconciliation/decoder/fetcher tests:
85 passed, 1 warning in 10.50 seconds. Evidence: work/midnight-fixture-red.log and
work/fractional-rollout-fixes.log.

A fresh full-suite run is active after those fixes, logging to
work/fractional-rollout-repository-final.log. Poll its returned session handle;
no terminal result is yet claimed. Current SSH diagnostic ended with hostname
resolution failure for blackm5mbp. No remote changes or service launch occurred.
Volume-unit verification, liquidity production, entry admission/settlement and
operator-dependent gates remain unfinished. Readiness is false; local edits are
uncommitted and no operational data was changed.


## September 17: full fractional-path verification and share-unit template intent

The active full-suite session reached successful completion: 3900 passed,
4 existing skips, 19 warnings in 100.77 seconds. Evidence:
work/fractional-rollout-repository-final.log. This validates the locally integrated
decoder, worker marker/serialization, client v2 canonicalization, exact storage,
version-aware readers and canonical data-fetcher changes, including the disconnect
guard and midnight-fixture fixes. Skips remain Docker containment availability,
the existing pairs-path integration TODO and a production-config environment case.
This is code verification, not operational paper readiness or profitability proof.

New primary-source investigation found the tracked IBC template accepts the
size-display notification but lacked SendMarketDataInLotsForUSstocks. Upstream
IBC documents that accepting/defering that dialog can set the lots checkbox.
Added SendMarketDataInLotsForUSstocks=no to the template, requesting shares while
retaining ReadOnlyApi=yes. Verified both entries and whitespace. The template-only
change occurred while the code suite ran; no runtime/test code changed during it.
No deployed config or Gateway state was changed.

Reviewed the upstream Java configuration action: it emits distinct already-set,
changed and missing-checkbox messages. Its start message alone is insufficient.
Documented the source links and required current-process/application evidence in
docs/FRACTIONAL_VOLUME_IMPLEMENTATION.md. A desired config value, stale log or
missing-checkbox error must not authorize units. Independently verified running
Gateway settings and binding/invalidation with the worker generation still need
implementation. All current canonical volume_unit values remain unknown.

Next: current-session unit evidence, liquidity production, final risk admission
and entry settlement, followed by operational backup/bootstrap/reconciliation and
operator-dependent startup gates. Remote name resolution remains unresolved.
Readiness remains false; no operational data, services, commits or remote branches
were changed.


## September 17: independent ledger verifier client and deadline validation

Added a dormant HTTPS RemoteMonotonicVerifier callable for the daily-fill ledger.
It requires an explicitly enrolled ledger ID, uses verified TLS and a fresh nonce,
and binds each response to the exact counters and chain heads. Redirects, cached
approval, auto-enrollment, duplicate/malformed JSON and oversized responses fail
closed. Private transport details are omitted from errors.

Review identified that a socket timeout cannot bound slow-progress responses.
Added an overall two-second caller deadline, one active transport worker per
client, permanent unavailability after timeout/error, and rejection of concurrent
calls. Late approvals cannot clear the latch. A stalled OS resolver may leave at
most one daemon worker per client; runtime must not reconstruct/retry clients.
Cleanup failures are sanitized, and interrupted waits/thread-start failures latch
before releasing the lock. Tests include a real HTTPResponse/socketpair slow-drip
body and actual DailyFilledNotional append/replay through the client interface.

Final focused validation: 143 passed in 4.67 seconds, covering the verifier and
existing daily-fill ledger suite (work/remote-verifier-final.log). New-file Black,
Flake8 and git diff --check pass. Earlier full-suite evidence remains 3900 passed;
the full suite was not rerun for this isolated, unwired module. The initial red
run failed collection because the new module did not yet exist.

The service itself, independent durable storage, enrollment/bootstrap procedure,
secret provider and runtime injection remain incomplete. The required protocol,
operator provisioning and recovery tests are documented in
FILLED_NOTIONAL_REMOTE_AUTHORITY.md. Asked the operator whether a separate host or
managed service is available; no deployment is inferred from that question.

Network evidence now distinguishes discovery from reachability: blackm5mbp.local
was resolved in the preceding diagnostic, but a fresh BatchMode SSH connection
with a five-second connect timeout ended with port-22 timeout (exit 255). The short
hostname resolution failure is not the sole current access blocker. No remote
configuration, service, trading data, commits or branches were changed.

Goal-turn classification: progress through implementation and fault validation.
Paper readiness remains false. Current-session volume units, liquidity evidence,
final entry admission/settlement, independent verifier deployment, actual paper
account ID, real operational backup/bootstrap/reconciliation and supervised launch
approval remain outstanding. This does not establish operational profitability.

Independent final review closed all reported client findings and passed all 35 verifier tests.


## September 17: settlement precision defect reproduced and corrected

Tracing the remaining entry/settlement integration exposed context-sensitive
arithmetic in the existing reduction path. Decimal.scaleb(-2) could round USD
commissions/rebates; abs(Decimal quantity) could round the number of shares used
for mark revaluation. The database trade projection also multiplied fill price
and quantity using ambient precision before converting to its legacy REAL field.
A synthetic persisted trade therefore recorded 7654.3 instead of 7654.34.

Added exact tuple-based minor-unit conversion, used copy_abs for share magnitude,
and reused validated exact multiplication for the legacy trade notional. No
existing trade/history rows were rewritten. These changes do not authorize BUYs
or alter the reduction-only settlement contract.

Failing-before evidence: 13 failed in work/settlement-context-red.log. Regression
coverage includes long/short reductions, commissions and rebates, low precision
with rounding traps both enabled and disabled, and actual temporary-database
trade/account/FIFO projections. The final focused run passed 44 tests with one
existing warning (work/settlement-context-focused.log). Independent review also
passed 44 tests and found no actionable defect. Changed-file Black/Flake8 checks
and git diff --check pass.

Full repository validation completed successfully: 3949 passed, 4 skipped,
19 warnings in 204.19 seconds (work/settlement-context-repository.log). The four
skips remain two Docker Compose availability cases, an existing pairs integration
TODO, and a production-config environment case. This run includes the prior
remote-verifier client plus the settlement precision fix. Entry admission remains
disabled, and PAPER_TERMINAL_SETTLEMENT_READY remains false.
The remaining BUY path requires separate admission/claim and atomic FIFO, cash,
position, outbox and journal-release integration; the current reduction request
explicitly rejects BUY. Do not broaden that request's allowed sides as a shortcut.
The prior volume-unit, independent-authority deployment and operational launch
gates remain outstanding. Goal-turn classification: progress through a reproduced
accounting defect, implementation and validation. No operational data, remote
configuration, services, commits or branches changed.


## September 17: separate opening-BUY accounting projection

Added paper_entry_settlement.project_flat_paper_buy for the missing entry
settlement path. It calculates exact cash, retained realized P&L, mark-to-market
daily P&L, quantity, cost basis and mark for a full BUY from an explicitly flat
position. Gate A's current zero-commission simulator policy is enforced; partial
fills, existing long/short quantity, invalid terminal states, nonzero fees and
insufficient cash reject. An unfilled terminal outcome preserves account values.

The existing database retains closed-position cost/mark/source metadata at zero
quantity. The projection therefore permits consistent retained history, preserves
it for an unfilled result and replaces cost/mark only for a new fill. It does not
require deleting closed-position history to permit re-entry. This was checked
against both an actual FIFO opening and an actual BUY/close/re-entry sequence.
Other-symbol realized/daily P&L and the persisted day-start baseline are preserved.

Validation: 47 tests passed in 0.38 seconds across the new projection, runtime FIFO
and settlement precision regressions (work/entry-settlement-final.log). The initial
red run failed collection because the new module was absent. Independent review
passed the then-current 26 projection tests and found no actionable defect; the
additional complete FIFO re-entry case was included in the final local run.
Black, Flake8 and git diff --check pass. The previous full-suite result remains
3949 passed; it was not repeated for this isolated, unwired calculation module.

This is a calculation, not a receipt, persistence implementation or order
capability. Next required integration: authenticate the consumed entry claim and
terminal execution, prove flatness under BEGIN IMMEDIATE, append FIFO, compare
this projection, atomically persist account/position/terminal/outbox state, and
release the journal reservation only after verified settlement. Replays and
fault recovery must preserve the same identities. The reduction-only request
continues rejecting BUY; its side validation was not broadened.

Readiness remains false and submit_baseline_entry still unconditionally rejects.
Volume-unit evidence, final risk admission, independent verifier deployment and
all operational launch gates remain open. Goal-turn classification: progress.
No operational data, services, remote configuration, commits or branches changed.


## September 17: durable entry-attempt claim before BUY settlement

The entry settlement writer needs an immutable attempt identity; capacity-only
reservations had none. Added ENTRY_SUBMISSION_CLAIMED and claim_entry_capacity.
The write transaction verifies the expected journal head, parent reservation
sequence/hash, scope and intent fingerprint, then records one claim ID and order
reference before its reservation expires. Exact duplicate/reopened attempts,
stale heads, unavailable parents and expired decisions reject. The claim is not
an execution permit and does not release any risk capacity.

Replay verifies the parent relationship and lifetime, rejects duplicate claims,
and keeps the original capacity reservation pending. Other journal paths cannot
reuse the claim's idempotency key. Existing events are retained unchanged; no
operational journal was opened or migrated. Older readers that do not recognize
the new event type must fail closed rather than drop it from replay.

Tests cover competing connections, process-style reopen, committed response
loss, rollback before commit, hash-valid malformed persisted claims, all pending
risk totals remaining reserved, and bootstrap refusal with an unresolved claim.
Review caught generated claim-ID collision handling: creation now checks every
prior event inside the same transaction before append, preserving readable
history on collision. Independent review closed that finding and passed the
then-current 22 claim tests. Final focused validation: 208 passed in 1.11 seconds
(work/entry-claim-final.log), including two additional containment regressions.
A missing-arguments error in the new bootstrap fixture was corrected; production
bootstrap validation was unchanged. Black/Flake8 and git diff --check pass.

Full repository validation completed: 4000 passed, 4 skipped, 19 warnings in
99.38 seconds (work/entry-claim-repository.log). Skips remain two Docker Compose
availability checks, the existing pairs-path integration TODO and a production
configuration environment case. This includes the preceding BUY projection and
the durable entry-claim changes; it does not establish operational readiness.
Next: bind the claim to the one-shot BUY execution boundary and atomic terminal
settlement/outbox, then enable release only from verified persistence evidence.
Final risk admission, volume-unit evidence, independent verifier deployment and
operator-dependent backup/account/reconciliation/startup gates remain open.
Readiness stays false. Goal-turn classification: progress. No operational data,
services, remote configuration, commits or branches changed.


## September 17: canonical proposed entry terminal record

Added build_paper_entry_terminal_record and PaperEntryTerminalRecord alongside
the exact BUY projection. The proposed record binds complete reservation/claim
snapshots, their payload hashes and structural relationship, the owned broker
quote and its reserved ID/generation, exact account pre-state, simulator outcome
and execution evidence, and derived post-state into canonical JSON plus SHA-256.
It requires the local paper execution domain and the claimed portfolio/order
reference, rejects an outcome before its claim, and bounds actual fill principal
by the durable reserved amount. Partial outcomes, nonzero commission, changed
quote evidence and held symbol exposure reject. Net and gross symbol quantities
are both explicit; offsetting portfolio holdings cannot represent flat exposure.

This is proposed data only. The future writer must authenticate the actual
journal history and consumed one-shot execution, verify all portfolio positions
and account/FIFO pre-state under its transaction, and persist terminal/outbox
state atomically. Constructing or fingerprinting this record does not authorize
any database mutation or journal release. Durable deserialization/replay and
atomic persistence are still outstanding; no existing reduction receipt/parser
was broadened to accept this record.

Validation: 78 focused tests passed in 0.73 seconds across proposed terminal
records, BUY projections, entry claims and runtime FIFO
(work/entry-terminal-record-final.log). Independent review passed the then-current
45 terminal/projection tests and found no actionable defect. Its reminder about
net-zero opposing positions led to the explicit gross quantity field and an
additional regression in the final run. Black/Flake8 and git diff --check pass.
These tests were added alongside/after implementation; no failing-before result
is claimed for this step. The prior full-suite result remains 4000 passed; it was
not repeated for this isolated, unwired record builder.

Next: persisted record validation/replay and the atomic entry settlement writer,
then one-shot execution and receipt-bound release through the final gateway.
Volume-unit evidence, final risk admission, independent verifier deployment and
operational backup/account/reconciliation/launch checks remain incomplete.
Readiness and BUY submission remain disabled. Goal-turn classification: progress.
No operational data, services, remote configuration, commits or branches changed.


### 2026-09-17 — Stored entry validation and exact outcome reconciliation

Added strict historical entry record validation. It rejects duplicate/unknown
fields, noncanonical scalar encodings, altered quote identity and accounting,
then recomputes the entire terminal record and matches its reservation and claim
to actual read-only journal replay. Missing journals are not created. Historical
quotes remain data structures, never producer-owned live evidence or permits.
Later unrelated journal events do not invalidate an authentic historical record.
The live builder still requires its actual quote producer.

Independent review exposed ambient Decimal rounding in the shared terminal
outcome's quantity conservation check. Four authentic 123-share full/no-fill
builder/replay regressions failed before the fix. Reconciliation now uses an
isolated context sized from operand coefficients and traps inexact addition;
nonnegative sums cannot round into an accepted mismatch. Regressions cover
partial fills, fractional carries, tiny imbalances, large positive/negative
exponents and restrictive caller exponent/rounding settings. This supports the
bounded trading quantity contract; arbitrary Decimal values below that context's
extreme subnormal range are not an additional diagnostic compatibility promise.

Focused validation: 175 passed in 2.47 seconds (entry-record-replay-focused.log),
including existing reduction submitter tests. Independent reviewer confirmed
50 replay/terminal tests and closed the original precision finding. Black,
Flake8 and git diff --check passed for the affected code.

The parser proves structural/accounting consistency and actual journal links,
not execution provenance or database pre-state. The next required step is the
atomic BUY settlement writer with FIFO/account/position/outbox updates,
idempotent recovery and receipt-bound release, then sealed one-shot execution
and final risk admission. Readiness remains false and baseline BUY submission
still rejects unconditionally. Deployment, actual account ID, independent
verifier provisioning, runtime volume-unit proof, backup/restore/reconciliation
and supervised launch remain outstanding. No user data or services changed.
Goal-turn classification: progress; the unrelated image turn made no goal
progress, and this continuation revalidated current files and the prior test
handle before advancing. HEAD remains cae45e0; no commit, push or launch.


### 2026-09-19 — Full-suite result and BUY persistence integration boundary

The interrupted September 17 run completed successfully: **4056 passed, 4
skipped, 19 warnings in 235.46 seconds**. Re-read its terminal log on September
19; the recorded tool process had exited zero. Log:
`work/entry-record-replay-repository.log`. Skips remain two unavailable Docker
Compose checks, the existing pairs integration TODO, and one environment-bound
production-config test. No new code has been changed since this result.

Inspection for the next writer identified a required integration boundary:
`paper_account_settlement_state`, `paper_position_settlement_state`, and
`paper_fifo_settlement_links` reference `paper_reduction_settlements`. Account and
position readers resolve those links, and the risk snapshot and outbox replay
currently deserialize every terminal row as a reduction. A separate BUY table
or a BUY payload inserted without typed reader support would break authenticated
recovery. Therefore the next implementation must add an explicit settlement kind
to the existing durable terminal table, retain the historical table name and all
existing row/payload identities, and dispatch strict entry versus reduction
parsers. A single table preserves the existing transaction insertion order and
foreign-key graph without rebuilding or copying operational history.

Implementation sequence and acceptance checks are in
`docs/superpowers/plans/2026-09-19-paper-entry-persistence.md`. Changes must remain
unwired until schema, writer, FIFO linkage, mixed replay/readers, execution
provenance and release are verified together. Merely adding the writer is not a
completed BUY path. No migration has been run on an operational database.
Prior goal turn: progress (parser and precision fix plus completed full-suite
validation). Current work revalidated that evidence and established the concrete
schema/reader dependency for implementation. Goal remains active; paper launch
is not yet verified.


### 2026-09-19 — Terminal-kind migration and historical dispatch implemented

Completed Task 1 of the entry persistence plan. Exact-state migration v4 adds
`settlement_kind` to the existing terminal table, with REDUCTION as the default
for historical rows and an explicit REDUCTION/ENTRY SQL constraint. The old
foreign keys, payloads, receipt fingerprints and row identities are retained.
Fresh initialization and offline bootstrap both use the ordered migration.
Hot-schema authentication checks the exact new column definition and constraint.

Strict historical dispatch selects the separate entry or reduction parser and
verifies its fingerprint. Neither parser grants execution authority. Until mixed
recovery lands, the reduction outbox, reduction writer, risk snapshot and
bootstrap journal crosslink explicitly reject ENTRY history instead of treating
it as reduction data or silently omitting it. Raw pre-v4 bootstrap sources retain
their historical reduction-only interpretation.

Five schema tests failed before implementation; parser collection failed on the
absent module. An additional regression reproduced old outbox acceptance of a
row marked ENTRY before the guard. Final focused integration: 175 passed,
1 existing monitoring warning, 8.33 seconds (`work/settlement-kind-integration.log`).
A subsequent upgrade rollback regression passed with the migration tests
(`work/settlement-kind-rollback.log`): rolling back restores v3 column shape and
all prior terminal data, then committing and repeating v4 preserves the original
reduction receipt. A test harness initially lacked the schema audit's required
transaction and was corrected; that was not a product defect. Black/Flake8 and
git diff --check pass.

Tasks 2–4 remain: consumed entry execution, atomic entry settlement, complete
mixed history and receipt-bound release. Readiness remains false; no baseline
BUY was enabled, no operational database was migrated, and no service, remote
configuration, commit or deployment changed. Goal-turn classification: progress.


Full repository validation for the terminal-kind changes completed successfully:
**4071 passed, 4 skipped, 19 warnings in 102.77 seconds**
(`work/settlement-kind-repository.log`, process exit 0). Skips are unchanged:
two Docker Compose checks unavailable on this host, one existing pairs-path
integration TODO and one production-configuration environment case. This proves
the tested schema/reader changes, not operational readiness or completion of
Tasks 2–4. Independent whole-plan review remains required after integration.


### 2026-09-19 — Atomic entry transaction engine staged (Task 2 in progress)

Added `paper_entry_persistence.stage_entry_settlement`, an internal transaction
engine that requires a caller-owned write transaction. It validates the proposed
record against the actual journal, checks paper/read-only runtime scope, database
descriptor identity, hot schema, exact account pre-state, integer gross-flat
symbol exposure and retained position metadata. It stages FIFO BUY evidence,
trade, ENTRY terminal/outbox row, exact and compatibility positions/accounts and
FIFO linkage together. A savepoint rolls back all engine mutations on ordinary
exceptions and BaseException, even if a caller catches the error and commits
its outer transaction. It never commits or releases journal capacity.

Full and zero-fill storage paths are exercised on synthetic ledgers. New opening
positions and re-entry over retained flat metadata work. Price improvement
updates exact cash/daily P&L and the compatibility equity delta; stale legacy cost
metadata rejects. Duplicate settlement identities reject without another fill.
Zero-fill outcomes create no trades or FIFO fills and preserve position state.

This is not the runtime-facing writer. Its result is plain uncommitted storage
identity data, not a producer-owned receipt. It does not authenticate consumed
execution authority, perform complete historical retry authentication, or bind a
public commit lifecycle to filesystem path replacement checks. Those checks and
the dispatch/receipt registry must be supplied by the integrated adapter before
runtime use. No runtime module calls this engine; all exercised writes are to
temporary test databases. Mixed history and final execution/release remain open.

Ruling: implement the storage engine before the consumed-dispatch adapter because
entry execution authority depends on final gateway admission. Exercise the
storage unit directly using a writable synthetic connection, without introducing
a test-only capability issuer or relaxing production execution. The public
`commit_paper_entry_outcome` API remains absent until genuine execution provenance
and commit/receipt recovery can be verified together. This changes implementation
order, not the required final behavior; Task 2 is not complete.

Validation: missing module failed collection before implementation. Subsequent
regressions failed for stale compatibility equity and retained cost mismatch,
then passed after fixes. **99 focused tests passed, 1 existing monitoring warning,
in 3.78 seconds** (`work/entry-persistence-integration.log`), including 33 engine
cases, entry record/projection and runtime FIFO coverage. Fourteen fault cases
cover seven mutation boundaries with RuntimeError and KeyboardInterrupt. All
participating tables are compared before/after rollback. Black/Flake8 and
`git diff --check` passed. Full-suite result remains 4071 passed before this
isolated, unwired engine; no new whole-repository result is claimed.

Goal-turn classification: progress. Prior turn also made verified implementation
progress. Readiness remains false and the baseline BUY gateway rejects. No
operational data, services, IBKR settings, commits or deployments changed.


### 2026-09-19 — Read-only entry recovery and exact retry validation

Added `paper_entry_storage_replay.read_entry_settlement`. In one existing read
snapshot it validates the proposed record against actual journal history, checks
all terminal envelope fields and storage fingerprint, verifies the exact trade,
and authenticates the existing FIFO fill and linkage without appending or
repairing anything. Zero-fill rows cannot claim a trade, FIFO link or fill under
their record identity. The reader binds the pathname, guardian descriptor and
SQLite descriptor before and after validation. Reopened-database recovery works
without the original quote producer. Results remain plain storage data, not
owned commit receipts or release authority.

Exact duplicate staging now uses this verifier and returns the original storage
identities without reapplying cash, position, trade or FIFO effects. Reusing a
claim with different accounting rejects. Corrupted terminal metadata, trade side
or price, FIFO quantity, missing link and changed link fingerprint all reject;
tests enforce SQLite query_only and compare every table plus total_changes to
show that validation performs no repair.

A new failing regression demonstrated that the staging engine previously did
not detect pathname replacement while the old SQLite file remained open. The
engine now holds a SQLitePathBinding through staging/retry and verifies it
inside the savepoint before release, rolling back all staged effects if the
path is swapped. Commit itself is still owned by the future integrated adapter;
its binding must extend through commit and receipt issuance. This change closes
the staging gap, not the whole runtime lifecycle.

Validation: absent-reader collection failed before implementation, exact retry
failed before wiring, and path replacement failed before the staging binding
fix. Final focused run: **119 passed, 1 existing monitoring warning in 5.00
seconds** (`work/entry-storage-replay-integration.log`). Black/Flake8 and git diff
--check pass. The prior full-suite result remains4071 passed; no newer full-suite
result is claimed for these isolated, unwired storage additions.

Task2 remains open for genuine consumed execution, the public authority-checking
commit adapter and producer-owned receipt issuance. Task3 mixed-history replay
and Task4 final gateway/release remain open. Baseline BUY rejection and readiness
false remain. No operational database, service, IBKR configuration, commit or
deployment changed. Prior and current goal turns both classify as progress.


### 2026-09-19 — Committed entry receipt recovery

Added `paper_entry_receipt.recover_committed_entry_receipt` and an owned-receipt
validator. Recovery opens a separate SQLite `mode=ro` connection, not a pooled
writer connection. It verifies terminal/trade/FIFO storage from an independent
transaction, closes that connection, rechecks the pinned file identity and only
then registers the receipt. The captured record is copied before any await, so
later mutation of the caller's record cannot change the recovered payload.
The registry checks object identity and canonical metadata; copying,
serialization, replacement records and metadata mutation cannot manufacture an
owned receipt. Scope and current file identity are checked on receipt use.

Tests hold the only pooled connection in an uncommitted write transaction:
recovery sees no terminal row and cannot issue a receipt. After commit, it
recovers the original fingerprint without changing tables. A write attempted
through the actual recovery connection fails with SQLite's readonly error.
Receipt mutation, dataclass replacement, deepcopy/pickle, wrong account scope,
record-shaped substitutes and a replaced file all reject. This proves committed
storage receipt recovery; it does not authenticate a new execution request or
grant journal release by itself.

The first test failed on the absent module. Initial implementation exposed the
new connection's default disabled foreign-key enforcement; enabling that
connection-local pragma fixed schema verification, and the pre-commit test now
specifically requires the missing-row error rather than accepting any failure.
Final validation: **127 passed, 1 existing monitoring warning in 6.29 seconds**
(`work/entry-receipt-final.log`). Black/Flake8 and git diff --check pass. The prior
full-suite result remains4071; no fresh whole-suite result is claimed for this
unwired module.

Task2 still needs genuine consumed entry execution and the runtime-facing commit
adapter spanning authority checks, transaction commit and receipt recovery.
Task3 mixed history/daily accounting and Task4 final admission/release remain
required. Receipt ownership proves registered committed storage; it is not a
submission permit. Baseline BUY rejection and readinessfalse remain. No user
data, services, IBKR configuration, commits or deployments changed. Current and
previous goal turns classify as progress.


### 2026-09-19 — Mixed terminal outbox and daily notional projection

Extended `iter_paper_terminal_receipts` with an explicit optional SafetyJournal.
With that journal it reads the common terminal envelope in durable row order,
recovers each ENTRY as an owned committed receipt and checks the entire envelope
against its parent read snapshot. Existing REDUCTION parsing remains strict.
Without a journal, entry history still rejects rather than being omitted.
The parent snapshot/file binding is retained across iteration, and recovery never
returns success for a partially consumed or invalid stream.

PaperFillAccounting now accepts owned entry receipts with the exact source
database, verifies runtime/scope/file ownership, and projects BUY principal using
the committed execution ID, exact quantity/price and execution timestamp. Its
existing independent monotonic ledger provides idempotent ingestion. Zero fills
count as terminal receipts but add no notional. Runtime-configured journal replay
is enabled for mixed accounting; missing journal configuration retains the
reduction-only guard. No journal file is initialized by this read path.

Tests cover a real synthetic reduction followed by an entry in another portfolio,
durable stream order, both sides' exact daily totals, repeated replay with zero
new fills, corruption after a valid reduction prefix, verifier refusal with
entry capacity retained, and all three zero-fill statuses. Entry-only replay
failed against the old guard before implementation. The first implementation
run exposed a missing canonical_json import; it was corrected before validation.
Focused integration: **70 passed, 1 existing monitoring warning in 6.40 seconds**
(`work/entry-accounting-integration.log`). Black/Flake8 and git diff --check pass.

Ruling: implement the read-only outbox/daily portion of Task3 while the execution
adapter portion of Task2 remains open, because its committed-receipt dependency
is now available and this adds no submission authority. Task3 is not complete:
collect_paper_risk_ledger_snapshot, bootstrap crosslink and the reduction writer
still reject ENTRY history. Full cash/position/FIFO history reconstruction must
be integrated before those guards change. The public entry commit adapter,
consumed execution and final admission/release remain required.

Goal-turn classification: progress, as was the preceding receipt-recovery turn.
Readinessfalse and unconditional baseline BUY rejection remain. No operational
history, services, IBKR settings, commit, push or deployment changed.


### 2026-09-19 — Mixed accounting repository verification

The first full run found one reduction callback compatibility regression:
4138 passed, 4 skipped, 1 failed. Mixed replay supplied its new entry-only
`database` keyword to a reduction ingestion callback that accepts only a
receipt. Replay now supplies that keyword only for ENTRY receipts, preserving
the existing reduction callback contract. The existing cancellation/replay
regression and mixed accounting checks then passed: 16 tests, 1 warning, 3.41s
(`work/entry-accounting-callback-fix.log`).

Fresh full repository verification after the fix: **4139 passed, 4 skipped,
19 warnings in 107.14 seconds**, exit 0, session7092
(`work/entry-accounting-repository-final.log`). Skips are two Docker Compose
containment cases (Compose unavailable), an existing pairs-path integration TODO,
and an existing production-config environment case. Those skipped gates are not
proven by this run. Black and Flake8 passed for the four affected Python files;
`git diff --check` passed.

Task3 remains partial: mixed outbox/daily projection is verified, while the risk
snapshot, bootstrap crosslink and reduction writer still reject ENTRY history.
Next implementation must reconstruct both kinds from the actual bootstrap in one
snapshot, including cash/quantity continuity, gross-flat entry and complete FIFO
correspondence, before removing those guards. The public authority-checking entry
commit adapter, consumed execution, final admission and verified release remain
open. Readiness is still false and baseline BUY submission still rejects.

The preceding image-only exchange made no trading-goal progress. This continuation
revalidated the stopped focused test process and completed fresh repository
verification, correcting stale plan/ledger status. No test process remains from
this verification. No operational data, service, IBKR setting, commit or deployment
changed. Operator-dependent account ID, remote access, independent verifier and
backup/reconciliation/launch gates remain unresolved; this suite does not prove
supervised paper readiness or profitability. The goal remains active.


### 2026-09-19 — Mixed entry risk snapshot and bootstrap lineage

The risk collector now dispatches ENTRY rows to strict entry storage/journal
verification on its existing SQLite read transaction. It reconstructs cash,
position quantity and position-source continuity from the sealed bootstrap,
requires account-wide gross-flat symbol history for an entry, and includes exact
BUY fills/links in the complete FIFO correspondence check. A previously absent
symbol can be introduced only by a validated filled entry; rejected/cancelled/
expired zero fills do not create positions. Reduction receipt parsing remains
unchanged. The snapshot grants no execution authority.

A genuine flat-bootstrap test exposed a storage integration defect: a newly
inserted position lacked origin_bootstrap_id. The entry engine now inherits that
origin from its exact account row when creating a new position. Existing position
origin is preserved; this is not a repair path for damaged history. No operational
ledger was changed.

Fourteen new tests cover bootstrap origin, full/zero outcomes, query-only snapshot
collection with unchanged table contents, reduction followed by entry and DB
reopen, corrupted trade/link/kind/cash/quantity/origin, coherent but unexplained
extra cash, and a missing journal that must not be created. Initial five cases
failed before implementation (missing origin and reduction-only snapshot guard).
Fixture authority uses a fresh instance of the existing production registry to
isolate current-time tests from historical synthetic clocks; all real ownership,
consumption and journal validation still run. Mixed test chronology was corrected
after FIFO correctly rejected an entry timestamp preceding its prior reduction.

Validation: **75 focused tests passed** in 6.30s; **4153 repository tests passed,
4 skipped, 19 warnings in 106.49s**, exit0/session4309
(`work/entry-snapshot-repository.log`). Black/Flake8 on the four affected Python
files and git diff --check passed. The same four skips remain: unavailable Docker
Compose containment checks (two), the existing pairs-path TODO, and the existing
production-config environment case. These are not proven by this run.

Task3 is still incomplete. The reduction writer and bootstrap journal crosslink
retain their ENTRY guards; a complete BUY/SELL/BUY cycle, global trade coverage,
and settled-entry release crosslink remain to be integrated. The public consumed-
execution commit adapter and final risk admission/release also remain required.
Readiness stays false, baseline BUY submission stays disabled, and IBKR remains
read-only. Account allow-list, remote access, runtime volume-unit evidence,
independent verifier provisioning, actual backup/reconciliation and supervised
launch gates remain unresolved. Profitability is not established.

Current and preceding goal turns classify as progress. Session4309 terminated;
no test process remains from this run. No operational history, services, IBKR
settings, commits, pushes or deployments changed. The goal remains active.


### 2026-09-19 — Reduction settlement after verified entry history

Reduction persistence now validates complete mixed history inside its existing
BEGIN IMMEDIATE before any mutation when ENTRY records exist. It reuses the same
bootstrap/cash/quantity/FIFO reconstruction as the risk collector, through a
transaction-owned helper that neither commits nor rolls back the caller's
transaction. This avoids separate snapshots and single-connection-pool deadlock.
Reduction-only history keeps its existing guard; malformed mixed history fails
before a new fill is appended. Existing reduction receipt identity and FIFO
idempotency checks remain in place.

Tests now exercise a genuine temporary flat bootstrap, stored BUY, close/reopen,
full SELL, another close/reopen and exact retry. The final cash is 100006 with no
open position, exactly two terminal fills, and retry changes no table. Corrupting
the entry trade rejects the reduction before mutation. Fault injection after
trade insertion and FIFO-link insertion restores every table. The initial test
failed on the old ENTRY guard. Two new fault tests initially used incorrect hook
names; corrected to the existing insertion-boundary hooks. The historical
relabelled-reduction test still rejects, now through mixed bootstrap validation.
Focused verification:85passed,1warning,9.15s. Formatting/lint/diff checks passed.

This is storage integration, not a runtime trading cycle. The entry reservation
remains pending even after storage of a closing SELL. Current journal rules also
prevent a runtime reduction reservation while that entry remains pending. The
next required step is owned independent-accounting confirmation and receipt-bound
entry release, including replay/duplicate/failure validation; only then can the
actual gateway BUY→SELL→BUY cycle be demonstrated. No pending journal event was
forged, deleted or bypassed. Bootstrap crosslink, global trade coverage, consumed
entry execution/commit authority and final admission remain open.

Current and preceding turns classify as progress. Runtime BUY rejection and the
false readiness flag remain. No operational data, service, IBKR setting, commit,
push or deployment changed. Operator-dependent launch gates remain unchanged.


Repository verification after this integration: **4157 passed, 4 skipped,
19 warnings in 108.33s**, exit0/session32664
(`work/mixed-reduction-repository.log`). The four unchanged skips cover Docker
Compose availability (two), an existing pairs-path TODO and a production-config
environment case. They remain unverified. No active test session remains.
The goal remains active; full operational readiness is not established.


### 2026-09-19 — Owned entry accounting confirmation for release

Added PaperFillAccounting.confirm_entry_settlement and a separately sealed,
one-use PaperEntryAccountingConfirmation. The producer validates the owned
committed receipt and exact accounting/database scope, performs actual idempotent
BUY ingestion, then reads the fill day's authenticated daily total through the
independent verifier. Zero-fill outcomes also require this read; ordinary
ingestion's zero-fill fast path cannot establish release evidence by itself.
The total must cover the exact filled principal.

Confirmation binds receipt fingerprint, settlement, account/portfolio, New York
trading date, exact principal/total, ledger/anchor paths and exact producer,
database and ledger object identities. Consumption rejects mutation, copies,
rebound producer, reuse and stale evidence. Wall-clock and monotonic age are both
limited to five seconds measured from verification start, including verification
latency. A retry obtains a new confirmation through fresh verification. Async
cancellation drains the durable accounting operation and returns no confirmation.
The proof alone does not release capacity, append a journal event or authorize
execution; pending entry capacity remains intact in every new test.

Four initial tests failed before implementation. Adversarial testing exposed a
mutable producer-token field that was absent from payload validation; the token
mutation test failed, then passed after explicit token checking. Twelve new
cases cover exact filled accounting/replay, all three zero-fill verifier denials,
zero-fill success, copy/amount/token/producer/age rejection, verifier denial after
previous ingestion, and cancellation drain.34 focused integration tests passed,
1warning,4.04s. Black/Flake8 and git diff --check passed.

Next required work is the receipt/confirmation-bound journal release event,
strict release replay and pending-capacity reconstruction, including duplicate
and failure recovery. Only after that can the runtime BUY→SELL→BUY sequence be
exercised. The consumed execution/public commit adapter, final gateway admission,
bootstrap release crosslink and operational launch gates remain open. No runtime
readiness flag, broker setting, service or operational history changed; no commit,
push or deployment occurred. Current and preceding goal turns classify as progress.


Fresh repository verification: **4169 passed, 4 skipped, 19 warnings in 105.48s**,
exit0/session49908 (`work/entry-confirmation-repository.log`). The four unchanged
skips remain unverified (two Docker Compose checks, pairs-path integration TODO,
production-config environment case). No running test handle remains. Goal remains
active; confirmation is a verified release prerequisite, not operational readiness.


### 2026-09-19 — Durable entry release and repeated-entry storage cycle

Added release_entry_capacity and ENTRY_SETTLEMENT_RELEASED. Release requires the
owned committed entry receipt, actual runtime journal path, exact journal
reservation/claim parents, expected current head and fresh one-use independent
accounting confirmation. The journal transaction appends release only after
consuming that confirmation. Filled and zero-fill outcomes have strict quantity,
price/notional, status, execution, chronology, trading-date and hash checks.
Replay removes only the exact released reservation; pending-capacity summaries
reconstruct that subset. Expiry still does not release unresolved entries.

A failed journal commit rolls back and retains capacity; its consumed accounting
confirmation cannot be reused. A fresh confirmation allows retry. Exact committed
retry returns the original release without another append and compares actual
committed terminal fields, not just fingerprints. Release keys are protected
against use by other journal paths. Copy/forged evidence, stale heads, invalid
parents/status/quantity/accounting data and hash-valid malformed/duplicate events
are rejected. New regression tests first exposed release-key reuse and acceptance
of a hash-valid release that misstated a fill as rejected; both were corrected.

A genuine temporary bootstrap now completes the storage sequence BUY→owned
accounting confirmation→journal release→full SELL→second BUY, with close/reopen
between stages. Final cash98008, AAPL quantity6, three terminal fills and exact
daily principal6000 are verified. Only the second entry remains pending. This
uses real journal/DB/FIFO/accounting components but direct storage settlement,
not the still-disabled runtime gateway execution path. No test-only authority
issuer or journal bypass was added.

Validation: initial five release tests failed before implementation.103 focused
checks passed with1warning in6.42s. Black/Flake8 on the seven changed Python files
and git diff --check passed. The full repository suite is recorded below when
terminal. Entry execution/public commit adapter, final gateway admission and
independent whole-plan review remain open.

Next startup requirement: authenticate entry release events against committed
terminal records, trades and accounting in the bootstrap/reconciliation
crosslink, then finish complete trade coverage. Standalone journal replay proves
structural consistency, not cross-database truth. The bootstrap ENTRY guard
remains in place until that integration is tested. Existing operational launch
gates (approved paper ID, remote access, runtime volume proof, independent
verifier provisioning, real backup/reconciliation and supervised launch) remain
unresolved. Readinessfalse and baseline BUY rejection remain; no broker setting,
service, operational history, commit, push or deployment changed. Current and
preceding goal turns classify as progress; the goal remains active.


The first full suite terminated with4184passed,4skipped,1failure: the safety
package's standard-library-only dependency test. Corrected the architecture by
moving owned receipt/accounting consumption and journal-write orchestration to
`robo_trader/paper_entry_release.py`. `safety/entry_release.py` now contains only
standard-library structural replay/pending-subset validation; exact bounded
multiplication uses its own Decimal Context rather than importing trading code.
The package-boundary test was preserved.108focused tests then passed,1warning,
10.64s, including the dormancy checks and a new low-precision Decimal regression.
A new full suite is running as session12488 after that actual fix.


Fresh full-suite verification after the boundary correction: **4186 passed,
4 skipped,19 warnings in197.65s**, exit0/session12488
(`work/entry-release-repository-final.log`). The same confirmed live process was
polled throughout the longer run; no duplicate suite was started. The four
unchanged skips remain unverified (two Docker Compose checks, pairs-path TODO,
production-config environment case). No active test process remains from this
run. Durable release and the repeated-entry storage cycle are verified; runtime
entry/public commit authority, bootstrap release crosslink, global trade coverage,
independent review and operational launch gates remain open. Goal stays active.


### 2026-09-29 — Bounded stop-clock and skipped safety-test repair

Recovered the final September 19 startup-crosslink result: 4195 passed, 1 failed,
4 skipped (the earlier 4186-pass release run was not the latest result).
The original stop timestamp matrix passes alone. Advancing its fixture clock two
hours after collection reproduces the exact DID NOT RAISE failure: the timestamp
was created at import time and no longer represented a future stop. The historical
log does not establish why its wall clock advanced. Production stop validation is
unchanged. Clock-relative test cases now reject naive, +1-hour and +5-second-plus-
1-microsecond timestamps and accept exactly the existing 5-second tolerance.
159 focused existing-position/crosslink checks passed; all four delayed-clock
cases passed. Independent review found no substantive concerns (11 targeted
checks independently passed).

Replaced the production-config exception-to-skip test with six real isolated
production-paper/read-only cases, without RT_TEST_MODE or operator dotenv loading.
Replaced the stale pairs TODO skip with two actual-runner dispatch cases proving
both directions stay quarantined before state mutation/orders, even with fresh
protective evidence. Enabled-pairs BUY risk validation remains explicitly deferred
until atomic two-leg admission; this replacement does not claim that coverage.
Eight final focused replacement cases passed. Docker's two containment skips stay
queued at lower priority; no Docker installation. Black/Flake8 on both changed
files and git diff --check passed. Existing uncommitted implementation is preserved.

Fresh full suite: 4200 passed, 6 failed, 2 skipped, 18 warnings in 121.02s, exit 1,
session 7511, work/paper-test-repair-repository-2026-09-29.log. The original failure
and repaired skips pass. Remaining failures require native host verification:
operational plist path not writable; two launcher stop tests cannot enumerate
processes (pgrep confirms process-list access failure); two localhost listener
binds denied; Yahoo external-data smoke test cannot resolve guce.yahoo.com. The
tests' finally blocks kill their own worker subprocesses. No skipped-test bypass
or production modification was made to hide these failures. Full suite is not
green; rerun from the native host before claiming success.

Gate A remains closed. Actual one-shot entry/public commit/gateway integration,
complete trade coverage, runtime BUY/SELL/re-entry/restart demonstration and real
provisioning/backup/reconciliation evidence remain open. No operational service,
DB/history, runtime readiness, branch or commit was changed by this bounded fix.
The voice session separately reports its authorized native startup attempt failed
unmodified preflight on stale July equity; its Gateway read-only handshake worked.
See outputs/paper-test-repair-2026-09-29.md in the task root for the detailed report.
