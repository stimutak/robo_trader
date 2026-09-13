# Paper readiness execution record

Goal: verified supervised local paper operation, with measured execution cost and performance. Profitability is an evaluation outcome, not a readiness claim.

Canonical specification: `docs/ROBOTRADER_REMEDIATION_PLAN_2026-07-20.md`.
Baseline: GitHub main `57f7634`. The original local checkout was `f1a5b7a`; the supplied PR2 summary is superseded by the remote remediation program.

## Constraints

- Preserve authoritative data and safety state; use synthetic databases for development tests.
- Keep IBKR read-only, paper execution containment, and the terminal readiness gate closed during integration.
- Use only START_TRADER.sh for eventual runtime startup after cumulative Gate A evidence and the operator confirmation required by the canonical plan.
- Do not infer launch readiness from passing unit tests or profitable backtests.

## Execution sequence

- [x] Fetch remote state and inspect canonical instructions and open PRs.
- [x] Create isolated branch from current main.
- [x] Verify baseline security/safety/preflight: 737 passed, 4 skipped. Docker Compose unavailable; two pre-existing trading-core test skips. Log: workspace work/baseline-tests.log.
- [ ] Integrate PR #122 backup/restore locally and run maintenance, portfolio, and settlement regressions. Verify synthetic WAL backup/restore behavior on this macOS host. Review remaining PR findings against current code.
- [ ] Integrate PR #124 reconciliation locally; verify startup, reconnect, periodic refresh, stale evidence, and reduction-only behavior against merged FIFO and backup code.
- [ ] Integrate PR #116 durable daily filled notional locally; verify replay, restart, tamper rejection, and fill accounting tests.
- [ ] Complete the Gate-A entry-risk production integration after inspecting exact producer interfaces: account-wide reservations, fresh canonical quotes, current signed state, all configured exposure limits, final-boundary revalidation, and exactly-once notional ingestion. Add failure/concurrency tests before implementation.
- [ ] Run full regression, formatting, changed-file lint, and synthetic restart/restore drills; retain exact commit and test results.
- [ ] Prepare operator-reviewable bootstrap/reconciliation evidence without applying changes to historical data. Obtain required explicit approval only for the concrete data operation, if needed.
- [ ] Close Gate A only when all implementation and operational evidence passes; request the canonical immediate pre-start confirmation, then start through START_TRADER.sh and verify health.
- [ ] Validate PR #121 deterministic backtests and complete Gate B strategy evaluation with holdout periods, transaction costs, slippage, drawdown, turnover, and baseline comparison before tuning strategy parameters.
- [ ] Continue remote-access, packaging, audit, dashboard, and soak work in canonical dependency order. Real-money trading remains outside this paper launch.

## Open PR inputs

| PR | Exact fetched head | Purpose |
|---|---|---|
| 122 | 5796133fa54fb859d9e302228cb3cb0f5696b766 | Backup and restore |
| 124 | 16d1f99d92237a951aded3d6b7172787f38ee8cf | Runtime reconciliation |
| 116 | ca9d0378659b399b294449990a275d3d650984f1 | Daily gross filled notional |
| 121 | 23c53ce65531d6727285e7a522472f1cc5dcb6b2 | Deterministic backtesting |

These are integration inputs, not independently approved launch evidence. No remote PR is merged by this record.

## Integration evidence

Local integration head: `1b8f6c0` (all four PR inputs above composed with current main).

- Backup/portfolio tests: 200 passed.
- Reconciliation/startup tests: 299 passed.
- Risk tests: 108 passed.
- Combined suite before offline backtesting: 3309 passed, 4 skipped.
- Offline backtesting tests: 169 passed.
- Final combined suite: 3478 passed, 4 skipped, 19 warnings, 111.59 seconds.
- Black and Flake8: all 54 changed Python files pass; whitespace diff check passes.
- macOS synthetic WAL drill: backup, manifest verification, clean-room restore, restored manifest verification, exact row comparison all pass. Only newly created synthetic data was used.

Tests used `/Users/oliver/Projects/robo_trader/.venv/bin/python` with the isolated checkout as cwd. Docker Compose is unavailable (two skips); two existing trading-core tests are skipped. These results are development evidence, not authoritative database restore or broker soak evidence.

Independent review reproduced two upgrade blockers in PR124:

1. `robo_trader/reconciliation/runtime_integration.py` compares historical bootstrap receipt runtime fingerprints to the current build-dependent fingerprint. Changing only `build_id` rejects a valid sealed bootstrap before fresh reconciliation. The existing sealed FIFO epoch cannot simply be bootstrapped again. Repair must authenticate the original runtime binding and its stable account/domain/database identity; removing receipt validation is not a safe remedy.
2. `_status_owner_binding` includes the same build-dependent fingerprint. A status file published by one build is rejected as belonging to another owner by the next build. Repair requires a durable ownership identity or explicit authenticated ownership transfer; arbitrary replacement remains prohibited.

Operational inspection checked configuration presence only, without printing secrets or opening historical trading databases. The original Mac checkout's `.env` lacks explicit paper account, approved account list, account type, namespace, model artifact set, and build identity. Intended trading host and independent monotonic authority are pending user clarification. No launcher or broker session was started.

### Status ownership upgrade fix

The second review finding is repaired with v2 status ownership bound to stable environment, mode, execution source, account/domain, database namespace, safety journal, and status path. Build/model updates and signing-key rotation do not change artifact ownership. Fresh reconciliation evidence still binds the full current runtime fingerprint. Different durable identities remain separate; all file identity and replacement protections remain enforced.

The real status publication regression failed before the fix and passes afterward. The focused module passes 43 tests, the related reconciliation/adapter/web suite passes 220 tests, and independent review found no actionable defect. Black, Flake8, and whitespace checks pass.

For any pre-release installation that wrote a v1 status artifact, preserve that file and set `RT_RECONCILIATION_STATUS_PATH` to a new absolute path before adopting v2. There is no automatic adoption, deletion, or overwrite of v1 evidence. This changes diagnostic artifact ownership only; it does not authorize startup or resolve historical bootstrap fingerprint compatibility.

### Historical bootstrap upgrade fix

Historical receipts now require one well-formed original runtime fingerprint per bootstrap epoch rather than equality with the current build fingerprint. Bootstrap application already authenticates receipts against that original runtime before their atomic append-only persistence. Startup retains exact immutable schema, account/domain/database/path/inode checks and exact signed-artifact coverage. Current reconciliation still authenticates the current runtime independently. No historical rows are rewritten or newly blessed by this change.

The upgrade regression failed before implementation. Tests prove valid build/model upgrades pass; mixed/malformed receipt origins and changed account/domain/database namespace fail without changing database bytes. Related bootstrap, receipt producer, and runtime tests: 165 passed. Independent review: no concrete regression, 49 focused tests passed. Full suite after both upgrade fixes: 3491 passed, 4 skipped, 19 warnings in 96.28 seconds. Black, Flake8, and diff whitespace checks pass. Both reproduced upgrade blockers are repaired; entry-risk integration and operational Gate A remain open.

## User-provided runtime share

The user identified `/Volumes/oliver/Projects/robo_trader`. The mount is an SMB share from `blackm5mbp`; its checkout is main at `51f0e99`, older than the integration baseline. Existing unrelated changes: a type change to `robo_trader.log.1` and untracked `.security_round2_commits.sh`; neither was touched.

- IBC configuration exists, with `TradingMode=paper` and `ReadOnlyApi=yes`. Credentials were not displayed or copied into deliverables.
- Gateway logs are updating on September 13. This is log evidence only, not a verified broker connection or process-health claim.
- Latest watchdog failure (September 11) reports 323 consecutive unsuccessful restarts and a preflight block on July 13 equity history. No bypass was attempted.
- `trading_data.db` has WAL and SHM companions. All three were copied as an offline diagnostic family; source hashes before/after and captured hashes matched. Original files were never opened by SQLite or modified.
- SQLite inspection occurred only on a second local working copy. Integrity check passes: 234 trades, 2 position rows, 1 account, 57 equity rows, 1 portfolio. Latest trade is `2026-07-13 13:55:32`; latest equity is `2026-07-13 14:04:20`. No FIFO or exact bootstrap tables exist.
- Raw diagnostic capture and hashes: workspace `work/blackm5mbp-evidence-mluk9kys`. This is not a broker-reviewed or process-quiescent operational backup and cannot authorize bootstrap or launch.
- Share `.env` also lacks the explicit paper-account/allow-list/type and newer runtime identity fields.
- `blackm5mbp.local` lacks a known-host entry; existing trusted alias `blackm5mbp` verifies but rejects available SSH authentication for user `oliver`. User was asked for the configured SSH username/alias. No host-key check was bypassed and no remote command ran.

The June handoff documents historical accounting and data-quality incidents already motivating the canonical remediation plan. It is historical context, not current pricing or launch evidence. Current work continues from the newer isolated integration branch; the mounted runtime remains unchanged.

## Terminal fill replay integration (September 13)

Added an account-wide read-only terminal outbox iterator and a dormant paper-fill
accounting adapter. Replay validates producer-owned persisted receipts, exact
paper/account/database scope, and the opened SQLite inode before every yielded
receipt. Normal iterator abandonment rolls back its snapshot without poisoning
the connection pool. Replaced paths and malformed outbox rows fail closed.

The adapter projects fills into the independently authenticated daily-notional
ledger using a separate simulator account namespace. All portfolio scopes must
share one database and anchor so execution-ID deduplication remains account-wide.
Repeated replay after a lost response does not duplicate notional. Cancellation
drains an in-progress durable append before propagating. No runtime admission is
wired or enabled by this adapter; complete replay remains a prerequisite for
future entry integration.

Independent review identified stale pooled inode acceptance and split-ledger
deduplication gaps; both have regression tests and fixes. Final review found no
additional actionable defect. Focused tests: 10 passed. Full suite: 3501 passed,
4 skipped, 19 warnings in 141.31 seconds. Black, changed-file Flake8, and diff
whitespace checks pass. Full log: workspace work/paper-replay-full.log.

### Remote access update

The user confirmed SSH username `oliver`. A native Terminal login established a
connection through the existing trusted `blackm5mbp` host entry. Read-only checks
observed the watchdog and Java listening on paper API port 4002, but no runner.
A subsequent log-summary request timed out connecting to port 22; current broker
login and remote health remain unverified. No runtime was restarted or changed.

The supplied `robo6trader` value appears to be a login name; an actual approved
DU/DUN paper account identifier is still needed. Independent monotonic-verifier
configuration, reviewed bootstrap/reconciliation, operational restore evidence,
and remaining entry-risk integration are still open. No launch is authorized by
these development test results.

## Entry admission limits and next integration boundary

The exact entry contract now requires explicit per-order notional policy and an
exact positive maximum-open-position count. A specified order cap participates
in exact flooring and the all-capacity postcondition; explicit None retains the
existing optional configuration semantics. Admission evidence must include held
plus pending account position slots, account-wide symbol duplicate state, and a
durable cooldown boundary. Missing evidence fails closed. Every added evidence
field participates in the sealed capability state and boundary revalidation.

Regression tests cover exact share-price boundaries, malformed configuration,
hostile Decimal context, occupied and pending slots at the limit, duplicate
entries, exact cooldown expiry, absent evidence, and post-seal mutation. Mutation
tests exposed the initial missing seal fields before they were fixed. Independent
review found no actionable defect in the final changes (186 tests for the order
cap; 22 targeted tests for admission evidence).

The contract still grants no runtime execution authority. Remaining work is a
connected implementation, in this order:

1. Implement a runtime evidence producer using the existing account-wide gateway
   lock, canonical broker-bound quote source, current signed allocations, and
   independently authenticated daily-risk ledger. Count held and reserved
   symbols across portfolios, derive cooldowns from durable recent terminal
   fills, and map explicit configured limits. Include account-level leverage and
   pending cash/exposure in addition to current portfolio/symbol/sector values.
2. Bind startup to complete terminal replay and fail closed if any portfolio
   accounting scope or independent monotonic authority is unavailable. Feed every
   new terminal receipt to accounting before admission can reopen.
3. Implement a separately authorized baseline BUY terminal path with atomic
   exact FIFO/account/cash persistence, producer-owned outcome evidence, and
   crash recovery. The current PaperTerminalSettlementRequest accepts only SELL
   and BUY_TO_COVER; removing that reduction restriction alone is not an entry
   implementation.
4. Connect one-shot baseline intents and risk decisions to the gateway. Recheck
   signed state, quotes, price ceiling, all capacities, reconciliation, and
   reservations at submission; keep the lock through settlement and risk ingestion.
   Verify simultaneous portfolios, cancellation, restart, uncertain fills, and
   price changes before enabling the terminal readiness constant.
5. Finish the operator-dependent paper account, monotonic authority, reviewed
   bootstrap/reconciliation, and operational restore requirements. Run the
   canonical pre-start consent and START_TRADER.sh flow only after Gate A passes.

Open GitHub PRs were checked again: no separate entry-runtime implementation is
available among the open PRs. Integrated PRs remain open remotely. The latest
bounded SSH retry timed out; no remote changes were attempted.


Validation note: all 208 focused entry-contract tests pass. The first full run
completed with 3535 passed, 4 skipped, and one migration-test subprocess timeout
(the outer 10-second test guard, before a report was returned). That unchanged
migration test passes in isolation in 3.85 seconds and verifies SQLite progress
interruption and rollback. No timeout was relaxed. The full rerun passed: 3536 passed, 4 skipped, 19 warnings in 333.22 seconds.
It is retained separately as work/paper-entry-limits-full-rerun.log; the first run remains in
work/paper-entry-limits-full.log for audit.


Performance observation during validation: the existing hyperparameter-tuning
test requests the complete random-forest search grid. ModelTrainer configures
both estimators and GridSearchCV with n_jobs=-1; the full rerun visibly spent
minutes in joblib worker processes during that stage. Resource-budgeting and
representative tuning benchmarks remain follow-up work; this observation does
not establish a trading latency or profitability improvement.

Black, changed-file Flake8, and diff whitespace checks pass. Entry authority remains disabled.


## Account leverage and pending capacity

The dormant exact risk contract now requires explicit account equity/gross
exposure and pending commitments for symbol, sector, portfolio, account, cash,
buying power, and daily notional. Account leverage is constrained to the same
1-through-4 range as RiskConfig. Each reservation reduces its relevant capacity
using isolated exact arithmetic before quantity flooring and the shared
postcondition. Absent or malformed values fail closed; new fields participate
in the sealed capability snapshot and consumption-time revalidation.

Focused contract suite: 296 passed. Independent review: no actionable defect,
95 targeted tests passed. Regression tests cover other portfolios exhausting
account capacity, every pending capacity, missing evidence, malformed decimals,
seal mutation, and fractional boundary arithmetic under a hostile Decimal context.
This is a necessary contract extension, not a runtime snapshot or reservation
implementation. Runtime must still collect coherent values under the account
order lock, supply balances before reservation deductions, and revalidate them
before submission. Paper readiness remains false.

Combined entry-contract and durable-risk suites: 408 passed, 1 warning in 12.44 seconds. Black, changed-file Flake8, and whitespace checks pass.

## Coherent paper ledger snapshot

Added a dormant, read-only account snapshot collector. It reconstructs exact
portfolio cash and contract quantities from authenticated bootstrap candidates
and durable terminal receipts within one SQLite read transaction. It checks
FIFO fills, commissions, settlement links, compatibility state, portfolio
coverage, and database identity before issuing sealed in-process evidence.
The shared bootstrap reader now supports the same transaction and explicitly
queries the main schema so temporary tables cannot hide persistent portfolios.

Regression tests exercise actual synthetic bootstrap and settlement history,
filled and rejected outcomes, cash/quantity rewrites, incomplete lineage,
concurrent writers, database replacement, cancellation and explicit pool
recovery, freshness measured before validation, and nested type substitutions.
A new test reproduced a temporary portfolios table hiding an unbootstrapped
portfolio; main-schema qualification fixes it. No historical user database was
modified. Snapshot collection itself makes no writes.

Validation: 166 related bootstrap, reconciliation, settlement and snapshot tests
passed (1 warning). The broader safety, security, risk and entry-contract run
passed 1003 tests with 4 existing skips and 1 warning before the final
main-schema qualification; the relevant 166-test run includes that correction.
Black, changed-file Flake8 and whitespace checks pass. Independent review
covered the snapshot and its mutation, FIFO link, freshness and cancellation
boundaries. Logs: work/paper-risk-snapshot-regressions.log and
work/paper-risk-snapshot-safety.log.

This establishes cash/quantity evidence only. Current quotes and valuations,
signed allocation evidence, pending reservations, daily-risk startup replay,
baseline BUY settlement and final gateway admission remain unconnected. The
paper readiness constant remains false; no trading process was started and no
performance or profitability improvement is claimed from this correctness work.
