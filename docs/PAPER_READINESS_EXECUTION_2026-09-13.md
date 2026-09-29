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

## Gateway account valuation integration

Portfolio-scoped entry serialization now collects the verified account ledger
under the existing account order lock before requesting quotes. Quote coverage
includes every held symbol, even when absent from the active stop list. The
active entry monitor authenticates current broker marks; exact contract IDs,
producer payloads, unique symbol records and current read-only transport
generation are checked before yielding entry context.

`entry_valuation` is accessible only to the task owning that context and
revalidates ledger provenance, quotes, generation and freshness on every access.
It derives portfolio and account equity from exact cash plus signed market
values, gross exposure from absolute values, and distinct held-symbol counts.
Arithmetic is independent of the ambient Decimal context. Ledger evidence has
a conservative five-second wall-clock and monotonic age bound including its
collection time. Context is cleared on normal exit, errors and cancellation.

Independent review identified and corrected an initial assumption that every
ledger portfolio has an executor registration: inactive cash and positions now
remain included without requiring execution registration. A test-only owned
multi-portfolio snapshot verifies inactive short valuation; the other tests use
real synthetic bootstrap history and exact monitor-produced quotes. No user
ledger or live runtime was modified.

Related gateway, protective-feed, failure-injection, snapshot and valuation
regressions: 121 passed, 1 warning in 10.24 seconds. Independent final review:
12 valuation tests passed, no remaining actionable defect. Black, changed-file
Flake8 and whitespace checks pass. Logs are retained in
work/paper-entry-valuation-regressions.log and the red-test evidence logs.

This connects ledger and quote collection to entry serialization, but does not
replace the runner's legacy risk validator or authorize BUY. Final submission
still needs fresh database/reconciliation revalidation, exact contract sizing,
sector/correlation/liquidity evidence, reservations, daily-risk replay and BUY
settlement. The readiness constant remains false and baseline BUY remains denied.

Broader post-change safety/security/risk/entry-contract/runner-event-time
verification: 1117 passed, 4 existing skips, 1 warning in 25.46 seconds
(work/paper-entry-valuation-safety.log). The skips are two unavailable Docker
Compose checks and two pre-existing conditional/incomplete integration checks;
they are not evidence of operational launch readiness.

## Gateway durable daily accounting integration

The gateway now accepts an exact PaperFillAccounting dependency. When supplied
at construction, startup validates complete authenticated bootstrap portfolio
coverage, replays the terminal outbox under the shared account order lock, and
performs independently authenticated reads for every scope before accounting
is ready. Empty outboxes do not skip authority validation. Explicit preparation
supports the same replay boundary. Cancelled or failed preparation leaves
entry accounting unavailable.

Task-owned entry contexts can request a fresh exact daily total, with portfolio
coverage checked and valuation/quote freshness revalidated after the awaited
read. Configured terminal accounting ingests each committed settlement receipt
before the safety journal releases the order, inside the existing cancellation-
drained completion task. Ingestion failure leaves accounting unavailable and
triggers existing gateway quarantine, preserving the committed outbox and the
unreleased journal reservation. Replay can recover the risk projection exactly
once; it does not clear gateway quarantine or authorize restart.

The adapter revalidates the constructor-bound risk database and anchor paths
for both reads and terminal ingestion. A regression reproduced redirection to
a different otherwise-valid risk ledger after initialization; the binding check
now rejects it. All tests use synthetic data and the existing test-only
monotonic authority. No test authority was installed in production.

Related accounting, valuation, gateway and failure-injection suites: 83 passed,
1 warning in 7.06 seconds. Tests include cancelled replay, independently denied
empty-state reads, terminal failure, cancellation drain, and idempotent recovery.
The gateway integration still requires production verifier/configuration
construction and runner injection. Reduction-only operation can omit the
adapter, but then no daily entry evidence is available. Baseline BUY remains
denied and paper readiness remains false. No services were started.

Broader post-change verification: 1216 passed, 4 existing skips, 1 warning in
28.10 seconds (work/paper-gateway-accounting-safety.log). Independent review
passed the initial 59-test set and 5 final targeted failure/binding tests with
no remaining actionable defect. Black, changed-file Flake8 and whitespace
checks pass.

## Bootstrap-day daily-history completeness

Runtime integration review exposed a distinction between successful outbox
replay and complete daily execution history. Authenticated bootstrap cash and
positions do not establish gross executions earlier on the bootstrap date.
Previously, an empty post-bootstrap outbox could yield zero daily notional for
that date. A regression reproduced this before the correction.

Owned ledger snapshots now carry each portfolio's authenticated bootstrap
effective timestamp, with exact scope coverage and mutation/future-time checks.
Entry daily accounting refuses the bootstrap's New York calendar date and any
earlier date. Later complete dates can use replayed terminal history. The daily
read is pinned to one explicit gateway UTC timestamp and its New York date is
checked again after the await. No legacy float trade rows are converted into
authoritative executions and no historical fill identifiers are invented.

This means a newly bootstrapped portfolio cannot use entry daily evidence until
a later New York date. Supporting its bootstrap date would require a separate
authenticated historical-execution import/completeness proof, which is not
implemented. Operator readiness must account for this boundary; successful
replay alone does not waive it.

Related snapshot, valuation, accounting, gateway and failure-injection suites:
111 passed, 1 warning in 17.02 seconds. Calendar tests cover UTC-midnight false
rollover, actual New York midnight, and spring/fall DST boundaries. A positive
entry read verifies that the gateway timestamp selects the intended day even
if the ledger's default clock differs. Independent review: 9 history-focused
tests passed, no actionable defect. Logs: work/paper-history-boundary-red.log,
work/paper-history-boundary-regressions.log and broader verification log.

Exact entry configuration binding, pending reservations, full evidence assembly,
production verifier construction/injection and baseline BUY settlement remain
open. No services or real-money execution were enabled.

Broader post-change safety/security/risk/runner/gateway verification: 1224 passed,
4 existing skips, 1 warning in 22.30 seconds
(work/paper-history-boundary-safety.log). Black, changed-file Flake8 and whitespace
checks pass. This is development evidence; paper readiness remains false.

## Durable symbol exposure and cooldown evidence

The task-owned gateway entry context now retains the requested symbol. Its
fresh valuation includes that symbol's absolute gross exposure across all
account portfolios, explicit held-position presence, and a durable allowed-at
cooldown timestamp. Long and short holdings never cancel in gross exposure;
inactive portfolios remain included. These are held-state values, not proof of
absent pending reservations.

The existing runtime's ten-minute BUY/SELL churn policy is derived from verified
nonzero terminal fill times across portfolios. The latest authenticated account
bootstrap provides a conservative lower bound when earlier symbol history is
unknown. Rejected/zero-fill outcomes do not extend this fill-based cooldown.
The risk contract must still combine this evidence with pending reservations
and enforce the resulting decision at final submission; BUY remains disabled.

Focused snapshot, valuation, accounting and exact entry-contract regressions:
351 passed, 1 warning in 10.96 seconds. Tests cover current holdings, real
committed fill history, rejected outcomes, a fill in an inactive portfolio, and
opposing signed holdings. Independent review: 4 targeted tests passed, no
actionable defect. The open PR list was refreshed; no separate pending entry
integration was available. No remote branch was merged or pushed.

Broader post-change safety/security/risk/runner/gateway verification: 1228 passed,
4 existing skips, 1 warning in 34.83 seconds (work/paper-entry-history-safety.log).
Black, changed-file Flake8 and whitespace checks pass. Operational launch and
profitability remain unverified; the paper readiness constant remains false.

## Explicit entry policy configuration

Added non-authorizing Config-level binding to the exact entry contract. Existing
position, sector, correlation, leverage, order, daily and slot limits map into
validated limits. The stricter correlation threshold applies. Three additional
portfolio/liquidity settings require explicit fixed-decimal values; freshness
settings default to five seconds and may only tighten that bound. Partial or
invalid explicit policy fails configuration load. Missing policy remains
unavailable. No operational thresholds were chosen or user environment edited.

Focused configuration and contract verification: 349 passed, 1 warning in
3.80 seconds. Independent review: all 18 new tests passed, no actionable defect.
Broader risk/security/entry/gateway/routing/settlement verification: 1012 passed,
4 existing skips, 4 warnings in 26.18 seconds. Black, changed-file Flake8 and
whitespace checks pass. Logs: work/entry-runtime-policy-regressions.log and
work/entry-runtime-policy-safety.log.

This slice does not consume runner-instance overrides or grant submission
authority. Override resolution, reservations, full evidence assembly, production
verifier injection and baseline BUY settlement remain open. A fresh read-only SSH
attempt to blackm5mbp again timed out on port 22; no remote changes were made.
The actual approved paper-account identifier remains outstanding. Paper readiness
is false and profitability remains unverified.

## Runner and portfolio policy resolution

The runner now has a current-limit resolver selecting exactly one active
portfolio from loaded configuration. It applies position/slot overrides and
runner order/daily caps to an isolated copy, retains the stricter runner/config
correlation threshold, and validates the result through the exact contract.
None inherits; zero, booleans, nonfinite values and invalid portfolio limits
fail closed. Missing/duplicate/inactive portfolio selection is rejected.
Rebuilding observes current overrides without mutating shared configuration.

Focused runner/config/contract regressions: 328 passed, 1 warning in 8.71 seconds.
Independent review: all 14 new tests passed, no actionable defect. Initial red
run confirmed the missing resolver. No admission capability is granted: final
submission still needs reservations, complete evidence and atomic BUY settlement.
Logs: work/runner-entry-policy-red.log and
work/runner-entry-policy-regressions.log. Operational gates remain unchanged.

Broader risk/security/entry/gateway/routing/settlement verification: 1026 passed,
4 existing skips, 4 warnings in 49.92 seconds (work/runner-entry-policy-safety.log).
Black, changed-file Flake8 and whitespace checks pass. Paper readiness remains false.

## Durable entry-capacity journal foundation

Added a separate reservation-only event family and risk-side adapter. The adapter
consumes an owned approved BUY decision after journal transaction/replay/head and
binding checks. Core persistence/replay remains standard-library-only. Records
include exact notional/quantity, contract, sector, quote/generation and lifetime;
an opaque intent-derived key retains durable duplicate detection. Same-symbol
entry conflicts and both directions of same-contract reduction conflicts block.
A failed commit rolls back the event and leaves the decision consumed.

Reservations survive expiry and restart. Coordinator startup rejects them.
Tests exposed bootstrap ignoring them and operator status returning CLEAN; both
consumers now block/report unresolved capacity. Offline reduction recovery also
refuses to treat an entry reservation as recoverable reduction authority. Existing
empty-journal migration already rejects any history. Older journal readers reject
this new event; rollback must preserve history and use a compatible reader.

Related journal/runtime/bootstrap/status/risk regressions: 590 passed in
13.15 seconds. Independent review: 19 reservation and 4 dormancy tests passed.
Coverage includes concurrent expected-head races, rollback, both conflict
directions, restart, expiry, read-only status and semantic-invalid persisted
payloads whose hash chains are valid. Black, changed-file Flake8 and whitespace
checks pass. Logs: work/entry-capacity-full-regressions.log and focused red/green
logs. Full repository verification is recorded below.

This is a durable reservation foundation, not entry admission. Pending aggregation,
head-bound complete account/sector/market evidence, authenticated terminal release
and recovery, verifier injection, and BUY settlement remain open. No submission
permit, production adapter invocation, service start or user-data mutation was
introduced. Paper readiness remains false.

Full repository verification completed: 3729 passed, 4 existing skips,
19 warnings in 303.96 seconds (work/entry-capacity-repository.log). No tests were
skipped or timeouts relaxed for this change. Final containment inspection confirms
PAPER_TERMINAL_SETTLEMENT_READY=False, baseline BUY rejection, and no production
call site for the new adapter. A fresh read-only SSH probe still timed out on
blackm5mbp port 22; no remote changes occurred.

## September 14: pending-capacity read model and gateway context

Added exact pending principal totals across account/portfolio/symbol/sector,
including inactive portfolio reservations and the union of held/pending symbols
for slots and duplicate detection. Unresolved reservations never age out. Cash
and daily totals are portfolio-scoped; buying power and gross are account-scoped.
The gateway obtains bound read-only coordinator replay before collecting ledger
and quotes, pins the journal head in its task-owned context, checks it before
yielding and on every pending read, and revalidates valuation after asynchronous
reads. Pending portfolios without ledger coverage and unresolved reductions block.
Cancellation drains journal workers before releasing the account gate.

Related risk/safety/gateway/routing/contract verification: 761 passed, 1 warning
in 14.31 seconds. Independent review: 9 focused tests passed, no actionable defect.
Tests cover exact arithmetic at precision 2, account/portfolio scoping, inactive
reservations, held/pending slot union, task ownership, journal changes during
quotes and after context creation, missing portfolio coverage and cancellation.
Logs: work/pending-capacity-related.log and initial focused red/green logs.

Final admission must still bind the complete decision inputs to this head and
atomically compare it when reserving. Commission/fee allowance, executable price
ceilings, authenticated terminal release/recovery and BUY settlement remain open.
No adapter submission call, paper readiness change or operational launch occurred.

Broader risk/safety/security/gateway/routing/settlement/status/bootstrap checks:
1294 passed, 4 existing skips, 1 warning in 28.52 seconds
(work/pending-capacity-safety.log). Black, changed-file Flake8 and whitespace
checks pass. This is component integration evidence; paper readiness remains false.

## September 14: deterministic paper execution pricing

Tracing the actual paper model confirmed producer-explicit zero commission and
slippage plus 0.0001 tick rounding. A new regression reproduced an authorized
fill changing from 12.3457 to 12.3456 under ambient Decimal precision 6 with
ROUND_DOWN. Replaced the Decimal sink arithmetic with exact integer ratios and
one half-even tick rounding. This fixes the execution result and provides one
shared model for proposed entry pricing; no commission schedule was invented.

Registered gateway bindings now retain the exact executor for a task-owned
entry_execution_cost read. It revalidates quote/ledger evidence, checks current
slippage and returns the modeled fill, explicit zero fee and max(reference, fill)
ceiling without changing quote evidence. Risk sizing and the reservation/final
execution boundary still need to bind this cost model; BUY remains disabled.

Related pricing/Decimal-sink/gateway/submitter/valuation tests: 164 passed,
1 warning in 8.39 seconds. Independent review: 47 focused tests passed, no
actionable defect. Coverage includes half-even ties, signed slippage, hostile
Decimal contexts, invalid slippage, round-down ceilings and changed registered
executor settings. Logs: work/paper-cost-rounding-red.log,
work/paper-cost-model-tests.log and work/paper-cost-related.log.

Broader risk/safety/security/pricing/gateway/submitter/routing/settlement checks:
1304 passed, 4 existing skips, 1 warning in 26.92 seconds
(work/paper-cost-safety.log). Black, changed-file Flake8 and whitespace checks
pass. Paper readiness remains false; operational launch and profitability are unverified.

## September 14: execution ceiling in exact entry sizing

The owned risk evidence now requires an explicit positive Decimal execution-price
ceiling, includes it in its mutation-detection fingerprint, and rejects missing
ceilings or values below the validated source quote. Quantity, approved principal
and every capacity postcondition use the ceiling; the broker quote is unchanged.
The durable reservation adapter consequently records the ceiling-based principal.

A regression demonstrates $333 plus 25 bps slippage: at $2,000 capacity the contract
permits five shares and reserves $1,669.1625; six would cost $2,002.995. Tests also
cover precision-2 arithmetic, missing/understated evidence, mutation rejection and
real journal replay of the principal. Related contract/reservation regressions:
322 passed in 1.38 seconds. Independent review: 301 focused tests passed.
Broader risk/safety/security/pricing/gateway/submitter/routing/settlement checks:
1309 passed, 4 existing skips, 1 warning in 26.59 seconds. Black, changed-file
Flake8 and whitespace checks pass. Logs: work/entry-ceiling-red.log,
work/entry-ceiling-related.log and work/entry-ceiling-safety.log.

The shared test producer explicitly models zero slippage when no test ceiling
is specified; production receives no inferred ceiling. Final gateway evidence
assembly must provide the registered executor's verified cost and bind the same
policy/reference at reservation and submission. Complete admission, authenticated
terminal release/BUY settlement and operational gates remain open; readiness is false.


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
