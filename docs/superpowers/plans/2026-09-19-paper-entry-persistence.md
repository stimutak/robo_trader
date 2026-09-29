# Paper Entry Persistence Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in the existing worktree. Steps use checkbox syntax. The user's standing instruction is to continue implementation autonomously; do not add a plan-approval pause or commit without authorization.

**Goal:** Persist a claimed local-paper opening BUY exactly once, preserve FIFO/cash/position continuity across BUY and reduction recovery, and release reserved capacity only after durable settlement and independently verified daily accounting.

**Architecture:** Keep the existing `paper_reduction_settlements` table as the common terminal envelope, adding an explicit `settlement_kind` discriminator. Its historical name is retained to preserve the foreign-key graph, row identities, payload fingerprints and insertion ordering. Separate strict entry and reduction parsers retain their contracts; no entry data is cast into a reduction request. The new entry writer shares the actual database transaction with FIFO and compatibility projections, and stays inaccessible from the runtime until the full execution/release path passes review.

**Tech Stack:** Python 3.10+, aiosqlite, SQLite, exact Decimal accounting, existing sealed capability and journal patterns.

**Spec:** `docs/ROBOTRADER_REMEDIATION_PLAN_2026-07-20.md`, especially the Gate-A/PR7 entry contract and September entry accounting/claim/record additions. Also read `docs/PAPER_READINESS_EXECUTION_2026-09-13.md`.

## Global Constraints

- Runtime execution is **paper-only and IBKR read-only** until the remediation plan's live-trading release gate is explicitly completed.
- Start or restart the system only with `./START_TRADER.sh`.
- Never delete or rewrite trading history, positions, equity history, account data, or other user data without explicit user approval and a verified backup.
- `PAPER_TERMINAL_SETTLEMENT_READY = False` and unconditional baseline entry rejection remain until the integrated path is independently verified.
- No automatic commit. Tests use temporary databases; operational migration/deployment remains a separate gated action.
- Only full FILLED or zero-fill REJECTED/CANCELLED/EXPIRED entry outcomes settle. Baseline entry is integral, account-wide gross-flat, explicit zero commission, within reserved principal, and tied to one consumed entry claim.

## Review Focus

- BUY/reduction/BUY after a complete close: retained zero-position metadata must not prevent a new entry, and mixed history must reconstruct cash, FIFO and provenance in commit order.
- Lost response after commit: retry returns the same authenticated receipt and adds no second trade/FIFO event; changed content under any reused claim/order/execution identity rejects.
- Opposing quantities in other portfolios: net-zero must still reject entry when gross exposure is nonzero.
- Crash at every mutation boundary: rollback restores every participating table; a committed outbox must recover without re-execution.
- Restored/copied/replaced databases and forged evidence: receipt database identity, schema, execution provenance, scope and journal links must all match before mutation or release.

## Task 1: Discriminated terminal storage and strict historical dispatch

**Files:** `robo_trader/database_migrations.py`, `robo_trader/database_async.py`, new `robo_trader/paper_settlement_record_dispatch.py`, new `tests/test_paper_settlement_kind_migration.py`.

**Interfaces:** Extend the existing terminal row with `settlement_kind TEXT NOT NULL DEFAULT 'REDUCTION' CHECK (settlement_kind IN ('REDUCTION','ENTRY'))`. Add `parse_terminal_record(kind, payload_json, fingerprint, *, journal)` returning the exact existing reduction request or validated `PaperEntryTerminalRecord`. Historical parsing never returns execution authority. Reduction receipt consumers continue requesting and checking kind REDUCTION until mixed support lands.

- [x] Add temporary-database tests for fresh creation and v3 upgrade. Populate a real existing reduction settlement, save all old column values, apply v4, and assert old values and request/receipt hashes are unchanged; kind is REDUCTION. Repeat migration and compare again.
- [x] Assert an unknown or NULL kind fails SQL insertion; deleting/replacing the check constraint is rejected by `assert_paper_settlement_hot_schema`; temp-table shadows remain rejected.
- [x] Add migration v4 using additive ALTER TABLE, update exact/hot schema contracts and both ordinary and bootstrap initialization paths. Do not modify prior migration functions to pretend historical databases already contained the new discriminator.
- [x] Dispatch on exact string enum values. ENTRY calls `validate_stored_paper_entry_terminal_record`; REDUCTION retains `PaperTerminalSettlementRequest.from_canonical_payload` and its existing fingerprint checks. Unknown kinds fail closed. Each parser must reject the other's payload.
- [x] Run the new migration/dispatch tests plus exact bootstrap, existing terminal persistence/replay and schema-containment suites before proceeding.

## Task 2: Entry receipt and atomic writer

**Files:** new `robo_trader/paper_entry_persistence.py`, `robo_trader/database_async.py`, `robo_trader/paper_entry_settlement.py`, `robo_trader/paper_execution_capability.py`, new `tests/test_paper_entry_persistence.py`.

**Interfaces:** `AsyncTradingDatabase.commit_paper_entry_outcome(record, *, runtime_contract, journal, execution_dispatch)` returns a producer-owned `PaperEntrySettlementReceipt`. Receipt fields: settlement_id, record, trade_id, database_path, database_identity, database_device, database_inode, committed_at, receipt_fingerprint. `execution_dispatch` must be a consumed one-shot entry dispatch; plain records, dictionaries or reduction dispatches cannot authorize this method. IDs remain `pset-<32 hex>` so existing state lineage formats remain valid.

- [ ] Implement the separate entry capability and consumed-dispatch types using the existing sealed registry pattern before the writer accepts them. Bind actual claim/order/quote/runtime and issue the consumed dispatch only from a one-shot local PaperExecutor invocation. Exercise real issuance in tests; do not invent a test-only authority bypass. Final production gateway attachment remains Task 4.
- [ ] Build tests using an actually bootstrapped temporary exact-state/FIFO database. Capture snapshots of every participating table before each request.
- [ ] For six shares at 333 with cash 100000 and mark 333, assert cash 98002, position 6, cost/mark 333, realized/daily unchanged, exactly one BUY trade, one FIFO fill, one terminal row of kind ENTRY and one FIFO link. Check the stored canonical record through the strict parser.
- [ ] For each zero-fill terminal status assert no trade/FIFO fill, identical money/position values, one terminal/outbox row and unchanged position source. Account history must still identify the committed terminal event.
- [ ] Reject stale account values, changed flat-position metadata, nonzero gross exposure across portfolios, missing sealed FIFO epoch, incompatible schema, altered DB inode, wrong account/domain/portfolio, future terminal time and unauthenticated execution dispatch before any mutation.
- [ ] Under `BEGIN IMMEDIATE`, authenticate database descriptor and hot schema, check every idempotency identity across both kinds, then load current exact/legacy pre-state. Revalidate the proposed record against actual journal history and current state; verify consumed dispatch binds its claim, order, symbol, contract, quantity, quote and terminal execution.
- [ ] Append `RuntimePaperFillEvidence` with side BUY through `append_runtime_fill_on_aiosqlite_worker` on the same connection. Require a fresh FIFO result, expected total signed quantity, zero fill realized P&L, matching portfolio total realized P&L and exact new average cost.
- [ ] Insert trade and kind ENTRY terminal row, update/insert exact and compatibility position/account projections, insert existing FIFO link with the entry record fingerprint, and bind source IDs. Recheck descriptor before commit. On any BaseException roll back; never release capacity from inside this transaction.
- [ ] Inject faults after begin, FIFO, trade, position, account, terminal row, FIFO link and before commit; compare all captured table snapshots. Simulate a lost committed response and authenticate exact replay without reapplying money/FIFO. Cross-kind ID collisions must reject.

## Task 3: Mixed recovery and all read models

**Files:** `robo_trader/database_async.py`, `robo_trader/risk/paper_ledger_snapshot.py`, `robo_trader/reconciliation/bootstrap_producer.py`, `robo_trader/paper_reduction_gateway.py`, `tests/risk/test_paper_ledger_snapshot.py`, `tests/test_paper_settlement_replay.py`, new `tests/test_paper_mixed_settlement_history.py`.

**Interfaces:** A mixed terminal iterator returns discriminated authenticated receipts in durable common-table order. Existing reduction-only iterator must explicitly filter/check REDUCTION and must not silently serve as complete daily history. Complete-history consumers switch to the mixed iterator before entry is wired.

- [x] Exercise bootstrapped flat state -> BUY -> full reduction -> BUY with actual FIFO and exact money updates. Reopen DB between every step, parse both receipt types and assert cash/quantity/source continuity. (Storage integration with owned entry accounting/release; runtime gateway cycle remains Task4.)
- [ ] Extend snapshot replay to validate both types, count BUY principal toward daily filled notional, authenticate every FIFO link/trade exactly once, and allow a previously absent symbol only for a verified flat entry. Reject missing rows, changed kinds, orphan trades/links and reordered/discontinuous pre-state.
- [ ] Extend outbox startup replay and independent daily verifier ingestion to both receipt types. Kill after commit before daily projection: startup consumes exactly once, and verifier refusal quarantines with capacity retained.
- [ ] Audit `get_paper_account_settlement_state`, `get_position`, `get_positions`, `get_account_info`, reconciliation bootstrap producer, and both snapshot FIFO correspondence checks. Identity-only joins can retain the common table; payload readers require explicit kind dispatch. Do not use COALESCE to mask missing evidence.
- [ ] Keep existing reduction receipt reconstruction and old records byte-for-byte compatible. Run all persistence/replay/bootstrap/risk/accounting suites plus mixed-history regressions.

## Task 4: Consumed execution, verified release, final gateway assembly

**Files:** `robo_trader/paper_execution_capability.py`, `robo_trader/paper_reduction_gateway.py`, `robo_trader/paper_reduction_submitter.py`, `robo_trader/safety/entry_capacity.py`, `robo_trader/safety/journal.py`, corresponding capability/gateway/risk tests.

**Interfaces:** Entry authority is separate from reduction authority. It binds the durable ENTRY_SUBMISSION_CLAIMED event, exact reservation, frozen admission evidence, refreshed producer-owned quote and validated runtime. Settlement consumes a dispatch produced by that one invocation. Release consumes a genuine committed entry receipt only after daily accounting is independently confirmed.

- [ ] Claim before dispatch; reject reused/copied/wrong-task authority, changed order fields, reservation expiry before claim, stale quote, changed transport generation or changed journal head. Keep task-owned account lock over final admission, reservation, execution, settlement and daily projection/release.
- [ ] Submit only through the local PaperExecutor BUY path with exact principal ceiling and zero commission contract. Never call IBKR order placement. Exceptions or uncertain outcomes retain/quarantine the claimed reservation; they cannot make the capability reusable.
- [ ] Release journal capacity only for the exact committed entry receipt after outbox projection and verifier acceptance. Fault between each stage and recover without submitting again. Zero-fill release still requires its committed terminal evidence.
- [ ] Run BUY -> protective SELL -> BUY integration through the actual gateway with journal, database, FIFO, daily ledger and quote producer, followed by a simulated process restart. Test verifier denial, DB failure, lost response, gross held exposure and concurrent attempts.
- [ ] Complete independent review and full suite. Do not infer launch permission from green unit tests: verify every operational gate in the canonical roadmap, including actual account allow-list, runtime volume units, independent verifier deployment, backup/restore, reconciliation and immediate launch consent.

## Validation commands

Use `/Users/oliver/Projects/robo_trader/.venv/bin/python -m pytest <affected tests> -q` and preserve failing-before evidence for each task. Use Black/Flake8 on affected Python files and `git diff --check`. Run the full suite once the integrated changes pass focused tests. No commit command is authorized by this plan.

## Current status

Task 1 is implemented: additive v4 migration and strict parser dispatch. Task 2 has a transaction staging engine, strict stored-row recovery and owned committed-receipt recovery; the authorized runtime commit adapter and consumed entry capability remain open. Task 3 has mixed outbox replay, daily BUY-notional ingestion and same-snapshot entry cash/position/FIFO reconstruction. Reduction persistence now validates mixed history in its existing transaction. Complete runtime mixed-cycle and bootstrap/release integration remain open. Task 4 remains open. Runtime entry rejection and the readiness gate remain unchanged. Full-suite results are recorded in the execution report.


### Task 2 execution note

The transaction engine is implemented and covered by33 storage tests; Task2 is
still open. It is deliberately not an authorized runtime writer. Implementation
order was adjusted to test the storage transaction before attaching genuine
consumed entry authority; see the September19 execution report and ledger.
Durable retry authentication, path-bound commit lifecycle, producer-owned receipt
and final one-shot dispatch remain required, followed by Tasks3–4.

Task2 further progress: strict read-only terminal/trade/FIFO validation and exact
storage retries are implemented (119 focused tests pass). Staging now binds
pathname and descriptors until its savepoint completes. Public committed-receipt
issuance and consumed execution remain incomplete; no task completion is inferred
from the internal storage result.

Owned committed-receipt recovery is now implemented through an independent
read-only connection and tested across the commit boundary (127 focused checks).
The authority-checking public commit adapter and consumed entry dispatch remain
open; the recovered receipt alone grants neither submission nor release.

Task3 is partially implemented: mixed terminal outbox and BUY daily-notional
projection now consume the owned entry receipt.70 focused checks pass. Risk
snapshot, bootstrap crosslink and reduction writer guards remain in place pending
complete mixed cash/quantity/FIFO reconstruction. Task3 is not complete.


Risk snapshot now reconstructs ENTRY and REDUCTION history in its own read
transaction, with strict entry storage/journal checks and exact FIFO
correspondence. A new bootstrapped-entry regression also fixed missing position
origin lineage.75 focused checks pass. Full mixed BUY/SELL/BUY, reduction-writer
integration, bootstrap/release crosslink and complete trade coverage remain open;
this does not complete Task3 or authorize runtime BUYs.


Reduction storage now supports an existing ENTRY only after complete same-
transaction bootstrap/cash/quantity/FIFO reconstruction. BUY→fullSELL storage,
restart and idempotent retry pass.85 focused checks pass. Pending entry capacity
is deliberately retained: the next step is owned accounting confirmation and
receipt-bound release with replay support, then the complete BUY→SELL→BUY cycle.
No runtime permit or journal bypass was introduced.


Task4 release prerequisite is implemented: fresh owned entry accounting
confirmation verifies the committed receipt, idempotent principal ingestion and
independent daily-total read (including zero fills). It is one-use and bound to
exact producer/database/ledger identity, with5s wall/monotonic lifetime measured
from verification start.34 focused checks pass. Journal release/event/replay and
full gateway attachment remain unimplemented; this proof alone changes no
reservation or execution authority.


Entry release/event/replay is implemented and pending-capacity summaries now
remove exactly validated releases. Exact retries are bound to committed outcome
fields and require fresh accounting confirmation. The bootstrapped storage
BUY→release→SELL→BUY cycle passes with reopen at each stage and daily total6000.
103focused checks pass. Bootstrap ENTRY-release crosslink and runtime consumed
execution/commit/gateway integration remain open; the bootstrap guard is retained
because journal structural replay alone cannot authenticate database truth.


Bootstrap entry crosslink is now implemented: released entries must match actual
journal parents, strict stored terminal/trade/FIFO evidence and exact released
outcome fields. Pending entries still block. Async recovery and synchronous
bootstrap share one read-only SQL/FIFO validator; held bootstrap collection and
its final file/snapshot validation are tested.101focused checks pass. Complete
global trade coverage and runtime consumed-execution/public-commit/gateway
integration remain open; Task3 and Task4 are not complete.
