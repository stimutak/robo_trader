# PR4C Runtime FIFO Settlement

Date: 2026-07-30

## Decision

An authenticated local-paper fill is appended to the sealed FIFO accounting
epoch in the same `BEGIN IMMEDIATE` transaction that writes the compatibility
trade, position, account, exact-state, terminal-settlement, and immutable link
rows. The transaction either commits all of those records or none of them.

This slice does not create or infer an accounting epoch. A runtime allocation
must have exactly one epoch established by the separately reviewed PR4B
bootstrap. Missing or ambiguous epoch authority fails closed.

## Evidence boundary

The sealed local `PaperExecutor` is the only runtime producer connected by this
slice. On a successful execution it emits:

- an opaque deterministic execution identifier;
- exact `Decimal` quantity and price;
- an aware UTC occurrence time; and
- an exact signed commission in USD minor units with a versioned producer
  source.

The current executor's explicit cost model reports zero commission. Zero is
therefore producer evidence, not a database default. The FIFO bridge supports
nonzero commissions and rebates, and its tests exercise both.

No order intent, position delta, compatibility trade row, or broker quote can
be converted into fill evidence. Positive outcomes without the producer-owned
evidence are rejected before settlement.

## Atomic write and replay

`FifoLedger.record_fill_in_transaction` leaves commit and rollback ownership
with the terminal-settlement transaction. The runtime bridge assigns the next
gap-free epoch sequence while that transaction holds SQLite's immediate write
lock, appends the fill and commission, derives FIFO lots/matches/snapshot, and
recomputes the complete epoch integrity chain. Before any mutation, it also
revalidates the durable FIFO tables and exact trigger bodies inside that same
transaction, preserving PR4B's schema-substitution protection.

The FIFO projection supplies the compatibility trade P&L, remaining quantity,
remaining cost, average cost, and total realized P&L. Settlement refuses to
commit if those values differ from the already-authorized exact terminal
request. Total realized P&L is the epoch baseline plus every lot match through
the current event sequence across all instruments; an instrument-local
snapshot is never treated as the account-wide total.

After acquiring the immediate write lock and before appending a fill, the
settlement path re-authenticates every non-FIFO table and trigger it can mutate.
It rejects temporary shadows and any persistent hot-table trigger other than
the exact append-only guards, matching trigger targets case-insensitively as
SQLite does. The audit compares complete canonical table and explicit-index
DDL, including constraints, uniqueness, foreign keys, defaults, and required
indexes. SQL normalization preserves quoted literal contents. The exact
multiuser-v1 compatibility definitions for `positions`, `trades`, and
`account` are separately allowlisted for both its populated-table rename path
and its partially initialized direct-create path, so an already-supported
upgraded ledger is not quarantined merely for predating the fresh-database
DDL. Every settlement statement is explicitly qualified to the durable `main`
schema.
The FIFO ledger separately performs the same in-transaction protection for all
FIFO tables and triggers. Thus a temporary settlement-link table cannot divert
lineage, a same-shaped table cannot substitute altered constraints, and an
injected trade trigger cannot add an unreviewed compatibility row after
projection checks.

Every fill-local and epoch-wide realized-P&L subtotal is accumulated in an
explicit 96-digit local Decimal context, independent of same-process ambient
context changes.

`paper_fifo_settlement_links` binds each terminal settlement to exactly one
FIFO fill, sequence, execution ID, commission source, and state fingerprint.
It is append-only and protected by uniqueness and composite foreign keys.

An exact retry authenticates the existing event payload, complete epoch, and
link, then returns the original terminal receipt without adding a fill or
changing a projection. An identity collision with different evidence fails
closed.

## Crash recovery

Stopped-system crash recovery now requires the linked FIFO event in addition
to the existing terminal outbox and compatibility projections. It opens the
ledger read-only, enables foreign-key validation, authenticates the complete
FIFO schema and epoch, recomputes the immutable event and snapshot chain, and
compares the link and exact projection with the terminal request. Missing,
ambiguous, or tampered FIFO evidence leaves the safety reservation quarantined.

## Current limitations

- The active local-paper executor still produces one complete fill. Partial and
  nonterminal outcomes remain quarantined; their order-lifecycle integration is
  not enabled by this PR.
- This slice does not authorize an IBKR write path or a live order.
- It does not apply the offline bootstrap to an operational database.
- It does not perform the backup/restore or broker-reconciliation drills needed
  to complete PR4 and Gate A.

The system remains paper-only and IBKR read-only. Gate A remains closed.
