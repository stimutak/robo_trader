# PR4A exact FIFO accounting foundation

## Status and safety boundary

This slice is dormant. It adds an exact append-only accounting contract and a
deterministic projector, but does not connect either to the runner, settlement,
broker, dashboard, startup script, or `trading_data.db`.

The only migration entry point is named `migrate_fifo_fixture_database`. It
accepts an in-memory database or a database whose filename ends in
`.fifo-fixture.sqlite3`; an ordinary production-style filename is rejected.
Production migration, backup/restore, legacy adoption, and runtime settlement
belong to later PR4 slices and require separate review.

In particular, PR4A cannot create or project a
`LEGACY_AGGREGATE_OPENING_BALANCE` epoch. Establishing that prospective legacy
cost-basis boundary requires an operator-reviewed candidate, verified backup,
and explicit user authorization in PR4B.

## Exact value convention

- Quantities, prices, gross P&L, realized P&L, and open position cost are
  canonical fixed-point decimal strings. Application inputs must be finite
  `Decimal` values with at most 38 digits and 18 fractional places. Floats are
  rejected.
- Commissions are signed integer USD cents. Positive values are expenses;
  negative values are rebates. Every cent is allocated exactly.
- Times are canonical UTC strings with microseconds and a trailing `Z`.
- Fill order is an explicit, gap-free epoch sequence. Event time cannot move
  backwards. Equal timestamps remain deterministic because the sequence is the
  primary ordering authority.

## FIFO and commission rules

Each BUY contributes positive quantity and each SELL contributes negative
quantity. A fill first closes opposite-direction lots in opening order. Any
remainder opens one new lot, so a single fill can cross through zero without
losing provenance.

The fill commission is allocated across match fragments and any new opening
fragment by integer largest remainder, with FIFO segment order as the stable
tie-breaker. An opening lot's commission is allocated cumulatively as it is
matched; the final match receives the exact remainder. Therefore allocations
are deterministic, replayable, and sum exactly to both the fill commission and
the opening-lot commission, including partial fills and rebates.

Realized P&L is:

`directional gross P&L - allocated opening commission - allocated closing commission`

No commission is embedded in a fill price, so costs are not counted twice.

## Durable records and invariants

The fixture schema contains append-only records for:

- accounting epochs;
- fills and one exact commission event per fill;
- lot openings;
- FIFO lot matches;
- chained position snapshots; and
- ordered migration evidence.

Primary keys, scope-local sequence/idempotency/execution uniqueness, composite
foreign keys, direction/state checks, and no-update/no-delete triggers are
enforced by SQLite. `FifoLedger.verify_epoch_integrity()` recomputes fill
fingerprints, sequence/time ordering, commission conservation, lot quantities,
gross/net P&L, complete opening-commission allocation, and snapshot chains.

One `record_fill` transaction appends the fill, commission, derived lots and
matches, and snapshot atomically. Any exception rolls back the entire event.

## Later PR4 slices

1. PR4B: reviewed prospective legacy bootstrap using one explicitly marked
   aggregate opening lot per current position. It must not fabricate historical
   FIFO or commissions.
2. PR4C: atomic runtime settlement integration and compatibility projections.
3. PR4D: descriptor-bound WAL-safe migration, backup, restore, dry-run reports,
   and fault-injection evidence.
