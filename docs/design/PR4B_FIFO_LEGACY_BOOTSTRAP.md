# PR4B FIFO legacy bootstrap

## Status and authority boundary

This slice binds the already reviewed exact-state bootstrap to the dormant FIFO
accounting foundation. It remains an offline operator tool. It does not run at
startup, connect to a broker, start any service, authorize an order, or open
Gate A. Every preview and apply result continues to report
`authorizes_startup=false`.

The only apply path remains `scripts/bootstrap_exact_paper_state.py apply`. It
requires the existing lifecycle lock, proof that the runner, dashboard,
WebSocket server, and Gateway are stopped, exact destination-bound operator
confirmation, a specific reason, authenticated reconciliation and protective
mark artifacts, and a new descriptor-verified SQLite online backup. No
authoritative database is changed automatically.

## Truthful legacy epoch boundary

Each reviewed portfolio receives one append-only FIFO epoch whose
`origin_kind` is `LEGACY_AGGREGATE_OPENING_BALANCE`. Its source fingerprint is
the full candidate fingerprint. A separate lineage record binds the epoch to:

- the exact-state bootstrap ID and candidate fingerprint;
- the reconciliation snapshot and authenticated report hash;
- the authenticated read-only broker snapshot hash;
- the complete reviewed legacy-ledger snapshot hash; and
- the administrator action recorded by the same transaction.

Every nonzero legacy position becomes one `fifo_opening_balances` row and one
corresponding open lot. The signed legacy quantity determines `LONG` or
`SHORT`; the lot quantity is its absolute magnitude; and its price is the
reviewed aggregate legacy cost basis. The positive IBKR contract ID is copied
from the authenticated protective-mark artifact and must match the candidate
exactly.

No `fifo_fills`, `fifo_commissions`, execution IDs, or position snapshots are
created at bootstrap. The opening lot records `opening_commission_minor=0` only
to define prospective epoch accounting: it means pre-epoch commission history
is unknown and not reconstructed. It is not a claim that historical broker
fees were zero. Later fills consume these aggregate opening lots through the
ordinary deterministic FIFO projector.

## Pre-epoch account baseline

One append-only baseline record preserves the candidate's exact cash, realized
P&L, daily P&L, daily P&L baseline, and baseline date. Post-epoch FIFO realized
P&L therefore starts at zero without erasing or relabeling pre-epoch economics.

## Atomicity and rollback

Schema preparation, the administrator action, exact-state bootstrap records,
evidence-consumption receipts, exact account/position adoption, FIFO epoch,
lineage, baseline, opening balances, and opening lots are appended inside the
same existing `BEGIN IMMEDIATE` transaction. The source descriptor, safety
journal, candidate, evidence, and sealed backup are revalidated at the existing
boundaries. Any exception before commit rolls back the complete schema and data
change. The verified backup remains the rollback artifact; PR4D owns the
operational clean-room restore drill.

## Read-only candidate report

`preview` reports the deterministic epoch ID, candidate fingerprint, exact
pre-epoch baseline, each contract-bound opening balance and lot, zero synthetic
fills, and `pre_epoch_history_reconstructed=false`. Preview only reads and
revalidates the candidate, evidence, ledger, and safety journal; it does not
create FIFO schema or rows.

## Non-goals

- No runtime settlement integration (PR4C).
- No automated or operational migration/restore execution (PR4D).
- No broker write capability or live-trading behavior.
- No reconstruction of historical lots, fills, executions, or commissions.
- No mutation of legacy account, position, trade, or equity-history rows.
