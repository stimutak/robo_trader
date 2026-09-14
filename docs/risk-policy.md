# Paper entry risk policy

The canonical operating requirements are in
`ROBOTRADER_REMEDIATION_PLAN_2026-07-20.md`, Gate A. Paper startup remains gated;
IBKR must stay read-only. The pure contract in
`robo_trader/risk/entry_contract.py` does not grant submission authority.

## Exact sizing contract

The contract floors whole-share quantity at the minimum remaining capacity:
requested allocation, symbol exposure, sector exposure, portfolio gross exposure,
liquidity, available cash, buying power, daily gross filled notional, account leverage, and any
explicitly configured per-order notional cap. Calculations use exact Decimal
arithmetic and verify the result against every capacity. The Gate-A symbol
position cap cannot exceed 2% of portfolio equity.

`max_order_notional_usd` must be explicitly supplied. A positive Decimal enables
that optional cap; explicit None corresponds to disabling the optional
`RiskConfig.max_order_notional` setting. Missing required risk evidence never
means zero exposure or unlimited capacity.

For example, a $1,000 per-order cap at a $333 share price permits at most three
shares ($999), provided all other limits allow them. A cap below one share's
price rejects the entry.

Account leverage uses total account equity multiplied by the configured exact
leverage ratio (1 through 4), minus gross holdings across all portfolios and
pending account exposure. Long and short gross notionals must not cancel each
other. Every relevant capacity separately subtracts its pending symbol, sector,
portfolio, cash, buying-power, or daily-notional commitments. Balances must be
provided before reservation deductions so commitments are counted exactly once.
Missing account or pending amounts block entry; zero must be explicit evidence.

## Admission state

A valid snapshot must contain an exact nonnegative account-wide count of held
and reserved position slots, an explicit indication of whether the symbol already
has a position or pending entry, and the durable timestamp at which a new entry
is allowed. The runtime producer must collect these under the account-wide order
lock and revalidate immediately before submission. Unknown state rejects entry.

A new symbol is rejected at the configured maximum number of open positions.
Any existing position or pending entry in the symbol rejects a duplicate. A
cooldown blocks while the evaluation time precedes its expiry; equality permits
evaluation of the remaining limits. Snapshot freshness and all existing quote,
contract, transport, portfolio, and symbol checks still apply.

## Daily history boundary

Successful terminal replay proves post-bootstrap fills. Bootstrap cash and
positions do not prove gross executions earlier on that date. Entry daily
accounting therefore requires a date later than the portfolio's authenticated
bootstrap date in America/New_York, plus complete replay and a fresh independent
ledger read. Unknown bootstrap-day history is never treated as zero. Supporting
that date requires additional authenticated historical-execution completeness
evidence. The gateway pins each read to one timestamp and rejects a date change
during the read.

## Explicit configuration binding

`Config.build_entry_risk_limits()` builds non-authorizing exact limits from the
current configuration. It maps existing position, sector, correlation, leverage,
order, daily-notional and position-count limits. The stricter of the risk and
correlation configuration thresholds applies. A disabled daily cap is rejected;
an explicitly absent optional per-order cap is preserved. Legacy numeric policy
scalars use their decimal spelling; this conversion is not for market or ledger
evidence.

Three additional fixed-decimal environment settings have no inferred defaults:
`ENTRY_MAX_PORTFOLIO_GROSS_FRACTION`,
`ENTRY_MINIMUM_AVERAGE_DAILY_DOLLAR_VOLUME_USD`, and
`ENTRY_MAX_ORDER_FRACTION_OF_DAILY_DOLLAR_VOLUME`. Fraction settings are positive
and at most one; dollar volume is positive USD. Any explicit entry-policy setting
requires a complete valid policy at configuration load. No settings leaves entry
policy unavailable and does not authorize entries. Optional
`ENTRY_MAX_QUOTE_AGE_SECONDS` and `ENTRY_MAX_ACCOUNT_EVIDENCE_AGE_SECONDS` default
to five seconds and may only tighten that bound, with exact microsecond precision.

`AsyncRunner._current_entry_risk_limits()` resolves one active portfolio from
its loaded configuration. Missing, duplicate or inactive portfolio selection
fails closed. Portfolio position/slot overrides and runner order/daily caps are
applied to an isolated copy; None inherits the configured value, while zero or
invalid overrides are rejected. Runner correlation may tighten the configured
threshold. Each call rebuilds from current values without changing shared
configuration. Final admission must invoke this resolution under serialization
and consume the resulting limits; no returned object grants entry authority.

## Durable entry capacity reservations

The dormant risk adapter `risk.entry_reservations.reserve_entry_capacity`
consumes an owned approved BUY decision inside the journal write transaction,
after verifying the bound journal identity and expected sequence/hash head.
It persists an `ENTRY_CAPACITY_RESERVED` event, never a submission claim or
permit. A changed head rejects the reservation. A prior entry in the same symbol
or contract, or an unresolved reduction in the same contract, conflicts. The
reverse conflict also prevents reduction authority while that contract has
unresolved entry capacity. Distinct contracts can retain separate reservations.

Replay validates the versioned payload, exact positive quantity/notional,
contract, decision lifetime at reservation time, identity and hash chain.
Reservation expiry does not release capacity. Startup and bootstrap reject
unresolved entries; read-only operator status reports them as blocked. The
reduction-only offline recovery path cannot release them. No entry release API
exists yet, and the production gateway does not invoke the adapter.

This adds an event to the existing journal schema. Older readers fail closed
when encountering it; code rollback must retain a compatible journal reader,
and must never discard journal history. Core payload validation is a persisted
format contract and must remain compatible when the risk model evolves. The
existing journal hash chain is not an independent rollback anchor.

Before activation, bind account/portfolio identity, sector, pending totals and
all decision evidence to the journal head under account serialization. Add authenticated atomic terminal release/recovery,
and final execution-capacity rechecks. The reservation record alone proves
neither complete risk evaluation nor permission to submit an order.

## Pending-capacity read model

The gateway's task-owned entry context now pins the verified journal head before
collecting ledger/quote evidence and rejects any change before yielding. Its
asynchronous pending-exposure read replays the bound journal again and revalidates
quote/ledger freshness after the await. Reads run off the event loop; cancellation
drains the worker before releasing account serialization. Unresolved reductions
and pending portfolios missing from the authenticated ledger block the context.

Pending symbol, sector and buying-power totals cover every account portfolio;
cash and daily principal totals cover the entry portfolio. Account gross pending
notional covers all reservations, including inactive portfolios. Position slots
count the union of held and pending symbols, so a symbol cannot consume two
slots. Every unresolved entry contributes regardless of age. Values use exact
Decimal arithmetic and retain the journal head for atomic reservation comparison.
These are principal-notional totals; final cash admission must also account for
any execution commissions or fees and enforce its executable price ceiling.

The aggregation function is a non-authorizing calculation over caller-verified
replay. The gateway supplies that provenance; constructing a totals object alone
cannot establish authoritative risk evidence. Final reservation must atomically
compare this context's head and bind all decision inputs, including sector.

## Current paper execution cost

The simulator explicitly produces zero commission. A shared pricing model now
applies signed basis-point slippage with exact integer-ratio arithmetic and one
half-even rounding to a 0.0001 USD tick. The authorized Decimal execution sink
uses that calculation, independent of the caller's Decimal precision or rounding
mode. A reproduced precision-6/ROUND_DOWN case previously filled at 12.3456 instead
of 12.3457; the corrected sink produces 12.3457.

Inside the owning entry context, the gateway reads the registered exact executor's
current finite slippage setting and the independently validated quote. It returns
the modeled BUY fill, explicit zero commission and a conservative price ceiling
of max(reference, modeled fill). The broker quote itself remains unchanged.
No cost estimate grants order authority. Risk sizing/reservation and final
submission must still consume and bind this ceiling to the executable policy;
any policy or reference-price change requires re-evaluation. Paper-performance
analysis must state the simulator's zero-commission assumption.

## Remaining runtime work

Owned account-wide snapshots, durable cooldown evidence, daily-risk replay and
ingestion, and Config-level policy binding now exist. Authenticated terminal release,
complete market/account evidence, production verifier
construction, final contract consumption, and baseline BUY settlement remain
open. These components must be integrated and verified before entry authority is
enabled; component tests do not prove complete operational enforcement.
Incomplete strategies, shorts, smart execution, AI/ML discovery, and take-profit
remain disabled for Gate A. Reductions retain their separate safety policy.
