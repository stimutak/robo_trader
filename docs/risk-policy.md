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

## Remaining runtime work

Authoritative account-wide snapshots and reservations, configuration binding,
durable cooldown production, daily-risk replay/ingestion, and the baseline BUY settlement
path must be integrated and verified before entry authority is enabled. Current
contract tests prove pure decisions, not complete operational enforcement.
Incomplete strategies, shorts, smart execution, AI/ML discovery, and take-profit
remain disabled for Gate A. Reductions retain their separate safety policy.
