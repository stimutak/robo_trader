# Gateway selection and evidence-backed equity freshness

This change prepares the supervised startup path; it does not install software,
start Gateway, connect an account, authorize trading, or complete Gate A.
PR130's rebuild and PR131's workflow changes remain separate.

## Gateway and IBC selection

`START_TRADER.sh` resolves one Gateway/IBC pair before modifying runtime
processes. Set `GATEWAY_VERSION` to the approved major version (for example
`10.51`) and optionally `ROBOTRADER_IBC_PATH` to an absolute official IBC
installation directory. These non-secret settings can be in `.env`; process
settings take precedence. Without a version pin the newest installed major
version is selected numerically. Invalid, missing or incomplete selections
block; there is no fallback to 10.37 or to an older compatible installation.
Startup exports its selection so internal recovery cannot switch versions
when another installation appears.

The selector requires application jars, `ibgateway.vmoptions`, and coherent
install4j build metadata. Gateway 10.48 onward requires IBC 3.24.2 or newer;
older supported installations require at least IBC 3.23.0. The selected IBC
must have its version file, jar and executable service scripts. This is a
known compatibility floor, not a certification of every future release.

Official IBC `gatewaystartmacos.sh` assigns its own settings, overwriting
inherited environment values. Both launch paths therefore invoke IBC's
`displaybannerandlaunch.sh` service script with the selected version, paths
and `APP=GATEWAY`. Vendor scripts and credential files are not rewritten.
Startup uses bundled Java; paper mode and the existing read-only validation
remain mandatory. An already-running Gateway must have a process classpath
matching both selected installations; otherwise startup/recovery blocks
before stopping processes. This proves installation identity, not login,
API handshake, or that an in-place software replacement updated a loaded JVM.
Process inspection requests untruncated command output on macOS so long JVM
classpaths retain the installation and IBC identity evidence.

Before connected-machine verification, install the approved Gateway and IBC
pair side by side, retain the previous pair and secure settings backups, and
coordinate a stopped maintenance window. Record exact build, CPU architecture,
bundled Java, artifact provenance, login/2FA, read-only handshake and read-only
requests. Do not infer supported build or retirement date from an API package
version. Rollback selects the previous pair only if IBKR still accepts it;
otherwise remain stopped. Never roll financial data back over newer activity.

Official references:

- https://github.com/IbcAlpha/IBC/releases/tag/3.24.2
- https://github.com/IbcAlpha/IBC/blob/3.24.2/resources/gatewaystartmacos.sh
- https://github.com/IbcAlpha/IBC/blob/3.24.2/resources/scripts/ibcstart.sh
- https://www.interactivebrokers.com/en/trading/ibgateway-latest.php

## Active portfolio freshness and recovery

Preflight reads the exact configured ledger in an identity-checked, query-only
SQLite transaction. It uses the same portfolio configuration parser as runtime
against the resolved environment snapshot. Every active portfolio must be
fresh; disabled portfolios cannot mask stale active ones. Malformed/no-active
configuration, missing history for an existing portfolio, malformed timestamps,
future timestamps and unreadable/replaced ledgers block. SQLite timestamps are
UTC and are converted to market time before counting trading days. The
one-trading-day freshness threshold is unchanged. A genuinely empty first-run
ledger retains its warning; unrelated readiness checks still apply.

For an **unbootstrapped legacy local-paper ledger**, the existing stopped-system
`bootstrap_exact_paper_state.py preview` and `apply` commands now accept
`--append-equity-checkpoint`. The preview requires the existing candidate,
authenticated reconciliation, zero-exposure broker evidence and all current
protective-mark artifacts. It displays the exact valuation and does not mutate
anything. Apply requires the existing lifecycle exclusion, journal validation,
fresh evidence revalidation, unchanged ledger identity/state, verified online
backup and destination-bound confirmation. The confirmation additionally
includes `append-equity-checkpoint=yes`; a plain bootstrap confirmation cannot
authorize this additional write.

The opt-in writes an immutable `bootstrap_equity_checkpoints` record in the
same transaction as the sealed exact/FIFO bootstrap. It records exact cash,
marked signed position value, equity and P&L with the authenticated observation
time and candidate fingerprint. It never copies stale equity forward, uses
broker cash as simulator capital, fabricates fills, backdates missing daily
records, or updates existing account/position/trade/equity-history rows.
The ordinary daily snapshot writer cannot overwrite this separate checkpoint.
Without the flag, existing bootstrap behavior is unchanged.

When daily history is stale, preflight can use this genuine fresh valuation
only after checking exact table/trigger structure, candidate fingerprint and
payload, runtime/account/portfolio and physical ledger identity, plus the
unchanged legacy ledger fingerprint in the same read snapshot. The checkpoint
then passes through the same timestamp/future/trading-day checks as history.
Fresh ordinary daily history remains authoritative after normal accounting
updates; a changed ledger with stale history cannot reuse the old checkpoint.
Diagnostic JSON identifies which valuation source was assessed.

This is deliberately **not** a recurring rebootstrap command. An already
bootstrapped stale ledger, changed allocation, incomplete evidence or expired
checkpoint remains blocked. It needs current reconciliation and a separately
reviewed accounting recovery; no historical timestamp or balance may be
rewritten to make the check pass.
Requesting a checkpoint on an existing bootstrap explicitly fails, including
when that bootstrap already has a checkpoint; replay cannot attach or refresh one.

Read-only broker evidence collection still requires a separately authorized
connected-machine task. A checkpoint fixes only the freshness condition; it
never releases kill switches, repairs the safety journal, opens BUY gates,
closes Gate A or authorizes startup. Connectivity verification must be scoped
to read-only requests without trading cycles; local paper simulation is a
separate authorization, and IBKR paper orders are another distinct action.

## Offline regression evidence

Tests use synthetic application directories, process results, SQLite ledgers,
and test-only signing keys. Coverage includes upgraded/pinned selection,
missing or mismatched installations, incompatible IBC before process changes,
old or unknown running processes, launch environment without vendor defaults,
active/disabled portfolio isolation, malformed/future timestamps,
authenticated checkpoint preview/application, untouched historical rows,
provenance tampering, copied database rejection, atomic rollback and distinct
operator confirmation. No test invokes Java/Gateway or a real broker account.

Local focused verification: 203 passed, 3 macOS lifecycle checks skipped.
Independent review found no remaining actionable defects; its selected
synthetic regression run passed 53 tests. Application/test Black, application
isort, application/test Flake8 and shell syntax checks pass. Two existing
stdlib import orders were adjusted for current isort's application gate.
Mac installer, runtime and connected-account verification remain outstanding.
