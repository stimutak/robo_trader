# Filled-notional ledger migration

Schema version 2 adds authenticated fill records, an account-level broker
execution identity, durable conflict quarantine, and a separately protected
chain-head/count anchor.

Version 1 ledgers are never changed in place. Opening one raises
`FilledNotionalMigrationRequired` before an anchor is created. The safe path is:

1. Preserve and checksum the version 1 database and all SQLite sidecars.
2. Reconcile every execution against a reviewed broker execution export.
3. Create a new version 2 ledger and anchor in new paths with an HMAC key from
   an independent secret authority.
4. Replay only confirmed executions using their immutable broker execution IDs.
5. Compare per-account, per-portfolio, per-currency, and New York trading-date
   totals, then retain the original version 1 files as read-only evidence.

Future schema versions also fail closed. Downgrade and in-place rewrite are not
supported. No migration may delete or overwrite the only copy of any ledger,
anchor, execution evidence, or conflict marker.

The anchor must live in a separate non-group/world-writable directory. Protect
and version it independently from the SQLite database. The HMAC key must not be
stored with either artifact. A coordinated rollback of the database, anchor,
and external key/monotonic authority is outside the local module's threat
boundary and requires an independently operated monotonic service.

Each append first fsyncs an authenticated pending transition to the anchor,
then commits SQLite, then atomically replaces and directory-fsyncs the stable
anchor. On restart, identity-bound read-write SQLite recovery runs before any
read-only validation. A pending anchor may resolve only to its authenticated
old or new state; a stable anchor is never advanced heuristically. Therefore an
old stable anchor, ledger tail truncation, database rollback, or replacement
fails closed instead of being mistaken for a process-crash window.
