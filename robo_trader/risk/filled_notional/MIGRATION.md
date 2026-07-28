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

The anchor must live in a separate non-group/world-writable directory. The
service binds that directory by device/inode and performs anchor operations
relative to an `O_DIRECTORY`/no-follow descriptor. Replacing or redirecting the
directory does not transfer anchor authority.

The HMAC key must not be stored with either local artifact, but key separation
alone does **not** prove freshness. An attacker can replay an older valid
database and its matching older valid anchor while leaving the HMAC key
unchanged. Every authoritative service construction, read, and append therefore
requires an independent monotonic verifier. Its accepted state must live
outside the database/anchor failure domain and reject any state older than the
last accepted fill/conflict heads and counts. Without such an independently
operated monotonic state or service, this ledger is non-authoritative.

Each append first fsyncs an authenticated pending transition to the anchor,
then commits SQLite, then atomically replaces and directory-fsyncs the stable
anchor. On restart, identity-bound read-write SQLite recovery runs before any
read-only validation. A pending anchor may resolve only to its authenticated
old or new state; a stable anchor is never advanced heuristically. Therefore an
old stable anchor, ledger tail truncation, database rollback, or replacement
fails closed instead of being mistaken for a process-crash window.

For an existing database, schema version detection is immutable and read-only
before any read-write recovery. A legacy v1 hot journal is preserved byte for
byte; it is never opened in a mode that can recover, remove, or rewrite the
database or sidecar. If the version cannot be established without recovery, the
service raises unavailable and leaves all evidence untouched.
