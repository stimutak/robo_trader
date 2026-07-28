# Filled-notional ledger migration

Schema version 3 adds an append-only HMAC checkpoint chain and authenticated
per-scope cumulative fill counts and totals. The service performs a full,
streaming validation of the fill, conflict, and checkpoint chains at startup.
After that audit, appends and authoritative reads validate the latest
checkpoint, the relevant authenticated scope-total record, the independent
anchor, and the external monotonic state without rescanning all history.

Version 1 and version 2 ledgers are never changed in place. Opening either one
raises `FilledNotionalMigrationRequired` before an anchor is created. The safe
path is:

1. Preserve and checksum the legacy database and all SQLite sidecars.
2. Reconcile every execution against a reviewed broker execution export.
3. Create a new version 3 ledger and anchor in new paths with an HMAC key from
   an independent secret authority.
4. Replay only confirmed executions using their immutable broker execution IDs.
5. Compare per-account, per-portfolio, per-currency, and New York trading-date
   totals, then retain the legacy files as read-only evidence.

Future schema versions also fail closed. Downgrade and in-place rewrite are not
supported. No migration may delete or overwrite the only copy of any ledger,
anchor, execution evidence, conflict marker, or SQLite sidecar.

The anchor must live in a separate non-group/world-writable directory. The
service binds that directory by device/inode and performs anchor operations
relative to an `O_DIRECTORY`/no-follow descriptor. Replacing or redirecting the
directory does not transfer anchor authority.

The HMAC key must not be stored with either local artifact, but key separation
alone does **not** prove freshness. An attacker can replay an older valid
database and its matching older valid anchor while leaving the HMAC key
unchanged. Every authoritative service construction, read, append, and review
therefore requires an independent monotonic verifier. Its accepted state must
live outside the database/anchor failure domain and reject any state older than
the last accepted fill/conflict heads and counts. Without such an independently
operated monotonic state or service, this ledger is non-authoritative.

Each append first fsyncs an authenticated pending transition to the anchor,
then commits SQLite, then atomically replaces and directory-fsyncs the stable
anchor. A pending anchor may resolve only to its authenticated old or new
state; a stable anchor is never advanced heuristically. Authoritative reads and
reviews use one SQLite snapshot and re-check both the exact anchor and external
monotonic state immediately before returning.

The hot path is deliberately bounded: the latest cumulative scope total is
found through the scope/date/sequence index, each append adds one checkpoint,
and no scope may exceed 100,000 fills for one New York trading date. Exceeding
that limit rejects the append before any durable mutation. Full history and
checkpoint-chain validation is startup-only and streams rows instead of
materializing them. The hot path detects forged or truncated tails and invalid
authenticated totals; the startup audit additionally detects interior record
tampering. This assumes the process and ledger paths retain their configured OS
access controls between startup and use. If that local-file integrity boundary
cannot be trusted, reconstruct on a reviewed copy or restart to force the full
audit before relying on the ledger.

For an existing database, schema version detection is immutable and read-only
before any read-write recovery. Read-write recovery of a hot rollback journal
is authorized only by a valid current-version anchor bound to that exact
database device/inode. A legacy or otherwise ambiguous hot journal without
that authority is preserved byte for byte and the service raises unavailable.
This is required because SQLite cache spill can place an uncommitted future
schema marker in the main database while the rollback journal still contains
the authoritative legacy image; trusting that marker and recovering would
destroy migration evidence.
