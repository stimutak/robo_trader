# PR 4D SQLite backup and clean-room restore

## Safety boundary

This tooling is dormant maintenance infrastructure. It never connects to IBKR,
starts RoboTrader, authorizes startup, deletes a database, renames a database
over another path, or rolls back by replacing an authoritative ledger. Every
output path must be absolute, canonical, and nonexistent. Existing files,
symlink leaves, hard-linked databases, and path substitution fail closed.
SQLite `-wal`, `-shm`, and `-journal` companions are checked for safe type,
link count, and stable identity while a snapshot is running. SQLite never opens
the requested output path or any other filesystem target. Online backup and
migration run in an in-memory SQLite database. The service serializes the exact
finished image into an unlinked regular-file inode (`st_nlink == 0`) on the
target filesystem. Linux uses `O_TMPFILE` and descriptor-only
`linkat(AT_EMPTY_PATH)` publication; macOS uses an unlinked temporary descriptor
and `fclonefileat`. Unsupported filesystems and platforms fail closed. The
service fsyncs the already-bound parent descriptor after publication. The
published descriptor is digested again and its lexical parent and target
identities are revalidated before success. A public `-wal`, `-shm`, or
`-journal` path is therefore never touched or deleted by the service or its
SQLite connection. There is no staging directory that a concurrent process
can replace with a symlink.

This is a filesystem pathname-integrity boundary, not isolation from actively
hostile code running as the same operating-system user. On Linux, such a process
may be able to inspect another process's file descriptors through `/proc`; on
macOS, creation of the temporary file used to obtain an unlinked descriptor can
be observed before unlinking. A hostile same-user process can also modify a
published artifact or its destination directory. Backup and restore therefore
require an exclusive maintenance window with no untrusted process running as
the maintenance user. Environments that need protection from hostile local
code must run this tooling under a dedicated OS identity, container, or sandbox
whose output directory is inaccessible to the trader and other user processes.
The in-process library cannot honestly provide that stronger isolation.

The supported operations are:

- a SQLite online backup, including committed WAL state, into an exclusively
  created owner-only file;
- integrity, foreign-key, schema, row-count, and deterministic table-hash
  verification;
- restoration of a manifest-verified backup into a different, exclusively
  created clean-room path;
- library-level migration dry runs on synthetic copies through a bounded,
  declarative `MigrationPlan`. No production migration is registered with the
  command-line tool, and callers never receive the SQLite connection.

Reports and manifests contain no paths, row values, environment values,
credentials, account identifiers, or broker configuration. They always state
`contains_secrets=false`, `mutated_authoritative_state=false`, and
`authorizes_startup=false`.

If backup, restore, or verification is interrupted, the authoritative main-file
bytes and committed logical state are not modified. A normal read connection to
a WAL-mode source may create SQLite-managed `-wal`/`-shm` companions; the
service validates and preserves them and never deletes them. The unlinked
partial output inode disappears when its held descriptor is closed; the
requested output path remains absent, there is no successful manifest, and no
partial file can be mistaken for a usable artifact.

## Commands

Use only explicit synthetic or operator-approved paths. These examples are
illustrative and must not be pointed at an authoritative database without the
separate review and authorization required by `AGENTS.md`.

```bash
python3 scripts/database_maintenance.py backup \
  --source /absolute/path/to/source.db \
  --target /absolute/path/to/new-backup.db \
  --manifest /absolute/path/to/new-backup-manifest.json

python3 scripts/database_maintenance.py verify \
  --database /absolute/path/to/new-backup.db \
  --manifest /absolute/path/to/new-backup-manifest.json

python3 scripts/database_maintenance.py restore-clean-room \
  --backup /absolute/path/to/new-backup.db \
  --backup-manifest /absolute/path/to/new-backup-manifest.json \
  --target /absolute/path/to/new-clean-room.db \
  --restore-manifest /absolute/path/to/new-restore-manifest.json
```

A restored clean-room database is evidence only. It is never promoted or
swapped into runtime by this service, and its manifest does not authorize
startup.

## Migration dry runs

`SQLiteMaintenanceService.dry_run_migration(...)` first creates a SQLite online
snapshot at a new synthetic path. The declarative plan runs within a
service-owned transaction on the same in-memory connection used to create and
verify the copy; there is no writable reopen gap or filesystem SQLite target.
Plan-supplied
transaction control, `ATTACH`, and `DETACH` are denied. Success and rollback
reports include before/after schema, row counts, content hashes, integrity
state, and a source-unchanged result. Final report evidence and the artifact
digest come directly from the descriptor-bound manifest captured before atomic
publication; the report never reopens the published target.

Active WAL sources are supported. Initial and final source evidence is captured
from bound SQLite read snapshots rather than by requiring a companion-free main
file. Normal checkpoints may change the physical main-file bytes while the held
read snapshot remains logically consistent; backup therefore does not treat a
checkpoint as authoritative data mutation.

A companion-free source is not assumed immutable. Ordinary backup and migration
sources use SQLite's normal locking even when a cleanly closed database retains
`journal_mode=WAL`; safe `-wal`/`-shm` files created while establishing the bound
connection are adopted into the identity set and monitored for the operation.
Only a companion-free, manifest-verified restore artifact uses SQLite immutable
mode. This preserves atomic snapshot semantics if a legitimate writer begins
after the initial filesystem inspection.

Before publication, the service reads the exact descriptor-bound bytes back,
deserializes that byte snapshot into SQLite memory, reruns integrity,
foreign-key, schema, row-count, and deterministic table-hash evidence, and
requires it to equal the evidence captured from the copied database. The
manifest digest is then required to match that same byte snapshot. This catches
retained-descriptor corruption within the supported trusted-identity boundary;
OS-identity isolation remains the deployment requirement described above.

## Legacy multiuser migration quarantine

`robo_trader.multiuser.migration.MultiuserMigration` is not called by any
sanctioned startup path in the reviewed PR 4D baseline. Its read-only version
inspection remains available and a database already at migration version 1 is
still a no-op. Missing databases remain a no-op for `migrate()`.

Mutation is disabled for an unmigrated database because the legacy path:

- copied only the main file instead of using SQLite's WAL-aware backup API;
- disabled foreign-key enforcement during table replacement;
- selected one account row while rebuilding the account table, which could
  collapse multi-row state;
- attempted rollback by copying a backup over the authoritative path.

The legacy backup, restore, and apply methods now raise
`LegacyMultiuserMigrationDisabled`. The unsafe table-rewrite SQL has been
removed rather than left as a callable dormant path. A future production
migration needs its own reviewed adapter, descriptor-bound backup manifest,
synthetic dry-run report, explicit operator approval, and post-operation
reconciliation.
