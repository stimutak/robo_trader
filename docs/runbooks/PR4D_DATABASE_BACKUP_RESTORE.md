# PR 4D SQLite backup and clean-room restore

## Safety boundary

This tooling is dormant maintenance infrastructure. It never connects to IBKR,
starts RoboTrader, authorizes startup, deletes a database, renames a database
over another path, or rolls back by replacing an authoritative ledger. Every
output path must be absolute, canonical, and nonexistent. Existing files,
symlink leaves, hard-linked databases, and path substitution fail closed.
SQLite `-wal`, `-shm`, and `-journal` companions are checked for safe type,
link count, and stable identity while a snapshot is running.

The supported operations are:

- a SQLite online backup, including committed WAL state, into an exclusively
  created owner-only file;
- integrity, foreign-key, schema, row-count, and deterministic table-hash
  verification;
- restoration of a manifest-verified backup into a different, exclusively
  created clean-room path;
- library-level migration dry runs on synthetic copies. No production
  migration is registered with the command-line tool; a reviewed callback is
  required so arbitrary code cannot be selected from command-line input.

Reports and manifests contain no paths, row values, environment values,
credentials, account identifiers, or broker configuration. They always state
`contains_secrets=false`, `mutated_authoritative_state=false`, and
`authorizes_startup=false`.

If backup, restore, or verification is interrupted, the source remains
untouched. Any already-reserved output is retained read-only for forensic
inspection; it has no successful manifest and cannot be reused as a target.

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
snapshot at a new synthetic path. The reviewed callback runs within a service-
owned transaction against that copy only. Transaction control, `ATTACH`, and
`DETACH` are denied. Success and rollback reports include before/after schema,
row counts, content hashes, integrity state, and a source-unchanged result.

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
