# PR 4D local review evidence

Date: 2026-07-28

Base: `101beb94921c884c050580973a084e8a8380be8d`

Branch: `codex/pr4d-backup-restore`

## Scope and safety result

- Added dormant, descriptor-bound SQLite backup, verification, clean-room
  restore, and synthetic migration dry-run tooling.
- Every output uses an exclusive new path. No operation deletes, renames over,
  restores over, or automatically replaces an authoritative database.
- Manifests contain no paths, row values, environment values, credentials, or
  broker/account configuration and always report
  `authorizes_startup=false`.
- Removed the callable raw-copy, overwrite-restore, foreign-key-disabled, and
  account-row-collapsing legacy multiuser implementation. Read-only inspection,
  missing-database no-op, and already-applied no-op behavior remain.
- Repository call-graph search found no non-test caller of
  `MultiuserMigration`; no sanctioned startup path was changed.
- All backup, corruption, restore, migration, WAL, alias, and interruption
  exercises used pytest temporary databases. No authoritative/user database,
  broker, runtime process, listener, or service was accessed or modified.
- This work does not authorize startup and does not advance Gate A.

## Verification

Focused maintenance, legacy migration, and database-isolation matrix:

```text
python -m pytest tests/maintenance/test_sqlite_service.py \
  tests/test_multiuser.py tests/security/test_db_isolation.py -q --tb=short
188 passed
```

The PR 4D maintenance module has 30 direct tests covering active WAL state,
multiportfolio preservation, backup/restore manifests, clean-room equivalence,
corruption, preexisting targets, symlinks, hardlinks, companion files,
substitution races, backup/restore interruption, migration rollback, callback
containment, CLI output, and legacy quarantine.

Final full repository regression:

```text
python -m pytest tests/ -q --tb=short
2854 passed, 5 skipped, 20 known warnings in 189.19s
```

Static checks:

```text
python -m black --check <changed Python paths>
python -m isort --check-only <changed Python paths>
python -m flake8 <changed Python paths>
python -m bandit -q -r robo_trader/maintenance \
  scripts/database_maintenance.py robo_trader/multiuser/migration.py
python -m mypy --follow-imports=skip --ignore-missing-imports \
  robo_trader/maintenance/models.py \
  robo_trader/maintenance/sqlite_service.py \
  scripts/database_maintenance.py robo_trader/multiuser/migration.py
git diff --check
```

All listed static checks passed. The focused mypy invocation intentionally
isolates changed modules because the repository retains unrelated, documented
type debt.

## Review notes

- The service reuses the reviewed SQLite-owned-descriptor proof from
  `robo_trader.safety.sqlite_identity` and independently holds an
  `O_NOFOLLOW` guardian.
- Main database and `-wal`/`-shm`/`-journal` identities are checked throughout
  online copy progress. Backup and restore artifacts are integrity-checked,
  foreign-key-checked, logically hashed, raw-file hashed, fsynced, and sealed
  read-only.
- Failed or interrupted outputs are retained read-only for forensic inspection
  and have no successful manifest. They are never promoted or reused.
- Declarative migration plans run only against the new synthetic copy in a
  service-owned transaction. Transaction control, `ATTACH`, `DETACH`, and
  dangerous path/schema pragmas are denied; failures roll back and return a
  secret-free report.

## 2026-07-30 exact-review follow-up

Three additional exact-head P1 findings and the Python 3.10 CI failures were
remediated without touching authoritative data or runtime services:

- Sidecar release now distinguishes a reservation SQLite consumed itself by
  checking the still-open descriptor's zero link count. Remaining reservations
  are atomically moved with the platform's no-replace rename into an exclusive
  sibling quarantine directory. The service performs no pathname unlink and
  leaves zero-byte tombstones in place.
- A substitution between pathname inspection and the atomic claim is preserved,
  restored to its original name when that can be done without replacement, and
  causes the operation to fail closed.
- A migration plan runs on the same descriptor-bound target connection used by
  the online copy. The synthetic database is never reopened writable between
  copy verification and migration.
- Service-owned commit and rollback replace the restrictive plan authorizer
  with a completion-only policy. This avoids Python 3.10's unsupported
  `set_authorizer(None)` behavior without allowing plan-supplied transaction
  control.

Current verification:

```text
focused maintenance + multiuser + DB isolation: 197 passed
full repository: 3036 passed, 5 skipped, 20 known warnings
Python 3.10 direct authorizer commit/rollback probe: passed
Black, isort, flake8, Bandit, mypy, and git diff checks on changed scope: passed
```

Repository-wide Black and flake8 remain red only for pre-existing files outside
this PR's changed scope; those unrelated files were not modified.
