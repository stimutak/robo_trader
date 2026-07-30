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
python3 -m pytest tests/maintenance/test_sqlite_service.py \
  tests/test_multiuser.py tests/security/test_db_isolation.py -q --tb=short
232 passed
```

The PR 4D maintenance module has 65 direct tests covering active WAL state,
multiportfolio preservation, backup/restore manifests, clean-room equivalence,
corruption, preexisting targets, symlinks, hardlinks, companion files,
substitution races, backup/restore interruption, migration rollback,
declarative-plan containment, hidden-rowid source-change detection,
planner-statistics evidence, bounded migration execution, `WITHOUT ROWID` and
shadowed-rowid compatibility, native-pointer and virtual-table denial,
schema-cookie protection and tracking, native-function denial, bounded
synthetic database growth, pre-publication final-evidence failure, CLI output,
embedded-schema-function rejection, per-opcode deadline enforcement, TEMP
storage containment, narrow migration-grammar rejection, and legacy quarantine.

Final full repository regression:

```text
python3 -m pytest tests/ -q --tb=short
3110 passed, 5 skipped, 20 known warnings in 82.67s
```

After PR 4C merged to `main`, that exact head was merge-integrated into this
branch. Its two settlement compatibility tests now install a synthetic copy of
the historical multiuser-v1 result directly under `tests/`; they do not invoke
or re-enable the quarantined in-place migrator. The complete settlement file
passes 30 tests on the combined tree.

Static checks:

```text
python3 -m black --check <changed Python paths>
python3 -m isort --check-only <changed Python paths>
python3 -m flake8 <changed Python paths>
python3 -m bandit -q -r robo_trader/maintenance \
  scripts/database_maintenance.py robo_trader/multiuser/migration.py
python3 -m mypy --follow-imports=skip --ignore-missing-imports \
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
- Failed or interrupted anonymous outputs disappear when their held descriptor
  closes and have no successful manifest or requested target path. They are
  never promoted or reused.
- Declarative migration plans run only against the new synthetic copy in a
  service-owned transaction. Transaction control, `ATTACH`, `DETACH`, and
  dangerous path/schema pragmas are denied; failures roll back and return a
  secret-free report.

## 2026-07-30 exact-review follow-up

The exact-head P1 findings and Python 3.10 CI failures were addressed without
touching authoritative data or runtime services. The latest same-UID attacker
reports were handled by correcting the documented trust boundary rather than
claiming isolation that an in-process filesystem library cannot provide:

- SQLite now operates only in memory. It never opens the requested output path,
  a staging-directory path, or any public sidecar pathname. The exact finished
  image is serialized into an unlinked inode on the target filesystem. Linux
  publishes its `O_TMPFILE` inode with `linkat(AT_EMPTY_PATH)`; macOS publishes
  from an unlinked descriptor with `fclonefileat`. Unsupported topology fails
  closed, and the service performs no pathname unlink.
- A planted staging-name symlink cannot redirect SQLite into an attacker-owned
  directory because no filesystem target connection exists. Exclusive file
  creation refuses the symlink and preserves planted `database.db-wal` bytes.
- The staged inode has no persistent directory entry (`st_nlink == 0`) before
  publication, which removes the prior pathname substitution and hard-link
  races. This is not claimed to isolate the service from actively hostile code
  running under the same OS user: Linux `/proc` descriptor access, macOS's
  temporary-file creation window, and post-publication directory access make
  that guarantee impossible for an in-process library. The reviewed operating
  boundary now requires an exclusive maintenance window; stronger adversarial
  isolation requires a dedicated OS identity, container, or sandbox.
- A cleanly closed database that retains `journal_mode=WAL` is opened with
  normal SQLite locking. Safe companions created while establishing the bound
  connection are adopted and monitored, preventing both self-rejection and the
  torn snapshots caused by incorrectly inferring immutability from absent
  sidecars. Only manifest-verified sealed restore input uses immutable mode.
- The exact descriptor bytes are read back and deserialized before publication.
  Their integrity, foreign-key, schema, row-count, and table-hash evidence must
  match the in-memory copy, and the manifest digest must match the same byte
  snapshot. A valid alternate database injected through a retained writable
  descriptor therefore fails before publication.
- Publication fsyncs the bound parent-directory descriptor and revalidates both
  the lexical parent and published target afterward. Parent replacement causes
  a failed operation rather than a false successful manifest.
- Migration dry runs capture before/after logical evidence through bound,
  WAL-aware read snapshots, so a normally active RoboTrader WAL source is
  accepted without relaxing source-change detection.
- A normal WAL checkpoint may change physical main-file bytes while SQLite's
  held read snapshot remains consistent. Online backup now relies on that bound
  logical snapshot instead of incorrectly rejecting checkpointed bytes.
- A public `-wal`, `-shm`, or `-journal` substitution while SQLite is active is
  preserved byte-for-byte and causes publication to fail closed. This closes
  the Linux behavior where SQLite itself could unlink a replacement before a
  later reservation check.
- A migration plan runs on the same descriptor-bound target connection used by
  the online copy. The synthetic database is never reopened writable between
  copy verification and migration.
- Migration report evidence and artifact hash come directly from the final
  descriptor-bound copy manifest. The published target is not reopened, so a
  later path substitution cannot change what artifact the report describes.
- Service-owned commit and rollback replace the restrictive plan authorizer
  with a completion-only policy. This avoids Python 3.10's unsupported
  `set_authorizer(None)` behavior without allowing plan-supplied transaction
  control.
- Migration plans cannot change SQLite's process-global hard or soft heap
  limits. Oversized integer binding failures are converted into the same
  rollback-only, secret-free failure report as rejected SQL.
- Database evidence includes SQLite's schema cookie, so transient create/drop
  sequences cannot hide a schema-generation change behind identical final DDL.
  The bound source cookie is preserved across SQLite's in-memory backup reset.
- Plan-invoked SQL functions are denied, including native functions that can
  allocate or execute without reaching the VM progress callback. SQLite's
  internal ALTER TABLE helper calls are reserved from plan SQL. The deadline is
  checked after every VM opcode and after each statement. Separate main and
  temporary page ceilings limit synthetic growth to 64 MiB each by default,
  and plan-controlled TEMP DDL is denied. Deadline rollback has a distinct
  `migration_deadline_exceeded` result, so its regression proves the interrupt
  without relying on CI filesystem speed; a ten-second outer ceiling still
  catches a lost interrupt or hang.
- Migration plans are not arbitrary SQL. Full-match validation permits only a
  small documented grammar of basic DDL plus parameter-only INSERT and
  predicate-required UPDATE/DELETE. Functions, expressions, comments,
  subqueries, PRAGMAs, TEMP objects, transaction control, and quoted
  identifiers fail before source/target access. The authorizer and resource
  controls are defense in depth rather than a claimed complete SQL sandbox.
- Functions embedded in persistent source schema can execute without an
  authorizer callback. The service now detects registered function names in
  copied schema SQL before plan execution and returns a rollback-only failure
  report rather than evaluating the schema expression.
- Final live-source evidence is captured after the descriptor-bound synthetic
  artifact is verified but before publication. A failed final evidence read
  closes the still-anonymous artifact and leaves no target path.
- Every CLI report path is checked against the full main/`-wal`/`-shm`/
  `-journal` family of each source, backup, and target database. Backup and
  restore manifest paths are exclusively reserved, fully written, fsynced,
  sealed read-only, and descriptor-checked before database publication. An
  existing, unwritable, or failed report path therefore cannot leave a new
  orphan database artifact.
- All documented Python invocations use the project-required `python3`
  executable name.

Current verification:

```text
focused maintenance + multiuser + DB isolation: 232 passed
full repository: 3110 passed, 5 skipped, 20 known warnings
PR 4C settlement integration file: 30 passed
Python 3.10 direct authorizer commit/rollback probe: passed
Black, isort, flake8, Bandit, mypy, and git diff checks on changed scope: passed
```

Repository-wide Black and flake8 remain red only for pre-existing files outside
this PR's changed scope; those unrelated files were not modified.
