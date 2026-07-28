# PR4B local review evidence — 2026-07-28

## Reviewed scope

Branch `codex/pr4b-fifo-bootstrap` was created from exact PR4A head
`101beb94921c884c050580973a084e8a8380be8d` in the dedicated worktree
`/private/tmp/robo-pr4b-fifo-bootstrap`.

The review covered only the PR4B bridge from the PR112 exact-state bootstrap to
the PR4A FIFO ledger:

- authenticated positive `con_id` binding for every nonzero candidate position;
- deterministic `LEGACY_AGGREGATE_OPENING_BALANCE` epoch identity;
- append-only candidate, reconciliation, broker, legacy-ledger, administrator,
  and exact-state bootstrap lineage;
- exact pre-epoch account baseline;
- explicit long/short opening balances and opening lots with no synthetic fill,
  execution, commission event, or position snapshot;
- read-only non-authorizing candidate preview; and
- one atomic offline schema/bootstrap transaction under the existing lifecycle,
  confirmation, evidence, safety-journal, descriptor, and verified-backup gates.

PR4C runtime settlement wiring and PR4D operational migration/restore drills
were not added. No startup, broker, runtime, service, authoritative database, or
user data was accessed or mutated.

## Adversarial evidence

Synthetic tests prove:

- candidate omission, duplication, or mismatch of authenticated contract IDs
  fails closed;
- long and short legacy quantities become aggregate opening balances rather
  than BUY/SELL fills;
- FIFO fill and commission tables remain empty at bootstrap;
- zero opening commission is explicitly reported as unknown pre-epoch history,
  not reconstructed fee evidence;
- the candidate fingerprint binds the FIFO epoch and exact-state lineage;
- a conflicting epoch for the same account/portfolio scope blocks the complete
  exact-state bootstrap;
- a fault after FIFO append rolls back schema and every bootstrap row to the
  byte-identical raw fixture database;
- preview leaves the fixture byte-identical and creates no FIFO objects;
- the pre-epoch cash, realized P&L, daily P&L, baseline, and date are preserved;
- a prospective post-epoch fill consumes the aggregate opening lot by strict
  FIFO with exact realized P&L; and
- existing stopped-runtime, evidence-replay, journal, backup, path, inode,
  hard-link, and post-commit backup checks remain green.

## Verification results

All commands used the repository virtual environment at
`/Users/oliver/Projects/robo_trader/.venv/bin/python3` and synthetic per-test
databases.

- Focused FIFO/exact-bootstrap/CLI suite: `124 passed`.
- Broader accounting/bootstrap/database/reconciliation/settlement matrix:
  `517 passed`.
- Full repository suite: `2849 passed, 5 skipped, 20 warnings` in 167.09s.
  Skips and warnings are the existing environment/test warnings reported by the
  suite; no test failed.
- Black, isort, Flake8, compilation, and `git diff --check`: passed for every
  changed Python file.
- Targeted mypy for FIFO/bootstrap sources: passed with no issues.
- Targeted Bandit: no medium or high findings. Three existing low findings are
  the bootstrap CLI's fixed-argv `subprocess` import/calls used for stopped-state
  checks.
- Scope/secret checks: no runner, launcher, or terminal-settlement file changed;
  no startup-authorizing value or private-key/access-key pattern was added.

## Review conclusion

No local correctness, data-integrity, trading-safety, security, or scope blocker
remains. This is code/test evidence only. It is not operational restore
evidence, broker evidence, startup approval, or Gate A approval.
