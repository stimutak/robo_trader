# Cloud source snapshot readiness — 2026-09-29

**Published:** initial full checkpoint `e0f3999` is available on
`codex/paper-rebuild-integration-2026-09-29` in
[draft PR #130](https://github.com/stimutak/robo_trader/pull/130). GitHub visibility
is public and native authentication/push were verified. The native owner scanned
the 96-commit history with Gitleaks 8.30.1 (zero findings) and the selected current
files (two deterministic test-key false positives, no real credential findings).
The exact 91 staged blobs matched the scanned candidate. The original inventory
and earlier restricted-environment limitations below are retained as history.
See [current state](../CURRENT_STATE.md) for remaining CI/review issues; source
publication does not enable Gate A or deploy the operational checkout.

**Historical pre-publication inventory.** The native integration owner is publishing the complete reviewed source checkpoint on `codex/paper-rebuild-integration-2026-09-29`; see `../CURRENT_STATE.md` and that branch's PR for current publication/check status. The facts below record the earlier local-only state. The remote is `github.com/stimutak/robo_trader.git` (credential-free identity only). Local rebuild branch `codex/paper-readiness-2026-09-13` is at `cae45e0`, 96 commits ahead of fetched `origin/main` (`57f7634`) and zero behind. The working tree has 34 modified tracked files and 50 untracked files, including the two September 29 handoffs. A cloud checkout of `origin/main` is therefore not the candidate being tested locally. No Git state or operational data was changed to publish it.

## Exact current inclusion candidates

The paths below are the complete `git status --porcelain=v1 -uall` source/documentation/test set at manifest creation. They are **candidates for a reviewed source-only snapshot**, not a preapproved `git add -A`. All existing uncommitted work belongs to the local owners until reviewed. A new snapshot branch must preserve the entire relevant implementation dependency graph and avoid dropping untracked modules.

### Modified tracked paths (34)

<!-- tracked-list-start -->
- `config/ibc/config.ini.template`
- `docs/PAPER_READINESS_EXECUTION_2026-09-13.md`
- `docs/ROBOTRADER_REMEDIATION_PLAN_2026-07-20.md`
- `robo_trader/clients/ibkr_subprocess_worker.py`
- `robo_trader/clients/subprocess_ibkr_client.py`
- `robo_trader/database_async.py`
- `robo_trader/database_migrations.py`
- `robo_trader/market_data_contract.py`
- `robo_trader/market_hours.py`
- `robo_trader/paper_reduction_gateway.py`
- `robo_trader/paper_reduction_submitter.py`
- `robo_trader/paper_terminal_settlement.py`
- `robo_trader/reconciliation/bootstrap_producer.py`
- `robo_trader/risk/entry_reservations.py`
- `robo_trader/risk/paper_fill_accounting.py`
- `robo_trader/risk/paper_ledger_snapshot.py`
- `robo_trader/runner/data_fetcher.py`
- `robo_trader/runner_async.py`
- `robo_trader/safety/entry_capacity.py`
- `robo_trader/safety/journal.py`
- `robo_trader/safety/models.py`
- `sync_db_reader.py`
- `tests/risk/test_paper_ledger_snapshot.py`
- `tests/security/test_trading_core.py`
- `tests/test_exact_state_bootstrap.py`
- `tests/test_ibkr_transport_generation_protocol.py`
- `tests/test_package_boundary.py`
- `tests/test_pr1a_existing_position_protection.py`
- `tests/test_pr1a_runner_event_time.py`
- `tests/test_pr2b1_contract_lineage.py`
- `tests/test_pr2b2_runner_routing.py`
- `tests/test_pr2b3_terminal_settlement_persistence.py`
- `tests/test_pr3_market_data_contract.py`
- `tests/test_reconciliation_ibkr_adapter.py`
<!-- tracked-list-end -->

### Untracked paths (50)

<!-- untracked-list-start -->
- `docs/FILLED_NOTIONAL_REMOTE_AUTHORITY.md`
- `docs/FRACTIONAL_VOLUME_IMPLEMENTATION.md`
- `docs/market-calendar.md`
- `docs/superpowers/plans/2026-09-19-paper-entry-persistence.md`
- `handoff/CLOUD_SOURCE_SNAPSHOT_READINESS_2026-09-29.md`
- `handoff/HANDOFF_2026-09-29_paper_rebuild.md`
- `robo_trader/clients/exact_historical_decoder.py`
- `robo_trader/paper_entry_persistence.py`
- `robo_trader/paper_entry_receipt.py`
- `robo_trader/paper_entry_record_replay.py`
- `robo_trader/paper_entry_release.py`
- `robo_trader/paper_entry_settlement.py`
- `robo_trader/paper_entry_storage_replay.py`
- `robo_trader/paper_settlement_record_dispatch.py`
- `robo_trader/reconciliation/entry_settlement_crosslink.py`
- `robo_trader/risk/canonical_correlation.py`
- `robo_trader/risk/completed_sessions.py`
- `robo_trader/risk/filled_notional/remote_verifier.py`
- `robo_trader/risk/paper_entry_accounting_confirmation.py`
- `robo_trader/safety/entry_release.py`
- `tests/canonical_batch_test_support.py`
- `tests/risk/test_canonical_batch_ownership.py`
- `tests/risk/test_canonical_correlation.py`
- `tests/risk/test_completed_sessions.py`
- `tests/risk/test_entry_accounting_confirmation.py`
- `tests/risk/test_entry_capacity_release.py`
- `tests/risk/test_entry_fill_accounting.py`
- `tests/risk/test_entry_submission_claim.py`
- `tests/risk/test_gateway_correlation.py`
- `tests/risk/test_gateway_sector_exposure.py`
- `tests/risk/test_paper_entry_ledger_snapshot.py`
- `tests/risk/test_remote_monotonic_verifier.py`
- `tests/test_entry_bootstrap_crosslink.py`
- `tests/test_exact_historical_decoder.py`
- `tests/test_fractional_data_fetcher.py`
- `tests/test_fractional_volume_contract.py`
- `tests/test_fractional_volume_readers.py`
- `tests/test_fractional_volume_storage.py`
- `tests/test_paper_entry_persistence.py`
- `tests/test_paper_entry_receipt.py`
- `tests/test_paper_entry_record_replay.py`
- `tests/test_paper_entry_settlement.py`
- `tests/test_paper_entry_storage_replay.py`
- `tests/test_paper_entry_terminal_record.py`
- `tests/test_paper_mixed_settlement_history.py`
- `tests/test_paper_outcome_decimal_context.py`
- `tests/test_paper_settlement_decimal_context.py`
- `tests/test_paper_settlement_kind_migration.py`
- `tests/test_paper_settlement_record_dispatch.py`
- `tests/test_regular_session_calendar.py`
<!-- untracked-list-end -->

## Explicit exclusions and review

- Do not stage ignored or local-only `.env`, `data/`, `dashboard.log`, `logs/`, `performance_results/`, `.pytest_cache/`, `.test_artifacts/`, `.superpowers/`, databases/SQLite/WAL, account snapshots, broker exports, credentials, private keys, model artifacts, or runtime output. Existing tracked historical test/model artifacts and public trust keys are part of old commits; do not accidentally add new binary or key material.
- `config/ibc/config.ini.template` is modified and must be reviewed as a **template** for real values. Review all modified/untracked documents, code, and staged diff for credential-like literals, account IDs, private endpoints, and test fixtures before publish. A filename check and limited signature scan found no obvious new secret file, but they do **not** certify content safe to push.
- Before a snapshot: coordinate current editor ownership; confirm `git status` still matches this manifest; stage only reviewed paths; inspect `git diff --cached --stat`, `git diff --cached --check`, and the full staged content; run a real secret scanner if available; verify ignored/runtime files are absent from the staged list; then commit a distinct source-only snapshot branch and push that exact branch. Do not reset, stash, delete, or overwrite concurrent work to make it clean. If the branch cannot be safely published from this host, transfer this manifest to a native Git owner instead of claiming cloud readiness.
- There is no `gitleaks` command available in this environment. The limited regex review of selected untracked source/document paths found no common key signatures; a full staged-content credential review remains outstanding.
- A second targeted scan of added lines across the 96 committed rebuild commits and of every currently modified/untracked text file found **zero** common private-key, AWS, GitHub/OpenAI token, or long configured provider/IBKR-password signatures. This is a narrow check, not proof against other secret formats. `gh repo view` could not reach `api.github.com` from this restricted environment, so remote privacy and publish rights were not verified. No push was attempted; the native Git owner must confirm those facts and review the exact staged snapshot.

## Bounded cloud work once the exact snapshot exists

Use the pushed snapshot branch and record its commit SHA. On an isolated cloud branch, implement only one-shot **local paper** BUY execution through the actual runner/gateway, a public atomic entry commit over the existing persistence/staging machinery, and receipt-bound projection. Remove legacy post-receipt accounting for that fill. Keep `robo_trader/safety/readiness.py` false; do not enable broker writes, change account data, or deploy. Exercise duplicate/lost responses and restart so the simulator runs once and cash, position, FIFO and trade each commit once. Add the actual BUY → protective SELL → BUY and projection/stop-install failure checks where supported.

Independent cloud checks can review global trade coverage, daily-filled-notional authority **mock contract tests only** (never an always-true runtime authority), canonical volume-unit rejection, and existing risk/admission tests using isolated temporary databases and fixtures. Cloud Linux may run deterministic Python tests and lint, but it cannot attest macOS launchd, the operator's Gateway/market-data entitlement, real account recovery, native browser/port ownership, or Docker Compose unless Docker is actually available and those tests run. Internet-dependent Yahoo smoke tests and localhost/process-enumeration tests require environment-specific interpretation, not silent skips. The local native result before new code was 4,206 passed, 0 failed, 2 Docker skips; repeat relevant checks after changes.

See `HANDOFF_2026-09-29_paper_rebuild.md` and the external forensic plan at `/Users/oliver/Documents/Codex/2026-09-13/prior-conversation-with-codex-conversation-role/outputs/paper-runtime-forensic-plan-2026-09-29.md` for the acceptance matrix and remaining Gate A evidence.
