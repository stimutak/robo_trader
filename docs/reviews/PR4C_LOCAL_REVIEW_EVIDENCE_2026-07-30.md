# PR4C Local Review Evidence

Date: 2026-07-30
Branch: `codex/pr4c-fifo-settlement`
Base: `main` at `4c7ea47`, including merged PR4B

## Scope reviewed

- producer-owned local-paper fill and exact commission evidence;
- transaction-owned FIFO append and exact replay;
- atomic FIFO, compatibility projection, and terminal outbox settlement;
- immutable settlement-to-FIFO lineage;
- stopped-system query-only recovery authentication;
- PR4B sealed-opening-manifest and trigger-body protections;
- PR4B in-transaction durable schema and trigger revalidation;
- paper/read-only containment and absence of runtime startup actions.

## Local verification

- Full repository suite: `3047 passed, 5 skipped, 20 warnings`.
- Combined FIFO, migration, exact-bootstrap, settlement, failure-injection,
  and recovery-status matrix: `220 passed`.
- Existing-position and stop-event suite: `168 passed`.
- Migration, exact bootstrap, and FIFO foundation suite: `120 passed`.
- Terminal failure-injection and recovery-status suite: `66 passed`.
- Focused runtime FIFO tests cover multiple partial fill events, exact
  commission allocation, a rebate, deterministic replay, conflicting replay,
  missing-epoch refusal, transaction rollback, and account-wide realized P&L
  accumulation across multiple symbols.
- End-to-end settlement proves that a later AAPL reduction preserves earlier
  MSFT realized P&L in the same epoch instead of replacing the account total
  with the current instrument's cumulative snapshot.
- Hot-schema regressions prove that a temporary
  `paper_fifo_settlement_links` shadow and a persistent `AFTER INSERT` trade
  injection trigger both fail before mutation, with no fill, commission,
  compatibility trade, settlement, or link partially committed.
- The persistent-trigger regression uses SQLite's case-insensitive `Trades`
  spelling, and complete-definition regressions reject both a same-column
  replacement table with an added `CHECK` and a malformed hot-table index.
- Quoted SQL literals retain their exact case during DDL authentication;
  lowercase `'usd'` is rejected. The reviewed multiuser-v1 compatibility DDL
  is accepted as an exact alternate schema and completes an end-to-end
  settlement. FIFO subtotal tests also lower the ambient Decimal precision and
  prove the 96-digit local context preserves an exact cross-symbol result.
- Settlement failure injection covers failures immediately after FIFO append,
  after compatibility trade insertion, and after immutable FIFO-link insertion;
  every case leaves zero fill, commission, trade, settlement, and link rows.
- Recovery tests reject a tampered FIFO link and a missing FIFO-link schema.
- Changed Python files pass Black, isort, and Flake8; compilation, dependency
  consistency, and whitespace/diff checks pass.
- The new FIFO runtime and terminal-request modules pass isolated mypy, and the
  new accounting/execution/terminal modules pass targeted Bandit.

The known full-suite warnings and expected skips predate this slice. No test
opened a broker connection or used an authoritative trading database.
Repository-wide Flake8 still reports nine pre-existing findings in the Claude
emoji hook and two training scripts; repository-wide mypy remains baseline
debt tracked outside PR4C. The broader changed-production Bandit scan reported
only four pre-existing low-severity findings and no medium/high findings.

## Safety conclusion

The code slice connects only the sealed local-paper execution result to the
exact FIFO ledger. It neither enables startup nor changes the live-order
boundary. Operational bootstrap, backup/restore, current reconciliation, and
Gate-A evidence remain outstanding.
