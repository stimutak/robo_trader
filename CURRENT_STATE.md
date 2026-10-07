# Current continuation point — September 29, 2026

This source checkpoint contains the rebuild previously held only on one Mac:
96 commits beyond `origin/main` at `57f7634`, plus the entry accounting,
market-data, calendar, risk and test work that was still uncommitted. Its review
branch is `codex/paper-rebuild-integration-2026-09-29`. Use that branch until its
PR is merged; after merge, use updated `main`. Do not assume an older `main`
checkout contains this checkpoint. Record `git rev-parse HEAD` when continuing.

The complete checkpoint was pushed as `e0f39996843b65ca53ed352b231bc36f78ed8565`
and is under [draft PR #130](https://github.com/stimutak/robo_trader/pull/130).
Publication is complete; merge is pending checks and review. The operational
checkout was not upgraded or deployed. Subsequent commits on this same branch
carry review/CI fixes; use its current HEAD when continuing implementation.

**Gate A remains closed. Source publication does not deploy or enable trading.**
Runtime readiness stays false, the gateway rejects BUY, IBKR connections remain
paper/read-only, and no live broker orders are authorized. Existing operational
data, secrets, credentials, logs and model artifacts are not part of the new
source snapshot. Do not reset a portfolio or bypass preflight to make it start.

## Read these in order

1. [Rebuild handoff](handoff/HANDOFF_2026-09-29_paper_rebuild.md): current code,
   verification, missing runtime integration and acceptance criteria.
2. [Operational recovery handoff](handoff/HANDOFF_2026-09-29_recovery.md): the
   separate paper checkout, Gateway/watchdog stopping state, accounting mismatch,
   account access and market-data entitlement findings.
3. [Recovery tracker](docs/PAPER_RECOVERY_STATUS_2026-09-29.md): required Gate A
   evidence, ownership and remaining tasks.
4. [Authoritative remediation plan](docs/ROBOTRADER_REMEDIATION_PLAN_2026-07-20.md)
   and [execution record](docs/PAPER_READINESS_EXECUTION_2026-09-13.md).

Local absolute paths in historical handoffs identify the original Mac's evidence
and checkout locations; use the repository-relative links above on another
computer. External `/tmp` probe logs and local reports are not shipped artifacts.
The local `feature/paper-relaunch-m5` checkout remains a distinct operational
candidate and must not be silently overwritten or restarted by this integration.

## Next work

- Implement consumed one-shot local-paper BUY dispatch, a public atomic entry
  commit and receipt-bound runner projection. Preserve the closed readiness
  gate while developing and testing these components. Eliminate duplicate legacy
  accounting for a receipted fill; test loss/retry/restart and BUY→SELL→BUY through
  actual runner/gateway boundaries.
- Complete the independent authority, quote freshness and volume provenance,
  risk wiring, backup/restore/reconciliation and final candidate review evidence
  before requesting a separate operator activation decision.
- Resolve the existing accounting mismatch only with the user's chosen basis
  and a verified backup. The old operational watchdog still retries against
  stale equity; its separately tested local fix has not been activated.
- Confirm the paper account's parent and live market-data sharing through IBKR.
  The user has two self-managed main accounts, personal and **Cap**. Robo paper
  may be linked to one; the relationship remains unverified. Both main logins use
  a currently unavailable 2FA phone. The paper portal showed `Individual (Demo)`
  and reset options, without parent/subscription controls. Do not reset it.

The latest native suite reported **4,206 passed, zero failed, two Docker Compose
skips** before this documentation/publication step. This is not proof of Gate A
readiness. PR checks and independent review must pass before source merge; retest
after code changes. No deployment is part of source integration.

Cloud or another computer can continue source work from the published exact
checkpoint. Real Gateway/account access, entitlement, production-data recovery,
watchdog activation and deployment evidence require the relevant local host and
operator access. Proposed future topology is one always-on Mac Studio Gateway
and a roaming laptop via private Tailscale/SSH; external-Gateway mode still needs
implementation and separate client-ID/state allocation. Another human such as
Carolyn needs authorized account access and verified data entitlement, not merely
a shared tunnel.

## Publication checks and review configuration

At the first PR run, the configured GitHub Actions ran despite draft status.
Docker build, container structure and Compose rendering passed, as did the
integration/performance matrices and BugBot. Full unit matrices were still
running when this note was written. Import-order checks failed; the follow-up
uses the same CI formatter versions and passes Black, isort and flake8 on all
141 Python paths changed from main. Whole-repository local checks also reported
older archive/script formatting debt outside those changed paths.

Remaining failures need resolution before merge: the Claude review workflow
returned HTTP 401 because its OAuth token was revoked and produced no review.
The six Bandit SQL findings were independently verified as fixed-string or
immutable allowlisted-identifier construction, with all external values bound.
Six line-specific explanations/suppressions leave query ASTs unchanged; the same
CI Bandit 1.9.4 command now passes with zero medium/high findings.
TruffleHog reported one unverified URI in
the synthetic credential-bearing `authority.example` test input used to assert
rejection before network access. It is not an operational credential. Preserve
the rejection test and handle that exact false positive without suppressing
unrelated secret scanning. Its same-line fixture annotation prevents future
HEAD-only findings, but the original `e0f3999` commit remains in the scanned PR
history. Clearing that historical result needs an exact reviewed exception or
an explicitly coordinated history rewrite; this publication did neither and
did not force-push. Check the live PR for subsequent results.

A separate cloud reviewer checked backtesting/maintenance at **exactly
`e0f39996843b65ca53ed352b231bc36f78ed8565`** and found no verified new integration
blocker or reproduced accounting duplication/look-ahead defect. Its system
Python 3.12 run of `tests/backtesting tests/maintenance` passed **253 tests in
5.80 seconds**. The initially bundled interpreter failed 52 maintenance checks
because its `_sqlite3` lacks `__file__`; those passed on system Python. This is
scoped external evidence, not a full-repository approval. After the import-order
follow-up, the local independent reviewer confirmed all 51 files retained the
same non-import AST and imported names, and repeated **236 focused passes**.

GitHub's repository API reports `allow_auto_merge=false`; this PR has no
auto-merge request. Main has no branch protection and no repository rulesets,
so required review/check enforcement is not configured. Configured reviews and
green CI therefore do **not** imply an automatic merge. External app check suites
were queued, but that is not evidence that those apps completed a review or that
their draft-PR policies were verified. No repository protections, secrets, or
auto-merge settings were changed during publication.

To restore the existing Claude review workflow, the account owner should run
`claude setup-token` in a private local terminal, complete the account sign-in,
and replace the repository Actions secret named `CLAUDE_CODE_OAUTH_TOKEN` under
GitHub **Settings → Secrets and variables → Actions**. Do not paste the token in
chat, a PR, or a tracked file. Then rerun the failed Claude Code Review job and
verify that an actual review appears. This follows the
[official action setup](https://github.com/anthropics/claude-code-action/blob/main/docs/setup.md).
The failed run was [36597299209](https://github.com/stimutak/robo_trader/actions/runs/36597299209);
re-running it before replacing the revoked credential cannot repair authentication.
