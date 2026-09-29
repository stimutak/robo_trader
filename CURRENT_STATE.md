# Current continuation point — September 29, 2026

This source checkpoint contains the rebuild previously held only on one Mac:
96 commits beyond `origin/main` at `57f7634`, plus the entry accounting,
market-data, calendar, risk and test work that was still uncommitted. Its review
branch is `codex/paper-rebuild-integration-2026-09-29`. Use that branch until its
PR is merged; after merge, use updated `main`. Do not assume an older `main`
checkout contains this checkpoint. Record `git rev-parse HEAD` when continuing.

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
