# Paper rebuild handoff — 2026-09-29

**Publication update:** the complete source checkpoint is now on
`codex/paper-rebuild-integration-2026-09-29` under
[draft PR #130](https://github.com/stimutak/robo_trader/pull/130), initially
published as `e0f3999`. Read [CURRENT_STATE.md](../CURRENT_STATE.md) for the current
continuation branch, CI/review results and merge status. The local-only inventory
below is the historical state before that publication, not a remaining transfer
requirement. Main and the operational paper checkout have not been upgraded.

This is a state snapshot for continuation, not an activation decision. Keep Gate A closed: `robo_trader/safety/readiness.py` remains false, the gateway BUY path still rejects, and the runner does not yet supply an entry authority. Do not infer trading readiness or profitability from passing tests.

## Exact checkout state

- Rebuild: this checkout, branch `codex/paper-readiness-2026-09-13`, HEAD `cae45e0`. Fetched `origin/main` is `57f7634`; rebuild is 96 commits ahead and zero behind. At handoff `git status --short` contained **82 modified or untracked paths** before this file. Preserve all of them; they include concurrent entry, accounting, market-data, and test work. This handoff adds one more untracked path. Review exact status and ownership before editing or committing.
- Operational checkout: `/Users/oliver/Projects/robo_trader`, branch `feature/paper-relaunch-m5`, HEAD `a09b567` plus the operational owner's local edits. It is a distinct, diverged execution candidate. Do not overwrite or silently deploy rebuild code there.
- The cloud checkout cannot see either local uncommitted tree, and `origin/main` lacks the 96 rebuild commits plus dirty files. **A reviewed remote snapshot of the complete rebuild is a prerequisite for cloud code implementation.** Select and verify what belongs in that snapshot with the local owners; do not infer that pushing the existing HEAD alone transfers the implementation. This handoff makes no commit or push.

## Verification and current boundary

The latest **native-host full suite** reported `4,206 passed, 0 failed, 2 skipped` (the two are Docker Compose checks), exit 0. This supersedes the restricted-sandbox `4,200 passed, 6 environment failures, 2 skipped` in the detailed test-repair report. The stop timestamp test was fixed by constructing timestamps against the validation clock; production validation was unchanged. Production read-only containment and pairs quarantine tests were enabled; the latter does not prove enabled-pairs BUY safety. Docker containment remains unverified until a Docker-capable host runs it. Repeat affected tests and the full suite after new implementation.

The operational Gateway's read-only API handshake and port 4002 listener were observed, but fresh quote requests returned IB error 354, so usable real-time protective data is unproven. A sanctioned startup refused stale July 13 equity; runner/dashboard did not start. The operational owner is separately preparing account recovery and startup evidence. See `/Users/oliver/Projects/robo_trader/docs/PAPER_RECOVERY_STATUS_2026-09-29.md` for its current findings; do not modify account data from this handoff.

## Next bounded rebuild code slice

Implement the **consumed one-shot local-paper BUY dispatch, public atomic entry commit, and receipt-bound runner projection** using the existing reserve/claim/staging/receipt components. Keep `readiness.py` false. In particular, `paper_reduction_gateway.py` still unconditionally rejects BUY around line 1379; `runner_async.py` leaves entry authority absent near 2894; `paper_entry_persistence.py` has staging but no consumed public commit adapter. Once an entry receipt exists, remove the runner's legacy post-BUY `record_trade` / `_update_position_atomic` accounting path for that result: applying both paths would double-write cash, positions, FIFO and trades. Do not activate broker order placement; this slice uses the local paper simulator.

Acceptance must exercise the actual runner and gateway: one admitted BUY invokes the simulator once and commits one durable cash/position/FIFO/trade transition; response loss or retry neither resubmits nor reapplies money; failed stop installation or projection freezes new entries while retaining committed evidence. Then test restart/replay, BUY → protective SELL → BUY, and failure injection across claim, commit, projection, release and reconnect. Keep pairs quarantined and other entry strategies out of baseline scope.

Independent work still required before Gate A: real market-data entitlement and fresh contract-bound quotes; trusted share-volume unit proof (unknown units must block admission); shared daily fill accounting with complete history; independently provisioned remote filled-notional authority and enrollment (never a permissive stub); complete signal-to-risk evidence wiring and account-wide serialization; actual backup/restore and account reconciliation; native Docker checks; candidate-specific startup, browser delivery, and independent review. No single test result or Gateway connectivity substitutes for these.

## Deployment gap and references

Historical launcher, WebSocket, and persistent-connection fixes should be reused. Several important fixes exist only in the rebuild: correlated subprocess replies, replacement-generation quote rewarm after reconnect, atomic reduction accounting, and listener-bind failure propagation. The operational checkout still has old paths, including stale-price stop rearm and separate accounting writes. Do not treat old historical handoffs as proof those fixes are deployed. The detailed evidence matrix and ordered Gate A plan are in `/Users/oliver/Documents/Codex/2026-09-13/prior-conversation-with-codex-conversation-role/outputs/paper-runtime-forensic-plan-2026-09-29.md`; test repair detail is in the adjacent `paper-test-repair-2026-09-29.md`.

**Cloud-suitable task after a complete reviewed remote snapshot exists:** On an isolated branch from that exact snapshot, implement only the one-shot local-paper BUY gateway dispatch, public atomic entry commit and receipt projection described above, with actual runner/gateway fault and restart tests. Do not change live/broker-write permissions, readiness flags, account data, strategy scope, or operational deployment. Return a reviewable diff and test evidence; further Gate A work remains separate.
