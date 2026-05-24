# Robust Startup Safety — Plan

**Status:** Plan approved 2026-05-23. MVP-1 (startup safety gate) selected as first implementation. This document is the input to the design-spec session.

## Incident that motivated this plan

On 2026-05-22 the trader was dead for ~4 hours after a normal NVDA -2.93% daily move tripped a hardcoded 2% per-position-loss kill switch. The switch state persisted to `data/kill_switch_state.json`. Every watchdog restart re-loaded `triggered=True`, the process died within seconds, and the watchdog logged 18 consecutive "Restart FAILED" lines without escalating — because its liveness check is mtime-based and the restart loop kept producing fresh log lines.

Root pattern: **failure modes that look healthy enough to fool monitoring**. Each layer (config → preflight → runtime → watchdog → escalation) was "working as designed," but no layer could see that the chain was stuck.

## Defense-in-depth model

```
   Layer            Concern                        2026-05-22 status
   ───────────────  ─────────────────────────────  ──────────────────────────────
   Configuration    Are thresholds realistic?      Hardcoded 2% in source
   Preflight        Is the system safe to start?   NO GATE — just runs and crashes
   Runtime          Detect operational anomalies   ConnectionHealth ✓ (this branch)
   Cleanup          Recovery on transient issues   recover_connection ✓ (this branch)
   Watchdog         Restart on hard failure        mtime-only, can't see livelock
   Escalation       Wake the operator              Layer 6 exists but trigger weak
   Observability    Was the right thing done?      Logs only, no synthesis
```

Runtime + Cleanup were strengthened by this session's commits. The weak links remaining are **Preflight, Watchdog content-liveness, and Escalation**.

## Already done (2026-05-23 audit + fix session)

Commits on `feature/persistent-ibkr-connection`:
- `b142d2a` C2 — RecoveryExhaustedError propagation (recovery exhaustion exits cleanly, not zombie loop)
- `c3a4cbe` C3 — RECOVERING state wired (no recovery re-entrancy)
- `009a7a4` C4 — stop_loss_monitor price re-warm after recovery (no blind window)
- `315ea6a` H1 — kill switch auto-reset on connection-related triggers
- `b8062bc` C1 — removed dead `restart_subprocess()` reference
- `a666015` M5 — `consecutive_failures` public property
- `0d7831f` M4 — `IBKRClientProtocol` type contract
- `baadd26` config — `max_position_loss_pct` env-configurable, default raised 2% → 5% (also absorbed M1's kill-switch log throttle + per-path check removals)
- `6f195a3` H2 — process-wide Gateway recovery lock (multi-portfolio race prevention)

Also: kill switch state files cleared, backup at `data/kill_switch_state.json.bak.2026-05-22-1627`.

## MVP — three components, ordered by leverage

### MVP-1: Startup safety gate (HIGHEST LEVERAGE — would have prevented 2026-05-22)

A preflight script `scripts/preflight_check.py` invoked by `START_TRADER.sh` BEFORE launching the runner. Exits non-zero if any of:

- `data/kill_switch_state.json` shows `triggered=True`
- `data/kill_switch.lock` exists
- Last `equity_history` row is >24h old (suggests stale state)
- Any of `RISK_MAX_*` env vars are below "sane minimum" (e.g., `MAX_POSITION_LOSS_PCT < 0.02` warns; `< 0.01` blocks)
- IBKR Gateway port not listening
- Zombie CLOSE_WAIT connections on port 4002

Each failure prints **what's wrong + what to do** in plain English. Critical: runs ONCE per startup, BEFORE the process forks into "trying to recover" mode. Operator sees the message immediately.

Converts a 4-hour silent livelock into a 30-second clear-message-on-startup. The persisted state stays the same — the *response* changes.

### MVP-2: Content-based watchdog liveness

Replace `mtime > T` check with a structured-event check: the runner must emit `event=cycle_heartbeat` every N seconds. The watchdog parses the last K log lines, finds the most recent heartbeat, escalates if older than 2N.

Distinguishes:
- **Healthy**: heartbeat fresh ✓
- **Degraded but alive**: process running, logs being written, but no heartbeat → escalate
- **Dead**: no logs at all → restart

### MVP-3: Fast-fail watchdog escalation

If the process dies within 60s of startup **3 times in a row**, the watchdog stops retrying and immediately:
- Sends macOS notification (Layer 6 already exists)
- Writes a `data/.startup_blocked` flag file
- Logs `event=startup_blocked_persistent` (queryable)
- Exits its restart loop entirely until the flag file is removed

"consecutive_failures: 18" should never happen. After 3 fast-fails, the system **declares defeat and asks for help**, not silently retries.

## Phase 2

- **P2-1**: Loud startup config logging (one structured event with every safety threshold)
- **P2-2**: H2 multi-portfolio Gateway lock — ✅ DONE (committed `6f195a3`)
- **P2-3**: Self-expiring kill switch (severity field: `transient` / `sticky` / `permanent`)
- **P2-4**: `/api/health` endpoint exposing single source of truth
- **P2-5**: Chaos drills in CI (`@pytest.mark.chaos`)

## Implementation order

1. **MVP-1** (Startup safety gate) ← biggest win, ~half day. THIS IS THE NEXT THING TO DESIGN.
2. MVP-3 (Fast-fail escalation) — ties into existing Layer 6, ~few hours
3. P2-1 (Loud config logging) — tiny change, big diagnostic payoff
4. MVP-2 (Content liveness) — needs heartbeat plumbing
5. P2-3 (Self-expiring kill switch)
6. P2-4, P2-5 (longer-term observability + CI)

## Related artifacts

- `CLAUDE.md` — project guidelines, especially "🚨 CRITICAL: NEVER DELETE USER DATA", common-mistakes table, gateway management section
- `START_TRADER.sh` — the entry point preflight must hook into
- `scripts/install_watchdog.sh`, `~/Library/LaunchAgents/com.robotrader.watchdog.plist` — watchdog setup
- `~/.claude/projects/-Users-oliver-Projects-robo-trader/memory/project_watchdog_layer6.md` — existing Layer 6 escalation
- `docs/superpowers/plans/2026-05-16-persistent-ibkr-connection.md` — prior plan style/format to mirror
- `docs/superpowers/specs/2026-05-16-persistent-ibkr-connection-design.md` — prior design spec style/format to mirror
- `data/kill_switch_state.json.bak.2026-05-22-1627` — the actual triggered state from the incident, useful as preflight test fixture
