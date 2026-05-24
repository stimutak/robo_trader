# Startup Safety Gate — Design Spec

**Status:** Draft — pending implementation
**Branch:** authored on `worktree-agent-ae6a40f59c516cb74` (worktree branched off `feature/persistent-ibkr-connection`); caller may cherry-pick or rebase onto whichever integration branch is current.
**Date:** 2026-05-23
**Supersedes:** Nothing — this is a new component (MVP-1 in `docs/superpowers/plans/2026-05-23-robust-startup-safety-plan.md`)
**Related:** `docs/superpowers/plans/2026-05-23-robust-startup-safety-plan.md` (motivating plan), `docs/superpowers/specs/2026-05-16-persistent-ibkr-connection-design.md` (style precedent)

---

## 1. Status & Context

This spec designs the **Startup Safety Gate**, the first of three MVP components in the robust-startup-safety plan. It is a preflight script (`scripts/preflight_check.py`) that runs **before** `runner_async.py` starts. Its job is to convert silently-fatal misconfigurations and stale state into a **30-second, plain-English error message at the operator's terminal**, instead of a 4-hour silent livelock.

It is bounded scope by design. It does NOT fix the runtime loops, it does NOT modify the watchdog liveness check, and it does NOT escalate. Those are MVP-2 and MVP-3.

The full plan (defense-in-depth layer model, the 2026-05-22 incident, and what was already fixed) is in `docs/superpowers/plans/2026-05-23-robust-startup-safety-plan.md`. Read that first if you haven't.

---

## 2. Motivation

### The 2026-05-22 incident (one paragraph)
The trader was dead for ~4 hours after a normal NVDA -2.93% daily move tripped a hardcoded 2% per-position-loss kill switch. The state persisted to `data/kill_switch_state.json` (`triggered=True`). Every subsequent watchdog restart loaded the persisted state, the runner died within seconds of startup, and the watchdog's mtime-based liveness check kept seeing fresh log lines from the rapid restart loop, so it never escalated. 18 consecutive "Restart FAILED" lines, no notification.

The root pattern: **failure modes that look healthy enough to fool monitoring.** The kill switch was *correctly* persisted (no bug there) and the watchdog was *correctly* restarting (no bug there). The thing missing was anyone telling the operator "you have a persisted kill switch — clear it or raise the threshold."

A preflight gate would have surfaced that within 30 seconds of the first restart, before the runner even called `connect()`.

### Why a gate, not a fix
The persisted kill switch state is **correct behavior** — we don't want it auto-cleared on every restart, because then loss-based triggers would lose their meaning. The fix is to make the persisted state **visible at startup**, so the operator can make an informed clear-or-keep decision, with the system refusing to proceed until they do.

### Coverage
The plan enumerates six classes of preflight failure that could waste hours of operator time. The 2026-05-22 incident hit class #1 (persisted kill switch). Classes #4 (misconfigured loss limit) and #5 (Gateway port not listening) are equally capable of producing the same "silent restart loop" pattern. Each check below covers one such class.

---

## 3. Goals

Numbered, testable, in priority order:

1. **G1 — Block on triggered kill switch.** If `data/kill_switch_state.json` shows `triggered=True` OR `data/kill_switch.lock` exists, exit non-zero with a message naming the state file, the trigger reason, and the remediation command.
2. **G2 — Block on stale equity history.** If the most recent `equity_history` row is more than 24h old (and the market has been open at least once in that window), exit non-zero. Stale equity is the strongest signal that a prior session died uncleanly.
3. **G3 — Block on dangerously-low loss thresholds.** If `RISK_MAX_POSITION_LOSS_PCT < 0.01` (i.e., less than 1%) exit non-zero. If `< 0.02` (less than 2%) warn but allow start.
4. **G4 — Block on unreachable Gateway.** If the Gateway API port (`4002` paper, `4001` live, determined by `EXECUTION_MODE`) is not in `LISTEN` state via `lsof`, exit non-zero with the relevant `START_TRADER.sh` step to run.
5. **G5 — Block on zombie connections.** If `lsof` shows any `CLOSE_WAIT` on the Gateway port, exit non-zero with the kill-or-restart instructions.
6. **G6 — All-pass output is observable.** Happy path prints one structured line per check (status=PASS) and a summary line, so the operator can see *what was checked* not just *that it passed*.
7. **G7 — Auditable bypass exists.** The operator can override any BLOCK with a single, logged action that survives in shell history (`--force` + reason string). No env var that silently disables checks forever.
8. **G8 — Latency budget ≤ 5s wall clock** on a healthy system. Slow checks (lsof, sqlite) are bounded by per-check timeouts and run in parallel where independent.
9. **G9 — Each check is independently testable.** Pure-Python check objects with a `tmp_path` pytest fixture for filesystem state and a subprocess mock for `lsof`. No real IBKR Gateway needed.
10. **G10 — Pluggable check registry.** Adding a new check (Phase 2 will add more) is one entry in a list, not a refactor.

---

## 4. Non-Goals

Explicit scope boundaries — these are NOT in this spec:

- **N1 — Does NOT modify state.** Never auto-clears a triggered kill switch, never kills zombies, never restarts Gateway. Preflight observes and reports; the operator (or a separate script) acts.
- **N2 — Does NOT replace `gateway_manager.py` checks.** That script's `status` command is more granular and stays the source of truth for Gateway diagnostics. Preflight uses lsof directly because shelling out to gateway_manager would couple us to its CLI evolution.
- **N3 — Does NOT touch the watchdog.** Content-based liveness is MVP-2.
- **N4 — Does NOT do escalation.** Fast-fail-after-3 is MVP-3.
- **N5 — Does NOT do health checks during operation.** This runs once, before the runner starts. The runtime `ConnectionHealth` module (already shipped on `feature/persistent-ibkr-connection`) handles in-flight health.
- **N6 — Does NOT call IBKR.** No `ib.connect()`, no socket connections to the API port. Port-listening check is `lsof -sTCP:LISTEN` only — never `socket.connect_ex` (creates zombies, see CLAUDE.md 2025-12-06 mistake row).
- **N7 — Does NOT modify config files.** If a threshold is wrong, preflight tells the operator the env var name and the fix command. It does not edit `.env`.
- **N8 — Does NOT block on warnings.** WARN status surfaces but proceeds with exit 0.
- **N9 — No multi-step ritual to bypass.** Bypass is a single flag (see §6.3). Operator paged at 3am should not have to navigate a menu.

---

## 5. Architecture

### 5.1 Script location and entry point

```
scripts/preflight_check.py        # CLI + main()
robo_trader/preflight/__init__.py # package
robo_trader/preflight/checks.py   # Check classes + registry
robo_trader/preflight/runner.py   # orchestration (parallelism, output)
robo_trader/preflight/result.py   # CheckResult dataclass
tests/test_preflight_checks.py    # unit tests per check
tests/test_preflight_runner.py    # orchestration tests
```

The thin script in `scripts/preflight_check.py` is just a CLI wrapper that imports from `robo_trader.preflight` and exits with the appropriate code. The actual logic lives in the package so it is unit-testable without subprocess invocation.

### 5.2 Exit codes

```
0    All checks PASS or WARN — safe to proceed
1    At least one check BLOCKED — abort startup
2    Operator passed --force on a BLOCK — proceeded with audit log
3    Preflight itself failed (uncaught exception, broken environment)
```

Code 2 distinguishes "we knew about a block and chose to bypass" from "no blocks at all." Useful in operational logs.

Note: code 3 must be impossible in normal operation — every check catches its own exceptions and converts them to `BLOCK` status with a "preflight check N raised" message. Code 3 only fires for things like `ImportError` at module-load time.

### 5.3 `CheckResult` dataclass

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class CheckStatus(Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"

@dataclass(frozen=True)
class CheckResult:
    name: str                          # short identifier: "kill_switch_state"
    status: CheckStatus                # PASS / WARN / BLOCK
    message: str                       # one-line summary for human display
    remediation: str = ""              # operator instructions, multiline OK
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0               # populated by runner
```

Why `frozen=True`: results are immutable once produced — easier to reason about in tests, and the runner serializes them to JSON without worrying about late mutation.

Why `remediation` as a separate field from `message`: the message goes on stdout next to the status badge; the remediation goes on its own indented block. JSON output exposes both as separate keys, useful for downstream tooling.

Why `details: dict`: machine-readable extras (file paths, threshold values, port numbers). Stays out of the human display unless `--verbose` is set, always present in JSON output.

### 5.4 Check protocol

```python
from typing import Protocol

class Check(Protocol):
    name: str                          # stable identifier (kebab-case)
    description: str                   # human label, e.g., "Kill switch state"
    timeout_seconds: float             # per-check soft budget (G8)

    def run(self, context: PreflightContext) -> CheckResult: ...
```

`PreflightContext` is a small dataclass passed to every check, holding shared inputs (project root path, `.env`-resolved env dict, target port, dry-run flag). Avoids re-reading `.env` per check and lets tests inject fakes.

### 5.5 Registry pattern (G10)

```python
# robo_trader/preflight/checks.py
ALL_CHECKS: list[Check] = [
    KillSwitchStateCheck(),
    KillSwitchLockCheck(),
    EquityHistoryFreshnessCheck(),
    RiskThresholdCheck(),
    GatewayPortListeningCheck(),
    ZombieConnectionsCheck(),
]
```

Adding a Phase 2 check (e.g., "DB schema migrations applied") is one line. Tests reference checks by index/name, not by import path of a module-internal function.

Checks are independent — order in the list determines display order but not pass/fail interdependence. The runner runs them all even if early ones fail; the operator wants to see *all* the problems at once, not fix one and re-run only to discover the next.

### 5.6 Severity model — flat vs. tiered (Q1)

**Decision: three-level (PASS / WARN / BLOCK), no REQUIRE_CONFIRM tier.**

Justification by check:

| Check | Why this severity model | Concrete example |
|---|---|---|
| Kill switch state | BLOCK only — no warning tier. If `triggered=True`, the system has affirmatively decided not to trade. Anything weaker than BLOCK is a regression to the 2026-05-22 behavior. | `triggered=True` → BLOCK |
| Kill switch lock file | BLOCK only — symmetric with state file (deny-by-default fail-closed signal per CLAUDE.md). | `data/kill_switch.lock` exists → BLOCK |
| Equity history stale | BLOCK if >24h, no WARN tier. The window itself is the threshold. | Last row at T-26h → BLOCK |
| Risk thresholds | **WARN + BLOCK tiers**. Below 2% is suspicious (the 2026-05-22 value), below 1% is almost certainly an accidental typo. Two thresholds catch both. | `0.018` → WARN; `0.005` → BLOCK |
| Gateway port | BLOCK only — the runner cannot do useful work without it. | port `4002` not in LISTEN → BLOCK |
| Zombies | BLOCK only — zombies guarantee a handshake failure within a few cycles. | Any `CLOSE_WAIT` count > 0 → BLOCK |

REQUIRE_CONFIRM was considered (force an interactive prompt) and rejected: at 3am with a watchdog auto-restart, there is no human at the keyboard. An interactive prompt becomes a hang. The bypass mechanism (§6.3) is the right knob for "I know about this block."

### 5.7 What checks **do not** do (cross-check vs. N1)

No check ever:
- Modifies a file (no `.unlink()`, no `.touch()`, no writing JSON state)
- Sends a network packet (no `socket.connect_ex`, no IBKR API call, no HTTP request — `lsof` is local kernel state)
- Spawns the gateway, the runner, or any other long-lived process
- Reads secrets out of `.env` (the threshold values are not secret; we don't need credentials)

Enforcing this in code review is easier when each check is a small class with a single `run()` method.

---

## 6. Operational behavior

### 6.1 Parallelism (G8)

The six initial checks split into:
- **I/O-cheap (parallel)**: kill switch state read, kill switch lock stat, risk threshold env-var read, equity-history sqlite SELECT — all bounded by per-check 1s timeout, all run concurrently in a thread pool.
- **subprocess-bound**: lsof for port-listening and lsof for zombies. These two run sequentially (same binary, fast) but the pair runs in parallel with the I/O checks.

Worst case latency on a healthy system: 1–2 seconds (lsof on macOS is fast). With degraded I/O (slow disk, large `equity_history` table), the per-check timeout caps each check at its `timeout_seconds`. Total budget is `max(check.timeout_seconds for check in ALL_CHECKS)`, not the sum.

If a check exceeds its timeout, it returns `CheckResult(status=BLOCK, message="check exceeded timeout", ...)`. We prefer to BLOCK over silently allowing start with unverified state.

### 6.2 Hook into `START_TRADER.sh` (Q4)

**Decision: invoke as a subprocess after the existing Gateway/zombie steps, before launching the dashboard and runner.**

Why subprocess instead of `import & call`:
- `START_TRADER.sh` already shells out to Python for the readonly API check pattern; consistent.
- Exit code is the contract — bash's `if !` natural fit.
- Failure isolation: a preflight import error doesn't take out the shell script's error reporting.

Why after the existing Gateway/zombie steps:
- The current Gateway-management block in START_TRADER.sh (steps 2 and 3 in the file) is what makes the port LISTEN and clears zombies in the first place. Preflight then **verifies** the state the shell script just established. Running preflight *before* the Gateway block would always block on "port not listening."
- However, preflight runs *before* the Python environment is fully exercised, so we don't waste time spinning up the dashboard if the safety gate blocks.

Exact slot — between current Step 4 ("Set up Python environment") and Step 5 ("Start dashboard"):

```bash
# Step 4.5: Preflight safety gate
echo "4.5. Running preflight safety gate..."
if ! $PYTHON scripts/preflight_check.py; then
    echo ""
    echo "=========================================="
    echo "❌ PREFLIGHT SAFETY GATE BLOCKED STARTUP"
    echo "=========================================="
    echo ""
    echo "Preflight reported blocking issues above."
    echo "Resolve each one and re-run ./START_TRADER.sh, or"
    echo "bypass with: $PYTHON scripts/preflight_check.py --force \"<reason>\""
    echo "(then re-run ./START_TRADER.sh — bypass is per-invocation, not persistent)"
    echo ""
    exit 1
fi
echo ""
```

Preflight's own non-zero exit already prints the BLOCK details. The wrapping bash block adds the meta-instruction (how to bypass).

### 6.3 Bypass mechanism (Q2, G7)

**Decision: `--force "<reason>"` CLI flag. No env var. Reason string is mandatory.**

```bash
# Operator runs (after explicit decision):
python3 scripts/preflight_check.py --force "cleared NVDA kill switch, raised threshold to 5%, ticket #1234"
```

Behavior of `--force`:
1. All checks still run and print their results (so the operator sees what they're overriding).
2. If any check returns BLOCK and `--force` is set:
   - The operator's reason string is logged to `robo_trader.log` at WARNING level with `event=preflight_bypass`.
   - A line is appended to `data/preflight_bypass.log` (newline-delimited JSON, one entry per bypass) with timestamp, reason, and the list of bypassed check names.
   - Exit code is **2** (not 0), so START_TRADER.sh can show "started with bypass" rather than "all clean."
3. If `--force` is given but no checks BLOCKed, exit 0 with a "force not needed" warning. (Prevents operators from leaving `--force` in their shell history as a copy-paste default.)
4. Reason string must be ≥10 characters and not match a small denylist of placeholders (`force`, `bypass`, `whatever`, `idk`). Forces a real-sentence rationale.

Why not an env var (e.g., `PREFLIGHT_BYPASS=1`):
- Env vars get set in `.env` and silently disable the gate forever. The whole point of the gate is that the operator stops and looks. A persistent bypass is worse than no gate.
- Bypass needs to be visible at the call site — `.bash_history` `grep` finds the `--force` invocations later.

Why not a delete-this-file mechanism (e.g., remove `data/.preflight_required`):
- Same problem as env var: easy to leave deleted, hard to audit.

Why a reason string:
- "What did I bypass and why" is the question 24 hours later. Capturing it at the moment of decision is much better than reconstructing from logs.
- Mirrors how `git commit --allow-empty -m "reason"` works — bypass requires an utterance.

Audit trail at `data/preflight_bypass.log` is **append-only by convention** — the gate writes to it, no other component reads from it. A Phase 2 dashboard view can render the log. For now, `tail data/preflight_bypass.log` is the operator query.

### 6.4 Output format (Q3)

**Decision: human-readable plaintext by default; `--json` flag for structured output. No mixing.**

The 3am-page operator sees plaintext. Tooling that wants to parse uses `--json`.

#### 6.4.1 Plaintext format

```
Preflight Safety Gate — checking 6 conditions
─────────────────────────────────────────────
[PASS] kill_switch_state      no triggered state                      (3ms)
[PASS] kill_switch_lock       no lock file                            (1ms)
[PASS] equity_history         last row 12h ago (within 24h window)    (18ms)
[PASS] risk_thresholds        max_position_loss_pct=0.05              (0ms)
[PASS] gateway_port           port 4002 listening                     (42ms)
[PASS] zombies                no CLOSE_WAIT connections               (39ms)
─────────────────────────────────────────────
6/6 checks passed. Safe to proceed.
```

Status badge widths (`[PASS]`, `[WARN]`, `[BLOCK]`) are uniform 7 chars including brackets. Check names left-padded to a column.

#### 6.4.2 Plaintext on BLOCK (the 2026-05-22 scenario)

```
Preflight Safety Gate — checking 6 conditions
─────────────────────────────────────────────
[BLOCK] kill_switch_state    triggered=True since 2026-05-22T22:58 ET  (4ms)
[PASS]  kill_switch_lock     no lock file                              (1ms)
[BLOCK] equity_history       last row 38h ago (stale; >24h)            (22ms)
[PASS]  risk_thresholds      max_position_loss_pct=0.05                (0ms)
[PASS]  gateway_port         port 4002 listening                       (44ms)
[PASS]  zombies              no CLOSE_WAIT connections                 (37ms)
─────────────────────────────────────────────
2/6 checks BLOCKED. Cannot proceed.

╔══════════════════════════════════════════════════════════════════════╗
║ BLOCK #1 — kill_switch_state                                         ║
╠══════════════════════════════════════════════════════════════════════╣
║ Kill switch is triggered.                                            ║
║                                                                      ║
║ Trigger reason: "Position loss limit exceeded for AAPL: 3.33% loss"  ║
║ Triggered at:   2026-05-22T22:58:12-04:00 (about 16 hours ago)       ║
║ State file:     /Users/oliver/Projects/robo_trader/data/kill_switch_state.json
║                                                                      ║
║ What to do:                                                          ║
║   1. Decide if the loss-trigger is still relevant.                   ║
║      - If a real loss event: review positions before clearing.       ║
║      - If a stale trip from a transient issue: clear it.             ║
║   2. To clear (DESTRUCTIVE — review first):                          ║
║        cp data/kill_switch_state.json data/kill_switch_state.json.bak.$(date +%Y-%m-%d-%H%M)
║        rm data/kill_switch_state.json data/kill_switch.lock          ║
║   3. Re-run: ./START_TRADER.sh                                       ║
║                                                                      ║
║ To proceed anyway (NOT recommended without step 1):                  ║
║   python3 scripts/preflight_check.py --force "<your reason here>"    ║
║   then re-run ./START_TRADER.sh                                      ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║ BLOCK #2 — equity_history                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║ Last equity_history row is 38 hours old (>24h threshold).            ║
║                                                                      ║
║ Most recent date: 2026-05-22                                         ║
║ Database:         /Users/oliver/Projects/robo_trader/trading_data.db ║
║                                                                      ║
║ This usually means a prior session died without writing an equity    ║
║ snapshot. State files may be inconsistent.                           ║
║                                                                      ║
║ What to do:                                                          ║
║   1. Check for an unclean shutdown:                                  ║
║        tail -100 robo_trader.log | grep -E "(Traceback|ERROR|killed)"║
║   2. Verify positions match IBKR before starting:                    ║
║        python3 scripts/reconcile_positions.py                        ║
║   3. If positions reconcile, you may proceed:                        ║
║        python3 scripts/preflight_check.py --force "verified positions reconcile, ticket #N"
║                                                                      ║
║ This check does NOT auto-clear. Operator must confirm system state.  ║
╚══════════════════════════════════════════════════════════════════════╝
```

Each BLOCK gets its own boxed block. Operator can fix them in any order — preflight is not transactional, just diagnostic.

#### 6.4.3 Plaintext on WARN

```
Preflight Safety Gate — checking 6 conditions
─────────────────────────────────────────────
[PASS] kill_switch_state    no triggered state                       (3ms)
[PASS] kill_switch_lock     no lock file                             (1ms)
[PASS] equity_history       last row 4h ago (within 24h window)      (19ms)
[WARN] risk_thresholds      max_position_loss_pct=0.018 (<0.02)      (0ms)
[PASS] gateway_port         port 4002 listening                      (44ms)
[PASS] zombies              no CLOSE_WAIT connections                (40ms)
─────────────────────────────────────────────
6/6 checks passed (1 with warnings). Safe to proceed.

⚠ WARN — risk_thresholds
  max_position_loss_pct is 0.018 (1.8%), below the recommended 2.0%
  floor. A normal daily move on a volatile stock can trip the kill
  switch at this level (see 2026-05-22 incident with NVDA at -2.93%).
  Consider RISK_MAX_POSITION_LOSS_PCT=0.05 in .env (current default).
  Not blocking — proceeding.
```

WARNs go after the summary line so the operator's eye still lands on "Safe to proceed."

#### 6.4.4 JSON output (`--json` flag)

```json
{
  "version": 1,
  "started_at": "2026-05-23T14:32:01.123-04:00",
  "completed_at": "2026-05-23T14:32:02.847-04:00",
  "duration_ms": 1724,
  "exit_code": 0,
  "summary": {
    "total": 6,
    "passed": 5,
    "warned": 1,
    "blocked": 0
  },
  "checks": [
    {
      "name": "kill_switch_state",
      "status": "PASS",
      "message": "no triggered state",
      "remediation": "",
      "details": {"state_path": "/Users/oliver/Projects/robo_trader/data/kill_switch_state.json", "exists": false},
      "duration_ms": 3
    },
    {
      "name": "risk_thresholds",
      "status": "WARN",
      "message": "max_position_loss_pct=0.018 (<0.02)",
      "remediation": "Consider RISK_MAX_POSITION_LOSS_PCT=0.05 in .env",
      "details": {"value": 0.018, "warn_threshold": 0.02, "block_threshold": 0.01, "env_var": "RISK_MAX_POSITION_LOSS_PCT"},
      "duration_ms": 0
    }
  ]
}
```

JSON output is suitable for piping into Phase 2's `/api/health` endpoint or for chaos-drill assertions.

---

## 7. The Checks

Each check below: rationale, inputs, decision logic, remediation text, and edge cases. The full implementation is sketched for the first check; subsequent checks follow the same pattern at less detail.

### 7.1 KillSwitchStateCheck (G1)

**Why this check exists.** This is the check that would have prevented 2026-05-22. The KillSwitch class loads state from disk on construction; once `triggered=True` is persisted, every new runner sees it and refuses to trade. Without this preflight, that state is invisible at startup.

**Inputs:**
- `data/kill_switch_state.json` (or path via `KILL_SWITCH_STATE_PATH` env var if set in future)

**Decision:**
- File missing → PASS
- File parseable, `triggered=False` → PASS (`details.previous_trigger_reason` populated if present)
- File parseable, `triggered=True` → BLOCK
- File unparseable (corrupt JSON, permission denied) → BLOCK (matches KillSwitch's own fail-closed behavior in `_load_persisted_state`)

**Sketch:**

```python
class KillSwitchStateCheck:
    name = "kill_switch_state"
    description = "Kill switch persisted state"
    timeout_seconds = 1.0

    def run(self, context: PreflightContext) -> CheckResult:
        path = context.project_root / "data" / "kill_switch_state.json"
        if not path.exists():
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message="no triggered state",
                details={"state_path": str(path), "exists": False},
            )
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"state file unreadable ({exc.__class__.__name__})",
                remediation=(
                    "Kill switch state file is corrupt. Fail-closed policy "
                    "blocks startup.\n"
                    f"  Inspect:   cat {path}\n"
                    f"  Backup:    cp {path} {path}.bak.$(date +%Y-%m-%d-%H%M)\n"
                    f"  Clear:     rm {path}\n"
                    "  Then re-run ./START_TRADER.sh"
                ),
                details={"state_path": str(path), "error": str(exc)},
            )

        if not payload.get("triggered"):
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message="no triggered state",
                details={
                    "state_path": str(path),
                    "exists": True,
                    "triggered": False,
                },
            )

        reason = payload.get("trigger_reason", "unknown")
        trigger_time = payload.get("trigger_time", "unknown")
        return CheckResult(
            name=self.name,
            status=CheckStatus.BLOCK,
            message=f'triggered=True since {trigger_time}',
            remediation=self._build_remediation(path, reason, trigger_time),
            details={
                "state_path": str(path),
                "trigger_reason": reason,
                "trigger_time": trigger_time,
            },
        )

    def _build_remediation(self, path, reason, trigger_time) -> str:
        return (
            f'Kill switch is triggered.\n\n'
            f'Trigger reason: "{reason}"\n'
            f'Triggered at:   {trigger_time}\n'
            f'State file:     {path}\n\n'
            'What to do:\n'
            '  1. Decide if the loss-trigger is still relevant.\n'
            '     - If a real loss event: review positions before clearing.\n'
            '     - If a stale trip from a transient issue: clear it.\n'
            '  2. To clear (DESTRUCTIVE — review first):\n'
            f'       cp {path} {path}.bak.$(date +%Y-%m-%d-%H%M)\n'
            f'       rm {path} data/kill_switch.lock\n'
            '  3. Re-run: ./START_TRADER.sh\n\n'
            'To proceed anyway (NOT recommended without step 1):\n'
            '  python3 scripts/preflight_check.py --force "<your reason here>"\n'
            '  then re-run ./START_TRADER.sh'
        )
```

**Test fixtures:**
- `tmp_path / "data" / "kill_switch_state.json"` containing each of: triggered, untriggered, missing, malformed, permission-denied.
- A `PreflightContext` factory that sets `project_root = tmp_path`.

**Edge cases:**
- `trigger_time` may be missing for old state files written before TC-M5 — display "unknown" not crash.
- `trigger_reason` may contain quotes — escape for display in the remediation text.
- Symlink at the path: follow it (matches KillSwitch behavior).

### 7.2 KillSwitchLockCheck (G1)

**Why this check exists.** The plan calls out `data/kill_switch.lock` as the deny-by-default fail-closed signal. The KillSwitch class's `_load_persisted_state` already treats the lock file's presence as triggered (`R2-M2`), but the lock can exist independently — for example, if an operator created it manually to force a stop. Preflight surfaces it explicitly rather than letting the runner spin up and immediately die.

**Inputs:** `data/kill_switch.lock`

**Decision:**
- File missing → PASS
- File exists → BLOCK

**Remediation:** "Kill switch lock file present. This is a deny-by-default safety signal. Either:
- Remove the lock: `rm data/kill_switch.lock` (after confirming nothing intentionally placed it)
- Or proceed with `--force "reason"`"

**Edge case:** The lock and the state file are *related but independent*. Don't dedupe — show both BLOCKs separately if both fire, because the remediation actions differ.

### 7.3 EquityHistoryFreshnessCheck (G2)

**Why this check exists.** Equity snapshots are written to `equity_history` at the end of every successful trading day. A row older than 24h means either (a) the system hasn't completed a full day in over a day (unclean shutdown candidate) or (b) the database itself is stale/wrong. Either case warrants an operator look before resuming trading. A stale state usually means stale positions.

**Inputs:**
- `trading_data.db` SQLite file
- Query: `SELECT MAX(date), MAX(timestamp) FROM equity_history` (across all portfolios)
- Current time (system clock)

**Decision:**
- DB file missing → BLOCK (system unconfigured)
- Empty `equity_history` table → WARN (first-run; not a livelock indicator)
- Max(timestamp) within last 24h → PASS
- Max(timestamp) older than 24h → BLOCK
- Sqlite read error → BLOCK (consistent fail-closed)

**Important nuance — weekends and holidays.** The market is closed Sat/Sun and on holidays. A Monday morning startup could legitimately see a "Friday" last-row that is 60+ hours old.

**Resolution:** Use a *trading-day* delta, not wall-clock delta. The check measures "how many trading days since last row." Threshold: more than 1 trading day → BLOCK.

```python
# Pseudocode
from robo_trader.utils.market_hours import MarketHours, get_market_time

trading_days_elapsed = MarketHours.count_trading_days(
    start=max_timestamp,
    end=get_market_time(),
)
if trading_days_elapsed > 1:
    return BLOCK
```

This handles Monday-after-Friday (0–1 trading days) correctly, AND catches "haven't run since last Tuesday" (3+ trading days, real problem).

**Remediation text:** "Last equity row is N trading days old. This usually means a prior session died without writing a snapshot. Verify positions match IBKR (`scripts/reconcile_positions.py`) before resuming, or `--force` if you've already confirmed."

**Edge case:** Multi-portfolio. `equity_history` is partitioned by `portfolio_id`. We check the max across all portfolios — if any portfolio is fresh, we PASS. (A portfolio that hasn't traded in weeks because it's disabled shouldn't block startup.)

**Edge case 2:** Test environment / CI. Skip via `PREFLIGHT_SKIP_CHECKS` env var? No — better to fail loudly and force CI to set up a fixture. The CheckResult's `details.skip_reason` field is reserved for future use but not exposed by current checks.

### 7.4 RiskThresholdCheck (G3)

**Why this check exists.** The 2026-05-22 incident's root cause was `max_position_loss_pct=0.02`. The audit raised the default to 0.05 (commit `baadd26`), but if a stale `.env` overrides it back to 0.02 or lower, the trader will trip again. This check surfaces the value at startup.

**Inputs:**
- Env var `RISK_MAX_POSITION_LOSS_PCT` (resolved through `.env` + shell environment)
- Compare against thresholds 0.02 (warn) and 0.01 (block)

**Decision:**
- Not set → PASS (use code default, which is 0.05 per `baadd26`)
- Set but unparseable as float → BLOCK ("config corrupt — preflight cannot evaluate")
- `< 0.01` → BLOCK
- `< 0.02` → WARN
- `>= 0.02` → PASS

**Remediation (BLOCK):** "RISK_MAX_POSITION_LOSS_PCT=N is below 1%. Almost any intraday move will trip the kill switch. Either remove the line from `.env` to use the default 0.05, or set to a realistic value (≥0.02). To proceed with this value anyway, `--force`."

**Remediation (WARN):** "RISK_MAX_POSITION_LOSS_PCT=N is below the 2% recommended floor. A normal daily move on a volatile stock can trip the kill switch at this level (see 2026-05-22 NVDA incident). Consider raising. Not blocking."

**Extensibility note:** Phase 2 may add similar checks for `RISK_MAX_DAILY_LOSS_PCT`, `RISK_MAX_DRAWDOWN_PCT`, `MAX_POSITION_PCT`. Each is a separate check class so each fails independently.

### 7.5 GatewayPortListeningCheck (G4)

**Why this check exists.** START_TRADER.sh tries to bring up Gateway and waits for `lsof -sTCP:LISTEN` on port 4002. If Gateway never came up successfully (2FA failure, license issue, IBC misconfig), START_TRADER fails its own check and exits before preflight runs. **But:** the watchdog can invoke runner_async directly in restart mode, bypassing START_TRADER. In that path, preflight is the first line of defense.

**Inputs:**
- `lsof -nP -iTCP:<port> -sTCP:LISTEN` (port from `EXECUTION_MODE`: `4002` for paper, `4001` for live, default 4002)
- Subprocess timeout: 3 seconds

**Decision:**
- lsof returns 0 with output → PASS
- lsof returns non-zero (no LISTEN found) → BLOCK
- lsof times out → BLOCK ("port check timed out — system likely overloaded")
- lsof binary missing → BLOCK ("lsof not installed — required for safety checks"; should never happen on macOS, but worth catching)

**Remediation:** "Gateway API port 4002 is not listening. Run `./scripts/start_gateway.sh` or `python3 scripts/gateway_manager.py start --paper` to bring it up. If 2FA is pending, check your IBKR Mobile app."

**Critical constraint (CLAUDE.md, 2025-12-06 row):** Use `lsof`, never `socket.connect_ex`. The latter creates a zombie connection that blocks subsequent API handshakes — exactly the failure class we're checking *for*. The implementation MUST go through `subprocess.run(["lsof", ...])` — there is NO Python-API alternative that is safe.

### 7.6 ZombieConnectionsCheck (G5)

**Why this check exists.** A `CLOSE_WAIT` connection on the Gateway port will block the runner's `ib.connect()` with cryptic timeouts. START_TRADER.sh already does a zombie sweep but preflight verifies it took.

**Inputs:**
- `lsof -nP -iTCP:<port> -sTCP:CLOSE_WAIT` (same port as 7.5)
- Subprocess timeout: 3 seconds

**Decision:**
- Output empty → PASS
- Any line in output → BLOCK with the zombie count and PIDs

**Remediation:** "Found N zombie connections on port <port>. These will block API handshakes. Either:
- Kill Python-owned zombies: `python3 scripts/gateway_manager.py clear-zombies`
- For Gateway-owned zombies: restart Gateway with `python3 scripts/gateway_manager.py restart`
- Then re-run ./START_TRADER.sh"

**Edge case:** Race with START_TRADER.sh's own zombie cleanup. Preflight runs after step 3 of the shell script, so this should not occur, but if it does the message is still actionable.

---

## 8. The H1 coordination question (Q7)

**Recap.** H1 (commit `315ea6a`) is the runner's in-flight kill-switch auto-reset: when `recover_connection` succeeds, if the kill switch was tripped *and* its `trigger_reason` contains connection-related keywords (`connection`, `handshake`, `gateway`, `timeout`, `subprocess`, `ibkr`), the runner force-resets it. The justification is "the proximate cause is resolved; the trigger was incidental." Loss-based triggers are explicitly preserved.

**The question.** Preflight runs BEFORE recover_connection has a chance to fire (the runner isn't up yet). If preflight blocks on a triggered kill switch, the operator might be in the position of "yeah, this is just the same connection-related trip from last night; the runner would have auto-cleared this in its first health cycle anyway." Should preflight apply the same heuristic?

**Decision: NO. Preflight always blocks on a triggered kill switch, even if the reason looks connection-related. The operator must explicitly clear or `--force`.**

**Justification (the trade-off).**

**Arguments for auto-clearing in preflight (rejected):**
1. Symmetric with H1 — operator already opted into "connection trips can self-heal."
2. Saves an operator wakeup when the trigger was a known-transient.
3. The 2026-05-22 incident itself would have been blocked correctly by preflight under either policy, because that trigger was loss-based, not connection-based.

**Arguments against auto-clearing in preflight (winning):**
1. **Different scope.** H1 runs *after* `recover_connection` actually succeeded — there is direct evidence the connection issue is resolved. Preflight runs before any connection attempt. "Looks connection-related" is a *guess* without that recovery evidence. Auto-clearing here would clear triggers for problems we haven't actually fixed yet.
2. **Reason-string fragility.** The keyword list (`connection`, `handshake`, etc.) is a heuristic that's fine when paired with explicit recovery success. As a standalone preflight rule, a future kill-switch trigger reason like `"Could not connect to historical data feed"` would auto-clear a real data-quality trip.
3. **The bypass exists.** `--force "transient connection trip from yesterday, ticket #1234"` is the supported escape hatch. It is one command, fast at 3am, and audited.
4. **Defense in depth.** H1 and preflight are different layers. H1 handles "we were running, hit a blip, recovered." Preflight handles "we are NOT running, why?" Conflating them weakens both.
5. **Plan intent.** The motivating plan explicitly frames preflight as *adding* a gate, not as *auto-fixing* state. "The persisted state stays the same — the response changes." (Plan §MVP-1.)

**Implication for operator UX.** After a connection-related trip:
- H1 *might* have auto-cleared it before the runner finally exited. If so, preflight passes — no issue.
- If H1 didn't fire (because the runner died too fast, or the trigger was loss-based, or the recovery itself failed), preflight blocks. Operator sees the trigger reason and the remediation. Worst case: 30-second `rm` + bypass. The 2026-05-22 4-hour loss is the failure mode we're optimizing against, not 30 seconds of operator decision-making.

**Future revisit trigger.** If post-deploy logs show frequent `event=preflight_bypass` entries with reason strings citing "transient connection," that's evidence the heuristic-symmetry argument has merit and we should reconsider. Until then: simpler is safer.

---

## 9. Testing Strategy (G9)

### 9.1 Unit tests — per check

Pattern: each check class has a corresponding test class in `tests/test_preflight_checks.py`. All checks accept a `PreflightContext` with `project_root=tmp_path`, so no real filesystem state is touched.

#### KillSwitchStateCheck
- `test_returns_pass_when_state_file_missing`
- `test_returns_pass_when_triggered_false`
- `test_returns_block_when_triggered_true`
- `test_returns_block_when_state_file_corrupt_json`
- `test_returns_block_when_state_file_unreadable_permissions`
- `test_remediation_text_includes_state_path_and_reason`
- `test_handles_missing_trigger_time_field` (regression: old state files)
- `test_does_not_modify_state_file` (verify file mtime unchanged after run)

#### KillSwitchLockCheck
- `test_returns_pass_when_lock_missing`
- `test_returns_block_when_lock_present`
- `test_does_not_remove_lock_file`

#### EquityHistoryFreshnessCheck
- `test_returns_block_when_db_missing`
- `test_returns_warn_when_table_empty`
- `test_returns_pass_when_max_row_within_24h`
- `test_returns_block_when_max_row_older_than_one_trading_day`
- `test_returns_pass_on_monday_morning_with_friday_max` (weekend gap)
- `test_returns_pass_on_post_holiday_morning` (holiday gap)
- `test_returns_block_when_three_trading_days_old`
- `test_aggregates_max_across_portfolios`
- `test_returns_block_on_sqlite_read_error`

#### RiskThresholdCheck
- `test_returns_pass_when_env_var_unset`
- `test_returns_pass_when_value_above_warn`
- `test_returns_warn_when_value_between_warn_and_block`
- `test_returns_block_when_value_below_block_threshold`
- `test_returns_block_when_value_unparseable`
- `test_returns_block_when_value_zero_or_negative`
- `test_remediation_names_env_var_and_recommended_value`

#### GatewayPortListeningCheck
- `test_returns_pass_when_lsof_finds_listener` (mock subprocess.run)
- `test_returns_block_when_lsof_finds_no_listener`
- `test_returns_block_when_lsof_times_out`
- `test_returns_block_when_lsof_binary_missing`
- `test_uses_paper_port_by_default`
- `test_uses_live_port_when_execution_mode_live`
- `test_does_not_use_socket_connect_ex` (static check; greps source for forbidden API)

#### ZombieConnectionsCheck
- `test_returns_pass_when_no_close_wait_lines`
- `test_returns_block_when_close_wait_lines_present`
- `test_includes_pid_list_in_details`

### 9.2 Integration tests — runner

`tests/test_preflight_runner.py`:

- `test_all_checks_pass_returns_exit_zero`
- `test_any_block_returns_exit_one`
- `test_warns_do_not_block`
- `test_force_with_blocks_returns_exit_two`
- `test_force_without_blocks_returns_exit_zero_with_warning`
- `test_force_rejects_empty_reason`
- `test_force_rejects_placeholder_reason`
- `test_force_logs_to_bypass_log_file`
- `test_runner_logs_event_preflight_bypass_to_main_log`
- `test_checks_run_in_parallel` (verify total duration < sum of individual)
- `test_check_timeout_returns_block` (mock a slow check)
- `test_check_exception_returns_block` (mock a raising check)
- `test_json_output_well_formed`
- `test_json_output_contains_all_fields`
- `test_plaintext_output_format_stable` (golden output comparison)

### 9.3 Required fixtures and helpers

| Fixture | Purpose |
|---|---|
| `tmp_path` (pytest built-in) | Sandbox for state files |
| `PreflightContext.for_test(tmp_path)` factory | Inject project_root, port, env dict |
| `lsof_subprocess_mock` (monkeypatch on `subprocess.run`) | Simulate `lsof` output for the two port checks |
| `sqlite_fixture(tmp_path, snapshots)` | Populate trading_data.db with N equity_history rows at given timestamps |
| `kill_switch_state_fixture(tmp_path, **payload)` | Write a state file with given fields |
| `frozen_market_time(monkeypatch, dt)` | Override `get_market_time` so trading-day math is deterministic |
| `bypass_log_isolated(tmp_path)` | Redirect `data/preflight_bypass.log` writes into tmp_path |

### 9.4 Out-of-scope tests

| Skipped | Why |
|---|---|
| Real `lsof` calls in CI | Non-portable; subprocess mock is sufficient |
| Real Gateway port checking | Requires running Gateway; manual verification only |
| Real `equity_history` with year of data | Tested with synthetic 10-row fixtures; large-table perf tested via `--timeout` |
| Concurrency stress (1000 parallel preflights) | Single-process by design; not a use case |

### 9.5 Regression: must stay green

- All existing `tests/security/` tests (per CLAUDE.md)
- All existing `tests/test_connection_health.py` (this branch's prior work)
- All existing `tests/test_recover_connection.py` (this branch's prior work)

Preflight is a new entry point — no existing tests cross-cut into it. If any unrelated test breaks during implementation, root-cause and don't paper over.

### 9.6 Manual canary

Before merging:
1. Trigger preflight against a clean state — expect `[PASS]` summary, exit 0, latency <2s.
2. Manually `touch data/kill_switch.lock` — re-run, expect BLOCK on `kill_switch_lock`, exit 1.
3. Restore `data/kill_switch_state.json.bak.2026-05-22-1627` to `data/kill_switch_state.json` — re-run, expect BLOCK on `kill_switch_state` with the actual trigger reason from the incident.
4. With both BLOCKs present, run with `--force "manual canary test"` — expect exit 2, log entry in `data/preflight_bypass.log`.
5. Remove the staged state, set `RISK_MAX_POSITION_LOSS_PCT=0.005` in shell env, re-run — expect BLOCK on threshold.
6. Set `RISK_MAX_POSITION_LOSS_PCT=0.018` — expect WARN, exit 0.
7. `pkill -f "IB Gateway"` — re-run, expect BLOCK on `gateway_port`.

Document outcomes in the implementation PR.

---

## 10. Performance Budget (G8)

**Target:** preflight wall-clock latency ≤ 5 seconds on a healthy system. 95th percentile (degraded I/O, slow `lsof`) ≤ 10 seconds.

**Per-check budgets:**

| Check | Soft timeout | Typical | Worst case |
|---|---|---|---|
| KillSwitchStateCheck | 1.0s | <5ms | Disk-full triggers timeout |
| KillSwitchLockCheck | 0.5s | <2ms | NFS-mounted path triggers timeout |
| EquityHistoryFreshnessCheck | 3.0s | <50ms | Sqlite WAL recovery on large DB |
| RiskThresholdCheck | 0.1s | <1ms | None plausible |
| GatewayPortListeningCheck | 3.0s | <100ms | `lsof` hang under load |
| ZombieConnectionsCheck | 3.0s | <100ms | Same as above |

**Total budget assuming parallelism:** ~3s (longest of the bunch). Sequential fallback: ~10.6s. Hence parallelism is required to meet G8.

**Threading model:** `concurrent.futures.ThreadPoolExecutor(max_workers=6)`. Each check submitted as a future with its `timeout_seconds` enforced via `future.result(timeout=...)`. Checks have no shared mutable state — pure functions over read-only context — so thread safety is structurally guaranteed.

**Subprocess timeout enforcement:** `subprocess.run(..., timeout=N)` raises `TimeoutExpired`; the check catches and returns BLOCK.

**Why not asyncio?** Could do it. Threading is simpler, the checks are I/O-bound, and there's no event loop to share with the runner (preflight is a separate process). Six threads vs. an event loop is the lower-overhead choice for a once-per-startup tool.

---

## 11. Open Questions

These need human input before implementation:

### Q11.1 — `equity_history` for first-run / newly-created portfolios
A brand-new portfolio (Phase 2 dynamic portfolio creation, hypothetical) would have zero `equity_history` rows. The check returns WARN under current spec, which means the operator sees the WARN line and proceeds. Is that the right default, or should "I am intentionally bootstrapping" need a `--force`? **Proposal:** WARN is fine; a bootstrap is rare enough that visibility is sufficient. **Need confirmation.**

### Q11.2 — `EXECUTION_MODE=live` port detection
The port check uses 4001 for live and 4002 for paper. The current codebase defaults to paper everywhere; the live path is rarely exercised. Worth a sanity test that a `live`-set environment correctly selects port 4001? **Proposal:** Yes, add unit test. **No spec change needed.**

### Q11.3 — bypass log retention / rotation
`data/preflight_bypass.log` will grow forever. Acceptable for now (entries are small JSON lines, write rate is low) but should rotate eventually. **Proposal:** Defer to Phase 2 (could roll into observability work P2-4). **Flagged.**

### Q11.4 — Should preflight also write a "last successful preflight" timestamp?
Phase 2 health endpoint could expose "last preflight passed at T." Cheap to add. **Proposal:** Defer; not in MVP-1 scope, but trivial future addition. **Flagged.**

### Q11.5 — How to handle multi-portfolio risk threshold overrides?
The plan check is on the *global* `RISK_MAX_POSITION_LOSS_PCT`. Multi-portfolio configs can override `max_position_pct` per-portfolio. If a portfolio overrides to a dangerous value, we don't currently catch it. **Proposal:** out of scope for MVP-1; multi-portfolio risk validation is a separate larger work item. **Acknowledged gap.**

### Q11.6 — Coexistence with the watchdog
If the watchdog auto-restarts the trader, it currently calls START_TRADER.sh, so preflight runs. But the launchd plist might evolve to invoke `runner_async` directly. **Proposal:** Keep preflight invocation in START_TRADER.sh AND add it as a recommended invocation in the watchdog plist as a follow-up. **Flagged for Phase 2.**

---

## 12. Sequenced Implementation Tasks

Each task is small enough for a single commit + atomic review. Strict TDD: write the test first.

1. **Skeleton + CheckResult dataclass.**
   - Add `robo_trader/preflight/__init__.py`, `result.py` with `CheckStatus` and `CheckResult`.
   - Test: instantiation, frozen-ness, JSON serializability.
   - Commit: `feat(preflight): scaffold preflight package with CheckResult`

2. **Check protocol + PreflightContext.**
   - `robo_trader/preflight/checks.py` with the `Check` Protocol and `PreflightContext` dataclass.
   - Test: nothing yet, but next commits build on it.
   - Commit: `feat(preflight): add Check protocol and PreflightContext`

3. **KillSwitchStateCheck.**
   - Write tests (8 cases above).
   - Implement.
   - Commit: `feat(preflight): kill-switch state check`

4. **KillSwitchLockCheck.**
   - Tests + impl.
   - Commit: `feat(preflight): kill-switch lock check`

5. **EquityHistoryFreshnessCheck.**
   - Tests including weekend/holiday cases.
   - Impl, with MarketHours.count_trading_days helper if it doesn't exist yet.
   - Commit: `feat(preflight): equity history freshness check`

6. **RiskThresholdCheck.**
   - Tests + impl.
   - Commit: `feat(preflight): risk threshold check`

7. **GatewayPortListeningCheck.**
   - Tests with `subprocess.run` mocking.
   - Impl using `lsof -nP -iTCP:<port> -sTCP:LISTEN`.
   - Commit: `feat(preflight): gateway port listening check`

8. **ZombieConnectionsCheck.**
   - Tests + impl.
   - Commit: `feat(preflight): zombie connections check`

9. **Runner orchestration (parallel execution, output formatting).**
   - `robo_trader/preflight/runner.py` with `run_all_checks(context) -> RunReport`.
   - Tests for parallelism, timeout, exception isolation, output formats.
   - Commit: `feat(preflight): parallel runner with plaintext + JSON output`

10. **CLI script + bypass mechanism.**
    - `scripts/preflight_check.py` with argparse, `--json`, `--force "reason"`, `--verbose`.
    - Bypass log writer (`data/preflight_bypass.log`).
    - Exit code logic (0/1/2/3).
    - Tests for CLI integration.
    - Commit: `feat(preflight): CLI entrypoint with auditable bypass`

11. **START_TRADER.sh integration.**
    - Add Step 4.5 block after Python env, before dashboard.
    - Manual canary per §9.6.
    - Commit: `feat(start-trader): wire preflight safety gate before runner launch`

12. **CLAUDE.md entry.**
    - Add row to Common Mistakes table:
      > "Skipping the preflight safety gate when restarting after an incident | Always run START_TRADER.sh (preflight runs automatically). If it blocks, fix the condition or use `--force "<reason>"`. The gate exists because of the 2026-05-22 4-hour silent livelock. | 2026-05-23"
    - Commit: `docs(claude-md): document preflight safety gate behavior`

13. **Final integration test + canary.**
    - End-to-end smoke: trigger each BLOCK condition in turn, verify message and exit code.
    - Commit: none (canary results recorded in PR description).

Total estimated effort: ~half a day, matching the plan's estimate.

---

## 13. References

- `docs/superpowers/plans/2026-05-23-robust-startup-safety-plan.md` — the motivating plan
- `docs/superpowers/specs/2026-05-16-persistent-ibkr-connection-design.md` — design spec style precedent
- `robo_trader/risk/advanced_risk.py:285-619` — KillSwitch class (state, lock, persistence)
- `robo_trader/runner_async.py:4240-4280` — H1 auto-reset heuristic (do NOT mirror in preflight; see §8)
- `START_TRADER.sh` — entry point; preflight slots in at the marked location
- `scripts/gateway_manager.py` — sibling operational tool; CLI shape precedent
- `data/kill_switch_state.json.bak.2026-05-22-1627` — actual triggered state from the incident, useful as test fixture
- `CLAUDE.md` "🚨 CRITICAL: NEVER DELETE USER DATA" — informs N1 (preflight does not modify state)
- `CLAUDE.md` row 2025-12-06 (use lsof, not socket.connect_ex) — informs 7.5 and 7.6
- `CLAUDE.md` row 2026-05-12 (load watchdog) — adjacent operational concern, referenced in Q11.6
