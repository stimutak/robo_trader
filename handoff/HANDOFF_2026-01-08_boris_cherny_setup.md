# Handoff: Boris Cherny-Style Claude Environment & P&L Fix

**Date:** 2026-01-08
**Session Focus:** Implemented Boris Cherny's efficient coding workflow + Fixed critical P&L bug

---

## Summary

This session implemented Boris Cherny's (Claude Code creator) workflow methodology and fixed a critical P&L calculation bug in the dashboard.

---

## 1. Boris Cherny-Style Environment Setup

### Slash Commands Created (`.claude/commands/`)

| Command | File | Purpose |
|---------|------|---------|
| `/review` | `review.md` | Multi-subagent code review (4 reviewers + 2 verifiers) |
| `/test-and-commit` | `test-and-commit.md` | Run tests, fix if failing, then commit |
| `/verify-trading` | `verify-trading.md` | Check Gateway, zombies, risk params, logs |
| `/pr` | `pr.md` | Full PR workflow with tests and linting |
| `/commit` | `commit.md` | Quick commit with proper format |
| `/code-simplifier` | `code-simplifier.md` | Review and simplify recent code |
| `/oncall-debug` | `oncall-debug.md` | Debug production issues systematically |

### Permissions Pre-Allowed (`.claude/settings.local.json`)

```json
"allow": [
  "Bash(python3 -m pytest:*)",
  "Bash(python3 -m black:*)",
  "Bash(python3 -m isort:*)",
  "Bash(python3 -m flake8:*)",
  "Bash(python3 scripts/gateway_manager.py:*)",
  "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)",
  "Bash(gh pr:*)", "Bash(gh issue:*)",
  ...
]
```

### PostToolUse Hook

Auto-formats Python files with `black` and `isort` after every Edit/Write operation.

### MCP Integration (`.mcp.json`)

- Filesystem server configured
- Memory server for persistent context
- Future integrations documented (Slack, Postgres, Prometheus)

### CLAUDE.md Enhancements

Added new sections:
- **Common Mistakes** - Auto-updated tables when errors found
- **Verification Checklist** - Pre-PR checklist
- **Slash Commands** - Reference table
- **Two-Phase Development Loop** - Plan mode → Execution

### Pre-Commit Hooks Updated (`.pre-commit-config.yaml`)

Added:
- **mypy** v1.8.0 - Type checking
- **bandit** v1.7.7 - Security scanning

### Documentation

- `docs/PARALLEL_CLAUDE_SETUP.md` - Guide for running 5+ Claude instances

---

## 2. Critical P&L Bug Fix

### Problem

`app.py:3750` assumed **1% profit on ALL sell trades** - completely wrong calculation:

```python
# OLD CODE - WRONG
if trade.get("side") == "SELL":
    profit = trade_value * 0.01  # Assumed 1% profit!
```

### Solution

Replaced with proper **FIFO cost basis tracking** using Decimal precision:

```python
# NEW CODE - Correct
position_tracker: Dict[str, Dict] = {}
for trade in sorted_trades:  # Chronological order
    if side == "BUY":
        # Update weighted average cost
        pos["avg_cost"] = total_cost / pos["quantity"]
    elif side == "SELL":
        # Calculate ACTUAL profit
        profit = (price - pos["avg_cost"]) * sell_qty
```

### Files Changed

| File | Change |
|------|--------|
| `app.py:16` | Added `from decimal import Decimal` |
| `app.py:19` | Added `from typing import Dict` |
| `app.py:3731-3739` | Unrealized P&L now uses Decimal |
| `app.py:3744-3793` | Replaced fake 1% with FIFO cost basis |
| `app.py:3798-3806` | Convert Decimal to float for JSON |

---

## 3. Code Review Findings (from `/review` command)

Ran multi-subagent code review with 4 reviewers + 2 verifiers:

### Critical (Fixed)
- P&L assumes 1% profit - **FIXED**
- Float arithmetic for P&L - **FIXED**

### High (Open)
- Missing 6 market holidays in `market_hours.py`
- `socket.connect_ex()` in `start_ai_trading.py`

### Medium (Open)
- No CSRF protection on `/api/start` and `/api/stop`
- SHA256 without salt for password hashing

---

## 4. System Status

At session end:
- Gateway: RUNNING (port 4002)
- Runner: RUNNING (PID 3354)
- Dashboard: RUNNING (PID 3355)
- Positions: 30/30 (at limit)
- Buy signals being rejected due to position limit

---

## 5. Next Steps

1. **Increase position limit** if needed: `RISK_MAX_OPEN_POSITIONS=40` in `.env`
2. **Fix market holidays** - Add MLK, Presidents, Memorial, Labor, Thanksgiving, Good Friday
3. **Add CSRF protection** to `/api/start` and `/api/stop` endpoints
4. **Test slash commands** - Run `/review`, `/verify-trading` etc.

---

## Files Created/Modified

### Created
- `.claude/commands/review.md`
- `.claude/commands/test-and-commit.md`
- `.claude/commands/verify-trading.md`
- `.claude/commands/pr.md`
- `.claude/commands/commit.md`
- `.claude/commands/code-simplifier.md`
- `.claude/commands/oncall-debug.md`
- `.claude/settings.local.json`
- `.mcp.json`
- `docs/PARALLEL_CLAUDE_SETUP.md`
- `handoff/HANDOFF_2026-01-08_boris_cherny_setup.md`

### Modified
- `app.py` - P&L calculation fix + Decimal imports
- `CLAUDE.md` - Added mistake tracking, slash commands, verification checklist
- `.pre-commit-config.yaml` - Added mypy and bandit
