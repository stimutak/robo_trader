# Multi-Agent Workflow Guide for RoboTrader

Based on [Boris Cherny's workflow](https://paddo.dev/blog/how-boris-uses-claude-code/) (creator of Claude Code) and adapted for algorithmic trading development.

---

## Quick Start (5 minutes)

### 1. Use Plan Mode First

Before any non-trivial task, plan first:

```
Shift+Tab twice → Plan Mode
```

Go back and forth refining the plan until you like it. Then switch to auto-accept mode and Claude can usually one-shot it.

> "A good plan is really important!" — Boris Cherny

### 2. Give Claude a Verification Loop

The single most important practice:

```
Write Code → Verify → Fix → Re-verify → Done
```

Always give Claude a way to test its work. This 2-3x the quality of results.

### 3. Use Slash Commands

Available commands (type `/` to see):

| Command | What It Does |
|---------|--------------|
| `/verify` | Run verification loop |
| `/review` | Multi-agent code review (6 subagents) |
| `/two-phase-review` | Review + challenge false positives |
| `/code-simplifier` | Simplify code after implementation |
| `/test-and-commit` | Verify in parallel, then commit |
| `/commit` | Quick commit with proper format |
| `/pr` | Full commit/push/PR workflow |
| `/verify-trading` | Check Gateway, zombies, risk params |
| `/oncall-debug` | Systematic production debugging |
| `/retrospective` | Extract learnings to CLAUDE.md |
| `/shared-knowledge` | Update CLAUDE.md with learnings |

### 4. Update CLAUDE.md When Claude Errs

When Claude makes a mistake:
1. Fix the immediate issue
2. Add a rule to CLAUDE.md's "Common Mistakes" table
3. Commit it

This makes all future sessions smarter.

---

## Core Concepts

### The Boris Cherny Philosophy

1. **Parallel > Sequential** — Run 5+ Claudes simultaneously
2. **Specialization > Generalization** — Each agent focuses on one thing
3. **Verification is Critical** — Always provide a feedback loop
4. **Two-Phase Loop** — Initial review + challenger filters false positives
5. **Shared Knowledge** — CLAUDE.md evolves with learnings

### Model Choice

Use **Opus 4.5 with thinking** for complex tasks:

> "Even though it's bigger & slower than Sonnet, since you have to steer it less and it's better at tool use, it is almost always faster than using a smaller model in the end."

### Directory Structure

```
.claude/
├── agents/              # Specialized subagent prompts
│   ├── bug-finder.md
│   ├── code-reviewer.md
│   ├── parallel-coordinator.md
│   ├── planner.md
│   ├── style-checker.md
│   ├── trading-validator.md
│   ├── verification-challenger.md
│   └── verifier.md
├── commands/            # Slash command definitions
│   ├── code-simplifier.md
│   ├── commit.md
│   ├── oncall-debug.md
│   ├── pr.md
│   ├── retrospective.md
│   ├── review.md
│   ├── shared-knowledge.md
│   ├── test-and-commit.md
│   ├── two-phase-review.md
│   ├── verify.md
│   └── verify-trading.md
├── skills/              # Skills (commands with scripts/templates)
│   └── verify-trading/
│       └── SKILL.md
└── settings.local.json  # Local permissions
```

---

## RoboTrader-Specific Workflows

### New Feature Workflow

```
/plan                  # Create implementation plan
  ↓
Implement              # Auto-accept mode after plan approval
  ↓
/verify                # Run verification loop
  ↓
/code-simplifier       # Remove unnecessary complexity
  ↓
/two-phase-review      # Full review with challenge phase
  ↓
/test-and-commit       # Run tests, then commit
```

### Bug Fix Workflow

```
Investigate → Fix      # Direct fix
  ↓
/verify                # Confirm fix works
  ↓
/review                # Quick review
  ↓
/test-and-commit       # Ship it
```

### Trading System Check Workflow

```
/verify-trading        # Check Gateway, zombies, risk params
  ↓
./START_TRADER.sh      # Restart if needed
  ↓
tail -f robo_trader.log  # Monitor startup
```

**Note:** The watchdog service auto-restarts the trader if stalled for 5+ minutes during market hours. Check `watchdog.log` for restart history. See CLAUDE.md for watchdog management commands.

### Production Debug Workflow

```
/oncall-debug          # Systematic production debugging
  ↓
Check logs             # tail -f robo_trader.log
  ↓
/verify-trading        # Verify system health
  ↓
Fix issue
  ↓
/test-and-commit       # Ship fix
```

### Code Quality Workflow

```
/review                # Comprehensive multi-agent review
  ↓
Fix confirmed issues   # Only real issues after challenge
  ↓
/shared-knowledge      # Document learnings in CLAUDE.md
```

---

## The Two-Phase Review Loop

Boris Cherny's signature technique. Filters 20-40% false positives.

### Phase 1: Fan-Out (Parallel)

Launch 6 agents simultaneously:
```
├── Bug Hunter         → Type mismatches, null checks, async issues
├── Trading Logic      → Risk calcs, position sizing, market hours
├── Style Checker      → CLAUDE.md compliance, conventions
├── Security Auditor   → SQL injection, input validation, secrets
├── False Positive     → Filter non-issues
└── Priority Ranker    → Categorize CRITICAL/HIGH/MEDIUM/LOW
```

### Phase 2: Challenge (Filter)

For EACH finding, the challenger asks:
1. Is this actually a problem?
2. Can I reproduce it realistically?
3. Is there defensive code elsewhere?
4. Is the severity correct?
5. Would the fix introduce new issues?

### Result

Only confirmed, actionable issues remain.

---

## Verification Patterns for Trading

### RoboTrader Verification Methods

| Domain | Verification Method |
|--------|---------------------|
| Unit tests | `venv/bin/python3 -m pytest tests/` |
| Type checking | `venv/bin/python3 -m mypy robo_trader/` |
| Linting | `venv/bin/python3 -m flake8 robo_trader/` |
| Format | `venv/bin/python3 -m black --check .` |
| Trading system | `/verify-trading` |
| Gateway | `lsof -nP -iTCP:4002 -sTCP:LISTEN` |
| Zombies | `lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT` |
| Dashboard | http://localhost:5001 |

### Database Verification

```bash
# Check positions
sqlite3 trading_data.db "SELECT symbol, quantity, avg_cost FROM positions WHERE quantity > 0"

# Check recent trades
sqlite3 trading_data.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10"

# Check equity
sqlite3 trading_data.db "SELECT * FROM equity_history ORDER BY date DESC LIMIT 5"
```

---

## Subagents Reference

### Available Agents

| Agent | When to Use |
|-------|-------------|
| `bug-finder` | Finding bugs, edge cases, race conditions |
| `code-reviewer` | Quality, security, maintainability checks |
| `style-checker` | Style guide and CLAUDE.md compliance |
| `trading-validator` | Trading logic, risk calculations |
| `verifier` | End-to-end functional verification |
| `verification-challenger` | Filtering false positives |
| `planner` | Creating implementation plans |
| `parallel-coordinator` | Orchestrating multiple agents |

### Invoke Directly

```
"Use the bug-finder agent on the recent changes"
"Run trading-validator on robo_trader/runner_async.py"
"Have the planner create a plan for adding X feature"
```

### Why Subagents Work

Complex tasks require X tokens of input context, accumulate Y tokens of working context, and produce Z tokens of answer. Subagents farm out the (X + Y) work and return only the Z token answer, keeping your main context clean.

---

## Parallel Sessions (Boris's Setup)

Run 5 terminal tabs, each with a Claude session:

| Tab | Name | Purpose | Key Command |
|-----|------|---------|-------------|
| 1 | MAIN | Primary development | (work happens here) |
| 2 | TEST | Continuous testing | `/test-and-commit` |
| 3 | REVIEW | Code review | `/review` |
| 4 | DOCS | Research/documentation | (research) |
| 5 | HOTFIX | Quick fixes | (emergency fixes) |

See [PARALLEL_CLAUDE_SETUP.md](./PARALLEL_CLAUDE_SETUP.md) for full setup guide.

### Standard Workflow

```
Tab 1 (MAIN):   Implement changes
Tab 3 (REVIEW): /review         → 6 subagents check code
Tab 1 (MAIN):   Fix issues found
Tab 2 (TEST):   /test-and-commit → Tests pass? → Commit
User:           git push
```

---

## Context Management

The 200k token limit requires strategy:

1. **`/clear` often** — Start fresh for new tasks
2. **"Document & Clear"** — For complex tasks:
   - Have Claude dump plan/progress to handoff file
   - Use `/clear`
   - New session reads the handoff and continues
3. **Use handoff/ directory** — Write session summaries here

### Resume Sessions

```bash
claude --resume     # Continue last session
claude --continue   # Same as --resume
```

---

## CLAUDE.md Best Practices

### Keep Critical Rules

RoboTrader's CLAUDE.md is comprehensive because trading systems have many gotchas. The "Common Mistakes" table is essential.

### Good Rules for Trading

- "Use `python3` not `python` on macOS"
- "Use `lsof` for port checking, not `socket.connect_ex()` (creates zombies)"
- "Use `side` column, not `action` in trades table"
- "Convert Decimal to float before math operations"
- "Market close is 4:00 PM ET, not 4:30"

### The Learning Loop

```
Claude makes mistake
       ↓
You fix immediate issue
       ↓
Add rule to CLAUDE.md "Common Mistakes" table
       ↓
Commit with message: "docs: Add [topic] to common mistakes"
       ↓
All future sessions benefit
```

---

## Daily Workflow Template

```
Morning:
1. Pull latest: git pull origin main
2. Start trader: ./START_TRADER.sh
3. Check dashboard: http://localhost:5001

During development:
4. Plan if complex: /plan
5. Implement changes
6. /verify frequently
7. /code-simplifier if needed

Before shipping:
8. /review or /two-phase-review
9. /test-and-commit

When you learn something:
10. /retrospective or /shared-knowledge
```

---

## The Golden Rule

**Always verify your work.**

```
Write Code → /verify → Fix Issues → /verify → Ship
```

This single habit will 2-3x your code quality.

---

## Sources

- [How Boris Cherny Uses Claude Code](https://paddo.dev/blog/how-boris-uses-claude-code/)
- [Writing a Good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Claude Code Hooks Guide](https://code.claude.com/docs/en/hooks-guide)
- [How to Run Coding Agents in Parallel](https://towardsdatascience.com/how-to-run-coding-agents-in-parallell/)
