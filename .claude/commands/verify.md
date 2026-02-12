---
description: Run full verification loop: tests, lint, types, trading checks
---

# Verification Loop

Run a complete verification loop on recent changes. This is the single most important practice for quality.

> "Give Claude a way to verify its work. If Claude has that feedback loop, it will 2-3x the quality of the final result." — Boris Cherny

## Verification Steps

### 1. Run Tests
```bash
python3 -m pytest tests/ -v --tb=short
```

If tests fail:
- Fix the failing tests
- Re-run until all pass
- Report which tests required fixes

### 2. Run Linters
```bash
python3 -m black --check .
python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

If linting fails:
- Run `python3 -m black .` to auto-fix formatting
- Fix remaining issues manually
- Re-run until clean

### 3. Type Check (if applicable)
```bash
python3 -m mypy robo_trader/ --ignore-missing-imports 2>/dev/null || echo "mypy not configured"
```

### 4. Trading-Specific Checks

#### Gateway Status
```bash
python3 scripts/gateway_manager.py status
```

#### Zombie Connections
```bash
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT | wc -l
```
Should be 0.

#### Safety Features
```bash
python3 test_safety_features.py
```

### 5. Import Verification
```bash
python3 -c "from robo_trader.runner_async import AsyncRunner; print('Import OK')"
```

## Verification Loop Pattern

```
Make changes
    ↓
Run /verify
    ↓
Fix any failures ←──┐
    ↓               │
Re-run /verify ─────┘ (until all pass)
    ↓
Changes verified ✅
```

## Output Format

```markdown
## Verification Results

| Check | Status | Notes |
|-------|--------|-------|
| Tests | ✅/❌ | X/Y passed |
| Black | ✅/❌ | |
| Flake8 | ✅/❌ | X errors |
| Gateway | ✅/❌ | |
| Zombies | ✅/❌ | Count: X |
| Safety | ✅/❌ | |
| Imports | ✅/❌ | |

**Overall:** VERIFIED / NEEDS FIXES

### Fixes Applied (if any)
1. ...
2. ...
```

## Critical Rule

**Do NOT mark verification as complete if any check fails.**

Keep iterating until all checks pass.
