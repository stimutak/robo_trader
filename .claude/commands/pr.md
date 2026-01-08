# Create Pull Request Workflow

Full PR workflow with tests, linting, and review.

## Pre-PR Checklist

### Step 1: Run Full Test Suite

```bash
python3 -m pytest tests/ --tb=short
```

All tests must pass.

### Step 2: Run Linting

```bash
python3 -m black --check .
python3 -m isort --check-only .
python3 -m flake8 .
```

Fix any issues before proceeding.

### Step 3: Run BugBot

```bash
./scripts/run_bugbot.sh
```

- No CRITICAL bugs allowed
- HIGH priority bugs should be addressed

### Step 4: Check for Uncommitted Changes

```bash
git status
git diff --stat
```

Ensure all changes are committed.

### Step 5: Create PR

```bash
# Get current branch
BRANCH=$(git branch --show-current)

# Get commits since main
git log main..HEAD --oneline

# Create PR
gh pr create --title "<type>: <description>" --body "$(cat <<'EOF'
## Summary
- Brief description of changes
- Key files modified
- Why these changes were made

## Changes
- [ ] Change 1
- [ ] Change 2

## Test Plan
- [ ] All tests pass (`pytest tests/`)
- [ ] Linting passes (`black`, `isort`, `flake8`)
- [ ] BugBot finds no critical issues
- [ ] Manual verification of key functionality

## Related Issues
Closes #XXX (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## PR Title Format

Use conventional commits:
- `feat: Add new feature`
- `fix: Fix bug in X`
- `refactor: Restructure Y`
- `test: Add tests for Z`
- `docs: Update documentation`
- `chore: Maintenance task`

## After PR Created

1. Note the PR URL
2. Run `/review` command on the PR if needed
3. Address any CI failures
4. Request review if required
