---
description: Full PR workflow: test, lint, push, create PR
---

# Create Pull Request

## Step 1: Verify Clean

```bash
python3 -m pytest tests/ --tb=short
python3 -m black --check .
python3 -m flake8 .
```

Fix any issues before proceeding.

## Step 2: Check State

```bash
git status
git log main..HEAD --oneline
```

## Step 3: Create PR

```bash
gh pr create --title "<type>: <description>" --body "$(cat <<'EOF'
## Summary
- Brief description of changes

## Test Plan
- [ ] All tests pass
- [ ] Linting passes
EOF
)"
```

Use conventional commit types for the PR title.

If `gh pr create` fails, provide a clickable compare URL with pre-filled title and body per `.claude/CLAUDE.md` instructions.
