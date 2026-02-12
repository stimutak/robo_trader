---
description: Run tests, fix failures, then commit
---

# Test and Commit

## Step 1: Run Tests

```bash
python3 -m pytest tests/ -x --tb=short -q
```

If tests fail: analyze, fix, and re-run until passing.

## Step 2: Lint Check

```bash
python3 -m black --check .
python3 -m flake8 .
```

## Step 3: Commit

Stage specific files (never `git add -A`) and commit with conventional format.

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
