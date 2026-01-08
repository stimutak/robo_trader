# Test and Commit Workflow

Run tests, then commit if passing. Fix issues if failing.

## Step 1: Run Tests

```bash
python3 -m pytest tests/ -x --tb=short -q
```

## Step 2: Evaluate Results

### If ALL tests pass:
1. Run linting checks:
   ```bash
   python3 -m black --check .
   python3 -m isort --check-only .
   python3 -m flake8 .
   ```

2. If linting passes, proceed to commit:
   ```bash
   git add -A
   git status
   ```

3. Generate commit message based on changes:
   - Look at staged files
   - Summarize the "why" not the "what"
   - Use conventional commit format (feat:, fix:, refactor:, etc.)

4. Create commit with generated message

### If tests FAIL:
1. Analyze the failure output
2. Identify root cause
3. Fix the issue
4. Re-run tests
5. Repeat until passing
6. Then proceed to commit

## Commit Message Format

```
<type>: <short description>

<longer description if needed>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Types
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `docs`: Documentation only
- `chore`: Maintenance tasks
