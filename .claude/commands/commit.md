# Quick Commit Workflow

Fast commit with proper message format.

## Steps

1. Check current changes:
```bash
git status
git diff --stat
```

2. Stage changes:
```bash
git add -A
```

3. Generate commit message based on:
- What files changed
- What the changes accomplish
- Why the changes were made

4. Commit with proper format:
```bash
git commit -m "$(cat <<'EOF'
<type>: <short description>

<optional longer description>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

## Commit Types

| Type | When to Use |
|------|-------------|
| `feat` | New feature or functionality |
| `fix` | Bug fix |
| `refactor` | Code restructuring, no behavior change |
| `test` | Adding or updating tests |
| `docs` | Documentation changes |
| `chore` | Maintenance, dependencies, configs |
| `perf` | Performance improvements |
| `style` | Formatting, no code change |

## Examples

```bash
# Feature
git commit -m "feat: add multi-subagent code review command"

# Bug fix
git commit -m "fix: correct market close time to 4:00 PM"

# Refactor
git commit -m "refactor: simplify position sizing calculation"
```
