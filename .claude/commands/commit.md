---
description: Quick conventional commit of staged changes
---

# Quick Commit

1. Check changes:
```bash
git status
git diff --stat
```

2. Stage specific files (never `git add -A`):
```bash
git add <relevant files>
```

3. Generate commit message:
   - Summarize the "why" not the "what"
   - Use conventional format: `<type>: <short description>`

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `style`
