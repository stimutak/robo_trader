---
allowed-tools: Bash(git:*)
description: Create a new git worktree that tracks the remote default branch
argument-hint: (no arguments needed)
---

# Git Worktree Command

## Context

Current repository info: !`git remote get-url origin 2>/dev/null || echo "No remote origin found"`

Current branch: !`git branch --show-current`

Existing worktrees: !`git worktree list`

Existing branches: !`git branch --list 'worktree*'`

Default remote branch: !`git remote show origin 2>/dev/null | grep "HEAD branch" | cut -d: -f2 | xargs || git branch -r | grep -E 'origin/(main|master)' | head -1 | sed 's/.*origin\///'`

## Your task

Create a new git worktree with an auto-generated branch name tracking the remote default branch.

1. `git fetch origin`
2. Determine default remote branch (origin/main or origin/master)
3. Find next available `worktreeN` number from existing branches
4. Create worktree: `git worktree add --track -b worktreeN /worktreeN origin/<default-branch>`
5. Display: new branch name, worktree path, upstream branch
