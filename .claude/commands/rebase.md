---
allowed-tools: Bash(git:*)
description: Rebase current branch onto remote upstream branch with smart stash handling
argument-hint: [optional: target-branch-name]
---

# Git Rebase Command

## Context

Current git status: !`git status --porcelain`

Current branch: !`git branch --show-current`

Remote tracking branch: !`git rev-parse --abbrev-ref @{upstream} 2>/dev/null || echo "No upstream branch"`

## Your task

Arguments: $ARGUMENTS

Rebase current branch onto the remote upstream branch.

1. Stash uncommitted changes if working directory is dirty
2. `git fetch origin`
3. Determine target: use argument if provided, otherwise use upstream tracking branch
4. `git rebase origin/<target-branch>`
5. Pop stash if one was created
6. If conflicts occur at any step, provide clear guidance to the user
