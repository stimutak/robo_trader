# Parallel Claude Instances Setup Guide

Based on Boris Cherny's workflow for maximum coding efficiency.

## Overview

Run 5+ Claude Code instances in parallel to dramatically increase development speed. While one agent runs tests, another refactors code, and a third reviews changes.

## Terminal Setup (5 Instances)

### Step 1: Open 5 Terminal Tabs

Use your preferred terminal (iTerm2, Terminal.app, VS Code terminal):

```bash
# Tab 1: Main development
cd ~/robo_trader && claude

# Tab 2: Test runner
cd ~/robo_trader && claude

# Tab 3: Code review
cd ~/robo_trader && claude

# Tab 4: Documentation/research
cd ~/robo_trader && claude

# Tab 5: Bug fixes/hotfixes
cd ~/robo_trader && claude
```

### Step 2: Label Your Tabs

Most terminals support tab naming:
- **iTerm2**: Right-click tab → Edit Tab Title
- **Terminal.app**: Shell → Edit Title
- **VS Code**: Tabs auto-label with directory

Suggested labels:
1. `MAIN` - Primary development
2. `TEST` - Continuous testing
3. `REVIEW` - Code review
4. `DOCS` - Research/documentation
5. `HOTFIX` - Quick fixes

### Step 3: Enable System Notifications

Claude Code sends notifications when it needs input. Enable them:

```bash
# macOS - notifications should work automatically
# Check System Preferences → Notifications → Terminal
```

## Task Distribution Strategy

### Tab 1: MAIN - Primary Development
- Feature implementation
- Major refactoring
- Complex problem solving

### Tab 2: TEST - Continuous Testing
```bash
# Keep running:
/test-and-commit
# Or continuous test watch
python3 -m pytest tests/ --watch
```

### Tab 3: REVIEW - Code Review
```bash
# After changes in MAIN:
/review
```

### Tab 4: DOCS - Research & Documentation
- Searching codebase for patterns
- Reading documentation
- Answering questions about architecture
- Updating handoff documents

### Tab 5: HOTFIX - Quick Fixes
- Bug fixes discovered during development
- Quick patches
- Emergency fixes

## Handoff Between Instances

### Using handoff/ Directory

When one Claude needs context from another:

1. **In Source Claude**: Write summary to handoff file
   ```bash
   # Create handoff document
   Write to: handoff/HANDOFF_<date>_<topic>.md
   ```

2. **In Target Claude**: Read the handoff
   ```bash
   Read: handoff/HANDOFF_<date>_<topic>.md
   ```

### Using Shared CLAUDE.md

All instances share the same CLAUDE.md knowledge:
- Mistakes get added once, all instances learn
- Guidelines apply consistently

### Session Teleportation (Advanced)

Transfer a session between devices:
```bash
# On source machine
claude --background &

# On target machine
claude --teleport <session-id>
```

## Parallel Workflow Examples

### Example 1: Feature + Tests in Parallel

**Tab 1 (MAIN)**: "Implement new position sizing algorithm"
**Tab 2 (TEST)**: "Write tests for position sizing as I implement"

### Example 2: Fix + Review in Parallel

**Tab 1 (MAIN)**: "Fix the Decimal/float type error"
**Tab 3 (REVIEW)**: "/review the fix for any other similar issues"

### Example 3: Research + Implement in Parallel

**Tab 4 (DOCS)**: "Research how other trading systems handle market hours"
**Tab 1 (MAIN)**: Start implementing based on research findings

## Best Practices

### 1. One Task Per Instance
Don't overload a single Claude with multiple unrelated tasks.

### 2. Use Plan Mode First
```
Shift+Tab twice → Plan Mode
```
Get the plan right before executing.

### 3. Let Claude Verify
Always give Claude a way to verify its work:
- Run tests
- Check linting
- Validate output

### 4. Share Knowledge Immediately
When you find an issue, add it to CLAUDE.md so all instances learn.

### 5. Use Slash Commands
```bash
/review      # Multi-subagent code review
/test-and-commit  # Test then commit
/verify-trading   # Check trading system
/pr          # Create pull request
/commit      # Quick commit
/oncall-debug    # Debug production issues
```

## Monitoring Instance Status

### Check Running Instances
```bash
pgrep -fl claude
```

### Kill All Instances
```bash
pkill -f claude
```

### Check CPU Usage
```bash
top -l 1 | grep claude
```

## Tips for Maximum Efficiency

1. **Start day with research Claude** - Catch up on what happened overnight
2. **Keep test Claude always running** - Immediate feedback on changes
3. **Use review Claude after significant changes** - Catch issues early
4. **Document in docs Claude** - Don't interrupt main development flow
5. **Hotfix Claude for emergencies** - Don't derail main work

## Troubleshooting

### Claude Not Responding
```bash
# Check if process is hung
ps aux | grep claude

# Kill and restart
pkill -9 -f claude
claude
```

### Too Many Instances
If system becomes slow:
- Close research/docs Claude (least critical)
- Keep main and test running
- Use single Claude for simple tasks

### Context Getting Stale
If Claude seems confused about recent changes:
- Check git status in that terminal
- Refresh with `/clear` command
- Re-read relevant files
