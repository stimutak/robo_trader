# Retrospective - Extract Learnings

Review the current session and extract learnings to update CLAUDE.md.

> "When Claude makes a mistake: Fix the immediate issue, add a rule to CLAUDE.md, commit it. This makes all future sessions smarter." — Boris Cherny

## Process

### Step 1: Identify Session Events

Review the conversation for:
- Errors encountered and how they were fixed
- Patterns that worked well
- Anti-patterns that caused problems
- New domain knowledge discovered
- Type mismatches or API surprises

### Step 2: Categorize Learnings

For each learning, categorize:

| Category | Example |
|----------|---------|
| **Type Errors** | Decimal vs float in financial calculations |
| **Connection/Socket** | Using lsof instead of socket.connect_ex() |
| **Async/Await** | Proper executor usage for stdin |
| **Database** | Column name mismatches (action vs side) |
| **Trading Logic** | Market close is 4:00 PM not 4:30 PM |
| **Config** | Attribute paths that don't exist |

### Step 3: Update CLAUDE.md

Add new entries to the "Common Mistakes" table in `/Users/oliver/robo_trader/CLAUDE.md`:

```markdown
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| [What went wrong] | [How to do it right] | [YYYY-MM-DD] |
```

### Step 4: Verify Entry Quality

Good entries:
- Specific and actionable
- Include the "why" not just "what"
- Reference actual code patterns if helpful

Bad entries:
- Too vague ("be careful with types")
- One-time issues unlikely to recur
- Personal preferences not project rules

## Output Format

```markdown
## Session Retrospective

### Learnings Identified

1. **[Category]:** [Brief description]
   - What happened: ...
   - Root cause: ...
   - Prevention: ...

2. ...

### CLAUDE.md Updates

Added to Common Mistakes table:
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| ... | ... | ... |

### No Updates Needed (if applicable)

Reason: Session went smoothly / Issues were one-time / Already documented
```

## Commit Message

If updates were made:
```
docs: add [topic] to CLAUDE.md common mistakes

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## When NOT to Update

- Temporary implementation details
- User-specific preferences
- Issues already documented
- Highly situational problems
