---
description: Extract session learnings and update CLAUDE.md
---

# Retrospective - Extract Learnings

Review the current session and extract learnings to update CLAUDE.md.

## Process

### Step 1: Identify Session Events

Review the conversation for:
- Errors encountered and how they were fixed
- Patterns that worked well
- Anti-patterns that caused problems
- New domain knowledge discovered

### Step 2: Update CLAUDE.md

Add new entries to the "Common Mistakes" table in CLAUDE.md:

```markdown
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| [What went wrong] | [How to do it right] | [YYYY-MM-DD] |
```

Good entries are specific, actionable, and include the "why". Skip one-time issues unlikely to recur.

### Step 3: Commit

If updates were made:
```
docs: add [topic] to CLAUDE.md common mistakes
```

## When NOT to Update

- Temporary implementation details
- User-specific preferences
- Issues already documented
- Highly situational problems
