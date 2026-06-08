---
name: "source-command-retrospective"
description: "Extract session learnings and update AGENTS.md"
---

# source-command-retrospective

Use this skill when the user asks to run the migrated source command `retrospective`.

## Command Template

# Retrospective - Extract Learnings

Review the current session and extract learnings to update AGENTS.md.

## Process

### Step 1: Identify Session Events

Review the conversation for:
- Errors encountered and how they were fixed
- Patterns that worked well
- Anti-patterns that caused problems
- New domain knowledge discovered

### Step 2: Update AGENTS.md

Add new entries to the "Common Mistakes" table in AGENTS.md:

```markdown
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| [What went wrong] | [How to do it right] | [YYYY-MM-DD] |
```

Good entries are specific, actionable, and include the "why". Skip one-time issues unlikely to recur.

### Step 3: Commit

If updates were made:
```
docs: add [topic] to AGENTS.md common mistakes
```

## When NOT to Update

- Temporary implementation details
- User-specific preferences
- Issues already documented
- Highly situational problems
