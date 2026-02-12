---
description: Multi-agent parallel code review
---

# Multi-Subagent Code Review

Review the current changes using multiple parallel perspectives, then verify findings.

## Phase 1: Parallel Analysis (Launch 4 subagents)

### Subagent 1: Bug Hunter
Search for:
- Type mismatches (especially Decimal/float in financial calculations)
- Null/None checks missing
- Async/await issues (missing await, unhandled exceptions)
- Off-by-one errors
- Resource leaks (unclosed connections, files)

### Subagent 2: Trading Logic Validator
Check:
- Risk calculations use Decimal precision
- Position sizing respects MAX_OPEN_POSITIONS
- Market hours logic is correct (4:00 PM close, not 4:30)
- Stop loss and take profit calculations
- Order validation before execution

### Subagent 3: Style & Guidelines Checker
Verify against CLAUDE.md:
- Uses `python3` not `python` on macOS
- No `socket.connect_ex()` (creates zombies)
- Uses `lsof` for port checking
- Proper error handling patterns
- Follows existing code conventions

### Subagent 4: Security Auditor
Check for:
- Hardcoded secrets or credentials
- SQL injection vulnerabilities
- Command injection in Bash calls
- Input validation at boundaries
- Sensitive data in logs (should be masked)

## Phase 2: Verification (Launch 2 subagents)

### Subagent 5: False Positive Filter
Review findings from Phase 1 and identify:
- False positives that aren't actual issues
- Low-priority items that can be ignored
- Duplicate findings across subagents

### Subagent 6: Priority Ranker
Categorize remaining issues:
- CRITICAL: Must fix before merge
- HIGH: Should fix, risk if not
- MEDIUM: Nice to fix
- LOW: Optional improvement

## Output Format

```markdown
## Code Review Summary

### Critical Issues (must fix)
- [ ] Issue description | File:line | Subagent

### High Priority
- [ ] Issue description | File:line | Subagent

### Medium Priority
- [ ] Issue description | File:line | Subagent

### Verified Non-Issues (false positives filtered)
- Item | Why it's not an issue
```
