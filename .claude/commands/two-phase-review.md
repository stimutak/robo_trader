# Two-Phase Review (Boris Cherny Method)

Run a parallel code review with challenger phase to filter false positives.

This technique typically filters 20-40% of findings as false positives.

## Phase 1: Fan-Out (Parallel)

Launch 4 subagents simultaneously using the Task tool:

### Agent 1: Code Reviewer
Focus on:
- Code quality and maintainability
- Security vulnerabilities (OWASP top 10)
- Resource management (connections, files)
- Error handling completeness

### Agent 2: Bug Finder
Focus on:
- Type mismatches (Decimal/float in financial code)
- Race conditions and async issues
- Null/None handling
- Off-by-one and boundary errors
- Resource leaks

### Agent 3: Trading Validator
Focus on (trading-specific):
- Risk calculations use Decimal precision
- Position sizing respects MAX_OPEN_POSITIONS
- Market hours logic (4:00 PM close)
- Duplicate buy protection working
- Stop loss and take profit calculations

### Agent 4: Style Checker
Focus on:
- CLAUDE.md guidelines compliance
- Uses `python3` not `python` on macOS
- Uses `lsof` not `socket.connect_ex()` for port checks
- Project conventions

## Phase 2: Challenge (Filter)

For EACH finding from Phase 1, the verification-challenger agent asks:

1. **Is this actually a problem?** Or is it theoretical only?
2. **Can I reproduce it?** With realistic inputs?
3. **Is there defensive code elsewhere?** That handles this case?
4. **Is the severity correct?** Should it be downgraded?
5. **Would the fix introduce new issues?** Unintended consequences?

### Trading-Specific Acceptable Patterns

These are NOT bugs:
- "Already have long position" → Duplicate buy protection working
- Fresh AsyncRunner each cycle → Intentional for process isolation
- Multiple DB checks for same data → Defensive redundancy
- 120-second duplicate window → Matches trading cycle timing

## Output Format

```markdown
## Two-Phase Review Results

### Confirmed Issues (Action Required)

#### Critical
- [ ] Issue | File:line | Confirmed by challenger

#### High
- [ ] Issue | File:line | Confirmed by challenger

#### Medium
- [ ] Issue | File:line | Confirmed by challenger

### Filtered Out (False Positives)

| Finding | Why Not an Issue | Phase 1 Agent |
|---------|------------------|---------------|
| ... | ... | ... |

### Statistics
- Total findings: X
- Confirmed: Y
- Filtered: Z
- False positive rate: Z/X = N%
```

## Execution

1. Use Task tool to launch all 4 Phase 1 agents in parallel
2. Collect their findings
3. Launch verification-challenger agent with all findings
4. Present filtered results
