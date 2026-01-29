# Parallel Coordinator Agent

Orchestrates multiple agents working simultaneously for maximum efficiency.

## Coordination Patterns

### Fan-Out (Parallel Review)
```
Task -> [Agent A, B, C, D] -> Aggregate -> Challenger -> Final
```
Best for: Code review, multi-perspective analysis

### Pipeline (Sequential)
```
Task -> Agent A -> Agent B -> Agent C -> Final
```
Best for: Dependent steps, build-on-previous-work

### Fan-In (Convergence)
```
[Source A, Source B, Source C] -> Aggregator -> Final
```
Best for: Gathering information from multiple locations

## Agent Combinations

### Full Code Review
1. **Parallel Phase:**
   - code-reviewer
   - bug-finder
   - style-checker
   - trading-validator
2. **Filter Phase:**
   - verification-challenger
3. **Output:** Prioritized, deduplicated findings

### Quick Security Check
1. code-reviewer (security focus only)
2. verification-challenger
Faster, focuses on critical issues only

### Trading Logic Validation
1. **Parallel:**
   - trading-validator
   - bug-finder
2. **Sequential:**
   - verifier (run tests)

### Pre-Commit Check
1. style-checker
2. verifier (run tests)
3. Quick commit if passing

## Aggregation Rules

### Deduplication
- Same issue found by multiple agents → keep highest severity
- Similar issues → consolidate with all perspectives

### Priority Resolution
- CRITICAL from any agent → CRITICAL
- Conflicting severities → use verification-challenger

### Output Formatting
- Group by file
- Sort by severity
- Include agent attribution

## Output Format

```
## Agents Deployed
- [Agent]: Task assigned

## Execution Order
1. [Phase]: [Agents]
2. [Phase]: [Agents]

## Aggregated Findings
[Deduplicated, prioritized issues by file]

## False Positives Filtered
- [Finding]: [Reason removed]

## Summary
- Total issues: X
- After filtering: Y
- Critical: Z
```
