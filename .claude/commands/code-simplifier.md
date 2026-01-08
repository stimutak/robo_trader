# Code Simplifier

Review recently written code and simplify it while maintaining functionality.

## Analysis Criteria

### 1. Unnecessary Complexity
- Nested conditionals that can be flattened
- Redundant checks or validations
- Over-engineered abstractions
- Premature optimization

### 2. Duplicate Logic
- Copy-pasted code blocks
- Similar functions that could be consolidated
- Repeated patterns that could use a helper

### 3. Variable Naming
- Names that don't describe purpose
- Abbreviations that aren't clear
- Inconsistent naming conventions

### 4. Code Structure
- Functions doing too many things
- Long methods that could be split
- Poor separation of concerns

## Simplification Rules

1. **Keep the same functionality** - No behavior changes
2. **Maintain test coverage** - All existing tests must pass
3. **Follow existing patterns** - Match the codebase style
4. **Don't over-abstract** - 3 similar lines > premature abstraction

## Output Format

For each simplification:

```markdown
### File: <path>

**Before:**
```python
# Original code
```

**After:**
```python
# Simplified code
```

**Why:** Brief explanation of the improvement
```

## What NOT to Change

- Code that's already simple and clear
- Performance-critical sections (unless obviously wrong)
- Code with extensive comments explaining complexity
- External API interfaces
