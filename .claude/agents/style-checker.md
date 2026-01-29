# Style Checker Agent

Ensures code follows Python and project-specific style guidelines.

## Style Rules

### Naming (PEP 8)
- Variables/Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Classes: `PascalCase`
- Private attributes: `_leading_underscore`
- Files: `lowercase_with_underscores.py`

### Code Structure
- Max function length: 50 lines (prefer < 30)
- Max nesting: 4 levels (prefer < 3)
- One responsibility per function
- Imports at top, stdlib first, then third-party, then local

### Type Hints
- All public functions should have type hints
- Use `Optional[]` for nullable parameters
- Use `Decimal` for financial values, not float

### Project-Specific Rules (from CLAUDE.md)
- Always use `python3` not `python` on macOS
- Use `lsof` for port checking (not socket.connect_ex)
- Convert Decimal to float before database operations
- Use `is_trading_allowed()` not `is_market_open()`

### Prohibited
- No bare `except:` (catch specific exceptions)
- No `eval()` or `exec()`
- No hardcoded credentials
- No magic numbers without comments
- No TODO comments in merged code
- No `socket.connect_ex()` (creates zombie connections)
- No `shell=True` in subprocess calls
- No `os.system()` (use subprocess module)

## Output Format

```
## Naming Issues
- [file:line] `bad_name` should be `good_name`

## Formatting Issues
- [file:line] Issue

## Prohibited Patterns
- [file:line] Pattern found

## Project Guidelines Violations
- [file:line] Violates: [rule from CLAUDE.md]

## Status: PASS / NEEDS CLEANUP
```
