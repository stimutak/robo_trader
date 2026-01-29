# Update Shared Knowledge

When Claude makes a mistake or discovers an important pattern, update CLAUDE.md so all future sessions learn from it.

## When to Update

1. **After fixing a bug** - Add to Common Mistakes table
2. **After discovering a pattern** - Document in appropriate section
3. **After a user reports an issue** - Capture the root cause and fix
4. **After completing a significant feature** - Update relevant sections

## Update Process

1. **Fix the immediate issue**
2. **Identify the root cause**
3. **Formulate a clear rule** - What should Claude do/not do?
4. **Find the right section in CLAUDE.md**
5. **Add the entry with today's date**
6. **Commit with message**: `docs: Add [topic] to CLAUDE.md common mistakes`

## Common Mistakes Table Format

```markdown
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| Brief description of error | What to do instead | YYYY-MM-DD |
```

## Categories in CLAUDE.md

- **CRITICAL - Data Destruction**: Never delete user data
- **Type Errors**: Decimal/float, int/datetime mismatches
- **Connection & Socket Errors**: Zombies, port checking
- **Async/Await Errors**: Race conditions, blocking
- **Database Performance Errors**: N+1 queries, caching
- **Config Attribute Errors**: Wrong config paths
- **Trading Logic Errors**: Position sizing, market hours, duplicates

## Example Entry

```markdown
| Using `price` (Decimal) in float division | Use `price_float` for calculations | 2025-12-29 |
```

## Output

After updating CLAUDE.md:

```
## Shared Knowledge Updated

### Added Entry
- Section: [section name]
- Mistake: [brief description]
- Correct Approach: [solution]

### Commit Ready
Run `/commit` to save this knowledge for future sessions.
```
