# Subprocess Async Issue - Root Cause Found

## Problem

Subprocess IBKR client works perfectly in isolation but times out when called from runner_async.

## Root Cause

**`asyncio.create_subprocess_exec` gets starved in busy async environments!**

### Evidence

1. ✅ `test_minimal_runner.py` works (simple async environment)
2. ❌ `runner_async.py` times out (complex async environment with many tasks)
3. ❌ `test_subprocess_in_async_env.py` times out (even with just 3 background tasks)

### Technical Explanation

When using `asyncio.create_subprocess_exec`:
- Subprocess stdout/stdin are async streams
- Reading/writing requires awaiting
- In a busy event loop with many tasks, these await calls can be delayed
- The subprocess worker sends a response, but the parent is busy with other tasks
- By the time the parent tries to read, the 15-second timeout has expired

## Solution

**Use regular `subprocess.Popen` with threading instead of `asyncio.create_subprocess_exec`**

### Why Threading Works

1. **Dedicated thread** for subprocess communication
2. **Not affected** by async event loop congestion
3. **Blocking I/O** is fine in a thread
4. **Async wrapper** makes it look async to the caller

### Implementation

Replace:
```python
# Current (BROKEN in busy async environments)
self.process = await asyncio.create_subprocess_exec(
    python_exe, str(worker_script),
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
```

With:
```python
# Fixed (works in any environment)
import subprocess
import threading

self.process = subprocess.Popen(
    [python_exe, str(worker_script)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    close_fds=True,
)

# Use threading for I/O
def _read_loop():
    while True:
        line = self.process.stdout.readline()
        if not line:
            break
        self._response_queue.put(line)

self._reader_thread = threading.Thread(target=_read_loop, daemon=True)
self._reader_thread.start()
```

## Timeline

- **Fix implementation:** 30 minutes
- **Testing:** 15 minutes
- **Total:** 45 minutes

## Status

Ready to implement fix.

---
**Date:** 2025-10-15
**Root cause:** asyncio.create_subprocess_exec starved in busy event loop
**Solution:** Use subprocess.Popen with threading

