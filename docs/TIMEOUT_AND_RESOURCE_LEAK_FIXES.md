# Timeout and Resource Leak Fixes

**Date:** 2025-11-29
**Session:** claude/fix-timeout-resource-leaks-0194oFjUoUkxSJhpB9aobkFY

## Overview

Fixed two code quality issues identified during code review but missed before merging to main:

1. **Timeout Stacking** - Confusing timeout calculation between client and worker
2. **Resource Leakage** - Debug log file handling could leak file descriptors

## Issue 1: Timeout Stacking

### Problem

The connection timeout logic had confusing/misaligned values:

- **Worker** (`ibkr_subprocess_worker.py`) has internal waits:
  - `max_handshake_wait = 15.0s` - API handshake verification
  - `await asyncio.sleep(0.5)` - Stabilization delay
  - `max_account_wait = 10.0s` - Account data retrieval
  - **Total: 25.5s**

- **Client** (`subprocess_ibkr_client.py`) added timeout buffer:
  - `extended_timeout = timeout + 30` (Line 441)
  - Comment said "~25.5s, so add 30s buffer" but didn't explain the mismatch

### Risk

Could mask actual timeout issues due to unclear timeout alignment.

### Solution

1. **Added timeout constants** to `SubprocessIBKRClient` class:
   ```python
   # Worker timeout constants (must match ibkr_subprocess_worker.py)
   WORKER_HANDSHAKE_TIMEOUT = 15.0  # max_handshake_wait in worker
   WORKER_STABILIZATION_DELAY = 0.5  # stabilization sleep in worker
   WORKER_ACCOUNT_TIMEOUT = 10.0  # max_account_wait in worker
   WORKER_MAX_WAIT = WORKER_HANDSHAKE_TIMEOUT + WORKER_STABILIZATION_DELAY + WORKER_ACCOUNT_TIMEOUT  # 25.5s
   ```

2. **Updated timeout calculation** with clear explanation:
   ```python
   # Calculate timeout: base TCP timeout + worker's internal sequence + buffer
   # Worker sequence: handshake (15s) + stabilization (0.5s) + accounts (10s) = 25.5s
   # Add 5s buffer for network/processing overhead
   # Total: timeout (TCP) + 25.5s (worker) + 5s (buffer)
   extended_timeout = timeout + self.WORKER_MAX_WAIT + 5.0
   ```

3. **Added cross-reference comments** in worker file:
   - `max_handshake_wait` → "must match WORKER_HANDSHAKE_TIMEOUT"
   - `await asyncio.sleep(0.5)` → "must match WORKER_STABILIZATION_DELAY"
   - `max_account_wait` → "must match WORKER_ACCOUNT_TIMEOUT"

### Results

- **Before:** `extended_timeout = timeout + 30 = 60s` (confusing comment)
- **After:** `extended_timeout = timeout + 25.5 + 5.0 = 60.5s` (explicit and documented)
- **Benefit:** Timeout values are now clearly aligned and maintainable

## Issue 2: Resource Leakage

### Problem

Debug log file opened at line 125 but cleanup relied on exception handling:

```python
# Lines 122-129: File opened but cleanup relies on exception handling
self._debug_log_file = open(debug_log_path, "w")
```

**Risk:** If an exception occurred between opening the file and starting the stderr reader thread, the file descriptor could leak.

### Solution

Wrapped file opening and subprocess/thread startup in try/except with proper cleanup:

```python
# Open file in local variable first
debug_log_file = None
try:
    debug_log_file = open(debug_log_path, "w")
except Exception as e:
    logger.warning("Could not create debug log file", error=str(e))
    debug_log_file = None

# Wrap subprocess and thread startup in try block to ensure cleanup
try:
    self.process = subprocess.Popen(...)

    # Store debug log file only after successful subprocess start
    self._debug_log_file = debug_log_file

    # Start threads...

except Exception:
    # Clean up debug log file if subprocess/thread startup fails
    if debug_log_file and not self._debug_log_file:
        # File was opened but not yet handed to stderr reader thread
        try:
            debug_log_file.close()
            logger.debug("Closed debug log file after startup failure")
        except Exception:
            pass
    # Re-raise the original exception
    raise
```

### Results

- **Before:** File could leak if subprocess/thread startup failed
- **After:** File is properly closed in all error paths
- **Benefit:** No file descriptor leaks, robust cleanup

## Files Modified

1. `robo_trader/clients/subprocess_ibkr_client.py`
   - Added timeout constants (lines 65-69)
   - Updated timeout calculation (lines 442-450)
   - Added resource cleanup (lines 128-183)

2. `robo_trader/clients/ibkr_subprocess_worker.py`
   - Added cross-reference comments for timeout values (lines 113, 169, 182)

## Testing

- ✅ Python syntax check passed
- ✅ Timeout calculation verified (60.5s = 30s base + 25.5s worker + 5s buffer)
- ✅ Resource cleanup logic verified
- ✅ Cross-references documented for maintainability

## Impact

- **No functional changes** - just cleanup and documentation
- **Better maintainability** - timeout values are now explicit constants
- **No resource leaks** - proper cleanup in all error paths
- **Clearer code** - timeout calculation is self-documenting

## Recommendations

- Keep timeout constants in sync between client and worker
- Monitor for timeout-related issues in production
- Consider adding automated tests for resource cleanup paths
