# Handoff: IBKR Zombie Connection Fix

**Date:** 2025-12-06
**Status:** ✅ COMPLETE - System stable for 1+ hour continuous connection

## Summary

Fixed critical bug causing IBKR Gateway API connection timeouts. The root cause was `socket.connect_ex()` calls in three locations that created CLOSE_WAIT zombie connections, blocking subsequent API handshakes.

## Root Cause

When checking if the IBKR Gateway port was open, the code used `socket.connect_ex()` which:
1. Creates a full TCP 3-way handshake with the Gateway
2. Immediately closes the socket without completing the IBKR API handshake
3. Gateway sees this as an improperly disconnected client
4. Creates a CLOSE_WAIT zombie connection that blocks ALL future API connections

## Files Modified

### 1. `app.py` (lines 2983-3041)
**Function:** `check_ibkr_connection()`
**Change:** Replaced `socket.connect_ex()` with `lsof` subprocess call
```python
# Before (creates zombies):
sock.connect_ex(("127.0.0.1", 4002))

# After (no zombies):
result = subprocess.run(
    ["lsof", "-nP", "-iTCP:4002", "-sTCP:LISTEN"],
    capture_output=True, text=True, timeout=2
)
```

### 2. `robo_trader/runner_async.py` (lines 460-497)
**Function:** `test_port_open()` → `test_port_open_lsof()`
**Change:** Replaced socket-based check with lsof subprocess
```python
def test_port_open_lsof(port=7497):
    """Check if port is listening using lsof (no zombies)."""
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
        capture_output=True, text=True, timeout=5
    )
    return result.returncode == 0 and "LISTEN" in result.stdout
```

### 3. `robo_trader/utils/tws_health.py` (lines 141-180)
**Function:** `is_port_listening()`
**Change:** Replaced socket-based check with lsof subprocess

### 4. `robo_trader/runner_async.py` (line 808)
**Bug:** Python 3.12+ variable scoping error
**Change:** Removed redundant `from pathlib import Path` import that shadowed module-level import

## Why `lsof` Instead of `socket.connect_ex()`

| Method | Creates TCP Connection | Creates Zombie | Safe for IBKR |
|--------|----------------------|----------------|---------------|
| `socket.connect_ex()` | ✅ Yes | ✅ Yes | ❌ No |
| `lsof -sTCP:LISTEN` | ❌ No | ❌ No | ✅ Yes |

`lsof` queries the kernel's socket table directly without creating any network connections.

## Testing Verification

After fix:
```
✓ No zombie connections detected on port 4002
✓ Port 4002 is open - proceeding to IBKR connect
✓ Connected to IBKR via subprocess
✅ Connection established successfully
✓ IBKR connection established successfully with robust connection
```

System ran stable for 1+ hour with continuous API connection.

## IBC Configuration Note

IBC expects config at `/Users/oliver/ibc/config.ini` but the project stores it at `/Users/oliver/robo_trader/config/ibc/config.ini`. Copied the file to the expected location.

## Related Files Added

- `scripts/gateway_manager.py` - Cross-platform Gateway management
- `scripts/start_gateway.sh` - Gateway launcher script
- `IBCMacos-3/` - IBC for macOS (automated Gateway management)
- `config/ibc/config.ini.template` - IBC config template

## Commands

```bash
# Start the system (handles everything automatically)
./START_TRADER.sh

# Check for zombies manually
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT

# Check Gateway status
python3 scripts/gateway_manager.py status
```

