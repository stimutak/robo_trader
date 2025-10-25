# IBKR Port Auto-Detection

## Overview

The RoboTrader system now automatically detects whether IBKR Gateway or TWS is running and selects the correct API port. This eliminates the common configuration issue where the system tries to connect to the wrong port.

## Port Reference

| Service | Environment | Port |
|---------|-------------|------|
| Gateway | Paper Trading | 4002 |
| Gateway | Live Trading | 4001 |
| TWS | Paper Trading | 7497 |
| TWS | Live Trading | 7496 |

## How It Works

### Detection Strategy

The auto-detection follows this priority order:

1. **Environment Variable** - If `IBKR_PORT` is set, uses that value
2. **Process Detection** - Checks for running Gateway or TWS processes
3. **Port Scanning** - Checks which IBKR ports are listening
4. **Fallback** - Defaults to Gateway paper port (4002)

### Detection Logic

```
1. Check IBKR_PORT environment variable
   ├─ If valid → use that port
   └─ If not set → continue to step 2

2. Check for Gateway process (pgrep -f "ibgateway")
   ├─ If found
   │  ├─ Port 4002 listening → use 4002 (paper)
   │  ├─ Port 4001 listening → use 4001 (live)
   │  └─ No port listening → default to 4002
   └─ If not found → continue to step 3

3. Check for TWS process (pgrep -f "tws")
   ├─ If found
   │  ├─ Port 7497 listening → use 7497 (paper)
   │  ├─ Port 7496 listening → use 7496 (live)
   │  └─ No port listening → default to 7497
   └─ If not found → continue to step 4

4. Check for listening ports (no process detected)
   ├─ Port 4002 listening → use 4002 (Gateway paper)
   ├─ Port 4001 listening → use 4001 (Gateway live)
   ├─ Port 7497 listening → use 7497 (TWS paper)
   ├─ Port 7496 listening → use 7496 (TWS live)
   └─ None found → default to 4002
```

## Usage

### START_TRADER.sh (Bash)

The startup script automatically detects and uses the correct port:

```bash
./START_TRADER.sh

# Output shows detected service and port:
# ==========================================
# RoboTrader Startup Script
# ==========================================
# Detected: Gateway Paper (port 4002)
```

### Python runner_async.py

The trading runner automatically detects the port during startup:

```python
# Auto-detection happens in setup()
# Logs show detection result:
# "Auto-detected IBKR port"
# "Detected IBKR service: Gateway Paper on port 4002"
```

### Manual Override

You can override auto-detection using the `IBKR_PORT` environment variable:

```bash
# Force use of TWS paper port
export IBKR_PORT=7497
./START_TRADER.sh

# Or inline
IBKR_PORT=7497 ./START_TRADER.sh
```

### Python API

You can use the detection utility directly in Python:

```python
from robo_trader.utils.ibkr_port_detection import get_ibkr_port, detect_ibkr_service

# Get recommended port
port, reason = get_ibkr_port()
print(f"Using port {port}: {reason}")

# Get detailed service info
service_type, port, reason = detect_ibkr_service()
print(f"Service: {service_type}, Port: {port}")
print(f"Reason: {reason}")

# Environment-based detection
from robo_trader.utils.ibkr_port_detection import get_ibkr_port_for_env

# Get paper trading port
port_paper, reason = get_ibkr_port_for_env(paper_trading=True)

# Get live trading port
port_live, reason = get_ibkr_port_for_env(paper_trading=False)
```

## Benefits

### Before Auto-Detection

**Problem:** Port configuration mismatch
- `START_TRADER.sh` hardcoded port 4002 (Gateway)
- `.env` defaulted to port 7497 (TWS)
- Users had to manually edit config files
- Connection failures due to wrong port

**Symptoms:**
- "Connection timeout" errors
- "Port not open" errors
- Confusion about which port to use

### After Auto-Detection

**Solution:** Automatic detection and selection
- ✅ Detects Gateway or TWS automatically
- ✅ Selects correct port (4002 vs 7497)
- ✅ Works with both paper and live trading
- ✅ Falls back to safe defaults
- ✅ Can be overridden if needed

**Benefits:**
- No manual configuration needed
- Fewer connection errors
- Clear logging of detected service
- Works in different environments

## Troubleshooting

### "No IBKR service detected"

**Cause:** Neither Gateway nor TWS is running, and no IBKR ports are listening.

**Solution:**
1. Start Gateway or TWS
2. Ensure API settings are configured:
   - File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Add 127.0.0.1 to Trusted IPs
   - Set correct socket port

### "Port X detected but connection fails"

**Cause:** Port is listening but API handshake times out.

**Solution:**
1. Check Gateway/TWS API settings (see above)
2. Verify no firewall blocking
3. Check for zombie connections:
   ```bash
   netstat -an | grep <port> | grep CLOSE_WAIT
   ```
4. Run diagnostics:
   ```bash
   python3 diagnose_gateway_api.py
   ```

### "Wrong port detected"

**Cause:** Both Gateway and TWS are running, but wrong one was selected.

**Solution:** Use environment variable to override:
```bash
export IBKR_PORT=7497  # Force TWS
./START_TRADER.sh
```

Or update `.env` file:
```
IBKR_PORT=7497
```

## Implementation Details

### Files Modified

1. **`robo_trader/utils/ibkr_port_detection.py`** (NEW)
   - Python port detection utility
   - Functions: `detect_ibkr_service()`, `get_ibkr_port()`, `get_ibkr_port_for_env()`

2. **`START_TRADER.sh`**
   - Added `detect_ibkr_port()` bash function
   - Auto-detects port before testing connectivity
   - Displays detected service in output

3. **`robo_trader/runner_async.py`**
   - Imports `get_ibkr_port()` in `setup()`
   - Auto-detection if configured port fails
   - Falls back to config port if detection fails
   - Uses detection in reconnection logic

### Detection Preference

The system prefers **Gateway over TWS** when both are running:

```python
if gateway_running:
    # Use Gateway ports (4002 or 4001)
elif tws_running:
    # Use TWS ports (7497 or 7496)
```

This is because Gateway is more commonly used for automated trading.

## Testing

### Test Auto-Detection

```bash
# Check what would be detected
python3 << 'EOF'
from robo_trader.utils.ibkr_port_detection import detect_ibkr_service

service_type, port, reason = detect_ibkr_service()
print(f"Service: {service_type or 'None'}")
print(f"Port: {port or 'None'}")
print(f"Reason: {reason}")
EOF
```

### Test Bash Function

```bash
# Run just the detection part of START_TRADER.sh
source START_TRADER.sh
PORT=$(detect_ibkr_port)
echo "Detected port: $PORT"
```

### Test Different Scenarios

1. **Gateway Paper Running:**
   ```bash
   # Expected: port 4002, service "Gateway Paper"
   ```

2. **TWS Paper Running:**
   ```bash
   # Expected: port 7497, service "TWS Paper"
   ```

3. **Nothing Running:**
   ```bash
   # Expected: port 4002 (default), "No service detected"
   ```

4. **Override with Environment:**
   ```bash
   IBKR_PORT=7497
   # Expected: port 7497, "Using IBKR_PORT from environment"
   ```

## Logging

Auto-detection produces clear log output:

### Success
```
Auto-detected IBKR port
Detected IBKR service: Gateway Paper on port 4002
✓ Port 4002 is open - proceeding to IBKR connect
```

### Fallback
```
Configured port 7497 is not open, attempting auto-detection...
Port detection: No IBKR service detected, defaulting to Gateway paper port 4002
✓ Auto-detected port 4002 is open, using it instead of config port 7497
```

### Failure
```
❌ IBKR PRE-FLIGHT CHECK FAILED
Neither config port 7497 nor detected port 4002 is open
Please ensure TWS or IB Gateway is running and configured properly
```

## See Also

- [ZOMBIE_CONNECTION_CLEANUP.md](ZOMBIE_CONNECTION_CLEANUP.md) - Zombie connection handling
- [FIXES_SUMMARY_2025-10-19.md](FIXES_SUMMARY_2025-10-19.md) - Subprocess IBKR implementation
- `diagnose_gateway_api.py` - Gateway diagnostics tool
- `force_gateway_reconnect.sh` - Gateway connection testing
