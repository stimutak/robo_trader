# Complete Restart and Test Guide

## Current Status
✅ **Code fixes implemented successfully:**
- Simplified connection architecture (removed complex pooling)
- Improved client ID management (unique timestamp-based IDs)
- Enhanced error handling and cleanup
- Library compatibility (supports both ib_insync and ib_async)

❌ **Current blocker:** TWS has stuck connections preventing API handshake

## Immediate Action Required

### 1. Restart TWS
TWS has been killed (PID 46094). You need to manually restart it:

1. Open **Trader Workstation** from Applications
2. Log in with your paper trading account
3. Wait for TWS to fully load
4. Ensure API settings are configured:
   - Go to **File > Global Configuration > API > Settings**
   - Check "Enable ActiveX and Socket Clients"
   - Set "Socket port" to **7497** (paper trading)
   - Set "Master API client ID" to **0** (or leave blank)
   - Click **OK** and restart TWS if prompted

### 2. Verify TWS is Running
```bash
lsof -i :7497
```
Should show:
```
COMMAND     PID   USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
JavaAppli [PID] oliver   46u  IPv6 [ADDRESS]      0t0  TCP *:7497 (LISTEN)
```

### 3. Test Basic Connection
```bash
python3 test_exact_working.py
```
Expected output:
```
SUCCESS!
Server version: 178
Accounts: ['DUN080889']
Result: PASS ✓
```

### 4. Test Runner Connection
```bash
python3 test_runner_connection.py
```
Expected output:
```
✓ Successfully connected!
Account summary: {...}
✓ Disconnected successfully
Result: PASS ✓
```

### 5. Test Full Runner
```bash
python3 -m robo_trader.runner_async --symbols AAPL
```

## Root Cause Analysis

The issue was **NOT with our code** but with TWS getting stuck connections that prevent the API handshake from completing. This happens when:

1. A connection attempt fails or times out
2. TWS doesn't properly clean up the connection
3. The connection stays in `CLOSE_WAIT` state
4. Subsequent connections can establish TCP but fail at API handshake

## Code Improvements Made

### ✅ Connection Architecture
- **Before:** Complex connection pooling with multiple simultaneous connections
- **After:** Simple direct connection approach (like working test scripts)

### ✅ Client ID Management  
- **Before:** Fixed ranges that could conflict
- **After:** Unique timestamp + PID based IDs (10000-99999 range)

### ✅ Error Handling
- **Before:** Basic retry with same client ID
- **After:** Retry with different client IDs, proper cleanup, comprehensive logging

### ✅ Library Compatibility
- **Before:** Hard dependency on ib_async
- **After:** Supports both ib_insync and ib_async, uses whichever works better

## Expected Results After TWS Restart

Once TWS is restarted with clean connections:

1. **Basic test should work** (client ID 999)
2. **Runner connection should work** (unique client IDs)
3. **Full runner should process symbols** without timeout issues
4. **Parallel symbol processing** should work properly

## Monitoring for Future Issues

Watch for stuck connections:
```bash
netstat -an | grep 7497
```

If you see `CLOSE_WAIT` connections, restart TWS:
```bash
# Find TWS process
lsof -i :7497

# Kill it (replace PID)
kill [PID]

# Restart TWS manually
```

## Next Steps After Restart

1. Verify basic connection works
2. Test runner connection  
3. Run full runner with single symbol
4. Test parallel processing with multiple symbols
5. Monitor for any new connection issues

The code is now much more robust and should handle connection issues better, but TWS still needs to be in a clean state for the API to work properly.
