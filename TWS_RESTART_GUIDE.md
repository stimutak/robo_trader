# TWS Restart Guide - Fix Connection Issues

## Current Problem
TWS has stuck connections preventing new API connections from completing the handshake.

## Evidence
```bash
# Stuck connections in CLOSE_WAIT state:
tcp4  127.0.0.1.7497  127.0.0.1.57324  CLOSE_WAIT 
tcp4  127.0.0.1.7497  127.0.0.1.57276  CLOSE_WAIT 
tcp4  127.0.0.1.7497  127.0.0.1.57268  CLOSE_WAIT 

# TWS Process:
JavaAppli 38912 oliver  /Users/oliver/Applications/Trader Workstation 10.39/
```

## Solution Steps

### 1. Restart TWS (Recommended)
```bash
# Kill TWS process
kill 38912

# Or force kill if needed
kill -9 38912

# Then restart TWS manually from Applications
```

### 2. Alternative: Try IB Gateway
If TWS continues to have issues, try IB Gateway instead:
- Download IB Gateway from IBKR website
- Configure for paper trading (port 4002) or live (port 4001)
- Update runner to use Gateway port

### 3. Verify Fix
After restarting TWS, test with:
```bash
python3 test_exact_working.py
```

Should see:
```
SUCCESS!
Server version: 176
Accounts: ['DU123456']
```

### 4. Test Runner
Once basic connection works:
```bash
python3 test_runner_connection.py
```

### 5. Full Runner Test
```bash
python3 -m robo_trader.runner_async --symbols AAPL
```

## Code Improvements Made

✅ **Simplified Connection Architecture**
- Removed complex connection pooling
- Direct connection approach
- Robust retry logic

✅ **Better Client ID Management**  
- Unique ID generation (timestamp + PID)
- Safe ID ranges (10000-99999)
- Conflict avoidance

✅ **Enhanced Error Handling**
- Proper cleanup of failed connections
- Async disconnect handling
- Comprehensive logging

✅ **Library Migration**
- Upgraded to ib_async v2.0.1
- Maintained backward compatibility

## Expected Outcome
After TWS restart, the runner should connect successfully and process symbols without the timeout issues.
