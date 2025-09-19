# IBKR Setup Requirements - CRITICAL

**THIS MUST NOT HAPPEN AGAIN: Trading system running without IBKR connection**

## Prerequisites

1. **Interactive Brokers Account**
   - Paper trading account recommended for testing
   - Live account for production trading
   - Account must be funded and active

2. **TWS or IB Gateway Installation**
   - Download TWS (Trader Workstation) or IB Gateway from Interactive Brokers
   - IB Gateway recommended for automated trading (lighter weight, no GUI)
   - TWS required if you want to monitor trades visually

## Critical TWS/IB Gateway Configuration

### API Settings (MUST BE ENABLED)

1. **Open TWS/IB Gateway**
2. **Navigate to Configuration:**
   - TWS: `File → Global Configuration → API → Settings`
   - IB Gateway: `Configure → Settings → API → Settings`

3. **Required Settings:**
   - ✅ **Enable ActiveX and Socket Clients** - MUST be checked
   - ✅ **Socket port:** 
     - Paper trading: `7497` (default)
     - Live trading: `7496` 
   - ✅ **Allow connections from localhost only** - For security
   - ✅ **Create API message log file** - For debugging
   - ❌ **Read-Only API** - Must be UNCHECKED for trading

4. **Master API Client ID:**
   - Leave blank or set to specific ID
   - System will use random IDs (100-999) to avoid conflicts

5. **Trusted IP Addresses:**
   - Add `127.0.0.1` to whitelist
   - This prevents connection prompts

## Daily Startup Checklist

### Before Market Open:

1. **Start TWS/IB Gateway**
   ```bash
   # Check if running on correct port
   lsof -i:7497  # For paper trading
   lsof -i:7496  # For live trading
   ```

2. **Test IBKR Connection**
   ```bash
   python3 test_ibkr_connection.py
   # MUST show "✅ IBKR CONNECTION TEST PASSED"
   ```

3. **Start Connection Monitor**
   ```bash
   python3 ibkr_connection_monitor.py &
   # Monitors connection every 30 seconds
   ```

4. **Start Trading System**
   ```bash
   # WebSocket server first
   python3 -m robo_trader.websocket_server &
   
   # Then trading runner (will fail if IBKR not connected)
   export LOG_FILE=/Users/oliver/robo_trader/robo_trader.log
   python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA
   ```

## Connection Safeguards Implemented

### 1. Pre-flight Check (runner_async.py)
- System will NOT start without verified IBKR connection
- Checks port availability
- Verifies API connection
- Tests account access
- Exit code 1 if connection fails

### 2. Connection Test Script (test_ibkr_connection.py)
- Standalone connection verification
- Tests:
  - Port open
  - API connection
  - Account access
  - Market data retrieval
- Run before starting trading

### 3. Connection Monitor (ibkr_connection_monitor.py)
- Continuous monitoring every 30 seconds
- Creates alert file if connection lost
- Attempts automatic restart after failures
- Logs all connection events
- Status file at `/tmp/ibkr_monitor_status.json`

## Common Issues and Solutions

### Issue: "Port 7497 is not open"
**Solution:** TWS/IB Gateway not running. Start the application.

### Issue: "Connection timeout after 30 seconds"
**Solutions:**
1. Check API settings are enabled
2. Restart TWS/IB Gateway
3. Check firewall settings
4. Verify correct port (7497 for paper, 7496 for live)

### Issue: "No managed accounts found"
**Solution:** Login to TWS/IB Gateway with your credentials

### Issue: "Client ID already in use"
**Solution:** Another process using same ID. System uses random IDs to avoid.

## Emergency Recovery Procedure

If connection is lost during trading:

1. **Check TWS/IB Gateway**
   - Is it running?
   - Are you logged in?
   - Check for error messages

2. **Run Connection Test**
   ```bash
   python3 test_ibkr_connection.py
   ```

3. **If Test Fails:**
   - Restart TWS/IB Gateway
   - Re-enable API settings
   - Clear any popup dialogs
   - Re-run connection test

4. **Restart Trading System:**
   ```bash
   # Kill all processes
   pkill -f "runner_async"
   
   # Restart with pre-flight check
   python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA
   ```

## Monitoring Commands

```bash
# Check if TWS is running
ps aux | grep -i "tws\|ibgateway"

# Check port availability
lsof -i:7497

# View connection monitor status
cat /tmp/ibkr_monitor_status.json

# Check for alerts
ls -la /tmp/IBKR_CONNECTION_CRITICAL.txt

# View monitor logs
tail -f /Users/oliver/robo_trader/ibkr_monitor.log
```

## Production Deployment

For production, run connection monitor as a service:

```bash
# Create systemd service or launchd plist
# Ensures monitor restarts if it crashes
python3 ibkr_connection_monitor.py --daemon
```

## Critical Rules

1. **NEVER** start trading without passing connection test
2. **ALWAYS** run connection monitor during trading hours
3. **IMMEDIATELY** investigate any connection alerts
4. **TEST** connection recovery procedures regularly
5. **DOCUMENT** any new connection issues discovered

## Support Resources

- IB API Documentation: https://interactivebrokers.github.io/
- TWS API Settings Guide: https://interactivebrokers.com/en/software/tws/twsguide.htm
- Connection Issues: https://interactivebrokers.com/en/software/api/api_trouble.htm

---

**Remember: NO TRADING WITHOUT VERIFIED IBKR CONNECTION**

The system now has multiple safeguards:
- Pre-flight checks prevent startup without connection
- Connection monitor alerts on failures
- Automatic recovery attempts
- Clear error messages and solutions

This documentation ensures the "no signals" issue NEVER happens again.