# RoboTrader Startup Guide

**Last Updated: 2025-11-27**

---

## ⚠️ CRITICAL WARNING - READ FIRST

**DO NOT run `test_gateway_connection_fix.py` immediately before starting the trader!**

Running the test script creates brief Gateway connections that, even with proper cleanup, can leave zombie connections that block trader startup. If you run the test and then immediately start the trader:

1. The test creates a connection to Gateway
2. Even with `safe_disconnect()`, Gateway may retain the connection state briefly
3. This creates a CLOSE_WAIT zombie that blocks the trader's connection
4. **Result: Trader fails to connect**

**If you need to test Gateway connectivity:**
- Use `./force_gateway_reconnect.sh` instead (quick test, minimal zombie risk)
- OR wait 30+ seconds after running `test_gateway_connection_fix.py`
- OR restart Gateway after testing (requires 2FA login)

**The safe workflow is:**
1. Check for zombies: `lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT`
2. Start directly: `./START_TRADER.sh`
3. Only run diagnostic tests if startup fails

---

## Pre-Flight Checklist

Before starting the trader, verify:

1. **IB Gateway is running** and you're logged in (requires 2FA)
2. **No zombie connections** exist (see Troubleshooting below)
3. **Virtual environment** is available at `.venv/`

---

## Quick Start (Recommended)

```bash
cd /home/user/robo_trader
./START_TRADER.sh
```

Or with custom symbols:
```bash
./START_TRADER.sh "AAPL,NVDA,TSLA,QQQ"
```

The script automatically:
- Kills existing Python trader processes
- Cleans up zombie CLOSE_WAIT connections
- Starts WebSocket server
- Starts trading system
- Starts dashboard on port 5555

---

## If Gateway Connection Fails

### Step 1: Check for Zombies
```bash
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT
```

If you see any output, there are zombie connections blocking the API.

### Step 2: Kill Python Zombies (Safe)
```bash
lsof -ti tcp:4002 -sTCP:CLOSE_WAIT | xargs kill -9
```

### Step 3: If Zombies Remain → Restart Gateway

Gateway-owned zombies **cannot** be killed without restarting Gateway:

1. **File → Exit** in IB Gateway (don't just close the window)
2. Wait 5 seconds
3. Relaunch IB Gateway
4. Login with 2FA
5. Try `./START_TRADER.sh` again

---

## Manual Startup (Advanced)

```bash
# 1. Navigate to project
cd /home/user/robo_trader

# 2. ACTIVATE VIRTUAL ENVIRONMENT (REQUIRED!)
source .venv/bin/activate

# 3. Kill existing processes
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"

# 4. Start WebSocket server (MUST BE FIRST)
python3 -m robo_trader.websocket_server &

# 5. Start trading system
export LOG_FILE=/home/user/robo_trader/robo_trader.log
python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA &

# 6. Start dashboard (ALWAYS port 5555)
export DASH_PORT=5555
python3 app.py &
```

**Dashboard URL: http://localhost:5555**

---

## Verify Connection Works (Troubleshooting Only)

⚠️ **DO NOT run this before starting the trader** - it creates zombies!

Only use this test script **after** startup fails to diagnose issues:
```bash
python3 test_gateway_connection_fix.py
```

Expected output: All 3 tests pass (Direct, Worker, Client)

**IMPORTANT:** After running this test, you MUST either:
- Wait 30+ seconds before starting the trader, OR
- Restart Gateway (File → Exit, relaunch with 2FA)

For quick connectivity checks without zombie risk, use:
```bash
./force_gateway_reconnect.sh
```

---

## Troubleshooting

### "API handshake timeout"
- Gateway is running but not responding to API protocol
- **Solution:** Restart Gateway (File → Exit, relaunch, 2FA)

### "No managed accounts"
- Handshake succeeded but account data not received
- **Solution:** Check Gateway API permissions, restart Gateway

### "Zombie connections detected"
- Old failed connections blocking new ones
- **Solution:** Kill zombies or restart Gateway (see above)

### Connection works in test but not in runner
- Multiple processes trying to connect simultaneously
- **Solution:** Kill all Python processes and start fresh:
```bash
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"
```

### "ModuleNotFoundError: No module named 'pandas'" or 'ib_async'
- Not using virtual environment
- **Solution:** `source .venv/bin/activate` then retry

---

## DO NOT

- ❌ Kill Gateway/TWS processes (`pkill -f tws` or `pkill -f gateway`)
- ❌ Use `lsof -ti:4002 | xargs kill` (kills Gateway!)
- ❌ Run multiple instances of `runner_async`
- ❌ Skip the virtual environment activation
- ❌ Start trader without checking for zombies first

---

## Diagnostic Commands

```bash
# Check Gateway process
ps aux | grep -i gateway

# Check what's on port 4002
lsof -nP -iTCP:4002

# Check for zombies specifically
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT

# View trader logs
tail -f robo_trader.log

# Full diagnostics
python3 diagnose_gateway_api.py

# Test connection only
python3 test_gateway_connection_fix.py
```

---

## Port Reference

| Port | Purpose |
|------|---------|
| 4002 | IB Gateway Paper Trading API |
| 4001 | IB Gateway Live Trading API |
| 7497 | TWS Paper Trading API |
| 7496 | TWS Live Trading API |
| 5555 | Dashboard web interface |

---

## Configuration

Edit `.env` file to configure:
- IB connection settings (host, port, client_id)
- Trading parameters
- Risk limits
- Symbol lists

### Key Risk Settings

```bash
# Trailing Stops (Recommended - lets winners run!)
USE_TRAILING_STOP=true          # Enable trailing stops
TRAILING_STOP_PERCENT=5.0       # 5% below high water mark

# Fixed Stops (Alternative)
USE_TRAILING_STOP=false         # Disable trailing
STOP_LOSS_PERCENT=2.0           # Fixed 2% stop
```

Default symbols are in `user_settings.json`

---

## Stopping the System

```bash
# Stop all trader processes (safe - does NOT touch Gateway)
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"
```

**NEVER** kill Gateway processes - they require manual 2FA login to restart.
