# Robo Trader Startup Guide

## CRITICAL: Virtual Environment Required (Updated 2025-09-25)
**ALWAYS use `.venv` virtual environment** - especially after macOS upgrades!

## Quick Start - Current Working Method

### Complete System Startup
```bash
# 1. Navigate to project
cd /Users/oliver/robo_trader

# 2. ACTIVATE VIRTUAL ENVIRONMENT (REQUIRED!)
source .venv/bin/activate

# 3. Kill any existing processes
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"

# 4. Start WebSocket server (MUST BE FIRST)
python3 -m robo_trader.websocket_server &

# 5. Start trading runner
export LOG_FILE=/Users/oliver/robo_trader/robo_trader.log
python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF,QQQ,QLD,BBIO,IMRX,CRGY &

# 6. Start dashboard (ALWAYS port 5555)
export DASH_PORT=5555
python3 app.py &
```

**Dashboard URL: http://localhost:5555**

### Legacy Script Method (May Need Updates)
```bash
# Start everything at once
./start_all.sh

# Or start components individually
./start_dashboard.sh  # In one terminal
./start_trading.sh    # In another terminal

# Stop everything
./stop_all.sh
```

### Automatic Startup on macOS

To have the trading system start automatically:

1. **Install the LaunchAgent** (runs when you log in):
```bash
# Copy the plist file to LaunchAgents
cp com.robotrader.trading.plist ~/Library/LaunchAgents/

# Load the service
launchctl load ~/Library/LaunchAgents/com.robotrader.trading.plist

# Start immediately (optional)
launchctl start com.robotrader.trading
```

2. **Check status**:
```bash
launchctl list | grep robotrader
```

3. **Stop/Disable automatic startup**:
```bash
# Stop the service
launchctl stop com.robotrader.trading

# Unload (disable automatic startup)
launchctl unload ~/Library/LaunchAgents/com.robotrader.trading.plist
```

4. **View logs**:
```bash
tail -f launchd_trading.log
tail -f launchd_trading_error.log
```

## Prerequisites

1. **IB Gateway or TWS must be running**
   - Paper trading: Port 7497
   - Live trading: Port 7496
   - API must be enabled in settings
   - May need periodic restart to clear stuck connections

2. **Virtual environment must be set up (.venv not venv)**:
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# CRITICAL: Install ib_async for IBKR connection
pip install ib_async
```

**Note**: After macOS upgrades, Python paths may reset. Always activate `.venv`!

## Scripts Overview

- `start_all.sh` - Starts both dashboard and trading system
- `start_trading.sh` - Starts only the trading system
- `start_dashboard.sh` - Starts only the dashboard
- `stop_all.sh` - Stops all components
- `com.robotrader.trading.plist` - macOS LaunchAgent for automatic startup

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'" or 'ib_async'
**Common after macOS upgrades!**
1. Not using virtual environment: `source .venv/bin/activate`
2. If persists: `pip install pandas ib_async`

### Trading system not connecting to IB
1. Ensure IB Gateway/TWS is running
2. Check API is enabled in IB settings
3. Verify port number (7497 for paper, 7496 for live)
4. **Restart TWS to clear stuck connections** (common issue)
5. Check for CLOSE_WAIT connections: `netstat -an | grep 7497 | grep CLOSE_WAIT`
6. Check logs: `tail -f /Users/oliver/robo_trader/robo_trader.log`

### Scripts not executing
```bash
# Make scripts executable
chmod +x *.sh
```

### Virtual environment issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Port already in use (Dashboard)
```bash
# Find and kill process using port 5555
lsof -i :5555
kill -9 <PID>
```

## Configuration

Edit `.env` file to configure:
- IB connection settings
- Trading parameters
- Risk limits
- Symbol lists

## Monitoring

- Dashboard: http://localhost:5555
- Trading logs: `tail -f ai_trading.log`
- Dashboard logs: `tail -f dashboard.log`
- Database: `trading_data.db`