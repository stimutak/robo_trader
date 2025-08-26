# Robo Trader Startup Guide

## Quick Start

### Manual Start (Recommended for Testing)
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

2. **Virtual environment must be set up**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Scripts Overview

- `start_all.sh` - Starts both dashboard and trading system
- `start_trading.sh` - Starts only the trading system
- `start_dashboard.sh` - Starts only the dashboard
- `stop_all.sh` - Stops all components
- `com.robotrader.trading.plist` - macOS LaunchAgent for automatic startup

## Troubleshooting

### Trading system not connecting to IB
1. Ensure IB Gateway/TWS is running
2. Check API is enabled in IB settings
3. Verify port number (7497 for paper, 7496 for live)
4. Check logs: `tail -f ai_trading.log`

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