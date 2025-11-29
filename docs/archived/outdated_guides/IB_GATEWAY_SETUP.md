# IB Gateway Setup Guide

## Quick Setup Steps

### 1. Launch IB Gateway
- Open IB Gateway application
- Select "IB API" (not "TWS")
- Log in with your IBKR credentials

### 2. Configure API Settings
**CRITICAL**: Enable API access in IB Gateway
1. Click "Configure" → "Settings" (or File → Global Configuration)
2. Go to "API" → "Settings"
3. Check these boxes:
   - ✅ **Enable ActiveX and Socket Clients**
   - ✅ **Download open orders on connection**
   - ✅ **Include FX positions when sending portfolio**
   - ✅ **Send instrument-specific account value**
4. Socket port: **4002** (live) or **4001** (paper)
5. Master API client ID: Leave blank
6. **IMPORTANT**: Add "127.0.0.1" to "Trusted IP Addresses" (create if needed)
7. Uncheck "Read-Only API" if you want to place orders
8. Click "Apply" and "OK"

### 3. Connection Ports
- **Paper Trading**: Port 4001 (IB Gateway) or 7497 (TWS)
- **Live Trading**: Port 4002 (IB Gateway) or 7496 (TWS)

### 4. Update RoboTrader Configuration
The system will automatically use the correct port based on what's running:
- Port 7497 for TWS Paper
- Port 4001 for IB Gateway Paper
- Port 4002 for IB Gateway Live

### 5. Test Connection
```bash
# Test IB Gateway connection
python3 -c "
from ib_insync import IB
ib = IB()
try:
    ib.connect('127.0.0.1', 4001, clientId=999)  # Paper trading port
    print('✅ Connected to IB Gateway!')
    print(f'Server version: {ib.client.serverVersion()}')
    ib.disconnect()
except Exception as e:
    print(f'❌ Connection failed: {e}')
"
```

### 6. Start RoboTrader
Once IB Gateway is running and configured:
```bash
# Kill any existing processes
pkill -9 -f "runner_async"

# Start the trading system
export LOG_FILE=/Users/oliver/robo_trader/robo_trader.log
python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA,QQQ
```

## Troubleshooting

### Common Issues:
1. **"API connection failed: TimeoutError()"**
   - Ensure API is enabled in Gateway settings
   - Check that 127.0.0.1 is in Trusted IP list
   - Verify correct port (4001 for paper, 4002 for live)

2. **"No security definition found"**
   - Make sure you're subscribed to market data
   - Check symbol is correct (use SMART exchange)

3. **Connection drops frequently**
   - IB Gateway is more stable than TWS for long-running connections
   - Consider setting auto-restart in Gateway settings

### Gateway vs TWS
- **IB Gateway**: Lightweight, headless, more stable for automation
- **TWS**: Full GUI, good for monitoring but less stable for 24/7 operation

### Auto-Login Setup (Optional)
You can configure IB Gateway to auto-login:
1. File → Global Configuration → Lock and Exit
2. Set up auto-login credentials
3. Enable "Auto logon at start"