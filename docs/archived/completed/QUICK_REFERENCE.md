# RoboTrader Quick Reference Card

## 🚀 Start System
```bash
./START_TRADER.sh
```

## 🛑 Stop System
```bash
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"
```

## 📊 View Dashboard
```
http://localhost:5555
```

## 📝 View Logs
```bash
tail -f robo_trader.log
```

## 🔍 Diagnostics
```bash
# Full diagnostics
python3 diagnose_gateway_api.py

# Test Gateway
./force_gateway_reconnect.sh

# Check zombies
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT
```

## 🔧 Gateway Settings
```
File → Global Configuration → API → Settings
☑️ Enable ActiveX and Socket Clients
Add: 127.0.0.1 to Trusted IPs
Port: 4002 (paper) or 4001 (live)
```

## 📍 Default Symbols
```
AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF
```

## 📚 Documentation
- **CLAUDE.md** - Startup commands
- **README.md** - Quick start
- **STARTUP_SUMMARY.md** - Current status
- **REAL_ISSUE_ANALYSIS.md** - Troubleshooting

## ⚠️ Important
- Always use `START_TRADER.sh` for clean startup
- Never kill Gateway/TWS processes
- Check Gateway API settings if connection fails
- Zombie cleanup is automatic at startup

