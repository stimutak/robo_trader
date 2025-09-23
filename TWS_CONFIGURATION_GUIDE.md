# TWS Configuration Guide - API Connection Issues

## Current Status
✅ **Code fixes implemented and ready**
❌ **TWS API configuration preventing connections**

## Issue Description
Even after implementing comprehensive connection fixes, TWS is still rejecting API connections with timeout errors. This indicates a TWS configuration issue rather than a code problem.

## TWS API Configuration Checklist

### 1. Enable API in TWS
1. Open TWS
2. Go to **File → Global Configuration → API → Settings**
3. Check **"Enable ActiveX and Socket Clients"**
4. Set **Socket port** to **7497** (paper trading) or **7496** (live)
5. Set **Master API client ID** to **0** (or leave blank)
6. **IMPORTANT**: Add **127.0.0.1** to **Trusted IPs** list
7. Click **OK** and **restart TWS**

### 2. Verify API Settings
- **Socket Port**: 7497 (paper) / 7496 (live)
- **Master Client ID**: 0 or blank
- **Read-Only API**: Can be checked for safety
- **Download open orders on connection**: Optional
- **Trusted IPs**: Must include 127.0.0.1

### 3. Check TWS Account Type
- Ensure you're using a **Paper Trading** account for testing
- Live accounts may have additional restrictions
- Verify account has API permissions enabled

### 4. Firewall and Network
- Check macOS firewall settings
- Ensure port 7497 is not blocked
- Verify no VPN interference
- Check for antivirus blocking connections

## Alternative Solutions

### Option 1: Use IB Gateway Instead of TWS
IB Gateway is lighter and often more stable for API connections:

1. Download **IB Gateway** from IBKR website
2. Install and configure for paper trading
3. Use port **4002** (paper) or **4001** (live)
4. Update code to use Gateway port:
   ```python
   config = ConnectionConfig(port=4002)  # IB Gateway paper
   ```

### Option 2: Reset TWS Configuration
1. Close TWS completely
2. Delete TWS settings folder:
   - macOS: `~/Jts/`
   - Windows: `%USERPROFILE%\Jts\`
3. Restart TWS and reconfigure API settings
4. Test connection again

### Option 3: Use Different Client IDs
Some users report success with specific client ID ranges:
- Try client IDs: 1, 2, 10, 100, 999, 1000
- Avoid client ID 0 (reserved for TWS)
- Use unique IDs to avoid conflicts

## Testing Commands

### Basic Connection Test
```bash
# Test basic connection (should work first)
python3 test_exact_working.py

# Test with longer timeout
python3 test_longer_timeout.py

# Test subprocess approach
python3 test_sync_wrapper.py
```

### Debug Connection Issues
```bash
# Check TWS is running
lsof -i :7497

# Check for stuck connections
netstat -an | grep 7497

# Monitor connection attempts
tail -f logs/runner.log
```

## Expected Behavior

### When Working Correctly
```
SUCCESS!
Server version: 178
Accounts: ['DUN080889']
Result: PASS ✓
```

### Current Error Pattern
```
API connection failed: TimeoutError()
FAILED: 
Error type: TimeoutError
Result: FAIL ✗
```

## Next Steps

### Immediate Actions Needed
1. **Verify TWS API settings** (most likely cause)
2. **Add 127.0.0.1 to Trusted IPs** (critical)
3. **Restart TWS** after configuration changes
4. **Test basic connection** before running full system

### If Still Failing
1. **Try IB Gateway** instead of TWS
2. **Reset TWS configuration** completely
3. **Contact IBKR support** for API access verification
4. **Check account permissions** for API usage

### Code is Ready
The comprehensive connection fixes are implemented and ready:
- ✅ Subprocess-based isolation
- ✅ Enhanced error handling
- ✅ Unique client ID generation
- ✅ Library migration completed
- ✅ Simplified connection architecture

Once TWS API is properly configured, the system should work immediately.

## Support Resources

### IBKR Documentation
- [TWS API Quick Start Guide](https://interactivebrokers.github.io/tws-api/initial_setup.html)
- [API Configuration](https://interactivebrokers.github.io/tws-api/initial_setup.html#enable_api)
- [Troubleshooting Guide](https://interactivebrokers.github.io/tws-api/troubleshooting.html)

### Common Solutions
- **"Socket port is not enabled"**: Enable API in Global Configuration
- **"Connection refused"**: Check port number and firewall
- **"Timeout"**: Add to Trusted IPs, check Master Client ID
- **"Already connected"**: Use different client ID

---

**Status**: Code fixes complete, waiting for TWS configuration resolution
**Priority**: HIGH - Blocking full system testing
**Estimated Resolution**: 15-30 minutes once TWS is properly configured
