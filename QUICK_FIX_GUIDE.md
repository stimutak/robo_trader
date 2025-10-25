# IBKR Gateway API Handshake Timeout - Quick Fix Guide

**Problem:** API handshakes timing out after 15 seconds despite successful TCP connections.

**Root Cause:** IBKR Gateway API settings not properly configured.

---

## Quick Fix (5 Minutes)

### Step 1: Open Gateway Configuration
1. In IB Gateway, click **File → Global Configuration**
2. Navigate to **API → Settings**

### Step 2: Enable API Socket Clients
Check the following settings:

```
☑️ Enable ActiveX and Socket Clients    ← CRITICAL!
Socket port: 4002                        ← For paper trading
Trusted IPs: 127.0.0.1                   ← Add if not present
Master API client ID: 0                  ← Or leave blank
☑️ Read-Only API                         ← Optional (recommended)
☑️ Create API message log file           ← Optional (for debugging)
```

### Step 3: Apply Settings
1. Click **Apply**
2. Click **OK**
3. **DO NOT restart Gateway** (settings apply immediately)

### Step 4: Test Connection
```bash
cd /Users/oliver/robo_trader
source .venv/bin/activate
python3 diagnose_gateway_api.py
```

**Expected Output:**
```
✅ Gateway Process
✅ Port Listening
✅ TCP Connection
✅ API Handshake
```

### Step 5: Start Trading System
```bash
python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA
```

---

## If Still Failing

### Option 1: Restart Gateway
1. Close IB Gateway completely
2. Relaunch IB Gateway
3. Login with credentials + 2FA
4. Verify API settings (Step 2)
5. Test again (Step 4)

### Option 2: Check API Logs
```bash
# View Gateway API logs
ls -la ~/Jts/api_logs/
tail -f ~/Jts/api_logs/*.log
```

Look for error messages related to:
- Client connection attempts
- Authentication failures
- API permission issues

### Option 3: Contact IBKR Support
If Gateway API settings cannot be saved or API still not responding:
- Call IBKR support: 1-877-442-2757
- Mention: "API socket clients not accepting connections"
- Provide: Gateway version (10.40), port (4002), error (handshake timeout)

---

## Verification Checklist

- [ ] Gateway process running
- [ ] Port 4002 listening
- [ ] TCP connection succeeds
- [ ] "Enable ActiveX and Socket Clients" is checked
- [ ] 127.0.0.1 in Trusted IPs
- [ ] Socket port set to 4002
- [ ] API handshake succeeds in < 5 seconds
- [ ] Managed accounts returned
- [ ] Trading system connects successfully

---

## Common Mistakes

❌ **Forgetting to click "Apply"** - Settings won't save  
❌ **Using wrong port** - 4002 for paper, 4001 for live  
❌ **Not adding 127.0.0.1 to Trusted IPs** - Connection will be rejected  
❌ **Restarting Gateway unnecessarily** - Settings apply without restart  

---

## Additional Resources

- **Full Remediation Plan:** `IBKR_GATEWAY_TIMEOUT_REMEDIATION_PLAN.md`
- **Enhanced Diagnostics:** `python3 diagnose_gateway_api.py`
- **Original Diagnostic:** `python3 debug_gateway_connection.py`
- **IB Gateway Guide:** `IB_GATEWAY_FIX_GUIDE.md`

---

**Last Updated:** 2025-10-23  
**Status:** Ready for Implementation

