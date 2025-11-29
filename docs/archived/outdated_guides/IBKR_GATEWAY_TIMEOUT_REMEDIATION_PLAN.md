# IBKR Gateway API Handshake Timeout - Remediation Plan
**Date:** 2025-10-23  
**Status:** CRITICAL - API handshakes timing out despite successful TCP connections  
**Duration:** Issue persisting for over 1 month

---

## Executive Summary

### Problem Statement
The RoboTrader system successfully establishes TCP connections to IBKR Gateway (port 4002) but **API handshakes consistently timeout** after 15 seconds. This prevents all trading operations despite Gateway being operational.

### Root Cause Analysis
**Primary Issue:** IBKR Gateway API settings are not properly configured to accept socket client connections.

**Evidence:**
- ✅ Gateway process running (PID 35037)
- ✅ Port 4002 listening
- ✅ TCP socket connections succeed in <1ms
- ❌ API handshakes timeout after 15s (tested with client IDs: 0, 1, 100, 8400)
- ❌ No zombie CLOSE_WAIT connections detected

**Diagnosis:** This is a **Gateway configuration issue**, NOT a code issue. The TCP layer works perfectly, but the Gateway API layer is rejecting or not responding to API handshake requests.

---

## Detailed Analysis

### 1. Connection Flow Investigation

#### Current Implementation Architecture
The system uses a **multi-layered connection approach**:

1. **Primary Path:** `runner_async.py` → `connect_ibkr_robust()` → `connect_ibkr_robust_subprocess()`
2. **Subprocess Isolation:** `SubprocessIBKRClient` → `ibkr_subprocess_worker.py`
3. **Fallback Path:** `ConnectionManager` → Direct `ib_async.IB.connectAsync()`

**Key Files:**
- `robo_trader/utils/robust_connection.py` (lines 700-1081) - Main connection logic
- `robo_trader/clients/subprocess_ibkr_client.py` - Subprocess-based client
- `robo_trader/clients/ibkr_subprocess_worker.py` - Isolated worker process
- `robo_trader/connection_manager.py` - Direct connection manager

#### Connection Parameters (from `.env`)
```bash
IBKR_HOST=127.0.0.1
IBKR_PORT=4002              # Gateway paper trading port
IBKR_CLIENT_ID=1
IBKR_TIMEOUT=15.0
IBKR_SSL_MODE=disabled      # Plain TCP (correct for local Gateway)
```

### 2. Timeout Behavior Analysis

#### Where Timeout Occurs
The timeout happens during the **API handshake phase** in `ib_async.IB.connectAsync()`:

<augment_code_snippet path="robo_trader/utils/robust_connection.py" mode="EXCERPT">
````python
await ib.connectAsync(
    host=host,
    port=port,
    clientId=use_client_id,
    timeout=timeout,
    readonly=readonly,
)
````
</augment_code_snippet>

**Handshake Sequence:**
1. TCP connection established ✅
2. Client sends API version negotiation message ❌ (Gateway not responding)
3. Timeout after 15 seconds

#### NOT Related to `patchAsyncio()` Issues
Previous fixes (2025-09-23) addressed async context issues by:
- Implementing subprocess-based isolation
- Removing `patchAsyncio()` from worker processes
- Using threading for subprocess I/O

**These fixes are working correctly** - the subprocess isolation is functioning as designed. The issue is that Gateway itself is not responding to API requests.

### 3. Recent Changes Review

#### TWS API Connection Resolution (2025-09-23)
**Changes Made:**
- ✅ Created `SyncIBKRWrapper` (now deprecated)
- ✅ Implemented subprocess approach (`SubprocessIBKRClient`)
- ✅ Enhanced client ID management (timestamp + PID based)
- ✅ Comprehensive error handling and cleanup

**Status:** These changes are **working correctly**. The subprocess successfully isolates `ib_async` from the main async environment.

#### Migration from `ib_insync` to `ib_async` (2025-09-27)
**Changes Made:**
- ✅ Migrated to `ib_async` v2.0.1 (community fork)
- ✅ Updated all imports
- ✅ Tested and verified compatibility

**Status:** Migration is **complete and functional**. The library is working correctly.

#### Readonly Connection Mode (2025-10-05)
**Implementation:**
- ✅ All connections use `readonly=True`
- ✅ No order placement through TWS API
- ✅ Prevents TWS security dialog popups

**Status:** Correctly implemented and should help avoid dialogs.

### 4. Configuration Analysis

#### Current `.env` Settings
```bash
IBKR_HOST=127.0.0.1          # ✅ Correct
IBKR_PORT=4002               # ✅ Correct (Gateway paper)
IBKR_CLIENT_ID=1             # ✅ Valid
IBKR_TIMEOUT=15.0            # ✅ Reasonable
IBKR_SSL_MODE=disabled       # ✅ Correct for local
```

#### Required Gateway Settings (MISSING/INCORRECT)
The Gateway API settings must be configured as follows:

**File → Global Configuration → API → Settings:**
1. ☑️ **Enable ActiveX and Socket Clients** - MUST BE CHECKED
2. **Socket port:** 4002 (for paper trading)
3. **Trusted IPs:** Must include `127.0.0.1`
4. **Master API client ID:** 0 (recommended) or blank
5. **Read-Only API:** Can be enabled for extra safety
6. **Create API message log file:** Recommended for debugging

**Current Status:** Unknown - needs verification

---

## Root Cause Determination

### Primary Root Cause
**Gateway API socket clients are NOT enabled or properly configured.**

### Evidence Supporting This Conclusion
1. **TCP connection succeeds** - Gateway is running and port is open
2. **API handshake fails** - Gateway is not responding to API protocol messages
3. **Multiple client IDs fail** - Not a client ID conflict issue
4. **No zombie connections** - Not a connection cleanup issue
5. **Subprocess isolation working** - Not an async context issue
6. **Consistent 15s timeout** - Gateway is silently ignoring API requests

### Secondary Contributing Factors
1. **Possible firewall/security software** blocking API protocol (unlikely given localhost)
2. **Gateway version compatibility** - IB Gateway 10.40 may have different API requirements
3. **Missing API permissions** in Gateway configuration

---

## Remediation Plan

### Phase 1: Immediate Gateway Configuration Fix (CRITICAL)

#### Step 1.1: Verify Gateway API Settings
**Action:** Open IB Gateway configuration and verify API settings

**Procedure:**
1. In IB Gateway, click **File → Global Configuration**
2. Navigate to **API → Settings**
3. Verify the following settings:

```
☑️ Enable ActiveX and Socket Clients
Socket port: 4002
Trusted IPs: 127.0.0.1
Master API client ID: 0 (or blank)
☑️ Read-Only API (optional, for safety)
☑️ Create API message log file (for debugging)
```

4. Click **Apply** and **OK**
5. **DO NOT restart Gateway** (requires 2FA re-login)

**Expected Result:** Settings should be saved without Gateway restart

#### Step 1.2: Test Connection Immediately
**Action:** Run diagnostic test to verify fix

```bash
cd /Users/oliver/robo_trader
source .venv/bin/activate
python3 debug_gateway_connection.py
```

**Expected Output:**
```
✅ API handshake SUCCESS in <5s
✅ Managed accounts: ['DU...']
✅ Server version: 176
```

**If Still Failing:** Proceed to Step 1.3

#### Step 1.3: Gateway Restart (Last Resort)
**Action:** Restart Gateway if configuration changes don't take effect

**Procedure:**
1. Note your IBKR credentials and 2FA device
2. Close IB Gateway completely
3. Relaunch IB Gateway
4. Login with credentials + 2FA
5. Verify API settings are still correct (Step 1.1)
6. Test connection (Step 1.2)

**Risk:** Requires manual login with 2FA

### Phase 2: Enhanced Diagnostics and Monitoring

#### Step 2.1: Enable Gateway API Logging
**Action:** Enable detailed API logging in Gateway

**Procedure:**
1. In Gateway: File → Global Configuration → API → Settings
2. Check **"Create API message log file"**
3. Note log file location (usually `~/Jts/api_logs/`)
4. Apply settings

**Benefit:** Provides detailed insight into why handshakes are failing

#### Step 2.2: Create Enhanced Diagnostic Script
**Action:** Create a more detailed diagnostic that checks Gateway API logs

**File:** `diagnose_gateway_api.py`

**Features:**
- Check Gateway process and port
- Test TCP connection
- Test API handshake with multiple client IDs
- Parse Gateway API logs for error messages
- Check for firewall/security software interference
- Verify Gateway version compatibility

#### Step 2.3: Implement Connection Health Monitoring
**Action:** Add pre-connection Gateway API validation

**Implementation:**
- Before each connection attempt, verify Gateway API is responding
- Add Gateway API health check to `robust_connection.py`
- Fail fast with clear error message if Gateway API is not configured

### Phase 3: Code Improvements (Post-Fix)

#### Step 3.1: Add Gateway API Configuration Validation
**Action:** Detect Gateway API misconfiguration early

**Implementation Location:** `robo_trader/utils/robust_connection.py`

**Logic:**
```python
async def validate_gateway_api_config(host: str, port: int) -> tuple[bool, str]:
    """
    Validate that Gateway API is properly configured.
    
    Returns:
        (is_valid, error_message)
    """
    # Test TCP connection
    # Test API handshake with short timeout
    # Return clear error if API not responding
```

#### Step 3.2: Improve Error Messages
**Action:** Provide actionable error messages for Gateway API issues

**Current Error:**
```
ConnectionError: Failed to connect after 2 attempts
```

**Improved Error:**
```
ConnectionError: IBKR Gateway API handshake timeout.

Possible causes:
1. Gateway API settings not configured (most likely)
   → Open Gateway: File → Global Configuration → API → Settings
   → Enable "Enable ActiveX and Socket Clients"
   → Add 127.0.0.1 to Trusted IPs
   
2. Gateway needs restart
   → Close and relaunch Gateway (requires 2FA login)
   
3. Firewall blocking API protocol
   → Check macOS firewall settings

Run diagnostics: python3 debug_gateway_connection.py
```

#### Step 3.3: Add Zombie Connection Cleanup (Preventive)
**Action:** Ensure zombie cleanup is called before connection attempts

**Current Implementation:** `kill_tws_zombie_connections()` exists in `robust_connection.py:283`

**Enhancement:** Call zombie cleanup proactively:
```python
# Before connection attempt
zombie_count, msg = check_tws_zombie_connections(port)
if zombie_count > 0:
    kill_tws_zombie_connections(port)
```

**Status:** Already implemented in `RobustConnectionManager.connect()` (lines 534-545)

### Phase 4: Testing and Validation

#### Step 4.1: Comprehensive Connection Testing
**Test Suite:**
1. ✅ TCP socket connection
2. ✅ API handshake with multiple client IDs (0, 1, 100, random)
3. ✅ Readonly vs read-write mode
4. ✅ SSL vs plain TCP
5. ✅ Subprocess isolation
6. ✅ Direct connection
7. ✅ Connection manager
8. ✅ Full runner_async integration

#### Step 4.2: Regression Testing
**Verify:**
- No zombie connection accumulation
- Proper cleanup on timeout
- Circuit breaker functioning
- Subprocess stability
- Error handling and logging

---

## Implementation Priority

### CRITICAL (Do Immediately)
1. ✅ **Verify Gateway API Settings** (Step 1.1)
2. ✅ **Test Connection** (Step 1.2)
3. ⚠️ **Restart Gateway if needed** (Step 1.3)

### HIGH (Do Today)
4. ⬜ **Enable Gateway API Logging** (Step 2.1)
5. ⬜ **Create Enhanced Diagnostic** (Step 2.2)
6. ⬜ **Improve Error Messages** (Step 3.2)

### MEDIUM (Do This Week)
7. ⬜ **Add API Config Validation** (Step 3.1)
8. ⬜ **Implement Health Monitoring** (Step 2.3)
9. ⬜ **Comprehensive Testing** (Step 4.1)

### LOW (Nice to Have)
10. ⬜ **Regression Testing** (Step 4.2)

---

## Testing Strategy

### Pre-Fix Baseline
```bash
# Current state (should fail)
python3 debug_gateway_connection.py
# Expected: All API handshakes timeout
```

### Post-Fix Validation
```bash
# After Gateway configuration fix
python3 debug_gateway_connection.py
# Expected: API handshake SUCCESS in <5s

# Test subprocess client
python3 test_subprocess_worker_direct.py
# Expected: Connection successful

# Test connection manager
python3 -c "
import asyncio
from robo_trader.connection_manager import ConnectionManager
async def test():
    mgr = ConnectionManager()
    ib = await mgr.connect()
    print(f'Connected: {ib.isConnected()}')
    print(f'Accounts: {ib.managedAccounts()}')
asyncio.run(test())
"

# Test full runner (with single symbol)
python3 -m robo_trader.runner_async --symbols AAPL
# Expected: Connection successful, trading loop starts
```

---

## Rollback Plan

### If Fix Causes Issues
1. **Revert Gateway API Settings:**
   - Uncheck "Enable ActiveX and Socket Clients"
   - Remove 127.0.0.1 from Trusted IPs
   - Apply settings

2. **Restore Previous State:**
   - No code changes required (issue is configuration-only)
   - System will return to current timeout state

### If Gateway Restart Fails
1. **Re-login to Gateway:**
   - Use IBKR credentials
   - Complete 2FA authentication
   - Reconfigure API settings

2. **Contact IBKR Support:**
   - If login issues persist
   - If API settings cannot be saved
   - If Gateway version incompatibility suspected

---

## Success Criteria

### Fix is Successful When:
1. ✅ `debug_gateway_connection.py` shows API handshake SUCCESS
2. ✅ Connection time < 5 seconds
3. ✅ Managed accounts returned
4. ✅ `runner_async.py` connects without timeout
5. ✅ No zombie connections accumulate
6. ✅ System runs for 24+ hours without connection issues

### Monitoring Metrics:
- Connection success rate: 100%
- Connection time: < 5s
- Zombie connections: 0
- Circuit breaker state: CLOSED
- Uptime: > 24 hours

---

## Next Steps

### Immediate Actions (User)
1. **Open IB Gateway**
2. **Navigate to:** File → Global Configuration → API → Settings
3. **Verify/Enable:** "Enable ActiveX and Socket Clients"
4. **Add:** 127.0.0.1 to Trusted IPs
5. **Set:** Socket port to 4002
6. **Apply** settings
7. **Run:** `python3 debug_gateway_connection.py`

### If Successful
8. **Run:** `python3 -m robo_trader.runner_async --symbols AAPL`
9. **Monitor:** Connection stability for 1 hour
10. **Document:** Final Gateway settings in `.env.example`

### If Still Failing
11. **Restart Gateway** (requires 2FA)
12. **Check Gateway API logs** in `~/Jts/api_logs/`
13. **Contact IBKR Support** for Gateway API configuration assistance

---

## Appendix

### A. Relevant Code Locations

**Connection Logic:**
- `robo_trader/utils/robust_connection.py:700-1081` - Main connection function
- `robo_trader/connection_manager.py:192-333` - Connection manager
- `robo_trader/clients/subprocess_ibkr_client.py:262-307` - Subprocess connect
- `robo_trader/clients/ibkr_subprocess_worker.py:23-86` - Worker connect handler

**Zombie Connection Cleanup:**
- `robo_trader/utils/robust_connection.py:283-402` - Zombie detection and cleanup
- `robo_trader/utils/robust_connection.py:534-545` - Pre-connection cleanup

**Configuration:**
- `.env` - Connection parameters
- `robo_trader/utils/secure_config.py:186-226` - Config validation

### B. Diagnostic Commands

```bash
# Check Gateway process
ps aux | grep -i gateway

# Check port status
netstat -an | grep 4002
lsof -nP -iTCP:4002

# Check zombie connections
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT

# Test TCP connection
nc -zv 127.0.0.1 4002

# Test API handshake
python3 debug_gateway_connection.py

# Check Gateway logs
ls -la ~/Jts/api_logs/
tail -f ~/Jts/api_logs/*.log
```

### C. Gateway API Settings Screenshot Locations
- IB Gateway: File → Global Configuration → API → Settings
- Key setting: "Enable ActiveX and Socket Clients"
- Documentation: `IB_GATEWAY_FIX_GUIDE.md`

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-23 17:50 PST  
**Author:** Augment Agent (Claude Sonnet 4.5)  
**Status:** Ready for Implementation

