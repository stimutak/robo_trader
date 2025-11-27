# Gateway API Handshake Timeout - Immediate Next Steps

**Status (2025-11-24 evening):** Gateway connectivity verified, official `ibapi` handshake succeeds instantly, robo_trader subprocess client still failing intermittently
**Current Issue:** Subprocess worker bails before IBKR pushes `nextValidId`/`managedAccounts`
**Confirmed Non-Causes:** IBKR account permissions, Gateway API configuration, TCP connectivity
**Likely Contributors:** Gateway 10.40/10.41 regression plus worker synchronization gaps

## 🚀 Forward Plan

### 1. Baseline & Instrumentation (15 min)

- Activate `.venv`, ensure Gateway is running on port 4002 (paper)
- Run the official ibapi probe (see `docs/diag/ibapi_handshake_probe.py`) to capture timestamps for `connectAck`, `managedAccounts`, and `nextValidId`
- Run `python3 diagnose_gateway_internal_state.py` and save output; this is the “known good” baseline to compare after each change

### 2. Subprocess Worker Synchronization (high priority)

- Add explicit event-based waits in `ibkr_subprocess_worker.py` so the worker does not return until it has received one of:
  - A non-empty `ib.managedAccounts()`
  - `ib.nextValidOrderId` callback (via `ib.nextValidIdEvent` in ib_async) or `ib.accountValuesEvent`
- Use `ib.waitOnUpdate()` or dedicated events in a loop with a 10s ceiling; capture detailed debug logs on timeout
- Update `test_subprocess_connection_fix.py` to assert that the worker stays alive until `managedAccounts` arrives (can mock IB object)

### 3. Gateway Version Matrix (manual but critical)

- For each available build (10.41 current, 10.40, 10.39/10.37):
  1. Exit Gateway via UI, install/launch target version
  2. Run `python3 diagnose_gateway_internal_state.py`
  3. Run the ibapi probe (expect `nextValidId` within <1s)
  4. Run `./START_TRADER.sh AAPL` ONLY if the probe succeeds
- Record results + timestamps in `handoff/LATEST_HANDOFF.md`
- If 10.39 (or older) works consistently, pin that version and note it here

### 4. Observability & Regression Guard

- Keep the ibapi probe script in the repo so on-call engineers can validate `nextValidId` in seconds
- Add a lightweight pytest (or CLI check) that shells out to the probe with a mock; CI should fail if the worker regresses to “empty accounts”
- Capture worker stderr to `logs/ibkr_worker_debug.log` with handshake timing metrics

### 5. Escalation Path (if all versions fail)

- Run the official ibapi sample (`Program.py` or sync wrapper) and capture the failure logs
- Open an IBKR support ticket with:
  - Gateway versions tested + results
  - ibapi probe logs showing whether `nextValidId` arrives
  - Confirmation that permissions and config are correct
- Mention that ActiveX/Socket Clients cannot be disabled in Gateway ≥10.41, so the issue is server-side

## 🎯 WHAT WE'VE FIXED

✅ **Gateway Connectivity**: Force restart resolved TCP connection issues
✅ **Subprocess Worker**: Timing race condition fixed and working
✅ **Library Compatibility**: Both ib_async and ib_insync tested - not the issue
✅ **Connection Parameters**: All variations tested - not the issue
✅ **Gateway API Configuration**: ActiveX/Socket Clients is permanently enabled - not the issue

## 🔍 CURRENT ISSUE ISOLATED TO

The API handshake **starts successfully** but **never completes**:
1. ✅ TCP connection succeeds
2. ✅ ib_async logs "Connected"
3. ❌ Waits for apiStart event (never arrives)
4. ❌ Times out after 5-15 seconds

This pattern indicates **Gateway version compatibility plus worker synchronization issues**, **not account permissions**.

## 📋 IF ACCOUNT PERMISSIONS AND GATEWAY VERSION DON'T FIX IT

### Try TWS Instead of Gateway
- Install TWS (Trader Workstation)
- Configure for paper trading on port 7497
- Test if TWS API works where Gateway fails

### Test Official IBKR Python API
- ✅ DONE (2025-11-24): `ibapi` 10.37.2 connects instantly and returns `managedAccounts: DUN264991`
- Keep this probe available to differentiate Gateway failures from worker defects

### Contact IBKR Support
- Provide exact error: "Subprocess client times out waiting for apiStart/managedAccounts even though ibapi probe succeeds"
- Gateway versions tested (10.41, 10.40, etc.)
- Account: DUN264991
- Note: Both ib_async and ib_insync fail on certain builds; ibapi succeeds

## 🎉 EXPECTED OUTCOME

After pinning a working Gateway build AND updating the worker synchronization:

```bash
./START_TRADER.sh AAPL
```

Should result in:
- ✅ Gateway connection successful
- ✅ Account data retrieved
- ✅ Trading system starts normally
- ✅ Dashboard shows connected status

---

**The technical infrastructure is working. The remaining work is (1) align the subprocess worker with IBKR handshake callbacks and (2) pin a Gateway build that doesn’t regress the API protocol.**
