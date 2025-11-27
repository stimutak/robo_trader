# ARCHIVED IBKR Connection Test Scripts

## DO NOT RUN THESE SCRIPTS

These test scripts were archived because **running them creates Gateway zombie connections**.

### Why they're dangerous:

1. They use raw `ib.disconnect()` which is a no-op (to protect Gateway)
2. But the connection they create **stays open until the script exits**
3. When the script exits without proper cleanup, it leaves a CLOSE_WAIT zombie
4. This zombie blocks ALL future API connections until Gateway restarts (requires 2FA login)

### If you need to test Gateway connectivity:

Use the safe startup script instead:
```bash
./START_TRADER.sh
```

### If you must run one of these scripts:

1. Set the force disconnect flag:
   ```bash
   export IBKR_FORCE_DISCONNECT=1
   ```
2. Import and use `safe_disconnect()` at the end:
   ```python
   from robo_trader.utils.ibkr_safe import safe_disconnect
   safe_disconnect(ib)
   ```

### Scripts in this folder:

All scripts here connect to IBKR Gateway/TWS and were not using `safe_disconnect()`.

See CLAUDE.md for the correct way to start the trading system.
