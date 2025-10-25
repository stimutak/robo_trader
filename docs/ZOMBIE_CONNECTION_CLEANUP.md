# Zombie Connection Cleanup System

## Overview

The zombie connection cleanup system detects and removes CLOSE_WAIT TCP connections that accumulate on the IBKR TWS/Gateway API port (typically 7497). These "zombie" connections are caused by failed handshake attempts and prevent new connections from establishing properly.

## What Are Zombie Connections?

Zombie connections are TCP sockets left in CLOSE_WAIT state after incomplete handshakes:
- **Cause**: When `ib_async.connectAsync()` times out, the socket remains open on the Gateway/TWS side even though `isConnected()` returns False
- **Impact**: Exhausts Gateway connection slots, causes "Already connected" errors, prevents new connections
- **Detection**: Found via `netstat -an | grep 7497 | grep CLOSE_WAIT`

## Components

### 1. Detection: `check_tws_zombie_connections(port=7497)`

Checks for CLOSE_WAIT connections on the specified port.

```python
from robo_trader.utils.robust_connection import check_tws_zombie_connections

zombie_count, error_msg = check_tws_zombie_connections(port=7497)
if zombie_count > 0:
    print(f"Found {zombie_count} zombies: {error_msg}")
```

**Returns:**
- `(0, "")` - Clean state, no zombies
- `(N, "error message")` - N zombies detected

**Error Handling:**
- Fails open on errors (returns `(0, "")`) to allow connection attempts
- Logs warnings for diagnostic failures

### 2. Cleanup: `kill_tws_zombie_connections(port=7497)`

Kills zombie processes safely, preserving Gateway/TWS/Java processes.

```python
from robo_trader.utils.robust_connection import kill_tws_zombie_connections

success, message = kill_tws_zombie_connections(port=7497)
if success:
    print(f"✅ {message}")
else:
    print(f"⚠️ {message}")
```

**Safety Features:**
- **NEVER kills Gateway/TWS/Java processes** - these require manual restart with 2FA
- Only kills Python-based runner/diagnostic processes
- Uses structured lsof output (`-Fpc`) to parse PID and command
- Verifies cleanup with post-kill netstat check

**Process Selection:**
- ✅ Kills: `python`, `runner`, `websocket_server` processes
- ❌ Skips: `java`, `gateway`, `tws` processes

**Returns:**
- `(True, "message")` - Cleanup successful, port is clean
- `(False, "message")` - Cleanup incomplete or failed

### 3. Integration: `RobustConnectionManager`

Automatically runs zombie cleanup during connection lifecycle:

```python
from robo_trader.utils.robust_connection import RobustConnectionManager

manager = RobustConnectionManager(
    connect_func=my_connect_func,
    max_retries=5,
    port=7497,
)

# Automatic zombie cleanup:
# - Before initial connection attempt
# - Before each retry attempt
connection = await manager.connect()
```

**Cleanup Points:**
1. **Pre-connection**: Checks and cleans before first attempt
2. **Pre-retry**: Cleans zombies before each retry (after attempt 1+)
3. **Non-blocking**: Cleanup failures don't prevent connection attempts (logged as warnings)

## Configuration

### File Lock Path

The file lock prevents concurrent IBKR handshakes across processes.

**Default:** `/tmp/ibkr_connect.lock`

**Configuration Options:**

1. **Environment Variable:**
   ```bash
   export IBKR_LOCK_FILE_PATH=/var/run/ibkr_connect.lock
   ```

2. **Direct Instantiation:**
   ```python
   from robo_trader.utils.robust_connection import _ConnectFileLock
   
   with _ConnectFileLock(lock_path="/custom/path/ibkr.lock"):
       # Serialized connection
       pass
   ```

### File Lock Timeout

Controls how long to wait for the lock before timing out.

**Default:** 30 seconds

**Configuration Options:**

1. **Environment Variable:**
   ```bash
   export IBKR_LOCK_TIMEOUT=60  # 60 seconds
   ```

2. **Direct Instantiation:**
   ```python
   with _ConnectFileLock(timeout=60.0):
       # Serialized connection with 60s timeout
       pass
   ```

**Timeout Behavior:**
- Uses `signal.SIGALRM` for timeout enforcement (Unix-based systems only)
- Raises `TimeoutError` if lock cannot be acquired within timeout
- Automatically cleans up on timeout or error

## Circuit Breaker Telemetry

The circuit breaker emits structured metrics for state changes:

```python
# Automatic emission on state changes
logger.info(
    "circuit_breaker_state_change",
    extra={
        "metric_type": "circuit_breaker_state",
        "state": "open",  # open, closed, half_open
        "failure_count": 5,
        "success_count": 0,
        "timestamp": "2025-10-19T12:34:56",
    },
)
```

**Metrics Emitted:**
- `circuit_breaker_state_change` - State transition events
- `circuit_breaker_metrics` - Counter snapshots

**Integration Points:**
- Prometheus: Parse structured logs or add custom exporter
- CloudWatch: Use log filters to extract metrics
- DataDog: Use log processing pipeline
- Custom: Subscribe to logger events

## Testing

Comprehensive integration tests are in `tests/test_zombie_cleanup_integration.py`:

```bash
# Run all zombie cleanup tests
pytest tests/test_zombie_cleanup_integration.py -v

# Run specific test class
pytest tests/test_zombie_cleanup_integration.py::TestZombieConnectionCleanup -v

# Run specific test
pytest tests/test_zombie_cleanup_integration.py::TestZombieConnectionDetection::test_check_tws_zombie_connections_found -v
```

**Test Coverage:**
- ✅ Zombie detection (clean state, zombies found, errors)
- ✅ Zombie cleanup (success, partial, Java process skipping)
- ✅ RobustConnectionManager integration
- ✅ End-to-end workflow (detect → clean → verify)
- ✅ Error handling (timeouts, missing commands, partial cleanup)

## Best Practices

### 1. Prevention Over Cleanup

**Recommended:**
- Use subprocess-based IBKR client (`use_subprocess=True`)
- Always call `ib.disconnect()` even on failed handshakes
- Set appropriate connection timeouts (10-30 seconds)
- Use file lock to serialize handshakes

**Avoid:**
- Concurrent handshakes from multiple processes
- Skipping disconnect on errors
- Very short timeouts causing premature failures

### 2. Monitoring

**Key Metrics:**
- Zombie connection count over time
- Cleanup success rate
- Circuit breaker state changes
- Connection retry counts

**Alerting Thresholds:**
- ⚠️ Warning: >5 zombies detected
- 🚨 Critical: >10 zombies detected
- 🚨 Critical: Circuit breaker opened

### 3. Manual Intervention

**When to Restart Gateway/TWS:**
- Zombies remain after cleanup (Gateway-owned connections)
- Circuit breaker stuck in OPEN state
- Connection failures persist despite cleanup
- "Already connected" errors continue

**Restart Procedure:**
1. Stop all Python processes: `pkill -9 -f "runner_async|app.py|websocket_server"`
2. Exit TWS/Gateway application
3. Wait 30 seconds for TCP state to clear
4. Restart TWS/Gateway with 2FA login
5. Restart Python processes

## Troubleshooting

### Issue: Zombies Not Cleaned

**Symptoms:**
- `kill_tws_zombie_connections()` returns `(False, "zombies remain")`
- netstat shows CLOSE_WAIT after cleanup

**Causes:**
- Zombies owned by Gateway/TWS/Java (cannot be killed)
- Process permissions issue

**Solutions:**
1. Check process ownership: `lsof -nP -iTCP:7497 -sTCP:CLOSE_WAIT`
2. If owned by Gateway: Manual TWS/Gateway restart required
3. If owned by Python: Check process permissions

### Issue: Lock Timeout

**Symptoms:**
- `TimeoutError: Failed to acquire file lock`

**Causes:**
- Another process holds lock indefinitely
- Stale lock file from crashed process

**Solutions:**
1. Check for stuck processes: `lsof /tmp/ibkr_connect.lock`
2. Increase timeout: `export IBKR_LOCK_TIMEOUT=60`
3. Remove stale lock: `rm /tmp/ibkr_connect.lock` (if no process holds it)

### Issue: lsof Not Available

**Symptoms:**
- `kill_tws_zombie_connections()` returns `(False, "lsof not available")`

**Causes:**
- lsof command not installed

**Solutions:**
1. Install lsof: `apt-get install lsof` (Ubuntu/Debian) or `yum install lsof` (RedHat/CentOS)
2. Use manual cleanup as fallback

## Implementation Details

### Why CLOSE_WAIT State?

When a TCP connection is closed by the remote peer (Gateway/TWS):
1. Remote sends FIN packet
2. Local acknowledges with ACK
3. Connection enters CLOSE_WAIT state
4. **Local must call close()** to complete shutdown
5. If close() never called → zombie remains indefinitely

### Why Zombie Cleanup Is Critical

Without cleanup:
- Gateway/TWS connection slots exhaust (default: 32 clients)
- New connections fail with "Already connected" or timeout
- Circuit breaker opens, blocking all traffic
- Manual intervention required (Gateway restart)

With cleanup:
- Zombies removed automatically on retry
- Connection slots freed for new attempts
- Circuit breaker remains closed
- System self-heals

## References

- [IBKR API Client Portal Documentation](https://interactivebrokers.github.io/cpwebapi/)
- [TCP State Diagram](https://en.wikipedia.org/wiki/Transmission_Control_Protocol#Protocol_operation)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- Project issue: Commit bd87fe5 (zombie connection bug fix)
- Project issue: Commit f55015c (enhanced zombie cleanup)

## See Also

- `robo_trader/utils/robust_connection.py` - Implementation
- `tests/test_zombie_cleanup_integration.py` - Integration tests
- `test_robust_connection.py` - Unit tests
- `.cursor/rules` - TWS management guidelines
