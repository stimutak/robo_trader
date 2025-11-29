# Subprocess-Based IBKR Wrapper Implementation Plan

## Problem Statement
The ib_async library has a fundamental incompatibility with complex async environments. While simple standalone scripts connect successfully to IBKR Gateway, the trading system's complex async environment causes API handshake timeouts.

**Root Cause:** ib_async's event loop management conflicts with our trading system's async architecture.

**Solution:** Complete process isolation using subprocess-based wrapper.

## Architecture Overview

### Current (Broken) Architecture
```
Trading System (Complex Async Environment)
  └─> ib_async.IB.connectAsync()
      └─> TCP connects ✅
      └─> API handshake times out ❌ (30 seconds)
```

### New (Subprocess) Architecture
```
Trading System (Complex Async Environment)
  └─> SubprocessIBKRClient (IPC via JSON)
      └─> Subprocess (Clean Async Environment)
          └─> ib_async.IB.connectAsync()
              └─> TCP connects ✅
              └─> API handshake succeeds ✅
```

## Implementation Phases

### Phase 1: Create Subprocess Worker Script
**File:** `robo_trader/clients/ibkr_subprocess_worker.py`

**Purpose:** Standalone script that runs in subprocess, handles all ib_async operations.

**Features:**
- Clean event loop (no interference from parent process)
- JSON-based command/response protocol
- Handles: connect, disconnect, get_accounts, get_positions, get_account_summary
- Graceful error handling and cleanup
- Heartbeat mechanism for health monitoring

**Interface (stdin/stdout JSON):**
```json
// Commands (stdin)
{"command": "connect", "host": "127.0.0.1", "port": 4002, "client_id": 1, "readonly": true, "timeout": 15.0}
{"command": "get_accounts"}
{"command": "get_positions"}
{"command": "disconnect"}
{"command": "ping"}

// Responses (stdout)
{"status": "success", "data": {"connected": true, "accounts": ["DUN264991"]}}
{"status": "success", "data": {"accounts": ["DUN264991"]}}
{"status": "success", "data": {"positions": [...]}}
{"status": "success", "data": {"disconnected": true}}
{"status": "success", "data": {"pong": true}}
{"status": "error", "error": "Connection timeout", "traceback": "..."}
```

### Phase 2: Create Subprocess Client
**File:** `robo_trader/clients/subprocess_ibkr_client.py`

**Purpose:** Async wrapper that manages subprocess, sends commands, receives responses.

**Features:**
- Subprocess lifecycle management (start, stop, restart)
- Async command/response handling
- Timeout handling for subprocess operations
- Automatic subprocess restart on crashes
- Resource cleanup (kill subprocess on exit)

**API:**
```python
class SubprocessIBKRClient:
    async def start(self) -> None:
        """Start the subprocess worker"""
        
    async def connect(self, host: str, port: int, client_id: int, 
                     readonly: bool = True, timeout: float = 15.0) -> bool:
        """Connect to IBKR via subprocess"""
        
    async def get_accounts(self) -> list[str]:
        """Get managed accounts"""
        
    async def get_positions(self) -> list[dict]:
        """Get current positions"""
        
    async def get_account_summary(self) -> dict:
        """Get account summary"""
        
    async def disconnect(self) -> None:
        """Disconnect from IBKR"""
        
    async def stop(self) -> None:
        """Stop the subprocess worker"""
        
    async def ping(self) -> bool:
        """Check if subprocess is alive"""
```

### Phase 3: Integrate with Robust Connection
**File:** `robo_trader/utils/robust_connection.py`

**Changes:**
1. Replace direct `ib_async.IB()` usage with `SubprocessIBKRClient`
2. Keep existing retry logic, circuit breaker, zombie cleanup
3. Add subprocess health monitoring
4. Handle subprocess crashes (restart subprocess, retry connection)

**Modified Flow:**
```python
async def connect_ibkr_robust(...):
    # Create subprocess client
    client = SubprocessIBKRClient()
    await client.start()
    
    # Existing retry logic
    for attempt in range(max_retries):
        try:
            # Connect via subprocess
            success = await client.connect(host, port, client_id, readonly, timeout)
            if success:
                accounts = await client.get_accounts()
                return client  # Return subprocess client instead of IB object
        except SubprocessCrashError:
            # Restart subprocess and retry
            await client.stop()
            await client.start()
            continue
```

### Phase 4: Update Runner
**File:** `robo_trader/runner_async.py`

**Changes:**
1. Accept `SubprocessIBKRClient` instead of `IB` object
2. Use subprocess client methods instead of direct ib_async calls
3. Add subprocess health monitoring in main loop
4. Graceful shutdown (stop subprocess on exit)

**Example:**
```python
# Old
self.ib = await connect_ibkr_robust(...)
positions = self.ib.positions()

# New
self.ibkr_client = await connect_ibkr_robust(...)
positions = await self.ibkr_client.get_positions()
```

## Implementation Details

### Subprocess Worker (ibkr_subprocess_worker.py)

**Key Components:**

1. **Command Handler:**
```python
async def handle_command(command: dict) -> dict:
    cmd = command.get("command")
    
    if cmd == "connect":
        return await handle_connect(command)
    elif cmd == "get_accounts":
        return await handle_get_accounts()
    elif cmd == "disconnect":
        return await handle_disconnect()
    elif cmd == "ping":
        return {"status": "success", "data": {"pong": True}}
    else:
        return {"status": "error", "error": f"Unknown command: {cmd}"}
```

2. **Connection Management:**
```python
ib = None  # Global IB instance

async def handle_connect(params: dict) -> dict:
    global ib
    try:
        ib = IB()
        await ib.connectAsync(
            host=params["host"],
            port=params["port"],
            clientId=params["client_id"],
            readonly=params.get("readonly", True),
            timeout=params.get("timeout", 15.0)
        )
        accounts = ib.managedAccounts()
        return {"status": "success", "data": {"connected": True, "accounts": accounts}}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
```

3. **Main Loop:**
```python
async def main():
    while True:
        try:
            # Read command from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            command = json.loads(line)
            response = await handle_command(command)
            
            # Write response to stdout
            print(json.dumps(response), flush=True)
        except Exception as e:
            error_response = {"status": "error", "error": str(e)}
            print(json.dumps(error_response), flush=True)
```

### Subprocess Client (subprocess_ibkr_client.py)

**Key Components:**

1. **Subprocess Management:**
```python
class SubprocessIBKRClient:
    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.lock = asyncio.Lock()
        
    async def start(self):
        """Start subprocess worker"""
        worker_script = Path(__file__).parent / "ibkr_subprocess_worker.py"
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, str(worker_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
```

2. **Command Execution:**
```python
async def _execute_command(self, command: dict, timeout: float = 30.0) -> dict:
    """Send command to subprocess and wait for response"""
    async with self.lock:
        if not self.process or self.process.returncode is not None:
            raise SubprocessCrashError("Subprocess not running")
            
        # Send command
        command_json = json.dumps(command) + "\n"
        self.process.stdin.write(command_json.encode())
        await self.process.stdin.drain()
        
        # Read response with timeout
        try:
            line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=timeout
            )
            response = json.loads(line.decode())
            
            if response.get("status") == "error":
                raise IBKRError(response.get("error"))
                
            return response.get("data", {})
        except asyncio.TimeoutError:
            raise IBKRTimeoutError(f"Command timeout after {timeout}s")
```

3. **Health Monitoring:**
```python
async def ping(self) -> bool:
    """Check if subprocess is alive and responsive"""
    try:
        await self._execute_command({"command": "ping"}, timeout=5.0)
        return True
    except Exception:
        return False
```

## Testing Strategy

### Unit Tests
1. **Test subprocess worker standalone:**
   ```bash
   echo '{"command": "connect", "host": "127.0.0.1", "port": 4002, "client_id": 1}' | python3 robo_trader/clients/ibkr_subprocess_worker.py
   ```

2. **Test subprocess client:**
   ```python
   client = SubprocessIBKRClient()
   await client.start()
   await client.connect("127.0.0.1", 4002, 1)
   accounts = await client.get_accounts()
   assert accounts == ["DUN264991"]
   ```

3. **Test with robust connection:**
   ```python
   client = await connect_ibkr_robust(...)
   assert isinstance(client, SubprocessIBKRClient)
   ```

### Integration Tests
1. **Test with trading system:**
   ```bash
   python3 -m robo_trader.runner_async --symbols AAPL
   ```

2. **Test reconnection after subprocess crash:**
   - Kill subprocess mid-operation
   - Verify automatic restart and reconnection

3. **Test long-running stability:**
   - Run for 24+ hours
   - Monitor for zombie connections
   - Verify no memory leaks

### Success Criteria
✅ Trading system connects successfully to Gateway  
✅ No API handshake timeouts  
✅ No zombie CLOSE_WAIT connections  
✅ Subprocess restarts automatically on crashes  
✅ Stable operation for 24+ hours  
✅ All existing tests pass  

## Rollback Plan
If subprocess approach fails:
1. Revert to main branch
2. Consider alternative solutions:
   - Use synchronous ib_insync in thread pool
   - Switch to different IBKR library
   - Use REST API instead of TWS API

## Timeline
- **Phase 1 (Subprocess Worker):** 2-3 hours
- **Phase 2 (Subprocess Client):** 2-3 hours
- **Phase 3 (Integration):** 1-2 hours
- **Phase 4 (Testing):** 2-4 hours
- **Total:** 7-12 hours

## Next Steps
1. ✅ Create handoff document
2. ✅ Create implementation plan
3. ⏳ Create branch `fix/subprocess-ibkr-wrapper`
4. ⏳ Implement Phase 1 (Subprocess Worker)
5. ⏳ Implement Phase 2 (Subprocess Client)
6. ⏳ Test standalone
7. ⏳ Implement Phase 3 (Integration)
8. ⏳ Implement Phase 4 (Runner Updates)
9. ⏳ Full integration testing
10. ⏳ 24-hour stability test
11. ⏳ Merge to main

---
**Document Created:** 2025-10-15 13:15 EDT  
**Status:** Ready for implementation

