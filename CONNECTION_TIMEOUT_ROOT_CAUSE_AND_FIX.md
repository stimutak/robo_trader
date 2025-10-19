# IBKR Gateway Connection Timeout - Root Cause Analysis & Complete Fix

**Date:** 2025-10-19  
**Status:** Root cause identified, comprehensive fix designed  
**Branch:** `cursor/investigate-and-fix-client-connection-timeouts-e1b3`

---

## Executive Summary

**Problem:** TCP socket connects successfully but API handshake times out after 15-30 seconds, making the trading system unusable.

**Root Cause:** Connection lifecycle mismanagement causing zombie CLOSE_WAIT connections to accumulate on Gateway port 4002, eventually blocking all new connections.

**Solution:** Implement connection pooling in `runner_async.py` to maintain a single persistent subprocess client connection instead of connect/disconnect cycles.

---

## Root Cause Analysis

### The Problem Chain

1. **Current Flow (BROKEN)**:
   ```
   runner_async.py starts
   → setup() calls connect_ibkr_robust()
   → Creates new IBKR connection
   → Runs trading logic
   → cleanup() calls disconnect()
   → Leaves CLOSE_WAIT zombie connection
   → Next run creates another zombie
   → After 3-4 zombies: Gateway stops accepting connections
   ```

2. **What Happens**:
   - Each `runner_async.py` execution creates a NEW connection
   - Each cleanup() disconnects, creating a CLOSE_WAIT zombie
   - Zombies accumulate faster than Gateway can clean them
   - After 3-4 zombies, Gateway's connection limit is hit
   - New connections timeout because Gateway won't accept them

3. **Why Simple Tests Work**:
   - Simple tests run once and exit cleanly
   - No zombie accumulation
   - Gateway has time to clean up between runs
   - But runner_async runs continuously or in quick succession

### Evidence

**From Handoff Documents:**
- ✅ Subprocess IBKR client works perfectly in isolation (test_minimal_runner.py)
- ✅ TCP connection succeeds (socket layer works)
- ❌ API handshake times out (Gateway not responding)
- ❌ CLOSE_WAIT connections accumulate on port 4002
- ❌ After 3-4 zombies, all new connections fail

**From Code Analysis:**
- `runner_async.py:506` - Creates new connection with `connect_ibkr_robust()`
- `runner_async.py:2087-2098` - Disconnects in cleanup, creating zombie
- No connection pooling or reuse mechanism exists
- Each test run/restart creates new client instances

### Why This Is In Our Code, Not Gateway

Gateway is configured correctly and works fine with simple clients. The issue is:

1. **Too Many Connections**: We're creating multiple clients rapidly
2. **Poor Lifecycle Management**: Connect/disconnect cycles leave zombies
3. **No Cleanup Timing**: Zombies created faster than Gateway can clean them
4. **No Connection Reuse**: Every operation creates a new connection

---

## The Complete Fix

### Strategy: Connection Pooling with Persistent Subprocess Client

Instead of:
```python
# CURRENT (BROKEN)
async def run():
    await setup()  # Creates new connection
    await process_symbols()
    await cleanup()  # Disconnects, creates zombie
```

We need:
```python
# FIXED
class AsyncRunner:
    async def __aenter__(self):
        await self._start_persistent_connection()  # Connect ONCE
        return self
    
    async def run(self):
        # Reuse self._ibkr_client for all operations
        await self.process_symbols()
    
    async def __aexit__(self):
        await self._stop_persistent_connection()  # Disconnect ONCE
```

### Implementation Plan

#### Phase 1: Add Connection Pool to AsyncRunner

**File:** `robo_trader/runner_async.py`

**Changes:**

1. **Add persistent client attribute** (line ~140):
   ```python
   class AsyncRunner:
       def __init__(self, ...):
           # Existing attributes...
           self._ibkr_client: Optional[SubprocessIBKRClient] = None
           self._connection_healthy = False
   ```

2. **Add connection pool methods**:
   ```python
   async def _start_persistent_connection(self) -> None:
       """Start and maintain persistent IBKR connection."""
       from .clients.subprocess_ibkr_client import SubprocessIBKRClient
       from .utils.robust_connection import kill_tws_zombie_connections
       
       # Kill any existing zombies FIRST
       success, msg = kill_tws_zombie_connections(self.cfg.ibkr.port)
       logger.info(f"Zombie cleanup: {msg}")
       
       # Wait for cleanup to complete
       await asyncio.sleep(1.0)
       
       # Create subprocess client
       self._ibkr_client = SubprocessIBKRClient()
       await self._ibkr_client.start()
       
       # Connect with retry logic
       max_retries = 3
       for attempt in range(max_retries):
           try:
               connected = await self._ibkr_client.connect(
                   host=self.cfg.ibkr.host,
                   port=self.cfg.ibkr.port,
                   client_id=self.cfg.ibkr.client_id + attempt,
                   readonly=self.cfg.ibkr.readonly,
                   timeout=self.cfg.ibkr.timeout,
               )
               
               if connected:
                   self._connection_healthy = True
                   logger.info("✓ Persistent IBKR connection established")
                   return
                   
           except Exception as e:
               logger.warning(f"Connection attempt {attempt+1} failed: {e}")
               if attempt < max_retries - 1:
                   await asyncio.sleep(2.0 ** attempt)  # Exponential backoff
       
       raise ConnectionError("Failed to establish persistent connection")
   
   async def _stop_persistent_connection(self) -> None:
       """Stop persistent IBKR connection."""
       if self._ibkr_client:
           try:
               await self._ibkr_client.disconnect()
               await self._ibkr_client.stop()
           except Exception as e:
               logger.error(f"Error stopping connection: {e}")
           finally:
               self._ibkr_client = None
               self._connection_healthy = False
   
   async def _check_connection_health(self) -> bool:
       """Check if connection is still healthy."""
       if not self._ibkr_client or not self._ibkr_client.is_connected:
           return False
       
       try:
           # Ping to verify subprocess is responsive
           return await self._ibkr_client.ping()
       except Exception:
           return False
   ```

3. **Modify setup() to use persistent connection** (line ~460):
   ```python
   async def setup(self, symbols: List[str]) -> None:
       """Setup trading environment with persistent connection."""
       logger.info("=== Setup Phase ===")
       
       # Load config
       self.cfg = load_config()
       
       # Start persistent connection if not already started
       if not self._connection_healthy:
           await self._start_persistent_connection()
       
       # Set self.ib to the subprocess client for compatibility
       self.ib = self._ibkr_client
       
       # Rest of setup (database, risk managers, etc.)
       # ... existing code ...
   ```

4. **Modify cleanup() to NOT disconnect** (line ~2080):
   ```python
   async def cleanup(self) -> None:
       """Cleanup resources (but keep connection alive)."""
       logger.info("=== Cleanup Phase ===")
       
       # Close database
       if self.db:
           await self.db.close()
           self.db = None
       
       # DO NOT disconnect IBKR here - connection stays alive
       # Only disconnect when runner context exits
       
       logger.info("Cleanup complete (connection kept alive)")
   ```

5. **Add context manager support**:
   ```python
   async def __aenter__(self):
       """Context manager entry - start persistent connection."""
       await self._start_persistent_connection()
       return self
   
   async def __aexit__(self, exc_type, exc_val, exc_tb):
       """Context manager exit - stop persistent connection."""
       await self._stop_persistent_connection()
   ```

#### Phase 2: Update Main Entry Point

**File:** `robo_trader/runner_async.py` (main function, line ~2200)

**Changes:**

```python
async def main() -> int:
    """Main entry point with persistent connection."""
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    
    # Use context manager for proper connection lifecycle
    async with AsyncRunner(
        duration=args.duration,
        bar_size=args.bar_size,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        # ... other args
    ) as runner:
        # Setup (uses persistent connection)
        await runner.setup(symbols)
        
        if args.once:
            # Single run
            await runner.run(symbols)
        else:
            # Continuous mode
            while True:
                try:
                    # Check connection health
                    if not await runner._check_connection_health():
                        logger.warning("Connection unhealthy, reconnecting...")
                        await runner._stop_persistent_connection()
                        await runner._start_persistent_connection()
                    
                    await runner.run(symbols)
                    
                    if not is_market_open():
                        wait_secs = seconds_until_market_open()
                        logger.info(f"Market closed, sleeping {wait_secs/60:.1f} minutes")
                        await asyncio.sleep(min(wait_secs, 300))
                    else:
                        await asyncio.sleep(60)
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in run loop: {e}")
                    await asyncio.sleep(60)
    
    # Connection automatically cleaned up when exiting context
    return 0
```

#### Phase 3: Enhance Subprocess Client

**File:** `robo_trader/clients/subprocess_ibkr_client.py`

**Add connection keep-alive**:

```python
class SubprocessIBKRClient:
    def __init__(self):
        # ... existing init ...
        self._last_activity = time.time()
        self._keepalive_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start subprocess and begin keepalive monitoring."""
        # ... existing start code ...
        
        # Start keepalive task
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
    
    async def _keepalive_loop(self) -> None:
        """Periodic ping to keep connection alive."""
        while self.process and self.process.poll() is None:
            try:
                await asyncio.sleep(30.0)  # Ping every 30 seconds
                
                if self._connected:
                    # Check if connection is still alive
                    if not await self.ping():
                        logger.warning("Keepalive ping failed")
                        self._connected = False
                        
            except Exception as e:
                logger.error(f"Keepalive error: {e}")
    
    async def stop(self) -> None:
        """Stop subprocess and cleanup."""
        # Cancel keepalive task
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
        
        # ... existing stop code ...
```

---

## Testing Strategy

### Test 1: Single Run (Should Already Work)
```bash
source .venv/bin/activate
python3 -m robo_trader.runner_async --symbols AAPL --once

# Expected: Success, no zombies created
# Verify: lsof -nP -iTCP:4002 | grep CLOSE_WAIT  # Should be empty
```

### Test 2: Consecutive Runs (Currently Fails, Should Pass After Fix)
```bash
source .venv/bin/activate

# Run 5 times in a row
for i in {1..5}; do
    echo "=== Run $i ==="
    python3 -m robo_trader.runner_async --symbols AAPL --once
    echo "Checking for zombies..."
    lsof -nP -iTCP:4002 | grep CLOSE_WAIT || echo "No zombies"
    sleep 2
done

# Expected: All runs succeed, no zombie accumulation
```

### Test 3: Continuous Mode (Long-Running)
```bash
source .venv/bin/activate

# Run for 5 minutes with health monitoring
timeout 300 python3 -m robo_trader.runner_async --symbols AAPL,NVDA

# Expected: 
# - Single connection maintained throughout
# - No reconnections unless health check fails
# - No zombies created
# - Clean shutdown after timeout
```

### Test 4: Connection Recovery
```bash
source .venv/bin/activate

# Start runner
python3 -m robo_trader.runner_async --symbols AAPL &
RUNNER_PID=$!

# Wait for connection
sleep 30

# Kill Gateway (simulate disconnect)
# ... restart Gateway ...

# Wait for runner to detect and reconnect
sleep 60

# Check runner is still running and healthy
ps -p $RUNNER_PID
```

---

## Success Criteria

✅ **Fixed when:**
1. Single run completes without errors
2. 10 consecutive runs complete without zombie accumulation
3. Continuous mode runs for 1+ hour without issues
4. Connection recovery works after Gateway restart
5. `lsof -nP -iTCP:4002 | grep CLOSE_WAIT` shows 0 zombies after runs
6. All existing tests still pass

---

## Migration Path

### For Existing Code

The fix maintains backward compatibility:
- `self.ib` still refers to the IBKR client
- All existing `self.ib` usage continues to work
- Just the lifecycle management changes

### Breaking Changes

None. The change is internal to `runner_async.py`.

---

## Files to Modify

1. ✅ `robo_trader/runner_async.py` - Add connection pooling
2. ✅ `robo_trader/clients/subprocess_ibkr_client.py` - Add keepalive
3. ✅ `CONNECTION_TIMEOUT_ROOT_CAUSE_AND_FIX.md` - This document

---

## Estimated Implementation Time

- **Phase 1 (Runner changes):** 2-3 hours
- **Phase 2 (Main entry point):** 1 hour  
- **Phase 3 (Subprocess enhancements):** 1-2 hours
- **Testing:** 2-3 hours
- **Total:** 6-9 hours

---

## Alternative Solutions Considered

### ❌ Option 1: Fix Gateway Configuration
**Rejected:** Gateway is working correctly. The problem is in our code.

### ❌ Option 2: More Aggressive Zombie Killing
**Rejected:** Doesn't solve root cause, just treats symptoms.

### ❌ Option 3: Use Different Client IDs
**Rejected:** Doesn't prevent zombie accumulation.

### ✅ Option 4: Connection Pooling (SELECTED)
**Reason:** Solves root cause by eliminating connect/disconnect cycles.

---

## Implementation Priority

**CRITICAL - P0**

This blocks all trading operations. Should be implemented immediately.

---

## Questions & Concerns

### Q: What if the connection dies?
**A:** Health monitoring detects failures and reconnects automatically.

### Q: Will this work with existing code?
**A:** Yes, maintains full backward compatibility.

### Q: What about zombies from previous runs?
**A:** `kill_tws_zombie_connections()` cleans them before first connection.

### Q: Performance impact?
**A:** Positive - no reconnection overhead between operations.

---

## Next Steps

1. ✅ Create this comprehensive analysis document
2. ⏳ Implement Phase 1: Connection pooling in runner_async.py
3. ⏳ Implement Phase 2: Update main entry point
4. ⏳ Implement Phase 3: Enhance subprocess client
5. ⏳ Run test suite to verify fix
6. ⏳ Test with 10 consecutive runs
7. ⏳ Test with 1-hour continuous run
8. ⏳ Document results and close issue

---

**Document Status:** Complete  
**Ready for Implementation:** YES  
**Estimated Fix Time:** 6-9 hours  
**Risk Level:** Low (maintains backward compatibility)
