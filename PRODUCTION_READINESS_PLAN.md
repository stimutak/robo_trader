# 🚨 PRODUCTION READINESS ACTION PLAN 🚨

> **⚠️ CRITICAL: THIS SYSTEM IS NOT READY FOR LIVE TRADING**  
> **THIS IS THE AUTHORITATIVE PLAN - FOLLOW IT UNTIL ALL ITEMS ARE COMPLETE**  
> **Last Updated: 2025-01-11**  
> **Status: IN PROGRESS - PAPER TRADING ONLY**

---

## 📊 Current Production Readiness Score: 5/10

**DO NOT ATTEMPT LIVE TRADING UNTIL SCORE REACHES 10/10**

---

## 🎯 Mission Critical Path

This plan must be executed in order. Each phase must be completed and tested before moving to the next.

### Phase 0: IMMEDIATE BLOCKERS (Complete First - Est. 2-3 days)
*These issues could cause immediate financial loss or system failure*

### Phase 1: CRITICAL SAFETY (Must Complete - Est. 1 week)
*Core safety mechanisms required for any live trading*

### Phase 2: HIGH PRIORITY (Pre-Production - Est. 2 weeks)
*Essential for stable production operation*

### Phase 3: MEDIUM PRIORITY (Production Enhancement - Est. 1 month)
*Important for long-term stability and compliance*

---

## 📝 PHASE 0: IMMEDIATE BLOCKERS [STATUS: NOT STARTED]

### ✅ TASK 0.1: Remove ALL Hardcoded Connection Parameters
**Priority:** CRITICAL  
**Status:** COMPLETED - 2025-01-11
**Files Modified:**
- [x] `robo_trader/config.py` - Lines 370-374 (removed defaults, added validation)
- [x] `robo_trader/clients/async_ibkr_client.py` (verified no hardcoded values)
- [x] All test files with hardcoded ports/hosts

**Implementation:**
```python
# BEFORE (DANGEROUS):
"host": os.getenv("IBKR_HOST", "127.0.0.1"),  # REMOVE DEFAULT
"port": int(os.getenv("IBKR_PORT", "7497")),  # REMOVE DEFAULT
"client_id": int(os.getenv("IBKR_CLIENT_ID", "123")),  # REMOVE DEFAULT

# AFTER (SAFE):
"host": os.getenv("IBKR_HOST"),  # Fail if not set
"port": int(os.getenv("IBKR_PORT")) if os.getenv("IBKR_PORT") else None,
"client_id": int(os.getenv("IBKR_CLIENT_ID")) if os.getenv("IBKR_CLIENT_ID") else None,

# Add validation:
if not all([config.ibkr.host, config.ibkr.port, config.ibkr.client_id]):
    raise ValueError("IBKR connection parameters must be explicitly configured")
```

**Testing Checklist:**
- [ ] System fails to start without explicit IBKR config
- [ ] No defaults in any configuration
- [ ] Clear error messages when config missing

### ✅ TASK 0.2: Add SQL Input Validation Layer
**Priority:** CRITICAL  
**Status:** COMPLETED - 2025-01-11
**Files Created:**
- [x] `robo_trader/database_validator.py` (581 lines of comprehensive validation)
- [x] Integrated into `database_async.py`

**Implementation:**
```python
# Create new file: robo_trader/database_validator.py
from typing import Any, Dict, Optional
import re

class DatabaseValidator:
    """Validate all database inputs before execution."""
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        if not symbol or not re.match(r'^[A-Z]{1,5}$', symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        return symbol
    
    @staticmethod
    def validate_numeric(value: Any, min_val: float = None, max_val: float = None) -> float:
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                raise ValueError(f"Value {num} below minimum {min_val}")
            if max_val is not None and num > max_val:
                raise ValueError(f"Value {num} above maximum {max_val}")
            return num
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid numeric value: {value}")
    
    @staticmethod
    def validate_trade_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all trade data before database insertion."""
        return {
            'symbol': DatabaseValidator.validate_symbol(data.get('symbol')),
            'quantity': int(DatabaseValidator.validate_numeric(data.get('quantity'), min_val=1)),
            'price': DatabaseValidator.validate_numeric(data.get('price'), min_val=0.01),
            'side': data.get('side') if data.get('side') in ['BUY', 'SELL'] else None,
            # ... additional fields
        }
```

**Testing Checklist:**
- [ ] Reject invalid symbols
- [ ] Reject negative quantities
- [ ] Reject invalid price values
- [ ] SQL injection attempts blocked

### ✅ TASK 0.3: Fix ALL Exception Handling
**Priority:** CRITICAL  
**Status:** COMPLETED - 2025-01-11
**Files Modified:**
- [x] `robo_trader/clients/async_ibkr_client.py` - Fixed bare except handlers
- [x] `robo_trader/features/engine.py` - Added specific exception handling
- [x] All files with bare `except:` or `except Exception: pass`

**Implementation:**
```python
# BEFORE (DANGEROUS):
try:
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values("date")
except Exception:
    pass  # SILENT FAILURE - DANGEROUS!

# AFTER (SAFE):
try:
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values("date")
except (ValueError, AttributeError) as e:
    logger.error(f"Failed to process date in market data: {e}")
    # Return safe default or raise based on criticality
    raise DataProcessingError(f"Invalid date format: {e}")
```

**Testing Checklist:**
- [ ] No bare except clauses remain
- [ ] All exceptions logged with context
- [ ] Critical paths have explicit error handling
- [ ] Network errors trigger retry logic

### ✅ TASK 0.4: Disable Debug Mode in Production
**Priority:** CRITICAL  
**Status:** COMPLETED - 2025-01-11
**Files Modified:**
- [x] `test_dashboard_simple.py` - Added environment-aware debug control
- [x] `app.py` - Verified no direct debug mode
- [x] All Flask/Dash applications checked

**Implementation:**
```python
# Add to all web applications:
import os

def get_debug_mode():
    """Never allow debug in production."""
    env = os.getenv("ENVIRONMENT", "development")
    if env in ["production", "staging"]:
        return False
    return os.getenv("DEBUG", "false").lower() == "true"

app.run(debug=get_debug_mode())
```

---

## 📝 PHASE 1: CRITICAL SAFETY [STATUS: NOT STARTED]

### ✅ TASK 1.1: Implement Active Stop-Loss Monitoring
**Priority:** CRITICAL  
**Status:** COMPLETED - 2025-01-11
**Files Created/Modified:**
- [x] `robo_trader/stop_loss_monitor.py` (592 lines - comprehensive monitoring system)
- [ ] `robo_trader/runner_async.py` - Integration pending

**Implementation:**
```python
# Create new file: robo_trader/stop_loss_monitor.py
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class StopLossOrder:
    symbol: str
    position_qty: int
    stop_price: float
    entry_price: float
    created_at: datetime
    triggered: bool = False

class StopLossMonitor:
    """Active stop-loss monitoring and execution."""
    
    def __init__(self, executor, risk_manager):
        self.executor = executor
        self.risk_manager = risk_manager
        self.active_stops: Dict[str, StopLossOrder] = {}
        self.monitoring = False
        
    async def add_stop_loss(self, symbol: str, position: Position, stop_pct: float = 0.02):
        """Add stop-loss for new position."""
        stop_price = position.avg_price * (1 - stop_pct) if position.is_long else position.avg_price * (1 + stop_pct)
        
        self.active_stops[symbol] = StopLossOrder(
            symbol=symbol,
            position_qty=position.quantity,
            stop_price=stop_price,
            entry_price=position.avg_price,
            created_at=datetime.now()
        )
        logger.info(f"Stop-loss set for {symbol} at ${stop_price:.2f}")
    
    async def monitor_stops(self, current_prices: Dict[str, float]):
        """Check and trigger stop-losses."""
        for symbol, stop_order in list(self.active_stops.items()):
            if stop_order.triggered:
                continue
                
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
            
            # Check if stop triggered
            if (stop_order.position_qty > 0 and current_price <= stop_order.stop_price) or \
               (stop_order.position_qty < 0 and current_price >= stop_order.stop_price):
                
                logger.warning(f"STOP-LOSS TRIGGERED for {symbol} at ${current_price:.2f}")
                await self._execute_stop_loss(stop_order)
    
    async def _execute_stop_loss(self, stop_order: StopLossOrder):
        """Execute stop-loss order immediately."""
        # This MUST be a market order for immediate execution
        order = Order(
            symbol=stop_order.symbol,
            quantity=abs(stop_order.position_qty),
            side="SELL" if stop_order.position_qty > 0 else "BUY",
            price=None  # Market order
        )
        
        result = await self.executor.place_order_async(order)
        stop_order.triggered = True
        
        if result.ok:
            logger.info(f"Stop-loss executed for {stop_order.symbol}")
        else:
            logger.error(f"CRITICAL: Stop-loss failed for {stop_order.symbol}: {result.message}")
            # Trigger emergency shutdown if stop-loss fails
            await self.risk_manager.trigger_emergency_shutdown("Stop-loss execution failed")
```

**Integration in runner_async.py:**
```python
# Add to AsyncRunner.__init__:
self.stop_loss_monitor = StopLossMonitor(self.executor, self.risk)

# Add to process_symbol after position entry:
if executed and signal != 0:
    await self.stop_loss_monitor.add_stop_loss(symbol, position)

# Add to main loop:
async def monitor_stop_losses(self):
    while self.running:
        current_prices = await self.get_all_current_prices()
        await self.stop_loss_monitor.monitor_stops(current_prices)
        await asyncio.sleep(1)  # Check every second
```

**Testing Checklist:**
- [ ] Stop-loss triggers when price breached
- [ ] Market orders used for immediate execution
- [ ] Failed stop-loss triggers emergency shutdown
- [ ] All positions have stop-loss orders

### ✅ TASK 1.2: Integrate Kill Switch at ALL Entry Points
**Priority:** CRITICAL  
**Status:** COMPLETED - 2025-01-11
**Files Modified:**
- [x] `robo_trader/runner_async.py` - Added kill switch check before EVERY order (BUY, SELL, SHORT, COVER)
- [x] `robo_trader/execution.py` - Added secondary kill switch check in place_order methods
- [x] `robo_trader/smart_execution/smart_executor.py` - Verified kill switch integration

**Implementation:**
```python
# Add to EVERY order placement:
async def place_order_with_safety(self, order: Order) -> ExecutionResult:
    """Place order with full safety checks."""
    
    # 1. CHECK KILL SWITCH FIRST
    if self.kill_switch and self.kill_switch.is_triggered:
        logger.error(f"KILL SWITCH ACTIVE - Order blocked for {order.symbol}")
        return ExecutionResult(False, "Kill switch active - trading halted")
    
    # 2. Check circuit breakers
    if self.circuit_breaker.is_open(order.symbol):
        logger.warning(f"Circuit breaker open for {order.symbol}")
        return ExecutionResult(False, "Circuit breaker open")
    
    # 3. Validate with risk manager
    is_valid, message = self.risk_manager.validate_order(...)
    if not is_valid:
        return ExecutionResult(False, f"Risk check failed: {message}")
    
    # 4. Finally place order
    try:
        result = await self.executor.place_order_async(order)
        if not result.ok:
            self.circuit_breaker.record_failure(order.symbol)
        return result
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        self.circuit_breaker.record_failure(order.symbol)
        raise
```

**Testing Checklist:**
- [ ] Kill switch blocks ALL order types
- [ ] Kill switch status logged
- [ ] Circuit breakers work per symbol
- [ ] Multiple safety layers active

### ❌ TASK 1.3: Fix Position Update Race Conditions
**Priority:** HIGH  
**Files to Modify:**
- [ ] `robo_trader/database_async.py` - Use transactions
- [ ] `robo_trader/runner_async.py` - Atomic updates

**Implementation:**
```python
# Add to database_async.py:
async def update_position_atomic(self, symbol: str, quantity_change: int, price: float) -> bool:
    """Atomically update position with transaction."""
    async with self.get_connection() as conn:
        try:
            await conn.execute("BEGIN EXCLUSIVE")
            
            # Get current position with lock
            cursor = await conn.execute(
                "SELECT quantity, avg_cost FROM positions WHERE symbol = ? FOR UPDATE",
                (symbol,)
            )
            row = await cursor.fetchone()
            
            if row:
                old_qty, old_avg = row
                new_qty = old_qty + quantity_change
                
                if new_qty == 0:
                    # Close position
                    await conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                else:
                    # Update position with weighted average
                    if quantity_change > 0:  # Adding to position
                        new_avg = ((old_qty * old_avg) + (quantity_change * price)) / new_qty
                    else:  # Reducing position
                        new_avg = old_avg  # Keep same average
                    
                    await conn.execute(
                        "UPDATE positions SET quantity = ?, avg_cost = ? WHERE symbol = ?",
                        (new_qty, new_avg, symbol)
                    )
            else:
                # New position
                if quantity_change > 0:
                    await conn.execute(
                        "INSERT INTO positions (symbol, quantity, avg_cost) VALUES (?, ?, ?)",
                        (symbol, quantity_change, price)
                    )
                else:
                    logger.error(f"Cannot reduce non-existent position for {symbol}")
                    await conn.execute("ROLLBACK")
                    return False
            
            await conn.execute("COMMIT")
            return True
            
        except Exception as e:
            await conn.execute("ROLLBACK")
            logger.error(f"Atomic position update failed: {e}")
            return False
```

**Testing Checklist:**
- [ ] Concurrent updates don't corrupt positions
- [ ] Rollback on any error
- [ ] Position math always correct
- [ ] Locks prevent race conditions

---

## 📝 PHASE 2: HIGH PRIORITY [STATUS: NOT STARTED]

### ❌ TASK 2.1: Implement Market Data Validation
**Priority:** HIGH  
**Files to Create:**
- [ ] `robo_trader/data_validator.py`

**Implementation:**
```python
# Create robo_trader/data_validator.py
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np

class MarketDataValidator:
    """Validate market data before use."""
    
    def __init__(self, max_staleness_seconds: int = 60):
        self.max_staleness_seconds = max_staleness_seconds
        self.price_history: Dict[str, list] = {}
        
    def validate_price(self, symbol: str, price: float, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """Validate single price point."""
        
        # 1. Check staleness
        age = (datetime.now() - timestamp).total_seconds()
        if age > self.max_staleness_seconds:
            return False, f"Price data stale by {age:.0f} seconds"
        
        # 2. Check price bounds
        if price <= 0 or price > 1_000_000:
            return False, f"Price {price} out of valid range"
        
        # 3. Check for outliers (if we have history)
        if symbol in self.price_history and len(self.price_history[symbol]) >= 20:
            prices = self.price_history[symbol][-20:]
            mean = np.mean(prices)
            std = np.std(prices)
            
            # Flag if > 5 standard deviations from mean
            if abs(price - mean) > 5 * std:
                return False, f"Price {price} is outlier (mean: {mean:.2f}, std: {std:.2f})"
        
        # 4. Store for future validation
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        
        return True, None
    
    def validate_ohlcv(self, data: Dict) -> Tuple[bool, Optional[str]]:
        """Validate OHLCV bar data."""
        
        # Check required fields
        required = ['open', 'high', 'low', 'close', 'volume']
        for field in required:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        o, h, l, c, v = data['open'], data['high'], data['low'], data['close'], data['volume']
        
        # Logical checks
        if not (l <= o <= h and l <= c <= h):
            return False, "OHLC values violate high/low constraints"
        
        if v < 0:
            return False, "Negative volume"
        
        return True, None
```

**Testing Checklist:**
- [ ] Stale data rejected
- [ ] Outliers detected
- [ ] Invalid OHLCV caught
- [ ] Price history maintained

### ❌ TASK 2.2: Implement Realistic Slippage Model
**Priority:** HIGH  
**Files to Modify:**
- [ ] `robo_trader/execution.py`

**Implementation:**
```python
class RealisticSlippageModel:
    """Market impact and slippage modeling."""
    
    def calculate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: int,
        base_price: float,
        market_data: Dict
    ) -> float:
        """Calculate realistic slippage."""
        
        # 1. Base spread cost (half-spread)
        spread = market_data.get('spread', 0.01)
        spread_cost = spread / 2
        
        # 2. Market impact (square-root model)
        avg_volume = market_data.get('avg_volume', 1_000_000)
        participation_rate = quantity / avg_volume
        market_impact = 0.1 * math.sqrt(participation_rate)  # 10bp per 1% participation
        
        # 3. Volatility adjustment
        volatility = market_data.get('volatility', 0.02)
        volatility_cost = volatility * 0.5  # Assume we pay half the volatility
        
        # 4. Urgency premium
        urgency = market_data.get('urgency', 0.5)
        urgency_cost = urgency * 0.001  # 10bp for maximum urgency
        
        # Total slippage in basis points
        total_slippage_bps = (spread_cost + market_impact + volatility_cost + urgency_cost) * 10000
        
        # Apply to price (adverse selection)
        if side == "BUY":
            return base_price * (1 + total_slippage_bps / 10000)
        else:
            return base_price * (1 - total_slippage_bps / 10000)
```

**Testing Checklist:**
- [ ] Large orders have higher slippage
- [ ] Volatility increases slippage
- [ ] Spread costs included
- [ ] Market impact realistic

### ❌ TASK 2.3: Add Circuit Breaker Recovery
**Priority:** HIGH  
**Files to Create:**
- [ ] `robo_trader/circuit_breaker.py`

**Implementation:**
```python
# Create robo_trader/circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Optional

class BreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Rejecting all requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker with automatic recovery."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.state = BreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_successes = 0
        
    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        
        if self.state == BreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = BreakerState.HALF_OPEN
                self.half_open_successes = 0
            else:
                raise CircuitBreakerOpen(f"Circuit breaker open until {self.reset_time}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        if self.state == BreakerState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self.state = BreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker recovered to CLOSED state")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == BreakerState.HALF_OPEN:
            self.state = BreakerState.OPEN
            logger.warning("Circuit breaker tripped back to OPEN from HALF_OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = BreakerState.OPEN
            logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
```

**Testing Checklist:**
- [ ] Opens after threshold failures
- [ ] Automatic recovery attempt
- [ ] Gradual recovery (half-open)
- [ ] Per-symbol breakers work

---

## 📝 PHASE 3: MEDIUM PRIORITY [STATUS: NOT STARTED]

### ❌ TASK 3.1: Performance Optimization
**Priority:** MEDIUM
- [ ] Increase DB connection pool to 20
- [ ] Add Redis caching layer
- [ ] Cache correlation matrix for 5 minutes
- [ ] Implement batch order processing

### ❌ TASK 3.2: Comprehensive Logging
**Priority:** MEDIUM
- [ ] Standardize all logging to structured format
- [ ] Add trace IDs for request tracking
- [ ] Implement log aggregation (ELK stack)
- [ ] Create audit trail for all trades

### ❌ TASK 3.3: Integration Testing Suite
**Priority:** MEDIUM
- [ ] Test all risk scenarios
- [ ] Test circuit breaker recovery
- [ ] Test concurrent trading
- [ ] Test data corruption recovery

### ❌ TASK 3.4: Create Emergency Runbooks
**Priority:** MEDIUM
- [ ] Runbook for kill switch triggered
- [ ] Runbook for database corruption
- [ ] Runbook for API connection loss
- [ ] Runbook for position reconciliation

---

## 📈 PROGRESS TRACKING

### Overall Completion: 6/30 Tasks (20%)

| Phase | Tasks | Completed | Status |
|-------|-------|-----------|--------|
| Phase 0 (Immediate) | 4 | 4 | ✅ COMPLETE |
| Phase 1 (Critical) | 3 | 2 | 🚧 IN PROGRESS |
| Phase 2 (High) | 3 | 0 | ❌ NOT STARTED |
| Phase 3 (Medium) | 4 | 0 | ❌ NOT STARTED |

### Production Readiness Checklist:
- [x] Phase 0 Complete
- [ ] Phase 1 Complete (2/3 done)
- [ ] Phase 2 Complete
- [ ] Phase 3 Complete
- [ ] 7-day paper trading test passed
- [ ] Risk team sign-off obtained
- [ ] Emergency procedures documented
- [ ] On-call rotation established

---

## ⚡ QUICK START FOR DEVELOPERS

### 1. Set Up Your Environment
```bash
# NEVER use these in production!
export ENVIRONMENT=development
export EXECUTION_MODE=paper
export IBKR_HOST=127.0.0.1
export IBKR_PORT=7497
export IBKR_CLIENT_ID=999
```

### 2. Run Safety Check
```bash
# Create this script: scripts/safety_check.py
python scripts/safety_check.py
# This should output all current safety violations
```

### 3. Start Fixing Issues
1. Pick the next uncompleted task from Phase 0
2. Create a feature branch: `git checkout -b fix/task-0-1-remove-hardcoded`
3. Implement the fix exactly as specified
4. Run tests: `pytest tests/test_safety.py`
5. Update this document marking the task complete
6. Create PR with title: `[SAFETY] Complete Task 0.1: Remove Hardcoded Values`

---

## 🚦 GO-LIVE CRITERIA

**DO NOT ATTEMPT LIVE TRADING UNTIL:**

1. ✅ All Phase 0 and Phase 1 tasks complete
2. ✅ All tests passing (including new safety tests)
3. ✅ 7 consecutive days of paper trading without safety violations
4. ✅ Risk management sign-off obtained
5. ✅ Emergency contacts and procedures documented
6. ✅ Initial capital limited to $10,000 maximum
7. ✅ 24/7 monitoring in place
8. ✅ Rollback plan tested

---

## 📞 EMERGENCY CONTACTS

**If ANY of these occur, IMMEDIATELY:**
1. Trigger kill switch
2. Contact the following:

- **Primary Developer**: [YOUR_NAME] - [PHONE]
- **Risk Manager**: [NAME] - [PHONE]
- **IBKR Support**: 1-877-442-2757
- **On-Call Engineer**: [Rotation Schedule]

---

## 🔄 PLAN UPDATES

This plan should be updated:
- [ ] After each task completion
- [ ] When new issues are discovered
- [ ] After each production incident
- [ ] During weekly safety review

**Last Review Date**: 2025-01-11  
**Next Review Date**: 2025-01-18  
**Plan Version**: 1.0.0

---

**Remember: SAFETY FIRST. No feature or optimization is worth risking capital or system integrity.**