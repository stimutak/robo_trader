# WebSocket Log Streaming Enhancement

**Date:** 2026-01-23
**Target Repo:** `/Users/oliver/robo_trader` (main repo, NOT the mobile worktree)
**File to Modify:** `robo_trader/websocket_server.py`

---

## Overview

The mobile app has a real-time log viewer that connects to `ws://localhost:8765`. Currently the WebSocket server broadcasts market data, trades, positions, and signals - but NOT log messages.

This enhancement adds structured log streaming so the mobile app can display live logs.

---

## Current State

**WebSocket Server (`robo_trader/websocket_server.py`)** already has:
- `send_market_update()` - market prices
- `send_trade_update()` - executed trades
- `send_position_update()` - position changes
- `send_signal_update()` - trading signals
- `send_performance_update()` - performance metrics

**Missing:** `send_log_message()` for streaming logs

---

## Required Changes

### 1. Add Log Message Method

Add to `WebSocketManager` class in `websocket_server.py`:

```python
def send_log_message(
    self,
    level: str,
    source: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
):
    """Queue a log message for broadcast to connected clients."""
    log_message = {
        "type": "log",
        "level": level.upper(),  # DEBUG, INFO, WARNING, ERROR
        "source": source,        # Module name or component
        "message": message,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
    }
    self.message_queue.put(log_message)
```

### 2. Create WebSocket Log Handler

Create a custom structlog processor or logging handler that sends logs to WebSocket:

```python
# In robo_trader/logger.py or new file robo_trader/websocket_log_handler.py

class WebSocketLogHandler:
    """Forwards log messages to WebSocket for real-time streaming."""

    def __init__(self, ws_manager):
        self.ws_manager = ws_manager
        self.min_level = "INFO"  # Don't stream DEBUG by default

    def __call__(self, logger, method_name, event_dict):
        """Structlog processor that sends logs to WebSocket."""
        level = method_name.upper()

        # Only stream INFO and above
        if level in ("INFO", "WARNING", "ERROR", "CRITICAL"):
            self.ws_manager.send_log_message(
                level=level,
                source=event_dict.get("logger", "robo_trader"),
                message=event_dict.get("event", str(event_dict)),
                context={k: v for k, v in event_dict.items()
                        if k not in ("event", "logger", "timestamp", "level")},
            )

        return event_dict  # Continue processing chain
```

### 3. Wire Handler into Logger

In `robo_trader/logger.py`, add the WebSocket handler to the structlog processor chain:

```python
from robo_trader.websocket_server import ws_manager

# Add to processor chain (after timestamp, before output)
processors = [
    # ... existing processors ...
    WebSocketLogHandler(ws_manager),
    # ... output processor ...
]
```

### 4. Mobile App Expected Format

The mobile app expects this message format:

```typescript
interface LogMessage {
  type: 'log';
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  source: string;      // e.g., "runner_async", "ml.training"
  message: string;     // The log message text
  context?: {          // Optional structured data
    symbol?: string;
    price?: number;
    // ... any key-value pairs
  };
  timestamp: string;   // ISO format
}
```

---

## Testing

1. Start WebSocket server: `python3 -m robo_trader.websocket_server`
2. Connect with wscat: `wscat -c ws://localhost:8765`
3. In another terminal, run trader: `./START_TRADER.sh`
4. Verify log messages appear in wscat output

---

## Mobile App Integration

The mobile app already has:
- `stores/logsStore.ts` - Zustand store for log state
- `hooks/useWebSocket.ts` - WebSocket connection hook
- `app/(tabs)/logs.tsx` - Log viewer UI with filters

Once the backend streams logs, the mobile app will automatically display them.

---

## Optional Enhancements

1. **Log Level Filter** - Allow clients to subscribe to specific levels only
2. **Source Filter** - Allow filtering by source module
3. **Rate Limiting** - Throttle high-volume DEBUG logs
4. **Backfill** - Send last N logs on connection

---

## Files to Modify

| File | Change |
|------|--------|
| `robo_trader/websocket_server.py` | Add `send_log_message()` method |
| `robo_trader/logger.py` | Add WebSocket handler to processor chain |
| (optional) `robo_trader/websocket_log_handler.py` | New file for handler class |

---

**Ready to implement in main repo.**
