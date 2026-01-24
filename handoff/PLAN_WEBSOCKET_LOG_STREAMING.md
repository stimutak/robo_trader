# WebSocket Log Streaming Implementation Plan

**Date:** 2026-01-23
**Status:** IN PROGRESS
**Branch:** `feature/websocket-log-streaming`

---

## Objective

Enable real-time log streaming to the mobile app via WebSocket.

---

## Implementation Steps

### Step 1: Add `send_log_message()` to WebSocketManager

**File:** `robo_trader/websocket_server.py`

Add method after `send_performance_update()` (around line 258):

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
        "level": level.upper(),
        "source": source,
        "message": message,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
    }
    self.message_queue.put(log_message)
```

### Step 2: Create WebSocket Log Processor

**File:** `robo_trader/logger.py`

Add a new structlog processor class that forwards logs to WebSocket:

```python
class WebSocketLogProcessor:
    """Structlog processor that forwards logs to WebSocket."""

    _ws_manager = None  # Lazy-loaded to avoid circular import

    @classmethod
    def set_ws_manager(cls, ws_manager):
        """Set the WebSocket manager instance."""
        cls._ws_manager = ws_manager

    def __call__(self, logger, method_name, event_dict):
        """Forward log to WebSocket if manager is set."""
        if self._ws_manager is None:
            return event_dict

        level = method_name.upper()

        # Only stream INFO and above
        if level in ("INFO", "WARNING", "ERROR", "CRITICAL"):
            try:
                self._ws_manager.send_log_message(
                    level=level,
                    source=event_dict.get("logger", "robo_trader"),
                    message=event_dict.get("event", str(event_dict)),
                    context={k: v for k, v in event_dict.items()
                            if k not in ("event", "logger", "timestamp", "level")},
                )
            except Exception:
                pass  # Don't let log streaming failures break logging

        return event_dict
```

Add to processor chain in `setup_structlog()` before the final renderer.

### Step 3: Wire Up in WebSocket Server

**File:** `robo_trader/websocket_server.py`

After creating `ws_manager`, register it with the logger:

```python
# At module level, after ws_manager = WebSocketManager()
from robo_trader.logger import WebSocketLogProcessor
WebSocketLogProcessor.set_ws_manager(ws_manager)
```

---

## Circular Import Prevention

The key challenge is avoiding circular imports:
- `websocket_server.py` imports `logger.py` for logging
- `logger.py` needs access to `ws_manager` for streaming

**Solution:** Use late binding via class method `set_ws_manager()`. The processor is added to the chain but only sends to WebSocket after the manager is registered.

---

## Testing

1. Start WebSocket server: `python3 -m robo_trader.websocket_server`
2. Connect with wscat: `wscat -c ws://localhost:8765`
3. Start trader: `./START_TRADER.sh`
4. Verify log messages appear in wscat

---

## Mobile App Expected Format

```typescript
interface LogMessage {
  type: 'log';
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  source: string;
  message: string;
  context?: Record<string, any>;
  timestamp: string;
}
```

---

## Checklist

- [x] Add `send_log_message()` to `websocket_server.py`
- [x] Add `WebSocketLogProcessor` class to `logger.py`
- [x] Add processor to structlog chain in `setup_structlog()`
- [x] Register ws_manager with processor in `websocket_server.py`
- [x] Test with wscat
- [x] Export `WebSocketLogProcessor` in `__all__`

---

**Status:** COMPLETE - Verified 2026-01-23
