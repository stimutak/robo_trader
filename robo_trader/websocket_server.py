"""
WebSocket server for real-time dashboard updates.
"""

import asyncio
import hmac
import json
import os
import threading
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, Optional, Set
from urllib.parse import parse_qs, urlsplit

import websockets
from websockets.server import WebSocketServerProtocol

from robo_trader.logger import get_logger

logger = get_logger(__name__)


def _allowed_ws_origins(port: int) -> Set[Optional[str]]:
    """Whitelist of acceptable Origin header values for the WS handshake."""
    allowed: Set[Optional[str]] = {
        None,
        "null",
        f"http://localhost:{port}",
        f"http://127.0.0.1:{port}",
    }
    extra = os.getenv("WS_ALLOWED_ORIGINS", "").strip()
    if extra:
        for origin in extra.split(","):
            origin = origin.strip()
            if origin:
                allowed.add(origin)
    return allowed


class WebSocketManager:
    """Manages WebSocket connections and broadcasts updates."""

    # W-H3: bind to loopback by default; override with WS_HOST env var.
    def __init__(self, host: Optional[str] = None, port: int = 8765):
        self.host = host if host is not None else os.getenv("WS_HOST", "127.0.0.1")
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.message_queue: Queue[Dict[str, Any]] = Queue()
        self.server = None
        self.loop = None
        self.thread = None
        # W-M1: optional shared secret for authenticating WS clients via ?token=...
        self.auth_token = os.getenv("WS_AUTH_TOKEN", "")

    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new client connection."""
        self.clients.add(websocket)
        # Log without including the websocket object itself to avoid serialization issues
        logger.info(
            f"WebSocket client connected",
            client_count=len(self.clients),
            remote_address=(
                str(websocket.remote_address) if hasattr(websocket, "remote_address") else None
            ),
        )

        # Send initial connection message
        await websocket.send(
            json.dumps(
                {
                    "type": "connection",
                    "status": "connected",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )

    def unregister_client(self, websocket: WebSocketServerProtocol):
        """Remove a client connection."""
        if websocket in self.clients:
            self.clients.remove(websocket)
            # Log without including the websocket object itself to avoid serialization issues
            logger.info(
                f"WebSocket client disconnected",
                client_count=len(self.clients),
                remote_address=(
                    str(websocket.remote_address) if hasattr(websocket, "remote_address") else None
                ),
            )

    def _authorize_handshake(self, websocket: WebSocketServerProtocol, path: str) -> bool:
        """Validate Origin and (optionally) auth token at handshake (W-H3, W-M1)."""
        # Origin whitelist
        origin = None
        try:
            origin = websocket.request_headers.get("Origin")
        except Exception:
            origin = None
        if origin not in _allowed_ws_origins(self.port):
            logger.warning(f"WS rejected: bad Origin {origin!r}")
            return False

        # Auth: prefer token if configured; else require loopback peer.
        if self.auth_token:
            try:
                query = urlsplit(path).query
                token = (parse_qs(query).get("token") or [""])[0]
            except Exception:
                token = ""
            if not token or not hmac.compare_digest(token, self.auth_token):
                logger.warning("WS rejected: missing/invalid token")
                return False
        else:
            # No token configured -> only allow connections from loopback.
            try:
                peer_ip = websocket.remote_address[0] if websocket.remote_address else ""
            except Exception:
                peer_ip = ""
            if peer_ip not in ("127.0.0.1", "::1"):
                logger.warning(f"WS rejected: non-loopback peer {peer_ip} without WS_AUTH_TOKEN")
                return False
        return True

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str = "/"):
        """Handle a client connection.

        Args:
            websocket: The WebSocket connection
            path: The request path (required by websockets library)
        """
        # W-H3 / W-M1: enforce origin and (optional) token before accepting.
        if not self._authorize_handshake(websocket, path):
            try:
                await websocket.close(code=1008, reason="unauthorized")
            except Exception:
                pass
            return

        await self.register_client(websocket)
        try:
            # Keep connection alive and handle any incoming messages.
            # W-H3: do NOT rebroadcast client-supplied payloads. Trusted producers
            # call broadcast() / send_*_update() directly via the queue.
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    if msg_type == "subscribe":
                        symbols = data.get("symbols", [])
                        logger.info(f"Client subscribed to symbols: {symbols}")
                    else:
                        # Log only; never echo to other clients.
                        logger.debug(f"Ignoring inbound WS message of type {msg_type}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.unregister_client(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.clients:
            return

        message_str = json.dumps(message)
        disconnected = set()

        for client in self.clients.copy():
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        for client in disconnected:
            self.unregister_client(client)

    async def process_queue(self):
        """Process messages from the queue and broadcast them."""
        while True:
            try:
                # Check for messages in queue
                while not self.message_queue.empty():
                    try:
                        message = self.message_queue.get_nowait()
                        await self.broadcast(message)
                    except Empty:
                        break
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(1)

    async def start_server(self):
        """Start the WebSocket server."""
        # SEC-C11: refuse to bind a non-loopback host without a token.
        # Anyone on the LAN would otherwise get full WS access. The
        # handshake check also enforces this, but a startup-time refusal
        # is louder and catches misconfiguration before connections land.
        if self.host not in ("127.0.0.1", "localhost", "::1") and not self.auth_token:
            raise RuntimeError(
                f"WS_HOST={self.host!r} is non-loopback but WS_AUTH_TOKEN is unset. "
                "Refusing to start unauthenticated WebSocket server on LAN. "
                "Set WS_AUTH_TOKEN to a 32+ char random string."
            )
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")

        # Disable websockets library's own logging to prevent serialization issues
        import logging as stdlib_logging

        stdlib_logging.getLogger("websockets").setLevel(stdlib_logging.WARNING)
        stdlib_logging.getLogger("websockets.server").setLevel(stdlib_logging.WARNING)

        # Create server and queue processor tasks
        server = await websockets.serve(self.handle_client, self.host, self.port)

        # Run both server and queue processor
        await asyncio.gather(server.wait_closed(), self.process_queue())

    def run_in_thread(self):
        """Run the WebSocket server in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self.start_server())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self.loop.close()

    def start(self):
        """Start the WebSocket server in a background thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run_in_thread, daemon=True)
            self.thread.start()
            logger.info("WebSocket server thread started")

    def stop(self):
        """Stop the WebSocket server."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("WebSocket server stopped")

    def send_market_update(
        self,
        symbol: str,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[int] = None,
    ):
        """Queue a market data update for broadcast."""
        if not self.thread or not self.thread.is_alive():
            logger.debug(f"WebSocket server not running, skipping market update for {symbol}")
            return

        message = {
            "type": "market_data",
            "symbol": symbol,
            "price": price,
            "timestamp": datetime.now().isoformat(),
        }

        if bid is not None:
            message["bid"] = bid
        if ask is not None:
            message["ask"] = ask
        if volume is not None:
            message["volume"] = volume

        self.message_queue.put(message)
        logger.debug(
            f"Queued market update for {symbol} @ ${price:.2f}, queue size: {self.message_queue.qsize()}"
        )

    def send_trade_update(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        status: str = "executed",
        portfolio_id: str = "default",
    ):
        """Queue a trade update for broadcast."""
        message = {
            "type": "trade",
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

        self.message_queue.put(message)

    def send_position_update(self, positions: Dict[str, Any], portfolio_id: str = "default"):
        """Queue a position update for broadcast."""
        message = {
            "type": "positions",
            "portfolio_id": portfolio_id,
            "positions": positions,
            "timestamp": datetime.now().isoformat(),
        }

        self.message_queue.put(message)

    def send_signal_update(
        self, symbol: str, signal: str, strength: float, portfolio_id: str = "default"
    ):
        """Queue a trading signal update for broadcast."""
        message = {
            "type": "signal",
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "signal": signal,
            "strength": strength,
            "timestamp": datetime.now().isoformat(),
        }

        self.message_queue.put(message)

    def send_performance_update(self, metrics: Dict[str, Any], portfolio_id: str = "default"):
        """Queue a performance metrics update for broadcast."""
        message = {
            "type": "performance",
            "portfolio_id": portfolio_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        self.message_queue.put(message)

    def send_log_message(
        self,
        level: str,
        source: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Queue a log message for broadcast to connected clients.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            source: Module or component name
            message: The log message text
            context: Optional structured data (symbol, price, etc.)
        """
        log_message = {
            "type": "log",
            "level": level.upper(),
            "source": source,
            "message": message,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.message_queue.put(log_message)


# Global WebSocket manager instance
ws_manager = WebSocketManager()

# Register with logger for log streaming (late binding to avoid circular import)
from robo_trader.logger import WebSocketLogProcessor  # noqa: E402

WebSocketLogProcessor.set_ws_manager(ws_manager)


if __name__ == "__main__":
    """Run the WebSocket server standalone."""
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\nShutting down WebSocket server...")
        ws_manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Starting WebSocket server on ws://localhost:8765")
    print("Press Ctrl+C to stop")

    # Run the server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(ws_manager.start_server())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        loop.close()
        print("WebSocket server stopped")
