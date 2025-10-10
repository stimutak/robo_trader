"""
WebSocket server for real-time dashboard updates.
"""

import asyncio
import json
import threading
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from robo_trader.logger import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts updates."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.message_queue: Queue[Dict[str, Any]] = Queue()
        self.server = None
        self.loop = None
        self.thread = None

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

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str = "/"):
        """Handle a client connection.

        Args:
            websocket: The WebSocket connection
            path: The request path (required by websockets library)
        """
        await self.register_client(websocket)
        try:
            # Keep connection alive and handle any incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.debug(f"Received message from client: {data.get('type', 'unknown')}")

                    # Handle client messages
                    if data.get("type") == "subscribe":
                        symbols = data.get("symbols", [])
                        logger.info(f"Client subscribed to symbols: {symbols}")
                    # Broadcast messages from runner client to all other clients
                    elif data.get("type") in ["market_data", "trade", "signal"]:
                        logger.info(
                            f"Broadcasting {data.get('type')} update for {data.get('symbol')}"
                        )
                        await self.broadcast(data)
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
        self, symbol: str, side: str, quantity: int, price: float, status: str = "executed"
    ):
        """Queue a trade update for broadcast."""
        message = {
            "type": "trade",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

        self.message_queue.put(message)

    def send_position_update(self, positions: Dict[str, Any]):
        """Queue a position update for broadcast."""
        message = {
            "type": "positions",
            "positions": positions,
            "timestamp": datetime.now().isoformat(),
        }

        self.message_queue.put(message)

    def send_signal_update(self, symbol: str, signal: str, strength: float):
        """Queue a trading signal update for broadcast."""
        message = {
            "type": "signal",
            "symbol": symbol,
            "signal": signal,
            "strength": strength,
            "timestamp": datetime.now().isoformat(),
        }

        self.message_queue.put(message)

    def send_performance_update(self, metrics: Dict[str, Any]):
        """Queue a performance metrics update for broadcast."""
        message = {
            "type": "performance",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        self.message_queue.put(message)


# Global WebSocket manager instance
ws_manager = WebSocketManager()


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
