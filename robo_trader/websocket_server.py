"""
WebSocket server for real-time dashboard updates.
"""

import asyncio
import json
import logging
from typing import Set, Optional, Dict, Any
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts updates."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.message_queue = Queue()
        self.server = None
        self.loop = None
        self.thread = None
        
    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial connection message
        await websocket.send(json.dumps({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        }))
        
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Remove a client connection."""
        if websocket in self.clients:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
            
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a client connection."""
        await self.register_client(websocket)
        try:
            # Keep connection alive and handle any incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Handle client messages if needed (e.g., subscribe to specific symbols)
                    if data.get("type") == "subscribe":
                        symbols = data.get("symbols", [])
                        logger.info(f"Client subscribed to symbols: {symbols}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
            
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
            await self.unregister_client(client)
            
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
        
        # Create server and queue processor tasks
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        # Run both server and queue processor
        await asyncio.gather(
            server.wait_closed(),
            self.process_queue()
        )
        
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
        
    def send_market_update(self, symbol: str, price: float, 
                          bid: Optional[float] = None, ask: Optional[float] = None,
                          volume: Optional[int] = None):
        """Queue a market data update for broadcast."""
        message = {
            "type": "market_data",
            "symbol": symbol,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
        
        if bid is not None:
            message["bid"] = bid
        if ask is not None:
            message["ask"] = ask
        if volume is not None:
            message["volume"] = volume
            
        self.message_queue.put(message)
        
    def send_trade_update(self, symbol: str, side: str, quantity: int, 
                         price: float, status: str = "executed"):
        """Queue a trade update for broadcast."""
        message = {
            "type": "trade",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_queue.put(message)
        
    def send_position_update(self, positions: Dict[str, Any]):
        """Queue a position update for broadcast."""
        message = {
            "type": "positions",
            "positions": positions,
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_queue.put(message)
        
    def send_signal_update(self, symbol: str, signal: str, strength: float):
        """Queue a trading signal update for broadcast."""
        message = {
            "type": "signal",
            "symbol": symbol,
            "signal": signal,
            "strength": strength,
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_queue.put(message)
        
    def send_performance_update(self, metrics: Dict[str, Any]):
        """Queue a performance metrics update for broadcast."""
        message = {
            "type": "performance",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_queue.put(message)


# Global WebSocket manager instance
ws_manager = WebSocketManager()