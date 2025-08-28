"""
WebSocket client for sending updates to the dashboard server.
This ensures runner_async can send updates to the already-running server.
"""

import json
import asyncio
import websockets
from typing import Optional
from datetime import datetime
import threading
from queue import Queue, Empty
from robo_trader.logger import get_logger

logger = get_logger(__name__)


class WebSocketClient:
    """Client for sending updates to the WebSocket server."""
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.connected = False
        self.message_queue = Queue()
        self.thread = None
        self.loop = None
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            logger.info(f"Connected to WebSocket server at {self.uri}")
            
            # Receive initial connection message
            await self.websocket.recv()
            
            return True
        except Exception as e:
            logger.debug(f"Could not connect to WebSocket server: {e}")
            self.connected = False
            return False
    
    async def send_message(self, message: dict):
        """Send a message to the server."""
        if not self.connected or not self.websocket:
            return False
            
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.debug(f"Error sending message: {e}")
            self.connected = False
            return False
    
    async def process_queue(self):
        """Process messages from the queue."""
        while True:
            try:
                # Try to connect if not connected
                if not self.connected:
                    await self.connect()
                    if not self.connected:
                        await asyncio.sleep(5)  # Wait before retry
                        continue
                
                # Process messages
                while not self.message_queue.empty():
                    try:
                        message = self.message_queue.get_nowait()
                        await self.send_message(message)
                    except Empty:
                        break
                        
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.debug(f"Error in process_queue: {e}")
                self.connected = False
                await asyncio.sleep(1)
    
    def run_in_thread(self):
        """Run the client in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self.process_queue())
        except Exception as e:
            logger.debug(f"WebSocket client error: {e}")
        finally:
            if self.websocket:
                self.loop.run_until_complete(self.websocket.close())
            self.loop.close()
    
    def start(self):
        """Start the client in a background thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run_in_thread, daemon=True)
            self.thread.start()
            logger.debug("WebSocket client thread started")
    
    def send_market_update(self, symbol: str, price: float,
                          bid: Optional[float] = None, ask: Optional[float] = None,
                          volume: Optional[int] = None):
        """Queue a market data update."""
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
        """Queue a trade update."""
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
    
    def send_signal_update(self, symbol: str, signal: str, strength: float,
                          reason: Optional[str] = None):
        """Queue a signal update."""
        message = {
            "type": "signal",
            "symbol": symbol,
            "signal": signal,
            "strength": strength,
            "timestamp": datetime.now().isoformat()
        }
        
        if reason:
            message["reason"] = reason
            
        self.message_queue.put(message)


# Global client instance for runner_async to use
ws_client = WebSocketClient()
ws_client.start()