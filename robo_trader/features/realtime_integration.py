"""
Real-time integration module connecting streaming features to trading system.
Completes Phase 3 S5 implementation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from robo_trader.features.streaming_features import (
    StreamingFeatureCalculator,
    StreamingFeatureStore,
)
from robo_trader.ml.online_inference import (
    ModelUpdateManager,
    OnlineModelInference,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class RealtimeFeaturePipeline:
    """
    Complete real-time feature pipeline integrating:
    - WebSocket data feed
    - Streaming feature calculation
    - Online model inference
    - Trading signal generation
    """

    def __init__(
        self,
        symbols: List[str],
        model_path: Optional[str] = None,
        enable_persistence: bool = True,
        enable_drift_detection: bool = True,
    ):
        """
        Initialize real-time pipeline.

        Args:
            symbols: List of symbols to track
            model_path: Path to ML model
            enable_persistence: Enable feature persistence
            enable_drift_detection: Enable drift monitoring
        """
        self.symbols = symbols
        self.enable_persistence = enable_persistence
        self.enable_drift_detection = enable_drift_detection

        # Initialize components
        self.feature_calculator = StreamingFeatureCalculator()
        self.inference_engine = OnlineModelInference(model_path=model_path)
        self.model_manager = ModelUpdateManager(self.inference_engine)

        if enable_persistence:
            self.feature_store = StreamingFeatureStore()
        else:
            self.feature_store = None

        # WebSocket integration
        self.websocket_clients: Dict[str, Any] = {}
        self.data_queues: Dict[str, asyncio.Queue] = {}

        # Signal tracking
        self.latest_signals: Dict[str, str] = {}
        self.signal_history: List[Dict] = []

        # Performance metrics
        self.metrics = {
            "updates_processed": 0,
            "predictions_made": 0,
            "signals_generated": 0,
            "drift_detections": 0,
            "errors": 0,
        }

        # Initialize symbols
        for symbol in symbols:
            self.data_queues[symbol] = asyncio.Queue(maxsize=100)
            self.latest_signals[symbol] = "HOLD"

    async def connect_websocket(self, websocket_url: str = "ws://localhost:8765"):
        """
        Connect to WebSocket server for real-time data.

        Args:
            websocket_url: WebSocket server URL
        """
        try:
            import websockets

            async with websockets.connect(websocket_url) as websocket:
                logger.info(f"Connected to WebSocket at {websocket_url}")

                # Subscribe to symbols
                subscribe_msg = {"action": "subscribe", "symbols": self.symbols}
                await websocket.send(str(subscribe_msg))

                # Listen for updates
                async for message in websocket:
                    await self._handle_websocket_message(message)

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.metrics["errors"] += 1

    async def _handle_websocket_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            import json

            data = json.loads(message)

            if "symbol" in data and data["symbol"] in self.symbols:
                # Queue data for processing
                await self.data_queues[data["symbol"]].put(data)

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            self.metrics["errors"] += 1

    async def process_market_update(self, update: Dict[str, Any]) -> Optional[PredictionResult]:
        """
        Process a market data update through the full pipeline.

        Args:
            update: Market data update dictionary

        Returns:
            PredictionResult if successful
        """
        try:
            symbol = update.get("symbol")
            if not symbol:
                return None

            # Extract market data
            price = update.get("price", update.get("close"))
            volume = update.get("volume", 0)
            high = update.get("high")
            low = update.get("low")
            timestamp = update.get("timestamp")

            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not timestamp:
                timestamp = datetime.now()

            # Update streaming features
            features = self.feature_calculator.update(
                symbol=symbol, price=price, volume=volume, high=high, low=low, timestamp=timestamp
            )

            self.metrics["updates_processed"] += 1

            # Check for feature drift
            if self.enable_drift_detection and self.metrics["updates_processed"] % 100 == 0:
                drift_result = self.feature_calculator.detect_drift(symbol)
                if drift_result["drift_detected"]:
                    logger.warning(f"Feature drift detected for {symbol}: {drift_result}")
                    self.metrics["drift_detections"] += 1

            # Store features if enabled
            if self.feature_store:
                self.feature_store.store_features(symbol, features, timestamp)

            # Make prediction
            model_name = self.model_manager.get_model_for_symbol(symbol)
            prediction = await self.inference_engine.predict_async(features, symbol, model_name)

            if prediction:
                self.metrics["predictions_made"] += 1

                # Generate trading signal
                signal = self.inference_engine.get_trading_signal(prediction)

                # Update signal tracking
                if signal != self.latest_signals[symbol]:
                    self.latest_signals[symbol] = signal
                    self.metrics["signals_generated"] += 1

                    # Record signal change
                    self.signal_history.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "signal": signal,
                            "prediction": prediction.prediction,
                            "confidence": prediction.confidence,
                        }
                    )

                    # Limit history
                    if len(self.signal_history) > 1000:
                        self.signal_history.pop(0)

                return prediction

        except Exception as e:
            logger.error(f"Error processing update for {symbol}: {e}")
            self.metrics["errors"] += 1

        return None

    async def run_streaming_pipeline(self):
        """Run the complete streaming pipeline."""
        logger.info(f"Starting streaming pipeline for {len(self.symbols)} symbols")

        # Create tasks for each symbol
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self._process_symbol_stream(symbol))
            tasks.append(task)

        # Run all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}")

    async def _process_symbol_stream(self, symbol: str):
        """Process streaming data for a single symbol."""
        logger.info(f"Processing stream for {symbol}")

        while True:
            try:
                # Get data from queue
                data = await self.data_queues[symbol].get()

                # Process through pipeline
                await self.process_market_update(data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stream processing for {symbol}: {e}")
                await asyncio.sleep(1)  # Brief pause on error

    def get_latest_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest calculated features for a symbol."""
        return self.feature_calculator.get_features(symbol)

    def get_latest_signal(self, symbol: str) -> str:
        """Get latest trading signal for a symbol."""
        return self.latest_signals.get(symbol, "HOLD")

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        inference_metrics = self.inference_engine.get_performance_metrics()

        return {
            "pipeline_metrics": self.metrics,
            "inference_metrics": inference_metrics,
            "active_symbols": len(self.symbols),
            "signals": dict(self.latest_signals),
            "feature_store_version": self.feature_store.get_latest_version()
            if self.feature_store
            else None,
        }

    async def update_model(self, new_model_path: str, rollout_pct: float = 0.1):
        """
        Update ML model with gradual rollout.

        Args:
            new_model_path: Path to new model
            rollout_pct: Percentage for initial rollout
        """
        model_name = f"model_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_manager.deploy_model(new_model_path, model_name, rollout_pct)

        if self.feature_store:
            self.feature_store.increment_version()

        logger.info(f"Deployed new model {model_name} with {rollout_pct*100}% rollout")

    def export_signal_history(self) -> pd.DataFrame:
        """Export signal history as DataFrame."""
        if not self.signal_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.signal_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df


class WebSocketIntegration:
    """
    Direct integration with existing WebSocket server.
    """

    def __init__(self, pipeline: RealtimeFeaturePipeline):
        """
        Initialize WebSocket integration.

        Args:
            pipeline: Real-time feature pipeline
        """
        self.pipeline = pipeline
        self.connected = False

    async def connect_to_server(self, host: str = "localhost", port: int = 8765):
        """Connect to existing WebSocket server."""
        try:
            from robo_trader.websocket_client import WebSocketClient

            self.client = WebSocketClient(f"ws://{host}:{port}")
            await self.client.connect()
            self.connected = True

            logger.info(f"Connected to WebSocket server at {host}:{port}")

            # Start processing messages
            await self._process_messages()

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server: {e}")
            self.connected = False

    async def _process_messages(self):
        """Process incoming WebSocket messages."""
        while self.connected:
            try:
                message = await self.client.receive()

                if message and isinstance(message, dict):
                    # Process market data updates
                    if message.get("type") == "market_data":
                        await self.pipeline.process_market_update(message.get("data", {}))

                    # Process price updates
                    elif message.get("type") == "price_update":
                        symbol = message.get("symbol")
                        if symbol in self.pipeline.symbols:
                            update = {
                                "symbol": symbol,
                                "price": message.get("price"),
                                "volume": message.get("volume", 0),
                                "timestamp": message.get("timestamp"),
                            }
                            await self.pipeline.process_market_update(update)

            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await asyncio.sleep(0.1)

    async def subscribe_to_symbols(self):
        """Subscribe to symbol updates."""
        if self.connected and self.client:
            for symbol in self.pipeline.symbols:
                await self.client.send({"action": "subscribe", "symbol": symbol})

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.connected and self.client:
            await self.client.disconnect()
            self.connected = False


async def run_realtime_system(
    symbols: List[str],
    model_path: Optional[str] = None,
    websocket_host: str = "localhost",
    websocket_port: int = 8765,
):
    """
    Run complete real-time feature and inference system.

    Args:
        symbols: List of symbols to track
        model_path: Path to ML model
        websocket_host: WebSocket server host
        websocket_port: WebSocket server port
    """
    # Initialize pipeline
    pipeline = RealtimeFeaturePipeline(
        symbols=symbols,
        model_path=model_path,
        enable_persistence=True,
        enable_drift_detection=True,
    )

    # Connect to WebSocket
    integration = WebSocketIntegration(pipeline)

    try:
        # Connect and subscribe
        await integration.connect_to_server(websocket_host, websocket_port)
        await integration.subscribe_to_symbols()

        # Run pipeline
        await pipeline.run_streaming_pipeline()

    except KeyboardInterrupt:
        logger.info("Shutting down real-time system")
    finally:
        await integration.disconnect()

        # Print final metrics
        metrics = pipeline.get_pipeline_metrics()
        logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    # Example usage
    symbols = ["AAPL", "NVDA", "TSLA"]

    asyncio.run(run_realtime_system(symbols))
