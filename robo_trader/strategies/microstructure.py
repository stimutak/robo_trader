"""Microstructure trading strategies for RoboTrader.

This module implements high-frequency strategies that exploit market microstructure patterns:
- Order book imbalance strategies
- Spread capture strategies  
- Liquidity provision strategies
- Sub-second momentum strategies
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Simple logger replacement
class SimpleLogger:
    def __init__(self, name):
        self.name = name

    def bind(self, **kwargs):
        return self

    def info(self, msg, **kwargs):
        print(f"INFO [{self.name}]: {msg}")

    def error(self, msg, **kwargs):
        print(f"ERROR [{self.name}]: {msg}")

logger = SimpleLogger(__name__)

# Standalone MarketMicrostructure to avoid import issues
@dataclass
class MarketMicrostructure:
    """Market microstructure data for execution decisions."""

    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    spread_bps: float
    mid_price: float
    imbalance: float  # (bid_size - ask_size) / (bid_size + ask_size)
    depth: Dict[str, List[Tuple[float, int]]]  # Price levels and sizes


class LiquidityProvider:
    """Mock liquidity provider."""

    async def get_liquidity_metrics(self, symbol: str) -> Dict[str, Any]:
        return {
            "avg_spread_bps": 5.0,
            "avg_volume": 1e6,
            "volatility": 0.02,
            "liquidity_score": 0.7,
        }

# Minimal strategy base to avoid import issues
class Strategy:
    def __init__(self, name: str):
        self.name = name

    async def analyze(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

logger = structlog.get_logger(__name__)


class MicrostructureSignal(Enum):
    """Microstructure-specific signal types."""
    
    SPREAD_CAPTURE = "spread_capture"
    IMBALANCE_LONG = "imbalance_long"
    IMBALANCE_SHORT = "imbalance_short"
    LIQUIDITY_PROVIDE = "liquidity_provide"
    MOMENTUM_MICRO = "momentum_micro"
    MEAN_REVERT_MICRO = "mean_revert_micro"


@dataclass
class OrderBookLevel:
    """Order book level data."""
    
    price: float
    size: int
    orders: int = 1


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_trade_price: float
    last_trade_size: int
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return 0.0
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2.0
        return self.last_trade_price
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance: (bid_size - ask_size) / (bid_size + ask_size)."""
        if self.best_bid and self.best_ask:
            bid_size = self.best_bid.size
            ask_size = self.best_ask.size
            total_size = bid_size + ask_size
            if total_size > 0:
                return (bid_size - ask_size) / total_size
        return 0.0


class OrderBookImbalanceStrategy(Strategy):
    """Strategy that trades on order book imbalances."""
    
    def __init__(
        self,
        name: str = "OrderBookImbalance",
        imbalance_threshold: float = 0.3,
        min_spread_bps: float = 2.0,
        max_spread_bps: float = 20.0,
        hold_time_seconds: int = 5,
        min_book_depth: int = 1000,
    ):
        super().__init__(name)
        self.imbalance_threshold = imbalance_threshold
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps
        self.hold_time_seconds = hold_time_seconds
        self.min_book_depth = min_book_depth
        
        # State tracking
        self.order_book_history: List[OrderBookSnapshot] = []
        self.liquidity_provider = LiquidityProvider()
        
        self.logger = logger.bind(strategy=name)
    
    async def analyze(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order book for imbalance opportunities."""
        
        # Get current order book snapshot
        order_book = await self._get_order_book_snapshot(symbol)
        
        if not order_book:
            return {"signal": SignalType.HOLD.value, "confidence": 0.0}
        
        # Store in history
        self.order_book_history.append(order_book)
        
        # Keep only recent history (last 100 snapshots)
        if len(self.order_book_history) > 100:
            self.order_book_history = self.order_book_history[-100:]
        
        # Check basic conditions
        if not self._check_basic_conditions(order_book):
            return {"signal": SignalType.HOLD.value, "confidence": 0.0}
        
        # Calculate imbalance signal
        imbalance = order_book.imbalance
        
        # Generate signal based on imbalance
        if abs(imbalance) > self.imbalance_threshold:
            if imbalance > 0:  # More bids than asks
                signal = SignalType.BUY.value
                confidence = min(abs(imbalance), 1.0)
            else:  # More asks than bids
                signal = SignalType.SELL.value
                confidence = min(abs(imbalance), 1.0)
            
            # Adjust confidence based on spread quality
            spread_bps = (order_book.spread / order_book.mid_price) * 10000
            if spread_bps < self.min_spread_bps:
                confidence *= 0.5  # Reduce confidence for tight spreads
            elif spread_bps > self.max_spread_bps:
                confidence *= 0.3  # Reduce confidence for wide spreads
            
            self.logger.info(
                "Imbalance signal generated",
                symbol=symbol,
                imbalance=imbalance,
                signal=signal,
                confidence=confidence,
                spread_bps=spread_bps,
            )
            
            return {
                "signal": signal,
                "confidence": confidence,
                "imbalance": imbalance,
                "spread_bps": spread_bps,
                "hold_time": self.hold_time_seconds,
                "strategy_type": MicrostructureSignal.IMBALANCE_LONG.value if signal == SignalType.BUY.value else MicrostructureSignal.IMBALANCE_SHORT.value,
            }
        
        return {"signal": SignalType.HOLD.value, "confidence": 0.0}
    
    async def _get_order_book_snapshot(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book snapshot."""
        
        try:
            # In production, this would connect to real market data
            # For now, simulate order book data
            
            # Get current price from recent data
            current_price = 100.0  # Mock price
            
            # Generate realistic order book
            spread_bps = np.random.uniform(2, 10)
            spread = current_price * spread_bps / 10000
            
            # Create bid/ask levels
            bids = []
            asks = []
            
            # Generate 5 levels each side
            for i in range(5):
                bid_price = current_price - spread/2 - i * spread * 0.5
                ask_price = current_price + spread/2 + i * spread * 0.5
                
                # Size decreases with distance from mid
                base_size = np.random.randint(500, 2000)
                bid_size = int(base_size * (1 - i * 0.2))
                ask_size = int(base_size * (1 - i * 0.2))
                
                bids.append(OrderBookLevel(bid_price, bid_size))
                asks.append(OrderBookLevel(ask_price, ask_size))
            
            return OrderBookSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                bids=bids,
                asks=asks,
                last_trade_price=current_price,
                last_trade_size=100,
            )
            
        except Exception as e:
            self.logger.error("Failed to get order book", symbol=symbol, error=str(e))
            return None
    
    def _check_basic_conditions(self, order_book: OrderBookSnapshot) -> bool:
        """Check basic conditions for trading."""
        
        # Must have valid bid/ask
        if not order_book.best_bid or not order_book.best_ask:
            return False
        
        # Spread must be reasonable
        spread_bps = (order_book.spread / order_book.mid_price) * 10000
        if spread_bps < self.min_spread_bps or spread_bps > self.max_spread_bps:
            return False
        
        # Must have sufficient depth
        total_depth = order_book.best_bid.size + order_book.best_ask.size
        if total_depth < self.min_book_depth:
            return False
        
        return True


class SpreadCaptureStrategy(Strategy):
    """Strategy that captures bid-ask spreads through market making."""
    
    def __init__(
        self,
        name: str = "SpreadCapture",
        min_spread_bps: float = 3.0,
        max_position_size: int = 1000,
        inventory_limit: int = 5000,
        skew_factor: float = 0.1,
    ):
        super().__init__(name)
        self.min_spread_bps = min_spread_bps
        self.max_position_size = max_position_size
        self.inventory_limit = inventory_limit
        self.skew_factor = skew_factor
        
        # Inventory tracking
        self.current_inventory = 0
        self.target_inventory = 0
        
        self.logger = logger.bind(strategy=name)
    
    async def analyze(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze for spread capture opportunities."""
        
        # Get market microstructure data
        microstructure = await self._get_microstructure_data(symbol)
        
        if not microstructure:
            return {"signal": SignalType.HOLD.value, "confidence": 0.0}
        
        # Check if spread is wide enough
        if microstructure.spread_bps < self.min_spread_bps:
            return {"signal": SignalType.HOLD.value, "confidence": 0.0}
        
        # Check inventory limits
        if abs(self.current_inventory) > self.inventory_limit:
            # Need to reduce inventory
            if self.current_inventory > 0:
                return {
                    "signal": SignalType.SELL.value,
                    "confidence": 0.8,
                    "strategy_type": MicrostructureSignal.SPREAD_CAPTURE.value,
                    "reason": "inventory_reduction",
                }
            else:
                return {
                    "signal": SignalType.BUY.value,
                    "confidence": 0.8,
                    "strategy_type": MicrostructureSignal.SPREAD_CAPTURE.value,
                    "reason": "inventory_reduction",
                }
        
        # Calculate optimal quotes with inventory skew
        inventory_skew = self.current_inventory * self.skew_factor / self.inventory_limit
        
        # Adjust quotes based on inventory
        bid_adjustment = -inventory_skew * microstructure.spread / 2
        ask_adjustment = inventory_skew * microstructure.spread / 2
        
        optimal_bid = microstructure.bid + bid_adjustment
        optimal_ask = microstructure.ask + ask_adjustment
        
        # Determine if we should provide liquidity
        confidence = min(microstructure.spread_bps / 10.0, 1.0)  # Higher confidence for wider spreads
        
        self.logger.info(
            "Spread capture analysis",
            symbol=symbol,
            spread_bps=microstructure.spread_bps,
            inventory=self.current_inventory,
            optimal_bid=optimal_bid,
            optimal_ask=optimal_ask,
            confidence=confidence,
        )
        
        return {
            "signal": MicrostructureSignal.LIQUIDITY_PROVIDE.value,
            "confidence": confidence,
            "optimal_bid": optimal_bid,
            "optimal_ask": optimal_ask,
            "position_size": min(self.max_position_size, self.inventory_limit - abs(self.current_inventory)),
            "strategy_type": MicrostructureSignal.SPREAD_CAPTURE.value,
        }
    
    async def _get_microstructure_data(self, symbol: str) -> Optional[MarketMicrostructure]:
        """Get market microstructure data."""
        
        try:
            # Mock microstructure data
            bid = 99.95
            ask = 100.05
            spread = ask - bid
            mid_price = (bid + ask) / 2
            spread_bps = (spread / mid_price) * 10000
            
            return MarketMicrostructure(
                bid=bid,
                ask=ask,
                bid_size=1500,
                ask_size=1200,
                spread=spread,
                spread_bps=spread_bps,
                mid_price=mid_price,
                imbalance=0.1,  # Slight bid imbalance
                depth={"bids": [(99.95, 1500), (99.94, 1000)], "asks": [(100.05, 1200), (100.06, 800)]},
            )
            
        except Exception as e:
            self.logger.error("Failed to get microstructure data", symbol=symbol, error=str(e))
            return None
    
    def update_inventory(self, trade_quantity: int, trade_side: str) -> None:
        """Update inventory after a trade."""
        
        if trade_side.upper() == "BUY":
            self.current_inventory += trade_quantity
        else:
            self.current_inventory -= trade_quantity
        
        self.logger.info(
            "Inventory updated",
            trade_quantity=trade_quantity,
            trade_side=trade_side,
            new_inventory=self.current_inventory,
        )


class MicroMomentumStrategy(Strategy):
    """Sub-second momentum strategy based on tick data."""
    
    def __init__(
        self,
        name: str = "MicroMomentum",
        lookback_ticks: int = 10,
        momentum_threshold: float = 0.0002,  # 2 bps
        max_hold_seconds: int = 30,
        volume_threshold: int = 1000,
    ):
        super().__init__(name)
        self.lookback_ticks = lookback_ticks
        self.momentum_threshold = momentum_threshold
        self.max_hold_seconds = max_hold_seconds
        self.volume_threshold = volume_threshold
        
        # Tick data storage
        self.tick_data: List[Dict[str, Any]] = []
        
        self.logger = logger.bind(strategy=name)
    
    async def analyze(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze micro momentum from tick data."""
        
        # Get latest tick data
        tick = await self._get_latest_tick(symbol)
        
        if not tick:
            return {"signal": SignalType.HOLD.value, "confidence": 0.0}
        
        # Store tick data
        self.tick_data.append(tick)
        
        # Keep only recent ticks
        if len(self.tick_data) > self.lookback_ticks * 2:
            self.tick_data = self.tick_data[-self.lookback_ticks * 2:]
        
        # Need sufficient history
        if len(self.tick_data) < self.lookback_ticks:
            return {"signal": SignalType.HOLD.value, "confidence": 0.0}
        
        # Calculate micro momentum
        recent_ticks = self.tick_data[-self.lookback_ticks:]
        
        # Price momentum
        price_changes = []
        volume_weighted_changes = []
        
        for i in range(1, len(recent_ticks)):
            price_change = (recent_ticks[i]["price"] - recent_ticks[i-1]["price"]) / recent_ticks[i-1]["price"]
            volume = recent_ticks[i]["volume"]
            
            price_changes.append(price_change)
            volume_weighted_changes.append(price_change * volume)
        
        if not price_changes:
            return {"signal": SignalType.HOLD.value, "confidence": 0.0}
        
        # Calculate momentum metrics
        momentum = np.mean(price_changes)
        volume_weighted_momentum = np.sum(volume_weighted_changes) / np.sum([t["volume"] for t in recent_ticks])
        momentum_consistency = np.mean([1 if pc * momentum > 0 else 0 for pc in price_changes])
        
        # Check volume threshold
        recent_volume = np.sum([t["volume"] for t in recent_ticks[-3:]])  # Last 3 ticks
        if recent_volume < self.volume_threshold:
            return {"signal": SignalType.HOLD.value, "confidence": 0.0}
        
        # Generate signal
        if abs(momentum) > self.momentum_threshold and momentum_consistency > 0.6:
            signal = SignalType.BUY.value if momentum > 0 else SignalType.SELL.value
            confidence = min(abs(momentum) / self.momentum_threshold * momentum_consistency, 1.0)
            
            self.logger.info(
                "Micro momentum signal",
                symbol=symbol,
                momentum=momentum,
                volume_weighted_momentum=volume_weighted_momentum,
                consistency=momentum_consistency,
                signal=signal,
                confidence=confidence,
            )
            
            return {
                "signal": signal,
                "confidence": confidence,
                "momentum": momentum,
                "volume_weighted_momentum": volume_weighted_momentum,
                "consistency": momentum_consistency,
                "hold_time": self.max_hold_seconds,
                "strategy_type": MicrostructureSignal.MOMENTUM_MICRO.value,
            }
        
        return {"signal": SignalType.HOLD.value, "confidence": 0.0}
    
    async def _get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest tick data."""
        
        try:
            # Mock tick data
            base_price = 100.0
            if self.tick_data:
                base_price = self.tick_data[-1]["price"]
            
            # Generate realistic tick
            price_change = np.random.normal(0, 0.0001)  # Small random walk
            new_price = base_price * (1 + price_change)
            volume = np.random.randint(100, 1000)
            
            return {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "price": new_price,
                "volume": volume,
                "side": "BUY" if price_change > 0 else "SELL",
            }
            
        except Exception as e:
            self.logger.error("Failed to get tick data", symbol=symbol, error=str(e))
            return None


def create_microstructure_strategies() -> List[Strategy]:
    """Create a suite of microstructure strategies."""
    
    strategies = [
        OrderBookImbalanceStrategy(
            name="OrderBookImbalance_Aggressive",
            imbalance_threshold=0.2,
            min_spread_bps=2.0,
            max_spread_bps=15.0,
            hold_time_seconds=3,
        ),
        OrderBookImbalanceStrategy(
            name="OrderBookImbalance_Conservative", 
            imbalance_threshold=0.4,
            min_spread_bps=3.0,
            max_spread_bps=10.0,
            hold_time_seconds=10,
        ),
        SpreadCaptureStrategy(
            name="SpreadCapture_Small",
            min_spread_bps=3.0,
            max_position_size=500,
            inventory_limit=2500,
        ),
        SpreadCaptureStrategy(
            name="SpreadCapture_Large",
            min_spread_bps=5.0,
            max_position_size=2000,
            inventory_limit=10000,
        ),
        MicroMomentumStrategy(
            name="MicroMomentum_Fast",
            lookback_ticks=5,
            momentum_threshold=0.0001,
            max_hold_seconds=15,
        ),
        MicroMomentumStrategy(
            name="MicroMomentum_Slow",
            lookback_ticks=20,
            momentum_threshold=0.0003,
            max_hold_seconds=60,
        ),
    ]
    
    return strategies
