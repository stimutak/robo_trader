"""
Microstructure Trading Strategies

High-frequency trading strategies based on order book dynamics and microstructure analysis.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from robo_trader.features.orderbook import OrderBookFeatures, OrderBookSnapshot


class BaseStrategy:
    """Simple base class for microstructure strategies"""
    
    def __init__(self):
        pass
    
    def should_buy(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """Determine if should buy"""
        raise NotImplementedError
    
    def should_sell(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """Determine if should sell"""
        raise NotImplementedError
    
    def calculate_position_size(self, symbol: str, current_price: float, capital: float, market_data: Dict) -> int:
        """Calculate position size"""
        return 100

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure strategies"""
    # Order flow imbalance thresholds
    ofi_entry_threshold: float = 0.3
    ofi_exit_threshold: float = 0.1
    
    # Spread trading parameters
    min_spread_bps: float = 5.0  # Minimum spread in basis points
    max_spread_bps: float = 50.0  # Maximum spread in basis points
    
    # Market making parameters
    quote_offset_bps: float = 2.0  # Offset from best bid/ask in basis points
    max_inventory: int = 100  # Maximum inventory per symbol
    inventory_skew_factor: float = 0.5  # Adjust quotes based on inventory
    
    # Tick momentum parameters
    tick_window: int = 20  # Window for tick direction calculation
    tick_threshold: float = 0.6  # Threshold for tick momentum signal
    
    # Risk parameters
    max_position_size: int = 1000
    stop_loss_bps: float = 20.0  # Stop loss in basis points
    take_profit_bps: float = 10.0  # Take profit in basis points
    
    # Execution parameters
    min_order_size: int = 100
    max_order_size: int = 500
    execution_delay_ms: int = 10  # Simulated execution delay
    
    # Feature calculation
    orderbook_levels: int = 10  # Number of order book levels to analyze
    feature_window: int = 50  # Number of snapshots for feature calculation


class OrderFlowImbalanceStrategy(BaseStrategy):
    """
    Strategy based on order flow imbalance (OFI)
    
    Trades in the direction of order book pressure when imbalance is significant.
    """
    
    def __init__(self, config: Optional[MicrostructureConfig] = None):
        super().__init__()
        self.config = config or MicrostructureConfig()
        self.orderbook_features = OrderBookFeatures(max_levels=self.config.orderbook_levels)
        self.positions: Dict[str, int] = {}
        self.entry_prices: Dict[str, float] = {}
        
    def should_buy(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """
        Buy when order flow imbalance indicates buying pressure
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            market_data: Market data including order book
            
        Returns:
            Tuple of (should_buy, metadata)
        """
        # Update order book features if available
        if 'orderbook' in market_data:
            snapshot = self._create_orderbook_snapshot(market_data['orderbook'])
            self.orderbook_features.add_snapshot(snapshot)
        
        # Calculate order flow imbalance
        ofi = self.orderbook_features.calculate_order_flow_imbalance()
        book_pressure = self.orderbook_features.calculate_book_pressure()
        
        # Check if we should buy
        should_buy = (
            ofi > self.config.ofi_entry_threshold and
            book_pressure > 0.2 and
            self.positions.get(symbol, 0) <= 0  # Not already long
        )
        
        metadata = {
            'strategy': 'order_flow_imbalance',
            'ofi': ofi,
            'book_pressure': book_pressure,
            'signal_strength': ofi * book_pressure
        }
        
        if should_buy:
            self.positions[symbol] = self.config.min_order_size
            self.entry_prices[symbol] = current_price
            logger.info(f"OFI Buy signal for {symbol}: OFI={ofi:.3f}, Pressure={book_pressure:.3f}")
        
        return should_buy, metadata
    
    def should_sell(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """
        Sell when order flow imbalance reverses or hits stop/target
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            market_data: Market data including order book
            
        Returns:
            Tuple of (should_sell, metadata)
        """
        # Update order book features if available
        if 'orderbook' in market_data:
            snapshot = self._create_orderbook_snapshot(market_data['orderbook'])
            self.orderbook_features.add_snapshot(snapshot)
        
        # Calculate order flow imbalance
        ofi = self.orderbook_features.calculate_order_flow_imbalance()
        book_pressure = self.orderbook_features.calculate_book_pressure()
        
        # Check position
        position = self.positions.get(symbol, 0)
        entry_price = self.entry_prices.get(symbol, current_price)
        
        # Calculate P&L
        pnl_bps = ((current_price - entry_price) / entry_price) * 10000 if entry_price > 0 else 0
        
        # Exit conditions
        should_sell = False
        exit_reason = None
        
        if position > 0:  # Long position
            if ofi < -self.config.ofi_exit_threshold:
                should_sell = True
                exit_reason = 'ofi_reversal'
            elif pnl_bps >= self.config.take_profit_bps:
                should_sell = True
                exit_reason = 'take_profit'
            elif pnl_bps <= -self.config.stop_loss_bps:
                should_sell = True
                exit_reason = 'stop_loss'
        elif position < 0:  # Short position (cover)
            if ofi > self.config.ofi_exit_threshold:
                should_sell = True
                exit_reason = 'ofi_reversal'
        
        metadata = {
            'strategy': 'order_flow_imbalance',
            'ofi': ofi,
            'book_pressure': book_pressure,
            'pnl_bps': pnl_bps,
            'exit_reason': exit_reason
        }
        
        if should_sell and position != 0:
            logger.info(f"OFI Sell signal for {symbol}: Reason={exit_reason}, PnL={pnl_bps:.1f}bps")
            self.positions[symbol] = 0
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]
        
        return should_sell, metadata
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              capital: float, market_data: Dict) -> int:
        """Calculate position size based on order flow strength"""
        ofi = self.orderbook_features.calculate_order_flow_imbalance()
        
        # Scale position size with OFI strength
        base_size = self.config.min_order_size
        size_multiplier = min(abs(ofi) / self.config.ofi_entry_threshold, 3.0)
        
        position_size = int(base_size * size_multiplier)
        position_size = min(position_size, self.config.max_order_size)
        
        return position_size
    
    def _create_orderbook_snapshot(self, orderbook_data: Dict) -> OrderBookSnapshot:
        """Create OrderBookSnapshot from market data"""
        return OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bid_prices=np.array(orderbook_data.get('bid_prices', [])),
            bid_sizes=np.array(orderbook_data.get('bid_sizes', [])),
            ask_prices=np.array(orderbook_data.get('ask_prices', [])),
            ask_sizes=np.array(orderbook_data.get('ask_sizes', []))
        )


class SpreadTradingStrategy(BaseStrategy):
    """
    Market making strategy that provides liquidity by quoting bid and ask prices
    
    Adjusts quotes based on inventory risk and market conditions.
    """
    
    def __init__(self, config: Optional[MicrostructureConfig] = None):
        super().__init__()
        self.config = config or MicrostructureConfig()
        self.orderbook_features = OrderBookFeatures(max_levels=self.config.orderbook_levels)
        self.inventory: Dict[str, int] = {}
        self.pending_orders: Dict[str, List[Dict]] = {}
        
    def calculate_quotes(self, symbol: str, market_data: Dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate bid and ask quotes based on market conditions and inventory
        
        Args:
            symbol: Stock symbol
            market_data: Market data including order book
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        if 'orderbook' not in market_data:
            return None, None
            
        snapshot = self._create_orderbook_snapshot(market_data['orderbook'])
        self.orderbook_features.add_snapshot(snapshot)
        
        # Get spread metrics
        spread_metrics = self.orderbook_features.calculate_spread_metrics()
        spread_bps = spread_metrics['spread_pct'] * 100
        
        # Check if spread is within tradeable range
        if spread_bps < self.config.min_spread_bps or spread_bps > self.config.max_spread_bps:
            return None, None
        
        # Calculate base quotes
        micro_price = self.orderbook_features.calculate_micro_price()
        if micro_price <= 0:
            return None, None
            
        offset = micro_price * (self.config.quote_offset_bps / 10000)
        
        # Adjust for inventory risk
        current_inventory = self.inventory.get(symbol, 0)
        inventory_ratio = current_inventory / self.config.max_inventory if self.config.max_inventory > 0 else 0
        
        # Skew quotes based on inventory
        inventory_adjustment = inventory_ratio * self.config.inventory_skew_factor * offset
        
        bid_price = micro_price - offset - inventory_adjustment
        ask_price = micro_price + offset - inventory_adjustment
        
        return bid_price, ask_price
    
    def should_buy(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """
        Place buy order at calculated bid price
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            market_data: Market data
            
        Returns:
            Tuple of (should_buy, metadata)
        """
        bid_price, _ = self.calculate_quotes(symbol, market_data)
        
        if bid_price is None:
            return False, {}
        
        # Check inventory limits
        current_inventory = self.inventory.get(symbol, 0)
        if current_inventory >= self.config.max_inventory:
            return False, {'reason': 'max_inventory'}
        
        # Check if our bid would be competitive
        if 'orderbook' in market_data:
            best_bid = market_data['orderbook'].get('best_bid', 0)
            if bid_price < best_bid * 0.995:  # Too far from best bid
                return False, {'reason': 'uncompetitive_bid'}
        
        metadata = {
            'strategy': 'spread_trading',
            'quote_type': 'bid',
            'bid_price': bid_price,
            'inventory': current_inventory,
            'spread_bps': self.orderbook_features.calculate_spread_metrics()['spread_pct'] * 100
        }
        
        # Update inventory (simulation)
        self.inventory[symbol] = current_inventory + self.config.min_order_size
        
        return True, metadata
    
    def should_sell(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """
        Place sell order at calculated ask price
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            market_data: Market data
            
        Returns:
            Tuple of (should_sell, metadata)
        """
        _, ask_price = self.calculate_quotes(symbol, market_data)
        
        if ask_price is None:
            return False, {}
        
        # Check inventory
        current_inventory = self.inventory.get(symbol, 0)
        if current_inventory <= -self.config.max_inventory:
            return False, {'reason': 'max_short_inventory'}
        
        # Check if our ask would be competitive
        if 'orderbook' in market_data:
            best_ask = market_data['orderbook'].get('best_ask', float('inf'))
            if ask_price > best_ask * 1.005:  # Too far from best ask
                return False, {'reason': 'uncompetitive_ask'}
        
        metadata = {
            'strategy': 'spread_trading',
            'quote_type': 'ask',
            'ask_price': ask_price,
            'inventory': current_inventory,
            'spread_bps': self.orderbook_features.calculate_spread_metrics()['spread_pct'] * 100
        }
        
        # Update inventory (simulation)
        self.inventory[symbol] = current_inventory - self.config.min_order_size
        
        return True, metadata
    
    def calculate_position_size(self, symbol: str, current_price: float,
                              capital: float, market_data: Dict) -> int:
        """Calculate order size based on spread and liquidity"""
        liquidity_metrics = self.orderbook_features.calculate_liquidity_metrics()
        
        # Scale size with market depth
        depth_ratio = min(liquidity_metrics['total_depth'] / 1000, 1.0)
        size = int(self.config.min_order_size * (1 + depth_ratio))
        
        return min(size, self.config.max_order_size)
    
    def _create_orderbook_snapshot(self, orderbook_data: Dict) -> OrderBookSnapshot:
        """Create OrderBookSnapshot from market data"""
        return OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bid_prices=np.array(orderbook_data.get('bid_prices', [])),
            bid_sizes=np.array(orderbook_data.get('bid_sizes', [])),
            ask_prices=np.array(orderbook_data.get('ask_prices', [])),
            ask_sizes=np.array(orderbook_data.get('ask_sizes', []))
        )


class TickMomentumStrategy(BaseStrategy):
    """
    High-frequency momentum strategy based on tick-level price movements
    
    Detects and trades short-term momentum in tick data.
    """
    
    def __init__(self, config: Optional[MicrostructureConfig] = None):
        super().__init__()
        self.config = config or MicrostructureConfig()
        self.orderbook_features = OrderBookFeatures(max_levels=self.config.orderbook_levels)
        self.tick_history: Dict[str, List[float]] = {}
        self.positions: Dict[str, int] = {}
        self.entry_times: Dict[str, datetime] = {}
        
    def should_buy(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """
        Buy when detecting upward tick momentum
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            market_data: Market data
            
        Returns:
            Tuple of (should_buy, metadata)
        """
        # Update tick history
        if symbol not in self.tick_history:
            self.tick_history[symbol] = []
        self.tick_history[symbol].append(current_price)
        
        # Keep only recent ticks
        if len(self.tick_history[symbol]) > self.config.tick_window:
            self.tick_history[symbol].pop(0)
        
        # Need enough history
        if len(self.tick_history[symbol]) < self.config.tick_window:
            return False, {'reason': 'insufficient_history'}
        
        # Update order book features if available
        if 'orderbook' in market_data:
            snapshot = self._create_orderbook_snapshot(market_data['orderbook'])
            self.orderbook_features.add_snapshot(snapshot)
        
        # Calculate tick momentum
        tick_direction = self.orderbook_features.calculate_tick_direction(self.config.tick_window)
        
        # Calculate price momentum
        ticks = self.tick_history[symbol]
        returns = [(ticks[i] - ticks[i-1]) / ticks[i-1] for i in range(1, len(ticks))]
        momentum = np.mean(returns) * 10000  # Convert to basis points
        
        # Check for strong upward momentum
        should_buy = (
            tick_direction > self.config.tick_threshold and
            momentum > 1.0 and  # Positive momentum in bps
            symbol not in self.positions  # Not already in position
        )
        
        metadata = {
            'strategy': 'tick_momentum',
            'tick_direction': tick_direction,
            'momentum_bps': momentum,
            'signal_strength': tick_direction * abs(momentum)
        }
        
        if should_buy:
            self.positions[symbol] = self.config.min_order_size
            self.entry_times[symbol] = datetime.now()
            logger.info(f"Tick momentum buy for {symbol}: Direction={tick_direction:.3f}, Momentum={momentum:.1f}bps")
        
        return should_buy, metadata
    
    def should_sell(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """
        Sell when momentum reverses or after holding period
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            market_data: Market data
            
        Returns:
            Tuple of (should_sell, metadata)
        """
        if symbol not in self.positions:
            return False, {}
        
        # Update order book features if available
        if 'orderbook' in market_data:
            snapshot = self._create_orderbook_snapshot(market_data['orderbook'])
            self.orderbook_features.add_snapshot(snapshot)
        
        # Calculate current momentum
        tick_direction = self.orderbook_features.calculate_tick_direction(self.config.tick_window)
        
        if symbol in self.tick_history and len(self.tick_history[symbol]) >= 2:
            ticks = self.tick_history[symbol]
            recent_return = (ticks[-1] - ticks[-2]) / ticks[-2] * 10000
        else:
            recent_return = 0
        
        # Check holding period (high-frequency exit)
        holding_time = (datetime.now() - self.entry_times.get(symbol, datetime.now())).total_seconds()
        
        # Exit conditions
        should_sell = (
            tick_direction < -self.config.tick_threshold * 0.5 or  # Momentum reversal
            recent_return < -self.config.stop_loss_bps or  # Stop loss
            recent_return > self.config.take_profit_bps or  # Take profit
            holding_time > 60  # Max holding time of 60 seconds for HFT
        )
        
        exit_reason = None
        if tick_direction < -self.config.tick_threshold * 0.5:
            exit_reason = 'momentum_reversal'
        elif recent_return < -self.config.stop_loss_bps:
            exit_reason = 'stop_loss'
        elif recent_return > self.config.take_profit_bps:
            exit_reason = 'take_profit'
        elif holding_time > 60:
            exit_reason = 'max_holding_time'
        
        metadata = {
            'strategy': 'tick_momentum',
            'tick_direction': tick_direction,
            'recent_return_bps': recent_return,
            'holding_time_seconds': holding_time,
            'exit_reason': exit_reason
        }
        
        if should_sell:
            logger.info(f"Tick momentum sell for {symbol}: Reason={exit_reason}, Return={recent_return:.1f}bps")
            del self.positions[symbol]
            if symbol in self.entry_times:
                del self.entry_times[symbol]
        
        return should_sell, metadata
    
    def calculate_position_size(self, symbol: str, current_price: float,
                              capital: float, market_data: Dict) -> int:
        """Calculate position size based on momentum strength"""
        tick_direction = self.orderbook_features.calculate_tick_direction(self.config.tick_window)
        
        # Scale with momentum strength
        strength_ratio = min(abs(tick_direction) / self.config.tick_threshold, 2.0)
        size = int(self.config.min_order_size * strength_ratio)
        
        return min(size, self.config.max_order_size)
    
    def _create_orderbook_snapshot(self, orderbook_data: Dict) -> OrderBookSnapshot:
        """Create OrderBookSnapshot from market data"""
        return OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bid_prices=np.array(orderbook_data.get('bid_prices', [])),
            bid_sizes=np.array(orderbook_data.get('bid_sizes', [])),
            ask_prices=np.array(orderbook_data.get('ask_prices', [])),
            ask_sizes=np.array(orderbook_data.get('ask_sizes', []))
        )


class MicrostructureEnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy combining multiple microstructure signals
    
    Combines order flow imbalance, spread trading, and tick momentum strategies.
    """
    
    def __init__(self, config: Optional[MicrostructureConfig] = None):
        super().__init__()
        self.config = config or MicrostructureConfig()
        
        # Initialize sub-strategies
        self.ofi_strategy = OrderFlowImbalanceStrategy(config)
        self.spread_strategy = SpreadTradingStrategy(config)
        self.tick_strategy = TickMomentumStrategy(config)
        
        # Ensemble weights
        self.weights = {
            'ofi': 0.4,
            'spread': 0.3,
            'tick': 0.3
        }
        
    def should_buy(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """
        Combine signals from all strategies
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            market_data: Market data
            
        Returns:
            Tuple of (should_buy, metadata)
        """
        # Get signals from each strategy
        ofi_signal, ofi_meta = self.ofi_strategy.should_buy(symbol, current_price, market_data)
        spread_signal, spread_meta = self.spread_strategy.should_buy(symbol, current_price, market_data)
        tick_signal, tick_meta = self.tick_strategy.should_buy(symbol, current_price, market_data)
        
        # Calculate weighted ensemble score
        ensemble_score = (
            self.weights['ofi'] * float(ofi_signal) +
            self.weights['spread'] * float(spread_signal) +
            self.weights['tick'] * float(tick_signal)
        )
        
        # Buy if ensemble score exceeds threshold
        should_buy = ensemble_score >= 0.5
        
        metadata = {
            'strategy': 'microstructure_ensemble',
            'ensemble_score': ensemble_score,
            'ofi_signal': ofi_signal,
            'spread_signal': spread_signal,
            'tick_signal': tick_signal,
            'sub_strategies': {
                'ofi': ofi_meta,
                'spread': spread_meta,
                'tick': tick_meta
            }
        }
        
        return should_buy, metadata
    
    def should_sell(self, symbol: str, current_price: float, market_data: Dict) -> Tuple[bool, Dict]:
        """
        Combine sell signals from all strategies
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            market_data: Market data
            
        Returns:
            Tuple of (should_sell, metadata)
        """
        # Get signals from each strategy
        ofi_signal, ofi_meta = self.ofi_strategy.should_sell(symbol, current_price, market_data)
        spread_signal, spread_meta = self.spread_strategy.should_sell(symbol, current_price, market_data)
        tick_signal, tick_meta = self.tick_strategy.should_sell(symbol, current_price, market_data)
        
        # Calculate weighted ensemble score
        ensemble_score = (
            self.weights['ofi'] * float(ofi_signal) +
            self.weights['spread'] * float(spread_signal) +
            self.weights['tick'] * float(tick_signal)
        )
        
        # Sell if ensemble score exceeds threshold
        should_sell = ensemble_score >= 0.5
        
        metadata = {
            'strategy': 'microstructure_ensemble',
            'ensemble_score': ensemble_score,
            'ofi_signal': ofi_signal,
            'spread_signal': spread_signal,
            'tick_signal': tick_signal,
            'sub_strategies': {
                'ofi': ofi_meta,
                'spread': spread_meta,
                'tick': tick_meta
            }
        }
        
        return should_sell, metadata
    
    def calculate_position_size(self, symbol: str, current_price: float,
                              capital: float, market_data: Dict) -> int:
        """Calculate position size as weighted average of sub-strategies"""
        ofi_size = self.ofi_strategy.calculate_position_size(symbol, current_price, capital, market_data)
        spread_size = self.spread_strategy.calculate_position_size(symbol, current_price, capital, market_data)
        tick_size = self.tick_strategy.calculate_position_size(symbol, current_price, capital, market_data)
        
        weighted_size = int(
            self.weights['ofi'] * ofi_size +
            self.weights['spread'] * spread_size +
            self.weights['tick'] * tick_size
        )
        
        return min(weighted_size, self.config.max_order_size)