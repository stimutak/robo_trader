"""
Order Book Features for Microstructure Analysis

This module provides features derived from order book dynamics for high-frequency trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Represents a snapshot of the order book at a point in time"""
    timestamp: pd.Timestamp
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray
    
    @property
    def best_bid(self) -> float:
        """Get best bid price"""
        return self.bid_prices[0] if len(self.bid_prices) > 0 else 0.0
    
    @property
    def best_ask(self) -> float:
        """Get best ask price"""
        return self.ask_prices[0] if len(self.ask_prices) > 0 else float('inf')
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.best_ask - self.best_bid
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.best_bid + self.best_ask) / 2 if self.best_bid > 0 else 0.0


class OrderBookFeatures:
    """Extracts microstructure features from order book data"""
    
    def __init__(self, max_levels: int = 10):
        """
        Initialize order book feature extractor
        
        Args:
            max_levels: Maximum number of order book levels to analyze
        """
        self.max_levels = max_levels
        self.snapshots: List[OrderBookSnapshot] = []
        
    def add_snapshot(self, snapshot: OrderBookSnapshot):
        """Add a new order book snapshot"""
        self.snapshots.append(snapshot)
        # Keep only recent snapshots (last 1000)
        if len(self.snapshots) > 1000:
            self.snapshots.pop(0)
    
    def calculate_order_flow_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order flow imbalance indicator
        
        OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Args:
            levels: Number of price levels to include
            
        Returns:
            Order flow imbalance ratio between -1 and 1
        """
        if not self.snapshots:
            return 0.0
            
        snapshot = self.snapshots[-1]
        
        # Sum volumes across specified levels
        bid_volume = np.sum(snapshot.bid_sizes[:min(levels, len(snapshot.bid_sizes))])
        ask_volume = np.sum(snapshot.ask_sizes[:min(levels, len(snapshot.ask_sizes))])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / total_volume
    
    def calculate_volume_weighted_average_price(self, side: str = 'both', levels: int = 5) -> float:
        """
        Calculate volume-weighted average price (VWAP) from order book
        
        Args:
            side: 'bid', 'ask', or 'both'
            levels: Number of price levels to include
            
        Returns:
            Volume-weighted average price
        """
        if not self.snapshots:
            return 0.0
            
        snapshot = self.snapshots[-1]
        
        if side == 'bid' or side == 'both':
            bid_prices = snapshot.bid_prices[:min(levels, len(snapshot.bid_prices))]
            bid_sizes = snapshot.bid_sizes[:min(levels, len(snapshot.bid_sizes))]
            bid_vwap = np.sum(bid_prices * bid_sizes) / np.sum(bid_sizes) if np.sum(bid_sizes) > 0 else 0
        else:
            bid_vwap = 0
            
        if side == 'ask' or side == 'both':
            ask_prices = snapshot.ask_prices[:min(levels, len(snapshot.ask_prices))]
            ask_sizes = snapshot.ask_sizes[:min(levels, len(snapshot.ask_sizes))]
            ask_vwap = np.sum(ask_prices * ask_sizes) / np.sum(ask_sizes) if np.sum(ask_sizes) > 0 else 0
        else:
            ask_vwap = 0
            
        if side == 'both':
            return (bid_vwap + ask_vwap) / 2 if bid_vwap > 0 and ask_vwap > 0 else 0
        elif side == 'bid':
            return bid_vwap
        else:
            return ask_vwap
    
    def calculate_book_pressure(self) -> float:
        """
        Calculate book pressure indicator
        
        Measures the ratio of bid pressure to ask pressure
        weighted by distance from mid price
        
        Returns:
            Book pressure ratio (positive = buy pressure, negative = sell pressure)
        """
        if not self.snapshots:
            return 0.0
            
        snapshot = self.snapshots[-1]
        mid_price = snapshot.mid_price
        
        if mid_price == 0:
            return 0.0
        
        # Calculate weighted bid pressure
        bid_pressure = 0.0
        for i in range(min(self.max_levels, len(snapshot.bid_prices))):
            distance = abs(snapshot.bid_prices[i] - mid_price) / mid_price
            weight = np.exp(-distance * 100)  # Exponential decay with distance
            bid_pressure += snapshot.bid_sizes[i] * weight
        
        # Calculate weighted ask pressure
        ask_pressure = 0.0
        for i in range(min(self.max_levels, len(snapshot.ask_prices))):
            distance = abs(snapshot.ask_prices[i] - mid_price) / mid_price
            weight = np.exp(-distance * 100)
            ask_pressure += snapshot.ask_sizes[i] * weight
        
        total_pressure = bid_pressure + ask_pressure
        if total_pressure == 0:
            return 0.0
            
        return (bid_pressure - ask_pressure) / total_pressure
    
    def calculate_micro_price(self) -> float:
        """
        Calculate micro price (weighted mid price based on order book imbalance)
        
        micro_price = bid * (ask_size / (bid_size + ask_size)) + ask * (bid_size / (bid_size + ask_size))
        
        Returns:
            Micro price estimate
        """
        if not self.snapshots:
            return 0.0
            
        snapshot = self.snapshots[-1]
        
        if snapshot.best_bid == 0 or snapshot.best_ask == float('inf'):
            return 0.0
            
        bid_size = snapshot.bid_sizes[0] if len(snapshot.bid_sizes) > 0 else 0
        ask_size = snapshot.ask_sizes[0] if len(snapshot.ask_sizes) > 0 else 0
        
        total_size = bid_size + ask_size
        if total_size == 0:
            return snapshot.mid_price
            
        return (snapshot.best_bid * ask_size + snapshot.best_ask * bid_size) / total_size
    
    def calculate_spread_metrics(self) -> Dict[str, float]:
        """
        Calculate various spread-based metrics
        
        Returns:
            Dictionary of spread metrics
        """
        if not self.snapshots:
            return {'spread': 0, 'spread_pct': 0, 'effective_spread': 0}
            
        snapshot = self.snapshots[-1]
        
        spread = snapshot.spread
        spread_pct = (spread / snapshot.mid_price * 100) if snapshot.mid_price > 0 else 0
        
        # Effective spread using micro price
        micro_price = self.calculate_micro_price()
        effective_spread = 2 * abs(micro_price - snapshot.mid_price) if micro_price > 0 else 0
        
        return {
            'spread': spread,
            'spread_pct': spread_pct,
            'effective_spread': effective_spread
        }
    
    def calculate_liquidity_metrics(self, levels: int = 5) -> Dict[str, float]:
        """
        Calculate liquidity metrics from order book
        
        Args:
            levels: Number of levels to analyze
            
        Returns:
            Dictionary of liquidity metrics
        """
        if not self.snapshots:
            return {'bid_depth': 0, 'ask_depth': 0, 'total_depth': 0, 'depth_imbalance': 0}
            
        snapshot = self.snapshots[-1]
        
        bid_depth = np.sum(snapshot.bid_sizes[:min(levels, len(snapshot.bid_sizes))])
        ask_depth = np.sum(snapshot.ask_sizes[:min(levels, len(snapshot.ask_sizes))])
        total_depth = bid_depth + ask_depth
        
        depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': total_depth,
            'depth_imbalance': depth_imbalance
        }
    
    def detect_order_book_anomalies(self) -> Dict[str, bool]:
        """
        Detect anomalies in order book structure
        
        Returns:
            Dictionary of detected anomalies
        """
        if not self.snapshots:
            return {'wide_spread': False, 'thin_book': False, 'one_sided': False}
            
        snapshot = self.snapshots[-1]
        spread_metrics = self.calculate_spread_metrics()
        liquidity_metrics = self.calculate_liquidity_metrics()
        
        # Wide spread detection (> 0.5% of mid price)
        wide_spread = spread_metrics['spread_pct'] > 0.5
        
        # Thin book detection (low liquidity)
        thin_book = liquidity_metrics['total_depth'] < 100  # Adjust threshold as needed
        
        # One-sided book detection (extreme imbalance)
        one_sided = abs(liquidity_metrics['depth_imbalance']) > 0.8
        
        return {
            'wide_spread': wide_spread,
            'thin_book': thin_book,
            'one_sided': one_sided
        }
    
    def calculate_tick_direction(self, window: int = 10) -> float:
        """
        Calculate tick direction indicator based on recent price movements
        
        Args:
            window: Number of snapshots to analyze
            
        Returns:
            Tick direction score between -1 (down) and 1 (up)
        """
        if len(self.snapshots) < 2:
            return 0.0
            
        recent_snapshots = self.snapshots[-min(window, len(self.snapshots)):]
        
        upticks = 0
        downticks = 0
        
        for i in range(1, len(recent_snapshots)):
            prev_mid = recent_snapshots[i-1].mid_price
            curr_mid = recent_snapshots[i].mid_price
            
            if curr_mid > prev_mid:
                upticks += 1
            elif curr_mid < prev_mid:
                downticks += 1
        
        total_ticks = upticks + downticks
        if total_ticks == 0:
            return 0.0
            
        return (upticks - downticks) / total_ticks
    
    def get_all_features(self) -> Dict[str, float]:
        """
        Calculate all microstructure features
        
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Order flow features
        features['order_flow_imbalance'] = self.calculate_order_flow_imbalance()
        features['book_pressure'] = self.calculate_book_pressure()
        
        # Price features
        features['micro_price'] = self.calculate_micro_price()
        features['vwap_bid'] = self.calculate_volume_weighted_average_price('bid')
        features['vwap_ask'] = self.calculate_volume_weighted_average_price('ask')
        
        # Spread features
        spread_metrics = self.calculate_spread_metrics()
        features.update(spread_metrics)
        
        # Liquidity features
        liquidity_metrics = self.calculate_liquidity_metrics()
        features.update(liquidity_metrics)
        
        # Tick direction
        features['tick_direction'] = self.calculate_tick_direction()
        
        # Anomaly detection
        anomalies = self.detect_order_book_anomalies()
        for key, value in anomalies.items():
            features[f'anomaly_{key}'] = float(value)
        
        return features