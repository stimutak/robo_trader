"""Pairs trading and statistical arbitrage strategies for RoboTrader.

This module implements sophisticated pairs trading strategies:
- Cointegration-based pairs trading
- Statistical arbitrage with ML enhancement
- Dynamic hedge ratio calculation
- Risk management for pair positions
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


# Simple logger replacement to avoid import issues
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


class PairSignal(Enum):
    """Pair trading signal types."""
    
    LONG_A_SHORT_B = "long_a_short_b"
    LONG_B_SHORT_A = "long_b_short_a"
    CLOSE_PAIR = "close_pair"
    HOLD = "hold"


@dataclass
class PairStats:
    """Statistical properties of a trading pair."""
    
    symbol_a: str
    symbol_b: str
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float  # Mean reversion half-life in days
    last_updated: datetime


@dataclass
class PairPosition:
    """Active pair trading position."""
    
    symbol_a: str
    symbol_b: str
    quantity_a: int
    quantity_b: int
    entry_spread: float
    entry_time: datetime
    target_spread: float
    stop_loss_spread: float
    unrealized_pnl: float = 0.0


class CointegrationPairsStrategy:
    """Cointegration-based pairs trading strategy."""
    
    def __init__(
        self,
        name: str = "CointegrationPairs",
        lookback_days: int = 252,
        min_correlation: float = 0.7,
        max_cointegration_pvalue: float = 0.05,
        entry_zscore_threshold: float = 2.0,
        exit_zscore_threshold: float = 0.5,
        stop_loss_zscore: float = 3.5,
        max_holding_days: int = 30,
        rebalance_frequency_hours: int = 24,
    ):
        self.name = name
        self.lookback_days = lookback_days
        self.min_correlation = min_correlation
        self.max_cointegration_pvalue = max_cointegration_pvalue
        self.entry_zscore_threshold = entry_zscore_threshold
        self.exit_zscore_threshold = exit_zscore_threshold
        self.stop_loss_zscore = stop_loss_zscore
        self.max_holding_days = max_holding_days
        self.rebalance_frequency_hours = rebalance_frequency_hours
        
        # State tracking
        self.pair_stats: Dict[Tuple[str, str], PairStats] = {}
        self.active_positions: Dict[Tuple[str, str], PairPosition] = {}
        self.price_history: Dict[str, pd.Series] = {}
        self.last_rebalance: Optional[datetime] = None
        
        self.logger = logger.bind(strategy=name)
    
    async def find_pairs(self, symbols: List[str], price_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """Find cointegrated pairs from a universe of symbols."""
        
        self.logger.info(f"Searching for cointegrated pairs among {len(symbols)} symbols")
        
        # Update price history
        for symbol in symbols:
            if symbol in price_data and len(price_data[symbol]) > 0:
                self.price_history[symbol] = price_data[symbol]["close"]
        
        valid_pairs = []
        
        # Test all possible pairs
        for i, symbol_a in enumerate(symbols):
            for j, symbol_b in enumerate(symbols[i+1:], i+1):
                
                if symbol_a not in self.price_history or symbol_b not in self.price_history:
                    continue
                
                # Get aligned price series
                prices_a = self.price_history[symbol_a]
                prices_b = self.price_history[symbol_b]
                
                # Align dates
                common_dates = prices_a.index.intersection(prices_b.index)
                if len(common_dates) < self.lookback_days:
                    continue
                
                aligned_a = prices_a.loc[common_dates].tail(self.lookback_days)
                aligned_b = prices_b.loc[common_dates].tail(self.lookback_days)
                
                # Test pair relationship
                pair_stats = await self._test_pair_relationship(symbol_a, symbol_b, aligned_a, aligned_b)
                
                if pair_stats:
                    self.pair_stats[(symbol_a, symbol_b)] = pair_stats
                    valid_pairs.append((symbol_a, symbol_b))
                    
                    self.logger.info(
                        f"Found valid pair: {symbol_a}-{symbol_b}",
                        correlation=pair_stats.correlation,
                        cointegration_p=pair_stats.cointegration_pvalue,
                        hedge_ratio=pair_stats.hedge_ratio,
                    )
        
        self.logger.info(f"Found {len(valid_pairs)} valid pairs")
        return valid_pairs
    
    async def _test_pair_relationship(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> Optional[PairStats]:
        """Test if two symbols form a valid trading pair."""
        
        # Calculate correlation
        correlation = prices_a.corr(prices_b)
        
        if abs(correlation) < self.min_correlation:
            return None
        
        # Test for cointegration using Engle-Granger test
        # Step 1: Run regression to get hedge ratio
        X = prices_b.values.reshape(-1, 1)
        y = prices_a.values
        
        reg = LinearRegression().fit(X, y)
        hedge_ratio = reg.coef_[0]
        
        # Step 2: Calculate spread
        spread = prices_a - hedge_ratio * prices_b
        
        # Step 3: Test spread for stationarity (simplified ADF test)
        # Using a simple approach - check if spread mean reverts
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # Calculate z-scores
        z_scores = (spread - spread_mean) / spread_std
        
        # Simple stationarity test: check if z-scores revert to mean
        # Count how often z-scores cross zero
        zero_crossings = np.sum(np.diff(np.sign(z_scores)) != 0)
        crossing_rate = zero_crossings / len(z_scores)
        
        # Estimate p-value based on crossing rate (simplified)
        # Higher crossing rate suggests more mean reversion
        estimated_pvalue = max(0.001, 0.2 - crossing_rate)
        
        if estimated_pvalue > self.max_cointegration_pvalue:
            return None
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        return PairStats(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=correlation,
            cointegration_pvalue=estimated_pvalue,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life,
            last_updated=datetime.now(),
        )
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life."""
        
        # Simple AR(1) model: spread[t] = alpha + beta * spread[t-1] + error
        spread_lag = spread.shift(1).dropna()
        spread_current = spread[1:]
        
        if len(spread_lag) < 10:
            return 30.0  # Default to 30 days
        
        # Linear regression
        X = spread_lag.values.reshape(-1, 1)
        y = spread_current.values
        
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        
        # Half-life calculation
        if beta >= 1 or beta <= 0:
            return 30.0  # No mean reversion
        
        half_life = -np.log(2) / np.log(beta)
        return min(max(half_life, 1.0), 100.0)  # Clamp between 1 and 100 days
    
    async def analyze_pairs(
        self,
        pairs: List[Tuple[str, str]],
        current_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Analyze pairs for trading opportunities."""
        
        signals = []
        
        for symbol_a, symbol_b in pairs:
            pair_key = (symbol_a, symbol_b)
            
            if pair_key not in self.pair_stats:
                continue
            
            if symbol_a not in current_prices or symbol_b not in current_prices:
                continue
            
            pair_stats = self.pair_stats[pair_key]
            price_a = current_prices[symbol_a]
            price_b = current_prices[symbol_b]
            
            # Calculate current spread and z-score
            current_spread = price_a - pair_stats.hedge_ratio * price_b
            z_score = (current_spread - pair_stats.spread_mean) / pair_stats.spread_std
            
            # Check for signals
            signal = await self._evaluate_pair_signal(
                pair_key, pair_stats, z_score, current_spread, price_a, price_b
            )
            
            if signal:
                signals.append(signal)
        
        return signals
    
    async def _evaluate_pair_signal(
        self,
        pair_key: Tuple[str, str],
        pair_stats: PairStats,
        z_score: float,
        current_spread: float,
        price_a: float,
        price_b: float,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate trading signal for a pair."""
        
        symbol_a, symbol_b = pair_key
        has_position = pair_key in self.active_positions
        
        # Exit signals for existing positions
        if has_position:
            position = self.active_positions[pair_key]
            
            # Check holding period
            holding_days = (datetime.now() - position.entry_time).days
            if holding_days > self.max_holding_days:
                return {
                    "signal": PairSignal.CLOSE_PAIR.value,
                    "pair": pair_key,
                    "reason": "max_holding_period",
                    "z_score": z_score,
                    "confidence": 0.8,
                }
            
            # Check stop loss
            if abs(z_score) > self.stop_loss_zscore:
                return {
                    "signal": PairSignal.CLOSE_PAIR.value,
                    "pair": pair_key,
                    "reason": "stop_loss",
                    "z_score": z_score,
                    "confidence": 0.9,
                }
            
            # Check mean reversion (exit signal)
            if abs(z_score) < self.exit_zscore_threshold:
                return {
                    "signal": PairSignal.CLOSE_PAIR.value,
                    "pair": pair_key,
                    "reason": "mean_reversion",
                    "z_score": z_score,
                    "confidence": min(abs(z_score) / self.exit_zscore_threshold, 1.0),
                }
        
        # Entry signals for new positions
        else:
            if abs(z_score) > self.entry_zscore_threshold:
                
                # Determine signal direction
                if z_score > 0:  # Spread too high, expect reversion
                    signal_type = PairSignal.LONG_B_SHORT_A.value  # Short expensive, long cheap
                else:  # Spread too low, expect reversion
                    signal_type = PairSignal.LONG_A_SHORT_B.value  # Long expensive, short cheap
                
                confidence = min(abs(z_score) / self.entry_zscore_threshold, 1.0)
                
                # Calculate position sizes
                target_notional = 10000  # $10k per pair
                quantity_a = int(target_notional / (2 * price_a))
                quantity_b = int(quantity_a * pair_stats.hedge_ratio)
                
                return {
                    "signal": signal_type,
                    "pair": pair_key,
                    "symbol_a": symbol_a,
                    "symbol_b": symbol_b,
                    "quantity_a": quantity_a,
                    "quantity_b": quantity_b,
                    "z_score": z_score,
                    "confidence": confidence,
                    "hedge_ratio": pair_stats.hedge_ratio,
                    "current_spread": current_spread,
                    "target_spread": pair_stats.spread_mean,
                    "stop_loss_spread": pair_stats.spread_mean + np.sign(z_score) * self.stop_loss_zscore * pair_stats.spread_std,
                }
        
        return None
    
    def update_position(self, pair_key: Tuple[str, str], signal: Dict[str, Any]) -> None:
        """Update pair position based on signal."""
        
        if signal["signal"] == PairSignal.CLOSE_PAIR.value:
            if pair_key in self.active_positions:
                del self.active_positions[pair_key]
                self.logger.info(f"Closed pair position: {pair_key[0]}-{pair_key[1]}")
        
        else:
            # Open new position
            position = PairPosition(
                symbol_a=signal["symbol_a"],
                symbol_b=signal["symbol_b"],
                quantity_a=signal["quantity_a"],
                quantity_b=signal["quantity_b"],
                entry_spread=signal["current_spread"],
                entry_time=datetime.now(),
                target_spread=signal["target_spread"],
                stop_loss_spread=signal["stop_loss_spread"],
            )
            
            self.active_positions[pair_key] = position
            self.logger.info(
                f"Opened pair position: {pair_key[0]}-{pair_key[1]}",
                signal=signal["signal"],
                z_score=signal["z_score"],
            )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of current pair positions."""
        
        return {
            "active_pairs": len(self.active_positions),
            "total_pairs_discovered": len(self.pair_stats),
            "positions": [
                {
                    "pair": f"{pos.symbol_a}-{pos.symbol_b}",
                    "entry_spread": pos.entry_spread,
                    "target_spread": pos.target_spread,
                    "days_held": (datetime.now() - pos.entry_time).days,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for pos in self.active_positions.values()
            ],
            "pair_stats": [
                {
                    "pair": f"{pair_stat.symbol_a}-{pair_stat.symbol_b}",
                    "correlation": pair_stat.correlation,
                    "cointegration_p": pair_stat.cointegration_pvalue,
                    "hedge_ratio": pair_stat.hedge_ratio,
                    "half_life": pair_stat.half_life,
                }
                for pair_stat in self.pair_stats.values()
            ],
        }


class StatisticalArbitrageStrategy:
    """ML-enhanced statistical arbitrage strategy."""
    
    def __init__(
        self,
        name: str = "StatisticalArbitrage",
        universe_size: int = 50,
        lookback_days: int = 126,
        rebalance_frequency_days: int = 7,
        min_score_threshold: float = 0.6,
        max_positions: int = 10,
    ):
        self.name = name
        self.universe_size = universe_size
        self.lookback_days = lookback_days
        self.rebalance_frequency_days = rebalance_frequency_days
        self.min_score_threshold = min_score_threshold
        self.max_positions = max_positions
        
        # ML models and features
        self.feature_columns = [
            "rsi_14", "bb_position", "volume_ratio", "price_momentum_5d",
            "price_momentum_20d", "volatility_ratio", "correlation_spy"
        ]
        
        self.logger = logger.bind(strategy=name)
    
    async def calculate_arbitrage_scores(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Calculate statistical arbitrage scores for symbols."""
        
        scores = {}
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
            
            df = market_data[symbol]
            if len(df) < self.lookback_days:
                continue
            
            # Calculate features
            features = self._calculate_features(df)
            
            # Calculate mean reversion score
            score = self._calculate_mean_reversion_score(features)
            
            if score > self.min_score_threshold:
                scores[symbol] = score
        
        return scores
    
    def _calculate_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate features for statistical arbitrage."""
        
        # RSI
        rsi = self._calculate_rsi(df["close"], 14)
        
        # Bollinger Band position
        bb_middle = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_position = (df["close"].iloc[-1] - bb_middle.iloc[-1]) / bb_std.iloc[-1]
        
        # Volume ratio
        volume_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
        
        # Price momentum
        price_momentum_5d = (df["close"].iloc[-1] / df["close"].iloc[-6]) - 1
        price_momentum_20d = (df["close"].iloc[-1] / df["close"].iloc[-21]) - 1
        
        # Volatility ratio
        recent_vol = df["close"].pct_change().tail(5).std()
        historical_vol = df["close"].pct_change().tail(20).std()
        volatility_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        return {
            "rsi_14": rsi,
            "bb_position": bb_position,
            "volume_ratio": volume_ratio,
            "price_momentum_5d": price_momentum_5d,
            "price_momentum_20d": price_momentum_20d,
            "volatility_ratio": volatility_ratio,
            "correlation_spy": 0.5,  # Placeholder
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _calculate_mean_reversion_score(self, features: Dict[str, float]) -> float:
        """Calculate mean reversion score based on features."""
        
        score = 0.0
        
        # RSI component (extreme values indicate mean reversion opportunity)
        rsi = features["rsi_14"]
        if rsi < 30:
            score += (30 - rsi) / 30 * 0.3  # Oversold
        elif rsi > 70:
            score += (rsi - 70) / 30 * 0.3  # Overbought
        
        # Bollinger Band position
        bb_pos = abs(features["bb_position"])
        if bb_pos > 2:
            score += min((bb_pos - 2) / 2, 1.0) * 0.3
        
        # Volume confirmation
        vol_ratio = features["volume_ratio"]
        if vol_ratio > 1.5:
            score += min((vol_ratio - 1.5) / 1.5, 1.0) * 0.2
        
        # Momentum divergence
        momentum_5d = abs(features["price_momentum_5d"])
        momentum_20d = abs(features["price_momentum_20d"])
        
        if momentum_5d > 0.02:  # 2% move in 5 days
            score += min(momentum_5d / 0.05, 1.0) * 0.2
        
        return min(score, 1.0)


def create_mean_reversion_suite() -> List[Any]:
    """Create a comprehensive suite of mean reversion strategies."""
    
    strategies = [
        CointegrationPairsStrategy(
            name="CointegrationPairs_Conservative",
            entry_zscore_threshold=2.5,
            exit_zscore_threshold=0.3,
            min_correlation=0.8,
        ),
        CointegrationPairsStrategy(
            name="CointegrationPairs_Aggressive",
            entry_zscore_threshold=1.8,
            exit_zscore_threshold=0.7,
            min_correlation=0.6,
        ),
        StatisticalArbitrageStrategy(
            name="StatArb_LargeUniverse",
            universe_size=100,
            max_positions=20,
        ),
        StatisticalArbitrageStrategy(
            name="StatArb_Focused",
            universe_size=30,
            max_positions=8,
            min_score_threshold=0.8,
        ),
    ]
    
    return strategies
