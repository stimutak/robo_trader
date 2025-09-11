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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    from ..logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    # Fallback logger
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
    ml_score: float = 0.0  # ML confidence score


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
        use_ml_enhancement: bool = True,
        ml_confidence_threshold: float = 0.65,
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
        self.use_ml_enhancement = use_ml_enhancement
        self.ml_confidence_threshold = ml_confidence_threshold

        # State tracking
        self.pair_stats: Dict[Tuple[str, str], PairStats] = {}
        self.active_positions: Dict[Tuple[str, str], PairPosition] = {}
        self.price_history: Dict[str, pd.Series] = {}
        self.last_rebalance: Optional[datetime] = None

        # ML models for pair selection and timing
        self.ml_pair_selector = None
        self.ml_timing_model = None
        self.feature_scaler = StandardScaler()

        self.logger = logger.bind(strategy=name)

    async def find_pairs(
        self, symbols: List[str], price_data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[str, str]]:
        """Find cointegrated pairs from a universe of symbols."""

        self.logger.info(f"Searching for cointegrated pairs among {len(symbols)} symbols")

        # Update price history
        for symbol in symbols:
            if symbol in price_data and len(price_data[symbol]) > 0:
                self.price_history[symbol] = price_data[symbol]["close"]

        valid_pairs = []

        # Test all possible pairs
        for i, symbol_a in enumerate(symbols):
            for j, symbol_b in enumerate(symbols[i + 1 :], i + 1):
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
                pair_stats = await self._test_pair_relationship(
                    symbol_a, symbol_b, aligned_a, aligned_b
                )

                if pair_stats:
                    # Apply ML enhancement for pair selection
                    if self.use_ml_enhancement:
                        ml_score = await self._evaluate_pair_with_ml(
                            pair_stats, aligned_a, aligned_b
                        )
                        if ml_score < self.ml_confidence_threshold:
                            continue
                        pair_stats.ml_score = ml_score

                    self.pair_stats[(symbol_a, symbol_b)] = pair_stats
                    valid_pairs.append((symbol_a, symbol_b))

                    self.logger.info(
                        f"Found valid pair: {symbol_a}-{symbol_b}",
                        correlation=pair_stats.correlation,
                        cointegration_p=pair_stats.cointegration_pvalue,
                        hedge_ratio=pair_stats.hedge_ratio,
                        ml_score=getattr(pair_stats, "ml_score", None),
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

            # Check for signals with ML enhancement
            if self.use_ml_enhancement and self.ml_timing_model:
                signal = await self._evaluate_pair_signal_with_ml(
                    pair_key, pair_stats, z_score, current_spread, price_a, price_b
                )
            else:
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
                    "stop_loss_spread": pair_stats.spread_mean
                    + np.sign(z_score) * self.stop_loss_zscore * pair_stats.spread_std,
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

    async def _evaluate_pair_with_ml(
        self, pair_stats: PairStats, prices_a: pd.Series, prices_b: pd.Series
    ) -> float:
        """
        Evaluate pair quality using ML models.
        Base implementation returns default confidence.
        Override in subclasses for ML enhancement.
        """
        # Default implementation without ML - just return a base confidence score
        # based on statistical metrics
        confidence = 0.5

        # Boost confidence based on cointegration strength
        if pair_stats.cointegration_pvalue < 0.01:
            confidence += 0.2
        elif pair_stats.cointegration_pvalue < 0.05:
            confidence += 0.1

        # Boost confidence based on half-life (prefer shorter half-lives)
        if pair_stats.half_life < 10:
            confidence += 0.2
        elif pair_stats.half_life < 20:
            confidence += 0.1

        return min(confidence, 1.0)


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
        use_sector_neutrality: bool = True,
        use_ml_ranking: bool = True,
    ):
        self.name = name
        self.universe_size = universe_size
        self.lookback_days = lookback_days
        self.rebalance_frequency_days = rebalance_frequency_days
        self.min_score_threshold = min_score_threshold
        self.max_positions = max_positions
        self.use_sector_neutrality = use_sector_neutrality
        self.use_ml_ranking = use_ml_ranking

        # ML models and features
        self.feature_columns = [
            "rsi_14",
            "bb_position",
            "volume_ratio",
            "price_momentum_5d",
            "price_momentum_20d",
            "volatility_ratio",
            "correlation_spy",
            "relative_strength",
            "mean_reversion_score",
            "liquidity_score",
        ]

        # ML components
        self.ml_ranker = None
        self.feature_scaler = StandardScaler()
        self.sector_mappings = {}  # Symbol to sector mapping
        self.active_positions = {}
        self.last_rebalance = None

        self.logger = logger.bind(strategy=name)

    async def calculate_arbitrage_scores(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Calculate statistical arbitrage scores for symbols."""

        scores = {}
        all_features = {}

        # Calculate features for all symbols
        for symbol in symbols:
            if symbol not in market_data:
                continue

            df = market_data[symbol]
            if len(df) < self.lookback_days:
                continue

            # Calculate features
            features = self._calculate_features(df, market_data)
            all_features[symbol] = features

        # Use ML ranking if available
        if self.use_ml_ranking and all_features:
            scores = await self._ml_rank_opportunities(all_features)
        else:
            # Fallback to rule-based scoring
            for symbol, features in all_features.items():
                score = self._calculate_mean_reversion_score(features)
                if score > self.min_score_threshold:
                    scores[symbol] = score

        # Apply sector neutrality if enabled
        if self.use_sector_neutrality:
            scores = self._apply_sector_neutrality(scores)

        return scores

    def _calculate_features(
        self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate enhanced features for statistical arbitrage."""

        # RSI
        rsi = self._calculate_rsi(df["close"], 14)

        # Bollinger Band position
        bb_middle = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_position = (df["close"].iloc[-1] - bb_middle.iloc[-1]) / (bb_std.iloc[-1] + 1e-10)

        # Volume ratio
        volume_ratio = df["volume"].iloc[-1] / (df["volume"].rolling(20).mean().iloc[-1] + 1e-10)

        # Price momentum
        price_momentum_5d = (df["close"].iloc[-1] / df["close"].iloc[-6]) - 1 if len(df) > 5 else 0
        price_momentum_20d = (
            (df["close"].iloc[-1] / df["close"].iloc[-21]) - 1 if len(df) > 20 else 0
        )

        # Volatility ratio
        recent_vol = df["close"].pct_change().tail(5).std()
        historical_vol = df["close"].pct_change().tail(20).std()
        volatility_ratio = recent_vol / (historical_vol + 1e-10)

        # Relative strength vs market
        relative_strength = self._calculate_relative_strength(df, market_data)

        # Mean reversion score
        mean_reversion_score = self._calculate_advanced_mean_reversion_score(df)

        # Liquidity score
        liquidity_score = self._calculate_liquidity_score(df)

        # Market correlation
        correlation_spy = self._calculate_market_correlation(df, market_data)

        return {
            "rsi_14": rsi,
            "bb_position": bb_position,
            "volume_ratio": volume_ratio,
            "price_momentum_5d": price_momentum_5d,
            "price_momentum_20d": price_momentum_20d,
            "volatility_ratio": volatility_ratio,
            "correlation_spy": correlation_spy,
            "relative_strength": relative_strength,
            "mean_reversion_score": mean_reversion_score,
            "liquidity_score": liquidity_score,
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
        """Calculate enhanced mean reversion score based on features."""

        score = 0.0
        weights = {
            "rsi": 0.25,
            "bb": 0.25,
            "volume": 0.15,
            "momentum": 0.15,
            "mean_reversion": 0.1,
            "liquidity": 0.1,
        }

        # RSI component (extreme values indicate mean reversion opportunity)
        rsi = features.get("rsi_14", 50)
        if rsi < 30:
            score += (30 - rsi) / 30 * weights["rsi"]  # Oversold
        elif rsi > 70:
            score += (rsi - 70) / 30 * weights["rsi"]  # Overbought

        # Bollinger Band position
        bb_pos = abs(features.get("bb_position", 0))
        if bb_pos > 2:
            score += min((bb_pos - 2) / 2, 1.0) * weights["bb"]

        # Volume confirmation
        vol_ratio = features.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            score += min((vol_ratio - 1.5) / 1.5, 1.0) * weights["volume"]

        # Momentum divergence
        momentum_5d = abs(features.get("price_momentum_5d", 0))
        momentum_20d = abs(features.get("price_momentum_20d", 0))

        if momentum_5d > 0.02:  # 2% move in 5 days
            if momentum_20d < momentum_5d:  # Acceleration
                score += min(momentum_5d / 0.05, 1.0) * weights["momentum"]

        # Add mean reversion and liquidity scores
        score += features.get("mean_reversion_score", 0) * weights["mean_reversion"]
        score += features.get("liquidity_score", 0) * weights["liquidity"]

        return min(score, 1.0)

    def _calculate_relative_strength(
        self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate relative strength vs market."""
        try:
            if market_data and "SPY" in market_data:
                spy_returns = market_data["SPY"]["close"].pct_change().tail(20).mean()
                stock_returns = df["close"].pct_change().tail(20).mean()
                return (stock_returns - spy_returns) / (abs(spy_returns) + 1e-10)
            return 0.0
        except Exception:
            return 0.0

    def _calculate_advanced_mean_reversion_score(self, df: pd.DataFrame) -> float:
        """Calculate advanced mean reversion score."""
        try:
            # Calculate z-score
            price = df["close"].iloc[-1]
            mean_20 = df["close"].tail(20).mean()
            std_20 = df["close"].tail(20).std()
            z_score = (price - mean_20) / (std_20 + 1e-10)

            # Check for mean crossing frequency
            prices = df["close"].tail(50)
            mean = prices.rolling(20).mean()
            crosses = np.sum(np.diff(np.sign(prices - mean)) != 0)
            crossing_freq = crosses / len(prices)

            # Combined score
            reversion_score = 0.0
            if abs(z_score) > 2:
                reversion_score += 0.5
            if crossing_freq > 0.2:  # Frequent mean crossings
                reversion_score += 0.3
            if abs(z_score) > 2.5 and crossing_freq > 0.15:
                reversion_score += 0.2

            return min(reversion_score, 1.0)
        except Exception:
            return 0.0

    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and spread."""
        try:
            # Volume consistency
            volume_mean = df["volume"].tail(20).mean()
            volume_std = df["volume"].tail(20).std()
            volume_cv = volume_std / (volume_mean + 1e-10)  # Coefficient of variation

            # Price impact (using high-low as proxy for spread)
            avg_spread = ((df["high"] - df["low"]) / df["close"]).tail(20).mean()

            # Dollar volume
            dollar_volume = (df["close"] * df["volume"]).tail(20).mean()

            # Score calculation
            liquidity = 0.0
            if volume_cv < 0.5:  # Consistent volume
                liquidity += 0.4
            if avg_spread < 0.01:  # Tight spread
                liquidity += 0.3
            if dollar_volume > 1e6:  # Good dollar volume
                liquidity += 0.3

            return min(liquidity, 1.0)
        except Exception:
            return 0.5

    def _calculate_market_correlation(
        self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate correlation with market index."""
        try:
            if market_data and "SPY" in market_data:
                spy_returns = market_data["SPY"]["close"].pct_change().tail(60)
                stock_returns = df["close"].pct_change().tail(60)

                # Align indices
                common_dates = spy_returns.index.intersection(stock_returns.index)
                if len(common_dates) > 20:
                    correlation = stock_returns.loc[common_dates].corr(
                        spy_returns.loc[common_dates]
                    )
                    return correlation
            return 0.5
        except Exception:
            return 0.5

    async def _ml_rank_opportunities(
        self, all_features: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Use ML to rank arbitrage opportunities."""
        try:
            # Train ML ranker if needed
            if self.ml_ranker is None:
                self._train_ml_ranker(all_features)

            if self.ml_ranker is None:
                # Fallback to rule-based
                scores = {}
                for symbol, features in all_features.items():
                    scores[symbol] = self._calculate_mean_reversion_score(features)
                return scores

            # Prepare feature matrix
            symbols = list(all_features.keys())
            X = np.array(
                [
                    [features.get(col, 0) for col in self.feature_columns]
                    for features in all_features.values()
                ]
            )

            # Scale features
            X_scaled = self.feature_scaler.transform(X)

            # Get ML predictions
            ml_scores = self.ml_ranker.predict_proba(X_scaled)[:, 1]

            # Combine with rule-based scores
            final_scores = {}
            for i, symbol in enumerate(symbols):
                rule_score = self._calculate_mean_reversion_score(all_features[symbol])
                # Weighted average of ML and rule-based
                final_scores[symbol] = 0.7 * ml_scores[i] + 0.3 * rule_score

            return final_scores

        except Exception as e:
            self.logger.error(f"ML ranking error: {e}")
            # Fallback to rule-based
            scores = {}
            for symbol, features in all_features.items():
                scores[symbol] = self._calculate_mean_reversion_score(features)
            return scores

    def _train_ml_ranker(self, sample_features: Dict[str, Dict[str, float]]):
        """Train ML model for opportunity ranking."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            # Generate synthetic training data
            n_samples = 5000
            n_features = len(self.feature_columns)

            X_train = np.random.randn(n_samples, n_features)

            # Create labels based on ideal arbitrage characteristics
            y_train = np.zeros(n_samples)
            for i in range(n_samples):
                # Good arbitrage: extreme RSI, high BB position, volume spike
                rsi = X_train[i, 0] * 15 + 50  # Scale to RSI range
                bb_pos = abs(X_train[i, 1])
                vol_ratio = X_train[i, 2] + 1

                if (rsi < 30 or rsi > 70) and bb_pos > 2 and vol_ratio > 1.5:
                    y_train[i] = 1

            # Fit scaler
            self.feature_scaler.fit(X_train)
            X_scaled = self.feature_scaler.transform(X_train)

            # Train model
            self.ml_ranker = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
            self.ml_ranker.fit(X_scaled, y_train)

            self.logger.info("ML ranker trained successfully")

        except Exception as e:
            self.logger.error(f"ML ranker training error: {e}")
            self.ml_ranker = None

    def _apply_sector_neutrality(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply sector neutrality to prevent sector concentration."""
        try:
            # Group by sectors (simplified - in production would use real sector data)
            sector_groups = {}
            for symbol in scores:
                # Simple sector assignment based on first letter (placeholder)
                sector = "Tech" if symbol[0] in "TNMA" else "Other"
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(symbol)

            # Normalize scores within sectors
            adjusted_scores = {}
            for sector, symbols in sector_groups.items():
                sector_scores = {s: scores[s] for s in symbols}
                if len(sector_scores) > 1:
                    # Z-score normalization within sector
                    values = list(sector_scores.values())
                    mean = np.mean(values)
                    std = np.std(values)
                    for symbol in symbols:
                        if std > 0:
                            adjusted_scores[symbol] = (scores[symbol] - mean) / std
                        else:
                            adjusted_scores[symbol] = scores[symbol]
                else:
                    adjusted_scores.update(sector_scores)

            # Re-scale to 0-1 range
            if adjusted_scores:
                min_score = min(adjusted_scores.values())
                max_score = max(adjusted_scores.values())
                if max_score > min_score:
                    for symbol in adjusted_scores:
                        adjusted_scores[symbol] = (adjusted_scores[symbol] - min_score) / (
                            max_score - min_score
                        )

            return adjusted_scores

        except Exception as e:
            self.logger.error(f"Sector neutrality error: {e}")
            return scores

    def generate_portfolio_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Generate portfolio weights from arbitrage scores."""
        # Sort by score
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Select top N
        selected = sorted_symbols[: self.max_positions]

        # Calculate weights (risk parity approach)
        weights = {}
        total_score = sum(score for _, score in selected)

        if total_score > 0:
            for symbol, score in selected:
                # Weight proportional to score with max cap
                weight = score / total_score
                weights[symbol] = min(weight, 1.0 / self.max_positions * 2)  # Max 2x equal weight

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w / total_weight for s, w in weights.items()}

        return weights


def create_mean_reversion_suite() -> List[Any]:
    """Create a comprehensive suite of mean reversion strategies."""

    strategies = [
        # Cointegration-based pairs trading
        CointegrationPairsStrategy(
            name="CointegrationPairs_Conservative",
            entry_zscore_threshold=2.5,
            exit_zscore_threshold=0.3,
            min_correlation=0.8,
            use_ml_enhancement=True,
        ),
        CointegrationPairsStrategy(
            name="CointegrationPairs_Aggressive",
            entry_zscore_threshold=1.8,
            exit_zscore_threshold=0.7,
            min_correlation=0.6,
            use_ml_enhancement=True,
        ),
        # ML-enhanced pairs trading
        MLEnhancedPairsTrading(
            name="MLPairs_HighFreq",
            lookback_days=126,
            entry_zscore_threshold=1.5,
            max_holding_days=10,
        ),
        # Statistical arbitrage
        StatisticalArbitrageStrategy(
            name="StatArb_LargeUniverse",
            universe_size=100,
            max_positions=20,
            use_sector_neutrality=True,
            use_ml_ranking=True,
        ),
        StatisticalArbitrageStrategy(
            name="StatArb_Focused",
            universe_size=30,
            max_positions=8,
            min_score_threshold=0.8,
            use_ml_ranking=True,
        ),
        StatisticalArbitrageStrategy(
            name="StatArb_SectorNeutral",
            universe_size=50,
            max_positions=15,
            use_sector_neutrality=True,
            use_ml_ranking=False,  # Pure sector-neutral
        ),
    ]

    return strategies


class MLEnhancedPairsTrading(CointegrationPairsStrategy):
    """ML-enhanced pairs trading with advanced feature engineering."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ml_features_cache = {}

    async def _evaluate_pair_with_ml(
        self, pair_stats: PairStats, prices_a: pd.Series, prices_b: pd.Series
    ) -> float:
        """Evaluate pair quality using ML models."""
        try:
            # Extract ML features
            features = self._extract_pair_features(pair_stats, prices_a, prices_b)

            # Train or use existing model
            if self.ml_pair_selector is None:
                self._train_pair_selector(features)
                return 0.7  # Default score during training

            # Predict pair quality
            X = np.array([features])
            X_scaled = self.feature_scaler.transform(X)
            ml_score = self.ml_pair_selector.predict_proba(X_scaled)[0, 1]

            return ml_score

        except Exception as e:
            self.logger.error(f"ML pair evaluation error: {e}")
            return 0.5

    def _extract_pair_features(
        self, pair_stats: PairStats, prices_a: pd.Series, prices_b: pd.Series
    ) -> List[float]:
        """Extract features for ML pair evaluation."""
        # Basic statistical features
        features = [
            pair_stats.correlation,
            pair_stats.cointegration_pvalue,
            pair_stats.hedge_ratio,
            pair_stats.half_life,
            pair_stats.spread_std / pair_stats.spread_mean if pair_stats.spread_mean != 0 else 1.0,
        ]

        # Price dynamics features
        returns_a = prices_a.pct_change().dropna()
        returns_b = prices_b.pct_change().dropna()

        features.extend(
            [
                returns_a.std(),
                returns_b.std(),
                returns_a.skew(),
                returns_b.skew(),
                returns_a.kurtosis(),
                returns_b.kurtosis(),
            ]
        )

        # Spread dynamics
        spread = prices_a - pair_stats.hedge_ratio * prices_b
        spread_returns = spread.pct_change().dropna()

        features.extend(
            [
                spread_returns.std(),
                spread_returns.skew(),
                spread_returns.kurtosis(),
                self._calculate_hurst_exponent(spread),
            ]
        )

        return features

    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """Calculate Hurst exponent to measure mean reversion tendency."""
        try:
            # Simplified Hurst calculation using R/S analysis
            lags = range(2, min(20, len(series) // 2))
            tau = []

            for lag in lags:
                # Calculate R/S statistic
                subseries = [series[i : i + lag] for i in range(0, len(series), lag)]
                rs_values = []

                for sub in subseries:
                    if len(sub) < 2:
                        continue
                    mean = sub.mean()
                    std = sub.std()
                    if std == 0:
                        continue

                    cumsum = (sub - mean).cumsum()
                    R = cumsum.max() - cumsum.min()
                    rs_values.append(R / std)

                if rs_values:
                    tau.append(np.mean(rs_values))

            if len(tau) > 2:
                # Fit log-log relationship
                log_lags = np.log(list(lags[: len(tau)]))
                log_tau = np.log(tau)

                # Linear regression for Hurst exponent
                reg = LinearRegression().fit(log_lags.reshape(-1, 1), log_tau)
                hurst = reg.coef_[0]

                return np.clip(hurst, 0, 1)

        except Exception:
            pass

        return 0.5  # Neutral value

    def _train_pair_selector(self, initial_features: List[float]):
        """Train ML model for pair selection."""
        try:
            # For initial training, create synthetic training data
            # In production, this would use historical pair performance
            from sklearn.ensemble import RandomForestClassifier

            # Generate synthetic training data (placeholder)
            n_samples = 1000
            n_features = len(initial_features)

            X_train = np.random.randn(n_samples, n_features)
            # Simple rule: good pairs have high correlation, low p-value
            y_train = (
                (X_train[:, 0] > 0.7)
                & (X_train[:, 1] < 0.05)  # High correlation
                & (X_train[:, 3] < 30)  # Low p-value  # Reasonable half-life
            ).astype(int)

            # Fit scaler
            self.feature_scaler.fit(X_train)
            X_scaled = self.feature_scaler.transform(X_train)

            # Train model
            self.ml_pair_selector = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
            self.ml_pair_selector.fit(X_scaled, y_train)

            self.logger.info("ML pair selector trained")

        except Exception as e:
            self.logger.error(f"ML training error: {e}")
            self.ml_pair_selector = None

    async def _evaluate_pair_signal_with_ml(
        self,
        pair_key: Tuple[str, str],
        pair_stats: PairStats,
        z_score: float,
        current_spread: float,
        price_a: float,
        price_b: float,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate pair signal with ML timing enhancement."""
        # Get base signal
        base_signal = await self._evaluate_pair_signal(
            pair_key, pair_stats, z_score, current_spread, price_a, price_b
        )

        if not base_signal:
            return None

        try:
            # Extract timing features
            timing_features = self._extract_timing_features(pair_stats, z_score, current_spread)

            # Train timing model if needed
            if self.ml_timing_model is None:
                self._train_timing_model()

            if self.ml_timing_model:
                # Predict timing quality
                X = np.array([timing_features])
                timing_confidence = self.ml_timing_model.predict_proba(X)[0, 1]

                # Adjust signal confidence
                base_signal["confidence"] = base_signal["confidence"] * (
                    0.5 + 0.5 * timing_confidence
                )
                base_signal["ml_timing_score"] = timing_confidence

                # Reject low confidence signals
                if base_signal["confidence"] < 0.4:
                    return None

        except Exception as e:
            self.logger.error(f"ML timing evaluation error: {e}")

        return base_signal

    def _extract_timing_features(
        self, pair_stats: PairStats, z_score: float, current_spread: float
    ) -> List[float]:
        """Extract features for timing prediction."""
        features = [
            abs(z_score),
            z_score**2,  # Non-linear relationship
            current_spread / pair_stats.spread_std if pair_stats.spread_std != 0 else 0,
            pair_stats.half_life,
            1.0 if abs(z_score) > 2.5 else 0.0,  # Extreme indicator
            (datetime.now().hour) / 24.0,  # Time of day feature
        ]

        # Add recent spread momentum if available
        if hasattr(self, "spread_history"):
            recent_spreads = self.spread_history.get(
                pair_stats.symbol_a + "_" + pair_stats.symbol_b, []
            )
            if len(recent_spreads) >= 5:
                spread_momentum = (recent_spreads[-1] - recent_spreads[-5]) / pair_stats.spread_std
                features.append(spread_momentum)
            else:
                features.append(0.0)
        else:
            features.append(0.0)

        return features

    def _train_timing_model(self):
        """Train ML model for entry/exit timing."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            # Generate synthetic training data
            n_samples = 2000
            n_features = 7

            X_train = np.random.randn(n_samples, n_features)
            # Simple rule: good timing when z-score is extreme but not too extreme
            y_train = (
                (np.abs(X_train[:, 0]) > 2.0)
                & (np.abs(X_train[:, 0]) < 3.5)  # Significant z-score
                & (X_train[:, 3] < 30)  # Not too extreme  # Reasonable half-life
            ).astype(int)

            # Train model
            self.ml_timing_model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            )
            self.ml_timing_model.fit(X_train, y_train)

            self.logger.info("ML timing model trained")

        except Exception as e:
            self.logger.error(f"ML timing training error: {e}")
            self.ml_timing_model = None
