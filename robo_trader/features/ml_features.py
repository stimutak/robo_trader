"""
Advanced ML features for trading strategies.

This module implements:
- Cross-asset correlations
- Market regime indicators
- Microstructure features
- Order flow imbalance
- Statistical arbitrage signals
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..logger import get_logger


@dataclass
class MarketRegime:
    """Market regime classification."""

    regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'
    confidence: float  # 0-1 confidence score
    volatility_percentile: float
    trend_strength: float
    change_probability: float  # Probability of regime change


class MLFeatureEngine:
    """Advanced ML feature engineering."""

    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.logger = get_logger("features.ml")

        # Data storage for cross-asset analysis
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.pca_components: Optional[np.ndarray] = None

        # Market microstructure
        self.order_flow: Dict[str, deque] = {}
        self.trade_imbalance: Dict[str, float] = {}

        # Regime detection
        self.regime_history: Dict[str, List[MarketRegime]] = {}
        self.regime_model = None

        # Statistical arbitrage
        self.pair_relationships: Dict[Tuple[str, str], Dict] = {}

    def calculate_cross_asset_features(
        self, symbol: str, price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate cross-asset correlation features."""
        features = {}

        try:
            # Store price data
            self.price_data = price_data

            # Calculate correlation matrix
            returns = pd.DataFrame()
            for sym, df in price_data.items():
                if len(df) >= 2:
                    # Handle both lowercase and uppercase column names
                    close_col = "close" if "close" in df.columns else "Close"
                    returns[sym] = df[close_col].pct_change()

            if len(returns.columns) >= 2:
                # Rolling correlation with market (SPY as proxy)
                if "SPY" in returns.columns and symbol in returns.columns:
                    features["correlation_spy"] = (
                        returns[symbol].tail(20).corr(returns["SPY"].tail(20))
                    )

                # Average correlation with other assets
                if symbol in returns.columns:
                    correlations = []
                    for other_symbol in returns.columns:
                        if other_symbol != symbol:
                            corr = returns[symbol].tail(20).corr(returns[other_symbol].tail(20))
                            if not pd.isna(corr):
                                correlations.append(corr)

                    if correlations:
                        features["avg_correlation"] = np.mean(correlations)
                        features["max_correlation"] = np.max(correlations)
                        features["min_correlation"] = np.min(correlations)

                # PCA features
                pca_features = self._calculate_pca_features(returns, symbol)
                features.update(pca_features)

            # Beta calculation
            if "SPY" in returns.columns and symbol in returns.columns:
                beta = self._calculate_beta(returns[symbol], returns["SPY"])
                if beta is not None:
                    features["beta"] = beta

        except Exception as e:
            self.logger.error(f"Error calculating cross-asset features: {e}")

        return features

    def _calculate_pca_features(self, returns: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate PCA-based features."""
        features = {}

        try:
            if len(returns.columns) < 3 or symbol not in returns.columns:
                return features

            # Prepare data
            returns_clean = returns.dropna()
            if len(returns_clean) < 20:
                return features

            # Standardize
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_clean.tail(self.lookback_window))

            # PCA
            pca = PCA(n_components=min(3, len(returns.columns)))
            components = pca.fit_transform(returns_scaled)

            # Get symbol's loadings on principal components
            symbol_idx = returns.columns.get_loc(symbol)
            features["pca_loading_1"] = pca.components_[0, symbol_idx]
            if len(pca.components_) > 1:
                features["pca_loading_2"] = pca.components_[1, symbol_idx]
            if len(pca.components_) > 2:
                features["pca_loading_3"] = pca.components_[2, symbol_idx]

            # Variance explained
            features["pca_variance_ratio_1"] = pca.explained_variance_ratio_[0]

        except Exception as e:
            self.logger.error(f"PCA calculation error: {e}")

        return features

    def _calculate_beta(
        self, asset_returns: pd.Series, market_returns: pd.Series
    ) -> Optional[float]:
        """Calculate beta relative to market."""
        try:
            # Align series
            aligned = pd.DataFrame({"asset": asset_returns, "market": market_returns}).dropna()

            if len(aligned) < 20:
                return None

            # Calculate beta
            covariance = aligned["asset"].cov(aligned["market"])
            market_variance = aligned["market"].var()

            if market_variance > 0:
                return covariance / market_variance

        except Exception:
            pass

        return None

    def detect_market_regime(self, symbol: str, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime."""
        try:
            # Handle both lowercase and uppercase column names
            close_col = "close" if "close" in df.columns else "Close"
            high_col = "high" if "high" in df.columns else "High"
            low_col = "low" if "low" in df.columns else "Low"
            open_col = "open" if "open" in df.columns else "Open"
            volume_col = "volume" if "volume" in df.columns else "Volume"
            if len(df) < 50:
                return MarketRegime(
                    regime="unknown",
                    confidence=0.0,
                    volatility_percentile=0.5,
                    trend_strength=0.0,
                    change_probability=0.0,
                )

            # Calculate returns
            returns = df[close_col].pct_change().dropna()

            # Calculate trend using linear regression
            x = np.arange(len(df.tail(20)))
            y = df[close_col].tail(20).values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Normalize slope by price level
            normalized_slope = slope / df[close_col].mean() * 100
            trend_strength = abs(r_value)  # R-squared as trend strength

            # Calculate volatility
            recent_vol = returns.tail(20).std()
            historical_vol = returns.std()
            volatility_percentile = (
                stats.percentileofscore(returns.rolling(20).std().dropna(), recent_vol) / 100
            )

            # Determine regime
            if trend_strength > 0.7:
                if normalized_slope > 0.5:
                    regime = "trending_up"
                elif normalized_slope < -0.5:
                    regime = "trending_down"
                else:
                    regime = "ranging"
            elif volatility_percentile > 0.8:
                regime = "volatile"
            else:
                regime = "ranging"

            # Calculate regime change probability
            change_prob = self._calculate_regime_change_probability(
                symbol, regime, volatility_percentile, trend_strength
            )

            market_regime = MarketRegime(
                regime=regime,
                confidence=max(trend_strength, volatility_percentile),
                volatility_percentile=volatility_percentile,
                trend_strength=trend_strength,
                change_probability=change_prob,
            )

            # Store in history
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            self.regime_history[symbol].append(market_regime)
            if len(self.regime_history[symbol]) > 100:
                self.regime_history[symbol] = self.regime_history[symbol][-100:]

            return market_regime

        except Exception as e:
            self.logger.error(f"Regime detection error: {e}")
            return MarketRegime(
                regime="unknown",
                confidence=0.0,
                volatility_percentile=0.5,
                trend_strength=0.0,
                change_probability=0.0,
            )

    def _calculate_regime_change_probability(
        self, symbol: str, current_regime: str, volatility_percentile: float, trend_strength: float
    ) -> float:
        """Calculate probability of regime change."""
        try:
            # Base probability increases with volatility
            base_prob = volatility_percentile * 0.3

            # Adjust based on trend strength (weak trends more likely to change)
            trend_adjustment = (1 - trend_strength) * 0.3

            # Check regime duration
            if symbol in self.regime_history and len(self.regime_history[symbol]) > 1:
                # Count consecutive same regime
                consecutive = 0
                for regime in reversed(self.regime_history[symbol][:-1]):
                    if regime.regime == current_regime:
                        consecutive += 1
                    else:
                        break

                # Longer duration increases change probability
                duration_adjustment = min(0.4, consecutive * 0.05)
            else:
                duration_adjustment = 0.1

            return min(1.0, base_prob + trend_adjustment + duration_adjustment)

        except Exception:
            return 0.5

    def calculate_microstructure_features(
        self, symbol: str, trades: List[Dict], quotes: List[Dict]
    ) -> Dict[str, float]:
        """Calculate market microstructure features."""
        features = {}

        try:
            if not trades or not quotes:
                return features

            # Order flow imbalance
            buy_volume = sum(t["size"] for t in trades if t.get("side") == "buy")
            sell_volume = sum(t["size"] for t in trades if t.get("side") == "sell")
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                features["order_flow_imbalance"] = (buy_volume - sell_volume) / total_volume
                features["buy_ratio"] = buy_volume / total_volume

            # Trade size distribution
            trade_sizes = [t["size"] for t in trades]
            if trade_sizes:
                features["avg_trade_size"] = np.mean(trade_sizes)
                features["trade_size_std"] = np.std(trade_sizes)
                features["large_trade_ratio"] = sum(
                    1 for s in trade_sizes if s > np.percentile(trade_sizes, 90)
                ) / len(trade_sizes)

            # Quote dynamics
            if len(quotes) >= 2:
                # Bid-ask spread statistics
                spreads = [(q["ask"] - q["bid"]) for q in quotes if q["ask"] > q["bid"]]
                if spreads:
                    features["avg_spread"] = np.mean(spreads)
                    features["spread_volatility"] = np.std(spreads)

                # Quote imbalance
                bid_sizes = [q["bid_size"] for q in quotes]
                ask_sizes = [q["ask_size"] for q in quotes]
                if bid_sizes and ask_sizes:
                    avg_bid_size = np.mean(bid_sizes)
                    avg_ask_size = np.mean(ask_sizes)
                    if avg_bid_size + avg_ask_size > 0:
                        features["quote_imbalance"] = (avg_bid_size - avg_ask_size) / (
                            avg_bid_size + avg_ask_size
                        )

            # Effective spread (if trades have prices)
            if trades and quotes:
                effective_spreads = []
                for trade in trades[-10:]:  # Last 10 trades
                    # Find nearest quote
                    trade_time = trade.get("timestamp")
                    if trade_time:
                        nearest_quote = min(
                            quotes, key=lambda q: abs(q.get("timestamp", 0) - trade_time)
                        )
                        mid_price = (nearest_quote["bid"] + nearest_quote["ask"]) / 2
                        effective_spread = 2 * abs(trade["price"] - mid_price)
                        effective_spreads.append(effective_spread)

                if effective_spreads:
                    features["effective_spread"] = np.mean(effective_spreads)

            # Kyle's lambda (price impact)
            if len(trades) >= 10:
                prices = [t["price"] for t in trades[-20:]]
                volumes = [t["size"] for t in trades[-20:]]
                if len(prices) == len(volumes):
                    price_changes = np.diff(prices)
                    cumulative_volume = np.array(volumes[:-1])

                    if len(price_changes) > 0 and cumulative_volume.sum() > 0:
                        # Simple price impact measure
                        features["price_impact"] = (
                            np.sum(np.abs(price_changes)) / cumulative_volume.sum()
                        )

        except Exception as e:
            self.logger.error(f"Microstructure features error: {e}")

        return features

    def calculate_statistical_arbitrage_signals(
        self, symbol1: str, symbol2: str, window: int = 60
    ) -> Dict[str, float]:
        """Calculate statistical arbitrage signals for pair trading."""
        features = {}

        try:
            if symbol1 not in self.price_data or symbol2 not in self.price_data:
                return features

            df1 = self.price_data[symbol1]
            df2 = self.price_data[symbol2]

            # Align data
            aligned = pd.DataFrame({"price1": df1["close"], "price2": df2["close"]}).dropna()

            if len(aligned) < window:
                return features

            # Calculate spread
            # Use log prices for better statistical properties
            log_price1 = np.log(aligned["price1"])
            log_price2 = np.log(aligned["price2"])

            # Calculate hedge ratio using rolling regression
            X = log_price2.tail(window).values.reshape(-1, 1)
            y = log_price1.tail(window).values

            # Simple OLS
            X_with_const = np.column_stack([np.ones(len(X)), X])
            try:
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                hedge_ratio = beta[1]

                # Calculate spread
                spread = log_price1 - hedge_ratio * log_price2
                current_spread = spread.iloc[-1]

                # Calculate z-score
                spread_mean = spread.tail(window).mean()
                spread_std = spread.tail(window).std()

                if spread_std > 0:
                    z_score = (current_spread - spread_mean) / spread_std
                    features[f"pair_{symbol1}_{symbol2}_zscore"] = z_score

                    # Mean reversion signal
                    if abs(z_score) > 2:
                        features[f"pair_{symbol1}_{symbol2}_signal"] = -np.sign(z_score)
                    else:
                        features[f"pair_{symbol1}_{symbol2}_signal"] = 0

                # Cointegration test (simplified)
                # Check if spread is stationary using Augmented Dickey-Fuller
                from statsmodels.tsa.stattools import adfuller

                adf_result = adfuller(spread.tail(window))
                features[f"pair_{symbol1}_{symbol2}_adf_pvalue"] = adf_result[1]
                features[f"pair_{symbol1}_{symbol2}_cointegrated"] = (
                    1.0 if adf_result[1] < 0.05 else 0.0
                )

                # Half-life of mean reversion
                spread_lag = spread.shift(1)
                spread_diff = spread - spread_lag
                spread_lag_centered = spread_lag - spread_mean

                # Remove NaN values
                valid_idx = ~(pd.isna(spread_diff) | pd.isna(spread_lag_centered))
                if valid_idx.sum() > 10:
                    theta = np.linalg.lstsq(
                        spread_lag_centered[valid_idx].values.reshape(-1, 1),
                        spread_diff[valid_idx].values,
                        rcond=None,
                    )[0][0]

                    if theta < 0:
                        half_life = -np.log(2) / theta
                        features[f"pair_{symbol1}_{symbol2}_halflife"] = half_life

            except Exception as e:
                self.logger.debug(f"Regression error in stat arb: {e}")

        except Exception as e:
            self.logger.error(f"Statistical arbitrage calculation error: {e}")

        return features

    def calculate_sentiment_features(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment and positioning features."""
        features = {}

        try:
            # Handle both lowercase and uppercase column names
            close_col = "close" if "close" in df.columns else "Close"
            high_col = "high" if "high" in df.columns else "High"
            low_col = "low" if "low" in df.columns else "Low"
            open_col = "open" if "open" in df.columns else "Open"
            volume_col = "volume" if "volume" in df.columns else "Volume"
            if len(df) < 20:
                return features

            # Put-call ratio proxy (using volatility skew if available)
            returns = df[close_col].pct_change().dropna()

            # Skewness as sentiment proxy
            if len(returns) >= 20:
                features["return_skewness"] = stats.skew(returns.tail(20))
                features["return_kurtosis"] = stats.kurtosis(returns.tail(20))

            # Volume-price divergence
            if "volume" in df.columns:
                # Calculate correlation between volume and absolute returns
                abs_returns = returns.abs()
                volume_changes = df[volume_col].pct_change()

                if len(abs_returns) >= 20 and len(volume_changes) >= 20:
                    vol_ret_corr = abs_returns.tail(20).corr(volume_changes.tail(20))
                    if not pd.isna(vol_ret_corr):
                        features["volume_return_correlation"] = vol_ret_corr

                # Accumulation/Distribution line
                clv = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (
                    df[high_col] - df[low_col] + 0.0001
                )
                adl = (clv * df[volume_col]).cumsum()

                if len(adl) >= 20:
                    # ADL trend
                    x = np.arange(20)
                    y = adl.tail(20).values
                    slope, _, r_value, _, _ = stats.linregress(x, y)
                    features["adl_trend"] = slope / (abs(adl.mean()) + 1)
                    features["adl_strength"] = r_value**2

            # Price position in range
            high_20 = df[high_col].tail(20).max()
            low_20 = df[low_col].tail(20).min()
            current_price = df[close_col].iloc[-1]

            if high_20 > low_20:
                features["price_position"] = (current_price - low_20) / (high_20 - low_20)

            # Momentum divergence
            if len(df) >= 40:
                price_change_10 = (df[close_col].iloc[-1] / df[close_col].iloc[-10] - 1) * 100
                price_change_20 = (df[close_col].iloc[-10] / df[close_col].iloc[-20] - 1) * 100
                features["momentum_divergence"] = price_change_10 - price_change_20

        except Exception as e:
            self.logger.error(f"Sentiment features error: {e}")

        return features

    def calculate_all_ml_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        trades: Optional[List[Dict]] = None,
        quotes: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """Calculate all ML features."""
        all_features = {}

        # Cross-asset features
        if price_data:
            cross_asset = self.calculate_cross_asset_features(symbol, price_data)
            all_features.update(cross_asset)

        # Market regime
        regime = self.detect_market_regime(symbol, df)
        all_features["regime_trending_up"] = 1.0 if regime.regime == "trending_up" else 0.0
        all_features["regime_trending_down"] = 1.0 if regime.regime == "trending_down" else 0.0
        all_features["regime_ranging"] = 1.0 if regime.regime == "ranging" else 0.0
        all_features["regime_volatile"] = 1.0 if regime.regime == "volatile" else 0.0
        all_features["regime_confidence"] = regime.confidence
        all_features["regime_change_probability"] = regime.change_probability
        all_features["volatility_percentile"] = regime.volatility_percentile
        all_features["trend_strength"] = regime.trend_strength

        # Microstructure features
        if trades and quotes:
            microstructure = self.calculate_microstructure_features(symbol, trades, quotes)
            all_features.update(microstructure)

        # Sentiment features
        sentiment = self.calculate_sentiment_features(symbol, df)
        all_features.update(sentiment)

        return all_features
