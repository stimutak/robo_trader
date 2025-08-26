"""
Technical indicators for feature engineering.

This module implements:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Volume indicators (VWAP, OBV)
- Moving averages (SMA, EMA, WMA)
- Additional indicators for comprehensive analysis
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR
    atr_period: int = 14
    
    # Moving Averages
    sma_periods: list = None
    ema_periods: list = None
    
    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    
    # ADX
    adx_period: int = 14
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26]


class TechnicalIndicators:
    """Calculator for technical indicators."""
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
    
    # Price-based indicators
    
    def sma(self, prices: pd.Series, period: int) -> Optional[float]:
        """Simple Moving Average."""
        try:
            if len(prices) < period:
                return None
            return prices.rolling(window=period).mean().iloc[-1]
        except Exception:
            return None
    
    def ema(self, prices: pd.Series, period: int) -> Optional[float]:
        """Exponential Moving Average."""
        try:
            if len(prices) < period:
                return None
            return prices.ewm(span=period, adjust=False).mean().iloc[-1]
        except Exception:
            return None
    
    def wma(self, prices: pd.Series, period: int) -> Optional[float]:
        """Weighted Moving Average."""
        try:
            if len(prices) < period:
                return None
            weights = np.arange(1, period + 1)
            return np.sum(prices.tail(period).values * weights) / np.sum(weights)
        except Exception:
            return None
    
    def rsi(self, prices: pd.Series, period: Optional[int] = None) -> Optional[float]:
        """Relative Strength Index."""
        try:
            period = period or self.config.rsi_period
            if len(prices) < period + 1:
                return None
            
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
            
        except Exception:
            return None
    
    def macd(self, prices: pd.Series, 
             fast: Optional[int] = None,
             slow: Optional[int] = None,
             signal: Optional[int] = None) -> Optional[Dict[str, float]]:
        """MACD (Moving Average Convergence Divergence)."""
        try:
            fast = fast or self.config.macd_fast
            slow = slow or self.config.macd_slow
            signal = signal or self.config.macd_signal
            
            if len(prices) < slow + signal:
                return None
            
            # Calculate MACD line
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.iloc[-1],
                'signal': signal_line.iloc[-1],
                'histogram': histogram.iloc[-1]
            }
            
        except Exception:
            return None
    
    def bollinger_bands(self, prices: pd.Series,
                       period: Optional[int] = None,
                       std_dev: Optional[float] = None) -> Optional[Dict[str, float]]:
        """Bollinger Bands."""
        try:
            period = period or self.config.bb_period
            std_dev = std_dev or self.config.bb_std
            
            if len(prices) < period:
                return None
            
            # Calculate middle band (SMA)
            middle = prices.rolling(window=period).mean()
            
            # Calculate standard deviation
            std = prices.rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            # Calculate bandwidth
            bandwidth = (upper - lower) / middle
            
            return {
                'upper': upper.iloc[-1],
                'middle': middle.iloc[-1],
                'lower': lower.iloc[-1],
                'bandwidth': bandwidth.iloc[-1] * 100  # As percentage
            }
            
        except Exception:
            return None
    
    def atr(self, df: pd.DataFrame, period: Optional[int] = None) -> Optional[float]:
        """Average True Range."""
        try:
            period = period or self.config.atr_period
            
            if len(df) < period + 1:
                return None
            
            # Calculate true range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=period).mean()
            
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None
            
        except Exception:
            return None
    
    def stochastic(self, df: pd.DataFrame,
                  k_period: Optional[int] = None,
                  d_period: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Stochastic Oscillator."""
        try:
            k_period = k_period or self.config.stoch_k_period
            d_period = d_period or self.config.stoch_d_period
            
            if len(df) < k_period + d_period:
                return None
            
            # Calculate %K
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
            
            # Calculate %D (SMA of %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k': k_percent.iloc[-1],
                'd': d_percent.iloc[-1]
            }
            
        except Exception:
            return None
    
    # Volume-based indicators
    
    def obv(self, df: pd.DataFrame) -> Optional[float]:
        """On-Balance Volume."""
        try:
            if len(df) < 2:
                return None
            
            # Calculate OBV
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df['volume'].iloc[0]
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv.iloc[-1]
            
        except Exception:
            return None
    
    def vwap(self, df: pd.DataFrame) -> Optional[float]:
        """Volume Weighted Average Price."""
        try:
            if len(df) < 1 or 'volume' not in df.columns:
                return None
            
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            return vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else None
            
        except Exception:
            return None
    
    def mfi(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Money Flow Index."""
        try:
            if len(df) < period + 1:
                return None
            
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate raw money flow
            raw_money_flow = typical_price * df['volume']
            
            # Determine positive and negative money flow
            money_flow = pd.DataFrame({
                'typical_price': typical_price,
                'raw_money_flow': raw_money_flow
            })
            
            money_flow['price_change'] = money_flow['typical_price'].diff()
            money_flow['positive_flow'] = money_flow['raw_money_flow'].where(money_flow['price_change'] > 0, 0)
            money_flow['negative_flow'] = money_flow['raw_money_flow'].where(money_flow['price_change'] < 0, 0)
            
            # Calculate money ratio
            positive_flow = money_flow['positive_flow'].rolling(window=period).sum()
            negative_flow = money_flow['negative_flow'].rolling(window=period).sum()
            
            money_ratio = positive_flow / (negative_flow + 1e-10)
            
            # Calculate MFI
            mfi = 100 - (100 / (1 + money_ratio))
            
            return mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else None
            
        except Exception:
            return None
    
    # Volatility indicators
    
    def historical_volatility(self, prices: pd.Series, period: int = 20) -> Optional[float]:
        """Historical Volatility (annualized)."""
        try:
            if len(prices) < period:
                return None
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.tail(period).std() * np.sqrt(252)  # Annualized
            
            return volatility if not pd.isna(volatility) else None
            
        except Exception:
            return None
    
    def keltner_channels(self, df: pd.DataFrame,
                        period: int = 20,
                        multiplier: float = 2.0) -> Optional[Dict[str, float]]:
        """Keltner Channels."""
        try:
            if len(df) < period:
                return None
            
            # Calculate middle line (EMA)
            middle = df['close'].ewm(span=period, adjust=False).mean()
            
            # Calculate ATR
            atr_value = self.atr(df, period)
            if atr_value is None:
                return None
            
            # Calculate channels
            upper = middle.iloc[-1] + (multiplier * atr_value)
            lower = middle.iloc[-1] - (multiplier * atr_value)
            
            return {
                'upper': upper,
                'middle': middle.iloc[-1],
                'lower': lower
            }
            
        except Exception:
            return None
    
    # Trend indicators
    
    def adx(self, df: pd.DataFrame, period: Optional[int] = None) -> Optional[float]:
        """Average Directional Index."""
        try:
            period = period or self.config.adx_period
            
            if len(df) < period * 2:
                return None
            
            # Calculate true range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Calculate directional movement
            up_move = df['high'] - df['high'].shift()
            down_move = df['low'].shift() - df['low']
            
            pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
            neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
            
            # Calculate directional indicators
            pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
            neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
            
            # Calculate DX and ADX
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else None
            
        except Exception:
            return None
    
    def cci(self, df: pd.DataFrame, period: int = 20) -> Optional[float]:
        """Commodity Channel Index."""
        try:
            if len(df) < period:
                return None
            
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate SMA of typical price
            sma = typical_price.rolling(window=period).mean()
            
            # Calculate mean deviation
            mean_dev = typical_price.rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            
            # Calculate CCI
            cci = (typical_price - sma) / (0.015 * mean_dev + 1e-10)
            
            return cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else None
            
        except Exception:
            return None
    
    def williams_r(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Williams %R."""
        try:
            if len(df) < period:
                return None
            
            # Calculate highest high and lowest low
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            # Calculate Williams %R
            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low + 1e-10))
            
            return williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else None
            
        except Exception:
            return None
    
    def roc(self, prices: pd.Series, period: int = 12) -> Optional[float]:
        """Rate of Change."""
        try:
            if len(prices) < period + 1:
                return None
            
            # Calculate ROC
            roc = ((prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1]) * 100
            
            return roc if not pd.isna(roc) else None
            
        except Exception:
            return None
    
    def momentum(self, prices: pd.Series, period: int = 10) -> Optional[float]:
        """Momentum indicator."""
        try:
            if len(prices) < period + 1:
                return None
            
            # Calculate momentum
            momentum = prices.iloc[-1] - prices.iloc[-period-1]
            
            return momentum if not pd.isna(momentum) else None
            
        except Exception:
            return None
    
    # Pattern detection helpers
    
    def pivot_points(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculate pivot points and support/resistance levels."""
        try:
            if len(df) < 1:
                return None
            
            # Use previous day's data
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            
            # Calculate pivot point
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                's1': s1,
                's2': s2,
                's3': s3
            }
            
        except Exception:
            return None
    
    def fibonacci_retracement(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        diff = high - low
        
        return {
            '0': high,
            '23.6': high - 0.236 * diff,
            '38.2': high - 0.382 * diff,
            '50': high - 0.5 * diff,
            '61.8': high - 0.618 * diff,
            '100': low
        }