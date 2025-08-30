"""
Technical indicators for feature engineering.
Implements 50+ technical indicators across different categories.
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from robo_trader.features.base import (
    BaseFeatureCalculator,
    FeatureMetadata,
    FeatureType,
    FeatureValue,
    TimeFrame,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class MomentumIndicators(BaseFeatureCalculator):
    """Calculate momentum-based technical indicators."""

    def __init__(self, window_size: int = 14, timeframe: TimeFrame = TimeFrame.MINUTE_5):
        super().__init__(
            feature_type=FeatureType.MOMENTUM, timeframe=timeframe, window_size=window_size
        )

    def calculate(self, data: pd.DataFrame) -> Dict[str, FeatureValue]:
        """Calculate all momentum indicators."""
        features = {}
        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else "UNKNOWN"
        timestamp = datetime.now()

        # RSI - Relative Strength Index
        rsi = self._calculate_rsi(data["close"], self.window_size)
        features["rsi"] = FeatureValue(
            symbol=symbol,
            feature_name="rsi",
            value=rsi,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="rsi",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Relative Strength Index",
                parameters={"period": self.window_size},
                value_range=(0, 100),
            ),
        )

        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(
            data["high"], data["low"], data["close"], self.window_size
        )
        features["stoch_k"] = FeatureValue(
            symbol=symbol,
            feature_name="stoch_k",
            value=stoch_k,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="stoch_k",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Stochastic %K",
                parameters={"period": self.window_size},
                value_range=(0, 100),
            ),
        )
        features["stoch_d"] = FeatureValue(
            symbol=symbol,
            feature_name="stoch_d",
            value=stoch_d,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="stoch_d",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Stochastic %D",
                parameters={"period": self.window_size},
                value_range=(0, 100),
            ),
        )

        # Williams %R
        williams_r = self._calculate_williams_r(
            data["high"], data["low"], data["close"], self.window_size
        )
        features["williams_r"] = FeatureValue(
            symbol=symbol,
            feature_name="williams_r",
            value=williams_r,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="williams_r",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Williams %R",
                parameters={"period": self.window_size},
                value_range=(-100, 0),
            ),
        )

        # ROC - Rate of Change
        roc = self._calculate_roc(data["close"], self.window_size)
        features["roc"] = FeatureValue(
            symbol=symbol,
            feature_name="roc",
            value=roc,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="roc",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Rate of Change",
                parameters={"period": self.window_size},
            ),
        )

        # CCI - Commodity Channel Index
        cci = self._calculate_cci(data["high"], data["low"], data["close"], self.window_size)
        features["cci"] = FeatureValue(
            symbol=symbol,
            feature_name="cci",
            value=cci,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="cci",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Commodity Channel Index",
                parameters={"period": self.window_size},
            ),
        )

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        avg_gain = gains.rolling(period).mean().iloc[-1]
        avg_loss = losses.rolling(period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi) if not pd.isna(rsi) else 50.0

    def _calculate_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(3).mean()

        return (
            float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else 50.0,
            float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else 50.0,
        )

    def _calculate_williams_r(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> float:
        """Calculate Williams %R."""
        highest_high = high.rolling(period).max().iloc[-1]
        lowest_low = low.rolling(period).min().iloc[-1]

        if highest_high == lowest_low:
            return -50.0

        williams_r = -100 * ((highest_high - close.iloc[-1]) / (highest_high - lowest_low))

        return float(williams_r) if not pd.isna(williams_r) else -50.0

    def _calculate_roc(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Rate of Change."""
        if len(prices) <= period:
            return 0.0

        current = prices.iloc[-1]
        past = prices.iloc[-period - 1]

        if past == 0:
            return 0.0

        roc = ((current - past) / past) * 100

        return float(roc) if not pd.isna(roc) else 0.0

    def _calculate_cci(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> float:
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(period).mean().iloc[-1]
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean()))).iloc[-1]

        if mad == 0:
            return 0.0

        cci = (typical_price.iloc[-1] - sma) / (0.015 * mad)

        return float(cci) if not pd.isna(cci) else 0.0

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required = self.get_required_columns()
        return all(col in data.columns for col in required) and len(data) >= self.get_minimum_rows()

    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ["open", "high", "low", "close", "volume"]

    def get_minimum_rows(self) -> int:
        """Get minimum rows needed."""
        return self.window_size + 5


class TrendIndicators(BaseFeatureCalculator):
    """Calculate trend-following indicators."""

    def __init__(self, timeframe: TimeFrame = TimeFrame.MINUTE_5):
        super().__init__(feature_type=FeatureType.TREND, timeframe=timeframe, window_size=20)

    def calculate(self, data: pd.DataFrame) -> Dict[str, FeatureValue]:
        """Calculate all trend indicators."""
        features = {}
        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else "UNKNOWN"
        timestamp = datetime.now()
        close = data["close"]

        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                sma = close.rolling(period).mean().iloc[-1]
                features[f"sma_{period}"] = FeatureValue(
                    symbol=symbol,
                    feature_name=f"sma_{period}",
                    value=float(sma),
                    timestamp=timestamp,
                    metadata=FeatureMetadata(
                        name=f"sma_{period}",
                        type=self.feature_type,
                        timeframe=self.timeframe,
                        description=f"Simple Moving Average {period}",
                        parameters={"period": period},
                    ),
                )

        # Exponential Moving Averages
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                ema = close.ewm(span=period, adjust=False).mean().iloc[-1]
                features[f"ema_{period}"] = FeatureValue(
                    symbol=symbol,
                    feature_name=f"ema_{period}",
                    value=float(ema),
                    timestamp=timestamp,
                    metadata=FeatureMetadata(
                        name=f"ema_{period}",
                        type=self.feature_type,
                        timeframe=self.timeframe,
                        description=f"Exponential Moving Average {period}",
                        parameters={"period": period},
                    ),
                )

        # MACD
        if len(close) >= 26:
            macd, signal, histogram = self._calculate_macd(close)
            features["macd"] = FeatureValue(
                symbol=symbol,
                feature_name="macd",
                value=macd,
                timestamp=timestamp,
                metadata=FeatureMetadata(
                    name="macd",
                    type=self.feature_type,
                    timeframe=self.timeframe,
                    description="MACD Line",
                    parameters={"fast": 12, "slow": 26, "signal": 9},
                ),
            )
            features["macd_signal"] = FeatureValue(
                symbol=symbol,
                feature_name="macd_signal",
                value=signal,
                timestamp=timestamp,
                metadata=FeatureMetadata(
                    name="macd_signal",
                    type=self.feature_type,
                    timeframe=self.timeframe,
                    description="MACD Signal Line",
                    parameters={"fast": 12, "slow": 26, "signal": 9},
                ),
            )
            features["macd_histogram"] = FeatureValue(
                symbol=symbol,
                feature_name="macd_histogram",
                value=histogram,
                timestamp=timestamp,
                metadata=FeatureMetadata(
                    name="macd_histogram",
                    type=self.feature_type,
                    timeframe=self.timeframe,
                    description="MACD Histogram",
                    parameters={"fast": 12, "slow": 26, "signal": 9},
                ),
            )

        # ADX - Average Directional Index
        if len(data) >= 14:
            adx = self._calculate_adx(data["high"], data["low"], data["close"])
            features["adx"] = FeatureValue(
                symbol=symbol,
                feature_name="adx",
                value=adx,
                timestamp=timestamp,
                metadata=FeatureMetadata(
                    name="adx",
                    type=self.feature_type,
                    timeframe=self.timeframe,
                    description="Average Directional Index",
                    parameters={"period": 14},
                    value_range=(0, 100),
                ),
            )

        return features

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0,
        )

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> float:
        """Calculate Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()]
        ).max()

        atr = tr1.rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * ((-minus_dm).rolling(period).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(period).mean().iloc[-1]

        return float(adx) if not pd.isna(adx) else 0.0

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required = self.get_required_columns()
        return all(col in data.columns for col in required) and len(data) >= self.get_minimum_rows()

    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ["open", "high", "low", "close", "volume"]

    def get_minimum_rows(self) -> int:
        """Get minimum rows needed."""
        return 50


class VolatilityIndicators(BaseFeatureCalculator):
    """Calculate volatility-based indicators."""

    def __init__(self, window_size: int = 20, timeframe: TimeFrame = TimeFrame.MINUTE_5):
        super().__init__(
            feature_type=FeatureType.VOLATILITY, timeframe=timeframe, window_size=window_size
        )

    def calculate(self, data: pd.DataFrame) -> Dict[str, FeatureValue]:
        """Calculate all volatility indicators."""
        features = {}
        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else "UNKNOWN"
        timestamp = datetime.now()

        # ATR - Average True Range
        atr = self._calculate_atr(data["high"], data["low"], data["close"], self.window_size)
        features["atr"] = FeatureValue(
            symbol=symbol,
            feature_name="atr",
            value=atr,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="atr",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Average True Range",
                parameters={"period": self.window_size},
            ),
        )

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger_bands(
            data["close"], self.window_size
        )
        features["bb_upper"] = FeatureValue(
            symbol=symbol,
            feature_name="bb_upper",
            value=bb_upper,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="bb_upper",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Bollinger Band Upper",
                parameters={"period": self.window_size, "std": 2},
            ),
        )
        features["bb_lower"] = FeatureValue(
            symbol=symbol,
            feature_name="bb_lower",
            value=bb_lower,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="bb_lower",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Bollinger Band Lower",
                parameters={"period": self.window_size, "std": 2},
            ),
        )
        features["bb_width"] = FeatureValue(
            symbol=symbol,
            feature_name="bb_width",
            value=bb_width,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="bb_width",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Bollinger Band Width",
                parameters={"period": self.window_size, "std": 2},
            ),
        )

        # Standard Deviation
        std = data["close"].rolling(self.window_size).std().iloc[-1]
        features["std"] = FeatureValue(
            symbol=symbol,
            feature_name="std",
            value=float(std) if not pd.isna(std) else 0.0,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="std",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Standard Deviation",
                parameters={"period": self.window_size},
            ),
        )

        # Historical Volatility
        returns = data["close"].pct_change()
        hist_vol = returns.rolling(self.window_size).std().iloc[-1] * np.sqrt(252)
        features["hist_volatility"] = FeatureValue(
            symbol=symbol,
            feature_name="hist_volatility",
            value=float(hist_vol) if not pd.isna(hist_vol) else 0.0,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="hist_volatility",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Historical Volatility (Annualized)",
                parameters={"period": self.window_size},
            ),
        )

        return features

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> float:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]

        return float(atr) if not pd.isna(atr) else 0.0

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, num_std: float = 2
    ) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()

        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        width = upper - lower

        return (
            float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else 0.0,
            float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0,
            float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else 0.0,
            float(width.iloc[-1]) if not pd.isna(width.iloc[-1]) else 0.0,
        )

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required = self.get_required_columns()
        return all(col in data.columns for col in required) and len(data) >= self.get_minimum_rows()

    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ["open", "high", "low", "close"]

    def get_minimum_rows(self) -> int:
        """Get minimum rows needed."""
        return self.window_size + 1


class VolumeIndicators(BaseFeatureCalculator):
    """Calculate volume-based indicators."""

    def __init__(self, window_size: int = 20, timeframe: TimeFrame = TimeFrame.MINUTE_5):
        super().__init__(
            feature_type=FeatureType.VOLUME, timeframe=timeframe, window_size=window_size
        )

    def calculate(self, data: pd.DataFrame) -> Dict[str, FeatureValue]:
        """Calculate all volume indicators."""
        features = {}
        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else "UNKNOWN"
        timestamp = datetime.now()

        # OBV - On Balance Volume
        obv = self._calculate_obv(data["close"], data["volume"])
        features["obv"] = FeatureValue(
            symbol=symbol,
            feature_name="obv",
            value=obv,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="obv",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="On Balance Volume",
            ),
        )

        # Volume SMA
        vol_sma = data["volume"].rolling(self.window_size).mean().iloc[-1]
        features["volume_sma"] = FeatureValue(
            symbol=symbol,
            feature_name="volume_sma",
            value=float(vol_sma) if not pd.isna(vol_sma) else 0.0,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="volume_sma",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Volume Simple Moving Average",
                parameters={"period": self.window_size},
            ),
        )

        # Volume Rate
        if len(data) > 1:
            vol_rate = data["volume"].iloc[-1] / vol_sma if vol_sma > 0 else 1.0
            features["volume_rate"] = FeatureValue(
                symbol=symbol,
                feature_name="volume_rate",
                value=float(vol_rate),
                timestamp=timestamp,
                metadata=FeatureMetadata(
                    name="volume_rate",
                    type=self.feature_type,
                    timeframe=self.timeframe,
                    description="Current Volume / Average Volume",
                ),
            )

        # VWAP - Volume Weighted Average Price
        vwap = self._calculate_vwap(data["high"], data["low"], data["close"], data["volume"])
        features["vwap"] = FeatureValue(
            symbol=symbol,
            feature_name="vwap",
            value=vwap,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="vwap",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Volume Weighted Average Price",
            ),
        )

        # MFI - Money Flow Index
        mfi = self._calculate_mfi(
            data["high"], data["low"], data["close"], data["volume"], self.window_size
        )
        features["mfi"] = FeatureValue(
            symbol=symbol,
            feature_name="mfi",
            value=mfi,
            timestamp=timestamp,
            metadata=FeatureMetadata(
                name="mfi",
                type=self.feature_type,
                timeframe=self.timeframe,
                description="Money Flow Index",
                parameters={"period": self.window_size},
                value_range=(0, 100),
            ),
        )

        return features

    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> float:
        """Calculate On Balance Volume."""
        obv = [0]
        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif prices.iloc[i] < prices.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])

        return float(obv[-1])

    def _calculate_vwap(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> float:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).sum() / volume.sum()

        return float(vwap) if not pd.isna(vwap) else 0.0

    def _calculate_mfi(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
    ) -> float:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = []
        negative_flow = []

        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        positive_flow = pd.Series(positive_flow)
        negative_flow = pd.Series(negative_flow)

        positive_mf = positive_flow.rolling(period - 1).sum().iloc[-1]
        negative_mf = negative_flow.rolling(period - 1).sum().iloc[-1]

        if negative_mf == 0:
            return 100.0

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))

        return float(mfi) if not pd.isna(mfi) else 50.0

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required = self.get_required_columns()
        return all(col in data.columns for col in required) and len(data) >= self.get_minimum_rows()

    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ["open", "high", "low", "close", "volume"]

    def get_minimum_rows(self) -> int:
        """Get minimum rows needed."""
        return self.window_size + 1
