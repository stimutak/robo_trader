"""
Base classes for feature engineering pipeline.
Provides abstract interfaces for feature calculators and feature sets.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class FeatureType(Enum):
    """Types of features for categorization."""

    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    MICROSTRUCTURE = "microstructure"
    REGIME = "regime"
    PATTERN = "pattern"
    COMPOSITE = "composite"


class TimeFrame(Enum):
    """Supported timeframes for feature calculation."""

    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


@dataclass
class FeatureMetadata:
    """Metadata for a calculated feature."""

    name: str
    type: FeatureType
    timeframe: TimeFrame
    version: str = "1.0.0"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    is_normalized: bool = False
    value_range: tuple = (-np.inf, np.inf)
    computation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FeatureValue:
    """Container for a single feature value with metadata."""

    symbol: str
    feature_name: str
    value: Union[float, np.ndarray]
    timestamp: datetime
    metadata: FeatureMetadata
    is_valid: bool = True
    error_msg: Optional[str] = None


class BaseFeatureCalculator(ABC):
    """
    Abstract base class for all feature calculators.
    Provides common interface and utilities for feature computation.
    """

    def __init__(
        self,
        feature_type: FeatureType,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
        window_size: int = 20,
        version: str = "1.0.0",
    ):
        self.feature_type = feature_type
        self.timeframe = timeframe
        self.window_size = window_size
        self.version = version
        self._cache: Dict[str, Any] = {}
        self._last_calculation: Optional[datetime] = None

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Dict[str, FeatureValue]:
        """
        Calculate features from input data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary of feature_name -> FeatureValue
        """
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns and sufficient rows.

        Args:
            data: Input DataFrame

        Returns:
            True if data is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get list of required DataFrame columns."""
        pass

    @abstractmethod
    def get_minimum_rows(self) -> int:
        """Get minimum number of rows needed for calculation."""
        pass

    async def calculate_async(self, data: pd.DataFrame) -> Dict[str, FeatureValue]:
        """Async wrapper for feature calculation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.calculate, data)

    def clear_cache(self):
        """Clear internal cache."""
        self._cache.clear()

    def get_feature_names(self) -> List[str]:
        """Get list of feature names this calculator produces."""
        return []


class CompositeFeatureCalculator(BaseFeatureCalculator):
    """
    Combines multiple feature calculators into a single calculator.
    Useful for creating feature sets.
    """

    def __init__(self, calculators: List[BaseFeatureCalculator]):
        super().__init__(feature_type=FeatureType.COMPOSITE, timeframe=TimeFrame.MINUTE_5)
        self.calculators = calculators

    def calculate(self, data: pd.DataFrame) -> Dict[str, FeatureValue]:
        """Calculate features from all sub-calculators."""
        all_features = {}

        for calc in self.calculators:
            if calc.validate_data(data):
                features = calc.calculate(data)
                all_features.update(features)

        return all_features

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Check if data is valid for at least one calculator."""
        return any(calc.validate_data(data) for calc in self.calculators)

    def get_required_columns(self) -> List[str]:
        """Get union of all required columns."""
        columns = set()
        for calc in self.calculators:
            columns.update(calc.get_required_columns())
        return list(columns)

    def get_minimum_rows(self) -> int:
        """Get maximum of all minimum row requirements."""
        return max(calc.get_minimum_rows() for calc in self.calculators)

    def add_calculator(self, calculator: BaseFeatureCalculator):
        """Add a new calculator to the composite."""
        self.calculators.append(calculator)

    def remove_calculator(self, calculator: BaseFeatureCalculator):
        """Remove a calculator from the composite."""
        if calculator in self.calculators:
            self.calculators.remove(calculator)


class FeatureSet:
    """
    Container for a set of related features.
    Manages feature computation and storage.
    """

    def __init__(self, name: str, calculators: List[BaseFeatureCalculator], description: str = ""):
        self.name = name
        self.description = description
        self.calculators = calculators
        self._feature_values: Dict[str, Dict[str, FeatureValue]] = {}
        self._last_update: Optional[datetime] = None

    async def compute_features(self, symbol: str, data: pd.DataFrame) -> Dict[str, FeatureValue]:
        """
        Compute all features for a symbol.

        Args:
            symbol: Stock symbol
            data: OHLCV data

        Returns:
            Dictionary of computed features
        """
        features = {}

        for calc in self.calculators:
            if calc.validate_data(data):
                calc_features = await calc.calculate_async(data)

                # Add symbol to each feature
                for fname, fvalue in calc_features.items():
                    fvalue.symbol = symbol
                    features[fname] = fvalue

        # Store in cache
        self._feature_values[symbol] = features
        self._last_update = datetime.now()

        return features

    def get_features(self, symbol: str) -> Optional[Dict[str, FeatureValue]]:
        """Get cached features for a symbol."""
        return self._feature_values.get(symbol)

    def get_feature_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get feature matrix for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            DataFrame with features as columns and symbols as rows
        """
        data = []

        for symbol in symbols:
            if symbol in self._feature_values:
                features = self._feature_values[symbol]
                row = {"symbol": symbol}

                for fname, fvalue in features.items():
                    if isinstance(fvalue.value, (int, float)):
                        row[fname] = fvalue.value

                data.append(row)

        if data:
            df = pd.DataFrame(data)
            df.set_index("symbol", inplace=True)
            return df

        return pd.DataFrame()

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached features."""
        if symbol:
            self._feature_values.pop(symbol, None)
        else:
            self._feature_values.clear()
