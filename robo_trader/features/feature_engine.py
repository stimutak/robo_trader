"""
Main feature engineering engine.
Orchestrates feature calculation, storage, and retrieval.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from robo_trader.database_async import AsyncTradingDatabase as AsyncDatabase
from robo_trader.features.base import (
    BaseFeatureCalculator,
    CompositeFeatureCalculator,
    FeatureSet,
    FeatureValue,
    TimeFrame,
)
from robo_trader.features.technical_indicators import (
    MomentumIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)
from robo_trader.monitoring.performance import PerformanceMonitor

logger = structlog.get_logger(__name__)


class FeatureEngine:
    """
    Main engine for feature calculation and management.
    Handles real-time feature computation and storage.
    """

    def __init__(
        self,
        db: AsyncDatabase,
        performance_monitor: Optional[PerformanceMonitor] = None,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
    ):
        self.db = db
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.timeframe = timeframe

        # Initialize feature sets
        self.feature_sets: Dict[str, FeatureSet] = {}
        self._initialize_default_features()

        # Cache for recent features
        self._feature_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_clear = datetime.now()

        logger.info(
            "FeatureEngine initialized",
            feature_sets=list(self.feature_sets.keys()),
            timeframe=timeframe.value,
        )

    def _initialize_default_features(self):
        """Initialize default feature calculators."""
        # Momentum features
        momentum_calc = MomentumIndicators(window_size=14, timeframe=self.timeframe)

        # Trend features
        trend_calc = TrendIndicators(timeframe=self.timeframe)

        # Volatility features
        volatility_calc = VolatilityIndicators(window_size=20, timeframe=self.timeframe)

        # Volume features
        volume_calc = VolumeIndicators(window_size=20, timeframe=self.timeframe)

        # Create composite calculator
        all_indicators = CompositeFeatureCalculator(
            [momentum_calc, trend_calc, volatility_calc, volume_calc]
        )

        # Create feature sets
        self.feature_sets["technical"] = FeatureSet(
            name="technical", calculators=[all_indicators], description="All technical indicators"
        )

        self.feature_sets["momentum"] = FeatureSet(
            name="momentum", calculators=[momentum_calc], description="Momentum-based indicators"
        )

        self.feature_sets["trend"] = FeatureSet(
            name="trend", calculators=[trend_calc], description="Trend-following indicators"
        )

        self.feature_sets["volatility"] = FeatureSet(
            name="volatility", calculators=[volatility_calc], description="Volatility indicators"
        )

        self.feature_sets["volume"] = FeatureSet(
            name="volume", calculators=[volume_calc], description="Volume-based indicators"
        )

    async def calculate_features(
        self, symbol: str, data: pd.DataFrame, feature_set_name: str = "technical"
    ) -> Dict[str, FeatureValue]:
        """
        Calculate features for a symbol using specified feature set.

        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            feature_set_name: Name of feature set to use

        Returns:
            Dictionary of calculated features
        """
        with self.performance_monitor.timer_context("feature_calculation"):
            # Check cache first
            cache_key = f"{symbol}_{feature_set_name}"
            if self._is_cache_valid(cache_key):
                logger.debug("Using cached features", symbol=symbol, feature_set=feature_set_name)
                return self._feature_cache[cache_key]["features"]

            # Get feature set
            feature_set = self.feature_sets.get(feature_set_name)
            if not feature_set:
                logger.error("Feature set not found", feature_set=feature_set_name)
                return {}

            # Add symbol to data if not present
            if "symbol" not in data.columns:
                data["symbol"] = symbol

            # Calculate features
            try:
                features = await feature_set.compute_features(symbol, data)

                # Update cache
                self._feature_cache[cache_key] = {"features": features, "timestamp": datetime.now()}

                # Store in database
                await self._store_features(symbol, features)

                self.performance_monitor.increment_counter("features_calculated", len(features))

                logger.info(
                    "Features calculated",
                    symbol=symbol,
                    feature_set=feature_set_name,
                    num_features=len(features),
                )

                return features

            except Exception as e:
                logger.error(
                    "Error calculating features",
                    symbol=symbol,
                    feature_set=feature_set_name,
                    error=str(e),
                )
                self.performance_monitor.increment_counter("feature_errors")
                return {}

    async def calculate_features_batch(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        feature_set_name: str = "technical",
    ) -> Dict[str, Dict[str, FeatureValue]]:
        """
        Calculate features for multiple symbols in parallel.

        Args:
            symbols: List of symbols
            data_dict: Dictionary of symbol -> DataFrame
            feature_set_name: Name of feature set to use

        Returns:
            Dictionary of symbol -> features
        """
        tasks = []
        for symbol in symbols:
            if symbol in data_dict:
                task = self.calculate_features(symbol, data_dict[symbol], feature_set_name)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_features = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error("Error in batch feature calculation", symbol=symbol, error=str(result))
                all_features[symbol] = {}
            else:
                all_features[symbol] = result

        return all_features

    async def _store_features(self, symbol: str, features: Dict[str, FeatureValue]):
        """Store calculated features in database."""
        if not features:
            return

        # Prepare data for storage
        timestamp = datetime.now()

        # Create features table if not exists
        await self._ensure_features_table()

        # Prepare batch insert data
        rows = []
        for fname, fvalue in features.items():
            if isinstance(fvalue.value, (int, float)):
                rows.append(
                    (
                        symbol,
                        fname,
                        float(fvalue.value),
                        fvalue.metadata.type.value,
                        fvalue.metadata.timeframe.value,
                        json.dumps(fvalue.metadata.parameters),
                        fvalue.metadata.version,
                        timestamp,
                    )
                )

        if rows:
            # Insert features
            query = """
                INSERT OR REPLACE INTO features 
                (symbol, feature_name, value, feature_type, timeframe, 
                 parameters, version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            await self.db.execute_many(query, rows)

            logger.debug("Features stored", symbol=symbol, count=len(rows))

    async def _ensure_features_table(self):
        """Ensure features table exists in database."""
        query = """
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                value REAL NOT NULL,
                feature_type TEXT,
                timeframe TEXT,
                parameters TEXT,
                version TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, feature_name, timestamp)
            )
        """
        await self.db.execute(query)

        # Create indices for performance
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp 
            ON features(symbol, timestamp DESC)
        """
        )
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_features_feature_name 
            ON features(feature_name, timestamp DESC)
        """
        )

    async def get_latest_features(
        self, symbol: str, feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get latest features for a symbol from database.

        Args:
            symbol: Stock symbol
            feature_names: Optional list of specific features to retrieve

        Returns:
            Dictionary of feature_name -> value
        """
        query = """
            SELECT DISTINCT feature_name, value
            FROM features
            WHERE symbol = ?
              AND timestamp = (
                  SELECT MAX(timestamp) 
                  FROM features f2 
                  WHERE f2.symbol = ? AND f2.feature_name = features.feature_name
              )
        """

        params = [symbol, symbol]

        if feature_names:
            placeholders = ",".join(["?" for _ in feature_names])
            query += f" AND feature_name IN ({placeholders})"
            params.extend(feature_names)

        rows = await self.db.fetch_all(query, params)

        return {row[0]: row[1] for row in rows}

    async def get_feature_history(
        self,
        symbol: str,
        feature_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get historical values for a specific feature.

        Args:
            symbol: Stock symbol
            feature_name: Name of feature
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with timestamp and value columns
        """
        query = """
            SELECT timestamp, value
            FROM features
            WHERE symbol = ? AND feature_name = ?
        """
        params = [symbol, feature_name]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        rows = await self.db.fetch_all(query, params)

        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "value"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            return df

        return pd.DataFrame()

    async def get_feature_correlation_matrix(
        self, symbols: List[str], feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for features across symbols.

        Args:
            symbols: List of symbols
            feature_names: Optional list of specific features

        Returns:
            Correlation matrix as DataFrame
        """
        # Get feature matrix
        feature_data = {}

        for symbol in symbols:
            features = await self.get_latest_features(symbol, feature_names)
            if features:
                feature_data[symbol] = features

        if not feature_data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(feature_data).T

        # Calculate correlation
        return df.corr()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._feature_cache:
            return False

        cache_entry = self._feature_cache[cache_key]
        age = datetime.now() - cache_entry["timestamp"]

        return age < self._cache_ttl

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear feature cache."""
        if symbol:
            # Clear cache for specific symbol
            keys_to_remove = [k for k in self._feature_cache.keys() if k.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self._feature_cache[key]
        else:
            # Clear all cache
            self._feature_cache.clear()

        self._last_cache_clear = datetime.now()

    async def cleanup_old_features(self, days_to_keep: int = 30):
        """Remove old features from database."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        query = "DELETE FROM features WHERE timestamp < ?"
        rows_deleted = await self.db.execute(query, [cutoff_date])

        logger.info("Cleaned up old features", rows_deleted=rows_deleted, days_kept=days_to_keep)

        return rows_deleted

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about available features."""
        info = {
            "feature_sets": {},
            "total_calculators": 0,
            "cache_size": len(self._feature_cache),
            "last_cache_clear": (
                self._last_cache_clear.isoformat() if self._last_cache_clear else None
            ),
        }

        for name, feature_set in self.feature_sets.items():
            info["feature_sets"][name] = {
                "description": feature_set.description,
                "num_calculators": len(feature_set.calculators),
                "last_update": (
                    feature_set._last_update.isoformat() if feature_set._last_update else None
                ),
            }
            info["total_calculators"] += len(feature_set.calculators)

        return info
