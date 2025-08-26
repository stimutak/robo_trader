"""
Feature store with versioning and persistence.

This module implements:
- Feature versioning and tracking
- Efficient storage and retrieval
- Feature metadata management
- Point-in-time feature retrieval
- Feature importance tracking
"""

import asyncio
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import numpy as np
import pandas as pd

from ..logger import get_logger


@dataclass
class FeatureMetadata:
    """Metadata for a feature set."""
    
    feature_id: str
    symbol: str
    timestamp: datetime
    version: int
    num_features: int
    feature_names: List[str]
    calculation_time_ms: float
    data_quality_score: float  # 0-1 score based on data completeness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'feature_id': self.feature_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version,
            'num_features': self.num_features,
            'feature_names': json.dumps(self.feature_names),
            'calculation_time_ms': self.calculation_time_ms,
            'data_quality_score': self.data_quality_score
        }


@dataclass
class FeatureImportance:
    """Track feature importance from ML models."""
    
    feature_name: str
    importance_score: float
    model_name: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'importance_score': self.importance_score,
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat()
        }


class FeatureStore:
    """Centralized feature store with versioning."""
    
    def __init__(self, db_path: str = "feature_store.db", max_versions: int = 10):
        self.db_path = db_path
        self.max_versions = max_versions
        self.logger = get_logger("features.store")
        
        # In-memory cache for recent features
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_metadata: Dict[str, FeatureMetadata] = {}
        self.cache_size = 100  # Max cached feature sets
        
        # Feature importance tracking
        self.feature_importance: Dict[str, List[FeatureImportance]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            'features_stored': 0,
            'features_retrieved': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_storage_time_ms': [],
            'avg_retrieval_time_ms': []
        }
        
        # Initialize database
        asyncio.create_task(self._init_database())
    
    async def _init_database(self) -> None:
        """Initialize database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")
            
            # Create features table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    feature_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    features TEXT NOT NULL,  -- JSON encoded features
                    metadata TEXT NOT NULL,  -- JSON encoded metadata
                    created_at TEXT NOT NULL,
                    INDEX idx_symbol_timestamp (symbol, timestamp),
                    INDEX idx_version (version)
                )
            """)
            
            # Create feature metadata table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    num_features INTEGER NOT NULL,
                    feature_names TEXT NOT NULL,
                    calculation_time_ms REAL NOT NULL,
                    data_quality_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (feature_id) REFERENCES features(feature_id)
                )
            """)
            
            # Create feature importance table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            await db.commit()
    
    async def store_features(
        self,
        symbol: str,
        features: pd.DataFrame,
        metadata: Optional[FeatureMetadata] = None
    ) -> str:
        """Store features with versioning."""
        start_time = datetime.now()
        
        try:
            # Generate feature ID
            timestamp = datetime.now()
            feature_id = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Get current version
            version = await self._get_next_version(symbol)
            
            # Create metadata if not provided
            if metadata is None:
                metadata = FeatureMetadata(
                    feature_id=feature_id,
                    symbol=symbol,
                    timestamp=timestamp,
                    version=version,
                    num_features=len(features.columns),
                    feature_names=list(features.columns),
                    calculation_time_ms=0,
                    data_quality_score=self._calculate_data_quality(features)
                )
            
            # Store in database
            async with aiosqlite.connect(self.db_path) as db:
                # Store features
                await db.execute("""
                    INSERT INTO features (
                        feature_id, symbol, timestamp, version, features, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_id,
                    symbol,
                    timestamp.isoformat(),
                    version,
                    features.to_json(),
                    json.dumps(metadata.to_dict()),
                    datetime.now().isoformat()
                ))
                
                # Store metadata
                await db.execute("""
                    INSERT INTO feature_metadata (
                        feature_id, symbol, timestamp, version, num_features,
                        feature_names, calculation_time_ms, data_quality_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_id,
                    symbol,
                    timestamp.isoformat(),
                    version,
                    metadata.num_features,
                    json.dumps(metadata.feature_names),
                    metadata.calculation_time_ms,
                    metadata.data_quality_score,
                    datetime.now().isoformat()
                ))
                
                await db.commit()
            
            # Update cache
            cache_key = f"{symbol}_latest"
            self.cache[cache_key] = features
            self.cache_metadata[cache_key] = metadata
            
            # Manage cache size
            if len(self.cache) > self.cache_size:
                oldest_key = min(self.cache_metadata.keys(), 
                               key=lambda k: self.cache_metadata[k].timestamp)
                del self.cache[oldest_key]
                del self.cache_metadata[oldest_key]
            
            # Cleanup old versions
            await self._cleanup_old_versions(symbol)
            
            # Update metrics
            self.metrics['features_stored'] += 1
            storage_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics['avg_storage_time_ms'].append(storage_time)
            if len(self.metrics['avg_storage_time_ms']) > 100:
                self.metrics['avg_storage_time_ms'] = self.metrics['avg_storage_time_ms'][-100:]
            
            self.logger.info(f"Stored features for {symbol} with ID {feature_id}")
            return feature_id
            
        except Exception as e:
            self.logger.error(f"Failed to store features: {e}")
            raise
    
    async def get_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
        version: Optional[int] = None
    ) -> Optional[Tuple[pd.DataFrame, FeatureMetadata]]:
        """Retrieve features from store."""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"{symbol}_latest" if timestamp is None else f"{symbol}_{timestamp}"
            if cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                return self.cache[cache_key], self.cache_metadata[cache_key]
            
            self.metrics['cache_misses'] += 1
            
            # Query database
            async with aiosqlite.connect(self.db_path) as db:
                if timestamp:
                    # Get features at specific timestamp
                    query = """
                        SELECT features, metadata FROM features
                        WHERE symbol = ? AND timestamp <= ?
                        ORDER BY timestamp DESC LIMIT 1
                    """
                    params = (symbol, timestamp.isoformat())
                elif version:
                    # Get specific version
                    query = """
                        SELECT features, metadata FROM features
                        WHERE symbol = ? AND version = ?
                        ORDER BY timestamp DESC LIMIT 1
                    """
                    params = (symbol, version)
                else:
                    # Get latest features
                    query = """
                        SELECT features, metadata FROM features
                        WHERE symbol = ?
                        ORDER BY timestamp DESC LIMIT 1
                    """
                    params = (symbol,)
                
                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        features_json, metadata_json = row
                        features = pd.read_json(features_json)
                        metadata_dict = json.loads(metadata_json)
                        metadata = FeatureMetadata(
                            feature_id=metadata_dict['feature_id'],
                            symbol=metadata_dict['symbol'],
                            timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
                            version=metadata_dict['version'],
                            num_features=metadata_dict['num_features'],
                            feature_names=json.loads(metadata_dict['feature_names']),
                            calculation_time_ms=metadata_dict['calculation_time_ms'],
                            data_quality_score=metadata_dict['data_quality_score']
                        )
                        
                        # Update cache
                        self.cache[cache_key] = features
                        self.cache_metadata[cache_key] = metadata
                        
                        # Update metrics
                        self.metrics['features_retrieved'] += 1
                        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
                        self.metrics['avg_retrieval_time_ms'].append(retrieval_time)
                        if len(self.metrics['avg_retrieval_time_ms']) > 100:
                            self.metrics['avg_retrieval_time_ms'] = \
                                self.metrics['avg_retrieval_time_ms'][-100:]
                        
                        return features, metadata
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve features: {e}")
            return None
    
    async def get_feature_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get historical features for backtesting."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = """
                    SELECT timestamp, features FROM features
                    WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                """
                
                features_list = []
                async with db.execute(query, (
                    symbol,
                    start_date.isoformat(),
                    end_date.isoformat()
                )) as cursor:
                    async for row in cursor:
                        timestamp_str, features_json = row
                        features = pd.read_json(features_json)
                        features['timestamp'] = datetime.fromisoformat(timestamp_str)
                        
                        if feature_names:
                            # Filter specific features
                            available_features = [f for f in feature_names if f in features.columns]
                            features = features[['timestamp'] + available_features]
                        
                        features_list.append(features)
                
                if features_list:
                    return pd.concat(features_list, ignore_index=True)
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"Failed to get feature history: {e}")
            return pd.DataFrame()
    
    async def update_feature_importance(
        self,
        importance_scores: Dict[str, float],
        model_name: str
    ) -> None:
        """Update feature importance scores from ML models."""
        try:
            timestamp = datetime.now()
            
            async with aiosqlite.connect(self.db_path) as db:
                for feature_name, score in importance_scores.items():
                    importance = FeatureImportance(
                        feature_name=feature_name,
                        importance_score=score,
                        model_name=model_name,
                        timestamp=timestamp
                    )
                    
                    # Store in database
                    await db.execute("""
                        INSERT INTO feature_importance (
                            feature_name, importance_score, model_name, timestamp, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        feature_name,
                        score,
                        model_name,
                        timestamp.isoformat(),
                        datetime.now().isoformat()
                    ))
                    
                    # Update in-memory tracking
                    self.feature_importance[feature_name].append(importance)
                
                await db.commit()
            
            self.logger.info(f"Updated feature importance for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to update feature importance: {e}")
    
    async def get_top_features(
        self,
        n: int = 50,
        model_name: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if model_name:
                    query = """
                        SELECT feature_name, AVG(importance_score) as avg_score
                        FROM feature_importance
                        WHERE model_name = ?
                        GROUP BY feature_name
                        ORDER BY avg_score DESC
                        LIMIT ?
                    """
                    params = (model_name, n)
                else:
                    query = """
                        SELECT feature_name, AVG(importance_score) as avg_score
                        FROM feature_importance
                        GROUP BY feature_name
                        ORDER BY avg_score DESC
                        LIMIT ?
                    """
                    params = (n,)
                
                async with db.execute(query, params) as cursor:
                    return await cursor.fetchall()
                    
        except Exception as e:
            self.logger.error(f"Failed to get top features: {e}")
            return []
    
    async def _get_next_version(self, symbol: str) -> int:
        """Get next version number for symbol."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT MAX(version) FROM features WHERE symbol = ?",
                    (symbol,)
                ) as cursor:
                    row = await cursor.fetchone()
                    return (row[0] or 0) + 1
                    
        except Exception:
            return 1
    
    async def _cleanup_old_versions(self, symbol: str) -> None:
        """Remove old versions exceeding max_versions."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get version count
                async with db.execute(
                    "SELECT COUNT(DISTINCT version) FROM features WHERE symbol = ?",
                    (symbol,)
                ) as cursor:
                    row = await cursor.fetchone()
                    version_count = row[0]
                
                # Delete old versions if necessary
                if version_count > self.max_versions:
                    versions_to_delete = version_count - self.max_versions
                    await db.execute("""
                        DELETE FROM features
                        WHERE symbol = ? AND version IN (
                            SELECT version FROM features
                            WHERE symbol = ?
                            ORDER BY version ASC
                            LIMIT ?
                        )
                    """, (symbol, symbol, versions_to_delete))
                    
                    await db.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old versions: {e}")
    
    def _calculate_data_quality(self, features: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness."""
        try:
            # Calculate percentage of non-null values
            total_values = features.size
            non_null_values = features.count().sum()
            
            if total_values == 0:
                return 0.0
            
            completeness = non_null_values / total_values
            
            # Check for infinite values
            inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
            inf_penalty = inf_count / total_values if total_values > 0 else 0
            
            # Final score
            quality_score = max(0.0, completeness - inf_penalty)
            
            return quality_score
            
        except Exception:
            return 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get store performance metrics."""
        return {
            'features_stored': self.metrics['features_stored'],
            'features_retrieved': self.metrics['features_retrieved'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': self.metrics['cache_hits'] / 
                             max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
            'avg_storage_time_ms': np.mean(self.metrics['avg_storage_time_ms']) 
                                  if self.metrics['avg_storage_time_ms'] else 0,
            'avg_retrieval_time_ms': np.mean(self.metrics['avg_retrieval_time_ms'])
                                    if self.metrics['avg_retrieval_time_ms'] else 0,
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size
        }