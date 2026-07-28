"""Backtesting framework for strategy validation and optimization."""

from .engine import BacktestEngine
from .execution_simulator import ExecutionSimulator, MarketImpactModel
from .manifest import (
    BacktestRunManifest,
    DataAssumptions,
    DatasetPartition,
    EnvironmentMetadata,
    ExecutionAssumptions,
    PackageVersion,
    ResultArtifact,
    SeedAssignment,
    SeedPolicy,
)
from .provenance import (
    ContentDigest,
    HashedInput,
    digest_bytes,
    digest_dataframe,
    digest_file,
    digest_file_set,
    digest_json,
)
from .walk_forward_optimization import WalkForwardOptimizer

__all__ = [
    "BacktestEngine",
    "BacktestRunManifest",
    "ContentDigest",
    "DataAssumptions",
    "DatasetPartition",
    "EnvironmentMetadata",
    "ExecutionSimulator",
    "ExecutionAssumptions",
    "HashedInput",
    "MarketImpactModel",
    "PackageVersion",
    "ResultArtifact",
    "SeedAssignment",
    "SeedPolicy",
    "WalkForwardOptimizer",
    "digest_bytes",
    "digest_dataframe",
    "digest_file",
    "digest_file_set",
    "digest_json",
]
