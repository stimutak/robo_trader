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
from .walk_forward import (
    BoundWalkForwardEvidence,
    EvaluationOutcome,
    HoldoutEvidence,
    OptimizationOutcome,
    WalkForwardPlan,
    WalkForwardResult,
    WalkForwardValidationError,
    WalkForwardValidator,
    WindowBoundary,
    WindowEvidence,
)
from .walk_forward_optimization import WalkForwardOptimizer

__all__ = [
    "BacktestEngine",
    "BacktestRunManifest",
    "BoundWalkForwardEvidence",
    "ContentDigest",
    "DataAssumptions",
    "DatasetPartition",
    "EnvironmentMetadata",
    "EvaluationOutcome",
    "ExecutionSimulator",
    "ExecutionAssumptions",
    "HashedInput",
    "HoldoutEvidence",
    "MarketImpactModel",
    "OptimizationOutcome",
    "PackageVersion",
    "ResultArtifact",
    "SeedAssignment",
    "SeedPolicy",
    "WalkForwardOptimizer",
    "WalkForwardPlan",
    "WalkForwardResult",
    "WalkForwardValidationError",
    "WalkForwardValidator",
    "WindowBoundary",
    "WindowEvidence",
    "digest_bytes",
    "digest_dataframe",
    "digest_file",
    "digest_file_set",
    "digest_json",
]
