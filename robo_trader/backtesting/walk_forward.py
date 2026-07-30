"""Leakage-resistant deterministic walk-forward validation.

This module operates only on caller-supplied offline DataFrames and callbacks.
It has no runtime, broker, database, credential, or order authority.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .manifest import BacktestRunManifest, SeedPolicy
from .provenance import ContentDigest, HashedInput, digest_bytes, digest_dataframe, digest_json

_SCHEMA_VERSION = "robotrader-walk-forward-v1"
_SELECTION_METRICS = frozenset({"mean_return", "cumulative_return", "return_to_volatility"})


class WalkForwardValidationError(RuntimeError):
    """A fail-closed validation failure."""


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _seed(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**64 - 1:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _timestamp(value: str, name: str) -> pd.Timestamp:
    _required_text(value, name)
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a finite timezone-aware timestamp")
    return timestamp


def _metrics(returns: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(tuple(returns), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise WalkForwardValidationError("evaluation must contain finite out-of-sample returns")
    mean = float(np.mean(values))
    deviation = float(np.std(values)) if len(values) > 1 else 0.0
    cumulative = float(np.prod(1.0 + values) - 1.0)
    # The evaluator contract does not declare a sampling frequency.  Report a
    # truthful unannualized return-to-volatility ratio rather than scaling by
    # sqrt(observation_count), which is a t-statistic and not a Sharpe ratio.
    return_to_volatility = mean / deviation if deviation > 0 else 0.0
    metrics = {
        "observation_count": float(len(values)),
        "mean_return": mean,
        "cumulative_return": cumulative,
        "volatility": deviation,
        "return_to_volatility": return_to_volatility,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise WalkForwardValidationError("evaluation produced non-finite aggregate metrics")
    return metrics


@dataclass(frozen=True)
class WalkForwardPlan:
    """Fixed-count rolling window and untouched-holdout policy."""

    split_id: str
    holdout_partition_id: str
    train_size: int
    validation_size: int
    test_size: int
    holdout_size: int
    step_size: int
    purge_size: int = 0
    embargo_size: int = 0
    root_seed: int = 0
    seed_generator_id: str = "numpy-pcg64"
    selection_metric: str = "mean_return"

    def __post_init__(self) -> None:
        _required_text(self.split_id, "split_id")
        _required_text(self.holdout_partition_id, "holdout_partition_id")
        for value, name in (
            (self.train_size, "train_size"),
            (self.validation_size, "validation_size"),
            (self.test_size, "test_size"),
            (self.holdout_size, "holdout_size"),
            (self.step_size, "step_size"),
        ):
            _positive_integer(value, name)
        _positive_integer(self.purge_size, "purge_size", allow_zero=True)
        _positive_integer(self.embargo_size, "embargo_size", allow_zero=True)
        if self.step_size < self.test_size + self.embargo_size:
            raise ValueError(
                "step_size must be at least test_size + embargo_size to prevent overlap"
            )
        if (
            isinstance(self.root_seed, bool)
            or not isinstance(self.root_seed, int)
            or not 0 <= self.root_seed <= 2**64 - 1
        ):
            raise ValueError("root_seed must be an unsigned 64-bit integer")
        _required_text(self.seed_generator_id, "seed_generator_id")
        if self.selection_metric not in _SELECTION_METRICS:
            raise ValueError(f"unsupported selection metric {self.selection_metric!r}")

    @property
    def minimum_development_size(self) -> int:
        return (
            self.train_size
            + self.purge_size
            + self.validation_size
            + self.purge_size
            + self.test_size
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_id": self.split_id,
            "holdout_partition_id": self.holdout_partition_id,
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "holdout_size": self.holdout_size,
            "step_size": self.step_size,
            "purge_size": self.purge_size,
            "embargo_size": self.embargo_size,
            "root_seed": self.root_seed,
            "seed_generator_id": self.seed_generator_id,
            "selection_metric": self.selection_metric,
        }


@dataclass(frozen=True)
class WindowBoundary:
    window_id: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def __post_init__(self) -> None:
        _required_text(self.window_id, "window_id")
        boundaries = (
            self.train_start,
            self.train_end,
            self.validation_start,
            self.validation_end,
            self.test_start,
            self.test_end,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in boundaries
        ):
            raise ValueError("window boundaries must be non-negative integers")
        if not (
            self.train_start
            < self.train_end
            <= self.validation_start
            < self.validation_end
            <= self.test_start
            < self.test_end
        ):
            raise ValueError("window boundaries must be strictly time ordered")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


@dataclass(frozen=True)
class OptimizationOutcome:
    """Versioned model/config output produced from train and validation only."""

    model_id: str
    model_version: str
    model_bytes: bytes
    selected_config: HashedInput
    validation_score: float

    def __post_init__(self) -> None:
        _required_text(self.model_id, "model_id")
        _required_text(self.model_version, "model_version")
        if not isinstance(self.model_bytes, bytes) or not self.model_bytes:
            raise ValueError("model_bytes must be non-empty bytes")
        if (
            not isinstance(self.selected_config, HashedInput)
            or self.selected_config.kind != "config"
        ):
            raise ValueError("selected_config must be a hashed config input")
        object.__setattr__(
            self, "validation_score", _finite(self.validation_score, "validation_score")
        )

    @property
    def model_digest(self) -> ContentDigest:
        return digest_bytes(self.model_bytes)


@dataclass(frozen=True)
class EvaluationOutcome:
    """Exact ordered returns observed only on the supplied OOS slice."""

    observation_returns: Tuple[float, ...]

    def __post_init__(self) -> None:
        returns = tuple(
            _finite(value, "out-of-sample return") for value in self.observation_returns
        )
        if not returns:
            raise ValueError("evaluation outcome must not be empty")
        object.__setattr__(self, "observation_returns", returns)

    @property
    def metrics(self) -> Dict[str, float]:
        return _metrics(self.observation_returns)


@dataclass(frozen=True)
class WindowEvidence:
    window_id: str
    boundary: WindowBoundary
    train_start_at: str
    train_end_at: str
    validation_start_at: str
    validation_end_at: str
    test_start_at: str
    test_end_at: str
    optimizer_seed: int
    evaluator_seed: int
    train_digest: ContentDigest
    validation_digest: ContentDigest
    test_digest: ContentDigest
    model_id: str
    model_version: str
    model_digest: ContentDigest
    selected_config: HashedInput
    validation_score: float
    oos_returns: Tuple[float, ...]

    def __post_init__(self) -> None:
        _required_text(self.window_id, "window_id")
        if self.window_id != self.boundary.window_id:
            raise ValueError("window evidence identifier mismatch")
        for value, name in (
            (self.model_id, "model_id"),
            (self.model_version, "model_version"),
            (self.train_start_at, "train_start_at"),
            (self.train_end_at, "train_end_at"),
            (self.validation_start_at, "validation_start_at"),
            (self.validation_end_at, "validation_end_at"),
            (self.test_start_at, "test_start_at"),
            (self.test_end_at, "test_end_at"),
        ):
            _required_text(value, name)
        train_start = _timestamp(self.train_start_at, "train_start_at")
        train_end = _timestamp(self.train_end_at, "train_end_at")
        validation_start = _timestamp(self.validation_start_at, "validation_start_at")
        validation_end = _timestamp(self.validation_end_at, "validation_end_at")
        test_start = _timestamp(self.test_start_at, "test_start_at")
        test_end = _timestamp(self.test_end_at, "test_end_at")
        if (
            not train_start
            <= train_end
            < validation_start
            <= validation_end
            < test_start
            <= test_end
        ):
            raise ValueError("window timestamps must be strictly phase ordered")
        if self.selected_config.kind != "config":
            raise ValueError("window selected_config must be a config input")
        for seed_value, name in (
            (self.optimizer_seed, "optimizer_seed"),
            (self.evaluator_seed, "evaluator_seed"),
        ):
            _seed(seed_value, name)
        for digest_value, name in (
            (self.train_digest, "train_digest"),
            (self.validation_digest, "validation_digest"),
            (self.test_digest, "test_digest"),
            (self.model_digest, "model_digest"),
        ):
            if not isinstance(digest_value, ContentDigest):
                raise TypeError(f"{name} must be a ContentDigest")
        object.__setattr__(
            self, "validation_score", _finite(self.validation_score, "validation_score")
        )
        returns = tuple(_finite(value, "OOS return") for value in self.oos_returns)
        if not returns:
            raise ValueError("window evidence requires OOS observations")
        object.__setattr__(self, "oos_returns", returns)

    @property
    def oos_metrics(self) -> Dict[str, float]:
        return _metrics(self.oos_returns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "boundary": self.boundary.to_dict(),
            "train_start_at": self.train_start_at,
            "train_end_at": self.train_end_at,
            "validation_start_at": self.validation_start_at,
            "validation_end_at": self.validation_end_at,
            "test_start_at": self.test_start_at,
            "test_end_at": self.test_end_at,
            "optimizer_seed": self.optimizer_seed,
            "evaluator_seed": self.evaluator_seed,
            "train_digest": self.train_digest.to_dict(),
            "validation_digest": self.validation_digest.to_dict(),
            "test_digest": self.test_digest.to_dict(),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_digest": self.model_digest.to_dict(),
            "selected_config": self.selected_config.to_dict(),
            "validation_score": self.validation_score,
            "oos_returns": list(self.oos_returns),
            "oos_metrics": self.oos_metrics,
        }


@dataclass(frozen=True)
class HoldoutEvidence:
    partition_id: str
    start_at: str
    end_at: str
    data_digest: ContentDigest
    evaluator_seed: int
    selected_window_id: str
    observation_returns: Tuple[float, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.partition_id, "holdout partition_id"),
            (self.start_at, "holdout start_at"),
            (self.end_at, "holdout end_at"),
            (self.selected_window_id, "selected_window_id"),
        ):
            _required_text(value, name)
        if _timestamp(self.start_at, "holdout start_at") > _timestamp(
            self.end_at, "holdout end_at"
        ):
            raise ValueError("holdout timestamps are reversed")
        returns = tuple(_finite(value, "holdout return") for value in self.observation_returns)
        if not returns:
            raise ValueError("holdout evidence requires observations")
        object.__setattr__(self, "observation_returns", returns)
        if not isinstance(self.data_digest, ContentDigest):
            raise TypeError("holdout data_digest must be a ContentDigest")
        _seed(self.evaluator_seed, "holdout evaluator_seed")

    @property
    def metrics(self) -> Dict[str, float]:
        return _metrics(self.observation_returns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "data_digest": self.data_digest.to_dict(),
            "evaluator_seed": self.evaluator_seed,
            "selected_window_id": self.selected_window_id,
            "observation_returns": list(self.observation_returns),
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class BoundWalkForwardEvidence:
    schema_version: str
    manifest_digest: ContentDigest
    validation_result_digest: ContentDigest
    source_data_input_id: str
    base_config_input_id: str
    model_evidence_input_id: str
    result_id: str

    def __post_init__(self) -> None:
        if self.schema_version != "robotrader-bound-walk-forward-v1":
            raise ValueError("unsupported bound walk-forward evidence schema")
        for value, name in (
            (self.source_data_input_id, "source_data_input_id"),
            (self.base_config_input_id, "base_config_input_id"),
            (self.model_evidence_input_id, "model_evidence_input_id"),
            (self.result_id, "result_id"),
        ):
            _required_text(value, name)
        if not isinstance(self.manifest_digest, ContentDigest) or not isinstance(
            self.validation_result_digest, ContentDigest
        ):
            raise TypeError("bound evidence digests must be ContentDigest values")


@dataclass(frozen=True)
class WalkForwardResult:
    schema_version: str
    plan: WalkForwardPlan
    source_data: HashedInput
    base_config: HashedInput
    timezone: str
    first_event_at: str
    last_event_at: str
    seed_policy: SeedPolicy
    windows: Tuple[WindowEvidence, ...]
    selected_window_id: str
    aggregate_oos_returns: Tuple[float, ...]
    holdout: HoldoutEvidence

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        if self.source_data.kind != "data" or self.base_config.kind != "config":
            raise ValueError("walk-forward result requires data and config input evidence")
        _required_text(self.timezone, "timezone")
        ZoneInfo(self.timezone)
        windows = tuple(self.windows)
        if not windows:
            raise ValueError("walk-forward result requires completed windows")
        if len({window.window_id for window in windows}) != len(windows):
            raise ValueError("walk-forward window identifiers must be unique")
        if self.selected_window_id not in {window.window_id for window in windows}:
            raise ValueError("selected_window_id is not a completed window")
        if self.seed_policy.root_seed != self.plan.root_seed:
            raise ValueError("result seed root does not match the validation plan")
        if self.seed_policy.generator_id != self.plan.seed_generator_id:
            raise ValueError("result seed generator does not match the validation plan")
        expected_components = {
            component
            for window in windows
            for component in (
                f"{window.window_id}:optimizer",
                f"{window.window_id}:evaluator",
            )
        } | {"holdout:evaluator"}
        actual_components = {assignment.component_id for assignment in self.seed_policy.assignments}
        if actual_components != expected_components:
            raise ValueError("result seed components do not match completed windows")
        for window in windows:
            if window.optimizer_seed != self.seed_policy.seed_for(
                f"{window.window_id}:optimizer"
            ) or window.evaluator_seed != self.seed_policy.seed_for(
                f"{window.window_id}:evaluator"
            ):
                raise ValueError("window seed evidence does not match the seed policy")
            boundary = window.boundary
            if (
                boundary.train_end - boundary.train_start != self.plan.train_size
                or boundary.validation_end - boundary.validation_start != self.plan.validation_size
                or boundary.test_end - boundary.test_start != self.plan.test_size
                or boundary.validation_start - boundary.train_end != self.plan.purge_size
                or boundary.test_start - boundary.validation_end != self.plan.purge_size
            ):
                raise ValueError("window evidence does not match fixed plan sizes")
        for previous, current in zip(windows, windows[1:]):
            if previous.boundary.test_end + self.plan.embargo_size > current.boundary.test_start:
                raise ValueError("window evidence violates test non-overlap or embargo")
        aggregate = tuple(
            _finite(value, "aggregate OOS return") for value in self.aggregate_oos_returns
        )
        expected = tuple(value for window in windows for value in window.oos_returns)
        if aggregate != expected:
            raise ValueError("aggregate returns must contain only ordered window OOS observations")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "aggregate_oos_returns", aggregate)
        if self.holdout.partition_id != self.plan.holdout_partition_id:
            raise ValueError("holdout partition identity does not match the validation plan")
        if self.holdout.selected_window_id != self.selected_window_id:
            raise ValueError("holdout was not evaluated with the preselected window model")
        if self.holdout.evaluator_seed != self.seed_policy.seed_for("holdout:evaluator"):
            raise ValueError("holdout seed evidence does not match the seed policy")
        first = _timestamp(self.first_event_at, "first_event_at")
        last = _timestamp(self.last_event_at, "last_event_at")
        holdout_start = _timestamp(self.holdout.start_at, "holdout start_at")
        holdout_end = _timestamp(self.holdout.end_at, "holdout end_at")
        latest_test = max(_timestamp(window.test_end_at, "test_end_at") for window in windows)
        if not first <= latest_test < holdout_start <= holdout_end <= last:
            raise ValueError("result timestamps violate test and holdout ordering")

    @property
    def aggregate_oos_metrics(self) -> Dict[str, float]:
        return _metrics(self.aggregate_oos_returns)

    def model_evidence_digest(self) -> ContentDigest:
        return digest_json(
            {
                "schema_version": self.schema_version,
                "models": [
                    {
                        "window_id": window.window_id,
                        "model_id": window.model_id,
                        "model_version": window.model_version,
                        "model_digest": window.model_digest.to_dict(),
                        "selected_config": window.selected_config.to_dict(),
                    }
                    for window in self.windows
                ],
                "selected_window_id": self.selected_window_id,
            }
        )

    def as_model_input(self, identifier: str) -> HashedInput:
        return HashedInput("model", identifier, self.model_evidence_digest())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan": self.plan.to_dict(),
            "source_data": self.source_data.to_dict(),
            "base_config": self.base_config.to_dict(),
            "timezone": self.timezone,
            "first_event_at": self.first_event_at,
            "last_event_at": self.last_event_at,
            "seed_policy": self.seed_policy.to_dict(),
            "windows": [window.to_dict() for window in self.windows],
            "selected_window_id": self.selected_window_id,
            "aggregate_oos_returns": list(self.aggregate_oos_returns),
            "aggregate_oos_metrics": self.aggregate_oos_metrics,
            "holdout": self.holdout.to_dict(),
            "model_evidence_digest": self.model_evidence_digest().to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    def evidence_digest(self) -> ContentDigest:
        return digest_bytes(self.to_bytes())

    def bind_to_manifest(
        self,
        manifest: BacktestRunManifest,
        *,
        model_evidence_input_id: str,
    ) -> BoundWalkForwardEvidence:
        manifest.verify_result(self.to_bytes())
        by_identity = {(item.kind, item.identifier): item for item in manifest.inputs}
        expected_inputs = (
            self.source_data,
            self.base_config,
            self.as_model_input(model_evidence_input_id),
        )
        for expected in expected_inputs:
            actual = by_identity.get((expected.kind, expected.identifier))
            if actual != expected:
                raise ValueError(
                    f"manifest {expected.kind} evidence mismatch for {expected.identifier}"
                )
        if manifest.seed_policy != self.seed_policy:
            raise ValueError("manifest seed policy does not match walk-forward evidence")
        if manifest.data_assumptions.timezone != self.timezone:
            raise ValueError("manifest timezone does not match walk-forward evidence")
        first = pd.Timestamp(self.first_event_at)
        last = pd.Timestamp(self.last_event_at)
        if first < pd.Timestamp(manifest.data_assumptions.start_at) or last > pd.Timestamp(
            manifest.data_assumptions.end_at
        ):
            raise ValueError("manifest data window does not contain validation events")
        matching_holdout = [
            partition
            for partition in manifest.partitions
            if partition.partition_id == self.plan.holdout_partition_id
            and partition.split_id == self.plan.split_id
            and partition.role == "holdout"
            and partition.untouched
        ]
        if len(matching_holdout) != 1:
            raise ValueError("manifest holdout identity does not match validation plan")
        holdout_partition = matching_holdout[0]
        if holdout_partition.content_digest != self.holdout.data_digest:
            raise ValueError("manifest holdout content digest does not match validation evidence")
        holdout_first = pd.Timestamp(self.holdout.start_at)
        holdout_last = pd.Timestamp(self.holdout.end_at)
        if holdout_first < pd.Timestamp(holdout_partition.start_at) or holdout_last > pd.Timestamp(
            holdout_partition.end_at
        ):
            raise ValueError("manifest holdout window does not contain validation holdout")
        return BoundWalkForwardEvidence(
            schema_version="robotrader-bound-walk-forward-v1",
            manifest_digest=manifest.manifest_digest(),
            validation_result_digest=self.evidence_digest(),
            source_data_input_id=self.source_data.identifier,
            base_config_input_id=self.base_config.identifier,
            model_evidence_input_id=model_evidence_input_id,
            result_id=manifest.result.result_id,
        )


Optimizer = Callable[[pd.DataFrame, pd.DataFrame, int], OptimizationOutcome]
Evaluator = Callable[[OptimizationOutcome, pd.DataFrame, int], EvaluationOutcome]


class WalkForwardValidator:
    """Execute fixed windows without exposing future or holdout rows to fitting."""

    def __init__(self, plan: WalkForwardPlan):
        if not isinstance(plan, WalkForwardPlan):
            raise TypeError("plan must be a WalkForwardPlan")
        self.plan = plan

    def build_windows(self, event_count: int) -> Tuple[WindowBoundary, ...]:
        count = _positive_integer(event_count, "event_count")
        holdout_start = count - self.plan.holdout_size
        development_end = holdout_start - self.plan.purge_size
        if development_end < self.plan.minimum_development_size:
            raise WalkForwardValidationError(
                "insufficient events for one fixed train/validation/test window, "
                "holdout purge, and holdout"
            )
        windows: List[WindowBoundary] = []
        start = 0
        while True:
            train_start = start
            train_end = train_start + self.plan.train_size
            validation_start = train_end + self.plan.purge_size
            validation_end = validation_start + self.plan.validation_size
            test_start = validation_end + self.plan.purge_size
            test_end = test_start + self.plan.test_size
            if test_end > development_end:
                break
            windows.append(
                WindowBoundary(
                    window_id=f"window-{len(windows):04d}",
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            start += self.plan.step_size
        if not windows:
            raise WalkForwardValidationError("walk-forward plan produced no complete windows")
        for previous, current in zip(windows, windows[1:]):
            if previous.test_end + self.plan.embargo_size > current.test_start:
                raise WalkForwardValidationError("test windows violate non-overlap or embargo")
        if any(window.test_end > development_end for window in windows):
            raise WalkForwardValidationError(
                "a test window overlaps the final holdout purge or holdout"
            )
        return tuple(windows)

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("data must not be empty")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("walk-forward data must use a DatetimeIndex")
        if data.index.tz is None:
            raise ValueError("walk-forward timestamps must be timezone-aware")
        if data.index.hasnans or data.index.has_duplicates:
            raise ValueError("walk-forward timestamps must be unique and finite")
        if not data.index.is_monotonic_increasing:
            raise ValueError("walk-forward data must be sorted in increasing time order")
        if data.columns.has_duplicates or any(
            not isinstance(column, str) for column in data.columns
        ):
            raise ValueError("walk-forward columns must be unique strings")
        if data.empty or not data.shape[1]:
            raise ValueError("walk-forward data must contain features")
        for column in data.columns:
            try:
                values = data[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"walk-forward column {column} must be numeric") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"walk-forward column {column} must be finite")

    @staticmethod
    def _slice(data: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
        frame = data.iloc[start:end].copy(deep=True)
        if frame.empty or len(frame) != end - start:
            raise WalkForwardValidationError("window slice is empty or unexpectedly short")
        return frame

    def run(
        self,
        data: pd.DataFrame,
        *,
        source_data: HashedInput,
        base_config: HashedInput,
        optimizer: Optimizer,
        evaluator: Evaluator,
    ) -> WalkForwardResult:
        self._validate_data(data)
        if source_data.kind != "data" or source_data.digest != digest_dataframe(data):
            raise ValueError("source_data evidence does not match supplied DataFrame")
        if base_config.kind != "config":
            raise ValueError("base_config must be a hashed config input")
        if not callable(optimizer) or not callable(evaluator):
            raise TypeError("optimizer and evaluator must be callable")
        windows = self.build_windows(len(data))
        component_ids = [
            component
            for window in windows
            for component in (
                f"{window.window_id}:optimizer",
                f"{window.window_id}:evaluator",
            )
        ] + ["holdout:evaluator"]
        seed_policy = SeedPolicy.from_root(
            self.plan.root_seed, component_ids, self.plan.seed_generator_id
        )
        window_evidence: List[WindowEvidence] = []
        outcomes: Dict[str, OptimizationOutcome] = {}

        for boundary in windows:
            train = self._slice(data, boundary.train_start, boundary.train_end)
            validation = self._slice(data, boundary.validation_start, boundary.validation_end)
            test = self._slice(data, boundary.test_start, boundary.test_end)
            optimizer_seed = seed_policy.seed_for(f"{boundary.window_id}:optimizer")
            evaluator_seed = seed_policy.seed_for(f"{boundary.window_id}:evaluator")
            try:
                outcome = optimizer(
                    train.copy(deep=True), validation.copy(deep=True), optimizer_seed
                )
            except Exception as exc:
                raise WalkForwardValidationError(
                    f"optimizer failed for {boundary.window_id}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(outcome, OptimizationOutcome):
                raise WalkForwardValidationError("optimizer returned an invalid or empty outcome")
            try:
                evaluation = evaluator(outcome, test.copy(deep=True), evaluator_seed)
            except Exception as exc:
                raise WalkForwardValidationError(
                    f"evaluator failed for {boundary.window_id}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(evaluation, EvaluationOutcome):
                raise WalkForwardValidationError("evaluator returned an invalid or empty outcome")
            evidence = WindowEvidence(
                window_id=boundary.window_id,
                boundary=boundary,
                train_start_at=train.index[0].isoformat(),
                train_end_at=train.index[-1].isoformat(),
                validation_start_at=validation.index[0].isoformat(),
                validation_end_at=validation.index[-1].isoformat(),
                test_start_at=test.index[0].isoformat(),
                test_end_at=test.index[-1].isoformat(),
                optimizer_seed=optimizer_seed,
                evaluator_seed=evaluator_seed,
                train_digest=digest_dataframe(train),
                validation_digest=digest_dataframe(validation),
                test_digest=digest_dataframe(test),
                model_id=outcome.model_id,
                model_version=outcome.model_version,
                model_digest=outcome.model_digest,
                selected_config=outcome.selected_config,
                validation_score=outcome.validation_score,
                oos_returns=evaluation.observation_returns,
            )
            window_evidence.append(evidence)
            outcomes[boundary.window_id] = outcome

        selected = max(
            window_evidence,
            key=lambda evidence: (
                evidence.oos_metrics[self.plan.selection_metric],
                evidence.window_id,
            ),
        )
        holdout_start = len(data) - self.plan.holdout_size
        holdout_frame = self._slice(data, holdout_start, len(data))
        holdout_seed = seed_policy.seed_for("holdout:evaluator")
        try:
            holdout_outcome = evaluator(
                outcomes[selected.window_id], holdout_frame.copy(deep=True), holdout_seed
            )
        except Exception as exc:
            raise WalkForwardValidationError(
                f"holdout evaluator failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(holdout_outcome, EvaluationOutcome):
            raise WalkForwardValidationError("holdout evaluator returned an invalid outcome")
        holdout = HoldoutEvidence(
            partition_id=self.plan.holdout_partition_id,
            start_at=holdout_frame.index[0].isoformat(),
            end_at=holdout_frame.index[-1].isoformat(),
            data_digest=digest_dataframe(holdout_frame),
            evaluator_seed=holdout_seed,
            selected_window_id=selected.window_id,
            observation_returns=holdout_outcome.observation_returns,
        )
        aggregate = tuple(value for evidence in window_evidence for value in evidence.oos_returns)
        return WalkForwardResult(
            schema_version=_SCHEMA_VERSION,
            plan=self.plan,
            source_data=source_data,
            base_config=base_config,
            timezone=str(data.index.tz),
            first_event_at=data.index[0].isoformat(),
            last_event_at=data.index[-1].isoformat(),
            seed_policy=seed_policy,
            windows=tuple(window_evidence),
            selected_window_id=selected.window_id,
            aggregate_oos_returns=aggregate,
            holdout=holdout,
        )


__all__: Tuple[str, ...] = (
    "BoundWalkForwardEvidence",
    "EvaluationOutcome",
    "HoldoutEvidence",
    "OptimizationOutcome",
    "WalkForwardPlan",
    "WalkForwardResult",
    "WalkForwardValidationError",
    "WalkForwardValidator",
    "WindowBoundary",
    "WindowEvidence",
)
