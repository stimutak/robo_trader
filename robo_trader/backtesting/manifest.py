"""Immutable reproducibility contract for deterministic offline backtests."""

import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .provenance import ContentDigest, HashedInput, digest_bytes, digest_json

_SCHEMA_VERSION = "robotrader-backtest-manifest-v1"
_PARTITION_ROLES = frozenset({"train", "validation", "test", "holdout"})
_FINALIZATION_POLICIES = frozenset({"liquidate", "mark_to_market"})
_REQUIRED_INPUT_KINDS = frozenset({"data", "config", "code", "model"})
_MAX_SEED = 2**64 - 1


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _aware_datetime(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value


def _iana_timezone_key(value: datetime) -> Any:
    """Return an exact IANA key for both zoneinfo and pandas 2/pytz datetimes."""

    if value.tzinfo is None:
        return None
    key = getattr(value.tzinfo, "key", None)
    if key is None:
        key = getattr(value.tzinfo, "zone", None)
    if not isinstance(key, str):
        return None
    try:
        return ZoneInfo(key).key
    except ZoneInfoNotFoundError:
        return None


def _parse_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 datetime") from exc
    return _aware_datetime(parsed, name)


@dataclass(frozen=True, order=True)
class SeedAssignment:
    component_id: str
    seed: int

    def __post_init__(self) -> None:
        _text(self.component_id, "seed component_id")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= _MAX_SEED
        ):
            raise ValueError("component seed must be an unsigned 64-bit integer")

    def to_dict(self) -> Dict[str, Any]:
        return {"component_id": self.component_id, "seed": self.seed}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SeedAssignment":
        _exact_keys(value, {"component_id", "seed"}, "seed assignment")
        return cls(component_id=value["component_id"], seed=value["seed"])


@dataclass(frozen=True)
class SeedPolicy:
    """Explicit deterministic seed ownership for every stochastic component."""

    root_seed: int
    generator_id: str
    derivation_id: str
    assignments: Tuple[SeedAssignment, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.root_seed, bool)
            or not isinstance(self.root_seed, int)
            or not 0 <= self.root_seed <= _MAX_SEED
        ):
            raise ValueError("root_seed must be an unsigned 64-bit integer")
        _text(self.generator_id, "seed generator_id")
        if self.derivation_id != "sha256-component-v1":
            raise ValueError("unsupported seed derivation policy")
        assignments = tuple(sorted(self.assignments))
        if not assignments:
            raise ValueError("seed policy requires at least one component assignment")
        if len({assignment.component_id for assignment in assignments}) != len(assignments):
            raise ValueError("seed component assignments must be unique")
        for assignment in assignments:
            if assignment.seed != self.derive_seed(self.root_seed, assignment.component_id):
                raise ValueError(f"seed assignment mismatch for {assignment.component_id}")
        object.__setattr__(self, "assignments", assignments)

    @staticmethod
    def derive_seed(root_seed: int, component_id: str) -> int:
        if (
            isinstance(root_seed, bool)
            or not isinstance(root_seed, int)
            or not 0 <= root_seed <= _MAX_SEED
        ):
            raise ValueError("root_seed must be an unsigned 64-bit integer")
        component = _text(component_id, "seed component_id")
        payload = f"sha256-component-v1\0{root_seed}\0{component}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    @classmethod
    def from_root(
        cls, root_seed: int, component_ids: Iterable[str], generator_id: str = "numpy-pcg64"
    ) -> "SeedPolicy":
        components = tuple(component_ids)
        if len(set(components)) != len(components):
            raise ValueError("seed component identifiers must be unique")
        assignments = tuple(
            SeedAssignment(component, cls.derive_seed(root_seed, component))
            for component in components
        )
        return cls(root_seed, generator_id, "sha256-component-v1", assignments)

    def seed_for(self, component_id: str) -> int:
        for assignment in self.assignments:
            if assignment.component_id == component_id:
                return assignment.seed
        raise KeyError(component_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_seed": self.root_seed,
            "generator_id": self.generator_id,
            "derivation_id": self.derivation_id,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SeedPolicy":
        _exact_keys(
            value,
            {"root_seed", "generator_id", "derivation_id", "assignments"},
            "seed policy",
        )
        return cls(
            root_seed=value["root_seed"],
            generator_id=value["generator_id"],
            derivation_id=value["derivation_id"],
            assignments=tuple(
                SeedAssignment.from_dict(_mapping(item, "seed assignment"))
                for item in _sequence(value["assignments"], "seed assignments")
            ),
        )


@dataclass(frozen=True)
class DataAssumptions:
    timezone: str
    calendar_id: str
    session_policy_id: str
    bar_interval_id: str
    start_at: datetime
    end_at: datetime
    corporate_action_policy_id: str
    price_adjustment_policy_id: str
    missing_quote_policy_id: str

    def __post_init__(self) -> None:
        _text(self.timezone, "timezone")
        try:
            zone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone {self.timezone}") from exc
        start = _aware_datetime(self.start_at, "start_at")
        end = _aware_datetime(self.end_at, "end_at")
        if _iana_timezone_key(start) != zone.key or _iana_timezone_key(end) != zone.key:
            raise ValueError("data window datetimes must use the declared IANA timezone")
        if start >= end:
            raise ValueError("data window start_at must precede end_at")
        for value, name in (
            (self.calendar_id, "calendar_id"),
            (self.session_policy_id, "session_policy_id"),
            (self.bar_interval_id, "bar_interval_id"),
            (self.corporate_action_policy_id, "corporate_action_policy_id"),
            (self.price_adjustment_policy_id, "price_adjustment_policy_id"),
            (self.missing_quote_policy_id, "missing_quote_policy_id"),
        ):
            _text(value, name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timezone": self.timezone,
            "calendar_id": self.calendar_id,
            "session_policy_id": self.session_policy_id,
            "bar_interval_id": self.bar_interval_id,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
            "corporate_action_policy_id": self.corporate_action_policy_id,
            "price_adjustment_policy_id": self.price_adjustment_policy_id,
            "missing_quote_policy_id": self.missing_quote_policy_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataAssumptions":
        keys = {
            "timezone",
            "calendar_id",
            "session_policy_id",
            "bar_interval_id",
            "start_at",
            "end_at",
            "corporate_action_policy_id",
            "price_adjustment_policy_id",
            "missing_quote_policy_id",
        }
        _exact_keys(value, keys, "data assumptions")
        timezone = value["timezone"]
        if not isinstance(timezone, str):
            raise TypeError("timezone must be a string")
        try:
            zone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone {timezone}") from exc
        start = _parse_datetime(value["start_at"], "start_at").astimezone(zone)
        end = _parse_datetime(value["end_at"], "end_at").astimezone(zone)
        return cls(
            timezone=timezone,
            calendar_id=value["calendar_id"],
            session_policy_id=value["session_policy_id"],
            bar_interval_id=value["bar_interval_id"],
            start_at=start,
            end_at=end,
            corporate_action_policy_id=value["corporate_action_policy_id"],
            price_adjustment_policy_id=value["price_adjustment_policy_id"],
            missing_quote_policy_id=value["missing_quote_policy_id"],
        )


@dataclass(frozen=True, order=True)
class DatasetPartition:
    partition_id: str
    split_id: str
    source_data_input_id: str
    role: str
    start_at: datetime
    end_at: datetime
    content_digest: ContentDigest
    untouched: bool

    def __post_init__(self) -> None:
        for value, name in (
            (self.partition_id, "partition_id"),
            (self.split_id, "split_id"),
            (self.source_data_input_id, "source_data_input_id"),
        ):
            _text(value, name)
        if self.role not in _PARTITION_ROLES:
            raise ValueError(f"unsupported dataset partition role {self.role!r}")
        start = _aware_datetime(self.start_at, "partition start_at")
        end = _aware_datetime(self.end_at, "partition end_at")
        if start >= end:
            raise ValueError("partition start_at must precede end_at")
        if not isinstance(self.content_digest, ContentDigest):
            raise TypeError("partition content_digest must be a ContentDigest")
        if not isinstance(self.untouched, bool):
            raise ValueError("partition untouched must be a boolean")
        if self.role != "holdout" and self.untouched:
            raise ValueError("only the holdout partition may be marked untouched")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "split_id": self.split_id,
            "source_data_input_id": self.source_data_input_id,
            "role": self.role,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
            "content_digest": self.content_digest.to_dict(),
            "untouched": self.untouched,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetPartition":
        keys = {
            "partition_id",
            "split_id",
            "source_data_input_id",
            "role",
            "start_at",
            "end_at",
            "content_digest",
            "untouched",
        }
        _exact_keys(value, keys, "dataset partition")
        return cls(
            partition_id=value["partition_id"],
            split_id=value["split_id"],
            source_data_input_id=value["source_data_input_id"],
            role=value["role"],
            start_at=_parse_datetime(value["start_at"], "partition start_at"),
            end_at=_parse_datetime(value["end_at"], "partition end_at"),
            content_digest=ContentDigest.from_dict(
                _mapping(value["content_digest"], "partition digest")
            ),
            untouched=value["untouched"],
        )


@dataclass(frozen=True)
class ExecutionAssumptions:
    commission_policy_id: str
    slippage_policy_id: str
    fill_policy_id: str
    market_impact_policy_id: str
    partial_fill_policy_id: str
    finalization_policy_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.commission_policy_id, "commission_policy_id"),
            (self.slippage_policy_id, "slippage_policy_id"),
            (self.fill_policy_id, "fill_policy_id"),
            (self.market_impact_policy_id, "market_impact_policy_id"),
            (self.partial_fill_policy_id, "partial_fill_policy_id"),
        ):
            _text(value, name)
        if self.finalization_policy_id not in _FINALIZATION_POLICIES:
            raise ValueError("finalization policy must be liquidate or mark_to_market")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commission_policy_id": self.commission_policy_id,
            "slippage_policy_id": self.slippage_policy_id,
            "fill_policy_id": self.fill_policy_id,
            "market_impact_policy_id": self.market_impact_policy_id,
            "partial_fill_policy_id": self.partial_fill_policy_id,
            "finalization_policy_id": self.finalization_policy_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionAssumptions":
        keys = {
            "commission_policy_id",
            "slippage_policy_id",
            "fill_policy_id",
            "market_impact_policy_id",
            "partial_fill_policy_id",
            "finalization_policy_id",
        }
        _exact_keys(value, keys, "execution assumptions")
        return cls(**{key: value[key] for key in keys})


@dataclass(frozen=True, order=True)
class PackageVersion:
    name: str
    version: str

    def __post_init__(self) -> None:
        _text(self.name, "package name")
        _text(self.version, "package version")

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PackageVersion":
        _exact_keys(value, {"name", "version"}, "package version")
        return cls(name=value["name"], version=value["version"])


@dataclass(frozen=True)
class EnvironmentMetadata:
    python_version: str
    implementation: str
    platform_id: str
    packages: Tuple[PackageVersion, ...]
    package_lock: HashedInput

    def __post_init__(self) -> None:
        for value, name in (
            (self.python_version, "python_version"),
            (self.implementation, "implementation"),
            (self.platform_id, "platform_id"),
        ):
            _text(value, name)
        packages = tuple(sorted(self.packages, key=lambda package: package.name.lower()))
        if not packages:
            raise ValueError("environment metadata requires package versions")
        names = [package.name.lower() for package in packages]
        if len(set(names)) != len(names):
            raise ValueError("environment package names must be unique")
        if self.package_lock.kind != "package-lock":
            raise ValueError("package_lock must use the package-lock input kind")
        object.__setattr__(self, "packages", packages)

    @classmethod
    def capture(
        cls, package_names: Iterable[str], package_lock: HashedInput
    ) -> "EnvironmentMetadata":
        requested = tuple(sorted({_text(name, "package name") for name in package_names}))
        if not requested:
            raise ValueError("at least one package name must be captured")
        packages = tuple(
            PackageVersion(name, importlib.metadata.version(name)) for name in requested
        )
        return cls(
            python_version=platform.python_version(),
            implementation=sys.implementation.name,
            platform_id=platform.platform(),
            packages=packages,
            package_lock=package_lock,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "implementation": self.implementation,
            "platform_id": self.platform_id,
            "packages": [package.to_dict() for package in self.packages],
            "package_lock": self.package_lock.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentMetadata":
        keys = {"python_version", "implementation", "platform_id", "packages", "package_lock"}
        _exact_keys(value, keys, "environment metadata")
        return cls(
            python_version=value["python_version"],
            implementation=value["implementation"],
            platform_id=value["platform_id"],
            packages=tuple(
                PackageVersion.from_dict(_mapping(item, "package version"))
                for item in _sequence(value["packages"], "packages")
            ),
            package_lock=HashedInput.from_dict(_mapping(value["package_lock"], "package lock")),
        )


@dataclass(frozen=True)
class ResultArtifact:
    result_id: str
    format_id: str
    content_digest: ContentDigest
    input_manifest_digest: ContentDigest

    def __post_init__(self) -> None:
        _text(self.result_id, "result_id")
        _text(self.format_id, "result format_id")
        if not isinstance(self.content_digest, ContentDigest) or not isinstance(
            self.input_manifest_digest, ContentDigest
        ):
            raise TypeError("result digests must be ContentDigest values")
        if self.content_digest.byte_length == 0:
            raise ValueError("result artifact must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "format_id": self.format_id,
            "content_digest": self.content_digest.to_dict(),
            "input_manifest_digest": self.input_manifest_digest.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResultArtifact":
        keys = {"result_id", "format_id", "content_digest", "input_manifest_digest"}
        _exact_keys(value, keys, "result artifact")
        return cls(
            result_id=value["result_id"],
            format_id=value["format_id"],
            content_digest=ContentDigest.from_dict(
                _mapping(value["content_digest"], "result digest")
            ),
            input_manifest_digest=ContentDigest.from_dict(
                _mapping(value["input_manifest_digest"], "input manifest digest")
            ),
        )


@dataclass(frozen=True)
class BacktestRunManifest:
    schema_version: str
    run_id: str
    created_at: datetime
    strategy_id: str
    strategy_version: str
    inputs: Tuple[HashedInput, ...]
    seed_policy: SeedPolicy
    data_assumptions: DataAssumptions
    partitions: Tuple[DatasetPartition, ...]
    execution_assumptions: ExecutionAssumptions
    environment: EnvironmentMetadata
    approval_eligible: bool
    recorded_errors: Tuple[str, ...]
    result: ResultArtifact

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        for value, name in (
            (self.run_id, "run_id"),
            (self.strategy_id, "strategy_id"),
            (self.strategy_version, "strategy_version"),
        ):
            _text(value, name)
        _aware_datetime(self.created_at, "created_at")
        inputs = tuple(sorted(self.inputs, key=lambda item: (item.kind, item.identifier)))
        if len({(item.kind, item.identifier) for item in inputs}) != len(inputs):
            raise ValueError("manifest input identities must be unique")
        kinds = {item.kind for item in inputs}
        if not _REQUIRED_INPUT_KINDS.issubset(kinds):
            raise ValueError(
                f"manifest requires hashed data/config/code/model inputs; found {sorted(kinds)}"
            )
        object.__setattr__(self, "inputs", inputs)

        partitions = tuple(sorted(self.partitions, key=lambda item: item.partition_id))
        if not partitions:
            raise ValueError("manifest requires dataset partitions")
        if len({partition.partition_id for partition in partitions}) != len(partitions):
            raise ValueError("dataset partition identifiers must be unique")
        split_ids = {partition.split_id for partition in partitions}
        if len(split_ids) != 1:
            raise ValueError("all partitions must belong to one explicit split_id")
        data_ids = {item.identifier for item in inputs if item.kind == "data"}
        if any(partition.source_data_input_id not in data_ids for partition in partitions):
            raise ValueError("partition references an unknown data input")
        holdouts = [partition for partition in partitions if partition.role == "holdout"]
        if len(holdouts) != 1 or not holdouts[0].untouched:
            raise ValueError("manifest requires exactly one untouched holdout partition")
        if not any(partition.role == "train" for partition in partitions):
            raise ValueError("manifest requires an explicit training partition")
        for earlier, later in zip(
            sorted(partitions, key=lambda item: item.start_at),
            sorted(partitions, key=lambda item: item.start_at)[1:],
        ):
            if earlier.end_at > later.start_at:
                raise ValueError("dataset partitions must not overlap")
        if any(
            partition.start_at < self.data_assumptions.start_at
            or partition.end_at > self.data_assumptions.end_at
            for partition in partitions
        ):
            raise ValueError("dataset partition lies outside the declared data window")
        object.__setattr__(self, "partitions", partitions)

        if not isinstance(self.approval_eligible, bool):
            raise ValueError("approval_eligible must be a boolean")
        errors = tuple(self.recorded_errors)
        if any(not isinstance(error, str) or not error.strip() for error in errors):
            raise ValueError("recorded errors must be non-empty strings")
        if errors and self.approval_eligible:
            raise ValueError("a run with recorded errors cannot be approval eligible")
        object.__setattr__(self, "recorded_errors", errors)
        if not isinstance(self.result, ResultArtifact):
            raise TypeError("manifest requires a result artifact")
        if self.result.input_manifest_digest != self.input_manifest_digest():
            raise ValueError("result input-manifest linkage does not match manifest provenance")

    @staticmethod
    def _input_payload(
        *,
        schema_version: str,
        run_id: str,
        created_at: datetime,
        strategy_id: str,
        strategy_version: str,
        inputs: Tuple[HashedInput, ...],
        seed_policy: SeedPolicy,
        data_assumptions: DataAssumptions,
        partitions: Tuple[DatasetPartition, ...],
        execution_assumptions: ExecutionAssumptions,
        environment: EnvironmentMetadata,
        approval_eligible: bool,
        recorded_errors: Tuple[str, ...],
    ) -> Dict[str, Any]:
        return {
            "schema_version": schema_version,
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "strategy_id": strategy_id,
            "strategy_version": strategy_version,
            "inputs": [item.to_dict() for item in inputs],
            "seed_policy": seed_policy.to_dict(),
            "data_assumptions": data_assumptions.to_dict(),
            "partitions": [partition.to_dict() for partition in partitions],
            "execution_assumptions": execution_assumptions.to_dict(),
            "environment": environment.to_dict(),
            "approval_eligible": approval_eligible,
            "recorded_errors": list(recorded_errors),
        }

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        created_at: datetime,
        strategy_id: str,
        strategy_version: str,
        inputs: Iterable[HashedInput],
        seed_policy: SeedPolicy,
        data_assumptions: DataAssumptions,
        partitions: Iterable[DatasetPartition],
        execution_assumptions: ExecutionAssumptions,
        environment: EnvironmentMetadata,
        approval_eligible: bool,
        recorded_errors: Iterable[str],
        result_id: str,
        result_format_id: str,
        result_payload: bytes,
    ) -> "BacktestRunManifest":
        normalized_inputs = tuple(sorted(inputs, key=lambda item: (item.kind, item.identifier)))
        normalized_partitions = tuple(sorted(partitions, key=lambda item: item.partition_id))
        normalized_errors = tuple(recorded_errors)
        input_payload = cls._input_payload(
            schema_version=_SCHEMA_VERSION,
            run_id=run_id,
            created_at=created_at,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            inputs=normalized_inputs,
            seed_policy=seed_policy,
            data_assumptions=data_assumptions,
            partitions=normalized_partitions,
            execution_assumptions=execution_assumptions,
            environment=environment,
            approval_eligible=approval_eligible,
            recorded_errors=normalized_errors,
        )
        result_linkage = {
            "manifest_inputs": input_payload,
            "result_id": result_id,
            "result_format_id": result_format_id,
        }
        result = ResultArtifact(
            result_id=result_id,
            format_id=result_format_id,
            content_digest=digest_bytes(result_payload),
            input_manifest_digest=digest_json(result_linkage),
        )
        return cls(
            schema_version=_SCHEMA_VERSION,
            run_id=run_id,
            created_at=created_at,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            inputs=normalized_inputs,
            seed_policy=seed_policy,
            data_assumptions=data_assumptions,
            partitions=normalized_partitions,
            execution_assumptions=execution_assumptions,
            environment=environment,
            approval_eligible=approval_eligible,
            recorded_errors=normalized_errors,
            result=result,
        )

    def _current_input_payload(self) -> Dict[str, Any]:
        return self._input_payload(
            schema_version=self.schema_version,
            run_id=self.run_id,
            created_at=self.created_at,
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            inputs=self.inputs,
            seed_policy=self.seed_policy,
            data_assumptions=self.data_assumptions,
            partitions=self.partitions,
            execution_assumptions=self.execution_assumptions,
            environment=self.environment,
            approval_eligible=self.approval_eligible,
            recorded_errors=self.recorded_errors,
        )

    def input_manifest_digest(self) -> ContentDigest:
        return digest_json(
            {
                "manifest_inputs": self._current_input_payload(),
                "result_id": self.result.result_id,
                "result_format_id": self.result.format_id,
            }
        )

    def manifest_digest(self) -> ContentDigest:
        return digest_json(self.to_dict())

    def verify_result(self, result_payload: bytes) -> None:
        if digest_bytes(result_payload) != self.result.content_digest:
            raise ValueError("result content does not match manifest digest")
        if self.result.input_manifest_digest != self.input_manifest_digest():
            raise ValueError("result provenance linkage does not match manifest inputs")

    def validate_for_approval(self, result_payload: bytes) -> None:
        self.verify_result(result_payload)
        if not self.approval_eligible:
            raise ValueError("manifest is not approval eligible")
        if self.recorded_errors:
            raise ValueError("manifest with recorded errors is not approval eligible")

    def to_dict(self) -> Dict[str, Any]:
        payload = self._current_input_payload()
        payload["result"] = self.result.to_dict()
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> "BacktestRunManifest":
        if not isinstance(payload, str) or not payload:
            raise ValueError("manifest payload must be non-empty JSON text")

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON constant {value} is forbidden")

        try:
            raw = json.loads(payload, parse_constant=reject_constant)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest payload is not valid JSON") from exc
        value = _mapping(raw, "manifest")
        keys = {
            "schema_version",
            "run_id",
            "created_at",
            "strategy_id",
            "strategy_version",
            "inputs",
            "seed_policy",
            "data_assumptions",
            "partitions",
            "execution_assumptions",
            "environment",
            "approval_eligible",
            "recorded_errors",
            "result",
        }
        _exact_keys(value, keys, "manifest")
        manifest = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            created_at=_parse_datetime(value["created_at"], "created_at"),
            strategy_id=value["strategy_id"],
            strategy_version=value["strategy_version"],
            inputs=tuple(
                HashedInput.from_dict(_mapping(item, "hashed input"))
                for item in _sequence(value["inputs"], "inputs")
            ),
            seed_policy=SeedPolicy.from_dict(_mapping(value["seed_policy"], "seed policy")),
            data_assumptions=DataAssumptions.from_dict(
                _mapping(value["data_assumptions"], "data assumptions")
            ),
            partitions=tuple(
                DatasetPartition.from_dict(_mapping(item, "dataset partition"))
                for item in _sequence(value["partitions"], "partitions")
            ),
            execution_assumptions=ExecutionAssumptions.from_dict(
                _mapping(value["execution_assumptions"], "execution assumptions")
            ),
            environment=EnvironmentMetadata.from_dict(
                _mapping(value["environment"], "environment")
            ),
            approval_eligible=value["approval_eligible"],
            recorded_errors=tuple(_sequence(value["recorded_errors"], "recorded errors")),
            result=ResultArtifact.from_dict(_mapping(value["result"], "result")),
        )
        if manifest.to_json() != payload:
            raise ValueError("manifest JSON is not in canonical form")
        return manifest


__all__: Tuple[str, ...] = (
    "BacktestRunManifest",
    "DataAssumptions",
    "DatasetPartition",
    "EnvironmentMetadata",
    "ExecutionAssumptions",
    "PackageVersion",
    "ResultArtifact",
    "SeedAssignment",
    "SeedPolicy",
)
