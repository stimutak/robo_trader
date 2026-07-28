"""Secret-free evidence models for dormant SQLite maintenance tooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Tuple


@dataclass(frozen=True, slots=True)
class TableEvidence:
    """Deterministic logical evidence for one SQLite table."""

    name: str
    row_count: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class DatabaseEvidence:
    """Portable database evidence containing no paths or row values."""

    schema_sha256: str
    tables: Tuple[TableEvidence, ...]
    quick_check: str
    integrity_check: str
    foreign_key_violations: int
    application_id: int
    user_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatabaseEvidence":
        _require_exact_keys(
            value,
            {
                "schema_sha256",
                "tables",
                "quick_check",
                "integrity_check",
                "foreign_key_violations",
                "application_id",
                "user_version",
            },
            "database evidence",
        )
        tables = value.get("tables")
        if not isinstance(tables, (list, tuple)):
            raise ValueError("database evidence tables must be a sequence")
        if not all(isinstance(item, Mapping) for item in tables):
            raise ValueError("database evidence table entries must be objects")
        for item in tables:
            _require_exact_keys(
                item,
                {"name", "row_count", "content_sha256"},
                "table evidence",
            )
        return cls(
            schema_sha256=_required_string(value, "schema_sha256"),
            tables=tuple(
                TableEvidence(
                    name=_required_string(item, "name"),
                    row_count=_required_int(item, "row_count"),
                    content_sha256=_required_string(item, "content_sha256"),
                )
                for item in tables
            ),
            quick_check=_required_string(value, "quick_check"),
            integrity_check=_required_string(value, "integrity_check"),
            foreign_key_violations=_required_int(value, "foreign_key_violations"),
            application_id=_required_int(value, "application_id"),
            user_version=_required_int(value, "user_version"),
        )


@dataclass(frozen=True, slots=True)
class MaintenanceManifest:
    """Portable manifest for one verified backup or clean-room restore."""

    manifest_version: int
    operation: str
    created_at: str
    artifact_sha256: str
    artifact_size: int
    evidence: DatabaseEvidence
    input_artifact_sha256: str | None = None
    contains_secrets: bool = False
    mutated_authoritative_state: bool = False
    authorizes_startup: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MaintenanceManifest":
        _require_exact_keys(
            value,
            {
                "manifest_version",
                "operation",
                "created_at",
                "artifact_sha256",
                "artifact_size",
                "evidence",
                "input_artifact_sha256",
                "contains_secrets",
                "mutated_authoritative_state",
                "authorizes_startup",
            },
            "maintenance manifest",
        )
        evidence = value.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("manifest evidence must be an object")
        manifest = cls(
            manifest_version=_required_int(value, "manifest_version"),
            operation=_required_string(value, "operation"),
            created_at=_required_string(value, "created_at"),
            artifact_sha256=_required_string(value, "artifact_sha256"),
            artifact_size=_required_int(value, "artifact_size"),
            evidence=DatabaseEvidence.from_mapping(evidence),
            input_artifact_sha256=_optional_string(value, "input_artifact_sha256"),
            contains_secrets=_required_bool(value, "contains_secrets"),
            mutated_authoritative_state=_required_bool(value, "mutated_authoritative_state"),
            authorizes_startup=_required_bool(value, "authorizes_startup"),
        )
        if manifest.manifest_version != 1:
            raise ValueError("unsupported maintenance manifest version")
        if manifest.operation not in {"backup", "restore"}:
            raise ValueError("unsupported maintenance manifest operation")
        if (
            manifest.contains_secrets
            or manifest.mutated_authoritative_state
            or manifest.authorizes_startup
        ):
            raise ValueError("maintenance manifests cannot carry authority or secrets")
        return manifest


@dataclass(frozen=True, slots=True)
class MigrationDryRunReport:
    """Result of applying a migration only to a disposable snapshot."""

    report_version: int
    migration_id: str
    created_at: str
    outcome: str
    before: DatabaseEvidence
    after: DatabaseEvidence
    source_unchanged: bool
    target_artifact_sha256: str
    error_code: str | None = None
    contains_secrets: bool = False
    mutated_authoritative_state: bool = False
    authorizes_startup: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be null or a non-empty string")
    return item


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{key} must be an integer")
    return item


def _required_bool(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValueError(f"{key} must be a boolean")
    return item


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has missing or unknown fields")
