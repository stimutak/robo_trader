"""Canonical hashing primitives for offline backtest provenance.

These helpers have no runtime or broker authority.  They turn already-selected
offline inputs into deterministic SHA-256 identities and reject values that do
not have a stable, finite representation.
"""

import base64
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

_SHA256_HEX_LENGTH = 64
_INPUT_KINDS = frozenset({"data", "config", "code", "model", "package-lock", "result"})


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


@dataclass(frozen=True, order=True)
class ContentDigest:
    """A validated content identity."""

    algorithm: str
    hexdigest: str
    byte_length: int

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ValueError("only sha256 content digests are supported")
        if (
            not isinstance(self.hexdigest, str)
            or len(self.hexdigest) != _SHA256_HEX_LENGTH
            or self.hexdigest.lower() != self.hexdigest
            or any(character not in "0123456789abcdef" for character in self.hexdigest)
        ):
            raise ValueError("hexdigest must be a lowercase SHA-256 digest")
        if (
            isinstance(self.byte_length, bool)
            or not isinstance(self.byte_length, int)
            or self.byte_length < 0
        ):
            raise ValueError("byte_length must be a non-negative integer")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "hexdigest": self.hexdigest,
            "byte_length": self.byte_length,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContentDigest":
        _require_exact_keys(value, {"algorithm", "hexdigest", "byte_length"}, "digest")
        return cls(
            algorithm=value["algorithm"],
            hexdigest=value["hexdigest"],
            byte_length=value["byte_length"],
        )


@dataclass(frozen=True, order=True)
class HashedInput:
    """One purpose-labelled immutable backtest input."""

    kind: str
    identifier: str
    digest: ContentDigest

    def __post_init__(self) -> None:
        if self.kind not in _INPUT_KINDS - {"result"}:
            raise ValueError(f"unsupported input kind {self.kind!r}")
        _required_text(self.identifier, "input identifier")
        if not isinstance(self.digest, ContentDigest):
            raise TypeError("input digest must be a ContentDigest")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "digest": self.digest.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HashedInput":
        _require_exact_keys(value, {"kind", "identifier", "digest"}, "hashed input")
        return cls(
            kind=value["kind"],
            identifier=value["identifier"],
            digest=ContentDigest.from_dict(_mapping(value["digest"], "input digest")),
        )


def _require_exact_keys(value: Mapping[str, Any], expected: set, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, extra={extra}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def canonicalize(value: Any) -> Any:
    """Return a JSON-safe, type-preserving canonical value.

    Ambiguous containers, non-string mapping keys, non-finite numbers, naive
    datetimes, and arbitrary objects are rejected instead of stringified.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return {"$int": str(int(value))}
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("canonical payload cannot contain non-finite floats")
        return {"$float": numeric.hex()}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical payload cannot contain non-finite decimals")
        return {"$decimal": str(value.normalize())}
    if isinstance(value, pd.Timestamp):
        if pd.isna(value) or value.tzinfo is None:
            raise ValueError("timestamps must be finite and timezone-aware")
        return {"$datetime": value.isoformat()}
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetimes must be timezone-aware")
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical mappings require string keys")
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return {"$tuple": [canonicalize(item) for item in value]}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    raise TypeError(f"unsupported canonical payload type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one value to deterministic UTF-8 JSON."""

    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_bytes(payload: bytes) -> ContentDigest:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return ContentDigest("sha256", hashlib.sha256(payload).hexdigest(), len(payload))


def digest_json(value: Any) -> ContentDigest:
    return digest_bytes(canonical_json_bytes(value))


def digest_file(path: Path) -> ContentDigest:
    """Hash one stable regular file without following a symlink."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("path must identify a regular, non-symlink file")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("secure file hashing requires O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ValueError("path changed or could not be opened without following links") from exc
    hasher = hashlib.sha256()
    length = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("path must identify a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
                length += len(chunk)
            after = os.fstat(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ValueError("file changed while it was being hashed")
    if length != before.st_size:
        raise ValueError("file size changed while it was being hashed")
    return ContentDigest("sha256", hasher.hexdigest(), length)


def digest_file_set(paths: Iterable[Path], root: Path) -> ContentDigest:
    """Hash a labelled file set independent of caller iteration order."""

    root_path = Path(root).resolve(strict=True)
    entries = []
    seen = set()
    for supplied in paths:
        candidate = Path(supplied)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("file-set members must be regular, non-symlink files")
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError("file-set member escapes the declared root") from exc
        if relative in seen:
            raise ValueError(f"duplicate file-set member {relative}")
        seen.add(relative)
        entries.append((relative, digest_file(candidate).to_dict()))
    if not entries:
        raise ValueError("file set must not be empty")
    return digest_json({"files": sorted(entries)})


def digest_dataframe(frame: pd.DataFrame) -> ContentDigest:
    """Hash DataFrame values, labels, dtypes, and index metadata canonically."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must not be empty")
    if frame.index.has_duplicates:
        raise ValueError("frame index must be unique")
    if frame.columns.has_duplicates:
        raise ValueError("frame columns must be unique")
    if any(not isinstance(column, str) for column in frame.columns):
        raise ValueError("frame columns must be strings")

    if isinstance(frame.index, pd.MultiIndex):
        index_values: Sequence[Any] = [tuple(value) for value in frame.index.tolist()]
        index_names = list(frame.index.names)
    else:
        index_values = frame.index.tolist()
        index_names = [frame.index.name]
    payload = {
        "schema": "dataframe-sha256-v1",
        "index_names": index_names,
        "index": list(index_values),
        "columns": list(frame.columns),
        "dtypes": [str(frame[column].dtype) for column in frame.columns],
        "rows": frame.to_numpy(dtype=object).tolist(),
    }
    return digest_bytes(canonical_json_bytes(payload))


def hashed_json_input(kind: str, identifier: str, value: Any) -> HashedInput:
    return HashedInput(kind, identifier, digest_json(value))


def hashed_file_input(kind: str, identifier: str, path: Path) -> HashedInput:
    return HashedInput(kind, identifier, digest_file(path))


def hashed_dataframe_input(identifier: str, frame: pd.DataFrame) -> HashedInput:
    return HashedInput("data", identifier, digest_dataframe(frame))


__all__: Tuple[str, ...] = (
    "ContentDigest",
    "HashedInput",
    "canonical_json_bytes",
    "canonicalize",
    "digest_bytes",
    "digest_dataframe",
    "digest_file",
    "digest_file_set",
    "digest_json",
    "hashed_dataframe_input",
    "hashed_file_input",
    "hashed_json_input",
)
