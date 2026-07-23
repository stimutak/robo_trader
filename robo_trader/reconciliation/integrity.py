"""Before/after evidence hashing for the non-mutating diagnostic."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

from .errors import IntegrityViolation


@dataclass(frozen=True)
class FileFingerprint:
    state: str
    size: int
    sha256: str


def _fingerprint(path: Path) -> FileFingerprint:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return FileFingerprint("absent", 0, "")
    except OSError as exc:
        raise IntegrityViolation("protected evidence metadata cannot be read") from exc

    if stat.S_ISLNK(metadata.st_mode):
        try:
            target = os.readlink(path)
        except OSError as exc:
            raise IntegrityViolation("protected evidence symlink cannot be read") from exc
        return FileFingerprint(
            "symlink",
            metadata.st_size,
            hashlib.sha256(target.encode("utf-8", errors="surrogateescape")).hexdigest(),
        )
    if not stat.S_ISREG(metadata.st_mode):
        return FileFingerprint("other", metadata.st_size, "")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise IntegrityViolation("protected evidence file cannot be hashed") from exc
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise IntegrityViolation("protected evidence changed while being hashed")
    return FileFingerprint("regular", metadata.st_size, digest.hexdigest())


def protected_evidence_paths(
    project_root: Path, resolved_env: Mapping[str, str]
) -> tuple[Path, ...]:
    db_value = str(resolved_env.get("RT_DB_PATH", "trading_data.db")).strip()
    db_path = Path(db_value).expanduser()
    if not db_path.is_absolute():
        db_path = project_root / db_path

    configured_log = str(resolved_env.get("LOG_FILE", "")).strip()
    log_paths = [project_root / "robo_trader.log"]
    if configured_log:
        configured_path = Path(configured_log).expanduser()
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        log_paths.append(configured_path)

    base_paths = [
        db_path,
        Path(f"{db_path}-wal"),
        Path(f"{db_path}-shm"),
        Path(f"{db_path}-journal"),
        project_root / "data" / "kill_switch_state.json",
        project_root / "data" / "kill_switch.lock",
        project_root / "data" / "preflight_bypass.log",
        project_root / "data" / ".preflight_last_ok",
        project_root / ".env",
        project_root / "config" / "ibc" / "config.ini",
    ]
    for log_path in log_paths:
        base_paths.append(log_path)
        base_paths.extend(Path(f"{log_path}.{index}") for index in range(1, 6))

    unique: dict[str, Path] = {}
    for path in base_paths:
        # Do not resolve symlinks here.  The link itself is protected evidence:
        # protecting only the current target would allow the configured path to
        # be swapped during collection.  Protect both identities so target
        # content changes remain detectable too.
        absolute = Path(os.path.abspath(os.fspath(path)))
        unique[str(absolute)] = absolute
        resolved = absolute.resolve(strict=False)
        unique[str(resolved)] = resolved
    return tuple(unique[key] for key in sorted(unique))


class EvidenceIntegrityGuard:
    """Raise if any protected path changes, appears, or disappears."""

    def __init__(self, paths: Iterable[Path]):
        self.paths = tuple(paths)
        self.before: dict[Path, FileFingerprint] = {}
        self.after: dict[Path, FileFingerprint] = {}

    def __enter__(self) -> "EvidenceIntegrityGuard":
        self.before = {path: _fingerprint(path) for path in self.paths}
        return self

    def verify_unchanged(self) -> None:
        self.after = {path: _fingerprint(path) for path in self.paths}
        if self.after != self.before:
            raise IntegrityViolation(
                "protected database, safety, audit, or logging evidence changed"
            )

    def __exit__(self, exc_type, exc, traceback) -> Literal[False]:
        try:
            self.verify_unchanged()
        except IntegrityViolation as integrity_error:
            if exc is not None:
                raise integrity_error from exc
            raise
        return False
