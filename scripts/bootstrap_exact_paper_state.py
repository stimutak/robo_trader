#!/usr/bin/env python3
"""Preview or apply one sealed exact paper-simulator accounting epoch.

This command never connects to IBKR and never places orders. The caller must
provide the reviewed reconciliation, zero-exposure broker snapshot, and every
protective mark artifact. ``preview`` is read-only. ``apply`` additionally
requires a stopped runtime, a destination-bound confirmation, and a new
descriptor-verified SQLite online backup before insert-only bootstrap writes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import sqlite3
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.config import RuntimeContract, load_runtime_contract_from_env  # noqa: E402
from robo_trader.database_async import AsyncTradingDatabase  # noqa: E402
from robo_trader.financial_state_bootstrap import (  # noqa: E402
    ExactStateBootstrapBackupReceipt,
    ExactStateBootstrapCandidate,
    ExactStateBootstrapError,
    ExactStateBootstrapEvidence,
    inspect_legacy_state,
    load_exact_state_bootstrap_evidence,
    sqlite_table_evidence,
)
from robo_trader.runtime_lifecycle_lock import RuntimeLifecycleLock  # noqa: E402
from robo_trader.safety.sqlite_identity import (  # noqa: E402
    SQLiteIdentityError,
    SQLitePathBinding,
    lexical_path_preserving_leaf,
    sqlite_connection_file_identity,
)

APPLY_CONFIRMATION_PREFIX = "APPLY_SEALED_EXACT_STATE_BOOTSTRAP"
_MAX_CANDIDATE_BYTES = 2 * 1024 * 1024


def _absolute_lexical_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ExactStateBootstrapError(f"{label} must be absolute")
    protected = lexical_path_preserving_leaf(path)
    if protected != path:
        raise ExactStateBootstrapError(f"{label} must use an existing canonical non-symlink parent")
    return protected


@dataclass(slots=True)
class _RegularFileBinding:
    path: Path
    descriptor: int
    device: int
    inode: int
    size: int
    modified_ns: int
    sealed: bool = False

    @classmethod
    def open_readonly(
        cls,
        path: Path,
        *,
        label: str,
        owner_only: bool = False,
    ) -> _RegularFileBinding:
        protected = _absolute_lexical_path(path, label)
        if not hasattr(os, "O_NOFOLLOW"):
            raise ExactStateBootstrapError("platform cannot reject symlink substitution")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(protected, flags)
        except OSError as exc:
            raise ExactStateBootstrapError(f"{label} cannot be opened safely") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ExactStateBootstrapError(f"{label} must be a single-link regular file")
            if metadata.st_uid != os.geteuid():
                raise ExactStateBootstrapError(f"{label} must be owned by this operator")
            if owner_only and stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ExactStateBootstrapError(f"{label} must be owner-only")
            binding = cls(
                path=protected,
                descriptor=descriptor,
                device=metadata.st_dev,
                inode=metadata.st_ino,
                size=metadata.st_size,
                modified_ns=metadata.st_mtime_ns,
            )
            binding.assert_identity()
            return binding
        except BaseException:
            os.close(descriptor)
            raise

    def assert_identity(self) -> None:
        try:
            guardian = os.fstat(self.descriptor)
            path_metadata = os.lstat(self.path)
        except OSError as exc:
            raise ExactStateBootstrapError(
                f"{self.path.name} identity is no longer authoritative"
            ) from exc
        expected = (self.device, self.inode, self.size, self.modified_ns)
        observed_guardian = (
            guardian.st_dev,
            guardian.st_ino,
            guardian.st_size,
            guardian.st_mtime_ns,
        )
        observed_path = (
            path_metadata.st_dev,
            path_metadata.st_ino,
            path_metadata.st_size,
            path_metadata.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(guardian.st_mode)
            or stat.S_ISLNK(path_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or guardian.st_nlink != 1
            or path_metadata.st_nlink != 1
            or (self.sealed and stat.S_IMODE(guardian.st_mode) & 0o222)
            or (self.sealed and stat.S_IMODE(path_metadata.st_mode) & 0o222)
            or observed_guardian != expected
            or observed_path != expected
        ):
            raise ExactStateBootstrapError(
                f"{self.path.name} changed after descriptor verification"
            )

    def read_bytes(self, *, maximum: int | None = None) -> bytes:
        self.assert_identity()
        if maximum is not None and self.size > maximum:
            raise ExactStateBootstrapError(f"{self.path.name} is too large")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = self.size
        while remaining:
            chunk = os.read(self.descriptor, min(remaining, 128 * 1024))
            if not chunk:
                raise ExactStateBootstrapError(f"{self.path.name} ended while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(self.descriptor, 1):
            raise ExactStateBootstrapError(f"{self.path.name} grew while reading")
        self.assert_identity()
        return b"".join(chunks)

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(slots=True)
class _CandidateBinding:
    file: _RegularFileBinding
    candidate: ExactStateBootstrapCandidate
    content_hash: str

    def assert_identity(self) -> None:
        self.file.assert_identity()
        current_hash = hashlib.sha256(
            self.file.read_bytes(maximum=_MAX_CANDIDATE_BYTES)
        ).hexdigest()
        if not hmac.compare_digest(current_hash, self.content_hash):
            raise ExactStateBootstrapError("candidate content changed after verification")

    def close(self) -> None:
        self.file.close()


def _open_candidate(path: Path) -> _CandidateBinding:
    binding = _RegularFileBinding.open_readonly(
        path,
        label="candidate",
        owner_only=True,
    )
    try:
        payload = binding.read_bytes(maximum=_MAX_CANDIDATE_BYTES)
        raw = json.loads(payload.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ExactStateBootstrapError("candidate document must be a JSON object")
        candidate = ExactStateBootstrapCandidate.from_mapping(raw)
        binding.assert_identity()
        return _CandidateBinding(binding, candidate, hashlib.sha256(payload).hexdigest())
    except (UnicodeError, json.JSONDecodeError) as exc:
        binding.close()
        raise ExactStateBootstrapError("candidate document cannot be read") from exc
    except BaseException:
        binding.close()
        raise


def _load_candidate(path: Path) -> ExactStateBootstrapCandidate:
    """Compatibility helper for read-only callers; apply retains a binding."""

    binding = _open_candidate(path)
    try:
        return binding.candidate
    finally:
        binding.close()


def _assert_stopped() -> None:
    lsof = Path("/usr/sbin/lsof")
    pgrep = Path("/usr/bin/pgrep")
    if not lsof.is_file() or not pgrep.is_file():
        raise RuntimeError("cannot prove the trader and Gateway are stopped")
    process_patterns = (
        r"(^|/)runner_async\.py([[:space:]]|$)",
        r"robo_trader\.runner_async",
        r"IB Gateway|ibgateway|tws\.jar",
    )
    for pattern in process_patterns:
        completed = subprocess.run(
            [str(pgrep), "-f", pattern],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            raise RuntimeError("trader and IBKR Gateway must remain stopped")
        if completed.returncode not in {0, 1}:
            raise RuntimeError("cannot prove process state")
    for port in (4001, 4002):
        completed = subprocess.run(
            [str(lsof), "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            raise RuntimeError("IBKR API listener must remain stopped")
        if completed.returncode not in {0, 1}:
            raise RuntimeError("cannot prove IBKR listener state")


@dataclass(slots=True)
class _OnlineBackup:
    receipt: ExactStateBootstrapBackupReceipt
    target_binding: _RegularFileBinding
    row_counts: tuple[tuple[str, int], ...]
    table_hashes: tuple[tuple[str, str], ...]

    def assert_restorable(self) -> None:
        """Re-prove the sealed backup, not only its inode metadata."""

        self.target_binding.assert_identity()
        current_hash = hashlib.sha256(self.target_binding.read_bytes()).hexdigest()
        if not hmac.compare_digest(current_hash, self.receipt.backup_content_hash):
            raise ExactStateBootstrapError("sealed backup content changed")
        state = inspect_legacy_state(self.target_binding.path)
        with sqlite3.connect(self.target_binding.path.as_uri() + "?mode=ro", uri=True) as conn:
            if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ExactStateBootstrapError("sealed backup integrity check failed")
            row_counts, table_hashes = sqlite_table_evidence(conn)
        self.target_binding.assert_identity()
        final_hash = hashlib.sha256(self.target_binding.read_bytes()).hexdigest()
        if (
            state["snapshot_hash"] != self.receipt.source_snapshot_hash
            or row_counts != self.receipt.row_counts
            or table_hashes != self.receipt.table_hashes
            or (state["database_device"], state["database_inode"])
            != (self.receipt.backup_device, self.receipt.backup_inode)
            or not hmac.compare_digest(final_hash, self.receipt.backup_content_hash)
        ):
            raise ExactStateBootstrapError("sealed backup no longer restores reviewed state")

    def assert_identity(self) -> None:
        """Compatibility alias retaining the stronger full verification."""

        self.assert_restorable()

    def report(self) -> dict[str, object]:
        return {
            "backup_path": self.receipt.backup_path,
            "backup_content_hash": self.receipt.backup_content_hash,
            "row_counts": dict(self.row_counts),
            "table_hashes": dict(self.table_hashes),
        }

    def close(self) -> None:
        self.target_binding.close()


def _reserve_backup_target(target: Path) -> _RegularFileBinding:
    protected = _absolute_lexical_path(target, "backup target")
    try:
        parent = os.lstat(protected.parent)
    except OSError as exc:
        raise RuntimeError("backup target parent cannot be inspected") from exc
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise RuntimeError("backup target parent must be a non-symlink directory")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(protected, flags, 0o600)
    except OSError as exc:
        raise RuntimeError("backup target must be a new exclusively-created file") from exc
    try:
        metadata = os.fstat(descriptor)
        binding = _RegularFileBinding(
            path=protected,
            descriptor=descriptor,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
        )
        binding.assert_identity()
        return binding
    except BaseException:
        os.close(descriptor)
        raise


def _refresh_binding_metadata(binding: _RegularFileBinding) -> None:
    metadata = os.fstat(binding.descriptor)
    binding.size = metadata.st_size
    binding.modified_ns = metadata.st_mtime_ns
    binding.assert_identity()


def _seal_binding_readonly(binding: _RegularFileBinding) -> None:
    """Remove write bits and replace the writable guardian with an O_RDONLY fd."""

    os.fsync(binding.descriptor)
    os.fchmod(binding.descriptor, 0o400)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    replacement = os.open(binding.path, flags)
    try:
        metadata = os.fstat(replacement)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (binding.device, binding.inode)
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise ExactStateBootstrapError("backup could not be sealed read-only")
    except BaseException:
        os.close(replacement)
        raise
    os.close(binding.descriptor)
    binding.descriptor = replacement
    binding.sealed = True
    _refresh_binding_metadata(binding)


def _make_backup_receipt(
    *,
    source: SQLitePathBinding,
    target: _RegularFileBinding,
    backup_content_hash: str,
    candidate: ExactStateBootstrapCandidate,
    row_counts: tuple[tuple[str, int], ...],
    table_hashes: tuple[tuple[str, str], ...],
) -> ExactStateBootstrapBackupReceipt:
    """Construct the core receipt from descriptor-bound backup evidence."""

    return ExactStateBootstrapBackupReceipt(
        schema_version=1,
        created_at=datetime.now(timezone.utc),
        candidate_fingerprint=candidate.fingerprint(),
        source_path=str(source.path),
        source_device=source.device,
        source_inode=source.inode,
        backup_path=str(target.path),
        backup_device=target.device,
        backup_inode=target.inode,
        integrity_check="ok",
        source_snapshot_hash=candidate.legacy_snapshot_hash,
        row_counts=row_counts,
        table_hashes=table_hashes,
        backup_content_hash=backup_content_hash,
    )


def _online_backup(
    source_path: Path,
    target_path: Path,
    candidate: ExactStateBootstrapCandidate,
) -> _OnlineBackup:
    source_metadata = os.lstat(source_path)
    if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_nlink != 1:
        raise ExactStateBootstrapError("backup source must be a single-link regular file")
    source = SQLitePathBinding.open_readonly(source_path)
    target: _RegularFileBinding | None = None
    source_connection: sqlite3.Connection | None = None
    target_connection: sqlite3.Connection | None = None
    try:
        target = _reserve_backup_target(target_path)
        source.assert_path_identity()
        if os.lstat(source.path).st_nlink != 1:
            raise ExactStateBootstrapError("backup source link count changed")
        source_connection = sqlite3.connect(source.path.as_uri() + "?mode=ro", uri=True)
        source = source.bind_sqlite_connection(sqlite_connection_file_identity(source_connection))
        source.assert_connection_identity(sqlite_connection_file_identity(source_connection))
        if os.lstat(source.path).st_nlink != 1:
            raise ExactStateBootstrapError("backup source link count changed")

        target.assert_identity()
        target_connection = sqlite3.connect(target.path.as_uri() + "?mode=rw", uri=True)
        target_sqlite_identity = sqlite_connection_file_identity(target_connection)
        if (target_sqlite_identity.device, target_sqlite_identity.inode) != (
            target.device,
            target.inode,
        ):
            raise SQLiteIdentityError("backup guardian and SQLite descriptor identities differ")
        target.assert_identity()
        source_connection.backup(target_connection)
        target_connection.commit()
        source.assert_connection_identity(sqlite_connection_file_identity(source_connection))
        current_target_identity = sqlite_connection_file_identity(target_connection)
        if current_target_identity.file_descriptor != target_sqlite_identity.file_descriptor or (
            current_target_identity.device,
            current_target_identity.inode,
        ) != (target.device, target.inode):
            raise SQLiteIdentityError("backup SQLite descriptor changed during copy")
        integrity = target_connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise RuntimeError("bootstrap backup failed SQLite integrity verification")
        row_counts, table_hashes = sqlite_table_evidence(target_connection)
        target_connection.close()
        target_connection = None
        _seal_binding_readonly(target)
        backup_content_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        receipt = _make_backup_receipt(
            source=source,
            target=target,
            backup_content_hash=backup_content_hash,
            candidate=candidate,
            row_counts=row_counts,
            table_hashes=table_hashes,
        )
        return _OnlineBackup(receipt, target, row_counts, table_hashes)
    except BaseException:
        if target_connection is not None:
            target_connection.close()
        if target is not None:
            target.close()
        # Deliberately leave the exclusive target path for forensic inspection.
        raise
    finally:
        if source_connection is not None:
            source_connection.close()
        source.close()


def _validate_evidence_binding(
    candidate: ExactStateBootstrapCandidate,
    evidence: ExactStateBootstrapEvidence,
    runtime_contract: RuntimeContract,
) -> None:
    expected = (
        candidate.reconciliation_snapshot_id,
        candidate.reconciliation_report_hash,
        candidate.broker_snapshot_hash,
        candidate.legacy_snapshot_hash,
        candidate.execution_domain_scope,
        candidate.account_scope,
        candidate.database_path,
        candidate.database_identity,
    )
    actual = (
        evidence.reconciliation_snapshot_id,
        evidence.reconciliation_report_hash,
        evidence.broker_snapshot_hash,
        evidence.legacy_snapshot_hash,
        evidence.execution_domain_scope,
        evidence.account_scope,
        evidence.database_path,
        evidence.database_identity,
    )
    if actual != expected:
        raise ExactStateBootstrapError("candidate does not match the reviewed evidence")
    if (
        evidence.runtime_fingerprint != runtime_contract.fingerprint
        or candidate.portfolio_id not in evidence.portfolio_ids
    ):
        raise ExactStateBootstrapError("evidence does not cover this runtime and portfolio")


def preview(
    candidate: ExactStateBootstrapCandidate,
    database_path: Path,
    evidence: ExactStateBootstrapEvidence | None = None,
    runtime_contract: RuntimeContract | None = None,
) -> dict[str, object]:
    if Path(candidate.database_path) != database_path:
        raise ExactStateBootstrapError("candidate database path does not match --db-path")
    if evidence is not None:
        if runtime_contract is None:
            raise ExactStateBootstrapError("runtime contract is required with evidence")
        _validate_evidence_binding(candidate, evidence, runtime_contract)
    legacy = inspect_legacy_state(database_path)
    if legacy["snapshot_hash"] != candidate.legacy_snapshot_hash:
        raise ExactStateBootstrapError("legacy ledger differs from the reviewed candidate")
    actual = {
        (row["portfolio_id"], row["symbol"]): int(row["quantity"])
        for row in legacy["position_rows"]
        if row["portfolio_id"] == candidate.portfolio_id
    }
    expected = {
        (candidate.portfolio_id, position.symbol): position.quantity
        for position in candidate.positions
    }
    if actual != expected:
        raise ExactStateBootstrapError(
            "candidate does not cover every nonzero position in its portfolio"
        )
    return {
        "authorizes_startup": False,
        "bootstrap_id": candidate.bootstrap_id,
        "candidate_fingerprint": candidate.fingerprint(),
        "legacy_snapshot_hash": candidate.legacy_snapshot_hash,
        "position_count": len(candidate.positions),
        "schema_version": 1,
        "status": "READY_FOR_OFFLINE_APPLY",
    }


async def _apply(
    candidate: ExactStateBootstrapCandidate,
    database_path: Path,
    operator_reason: str,
    *,
    evidence: ExactStateBootstrapEvidence,
    backup_receipt: ExactStateBootstrapBackupReceipt,
    runtime_contract: RuntimeContract,
) -> dict[str, object]:
    database = AsyncTradingDatabase(database_path)
    try:
        await database.initialize()
        receipt = await database.apply_exact_state_bootstrap(
            candidate,
            evidence=evidence,
            backup_receipt=backup_receipt,
            operator_reason=operator_reason,
            runtime_contract=runtime_contract,
        )
    finally:
        await database.close()
    return {
        "authorizes_startup": False,
        "bootstrap_id": receipt.bootstrap_id,
        "candidate_fingerprint": receipt.candidate_fingerprint,
        "committed_at": receipt.committed_at.isoformat(),
        "database_device": receipt.database_device,
        "database_inode": receipt.database_inode,
        "operator_action_id": receipt.operator_action_id,
        "schema_version": 1,
        "status": "BOOTSTRAPPED_GATE_A_STILL_CLOSED",
    }


def _required_confirmation(
    candidate: ExactStateBootstrapCandidate,
    runtime_contract: RuntimeContract,
    backup_path: Path,
) -> str:
    protected_backup = _absolute_lexical_path(backup_path, "backup target")
    return (
        f"{APPLY_CONFIRMATION_PREFIX} candidate={candidate.fingerprint()} "
        f"database={runtime_contract.database_identity} backup={protected_backup}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "apply"):
        child = subparsers.add_parser(command)
        child.add_argument("--candidate", type=Path, required=True)
        child.add_argument("--db-path", type=Path, required=True)
        child.add_argument("--reconciliation-evidence", type=Path, required=True)
        child.add_argument("--broker-snapshot", type=Path, required=True)
        child.add_argument(
            "--protective-mark",
            type=Path,
            action="append",
            default=[],
            dest="protective_marks",
        )
        child.add_argument("--json", action="store_true", required=True)
        if command == "apply":
            child.add_argument("--backup-path", type=Path, required=True)
            child.add_argument("--reason", required=True)
            child.add_argument("--confirm", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    candidate_binding: _CandidateBinding | None = None
    backup: _OnlineBackup | None = None
    try:
        args = _parser().parse_args(argv)
        database_path = _absolute_lexical_path(args.db_path, "--db-path")
        runtime_contract = load_runtime_contract_from_env(project_root=PROJECT_ROOT)
        if Path(runtime_contract.database_path) != database_path:
            raise ExactStateBootstrapError("--db-path does not match the sealed runtime contract")
        candidate_binding = _open_candidate(args.candidate)
        candidate = candidate_binding.candidate
        evidence = load_exact_state_bootstrap_evidence(
            reconciliation_path=args.reconciliation_evidence,
            broker_snapshot_path=args.broker_snapshot,
            protective_mark_paths=args.protective_marks,
            expected_runtime_contract=runtime_contract,
        )
        candidate_binding.assert_identity()
        report = preview(candidate, database_path, evidence, runtime_contract)
        if args.command == "apply":
            required_confirmation = _required_confirmation(
                candidate,
                runtime_contract,
                args.backup_path,
            )
            if not hmac.compare_digest(args.confirm, required_confirmation):
                raise ExactStateBootstrapError(
                    f"confirmation must be exactly {required_confirmation}"
                )
            if len(args.reason.strip()) < 10:
                raise ExactStateBootstrapError("--reason must be a specific sentence")
            lock = RuntimeLifecycleLock()
            if not lock.acquire():
                raise RuntimeError("runtime lifecycle lock is already held")
            try:
                candidate_binding.assert_identity()
                _assert_stopped()
                backup = _online_backup(database_path, args.backup_path, candidate)
                candidate_binding.assert_identity()
                backup.assert_restorable()
                report = asyncio.run(
                    _apply(
                        candidate,
                        database_path,
                        args.reason.strip(),
                        evidence=evidence,
                        backup_receipt=backup.receipt,
                        runtime_contract=runtime_contract,
                    )
                )
                candidate_binding.assert_identity()
                backup.assert_restorable()
                report["backup"] = backup.report()
            finally:
                lock.release()
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "authorizes_startup": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "schema_version": 1,
                    "status": "BLOCKED",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    finally:
        if backup is not None:
            backup.close()
        if candidate_binding is not None:
            candidate_binding.close()


if __name__ == "__main__":
    raise SystemExit(main())
