#!/usr/bin/env python3
"""Generate paper safety identity and explicitly provision its empty journal.

This command never edits ``.env``.  Journal creation is a one-time, separately
confirmed operator action; normal trader startup only replays an existing
journal and will never create or repair one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.config import (  # noqa: E402
    RuntimeContract,
    _derive_safety_account_scope,
    load_runtime_contract_from_env,
)
from robo_trader.runtime_contract_constants import (  # noqa: E402
    PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
)
from robo_trader.runtime_lifecycle_lock import RuntimeLifecycleLock  # noqa: E402
from robo_trader.safety import (  # noqa: E402
    PaperExecutionIdentity,
    SafetyJournal,
    SafetyRuntimeCoordinator,
)
from robo_trader.utils.secure_config import ConfigValidationError  # noqa: E402

CREATE_CONFIRMATION = "CREATE-EMPTY-PAPER-SAFETY-JOURNAL"
MIGRATE_CONFIRMATION = "MIGRATE-EMPTY-LEGACY-PAPER-SAFETY-JOURNAL"
_ACCOUNT_SCOPE_RE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_SQLITE_RESOURCE_SUFFIXES = ("", "-wal", "-shm", "-journal")


@dataclass(frozen=True)
class EmptyLegacyJournalMigration:
    """Operator-facing result for one non-destructive journal cutover."""

    safety_account_scope_key: str
    safety_account_scope: str
    safety_journal_path: str
    receipt: str


def generate_account_scope(account: str) -> tuple[str, str]:
    """Return one fresh secret key and its exact account-bound scope."""

    normalized_account = str(account).strip()
    if not normalized_account:
        raise ValueError("IBKR_ACCOUNT is required to generate a bound scope")
    key = secrets.token_hex(32)
    return key, _derive_safety_account_scope(key, normalized_account)


def _resolved_environment() -> dict[str, str]:
    values = dotenv_values(PROJECT_ROOT / ".env")
    malformed = sorted(key for key, value in values.items() if value is None)
    if malformed:
        raise ValueError("malformed .env entries: " + ", ".join(malformed))

    resolved = {key: value for key, value in values.items() if value is not None}
    resolved.update(os.environ)
    return resolved


def _paper_contract(environ: Mapping[str, str]) -> RuntimeContract:
    contract = load_runtime_contract_from_env(environ, project_root=PROJECT_ROOT)
    if contract.execution_mode != "paper":
        raise ValueError("paper safety journal management requires EXECUTION_MODE=paper")
    if not contract.safety_journal_path:
        raise ValueError("SAFETY_JOURNAL_PATH is required")
    return contract


def _project_path(value: object) -> Path:
    """Anchor one configured path while preserving its lexical final leaf."""

    raw_path = str(value or "").strip()
    if not raw_path:
        raise ValueError("configured journal path cannot be empty")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.parent.resolve(strict=False) / path.name


def _sqlite_resource_family(path: Path) -> frozenset[str]:
    """Return normalized keys for a SQLite main path and its companions."""

    anchored = _project_path(path)
    return frozenset(
        unicodedata.normalize("NFC", os.fspath(Path(f"{anchored}{suffix}"))).casefold()
        for suffix in _SQLITE_RESOURCE_SUFFIXES
    )


def _assert_disjoint_sqlite_resource_families(*paths: Path) -> None:
    """Reject aliases between journals and the allocation-ledger namespace."""

    families = [_sqlite_resource_family(path) for path in paths]
    for index, family in enumerate(families):
        for other in families[index + 1 :]:
            if family & other:
                raise ValueError(
                    "legacy journal, migration target, and allocation ledger "
                    "SQLite resource families must be pairwise disjoint"
                )


def _configured_sqlite_path(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str | None = None,
) -> Path | None:
    """Resolve a configured SQLite path using the runtime contract's defaults."""

    raw_value = environ.get(name, default)
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        if default is None:
            return None
        raise ValueError(f"{name} cannot be empty")
    return _project_path(value)


def _read_regular_file(path: Path) -> tuple[bytes, tuple[int, int]]:
    """Read one exact non-symlink regular file through its owned descriptor."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("legacy safety journal cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("legacy safety journal must be a non-symlink regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)


def _assert_same_source(
    path: Path,
    expected_bytes: bytes,
    expected_identity: tuple[int, int],
) -> None:
    current_bytes, current_identity = _read_regular_file(path)
    if current_identity != expected_identity or current_bytes != expected_bytes:
        raise RuntimeError("legacy safety journal changed during migration")


def _migration_receipt(
    *,
    source_path: Path,
    source_bytes: bytes,
    source_identity: tuple[int, int],
    target_path: Path,
    target_bytes: bytes,
    target_identity: tuple[int, int],
) -> str:
    """Return a redacted, deterministic receipt without broker-account data."""

    payload = {
        "migration": "empty-legacy-paper-safety-journal-v1",
        "source_content_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_identity": f"{source_identity[0]}:{source_identity[1]}",
        "source_path_sha256": hashlib.sha256(str(source_path).encode()).hexdigest(),
        "target_content_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "target_identity": f"{target_identity[0]}:{target_identity[1]}",
        "target_path_sha256": hashlib.sha256(str(target_path).encode()).hexdigest(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return serialized


def migrate_empty_legacy_journal(
    environ: Mapping[str, str],
    *,
    target_path: str | Path,
    confirmation: str,
) -> EmptyLegacyJournalMigration:
    """Create a new account-bound journal without modifying the legacy source.

    This deliberately supports only the empty journals provisioned while PR2B1
    was dormant. Any event history requires a separately reviewed append-only
    schema migration; it is never rewritten, copied under a new identity, or
    silently discarded here.
    """

    if confirmation != MIGRATE_CONFIRMATION:
        raise ValueError(f"confirmation must be exactly {MIGRATE_CONFIRMATION}")

    lifecycle_lock = RuntimeLifecycleLock()
    if not lifecycle_lock.acquire():
        raise RuntimeError(
            "trader must be stopped and the runtime lifecycle lock must be available"
        )
    try:
        account = str(environ.get("IBKR_ACCOUNT", "")).strip()
        legacy_scope = str(environ.get("SAFETY_ACCOUNT_SCOPE", "")).strip()
        if not account:
            raise ValueError("IBKR_ACCOUNT is required for legacy journal migration")
        if not _ACCOUNT_SCOPE_RE.fullmatch(legacy_scope):
            raise ValueError("legacy SAFETY_ACCOUNT_SCOPE must be acct_v1_<64 lowercase hex>")
        configured_scope_key = str(environ.get("SAFETY_ACCOUNT_SCOPE_KEY", "")).strip()
        if configured_scope_key:
            try:
                derived_scope = _derive_safety_account_scope(
                    configured_scope_key,
                    account,
                )
            except ConfigValidationError as exc:
                raise ValueError(
                    "configured SAFETY_ACCOUNT_SCOPE_KEY is malformed; refusing legacy migration"
                ) from exc
            if secrets.compare_digest(derived_scope, legacy_scope):
                raise ValueError(
                    "configured safety journal is already account-bound; "
                    "verify and use the existing journal"
                )

        source_path = _project_path(environ.get("SAFETY_JOURNAL_PATH"))
        target = _project_path(target_path)
        allocation_ledger = _configured_sqlite_path(
            environ,
            "RT_DB_PATH",
            default="trading_data.db",
        )
        if allocation_ledger is None:
            raise RuntimeError("RT_DB_PATH resolution unexpectedly returned no path")
        protected_paths = [source_path, target, allocation_ledger]
        for name in ("LIVE_RT_DB_PATH", "LIVE_SAFETY_JOURNAL_PATH"):
            configured_path = _configured_sqlite_path(environ, name)
            if configured_path is not None:
                protected_paths.append(configured_path)
        _assert_disjoint_sqlite_resource_families(
            *protected_paths,
        )
        if source_path == target or source_path.resolve(strict=False) == target.resolve(
            strict=False
        ):
            raise ValueError("migration target must be distinct from the legacy journal")
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                "migration target already exists; refusing to overwrite or repair it"
            )
        if not target.parent.is_dir():
            raise FileNotFoundError(
                "migration target parent directory does not exist; create it explicitly"
            )

        source_bytes, source_identity = _read_regular_file(source_path)
        source_journal = SafetyJournal(source_path)
        source_state = source_journal.replay_and_bind_runtime_path(
            expected_execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
            expected_account_scope=legacy_scope,
        )
        if source_journal.runtime_path_identity != source_identity:
            raise RuntimeError("replayed legacy journal identity changed")
        if (
            source_state.events
            or source_state.reservations
            or source_state.active_reservations
            or source_state.quarantined_reservations
        ):
            raise RuntimeError(
                "legacy safety journal is not empty; refusing a history-dropping migration"
            )
        _assert_same_source(source_path, source_bytes, source_identity)

        scope_key, account_scope = generate_account_scope(account)
        candidate_environment = dict(environ)
        candidate_environment.update(
            {
                "SAFETY_ACCOUNT_SCOPE_KEY": scope_key,
                "SAFETY_ACCOUNT_SCOPE": account_scope,
                "SAFETY_JOURNAL_PATH": str(target),
            }
        )
        contract = load_runtime_contract_from_env(
            candidate_environment,
            project_root=PROJECT_ROOT,
        )
        if contract.safety_journal_path != str(target):
            raise RuntimeError("validated migration target identity changed")

        target_journal = SafetyJournal(target)
        target_state = target_journal.initialize_new(
            execution_domain_scope=contract.safety_execution_domain_scope,
            account_scope=contract.safety_account_scope,
        )
        if target_state.events or target_state.reservations:
            raise RuntimeError("new account-bound safety journal is not empty")

        target_bytes, target_identity = _read_regular_file(target)
        if target_journal.runtime_path_identity != target_identity:
            raise RuntimeError("replayed migration target identity changed")
        _assert_same_source(source_path, source_bytes, source_identity)
        _assert_same_source(target, target_bytes, target_identity)
        receipt = _migration_receipt(
            source_path=source_path,
            source_bytes=source_bytes,
            source_identity=source_identity,
            target_path=target,
            target_bytes=target_bytes,
            target_identity=target_identity,
        )
        _assert_same_source(source_path, source_bytes, source_identity)
        _assert_same_source(target, target_bytes, target_identity)
        return EmptyLegacyJournalMigration(
            safety_account_scope_key=scope_key,
            safety_account_scope=account_scope,
            safety_journal_path=str(target),
            receipt=receipt,
        )
    finally:
        lifecycle_lock.release()


def verify_journal(environ: Mapping[str, str]) -> RuntimeContract:
    """Read-only identity-bound replay with startup quarantine enforcement."""

    contract = _paper_contract(environ)
    identity = PaperExecutionIdentity(
        contract.safety_execution_domain_scope,
        contract.safety_account_scope,
    )
    SafetyRuntimeCoordinator(
        identity,
        SafetyJournal(contract.safety_journal_path),
    ).start()
    return contract


def initialize_journal(
    environ: Mapping[str, str],
    *,
    confirmation: str,
) -> RuntimeContract:
    """Create one new empty journal after exact typed confirmation."""

    if confirmation != CREATE_CONFIRMATION:
        raise ValueError(f"confirmation must be exactly {CREATE_CONFIRMATION}")
    contract = _paper_contract(environ)
    path = Path(contract.safety_journal_path).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError("configured safety journal already exists; refusing to modify it")
    if not path.parent.is_dir():
        raise FileNotFoundError(
            "safety journal parent directory does not exist; create and permission it explicitly"
        )
    journal = SafetyJournal(path)
    journal.initialize(
        execution_domain_scope=contract.safety_execution_domain_scope,
        account_scope=contract.safety_account_scope,
    )
    SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            contract.safety_execution_domain_scope,
            contract.safety_account_scope,
        ),
        journal,
    ).start()
    return contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "generate-scope",
        help=(
            "Print a new SAFETY_ACCOUNT_SCOPE_KEY and matching account-bound "
            "SAFETY_ACCOUNT_SCOPE; never edits .env."
        ),
    )
    subparsers.add_parser("verify", help="Read-only replay of the configured journal.")
    initialize = subparsers.add_parser(
        "initialize",
        help="Create a new empty configured journal; refuses any existing path.",
    )
    initialize.add_argument("--confirm", required=True, metavar=CREATE_CONFIRMATION)
    migrate = subparsers.add_parser(
        "migrate-empty-legacy",
        help=(
            "Create a distinct account-bound journal from an empty legacy journal; "
            "never edits or removes the source or .env."
        ),
    )
    migrate.add_argument("--target", required=True, type=Path)
    migrate.add_argument("--confirm", required=True, metavar=MIGRATE_CONFIRMATION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate-scope":
            environ = _resolved_environment()
            key, scope = generate_account_scope(environ.get("IBKR_ACCOUNT", ""))
            print(f"SAFETY_ACCOUNT_SCOPE_KEY={key}")
            print(f"SAFETY_ACCOUNT_SCOPE={scope}")
            return 0
        environ = _resolved_environment()
        if args.command == "verify":
            contract = verify_journal(environ)
            print("Paper safety journal verified: " f"{contract.safety_journal_identity}")
            return 0
        if args.command == "migrate-empty-legacy":
            migration = migrate_empty_legacy_journal(
                environ,
                target_path=args.target,
                confirmation=args.confirm,
            )
            print(f"SAFETY_ACCOUNT_SCOPE_KEY={migration.safety_account_scope_key}")
            print(f"SAFETY_ACCOUNT_SCOPE={migration.safety_account_scope}")
            print(f"SAFETY_JOURNAL_PATH={migration.safety_journal_path}")
            print(f"MIGRATION_RECEIPT={migration.receipt}")
            print("Legacy journal preserved unchanged; .env was not edited.")
            return 0
        contract = initialize_journal(environ, confirmation=args.confirm)
        print(
            "Created and verified empty paper safety journal: "
            f"{contract.safety_journal_identity}"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
