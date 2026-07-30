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
import math
import os
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.accounting.fifo_runtime import (  # noqa: E402
    RuntimePaperFillEvidence,
    reduction_side_to_fifo,
    verify_runtime_fill_in_transaction,
)
from robo_trader.config import (  # noqa: E402
    RuntimeContract,
    _derive_safety_account_scope,
    load_runtime_contract_from_env,
)
from robo_trader.paper_terminal_settlement import (  # noqa: E402
    PaperTerminalSettlementError,
    PaperTerminalSettlementRequest,
)
from robo_trader.runtime_contract_constants import (  # noqa: E402
    PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
)
from robo_trader.runtime_lifecycle_lock import RuntimeLifecycleLock  # noqa: E402
from robo_trader.safety import (  # noqa: E402
    LocalPaperTerminalEvidence,
    PaperExecutionIdentity,
    ReplayReservation,
    SafetyJournal,
    SafetyRuntimeCoordinator,
    TerminalOrderStatus,
    canonical_json,
    decimal_to_fixed,
)
from robo_trader.safety.models import parse_utc_text  # noqa: E402
from robo_trader.safety.sqlite_identity import (  # noqa: E402
    SQLiteIdentityError,
    sqlite_connection_file_identity,
)
from robo_trader.utils.secure_config import ConfigValidationError  # noqa: E402

CREATE_CONFIRMATION = "CREATE-EMPTY-PAPER-SAFETY-JOURNAL"
MIGRATE_CONFIRMATION = "MIGRATE-EMPTY-LEGACY-PAPER-SAFETY-JOURNAL"
RECOVER_CONFIRMATION = "RECOVER-EXACT-LOCAL-PAPER-SETTLEMENT"
_ACCOUNT_SCOPE_RE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_SQLITE_RESOURCE_SUFFIXES = ("", "-wal", "-shm", "-journal")
_STATUS_SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

_SETTLEMENT_PROJECTION_SCHEMA = {
    "paper_reduction_settlements": {
        "protective_quote_payload",
    },
    "trades": {
        "id",
        "portfolio_id",
        "symbol",
        "side",
        "quantity",
        "price",
        "notional",
        "slippage",
        "commission",
        "pnl",
        "timestamp",
    },
    "positions": {
        "market_price",
        "portfolio_id",
        "symbol",
        "quantity",
        "avg_cost",
    },
    "account": {
        "portfolio_id",
        "cash",
        "daily_pnl",
        "realized_pnl",
        "timestamp",
        "unrealized_pnl",
    },
    "paper_account_settlement_state": {
        "portfolio_id",
        "cash_text",
        "daily_pnl_baseline_text",
        "daily_pnl_date",
        "daily_pnl_text",
        "realized_pnl_text",
        "updated_at",
        "source_settlement_id",
    },
    "paper_position_settlement_state": {
        "cost_basis_text",
        "mark_price_text",
        "portfolio_id",
        "source_settlement_id",
        "symbol",
        "updated_at",
    },
    "paper_fifo_settlement_links": {
        "commission_currency",
        "commission_minor",
        "commission_source",
        "epoch_id",
        "event_sequence",
        "execution_id",
        "fifo_state_fingerprint",
        "fill_id",
        "request_fingerprint",
        "settlement_id",
    },
}


@dataclass(frozen=True)
class EmptyLegacyJournalMigration:
    """Operator-facing result for one non-destructive journal cutover."""

    safety_account_scope_key: str
    safety_account_scope: str
    safety_journal_path: str
    receipt: str


@dataclass(frozen=True)
class LocalSettlementCorrelation:
    """One query-only correlation result for an unresolved reservation."""

    status: str
    evidence: LocalPaperTerminalEvidence | None = None


@dataclass(frozen=True)
class OfflineRecoveryResult:
    """Redacted result of one append-only offline recovery."""

    journal_identity: str
    reservation_id_sha256: str
    claim_id_sha256: str
    order_ref_sha256: str
    terminal_sequence: int


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


def _redacted_identifier(kind: str, value: str) -> str:
    """Hash one identifier with a domain separator; never return its raw value."""

    payload = f"paper-safety-status-v1\x00{kind}\x00{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular_file_identity(path: Path, label: str) -> tuple[int, int]:
    """Inspect a configured database path without following its final leaf."""

    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} cannot be inspected safely") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a non-symlink regular file")
    return metadata.st_dev, metadata.st_ino


def _verified_request_for_reservation(
    payload_json: object,
    request_fingerprint: object,
    reservation: ReplayReservation,
) -> PaperTerminalSettlementRequest | None:
    """Rebuild one exact canonical request bound to the journal reservation."""

    if not isinstance(payload_json, str) or not isinstance(request_fingerprint, str):
        return None
    if not _DIGEST_RE.fullmatch(request_fingerprint):
        return None
    if not secrets.compare_digest(
        hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        request_fingerprint,
    ):
        return None
    try:
        request = PaperTerminalSettlementRequest.from_canonical_payload(payload_json)
    except PaperTerminalSettlementError:
        return None
    claim_id = getattr(reservation, "claim_id", None)
    claim_sequence = getattr(reservation, "claim_sequence", None)
    descriptor_fingerprint = getattr(
        reservation,
        "submission_descriptor_fingerprint",
        None,
    )
    order_ref = getattr(reservation, "order_ref", None)
    exact_pairs = (
        (request.execution_domain_scope, reservation.execution_domain_scope),
        (request.account_scope, reservation.account_scope),
        (request.portfolio_id, reservation.portfolio_id),
        (request.con_id, reservation.con_id),
        (request.symbol, reservation.symbol),
        (request.reservation_id, reservation.reservation_id),
        (request.claim_id, claim_id),
        (request.claim_sequence, claim_sequence),
        (request.submission_descriptor_fingerprint, descriptor_fingerprint),
        (request.order_ref, order_ref),
        (request.side.value, reservation.side.value),
        (
            decimal_to_fixed(request.requested_quantity),
            decimal_to_fixed(reservation.quantity),
        ),
    )
    if not all(actual == expected for actual, expected in exact_pairs):
        return None
    return request


def _settlement_projection_schema_available(connection: sqlite3.Connection) -> bool:
    """Return whether every immutable-settlement projection can be inspected."""

    for table, required_columns in _SETTLEMENT_PROJECTION_SCHEMA.items():
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        if not rows or not required_columns.issubset({row[1] for row in rows}):
            return False
    return True


def _float_projection_matches(actual: object, exact: Decimal) -> bool:
    """Match one legacy REAL value to the deterministic float written at commit."""

    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    try:
        expected = float(exact)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(float(actual)) and math.isfinite(expected) and actual == expected


def _legacy_timestamp_not_after(value: object, upper_bound: datetime) -> bool:
    """Validate either canonical UTC or SQLite CURRENT_TIMESTAMP text."""

    try:
        observed = parse_utc_text(value, "account timestamp")
    except (TypeError, ValueError):
        if type(value) is not str:
            return False
        try:
            observed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return False
    return observed <= upper_bound


def _linked_settlement_projection_matches(
    connection: sqlite3.Connection,
    *,
    settlement_id: object,
    trade_id: object,
    committed_at: object,
    protective_quote_payload: object,
    request: PaperTerminalSettlementRequest,
) -> bool:
    """Prove the outbox row's exact atomic trade/position/account effects.

    This is deliberately stricter than receipt authentication. A receipt can
    be copied or forged into a partial database; recovery is permitted only
    when every deterministic projection written by the settlement transaction
    still agrees in the same query-only SQLite snapshot.
    """

    if (
        type(settlement_id) is not str
        or type(committed_at) is not str
        or protective_quote_payload != request.protective_quote_payload
        or request.expected_position_cost_basis is None
    ):
        return False

    if request.filled_quantity > 0:
        if (
            type(trade_id) is not int
            or trade_id <= 0
            or request.fill_price is None
            or request.fill_commission_minor is None
        ):
            return False
        trade_rows = connection.execute(
            """
            SELECT portfolio_id, symbol, side, quantity, typeof(quantity),
                   price, notional, slippage, commission, pnl, timestamp
            FROM trades WHERE id = ?
            """,
            (trade_id,),
        ).fetchall()
        if len(trade_rows) != 1:
            return False
        (
            trade_portfolio,
            trade_symbol,
            trade_side,
            trade_quantity,
            trade_quantity_type,
            trade_price,
            trade_notional,
            trade_slippage,
            trade_commission,
            trade_pnl,
            trade_timestamp,
        ) = trade_rows[0]
        exact_notional = request.fill_price * request.filled_quantity
        exact_commission = Decimal(request.fill_commission_minor) / Decimal("100")
        exact_pnl = request.expected_post_realized_pnl - request.expected_pre_realized_pnl
        trade_matches = (
            trade_portfolio == request.portfolio_id
            and trade_symbol == request.symbol
            and trade_side == request.side.value
            and trade_quantity_type == "integer"
            and type(trade_quantity) is int
            and trade_quantity == int(request.filled_quantity)
            and _float_projection_matches(trade_price, request.fill_price)
            and _float_projection_matches(trade_notional, exact_notional)
            and _float_projection_matches(trade_slippage, Decimal(0))
            and _float_projection_matches(trade_commission, exact_commission)
            and _float_projection_matches(trade_pnl, exact_pnl)
            and trade_timestamp == committed_at
        )
        if not trade_matches:
            return False

        matching_trade_count = connection.execute(
            """
            SELECT COUNT(*) FROM trades
            WHERE portfolio_id = ? AND symbol = ? AND side = ? AND quantity = ?
              AND price = ? AND notional = ? AND slippage = 0 AND commission = ?
              AND pnl = ? AND timestamp = ?
            """,
            (
                request.portfolio_id,
                request.symbol,
                request.side.value,
                int(request.filled_quantity),
                float(request.fill_price),
                float(exact_notional),
                float(exact_commission),
                float(exact_pnl),
                committed_at,
            ),
        ).fetchone()
        if matching_trade_count != (1,):
            return False
    else:
        if (
            trade_id is not None
            or request.fill_price is not None
            or request.terminal_status
            not in {
                TerminalOrderStatus.CANCELLED,
                TerminalOrderStatus.REJECTED,
                TerminalOrderStatus.EXPIRED,
                TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
            }
            or request.expected_pre_position_quantity != request.expected_post_position_quantity
            or request.expected_pre_aggregate_quantity != request.expected_post_aggregate_quantity
            or request.expected_pre_cash != request.expected_post_cash
            or request.expected_pre_realized_pnl != request.expected_post_realized_pnl
            or request.expected_pre_daily_pnl != request.expected_post_daily_pnl
        ):
            return False
        # A zero-fill settlement must not have produced even an unlinked trade
        # row at its exact transaction timestamp for this allocation.
        if (
            connection.execute(
                """
            SELECT COUNT(*) FROM trades
            WHERE portfolio_id = ? AND symbol = ? AND timestamp = ?
            """,
                (request.portfolio_id, request.symbol, committed_at),
            ).fetchone()
            != (0,)
        ):
            return False

    position_rows = connection.execute(
        """
        SELECT quantity, typeof(quantity), avg_cost, market_price
        FROM positions WHERE portfolio_id = ? AND symbol = ?
        """,
        (request.portfolio_id, request.symbol),
    ).fetchall()
    if len(position_rows) != 1:
        return False
    position_quantity, position_quantity_type, avg_cost, market_price = position_rows[0]
    expected_mark_price = (
        request.protective_mark_price
        if request.filled_quantity > 0
        else request.expected_pre_position_mark_price
    )
    if (
        position_quantity_type != "integer"
        or type(position_quantity) is not int
        or position_quantity != int(request.expected_post_position_quantity)
        or not _float_projection_matches(avg_cost, request.expected_position_cost_basis)
        or (
            market_price is not None
            if expected_mark_price is None
            else not _float_projection_matches(market_price, expected_mark_price)
        )
    ):
        return False

    aggregate_rows = connection.execute(
        "SELECT quantity, typeof(quantity) FROM positions WHERE symbol = ?",
        (request.symbol,),
    ).fetchall()
    if any(
        storage_type != "integer" or type(quantity) is not int
        for quantity, storage_type in aggregate_rows
    ):
        return False
    if sum(quantity for quantity, _ in aggregate_rows) != int(
        request.expected_post_aggregate_quantity
    ):
        return False

    position_state_rows = connection.execute(
        """
        SELECT cost_basis_text, mark_price_text, source_settlement_id, updated_at
        FROM paper_position_settlement_state
        WHERE portfolio_id = ? AND symbol = ?
        """,
        (request.portfolio_id, request.symbol),
    ).fetchall()
    if len(position_state_rows) != 1:
        return False
    try:
        exact_cost_basis = Decimal(position_state_rows[0][0])
        exact_mark_price = (
            None if position_state_rows[0][1] is None else Decimal(position_state_rows[0][1])
        )
    except (TypeError, InvalidOperation):
        return False
    if (
        not exact_cost_basis.is_finite()
        or decimal_to_fixed(exact_cost_basis) != position_state_rows[0][0]
        or exact_cost_basis != request.expected_position_cost_basis
        or (exact_mark_price is not None and not exact_mark_price.is_finite())
        or (
            exact_mark_price is not None
            and decimal_to_fixed(exact_mark_price) != position_state_rows[0][1]
        )
        or exact_mark_price != expected_mark_price
    ):
        return False
    expected_position_source = (
        settlement_id
        if request.filled_quantity > 0
        else request.expected_pre_position_source_settlement_id
    )
    if position_state_rows[0][2] != expected_position_source:
        return False
    try:
        position_updated_at = parse_utc_text(
            position_state_rows[0][3],
            "position updated_at",
        )
        exact_committed_at = parse_utc_text(committed_at, "committed_at")
    except (TypeError, ValueError):
        return False
    if request.filled_quantity > 0:
        if position_state_rows[0][3] != committed_at:
            return False
    elif position_updated_at > exact_committed_at:
        return False

    account_state_rows = connection.execute(
        """
        SELECT cash_text, realized_pnl_text, daily_pnl_text,
               daily_pnl_baseline_text, daily_pnl_date, updated_at,
               source_settlement_id
        FROM paper_account_settlement_state WHERE portfolio_id = ?
        """,
        (request.portfolio_id,),
    ).fetchall()
    if len(account_state_rows) != 1:
        return False
    (
        cash_text,
        realized_pnl_text,
        daily_pnl_text,
        daily_pnl_baseline_text,
        daily_pnl_date,
        account_updated_at,
        source_settlement_id,
    ) = account_state_rows[0]
    try:
        exact_cash = Decimal(cash_text)
        exact_realized_pnl = Decimal(realized_pnl_text)
        exact_daily_pnl = Decimal(daily_pnl_text)
        exact_daily_pnl_baseline = Decimal(daily_pnl_baseline_text)
    except (TypeError, InvalidOperation):
        return False
    if (
        not exact_cash.is_finite()
        or not exact_realized_pnl.is_finite()
        or not exact_daily_pnl.is_finite()
        or not exact_daily_pnl_baseline.is_finite()
        or decimal_to_fixed(exact_cash) != cash_text
        or decimal_to_fixed(exact_realized_pnl) != realized_pnl_text
        or decimal_to_fixed(exact_daily_pnl) != daily_pnl_text
        or decimal_to_fixed(exact_daily_pnl_baseline) != daily_pnl_baseline_text
        or exact_cash != request.expected_post_cash
        or exact_realized_pnl != request.expected_post_realized_pnl
        or exact_daily_pnl != request.expected_post_daily_pnl
        or exact_daily_pnl_baseline != request.expected_daily_pnl_baseline
        or daily_pnl_date != request.expected_daily_pnl_date
        or source_settlement_id != settlement_id
    ):
        return False

    try:
        exact_account_updated_at = parse_utc_text(account_updated_at, "updated_at")
        exact_committed_at = parse_utc_text(committed_at, "committed_at")
        exact_daily_pnl_date = date.fromisoformat(daily_pnl_date)
    except (TypeError, ValueError):
        return False
    if (
        exact_daily_pnl_date.isoformat() != daily_pnl_date
        or exact_daily_pnl_date > exact_committed_at.date()
    ):
        return False
    if request.filled_quantity > 0:
        if account_updated_at != committed_at:
            return False
    elif exact_account_updated_at > exact_committed_at:
        return False

    account_rows = connection.execute(
        """
        SELECT cash, realized_pnl, daily_pnl, unrealized_pnl, timestamp
        FROM account WHERE portfolio_id = ?
        """,
        (request.portfolio_id,),
    ).fetchall()
    if len(account_rows) != 1 or not (
        _float_projection_matches(account_rows[0][0], request.expected_post_cash)
        and _float_projection_matches(account_rows[0][1], request.expected_post_realized_pnl)
        and _float_projection_matches(account_rows[0][2], request.expected_post_daily_pnl)
        and _float_projection_matches(
            account_rows[0][3],
            request.expected_post_daily_pnl
            + request.expected_daily_pnl_baseline
            - request.expected_post_realized_pnl,
        )
    ):
        return False
    if request.filled_quantity > 0:
        return account_rows[0][4] == committed_at
    return _legacy_timestamp_not_after(account_rows[0][4], exact_committed_at)


def _fifo_settlement_projection_matches(
    connection: sqlite3.Connection,
    *,
    settlement_id: object,
    request_fingerprint: object,
    request: PaperTerminalSettlementRequest,
) -> bool:
    """Authenticate the immutable FIFO event linked to one settlement."""

    if type(settlement_id) is not str or type(request_fingerprint) is not str:
        return False
    rows = connection.execute(
        """
        SELECT epoch_id,fill_id,event_sequence,execution_id,commission_minor,
               commission_currency,commission_source,fifo_state_fingerprint
        FROM paper_fifo_settlement_links WHERE settlement_id=?
        """,
        (settlement_id,),
    ).fetchall()
    if request.filled_quantity == 0:
        return not rows
    if (
        len(rows) != 1
        or request.fill_price is None
        or request.fill_execution_id is None
        or request.fill_commission_minor is None
        or request.fill_commission_currency is None
        or request.fill_commission_source is None
    ):
        return False
    (
        epoch_id,
        fill_id,
        event_sequence,
        execution_id,
        commission_minor,
        commission_currency,
        commission_source,
        state_fingerprint,
    ) = rows[0]
    try:
        evidence = RuntimePaperFillEvidence(
            execution_domain_scope=request.execution_domain_scope,
            account_scope=request.account_scope,
            portfolio_id=request.portfolio_id,
            con_id=request.con_id,
            symbol=request.symbol,
            side=reduction_side_to_fifo(request.side.value),
            quantity=request.filled_quantity,
            price=request.fill_price,
            execution_id=request.fill_execution_id,
            idempotency_key=request_fingerprint,
            commission_minor=request.fill_commission_minor,
            commission_currency=request.fill_commission_currency,
            commission_source=request.fill_commission_source,
            occurred_at=request.outcome_at,
        )
        projection = verify_runtime_fill_in_transaction(connection, evidence)
    except (sqlite3.Error, RuntimeError, TypeError, ValueError):
        return False
    return (
        epoch_id == projection.epoch_id
        and fill_id == projection.fill_id
        and event_sequence == projection.event_sequence
        and execution_id == request.fill_execution_id
        and commission_minor == request.fill_commission_minor
        and commission_currency == request.fill_commission_currency
        and commission_source == request.fill_commission_source
        and state_fingerprint == projection.state_fingerprint
        and projection.replayed
        and projection.signed_quantity == request.expected_post_position_quantity
        and (
            projection.average_cost == request.expected_position_cost_basis
            if projection.signed_quantity != 0
            else projection.average_cost is None and projection.open_cost is None
        )
        and projection.fill_realized_pnl
        == request.expected_post_realized_pnl - request.expected_pre_realized_pnl
        and projection.total_realized_pnl == request.expected_post_realized_pnl
    )


def _local_settlement_correlations(
    contract: RuntimeContract,
    reservations: tuple[ReplayReservation, ...],
) -> dict[str, LocalSettlementCorrelation]:
    """Verify exact persisted receipts through one query-only SQLite snapshot."""

    if not reservations:
        return {}
    ledger_path = _project_path(contract.database_path)
    expected_identity = _regular_file_identity(
        ledger_path,
        "paper allocation ledger",
    )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{ledger_path.as_uri()}?mode=ro",
            uri=True,
            isolation_level=None,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA foreign_keys").fetchone() != (1,):
            raise RuntimeError("paper allocation ledger foreign keys cannot be enforced")
        connection.execute("PRAGMA query_only = ON")
        if connection.execute("PRAGMA query_only").fetchone() != (1,):
            raise RuntimeError("paper allocation ledger query-only mode cannot be proven")
        descriptor_identity = sqlite_connection_file_identity(connection)
        if (descriptor_identity.device, descriptor_identity.inode) != expected_identity:
            raise RuntimeError("paper allocation ledger identity changed while opening")
        connection.execute("BEGIN")
        schema_row = connection.execute("""
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'paper_reduction_settlements'
            """).fetchone()
        if schema_row is None:
            return {
                reservation.reservation_id: LocalSettlementCorrelation("SCHEMA_MISSING")
                for reservation in reservations
            }
        if not _settlement_projection_schema_available(connection):
            return {
                reservation.reservation_id: LocalSettlementCorrelation("SCHEMA_MISSING")
                for reservation in reservations
            }

        correlations: dict[str, LocalSettlementCorrelation] = {}
        for reservation in reservations:
            if (
                reservation.claim_id is None
                or reservation.claim_sequence is None
                or reservation.submission_descriptor_fingerprint is None
                or reservation.order_ref is None
            ):
                correlations[reservation.reservation_id] = LocalSettlementCorrelation(
                    "JOURNAL_IDENTITY_INCOMPLETE"
                )
                continue
            rows = connection.execute(
                """
                SELECT settlement_id, execution_domain_scope, account_scope,
                       portfolio_id, con_id, symbol, reservation_id, claim_id,
                       order_ref, protective_quote_payload, request_fingerprint,
                       request_payload_json,
                       terminal_status, trade_id, database_path, database_identity,
                       database_device, database_inode, committed_at,
                       receipt_fingerprint, schema_version
                FROM paper_reduction_settlements
                WHERE reservation_id = ? OR claim_id = ? OR (
                    execution_domain_scope = ? AND account_scope = ? AND order_ref = ?
                )
                """,
                (
                    reservation.reservation_id,
                    reservation.claim_id,
                    reservation.execution_domain_scope,
                    reservation.account_scope,
                    reservation.order_ref,
                ),
            ).fetchall()
            if not rows:
                correlations[reservation.reservation_id] = LocalSettlementCorrelation("ABSENT")
                continue
            if len(rows) != 1:
                correlations[reservation.reservation_id] = LocalSettlementCorrelation(
                    "IDENTITY_CONFLICT"
                )
                continue
            (
                settlement_id,
                execution_domain_scope,
                account_scope,
                portfolio_id,
                con_id,
                symbol,
                reservation_id,
                claim_id,
                order_ref,
                protective_quote_payload,
                request_fingerprint,
                payload_json,
                terminal_status,
                trade_id,
                database_path,
                database_identity,
                database_device,
                database_inode,
                committed_at,
                receipt_fingerprint,
                schema_version,
            ) = rows[0]
            try:
                request = _verified_request_for_reservation(
                    payload_json,
                    request_fingerprint,
                    reservation,
                )
            except (AttributeError, InvalidOperation, ValueError):
                request = None
            try:
                committed = parse_utc_text(committed_at, "committed_at")
            except (TypeError, ValueError):
                committed = None
            receipt_payload = (
                canonical_json(
                    {
                        "committed_at": committed,
                        "database_device": database_device,
                        "database_identity": database_identity,
                        "database_inode": database_inode,
                        "database_path": database_path,
                        "request_fingerprint": request_fingerprint,
                        "schema_version": schema_version,
                        "settlement_id": settlement_id,
                        "trade_id": trade_id,
                    }
                )
                if committed is not None
                else ""
            )
            receipt_matches = (
                isinstance(receipt_fingerprint, str)
                and _DIGEST_RE.fullmatch(receipt_fingerprint) is not None
                and secrets.compare_digest(
                    hashlib.sha256(receipt_payload.encode("utf-8")).hexdigest(),
                    receipt_fingerprint,
                )
            )
            provenance_matches = (
                request is not None
                and execution_domain_scope == request.execution_domain_scope
                and account_scope == request.account_scope
                and portfolio_id == request.portfolio_id
                and con_id == request.con_id
                and symbol == request.symbol
                and reservation_id == request.reservation_id
                and claim_id == request.claim_id
                and order_ref == request.order_ref
                and database_path == str(ledger_path)
                and database_identity == contract.database_identity
                and database_device == expected_identity[0]
                and database_inode == expected_identity[1]
                and schema_version == 1
                and terminal_status == request.terminal_status.value
                and committed is not None
                and (
                    (request.filled_quantity > 0 and type(trade_id) is int and trade_id > 0)
                    or (request.filled_quantity == 0 and trade_id is None)
                )
                and _linked_settlement_projection_matches(
                    connection,
                    settlement_id=settlement_id,
                    trade_id=trade_id,
                    committed_at=committed_at,
                    protective_quote_payload=protective_quote_payload,
                    request=request,
                )
                and _fifo_settlement_projection_matches(
                    connection,
                    settlement_id=settlement_id,
                    request_fingerprint=request_fingerprint,
                    request=request,
                )
            )
            evidence = None
            if (
                request is not None
                and committed is not None
                and provenance_matches
                and receipt_matches
            ):
                try:
                    if committed < request.outcome_at:
                        raise ValueError("settlement predates terminal outcome")
                    evidence = LocalPaperTerminalEvidence(
                        execution_domain_scope=request.execution_domain_scope,
                        account_scope=request.account_scope,
                        portfolio_id=request.portfolio_id,
                        con_id=request.con_id,
                        symbol=request.symbol,
                        reservation_id=request.reservation_id,
                        claim_id=request.claim_id,
                        claim_sequence=request.claim_sequence,
                        submission_descriptor_fingerprint=(
                            request.submission_descriptor_fingerprint
                        ),
                        protective_quote_fingerprint=request.protective_quote_fingerprint,
                        order_ref=request.order_ref,
                        settlement_id=settlement_id,
                        settlement_request_fingerprint=request_fingerprint,
                        settlement_receipt_fingerprint=receipt_fingerprint,
                        database_path=database_path,
                        database_identity=database_identity,
                        database_device=database_device,
                        database_inode=database_inode,
                        committed_at=committed,
                        terminal_status=TerminalOrderStatus(terminal_status),
                        filled_quantity=request.filled_quantity,
                        remaining_quantity=request.remaining_quantity,
                        pre_position_quantity=request.expected_pre_position_quantity,
                        final_position_quantity=request.expected_post_position_quantity,
                        pre_aggregate_quantity=request.expected_pre_aggregate_quantity,
                        final_aggregate_quantity=request.expected_post_aggregate_quantity,
                        source="LOCAL_PAPER_SETTLEMENT_LEDGER",
                        schema_version=schema_version,
                    )
                except (TypeError, ValueError, PaperTerminalSettlementError):
                    evidence = None
            correlations[reservation.reservation_id] = LocalSettlementCorrelation(
                "MATCH" if evidence is not None else "MISMATCH",
                evidence,
            )

        final_descriptor_identity = sqlite_connection_file_identity(connection)
        if final_descriptor_identity != descriptor_identity:
            raise RuntimeError("paper allocation ledger descriptor identity changed")
        return correlations
    except (sqlite3.Error, SQLiteIdentityError) as exc:
        raise RuntimeError("paper allocation ledger cannot be inspected read-only") from exc
    finally:
        if connection is not None:
            connection.close()
        if _regular_file_identity(ledger_path, "paper allocation ledger") != expected_identity:
            raise RuntimeError("paper allocation ledger identity changed during status read")


def _local_settlement_statuses(
    contract: RuntimeContract,
    reservations: tuple[ReplayReservation, ...],
) -> dict[str, str]:
    """Return only public correlation codes for the read-only status report."""

    return {
        reservation_id: correlation.status
        for reservation_id, correlation in _local_settlement_correlations(
            contract,
            reservations,
        ).items()
    }


def _assert_gateway_stopped() -> None:
    """Prove trader/Gateway processes and IBKR API listeners are absent."""

    lsof = Path("/usr/sbin/lsof")
    pgrep = Path("/usr/bin/pgrep")
    if not lsof.is_file():
        raise RuntimeError("cannot prove Gateway is stopped because /usr/sbin/lsof is absent")
    if not pgrep.is_file():
        raise RuntimeError("cannot prove trader and Gateway are stopped because pgrep is absent")
    for pattern in (r"(^|/)runner_async\.py( |$)", r"IB Gateway|ibgateway|tws\.jar"):
        completed = subprocess.run(
            [str(pgrep), "-f", pattern],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            raise RuntimeError("trader and IBKR Gateway must remain stopped during recovery")
        if completed.returncode not in {0, 1}:
            raise RuntimeError("cannot prove trader and IBKR Gateway processes are stopped")
    for port in (4001, 4002):
        completed = subprocess.run(
            [str(lsof), "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            raise RuntimeError("IBKR Gateway must remain stopped during offline recovery")
        if completed.returncode not in {0, 1}:
            raise RuntimeError("cannot prove IBKR Gateway API ports are stopped")


def recover_exact_local_paper_settlement(
    environ: Mapping[str, str],
    *,
    confirmation: str,
) -> OfflineRecoveryResult:
    """Append one terminal release for one exact crash-after-commit receipt.

    The allocation ledger is opened query-only. No trade, position, account,
    settlement, broker, or process action is available on this path. The only
    permitted mutation is one terminal event in the configured safety journal.
    """

    if confirmation != RECOVER_CONFIRMATION:
        raise ValueError(f"confirmation must be exactly {RECOVER_CONFIRMATION}")

    lifecycle_lock = RuntimeLifecycleLock()
    if not lifecycle_lock.acquire():
        raise RuntimeError(
            "trader must be stopped and the runtime lifecycle lock must be available"
        )
    try:
        _assert_gateway_stopped()
        contract = _paper_contract(environ)
        journal_path = _project_path(contract.safety_journal_path)
        ledger_path = _project_path(contract.database_path)
        _assert_disjoint_sqlite_resource_families(journal_path, ledger_path)

        journal = SafetyJournal(journal_path)
        state = journal.replay_and_bind_runtime_path(
            expected_execution_domain_scope=contract.safety_execution_domain_scope,
            expected_account_scope=contract.safety_account_scope,
        )
        unresolved = state.active_reservations
        if len(unresolved) != 1:
            raise RuntimeError("offline recovery requires exactly one unresolved paper reservation")
        reservation = unresolved[0]
        if not reservation.outcome_unknown or not reservation.quarantined:
            raise RuntimeError("unresolved reservation is not a quarantined post-dispatch outcome")

        correlation = _local_settlement_correlations(contract, unresolved).get(
            reservation.reservation_id
        )
        if correlation is None or correlation.status != "MATCH" or correlation.evidence is None:
            status = "NOT_CHECKED" if correlation is None else correlation.status
            raise RuntimeError(
                f"offline recovery blocked: local terminal settlement status is {status}"
            )
        if reservation.claim_id is None or reservation.order_ref is None:
            raise RuntimeError("offline recovery reservation identity is incomplete")

        coordinator = SafetyRuntimeCoordinator(
            PaperExecutionIdentity(
                contract.safety_execution_domain_scope,
                contract.safety_account_scope,
            ),
            journal,
        )
        before_sequence = state.last_sequence
        released = coordinator.recover_after_verified_local_paper_settlement(
            reservation.idempotency_key,
            reservation.intent_fingerprint,
            correlation.evidence,
        )
        if (
            not released.released
            or released.terminal_sequence is None
            or released.terminal_sequence != before_sequence + 1
        ):
            raise RuntimeError("offline recovery did not append exactly one terminal event")

        recovered_state = journal.replay_and_bind_runtime_path(
            expected_execution_domain_scope=contract.safety_execution_domain_scope,
            expected_account_scope=contract.safety_account_scope,
        )
        if recovered_state.last_sequence != before_sequence + 1 or any(
            item.reservation_id == reservation.reservation_id
            for item in recovered_state.active_reservations
        ):
            raise RuntimeError("offline recovery terminal state could not be verified")

        return OfflineRecoveryResult(
            journal_identity=contract.safety_journal_identity,
            reservation_id_sha256=_redacted_identifier(
                "reservation_id", reservation.reservation_id
            ),
            claim_id_sha256=_redacted_identifier("claim_id", reservation.claim_id),
            order_ref_sha256=_redacted_identifier("order_ref", reservation.order_ref),
            terminal_sequence=released.terminal_sequence,
        )
    finally:
        lifecycle_lock.release()


def paper_safety_status(environ: Mapping[str, str]) -> dict[str, Any]:
    """Return a redacted, strictly read-only journal and settlement status."""

    contract = _paper_contract(environ)
    journal = SafetyJournal(contract.safety_journal_path)
    state = journal.replay_and_bind_runtime_path(
        expected_execution_domain_scope=contract.safety_execution_domain_scope,
        expected_account_scope=contract.safety_account_scope,
    )
    unresolved = state.active_reservations
    settlement_statuses = _local_settlement_statuses(contract, unresolved)
    observed_at = datetime.now(timezone.utc)
    reservations = []
    for reservation in unresolved:
        reference_time = reservation.claim_time or reservation.acquired_at
        age_seconds = max(0.0, (observed_at - reference_time).total_seconds())
        settlement_status = settlement_statuses.get(
            reservation.reservation_id,
            "NOT_CHECKED",
        )
        phase = (
            "OUTCOME_UNKNOWN"
            if reservation.outcome_unknown
            else (
                "AUTHORIZED_NOT_DISPATCHED"
                if reservation.claim_id is not None
                else "RESERVATION_ACQUIRED"
            )
        )
        reason_codes = [phase]
        if reservation.quarantined:
            reason_codes.append("JOURNAL_QUARANTINED")
        if settlement_status == "MATCH":
            reason_codes.append("LOCAL_TERMINAL_SETTLEMENT_PRESENT")
        elif settlement_status == "ABSENT":
            reason_codes.append("LOCAL_TERMINAL_SETTLEMENT_ABSENT")
        else:
            reason_codes.append(f"LOCAL_TERMINAL_SETTLEMENT_{settlement_status}")
        reservations.append(
            {
                "age_seconds": round(age_seconds, 6),
                "claim_id_sha256": (
                    _redacted_identifier("claim_id", reservation.claim_id)
                    if reservation.claim_id is not None
                    else None
                ),
                "exact_local_settlement_exists": settlement_status == "MATCH",
                "local_settlement_status": settlement_status,
                "order_ref_sha256": (
                    _redacted_identifier("order_ref", reservation.order_ref)
                    if reservation.order_ref is not None
                    else None
                ),
                "outcome_unknown": reservation.outcome_unknown,
                "phase": phase,
                "portfolio_id": reservation.portfolio_id,
                "quarantined": reservation.quarantined,
                "reason_codes": reason_codes,
                "reservation_id_sha256": _redacted_identifier(
                    "reservation_id",
                    reservation.reservation_id,
                ),
                "symbol": reservation.symbol,
            }
        )
    reservations.sort(
        key=lambda item: (
            item["portfolio_id"],
            item["symbol"],
            item["reservation_id_sha256"],
        )
    )
    return {
        "journal_identity": contract.safety_journal_identity,
        "reason_codes": (["UNRESOLVED_PAPER_SUBMISSION_AUTHORITY"] if reservations else []),
        "schema_version": _STATUS_SCHEMA_VERSION,
        "status": "BLOCKED" if reservations else "CLEAN",
        "unresolved_count": len(reservations),
        "unresolved_reservations": reservations,
    }


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
    status_parser = subparsers.add_parser(
        "status",
        help="Print redacted unresolved paper authority as read-only JSON.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        required=True,
        help="Emit the versioned machine-readable status document.",
    )
    recover = subparsers.add_parser(
        "recover-exact-local-settlement",
        help=(
            "While trader and Gateway are stopped, append one terminal journal "
            "release for exactly one verified local-paper settlement."
        ),
    )
    recover.add_argument("--confirm", required=True, metavar=RECOVER_CONFIRMATION)
    recover.add_argument(
        "--json",
        action="store_true",
        required=True,
        help="Emit only the redacted machine-readable recovery receipt.",
    )
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
        if args.command == "status":
            print(
                json.dumps(
                    paper_safety_status(environ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.command == "recover-exact-local-settlement":
            recovery = recover_exact_local_paper_settlement(
                environ,
                confirmation=args.confirm,
            )
            print(
                json.dumps(
                    {
                        "appended_terminal_events": 1,
                        "claim_id_sha256": recovery.claim_id_sha256,
                        "journal_identity": recovery.journal_identity,
                        "order_ref_sha256": recovery.order_ref_sha256,
                        "reservation_id_sha256": recovery.reservation_id_sha256,
                        "schema_version": 1,
                        "status": "RECOVERED",
                        "terminal_sequence": recovery.terminal_sequence,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
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
