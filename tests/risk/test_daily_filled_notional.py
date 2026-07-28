"""Adversarial tests for durable daily gross-filled-notional accounting."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext, setcontext
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import robo_trader.risk.filled_notional.ledger as ledger_module
from robo_trader.risk.filled_notional import (
    DailyFilledNotional,
    ExecutedFill,
    FilledNotionalConflict,
    FilledNotionalError,
    FilledNotionalIntegrityError,
    FilledNotionalMigrationRequired,
    FilledNotionalUnavailable,
    FillSide,
    MonotonicLedgerState,
)

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
ANCHOR_KEY = b"test-only-independent-anchor-key-32-bytes-minimum"
_MONOTONIC_VERIFIERS: dict[str, "TestMonotonicVerifier"] = {}


class TestMonotonicVerifier:
    __test__ = False

    def __init__(self) -> None:
        self.state: MonotonicLedgerState | None = None

    def __call__(self, candidate: MonotonicLedgerState) -> bool:
        prior = self.state
        if prior is None:
            self.state = candidate
            return True
        if candidate == prior:
            return True
        if candidate.ledger_id != prior.ledger_id:
            return False
        fill_delta = candidate.fill_count - prior.fill_count
        conflict_delta = candidate.conflict_count - prior.conflict_count
        if fill_delta < 0 or conflict_delta < 0 or fill_delta + conflict_delta != 1:
            return False
        if fill_delta == 0 and candidate.fill_head != prior.fill_head:
            return False
        if conflict_delta == 0 and candidate.conflict_head != prior.conflict_head:
            return False
        self.state = candidate
        return True


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _fill(
    execution_id: str,
    *,
    side: FillSide = FillSide.BUY,
    quantity: str = "2",
    price: str = "10.25",
    executed_at: datetime = datetime(2026, 7, 28, 15, 0, tzinfo=UTC),
) -> ExecutedFill:
    return ExecutedFill(
        broker_execution_id=execution_id,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        currency="USD",
        executed_at=executed_at,
    )


def _service(
    path: Path,
    *,
    account_id: str = "DU12345",
    portfolio_id: str = "default",
    clock: MutableClock | None = None,
    monotonic_verifier=None,
) -> DailyFilledNotional:
    anchor_directory = path.parent / "protected-anchor"
    anchor_directory.mkdir(mode=0o700, exist_ok=True)
    verifier = monotonic_verifier or _MONOTONIC_VERIFIERS.setdefault(
        str(path), TestMonotonicVerifier()
    )
    return DailyFilledNotional(
        path,
        anchor_path=anchor_directory / f"{path.name}.anchor",
        anchor_key=ANCHOR_KEY,
        monotonic_verifier=verifier,
        account_id=account_id,
        portfolio_id=portfolio_id,
        clock=clock or MutableClock(datetime(2026, 7, 28, 20, 0, tzinfo=UTC)),
    )


def _anchor_path(database_path: Path) -> Path:
    return database_path.parent / "protected-anchor" / f"{database_path.name}.anchor"


def _file_snapshot(path: Path) -> tuple[bytes, int, int]:
    metadata = os.lstat(path)
    return path.read_bytes(), metadata.st_dev, metadata.st_ino


def _path_inventory(root: Path) -> tuple[str, ...]:
    return tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*")))


def test_counts_only_executed_fill_evidence_with_exact_decimal_math(tmp_path):
    ledger = _service(tmp_path / "notional.db")

    results = [
        ledger.record_fill(_fill("buy", side=FillSide.BUY, quantity="0.1", price="0.2")),
        ledger.record_fill(_fill("sell", side=FillSide.SELL, quantity="-3", price="4.01")),
        ledger.record_fill(_fill("short", side=FillSide.SELL_SHORT, quantity="1.25", price="8")),
        ledger.record_fill(
            _fill("cover", side=FillSide.BUY_TO_COVER, quantity="-0.5", price="9.5")
        ),
    ]

    assert [result.recorded for result in results] == [True, True, True, True]
    assert ledger.current_gross_filled_notional() == Decimal("26.80")
    assert isinstance(ledger.current_gross_filled_notional(), Decimal)


def test_partial_fills_count_by_unique_broker_execution_not_parent_order(tmp_path):
    ledger = _service(tmp_path / "notional.db")

    first = ledger.record_fill(_fill("order-7.partial-1", quantity="2", price="100.10"))
    second = ledger.record_fill(_fill("order-7.partial-2", quantity="3", price="100.20"))

    assert first.gross_filled_notional == Decimal("200.2")
    assert second.gross_filled_notional == Decimal("500.8")


def test_decimal_product_beyond_default_context_precision_is_not_rounded(tmp_path):
    ledger = _service(tmp_path / "notional.db")

    result = ledger.record_fill(
        _fill(
            "high-precision",
            quantity="12345678901234567890.123456789",
            price="1.000000001",
        )
    )

    assert result.fill_notional == Decimal("12345678913580246791.358024679123456789")
    assert result.gross_filled_notional == Decimal("12345678913580246791.358024679123456789")


def test_exact_execution_replay_is_idempotent(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    fill = _fill("broker.exec.1", quantity="7", price="12.345")

    original = ledger.record_fill(fill)
    replay = ledger.record_fill(fill)

    assert original.recorded is True
    assert replay.recorded is False
    assert replay.fill_notional == Decimal("86.415")
    assert replay.gross_filled_notional == Decimal("86.415")
    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (1,)


def test_exact_replay_rechecks_monotonic_authority_at_return_boundary(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    fill = _fill("broker.exec.return-boundary")
    ledger.record_fill(fill)

    class RejectSecondVerification:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, state) -> bool:
            del state
            self.calls += 1
            return self.calls == 1

    verifier = RejectSecondVerification()
    ledger._monotonic_verifier = verifier

    with pytest.raises(FilledNotionalIntegrityError, match="monotonic authority rejected"):
        ledger.record_fill(fill)
    assert verifier.calls == 2
    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (1,)


def test_conflicting_duplicate_fails_closed_and_latches_instance(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    original = _fill("broker.exec.1")
    ledger.record_fill(original)

    with pytest.raises(FilledNotionalConflict, match="durable conflicting evidence"):
        ledger.record_fill(replace(original, price=Decimal("999")))
    with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
        ledger.current_gross_filled_notional()

    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_conflicts"
        ).fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM daily_filled_notional_conflicts")
    with pytest.raises(FilledNotionalConflict, match="requires review"):
        _service(ledger.database_path)

    review = DailyFilledNotional.review_quarantine(
        ledger.database_path,
        anchor_path=ledger.anchor_path,
        anchor_key=ANCHOR_KEY,
        monotonic_verifier=_MONOTONIC_VERIFIERS[str(ledger.database_path)],
    )
    assert len(review) == 1
    assert review[0].broker_execution_id == "broker.exec.1"
    assert review[0].existing_portfolio_id == "default"
    assert review[0].claimed_portfolio_id == "default"


def test_preexisting_instances_observe_durable_conflict_on_every_operation(tmp_path):
    path = tmp_path / "notional.db"
    writer = _service(path)
    reader = _service(path)
    appender = _service(path)
    original = _fill("shared-conflict")
    writer.record_fill(original)

    with pytest.raises(FilledNotionalConflict):
        writer.record_fill(replace(original, price=Decimal("999")))
    with pytest.raises(FilledNotionalConflict, match="requires review"):
        reader.current_gross_filled_notional()
    with pytest.raises(FilledNotionalConflict, match="requires review"):
        appender.record_fill(_fill("must-not-append"))

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (1,)


def test_restart_restores_same_day_total_from_append_only_ledger(tmp_path):
    path = tmp_path / "notional.db"
    first_process = _service(path)
    first_process.record_fill(_fill("exec-1", quantity="2", price="10"))
    first_process.record_fill(_fill("exec-2", quantity="4", price="2.5"))

    restarted = _service(path)

    assert restarted.restored_trading_date.isoformat() == "2026-07-28"
    assert restarted.restored_gross_filled_notional == Decimal("30")
    assert restarted.current_gross_filled_notional() == Decimal("30")


def test_hot_path_uses_authenticated_checkpoints_without_history_rescans(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    ledger = _service(path)

    def forbid_history_scan(*args, **kwargs):
        pytest.fail("hot accounting path performed an unbounded history scan")

    monkeypatch.setattr(ledger, "_validate_fills", forbid_history_scan)
    monkeypatch.setattr(ledger, "_validate_conflicts", forbid_history_scan)
    for index in range(400):
        ledger.record_fill(_fill(f"bounded-{index}"))

    assert ledger.current_gross_filled_notional() == Decimal("8200")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_checkpoints"
        ).fetchone() == (401,)

    monkeypatch.undo()
    assert _service(path).restored_gross_filled_notional == Decimal("8200")


def test_daily_scope_limit_rejects_append_before_any_durable_mutation(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    monkeypatch.setattr(ledger_module, "_MAX_DAILY_FILL_ROWS", 2)
    ledger = _service(path)
    ledger.record_fill(_fill("bounded-one"))
    ledger.record_fill(_fill("bounded-two"))

    with pytest.raises(FilledNotionalUnavailable, match="bounded accounting limit"):
        ledger.record_fill(_fill("must-not-persist"))

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (2,)
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_checkpoints"
        ).fetchone() == (3,)
    assert ledger.current_gross_filled_notional() == Decimal("41")


def test_scope_isolated_by_account_and_portfolio(tmp_path):
    path = tmp_path / "notional.db"
    default = _service(path)
    second_portfolio = _service(path, portfolio_id="income")
    second_account = _service(path, account_id="DU99999")

    default.record_fill(_fill("same-exec", quantity="1", price="10"))
    second_account.record_fill(_fill("same-exec", quantity="3", price="10"))

    assert default.current_gross_filled_notional() == Decimal("10")
    assert second_account.current_gross_filled_notional() == Decimal("30")
    with pytest.raises(FilledNotionalConflict, match="durable conflicting evidence"):
        second_portfolio.record_fill(_fill("same-exec", quantity="2", price="10"))
    review = DailyFilledNotional.review_quarantine(
        path,
        anchor_path=default.anchor_path,
        anchor_key=ANCHOR_KEY,
        monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
    )
    assert review[0].existing_portfolio_id == "default"
    assert review[0].claimed_portfolio_id == "income"


def test_new_york_midnight_rollover_and_late_prior_day_fill(tmp_path):
    clock = MutableClock(datetime(2026, 7, 29, 3, 59, 59, tzinfo=UTC))
    ledger = _service(tmp_path / "notional.db", clock=clock)
    ledger.record_fill(
        _fill("before-midnight", executed_at=datetime(2026, 7, 29, 3, 59, tzinfo=UTC))
    )
    assert ledger.current_gross_filled_notional() == Decimal("20.5")

    clock.value = datetime(2026, 7, 29, 4, 0, tzinfo=UTC)
    assert ledger.current_gross_filled_notional() == Decimal("0")
    ledger.record_fill(_fill("after-midnight", executed_at=datetime(2026, 7, 29, 4, 0, tzinfo=UTC)))
    assert ledger.current_gross_filled_notional() == Decimal("20.5")

    ledger.record_fill(
        _fill("late-prior-day", executed_at=datetime(2026, 7, 29, 3, 30, tzinfo=UTC))
    )
    assert ledger.current_gross_filled_notional() == Decimal("20.5")
    assert ledger.current_gross_filled_notional(
        as_of=datetime(2026, 7, 29, 3, 59, tzinfo=UTC)
    ) == Decimal("41")


def test_dst_fold_maps_both_real_instants_to_same_new_york_date(tmp_path):
    clock = MutableClock(datetime(2026, 11, 1, 8, 0, tzinfo=UTC))
    ledger = _service(tmp_path / "notional.db", clock=clock)
    first_one_thirty = datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=0)
    second_one_thirty = datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=1)

    ledger.record_fill(_fill("dst-edt", executed_at=first_one_thirty))
    ledger.record_fill(_fill("dst-est", executed_at=second_one_thirty))

    assert first_one_thirty.astimezone(UTC) != second_one_thirty.astimezone(UTC)
    assert ledger.current_gross_filled_notional() == Decimal("41")


def test_spring_dst_and_utc_date_boundary_are_deterministic(tmp_path):
    clock = MutableClock(datetime(2026, 3, 9, 1, 0, tzinfo=UTC))  # Mar 8, 21:00 EDT
    ledger = _service(tmp_path / "notional.db", clock=clock)
    ledger.record_fill(_fill("spring-before", executed_at=datetime(2026, 3, 8, 6, 59, tzinfo=UTC)))
    ledger.record_fill(_fill("spring-after", executed_at=datetime(2026, 3, 8, 7, 1, tzinfo=UTC)))

    assert ledger.current_gross_filled_notional() == Decimal("41")


def test_non_execution_and_inexact_or_malformed_evidence_is_rejected(tmp_path):
    ledger = _service(tmp_path / "notional.db")

    with pytest.raises(FilledNotionalError, match="only exact ExecutedFill"):
        ledger.record_fill(object())  # type: ignore[arg-type]
    with pytest.raises(FilledNotionalError, match="side must be a FillSide"):
        ledger.record_fill(replace(_fill("submitted"), side="SUBMITTED"))  # type: ignore[arg-type]
    with pytest.raises(FilledNotionalError, match="finite non-zero Decimal"):
        ledger.record_fill(replace(_fill("float"), quantity=1.0))  # type: ignore[arg-type]
    with pytest.raises(FilledNotionalError, match="aware datetime"):
        ledger.record_fill(replace(_fill("naive"), executed_at=datetime(2026, 7, 28)))
    assert ledger.current_gross_filled_notional() == Decimal("0")


def test_database_triggers_deny_update_and_delete(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    ledger.record_fill(_fill("immutable"))

    with sqlite3.connect(ledger.database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE daily_filled_notional_records SET price_text = '1'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM daily_filled_notional_records")
        with pytest.raises(sqlite3.IntegrityError, match="append to the chain"):
            connection.execute("""
                INSERT OR REPLACE INTO daily_filled_notional_records
                SELECT sequence, account_id, portfolio_id, broker_execution_id,
                       side, quantity_text, '1', currency, executed_at_utc,
                       trading_date, notional_text, scope_fill_count, scope_total_text,
                       previous_hash, record_hash
                FROM daily_filled_notional_records WHERE sequence = 1
                """)


def test_trigger_removal_and_row_tamper_fail_closed_on_restart(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("tamper-target"))
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER daily_filled_notional_records_no_update")
        connection.execute(
            "UPDATE daily_filled_notional_records SET notional_text = '1' WHERE sequence = 1"
        )

    with pytest.raises(FilledNotionalIntegrityError, match="schema objects do not match"):
        _service(path)


@pytest.mark.parametrize(
    "protected_table",
    [
        "daily_filled_notional_schema",
        "daily_filled_notional_records",
        "daily_filled_notional_conflicts",
        "daily_filled_notional_checkpoints",
    ],
)
@pytest.mark.parametrize("target_case", ["mixed", "uppercase"])
def test_case_variant_foreign_trigger_on_every_protected_table_fails_closed_without_mutation(
    tmp_path, protected_table, target_case
):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    target_identifier = (
        protected_table.upper()
        if target_case == "uppercase"
        else "".join(
            character.upper() if index % 2 == 0 else character.lower()
            for index, character in enumerate(protected_table)
        )
    )
    trigger_name = f"foreign_suppress_{target_case}_{protected_table}"
    with sqlite3.connect(path) as connection:
        connection.execute(f"""
            CREATE TRIGGER "{trigger_name}"
            BEFORE INSERT ON "{target_identifier}"
            BEGIN
                SELECT RAISE(IGNORE);
            END
            """)
        assert connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (trigger_name,),
        ).fetchone() == (target_identifier,)

    database_before = _file_snapshot(path)
    anchor_before = _file_snapshot(ledger.anchor_path)
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(FilledNotionalIntegrityError, match="schema objects do not match"):
        ledger.record_fill(_fill(f"must-not-be-ignored-{target_case}-{protected_table}"))

    assert _file_snapshot(path) == database_before
    assert _file_snapshot(ledger.anchor_path) == anchor_before
    assert _path_inventory(tmp_path) == paths_before

    with pytest.raises(FilledNotionalIntegrityError, match="schema objects do not match"):
        DailyFilledNotional.review_quarantine(
            path,
            anchor_path=ledger.anchor_path,
            anchor_key=ANCHOR_KEY,
            monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
        )

    assert _file_snapshot(path) == database_before
    assert _file_snapshot(ledger.anchor_path) == anchor_before
    assert _path_inventory(tmp_path) == paths_before


@pytest.mark.parametrize("suppressed_append", ["fill", "checkpoint"])
def test_record_fill_verifies_uncommitted_append_advanced_before_anchor_publication(
    tmp_path, monkeypatch, suppressed_append
):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    database_before = path.read_bytes()
    database_identity = (os.lstat(path).st_dev, os.lstat(path).st_ino)
    anchor_before = ledger.anchor_path.read_bytes()
    anchor_identity = (
        os.lstat(ledger.anchor_path).st_dev,
        os.lstat(ledger.anchor_path).st_ino,
    )

    if suppressed_append == "fill":
        monkeypatch.setattr(ledger, "_append_fill", lambda *args, **kwargs: "f" * 64)
    else:
        monkeypatch.setattr(
            ledger,
            "_append_checkpoint",
            lambda connection, before, after: replace(
                after,
                checkpoint_sequence=before.checkpoint_sequence + 1,
                checkpoint_head="f" * 64,
            ),
        )

    with pytest.raises(FilledNotionalIntegrityError, match="checkpoint|append"):
        ledger.record_fill(_fill(f"suppressed-{suppressed_append}"))

    assert path.read_bytes() == database_before
    assert (os.lstat(path).st_dev, os.lstat(path).st_ino) == database_identity
    assert ledger.anchor_path.read_bytes() == anchor_before
    assert (
        os.lstat(ledger.anchor_path).st_dev,
        os.lstat(ledger.anchor_path).st_ino,
    ) == anchor_identity
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_checkpoints"
        ).fetchone() == (1,)


def test_checkpoint_text_in_integer_column_fails_closed_with_typed_error(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    malformed_row = {
        "event_sequence": 0,
        "ledger_id": ledger._ledger_id,
        "database_device": "oops",
        "database_inode": 1,
        "fill_count": 0,
        "fill_head": "0" * 64,
        "conflict_count": 0,
        "conflict_head": "0" * 64,
        "previous_checkpoint_hash": "0" * 64,
        "checkpoint_hash": "0" * 64,
    }

    class MalformedCheckpointConnection:
        @staticmethod
        def execute(_statement):
            return (malformed_row,)

    with pytest.raises(FilledNotionalIntegrityError, match="checkpoint counters"):
        ledger._validate_checkpoints(
            MalformedCheckpointConnection(),  # type: ignore[arg-type]
            ledger_id=ledger._ledger_id,
            database_device=1,
            database_inode=1,
            fill_count=0,
            fill_head="0" * 64,
            conflict_count=0,
            conflict_head="0" * 64,
        )


def test_service_connection_has_no_update_delete_or_schema_authority(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    ledger.record_fill(_fill("immutable"))

    with ledger._connection(readonly=False) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("UPDATE daily_filled_notional_records SET price_text = '1'")
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("DELETE FROM daily_filled_notional_records")
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("DROP TABLE daily_filled_notional_records")


def test_hash_tamper_fails_closed_even_with_required_schema_restored(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("tamper-target"))
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER daily_filled_notional_records_no_update")
        connection.execute(
            "UPDATE daily_filled_notional_records SET price_text = '11' WHERE sequence = 1"
        )
        connection.execute(ledger_module._TRIGGER_SQL["daily_filled_notional_records_no_update"])

    with pytest.raises(FilledNotionalIntegrityError, match="notional is inconsistent"):
        _service(path)


def test_database_read_failure_latches_service_unavailable(tmp_path, monkeypatch):
    ledger = _service(tmp_path / "notional.db")
    ledger.record_fill(_fill("exec-1"))
    original_connect = ledger_module.sqlite3.connect

    def fail_connect(*args, **kwargs):
        raise sqlite3.OperationalError("injected database outage")

    monkeypatch.setattr(ledger_module.sqlite3, "connect", fail_connect)
    with pytest.raises(FilledNotionalUnavailable, match="read failed closed"):
        ledger.current_gross_filled_notional()
    monkeypatch.setattr(ledger_module.sqlite3, "connect", original_connect)
    with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
        ledger.current_gross_filled_notional()


def test_corrupt_database_fails_closed_without_replacing_it(tmp_path):
    path = tmp_path / "notional.db"
    contents = b"not a sqlite database; preserve this evidence"
    path.write_bytes(contents)
    before = hashlib.sha256(contents).hexdigest()

    with pytest.raises(FilledNotionalUnavailable, match="cannot be detected"):
        _service(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_current_query_is_read_only(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    ledger.record_fill(_fill("exec-1"))
    before = ledger.database_path.read_bytes()
    anchor_before = ledger.anchor_path.read_bytes()

    assert ledger.current_gross_filled_notional() == Decimal("20.5")

    assert ledger.database_path.read_bytes() == before
    assert ledger.anchor_path.read_bytes() == anchor_before
    assert not Path(f"{ledger.database_path}-wal").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("sidecar_kind", ["plain", "hardlink", "symlink", "directory"])
def test_missing_database_preserves_and_rejects_every_sqlite_sidecar(
    tmp_path, suffix, sidecar_kind
):
    path = tmp_path / "notional.db"
    sidecar = Path(f"{path}{suffix}")
    preserved_target = tmp_path / f"preserved-{sidecar_kind}{suffix}"
    marker = None

    if sidecar_kind == "plain":
        sidecar.write_bytes(b"preserve plain sidecar evidence")
    elif sidecar_kind == "hardlink":
        preserved_target.write_bytes(b"preserve hardlinked sidecar evidence")
        os.link(preserved_target, sidecar)
    elif sidecar_kind == "symlink":
        preserved_target.write_bytes(b"preserve symlink target evidence")
        sidecar.symlink_to(preserved_target)
    else:
        sidecar.mkdir()
        marker = sidecar / "preserve.marker"
        marker.write_bytes(b"preserve non-regular sidecar evidence")

    def stable_metadata(candidate: Path):
        metadata = candidate.lstat()
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    sidecar_metadata = stable_metadata(sidecar)
    sidecar_link = os.readlink(sidecar) if sidecar_kind == "symlink" else None
    target_metadata = (
        stable_metadata(preserved_target) if sidecar_kind in {"hardlink", "symlink"} else None
    )
    target_bytes = (
        preserved_target.read_bytes() if sidecar_kind in {"hardlink", "symlink"} else None
    )
    sidecar_bytes = sidecar.read_bytes() if sidecar_kind == "plain" else None
    marker_metadata = stable_metadata(marker) if marker is not None else None
    marker_bytes = marker.read_bytes() if marker is not None else None

    with pytest.raises(FilledNotionalUnavailable, match="main database is absent"):
        _service(path)

    assert not path.exists()
    assert not _anchor_path(path).exists()
    assert stable_metadata(sidecar) == sidecar_metadata
    if sidecar_kind == "plain":
        assert sidecar.read_bytes() == sidecar_bytes
    elif sidecar_kind in {"hardlink", "symlink"}:
        assert stable_metadata(preserved_target) == target_metadata
        assert preserved_target.read_bytes() == target_bytes
        if sidecar_kind == "hardlink":
            assert sidecar.samefile(preserved_target)
        else:
            assert os.readlink(sidecar) == sidecar_link
    else:
        assert marker is not None
        assert stable_metadata(marker) == marker_metadata
        assert marker.read_bytes() == marker_bytes


def test_missing_database_with_surviving_anchor_and_lock_creates_nothing(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    preserved_database = tmp_path / "preserved-notional.db"
    path.rename(preserved_database)
    lock_path = ledger.anchor_path.parent / f".{ledger.anchor_path.name}.lock"
    assert lock_path.exists()
    preserved_database_before = _file_snapshot(preserved_database)
    anchor_before = _file_snapshot(ledger.anchor_path)
    lock_before = _file_snapshot(lock_path)
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(
        FilledNotionalIntegrityError, match="main database is absent while anchor survives"
    ):
        _service(path)

    assert not path.exists()
    assert _file_snapshot(preserved_database) == preserved_database_before
    assert _file_snapshot(ledger.anchor_path) == anchor_before
    assert _file_snapshot(lock_path) == lock_before
    assert _path_inventory(tmp_path) == paths_before
    assert all(not Path(f"{path}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


def test_missing_database_with_surviving_anchor_does_not_create_lock(tmp_path):
    path = tmp_path / "notional.db"
    anchor_directory = tmp_path / "protected-anchor"
    anchor_directory.mkdir(mode=0o700)
    anchor_path = anchor_directory / "notional.db.anchor"
    anchor_path.write_bytes(b"preserved-anchor-evidence")
    os.chmod(anchor_path, 0o600)
    lock_path = anchor_directory / ".notional.db.anchor.lock"
    anchor_before = _file_snapshot(anchor_path)
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(
        FilledNotionalIntegrityError, match="main database is absent while anchor survives"
    ):
        DailyFilledNotional(
            path,
            anchor_path=anchor_path,
            anchor_key=ANCHOR_KEY,
            monotonic_verifier=TestMonotonicVerifier(),
            account_id="DU12345",
            portfolio_id="default",
        )

    assert not path.exists()
    assert not lock_path.exists()
    assert _file_snapshot(anchor_path) == anchor_before
    assert _path_inventory(tmp_path) == paths_before
    assert all(not Path(f"{path}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


@pytest.mark.parametrize("injected_artifact", ["anchor", "-wal", "-shm", "-journal"])
def test_missing_database_race_rejects_without_creating_transition_lock(
    tmp_path, monkeypatch, injected_artifact
):
    path = tmp_path / "notional.db"
    anchor_directory = tmp_path / "protected-anchor"
    anchor_directory.mkdir(mode=0o700)
    anchor_path = anchor_directory / "notional.db.anchor"
    lock_path = anchor_directory / ".notional.db.anchor.lock"
    original_transition_lock = DailyFilledNotional._anchor_transition_lock
    artifact_path = (
        anchor_path if injected_artifact == "anchor" else Path(f"{path}{injected_artifact}")
    )
    artifact_before: list[tuple[bytes, int, int]] = []
    paths_after_injection: list[tuple[str, ...]] = []

    @contextmanager
    def inject_before_lock(self, **kwargs):
        artifact_path.write_bytes(b"preserved evidence injected before lock acquisition")
        if artifact_path == anchor_path:
            os.chmod(artifact_path, 0o600)
        artifact_before.append(_file_snapshot(artifact_path))
        paths_after_injection.append(_path_inventory(tmp_path))
        with original_transition_lock(self, **kwargs):
            yield

    monkeypatch.setattr(DailyFilledNotional, "_anchor_transition_lock", inject_before_lock)

    with pytest.raises(FilledNotionalUnavailable):
        DailyFilledNotional(
            path,
            anchor_path=anchor_path,
            anchor_key=ANCHOR_KEY,
            monotonic_verifier=TestMonotonicVerifier(),
            account_id="DU12345",
            portfolio_id="default",
        )

    assert not path.exists()
    assert not lock_path.exists()
    assert _file_snapshot(artifact_path) == artifact_before[0]
    assert _path_inventory(tmp_path) == paths_after_injection[0]


def test_wal_shm_hardlinks_are_rejected_without_mutating_targets(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    with sqlite3.connect(path, isolation_level=None) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)

    wal_target = tmp_path / "preserved.wal"
    shm_target = tmp_path / "preserved.shm"
    wal_target.write_bytes(b"preserve WAL evidence")
    shm_target.write_bytes(b"preserve SHM evidence")
    wal_path = Path(f"{path}-wal")
    shm_path = Path(f"{path}-shm")
    os.link(wal_target, wal_path)
    os.link(shm_target, shm_path)
    wal_before = wal_target.read_bytes()
    shm_before = shm_target.read_bytes()
    wal_stat = wal_target.stat()
    shm_stat = shm_target.stat()

    with pytest.raises(FilledNotionalUnavailable, match="WAL/SHM sidecars are unsupported"):
        _service(path)

    assert wal_target.read_bytes() == wal_before
    assert shm_target.read_bytes() == shm_before
    assert (wal_target.stat().st_ino, wal_target.stat().st_size) == (
        wal_stat.st_ino,
        wal_stat.st_size,
    )
    assert (shm_target.stat().st_ino, shm_target.stat().st_size) == (
        shm_stat.st_ino,
        shm_stat.st_size,
    )


def test_clean_wal_mode_is_rejected_before_sqlite_creates_sidecars(tmp_path):
    path = tmp_path / "notional.db"
    _service(path)
    with sqlite3.connect(path, isolation_level=None) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()

    with pytest.raises(FilledNotionalUnavailable, match="rollback-journal mode"):
        _service(path)

    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()


def test_hardlinked_rollback_journal_is_rejected_without_mutation(tmp_path):
    path = tmp_path / "notional.db"
    _service(path)
    target = tmp_path / "preserved.journal"
    target.write_bytes(b"preserve rollback evidence")
    journal_path = Path(f"{path}-journal")
    os.link(target, journal_path)
    before = target.read_bytes()
    target_stat = target.stat()

    with pytest.raises(FilledNotionalUnavailable, match="journal prevents safe"):
        _service(path)

    assert target.read_bytes() == before
    assert (target.stat().st_ino, target.stat().st_size, target.stat().st_nlink) == (
        target_stat.st_ino,
        target_stat.st_size,
        target_stat.st_nlink,
    )


def test_hardlinked_main_ledger_is_rejected_before_fill_mutation(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    preserved_alias = tmp_path / "preserved-ledger.db"
    os.link(path, preserved_alias)
    database_before = _file_snapshot(path)
    alias_before = _file_snapshot(preserved_alias)
    anchor_before = _file_snapshot(ledger.anchor_path)
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(FilledNotionalIntegrityError, match="exclusive non-symlink regular"):
        ledger.record_fill(_fill("must-not-mutate-hardlinked-ledger"))

    assert _file_snapshot(path) == database_before
    assert _file_snapshot(preserved_alias) == alias_before
    assert _file_snapshot(ledger.anchor_path) == anchor_before
    assert _path_inventory(tmp_path) == paths_before


def test_anchor_write_crash_window_recovers_one_authenticated_fill(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    original_replace = ledger._replace_anchor

    def fail_final_anchor(desired, *, expected):
        if expected.pending_state is not None and desired.pending_state is None:
            raise OSError("injected crash before final atomic anchor replace")
        return original_replace(desired, expected=expected)

    monkeypatch.setattr(ledger, "_replace_anchor", fail_final_anchor)
    with pytest.raises(FilledNotionalUnavailable, match="write failed closed"):
        ledger.record_fill(_fill("committed-before-anchor"))

    restarted = _service(path)
    assert restarted.restored_gross_filled_notional == Decimal("20.5")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (1,)


def test_crash_before_database_commit_resolves_pending_anchor_to_old_state(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    ledger = _service(path)

    def fail_commit(connection):
        raise sqlite3.OperationalError("injected crash before database commit")

    monkeypatch.setattr(ledger, "_commit_database", fail_commit)
    with pytest.raises(FilledNotionalUnavailable, match="write failed closed"):
        ledger.record_fill(_fill("never-committed"))

    restarted = _service(path)
    assert restarted.restored_gross_filled_notional == Decimal("0")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (0,)


def test_empty_initialized_database_recovers_when_initial_anchor_creation_failed(
    tmp_path, monkeypatch
):
    path = tmp_path / "notional.db"
    original_create = DailyFilledNotional._create_initial_anchor

    def fail_initial_anchor(self, state):
        del self, state
        raise OSError("injected failure after initial database commit")

    monkeypatch.setattr(DailyFilledNotional, "_create_initial_anchor", fail_initial_anchor)
    with pytest.raises(FilledNotionalUnavailable, match="startup failed closed"):
        _service(path)

    assert path.is_file()
    assert not _anchor_path(path).exists()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_conflicts"
        ).fetchone() == (0,)

    monkeypatch.setattr(DailyFilledNotional, "_create_initial_anchor", original_create)
    restarted = _service(path)

    assert restarted.restored_gross_filled_notional == Decimal("0")
    assert restarted.anchor_path.is_file()


def test_missing_anchor_never_reinitializes_nonempty_ledger(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("durable-before-anchor-loss"))
    database_before = _file_snapshot(path)
    ledger.anchor_path.unlink()
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(
        FilledNotionalIntegrityError, match="cannot authenticate a non-empty ledger"
    ):
        _service(path)

    assert _file_snapshot(path) == database_before
    assert not ledger.anchor_path.exists()
    assert _path_inventory(tmp_path) == paths_before


def test_slow_monotonic_read_verifier_serializes_writer_without_sqlite_timeout(tmp_path):
    path = tmp_path / "notional.db"
    writer = _service(path)
    reader = _service(path)
    verifier_started = threading.Event()
    release_verifier = threading.Event()
    verification_state = TestMonotonicVerifier()
    read_results: list[object] = []
    write_results: list[object] = []

    def slow_verifier(state):
        verifier_started.set()
        if not release_verifier.wait(timeout=5):
            raise RuntimeError("test timed out releasing monotonic verifier")
        return verification_state(state)

    reader._monotonic_verifier = slow_verifier

    def read() -> None:
        try:
            read_results.append(reader.current_gross_filled_notional())
        except BaseException as exc:  # pragma: no cover - asserted below
            read_results.append(exc)

    def write() -> None:
        try:
            write_results.append(writer.record_fill(_fill("during-slow-verification")))
        except BaseException as exc:  # pragma: no cover - asserted below
            write_results.append(exc)

    reader_thread = threading.Thread(target=read)
    writer_thread = threading.Thread(target=write)
    reader_thread.start()
    assert verifier_started.wait(timeout=5)
    writer_thread.start()
    writer_thread.join(timeout=0.2)

    try:
        assert writer_thread.is_alive(), "writer bypassed the transition-boundary lock"
        assert write_results == []
    finally:
        release_verifier.set()
        reader_thread.join(timeout=5)
        writer_thread.join(timeout=5)

    assert not reader_thread.is_alive()
    assert not writer_thread.is_alive()
    assert read_results == [Decimal("0")]
    assert len(write_results) == 1
    assert not isinstance(write_results[0], BaseException)
    assert reader.current_gross_filled_notional() == Decimal("20.5")


def test_reader_waits_for_concurrent_pending_writer_without_latching(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    writer = _service(path)
    reader = _service(path)
    pending_ready = threading.Event()
    release_writer = threading.Event()
    original_commit = writer._commit_database
    results: list[object] = []

    def gated_commit(connection):
        pending_ready.set()
        if not release_writer.wait(timeout=5):
            raise sqlite3.OperationalError("test timed out waiting to release writer")
        original_commit(connection)

    monkeypatch.setattr(writer, "_commit_database", gated_commit)

    def write() -> None:
        try:
            results.append(writer.record_fill(_fill("concurrent-fill")))
        except BaseException as exc:  # pragma: no cover - asserted below
            results.append(exc)

    def read() -> None:
        try:
            results.append(reader.current_gross_filled_notional())
        except BaseException as exc:  # pragma: no cover - asserted below
            results.append(exc)

    writer_thread = threading.Thread(target=write)
    reader_thread = threading.Thread(target=read)
    writer_thread.start()
    assert pending_ready.wait(timeout=5)
    reader_thread.start()
    reader_thread.join(timeout=0.2)
    assert reader_thread.is_alive(), "reader bypassed the writer transition lock"
    assert len(results) == 0
    release_writer.set()
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)

    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert not any(isinstance(result, BaseException) for result in results)
    assert Decimal("20.5") in results
    assert reader.current_gross_filled_notional() == Decimal("20.5")


def test_reader_snapshot_rechecks_anchor_before_returning_during_writer_race(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    writer = _service(path)
    reader = _service(path)
    snapshot_total_ready = threading.Event()
    release_reader = threading.Event()
    original_total = reader._total_for_date
    total_calls = 0
    results: dict[str, object] = {}

    def gated_total(connection, trading_day):
        nonlocal total_calls
        total = original_total(connection, trading_day)
        total_calls += 1
        if total_calls == 1:
            snapshot_total_ready.set()
            if not release_reader.wait(timeout=5):
                raise sqlite3.OperationalError("test timed out releasing reader snapshot")
        return total

    monkeypatch.setattr(reader, "_total_for_date", gated_total)

    def read() -> None:
        try:
            results["read"] = reader.current_gross_filled_notional()
        except BaseException as exc:  # pragma: no cover - asserted below
            results["read"] = exc

    def write() -> None:
        try:
            results["write"] = writer.record_fill(_fill("snapshot-race"))
        except BaseException as exc:  # pragma: no cover - asserted below
            results["write"] = exc

    reader_thread = threading.Thread(target=read)
    writer_thread = threading.Thread(target=write)
    reader_thread.start()
    assert snapshot_total_ready.wait(timeout=5)
    writer_thread.start()
    writer_thread.join(timeout=0.2)
    assert writer_thread.is_alive(), "writer bypassed the reader transition lock"
    assert "write" not in results
    release_reader.set()
    reader_thread.join(timeout=5)
    writer_thread.join(timeout=5)

    assert not reader_thread.is_alive()
    assert not writer_thread.is_alive()
    assert not isinstance(results.get("read"), BaseException)
    assert not isinstance(results.get("write"), BaseException)
    assert results["read"] == Decimal("0")
    assert total_calls >= 2
    assert reader._failed_reason is None
    assert reader.current_gross_filled_notional() == Decimal("20.5")


def test_concurrent_writers_serialize_through_stable_anchor_publication(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    first = _service(path)
    second = _service(path)
    first_reached_post_commit = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_done = threading.Event()
    results: dict[str, object] = {}
    original_replace = first._replace_anchor

    def gate_final_anchor(desired, *, expected):
        if expected.pending_state is not None and desired.pending_state is None:
            first_reached_post_commit.set()
            if not release_first.wait(timeout=5):
                raise sqlite3.OperationalError("test timed out releasing first writer")
        return original_replace(desired, expected=expected)

    monkeypatch.setattr(first, "_replace_anchor", gate_final_anchor)

    def write_first() -> None:
        try:
            results["first"] = first.record_fill(_fill("writer-one"))
        except BaseException as exc:  # pragma: no cover - asserted below
            results["first"] = exc

    def write_second() -> None:
        second_started.set()
        try:
            results["second"] = second.record_fill(_fill("writer-two"))
        except BaseException as exc:  # pragma: no cover - asserted below
            results["second"] = exc
        finally:
            second_done.set()

    first_thread = threading.Thread(target=write_first)
    second_thread = threading.Thread(target=write_second)
    first_thread.start()
    assert first_reached_post_commit.wait(timeout=5)
    second_thread.start()
    assert second_started.wait(timeout=5)
    assert not second_done.wait(timeout=0.1)
    release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not isinstance(results.get("first"), BaseException)
    assert not isinstance(results.get("second"), BaseException)
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT broker_execution_id FROM daily_filled_notional_records ORDER BY sequence"
        ).fetchall() == [("writer-one",), ("writer-two",)]
    assert _service(path).restored_gross_filled_notional == Decimal("41")


def test_transition_lock_revalidates_path_identity_after_blocked_flock(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    lock_path = ledger.anchor_path.parent / f".{ledger.anchor_path.name}.lock"
    preserved_lock = ledger.anchor_path.parent / "preserved-transition.lock"
    blocker = os.open(lock_path, os.O_RDWR)
    fcntl.flock(blocker, fcntl.LOCK_EX)
    worker_opened_lock = threading.Event()
    worker_entered_section = threading.Event()
    worker_result: list[BaseException] = []
    worker_ident: list[int] = []
    original_open = os.open

    def observe_worker_open(target, flags, mode=0o777, *, dir_fd=None):
        descriptor = original_open(target, flags, mode, dir_fd=dir_fd)
        if (
            worker_ident
            and threading.get_ident() == worker_ident[0]
            and target == ledger._anchor_lock_name
        ):
            worker_opened_lock.set()
        return descriptor

    monkeypatch.setattr(ledger_module.os, "open", observe_worker_open)

    def acquire_transition_lock() -> None:
        worker_ident.append(threading.get_ident())
        try:
            with ledger._anchor_transition_lock():
                worker_entered_section.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            worker_result.append(exc)

    worker = threading.Thread(target=acquire_transition_lock)
    worker.start()
    assert worker_opened_lock.wait(timeout=5)
    lock_path.rename(preserved_lock)
    replacement = original_open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(replacement)
    fcntl.flock(blocker, fcntl.LOCK_UN)
    os.close(blocker)
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert not worker_entered_section.is_set()
    assert len(worker_result) == 1
    assert isinstance(worker_result[0], FilledNotionalIntegrityError)
    assert "transition lock" in str(worker_result[0])


@pytest.mark.parametrize("mutation", ["mode", "link-count"])
def test_transition_lock_revalidates_safety_metadata_after_flock(tmp_path, monkeypatch, mutation):
    ledger = _service(tmp_path / "notional.db")
    lock_path = ledger.anchor_path.parent / f".{ledger.anchor_path.name}.lock"
    extra_link = ledger.anchor_path.parent / "extra-transition-lock-link"
    lock_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    original_flock = fcntl.flock
    mutated = False
    entered_section = False

    def mutate_after_acquire(descriptor, operation):
        nonlocal mutated
        result = original_flock(descriptor, operation)
        metadata = os.fstat(descriptor)
        if (
            not mutated
            and operation & fcntl.LOCK_EX
            and (metadata.st_dev, metadata.st_ino) == lock_identity
        ):
            mutated = True
            if mutation == "mode":
                os.chmod(lock_path, 0o644)
            else:
                os.link(lock_path, extra_link)
        return result

    monkeypatch.setattr(ledger_module.fcntl, "flock", mutate_after_acquire)

    with pytest.raises(FilledNotionalIntegrityError, match="transition lock"):
        with ledger._anchor_transition_lock():
            entered_section = True

    assert mutated
    assert not entered_section


def test_invalid_pending_anchor_identity_still_fails_closed(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    payload = json.loads(ledger.anchor_path.read_text(encoding="ascii"))
    payload["pending_state"] = dict(payload["state"])
    payload["pending_state"]["database_inode"] += 1
    unsigned = {key: payload[key] for key in payload if key != "mac"}
    payload["mac"] = ledger_module._keyed_hash(ANCHOR_KEY, unsigned)
    ledger.anchor_path.write_text(ledger_module._canonical_json(payload) + "\n", encoding="ascii")
    os.chmod(ledger.anchor_path, 0o600)

    with pytest.raises(FilledNotionalIntegrityError, match="pending anchor"):
        ledger.current_gross_filled_notional()


def test_pending_writer_timeout_is_bounded_and_does_not_latch_reader(tmp_path, monkeypatch):
    ledger = _service(tmp_path / "notional.db")
    with ledger._connection(readonly=True) as connection:
        state = ledger._validate_ledger(connection)
    ledger._anchor = ledger._replace_anchor(
        ledger._anchor_for_state(state, pending_state=state),
        expected=ledger._anchor,
    )
    original_resolve = ledger._resolve_pending_anchor

    def always_busy():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ledger, "_resolve_pending_anchor", always_busy)
    with pytest.raises(FilledNotionalUnavailable, match="remained busy"):
        ledger.current_gross_filled_notional()
    assert ledger._failed_reason is None

    monkeypatch.setattr(ledger, "_resolve_pending_anchor", original_resolve)
    assert ledger.current_gross_filled_notional() == Decimal("0")


def test_sigkill_hot_journal_is_recovered_rw_before_readonly_validation(tmp_path):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("SIGKILL unavailable on this host")
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("durable"))
    script = f"""
import signal
from datetime import datetime, timezone
from decimal import Decimal
from robo_trader.risk.filled_notional import DailyFilledNotional, ExecutedFill, FillSide

service = DailyFilledNotional(
    {str(path)!r},
    anchor_path={str(ledger.anchor_path)!r},
    anchor_key={ANCHOR_KEY!r},
    monotonic_verifier=lambda state: True,
    account_id='DU12345',
    portfolio_id='default',
    clock=lambda: datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc),
)
def pause_before_commit(connection):
    print('READY', flush=True)
    signal.pause()
service._commit_database = pause_before_commit
service.record_fill(ExecutedFill(
    broker_execution_id='uncommitted-crash',
    side=FillSide.BUY,
    quantity=Decimal('2'),
    price=Decimal('10.25'),
    currency='USD',
    executed_at=datetime(2026, 7, 28, 15, 0, tzinfo=timezone.utc),
))
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    journal = Path(f"{path}-journal")
    assert journal.exists() and journal.stat().st_size > 0
    restarted = _service(path)
    assert restarted.restored_gross_filled_notional == Decimal("20.5")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT broker_execution_id FROM daily_filled_notional_records"
        ).fetchall() == [("durable",)]


def test_hot_journal_monotonic_rejection_preserves_all_original_evidence(tmp_path):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("SIGKILL unavailable on this host")
    path = tmp_path / "monotonic-replay-hot.db"
    verifier = TestMonotonicVerifier()
    ledger = _service(path, monotonic_verifier=verifier)
    ledger.record_fill(_fill("older"))
    old_database = path.read_bytes()
    old_anchor = ledger.anchor_path.read_bytes()
    ledger.record_fill(_fill("newer"))

    path.write_bytes(old_database)
    ledger.anchor_path.write_bytes(old_anchor)
    os.chmod(ledger.anchor_path, 0o600)
    script = f"""
import signal
import sqlite3

connection = sqlite3.connect({str(path)!r}, isolation_level=None)
connection.execute('PRAGMA journal_mode=DELETE')
connection.execute('PRAGMA synchronous=FULL')
connection.execute('PRAGMA cache_size=1')
connection.execute('PRAGMA cache_spill=ON')
connection.execute('BEGIN IMMEDIATE')
connection.execute('CREATE TABLE spill_payload (id INTEGER PRIMARY KEY, value BLOB)')
connection.executemany(
    'INSERT INTO spill_payload VALUES (?, randomblob(4096))',
    ((index,) for index in range(1, 513)),
)
print('READY', flush=True)
signal.pause()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    journal = Path(f"{path}-journal")
    assert journal.exists() and journal.stat().st_size > 0
    database_before = _file_snapshot(path)
    journal_before = _file_snapshot(journal)
    anchor_before = _file_snapshot(ledger.anchor_path)
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(FilledNotionalIntegrityError, match="monotonic authority rejected"):
        _service(path, monotonic_verifier=verifier)

    assert _file_snapshot(path) == database_before
    assert _file_snapshot(journal) == journal_before
    assert _file_snapshot(ledger.anchor_path) == anchor_before
    assert _path_inventory(tmp_path) == paths_before


def test_hot_journal_is_revalidated_under_lock_immediately_before_recovery(tmp_path, monkeypatch):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("SIGKILL unavailable on this host")
    path = tmp_path / "replaced-hot-journal.db"
    ledger = _service(path)
    ledger.record_fill(_fill("durable-before-replacement"))
    script = f"""
import signal
import sqlite3

connection = sqlite3.connect({str(path)!r}, isolation_level=None)
connection.execute('PRAGMA journal_mode=DELETE')
connection.execute('PRAGMA synchronous=FULL')
connection.execute('PRAGMA cache_size=1')
connection.execute('PRAGMA cache_spill=ON')
connection.execute('BEGIN IMMEDIATE')
connection.execute('CREATE TABLE spill_payload (id INTEGER PRIMARY KEY, value BLOB)')
connection.executemany(
    'INSERT INTO spill_payload VALUES (?, randomblob(4096))',
    ((index,) for index in range(1, 513)),
)
print('READY', flush=True)
signal.pause()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    journal = Path(f"{path}-journal")
    replacement = tmp_path / "replacement.journal"
    replacement_bytes = bytearray(journal.read_bytes())
    replacement_bytes[0] ^= 0xFF
    replacement.write_bytes(replacement_bytes)
    original_validator = DailyFilledNotional._validate_hot_journal_recovery_on_copy
    validation_calls = 0

    def replace_after_first_validation(self):
        nonlocal validation_calls
        validation_calls += 1
        evidence = original_validator(self)
        if validation_calls == 1:
            os.replace(replacement, journal)
        return evidence

    monkeypatch.setattr(
        DailyFilledNotional,
        "_validate_hot_journal_recovery_on_copy",
        replace_after_first_validation,
    )
    database_before = _file_snapshot(path)
    anchor_before = _file_snapshot(ledger.anchor_path)

    with pytest.raises(FilledNotionalError):
        _service(path)

    assert validation_calls == 2
    assert _file_snapshot(path) == database_before
    assert journal.read_bytes() == bytes(replacement_bytes)
    assert _file_snapshot(ledger.anchor_path) == anchor_before


def test_sigkill_spilled_hot_journal_rejects_hardlinked_main_without_mutation(tmp_path):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("SIGKILL unavailable on this host")
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("durable-before-spill"))
    stable_database = path.read_bytes()
    script = f"""
import signal
import sqlite3

connection = sqlite3.connect({str(path)!r}, isolation_level=None)
connection.execute('PRAGMA journal_mode=DELETE')
connection.execute('PRAGMA synchronous=FULL')
connection.execute('PRAGMA cache_size=1')
connection.execute('PRAGMA cache_spill=ON')
connection.execute('BEGIN IMMEDIATE')
connection.execute('CREATE TABLE spill_payload (id INTEGER PRIMARY KEY, value BLOB)')
connection.executemany(
    'INSERT INTO spill_payload VALUES (?, randomblob(4096))',
    ((index,) for index in range(1, 513)),
)
print('READY', flush=True)
signal.pause()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    journal = Path(f"{path}-journal")
    assert journal.exists() and journal.stat().st_size > 0
    assert path.read_bytes() != stable_database
    preserved_alias = tmp_path / "preserved-hot-ledger.db"
    os.link(path, preserved_alias)
    database_before = _file_snapshot(path)
    alias_before = _file_snapshot(preserved_alias)
    journal_before = _file_snapshot(journal)
    anchor_before = _file_snapshot(ledger.anchor_path)
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(FilledNotionalUnavailable, match="exclusive non-symlink regular"):
        _service(path)

    assert _file_snapshot(path) == database_before
    assert _file_snapshot(preserved_alias) == alias_before
    assert _file_snapshot(journal) == journal_before
    assert _file_snapshot(ledger.anchor_path) == anchor_before
    assert path.samefile(preserved_alias)
    assert _path_inventory(tmp_path) == paths_before


def test_ledger_tail_deletion_is_detected_against_independent_anchor(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("one"))
    ledger.record_fill(_fill("two"))
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER daily_filled_notional_records_no_delete")
        connection.execute("DELETE FROM daily_filled_notional_records WHERE sequence = 2")
        connection.execute(ledger_module._TRIGGER_SQL["daily_filled_notional_records_no_delete"])

    with pytest.raises(FilledNotionalIntegrityError, match="rollback, tail deletion"):
        _service(path)


def test_conflict_quarantine_tail_deletion_is_detected_by_anchor(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    original = _fill("conflicted")
    ledger.record_fill(original)
    with pytest.raises(FilledNotionalConflict):
        ledger.record_fill(replace(original, price=Decimal("999")))
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER daily_filled_notional_conflicts_no_delete")
        connection.execute("DELETE FROM daily_filled_notional_conflicts WHERE sequence = 1")
        connection.execute(ledger_module._TRIGGER_SQL["daily_filled_notional_conflicts_no_delete"])

    with pytest.raises(FilledNotionalIntegrityError, match="rollback, tail deletion"):
        _service(path)


def test_ledger_rollback_is_detected_against_newer_anchor(tmp_path):
    path = tmp_path / "notional.db"
    snapshot = tmp_path / "old-ledger-copy.db"
    ledger = _service(path)
    ledger.record_fill(_fill("one"))
    shutil.copyfile(path, snapshot)
    ledger.record_fill(_fill("two"))
    shutil.copyfile(snapshot, path)

    with pytest.raises(FilledNotionalIntegrityError, match="rollback, tail deletion"):
        _service(path)


def test_stable_anchor_rollback_is_not_mistaken_for_a_crash_window(tmp_path):
    path = tmp_path / "notional.db"
    old_anchor = tmp_path / "old.anchor"
    ledger = _service(path)
    ledger.record_fill(_fill("one"))
    shutil.copyfile(ledger.anchor_path, old_anchor)
    ledger.record_fill(_fill("two"))
    shutil.copyfile(old_anchor, ledger.anchor_path)
    os.chmod(ledger.anchor_path, 0o600)

    with pytest.raises(FilledNotionalIntegrityError, match="stable-anchor rollback"):
        _service(path)


def test_coordinated_old_database_and_anchor_replay_requires_monotonic_rejection(tmp_path):
    path = tmp_path / "notional.db"
    verifier = TestMonotonicVerifier()
    ledger = _service(path, monotonic_verifier=verifier)
    ledger.record_fill(_fill("one"))
    old_database = path.read_bytes()
    old_anchor = ledger.anchor_path.read_bytes()
    ledger.record_fill(_fill("two"))

    path.write_bytes(old_database)
    ledger.anchor_path.write_bytes(old_anchor)
    os.chmod(ledger.anchor_path, 0o600)

    with pytest.raises(FilledNotionalIntegrityError, match="monotonic authority rejected"):
        _service(path, monotonic_verifier=verifier)


def test_review_reverifies_monotonic_state_after_constructor_before_return(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    verifier = TestMonotonicVerifier()
    ledger = _service(path, monotonic_verifier=verifier)
    ledger.record_fill(_fill("older"))
    old_database = path.read_bytes()
    old_anchor = ledger.anchor_path.read_bytes()
    ledger.record_fill(_fill("newer"))
    original_connection = DailyFilledNotional._connection
    authoritative_read_count = 0

    @contextmanager
    def restore_before_review(self, *, readonly, immutable=False):
        nonlocal authoritative_read_count
        if readonly:
            authoritative_read_count += 1
            if authoritative_read_count == 2:
                path.write_bytes(old_database)
                ledger.anchor_path.write_bytes(old_anchor)
                os.chmod(ledger.anchor_path, 0o600)
        with original_connection(self, readonly=readonly, immutable=immutable) as connection:
            yield connection

    monkeypatch.setattr(DailyFilledNotional, "_connection", restore_before_review)

    with pytest.raises(FilledNotionalIntegrityError, match="monotonic authority rejected"):
        DailyFilledNotional.review_quarantine(
            path,
            anchor_path=ledger.anchor_path,
            anchor_key=ANCHOR_KEY,
            monotonic_verifier=verifier,
        )


def test_review_quarantine_success_is_byte_inode_and_path_read_only(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("review-read-only"))
    database_before = _file_snapshot(path)
    anchor_before = _file_snapshot(ledger.anchor_path)
    paths_before = _path_inventory(tmp_path)

    evidence = DailyFilledNotional.review_quarantine(
        path,
        anchor_path=ledger.anchor_path,
        anchor_key=ANCHOR_KEY,
        monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
    )

    assert evidence == ()
    assert _file_snapshot(path) == database_before
    assert _file_snapshot(ledger.anchor_path) == anchor_before
    assert _path_inventory(tmp_path) == paths_before


def test_review_quarantine_retries_when_conflict_commits_during_verification(tmp_path):
    path = tmp_path / "notional.db"
    verifier = TestMonotonicVerifier()
    ledger = _service(path, monotonic_verifier=verifier)
    original = _fill("conflict-during-review-verification")
    ledger.record_fill(original)
    writer = _service(path, monotonic_verifier=verifier)

    class CommitConflictDuringSecondVerification:
        def __init__(self) -> None:
            self.calls = 0
            self.conflict_committed = False

        def __call__(self, state) -> bool:
            self.calls += 1
            accepted = verifier(state)
            if self.calls == 2:
                try:
                    writer.record_fill(replace(original, price=Decimal("999")))
                except FilledNotionalConflict:
                    self.conflict_committed = True
                else:  # pragma: no cover - durable conflict must reject the writer
                    raise AssertionError("conflicting execution unexpectedly succeeded")
            return accepted

    race_verifier = CommitConflictDuringSecondVerification()

    evidence = DailyFilledNotional.review_quarantine(
        path,
        anchor_path=ledger.anchor_path,
        anchor_key=ANCHOR_KEY,
        monotonic_verifier=race_verifier,
    )

    assert race_verifier.conflict_committed
    assert race_verifier.calls >= 3
    assert len(evidence) == 1
    assert evidence[0].broker_execution_id == original.broker_execution_id


def test_review_quarantine_translates_concurrent_pending_writer_without_mutation(
    tmp_path, monkeypatch
):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    writer = _service(path)
    pending_published = threading.Event()
    release_writer = threading.Event()
    writer_result: list[object] = []
    original_commit = writer._commit_database

    def hold_after_pending(connection):
        pending_published.set()
        if not release_writer.wait(timeout=5):
            raise sqlite3.OperationalError("test timed out releasing pending writer")
        original_commit(connection)

    monkeypatch.setattr(writer, "_commit_database", hold_after_pending)

    def write() -> None:
        try:
            writer_result.append(writer.record_fill(_fill("pending-during-review")))
        except BaseException as exc:  # pragma: no cover - asserted below
            writer_result.append(exc)

    writer_thread = threading.Thread(target=write)
    writer_thread.start()
    assert pending_published.wait(timeout=5)
    database_before = _file_snapshot(path)
    anchor_before = _file_snapshot(ledger.anchor_path)
    paths_before = _path_inventory(tmp_path)
    try:
        with pytest.raises(FilledNotionalUnavailable, match="pending anchor transition"):
            DailyFilledNotional.review_quarantine(
                path,
                anchor_path=ledger.anchor_path,
                anchor_key=ANCHOR_KEY,
                monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
            )

        assert _file_snapshot(path) == database_before
        assert _file_snapshot(ledger.anchor_path) == anchor_before
        assert _path_inventory(tmp_path) == paths_before
    finally:
        release_writer.set()
        writer_thread.join(timeout=5)

    assert not writer_thread.is_alive()
    assert len(writer_result) == 1
    assert not isinstance(writer_result[0], BaseException)


def test_review_quarantine_translates_pending_writer_after_constructor(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    writer = _service(path)
    pending_published = threading.Event()
    release_writer = threading.Event()
    writer_result: list[object] = []
    writer_thread: list[threading.Thread] = []
    original_commit = writer._commit_database
    original_require = DailyFilledNotional._require_exact_anchor
    review_require_calls = 0
    database_at_pending: list[tuple[bytes, int, int]] = []
    anchor_at_pending: list[tuple[bytes, int, int]] = []
    paths_at_pending: list[tuple[str, ...]] = []

    def hold_after_pending(connection):
        pending_published.set()
        if not release_writer.wait(timeout=5):
            raise sqlite3.OperationalError("test timed out releasing pending writer")
        original_commit(connection)

    monkeypatch.setattr(writer, "_commit_database", hold_after_pending)

    def write() -> None:
        try:
            writer_result.append(writer.record_fill(_fill("pending-after-review-construction")))
        except BaseException as exc:  # pragma: no cover - asserted below
            writer_result.append(exc)

    def start_writer_before_review_read(self, state):
        nonlocal review_require_calls
        if self._review_only:
            review_require_calls += 1
            if review_require_calls == 3:
                thread = threading.Thread(target=write)
                writer_thread.append(thread)
                thread.start()
                if not pending_published.wait(timeout=5):
                    raise RuntimeError("test timed out waiting for pending writer")
                database_at_pending.append(_file_snapshot(path))
                anchor_at_pending.append(_file_snapshot(ledger.anchor_path))
                paths_at_pending.append(_path_inventory(tmp_path))
        return original_require(self, state)

    monkeypatch.setattr(
        DailyFilledNotional, "_require_exact_anchor", start_writer_before_review_read
    )

    try:
        with pytest.raises(FilledNotionalUnavailable, match="pending anchor transition"):
            DailyFilledNotional.review_quarantine(
                path,
                anchor_path=ledger.anchor_path,
                anchor_key=ANCHOR_KEY,
                monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
            )

        assert _file_snapshot(path) == database_at_pending[0]
        assert _file_snapshot(ledger.anchor_path) == anchor_at_pending[0]
        assert _path_inventory(tmp_path) == paths_at_pending[0]
    finally:
        release_writer.set()
        if writer_thread:
            writer_thread[0].join(timeout=5)

    assert review_require_calls == 3
    assert writer_thread and not writer_thread[0].is_alive()
    assert len(writer_result) == 1
    assert not isinstance(writer_result[0], BaseException)


@pytest.mark.parametrize("missing_artifact", ["database", "anchor"])
def test_review_requires_existing_artifacts_without_creating_or_mutating(
    tmp_path, missing_artifact
):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    target = path if missing_artifact == "database" else ledger.anchor_path
    survivor = ledger.anchor_path if missing_artifact == "database" else path
    target.unlink()
    survivor_before = _file_snapshot(survivor)
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(FilledNotionalUnavailable, match="review requires existing"):
        DailyFilledNotional.review_quarantine(
            path,
            anchor_path=ledger.anchor_path,
            anchor_key=ANCHOR_KEY,
            monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
        )

    assert not target.exists()
    assert _file_snapshot(survivor) == survivor_before
    assert _path_inventory(tmp_path) == paths_before


@pytest.mark.parametrize("mistyped_artifact", ["database", "anchor"])
def test_review_rejects_mistyped_artifacts_without_mutating_paths(tmp_path, mistyped_artifact):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    target = path if mistyped_artifact == "database" else ledger.anchor_path
    survivor = ledger.anchor_path if mistyped_artifact == "database" else path
    preserved = tmp_path / f"preserved-{mistyped_artifact}"
    target.rename(preserved)
    target.mkdir(mode=0o700)
    target_identity = (os.lstat(target).st_dev, os.lstat(target).st_ino)
    preserved_before = _file_snapshot(preserved)
    survivor_before = _file_snapshot(survivor)
    paths_before = _path_inventory(tmp_path)

    with pytest.raises(FilledNotionalUnavailable, match="non-symlink regular"):
        DailyFilledNotional.review_quarantine(
            path,
            anchor_path=ledger.anchor_path,
            anchor_key=ANCHOR_KEY,
            monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
        )

    assert target.is_dir()
    assert (os.lstat(target).st_dev, os.lstat(target).st_ino) == target_identity
    assert _file_snapshot(preserved) == preserved_before
    assert _file_snapshot(survivor) == survivor_before
    assert _path_inventory(tmp_path) == paths_before


@pytest.mark.parametrize("deleted_artifact", ["database", "anchor"])
def test_review_does_not_recreate_artifact_deleted_after_preflight(
    tmp_path, monkeypatch, deleted_artifact
):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    target = path if deleted_artifact == "database" else ledger.anchor_path
    survivor = ledger.anchor_path if deleted_artifact == "database" else path
    preserved = tmp_path / f"deleted-during-review-{deleted_artifact}"
    target_before = _file_snapshot(target)
    survivor_before = _file_snapshot(survivor)
    paths_after_deletion: list[tuple[str, ...]] = []
    original_require = DailyFilledNotional._require_review_artifacts

    def delete_after_preflight(self):
        original_require(self)
        target.rename(preserved)
        paths_after_deletion.append(_path_inventory(tmp_path))

    monkeypatch.setattr(DailyFilledNotional, "_require_review_artifacts", delete_after_preflight)

    with pytest.raises(FilledNotionalUnavailable):
        DailyFilledNotional.review_quarantine(
            path,
            anchor_path=ledger.anchor_path,
            anchor_key=ANCHOR_KEY,
            monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
        )

    assert not target.exists()
    assert _file_snapshot(preserved) == target_before
    assert _file_snapshot(survivor) == survivor_before
    assert _path_inventory(tmp_path) == paths_after_deletion[0]


def test_authoritative_service_requires_independent_monotonic_verifier(tmp_path):
    path = tmp_path / "notional.db"
    anchor_directory = tmp_path / "protected-anchor"
    anchor_directory.mkdir(mode=0o700)

    with pytest.raises(FilledNotionalUnavailable, match="monotonic verifier is required"):
        DailyFilledNotional(
            path,
            anchor_path=anchor_directory / "notional.db.anchor",
            anchor_key=ANCHOR_KEY,
            monotonic_verifier=None,  # type: ignore[arg-type]
            account_id="DU12345",
            portfolio_id="default",
        )

    assert not path.exists()
    assert not (anchor_directory / "notional.db.anchor").exists()


def test_ledger_replacement_is_detected_by_anchor_inode_binding(tmp_path):
    path = tmp_path / "notional.db"
    original = _service(path)
    original.record_fill(_fill("original"))
    other_path = tmp_path / "other" / "replacement.db"
    other_path.parent.mkdir(mode=0o700)
    replacement = _service(other_path)
    replacement.record_fill(_fill("replacement"))

    os.replace(other_path, path)

    with pytest.raises(FilledNotionalIntegrityError, match="bind this ledger identity"):
        _service(path)


def test_anchor_replacement_with_other_valid_anchor_fails_closed(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("original"))
    other_path = tmp_path / "other" / "other.db"
    other_path.parent.mkdir(mode=0o700)
    other = _service(other_path)
    other.record_fill(_fill("other"))

    shutil.copyfile(other.anchor_path, ledger.anchor_path)
    os.chmod(ledger.anchor_path, 0o600)

    with pytest.raises(FilledNotionalIntegrityError, match="bind this ledger identity"):
        _service(path)


def test_anchor_directory_replacement_cannot_transfer_authority(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("original"))
    anchor_directory = ledger.anchor_path.parent
    moved_directory = tmp_path / "original-anchor-directory"
    original_anchor = ledger.anchor_path.read_bytes()

    os.rename(anchor_directory, moved_directory)
    anchor_directory.mkdir(mode=0o700)
    replacement_anchor = anchor_directory / ledger.anchor_path.name
    replacement_anchor.write_bytes(original_anchor)
    os.chmod(replacement_anchor, 0o600)

    with pytest.raises(FilledNotionalIntegrityError, match="directory identity changed"):
        ledger.current_gross_filled_notional()
    assert replacement_anchor.read_bytes() == original_anchor


def test_anchor_directory_swap_during_replace_is_rejected_without_authority_transfer(
    tmp_path, monkeypatch
):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    anchor_directory = ledger.anchor_path.parent
    moved_directory = tmp_path / "moved-bound-anchor"
    original_anchor = ledger.anchor_path.read_bytes()
    real_replace = ledger_module.os.replace
    swapped = False

    def race_replace(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped and destination == ledger.anchor_path.name:
            swapped = True
            os.rename(anchor_directory, moved_directory)
            anchor_directory.mkdir(mode=0o700)
            replacement_anchor = anchor_directory / ledger.anchor_path.name
            replacement_anchor.write_bytes(original_anchor)
            os.chmod(replacement_anchor, 0o600)
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(ledger_module.os, "replace", race_replace)

    with pytest.raises(FilledNotionalIntegrityError, match="identity changed during"):
        ledger.record_fill(_fill("racing-fill"))

    replacement_anchor = anchor_directory / ledger.anchor_path.name
    assert swapped is True
    assert replacement_anchor.read_bytes() == original_anchor
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM daily_filled_notional_records"
        ).fetchone() == (0,)


def test_anchor_content_or_external_key_mismatch_fails_closed(tmp_path):
    path = tmp_path / "notional.db"
    ledger = _service(path)
    ledger.record_fill(_fill("anchored"))

    with pytest.raises(FilledNotionalIntegrityError, match="HMAC is invalid"):
        DailyFilledNotional(
            path,
            anchor_path=ledger.anchor_path,
            anchor_key=b"different-independent-key-material-32-bytes",
            monotonic_verifier=_MONOTONIC_VERIFIERS[str(path)],
            account_id="DU12345",
            portfolio_id="default",
        )

    payload = bytearray(ledger.anchor_path.read_bytes())
    payload[payload.index(b"a")] = ord("b")
    ledger.anchor_path.write_bytes(payload)
    os.chmod(ledger.anchor_path, 0o600)
    with pytest.raises(FilledNotionalIntegrityError):
        _service(path)


def test_hostile_global_decimal_context_cannot_round_or_overflow_accounting(tmp_path):
    saved = getcontext().copy()
    try:
        hostile = getcontext()
        hostile.prec = 1
        hostile.Emax = 1
        hostile.Emin = -1
        hostile.clamp = 1
        hostile.traps[InvalidOperation] = True
        ledger = _service(tmp_path / "notional.db")
        result = ledger.record_fill(
            _fill(
                "hostile-context",
                quantity="12345678901234567890.123456789",
                price="1.000000001",
            )
        )
        assert result.gross_filled_notional == Decimal("12345678913580246791.358024679123456789")
    finally:
        setcontext(saved)


def test_decimal_exception_is_typed_and_latches_unavailable(tmp_path, monkeypatch):
    ledger = _service(tmp_path / "notional.db")

    def fail_decimal(*args, **kwargs):
        raise InvalidOperation("injected decimal fault")

    monkeypatch.setattr(ledger_module, "_exact_multiply", fail_decimal)
    with pytest.raises(FilledNotionalUnavailable, match="decimal failed closed"):
        ledger.record_fill(_fill("decimal-fault"))
    with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
        ledger.current_gross_filled_notional()


@pytest.mark.parametrize("schema_version", [1, 2])
def test_legacy_schema_requires_explicit_copy_migration_without_anchor_creation(
    tmp_path, schema_version
):
    path = tmp_path / f"legacy-v{schema_version}.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE daily_filled_notional_schema (
                singleton INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL
            )
            """)
        connection.execute(
            "INSERT INTO daily_filled_notional_schema VALUES (1, ?)", (schema_version,)
        )
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(FilledNotionalMigrationRequired, match="reviewed copy migration to v3"):
        _service(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not _anchor_path(path).exists()


def test_sigkill_v1_hot_journal_is_preserved_byte_for_byte(tmp_path):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("SIGKILL unavailable on this host")
    path = tmp_path / "legacy-hot-v1.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("""
            CREATE TABLE daily_filled_notional_schema (
                singleton INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL
            )
            """)
        connection.execute("INSERT INTO daily_filled_notional_schema VALUES (1, 1)")
        connection.execute("CREATE TABLE legacy_payload (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO legacy_payload VALUES (1, 'original')")
    script = f"""
import signal
import sqlite3

connection = sqlite3.connect({str(path)!r}, isolation_level=None)
connection.execute('PRAGMA journal_mode=DELETE')
connection.execute('PRAGMA synchronous=FULL')
connection.execute('BEGIN IMMEDIATE')
connection.execute("UPDATE legacy_payload SET value = 'uncommitted' WHERE id = 1")
print('READY', flush=True)
signal.pause()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    journal = Path(f"{path}-journal")
    assert journal.exists() and journal.stat().st_size > 0
    database_before = path.read_bytes()
    journal_before = journal.read_bytes()
    database_stat_before = path.stat()
    journal_stat_before = journal.stat()

    with pytest.raises(
        FilledNotionalUnavailable, match="authenticated current-schema recovery anchor"
    ):
        _service(path)

    assert path.read_bytes() == database_before
    assert journal.read_bytes() == journal_before
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (
        database_stat_before.st_ino,
        database_stat_before.st_mtime_ns,
    )
    assert (journal.stat().st_ino, journal.stat().st_mtime_ns) == (
        journal_stat_before.st_ino,
        journal_stat_before.st_mtime_ns,
    )
    assert not _anchor_path(path).exists()


def test_spilled_uncommitted_current_version_cannot_authorize_legacy_recovery(tmp_path):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("SIGKILL unavailable on this host")
    path = tmp_path / "legacy-spilled-current-version.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("""
            CREATE TABLE daily_filled_notional_schema (
                singleton INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL
            )
            """)
        connection.execute("INSERT INTO daily_filled_notional_schema VALUES (1, 1)")
        connection.execute("CREATE TABLE legacy_payload (id INTEGER PRIMARY KEY, value BLOB)")
        connection.executemany(
            "INSERT INTO legacy_payload VALUES (?, ?)",
            ((index, b"a" * 4096) for index in range(1, 257)),
        )
    script = f"""
import signal
import sqlite3

connection = sqlite3.connect({str(path)!r}, isolation_level=None)
connection.execute('PRAGMA journal_mode=DELETE')
connection.execute('PRAGMA synchronous=FULL')
connection.execute('PRAGMA cache_size=1')
connection.execute('PRAGMA cache_spill=ON')
connection.execute('BEGIN IMMEDIATE')
connection.execute('UPDATE daily_filled_notional_schema SET schema_version = 3')
connection.execute("UPDATE legacy_payload SET value = randomblob(4096)")
print('READY', flush=True)
signal.pause()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    journal = Path(f"{path}-journal")
    assert journal.exists() and journal.stat().st_size > 0
    with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute(
            "SELECT schema_version FROM daily_filled_notional_schema"
        ).fetchone() == (3,)
    database_before = path.read_bytes()
    journal_before = journal.read_bytes()
    database_stat_before = path.stat()
    journal_stat_before = journal.stat()

    with pytest.raises(
        FilledNotionalUnavailable, match="authenticated current-schema recovery anchor"
    ):
        _service(path)

    assert path.read_bytes() == database_before
    assert journal.read_bytes() == journal_before
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (
        database_stat_before.st_ino,
        database_stat_before.st_mtime_ns,
    )
    assert (journal.stat().st_ino, journal.stat().st_mtime_ns) == (
        journal_stat_before.st_ino,
        journal_stat_before.st_mtime_ns,
    )
    assert not _anchor_path(path).exists()


def test_authenticated_anchor_does_not_destroy_spilled_legacy_journal(tmp_path):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("SIGKILL unavailable on this host")
    path = tmp_path / "anchored-legacy-spill.db"
    ledger = _service(path)

    with sqlite3.connect(path) as connection:
        for trigger in ledger_module._TRIGGER_SQL:
            connection.execute(f'DROP TRIGGER "{trigger}"')
        connection.execute("ALTER TABLE daily_filled_notional_schema RENAME TO discarded_v3_schema")
        connection.execute("""
            CREATE TABLE daily_filled_notional_schema (
                singleton INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL
            )
            """)
        connection.execute("INSERT INTO daily_filled_notional_schema VALUES (1, 1)")
        connection.execute("DROP TABLE discarded_v3_schema")
        connection.execute("CREATE TABLE spill_payload (id INTEGER PRIMARY KEY, value BLOB)")
        connection.executemany(
            "INSERT INTO spill_payload VALUES (?, ?)",
            ((index, b"a" * 4096) for index in range(1, 257)),
        )

    script = f"""
import signal
import sqlite3

connection = sqlite3.connect({str(path)!r}, isolation_level=None)
connection.execute('PRAGMA journal_mode=DELETE')
connection.execute('PRAGMA synchronous=FULL')
connection.execute('PRAGMA cache_size=1')
connection.execute('PRAGMA cache_spill=ON')
connection.execute('BEGIN IMMEDIATE')
connection.execute('UPDATE daily_filled_notional_schema SET schema_version = 2')
connection.execute('UPDATE spill_payload SET value = randomblob(4096)')
print('READY', flush=True)
signal.pause()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    journal = Path(f"{path}-journal")
    assert journal.exists() and journal.stat().st_size > 0
    with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute(
            "SELECT schema_version FROM daily_filled_notional_schema"
        ).fetchone() == (2,)
    database_before = _file_snapshot(path)
    journal_before = _file_snapshot(journal)
    anchor_before = _file_snapshot(ledger.anchor_path)

    with pytest.raises(FilledNotionalMigrationRequired, match="reviewed copy migration"):
        _service(path)

    assert _file_snapshot(path) == database_before
    assert _file_snapshot(journal) == journal_before
    assert _file_snapshot(ledger.anchor_path) == anchor_before


def test_future_schema_version_fails_closed_without_downgrade(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE daily_filled_notional_schema (
                singleton INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL
            )
            """)
        connection.execute("INSERT INTO daily_filled_notional_schema VALUES (1, 99)")

    with pytest.raises(FilledNotionalIntegrityError, match="unsupported future"):
        _service(path)
    assert not _anchor_path(path).exists()
