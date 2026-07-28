"""Adversarial tests for durable daily gross-filled-notional accounting."""

from __future__ import annotations

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


def test_reader_waits_for_concurrent_pending_writer_without_latching(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    writer = _service(path)
    reader = _service(path)
    pending_ready = threading.Event()
    reader_reconciling = threading.Event()
    reader_retrying = threading.Event()
    release_writer = threading.Event()
    original_commit = writer._commit_database
    original_resolve = reader._resolve_pending_anchor
    resolve_attempts = 0
    results: list[object] = []

    def gated_commit(connection):
        pending_ready.set()
        if not release_writer.wait(timeout=5):
            raise sqlite3.OperationalError("test timed out waiting to release writer")
        original_commit(connection)

    def observed_resolve():
        nonlocal resolve_attempts
        resolve_attempts += 1
        reader_reconciling.set()
        if resolve_attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        reader_retrying.set()
        original_resolve()

    monkeypatch.setattr(writer, "_commit_database", gated_commit)
    monkeypatch.setattr(reader, "_resolve_pending_anchor", observed_resolve)

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
    assert reader_reconciling.wait(timeout=5)
    assert reader_retrying.wait(timeout=5)
    release_writer.set()
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)

    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert not any(isinstance(result, BaseException) for result in results)
    assert resolve_attempts >= 2
    assert Decimal("20.5") in results
    assert reader.current_gross_filled_notional() == Decimal("20.5")


def test_reader_snapshot_rechecks_anchor_before_returning_during_writer_race(tmp_path, monkeypatch):
    path = tmp_path / "notional.db"
    writer = _service(path)
    reader = _service(path)
    snapshot_total_ready = threading.Event()
    writer_pending = threading.Event()
    original_total = reader._total_for_date
    original_commit = writer._commit_database
    total_calls = 0
    results: dict[str, object] = {}

    def gated_total(connection, trading_day):
        nonlocal total_calls
        total = original_total(connection, trading_day)
        total_calls += 1
        if total_calls == 1:
            snapshot_total_ready.set()
            if not writer_pending.wait(timeout=5):
                raise sqlite3.OperationalError("test writer did not reach pending state")
        return total

    def observed_commit(connection):
        writer_pending.set()
        original_commit(connection)

    monkeypatch.setattr(reader, "_total_for_date", gated_total)
    monkeypatch.setattr(writer, "_commit_database", observed_commit)

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
    reader_thread.join(timeout=5)
    writer_thread.join(timeout=5)

    assert not reader_thread.is_alive()
    assert not writer_thread.is_alive()
    assert not isinstance(results.get("read"), BaseException)
    assert not isinstance(results.get("write"), BaseException)
    assert results["read"] == Decimal("20.5")
    assert total_calls >= 2
    assert reader._failed_reason is None


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
        if readonly and not immutable:
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
