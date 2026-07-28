"""Adversarial tests for durable daily gross-filled-notional accounting."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
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
    FilledNotionalUnavailable,
    FillSide,
)

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")


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
) -> DailyFilledNotional:
    return DailyFilledNotional(
        path,
        account_id=account_id,
        portfolio_id=portfolio_id,
        clock=clock or MutableClock(datetime(2026, 7, 28, 20, 0, tzinfo=UTC)),
    )


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


def test_conflicting_duplicate_fails_closed_and_latches_instance(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    original = _fill("broker.exec.1")
    ledger.record_fill(original)

    with pytest.raises(FilledNotionalConflict, match="conflicting immutable evidence"):
        ledger.record_fill(replace(original, price=Decimal("999")))
    with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
        ledger.current_gross_filled_notional()

    # The conflicting claim was never written.  A fresh process can restore the
    # one durable execution and will fail again if the bad claim is replayed.
    restarted = _service(ledger.database_path)
    assert restarted.restored_gross_filled_notional == Decimal("20.5")


def test_restart_restores_same_day_total_from_append_only_ledger(tmp_path):
    path = tmp_path / "notional.db"
    first_process = _service(path)
    first_process.record_fill(_fill("exec-1", quantity="2", price="10"))
    first_process.record_fill(_fill("exec-2", quantity="4", price="2.5"))

    restarted = _service(path)

    assert restarted.restored_trading_date.isoformat() == "2026-07-28"
    assert restarted.restored_gross_filled_notional == Decimal("30")
    assert restarted.current_gross_filled_notional() == Decimal("30")


def test_scope_isolated_by_account_and_portfolio(tmp_path):
    path = tmp_path / "notional.db"
    default = _service(path)
    second_portfolio = _service(path, portfolio_id="income")
    second_account = _service(path, account_id="DU99999")

    default.record_fill(_fill("same-exec", quantity="1", price="10"))
    second_portfolio.record_fill(_fill("same-exec", quantity="2", price="10"))
    second_account.record_fill(_fill("same-exec", quantity="3", price="10"))

    assert default.current_gross_filled_notional() == Decimal("10")
    assert second_portfolio.current_gross_filled_notional() == Decimal("20")
    assert second_account.current_gross_filled_notional() == Decimal("30")


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
                       trading_date, notional_text, previous_hash, record_hash
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

    with pytest.raises(FilledNotionalIntegrityError, match="notional does not match"):
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

    with pytest.raises(FilledNotionalUnavailable, match="startup failed closed"):
        _service(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_current_query_is_read_only(tmp_path):
    ledger = _service(tmp_path / "notional.db")
    ledger.record_fill(_fill("exec-1"))
    before = ledger.database_path.read_bytes()

    assert ledger.current_gross_filled_notional() == Decimal("20.5")

    assert ledger.database_path.read_bytes() == before
    assert not Path(f"{ledger.database_path}-wal").exists()
