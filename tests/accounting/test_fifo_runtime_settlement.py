"""PR4C atomic runtime FIFO settlement and recovery evidence."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from robo_trader.accounting import AccountingEpoch, FifoAccountingConflict, FifoLedger
from robo_trader.accounting.fifo import FillSide
from robo_trader.accounting.fifo_fixture_migration import (
    _TABLE_SQL,
    _TRIGGER_SQL,
    FifoFixtureMigrationError,
    migrate_fifo_fixture_database,
)
from robo_trader.accounting.fifo_runtime import (
    LOCAL_PAPER_COMMISSION_SOURCE,
    FifoRuntimeSettlementError,
    RuntimePaperFillEvidence,
    append_runtime_fill_in_transaction,
    reduction_side_to_fifo,
    verify_runtime_fill_in_transaction,
)


def _connection() -> tuple[sqlite3.Connection, datetime]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    migrate_fifo_fixture_database(connection, expected_path=None)
    effective = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
    FifoLedger(connection).create_epoch(
        AccountingEpoch(
            epoch_id="fepoch-" + ("1" * 32),
            execution_domain_scope="paper-simulator-v1",
            account_scope="acct_v1_" + ("2" * 64),
            portfolio_id="portfolio-a",
            source_fingerprint="3" * 64,
            effective_at=effective,
            created_at=effective,
        )
    )
    return connection, effective


def _evidence(
    sequence: int,
    *,
    side: FillSide,
    quantity: str,
    price: str,
    commission_minor: int,
    occurred_at: datetime,
    con_id: int = 265598,
    symbol: str = "AAPL",
) -> RuntimePaperFillEvidence:
    identity = hashlib.sha256(f"runtime-fill-{sequence}".encode()).hexdigest()
    return RuntimePaperFillEvidence(
        execution_domain_scope="paper-simulator-v1",
        account_scope="acct_v1_" + ("2" * 64),
        portfolio_id="portfolio-a",
        con_id=con_id,
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        execution_id=f"lpfill-{identity[:32]}",
        idempotency_key=identity,
        commission_minor=commission_minor,
        commission_currency="USD",
        commission_source=LOCAL_PAPER_COMMISSION_SOURCE,
        occurred_at=occurred_at,
    )


def test_partial_fills_and_commissions_project_exact_fifo() -> None:
    connection, effective = _connection()
    try:
        events = (
            _evidence(
                1,
                side=FillSide.BUY,
                quantity="10",
                price="100",
                commission_minor=101,
                occurred_at=effective,
            ),
            _evidence(
                2,
                side=FillSide.SELL,
                quantity="4",
                price="110",
                commission_minor=51,
                occurred_at=effective + timedelta(seconds=1),
            ),
            _evidence(
                3,
                side=FillSide.SELL,
                quantity="6",
                price="120",
                commission_minor=-25,
                occurred_at=effective + timedelta(seconds=2),
            ),
        )
        projections = []
        for event in events:
            connection.execute("BEGIN IMMEDIATE")
            projections.append(append_runtime_fill_in_transaction(connection, event))
            connection.commit()

        first, partial, final = projections
        assert first.signed_quantity == Decimal("10")
        assert first.average_cost == Decimal("100")
        assert partial.signed_quantity == Decimal("6")
        assert partial.fill_realized_pnl == Decimal("39.09")
        assert final.signed_quantity == Decimal("0")
        assert final.average_cost is None
        assert final.fill_realized_pnl == Decimal("119.64")
        assert final.total_realized_pnl == Decimal("158.73")
        assert final.cumulative_commission_minor == 127
        assert connection.execute("PRAGMA foreign_key_check").fetchone() is None
    finally:
        connection.close()


def test_epoch_realized_pnl_accumulates_across_symbols() -> None:
    connection, effective = _connection()
    try:
        events = (
            _evidence(
                1,
                side=FillSide.BUY,
                quantity="1",
                price="100",
                commission_minor=0,
                occurred_at=effective,
            ),
            _evidence(
                2,
                side=FillSide.SELL,
                quantity="1",
                price="110",
                commission_minor=0,
                occurred_at=effective + timedelta(seconds=1),
            ),
            _evidence(
                3,
                side=FillSide.BUY,
                quantity="1",
                price="100",
                commission_minor=0,
                occurred_at=effective + timedelta(seconds=2),
                con_id=272093,
                symbol="MSFT",
            ),
            _evidence(
                4,
                side=FillSide.SELL,
                quantity="1",
                price="120",
                commission_minor=0,
                occurred_at=effective + timedelta(seconds=3),
                con_id=272093,
                symbol="MSFT",
            ),
        )
        projections = []
        for event in events:
            connection.execute("BEGIN IMMEDIATE")
            projections.append(append_runtime_fill_in_transaction(connection, event))
            connection.commit()

        first_close, second_close = projections[1], projections[3]
        assert first_close.fill_realized_pnl == Decimal("10")
        assert first_close.epoch_realized_pnl == Decimal("10")
        assert second_close.fill_realized_pnl == Decimal("20")
        assert second_close.epoch_realized_pnl == Decimal("30")
        assert second_close.total_realized_pnl == Decimal("30")
    finally:
        connection.close()


def test_epoch_realized_pnl_is_independent_of_ambient_decimal_precision() -> None:
    connection, effective = _connection()
    try:
        events = (
            _evidence(
                1,
                side=FillSide.BUY,
                quantity="1",
                price="10000",
                commission_minor=0,
                occurred_at=effective,
            ),
            _evidence(
                2,
                side=FillSide.SELL,
                quantity="1",
                price="22345.67",
                commission_minor=0,
                occurred_at=effective + timedelta(seconds=1),
            ),
            _evidence(
                3,
                side=FillSide.BUY,
                quantity="1",
                price="20000",
                commission_minor=0,
                occurred_at=effective + timedelta(seconds=2),
                con_id=272093,
                symbol="MSFT",
            ),
            _evidence(
                4,
                side=FillSide.SELL,
                quantity="1",
                price="7654.34",
                commission_minor=0,
                occurred_at=effective + timedelta(seconds=3),
                con_id=272093,
                symbol="MSFT",
            ),
        )
        with localcontext() as context:
            context.prec = 6
            for event in events:
                connection.execute("BEGIN IMMEDIATE")
                projection = append_runtime_fill_in_transaction(connection, event)
                connection.commit()

        assert projection.fill_realized_pnl == Decimal("-12345.66")
        assert projection.epoch_realized_pnl == Decimal("0.01")
        assert projection.total_realized_pnl == Decimal("0.01")
    finally:
        connection.close()


def test_runtime_fill_rejects_case_changed_fifo_constraint_before_writing() -> None:
    connection, effective = _connection()
    try:
        connection.execute("DROP TABLE fifo_commissions")
        connection.execute(_TABLE_SQL["fifo_commissions"].replace("'USD'", "'usd'"))
        for name, statement in _TRIGGER_SQL.items():
            if name.startswith("fifo_commissions_no_"):
                connection.execute(statement)

        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(FifoFixtureMigrationError, match="fifo_commissions is malformed"):
            append_runtime_fill_in_transaction(
                connection,
                _evidence(
                    1,
                    side=FillSide.BUY,
                    quantity="1",
                    price="100",
                    commission_minor=0,
                    occurred_at=effective,
                ),
            )
        connection.rollback()

        assert connection.execute("SELECT COUNT(*) FROM fifo_fills").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM fifo_commissions").fetchone() == (0,)
    finally:
        connection.close()


def test_exact_replay_is_read_only_and_conflict_fails_closed() -> None:
    connection, effective = _connection()
    evidence = _evidence(
        1,
        side=FillSide.BUY,
        quantity="2",
        price="10",
        commission_minor=3,
        occurred_at=effective,
    )
    try:
        connection.execute("BEGIN IMMEDIATE")
        first = append_runtime_fill_in_transaction(connection, evidence)
        connection.commit()
        before = connection.total_changes

        connection.execute("BEGIN IMMEDIATE")
        replay = append_runtime_fill_in_transaction(connection, evidence)
        connection.rollback()
        assert replay.replayed is True
        assert replay.state_fingerprint == first.state_fingerprint
        assert connection.total_changes == before

        connection.execute("BEGIN")
        verified = verify_runtime_fill_in_transaction(connection, evidence)
        connection.rollback()
        assert verified.replayed is True
        assert verified.state_fingerprint == first.state_fingerprint
        assert connection.total_changes == before

        conflicting = RuntimePaperFillEvidence(
            **{
                **{field: getattr(evidence, field) for field in evidence.__dataclass_fields__},
                "price": Decimal("11"),
            }
        )
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(FifoAccountingConflict, match="different data"):
            append_runtime_fill_in_transaction(connection, conflicting)
        connection.rollback()
        assert connection.execute("SELECT COUNT(*) FROM fifo_fills").fetchone() == (1,)
    finally:
        connection.close()


def test_reduction_side_mapping_handles_short_cover_without_guessing() -> None:
    assert reduction_side_to_fifo("SELL") is FillSide.SELL
    assert reduction_side_to_fifo("BUY_TO_COVER") is FillSide.BUY
    with pytest.raises(FifoRuntimeSettlementError, match="only reduction"):
        reduction_side_to_fifo("BUY")


def test_failure_injection_rolls_back_fifo_and_compatibility_projection() -> None:
    connection, effective = _connection()
    connection.execute(
        "CREATE TABLE compatibility_position(symbol TEXT PRIMARY KEY, quantity_text TEXT NOT NULL)"
    )
    connection.execute("INSERT INTO compatibility_position VALUES ('AAPL','0')")
    evidence = _evidence(
        1,
        side=FillSide.BUY,
        quantity="5",
        price="10",
        commission_minor=0,
        occurred_at=effective,
    )
    try:
        connection.execute("BEGIN IMMEDIATE")
        append_runtime_fill_in_transaction(connection, evidence)
        connection.execute(
            "UPDATE compatibility_position SET quantity_text='5' WHERE symbol='AAPL'"
        )
        connection.rollback()

        assert connection.execute("SELECT COUNT(*) FROM fifo_fills").fetchone() == (0,)
        assert connection.execute(
            "SELECT quantity_text FROM compatibility_position WHERE symbol='AAPL'"
        ).fetchone() == ("0",)
    finally:
        connection.close()


def test_missing_epoch_never_invents_a_fill() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    migrate_fifo_fixture_database(connection, expected_path=None)
    effective = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
    evidence = _evidence(
        1,
        side=FillSide.BUY,
        quantity="1",
        price="10",
        commission_minor=0,
        occurred_at=effective,
    )
    try:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(FifoRuntimeSettlementError, match="exactly one sealed FIFO epoch"):
            append_runtime_fill_in_transaction(connection, evidence)
        connection.rollback()
        assert connection.execute("SELECT COUNT(*) FROM fifo_fills").fetchone() == (0,)
    finally:
        connection.close()
