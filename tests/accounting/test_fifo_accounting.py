from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from robo_trader.accounting import (
    AccountingEpoch,
    FifoAccountingConflict,
    FifoAccountingOrderingError,
    FifoAccountingValidationError,
    FifoLedger,
    FillEvent,
    FillSide,
    assert_fifo_accounting_schema,
    migrate_fifo_fixture_database,
)
from robo_trader.accounting.fifo_fixture_migration import FifoFixtureMigrationError

NOW = datetime(2026, 7, 28, 16, 0, tzinfo=timezone.utc)
EPOCH_ID = "fepoch-" + "1" * 32


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identifier(prefix: str, number: int) -> str:
    return f"{prefix}-{number:032x}"


@pytest.fixture
def connection(tmp_path):
    path = tmp_path / "accounting.fifo-fixture.sqlite3"
    connection = sqlite3.connect(path)
    migrate_fifo_fixture_database(connection, expected_path=path)
    yield connection
    connection.close()


@pytest.fixture
def ledger(connection):
    ledger = FifoLedger(connection)
    ledger.create_epoch(
        AccountingEpoch(
            epoch_id=EPOCH_ID,
            execution_domain_scope="paper-simulator-v1",
            account_scope="acct_v1_test_fixture",
            portfolio_id="default",
            source_fingerprint=_digest("empty fixture epoch"),
            effective_at=NOW,
            created_at=NOW,
        )
    )
    return ledger


def _fill(
    sequence: int,
    side: FillSide,
    quantity: str,
    price: str,
    commission_minor: int = 0,
    *,
    con_id: int = 265598,
    symbol: str = "AAPL",
) -> FillEvent:
    return FillEvent(
        epoch_id=EPOCH_ID,
        fill_id=_identifier("ffill", sequence),
        commission_id=_identifier("fcomm", sequence),
        event_sequence=sequence,
        execution_id=f"execution-{sequence}",
        idempotency_key=f"idempotency-{sequence}",
        con_id=con_id,
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        commission_minor=commission_minor,
        occurred_at=NOW + timedelta(seconds=sequence),
        recorded_at=NOW + timedelta(seconds=sequence),
    )


def _table_count(connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _insert_fill_row(connection: sqlite3.Connection, event: FillEvent) -> None:
    connection.execute(
        """
        INSERT INTO fifo_fills(
            fill_id, epoch_id, event_sequence, execution_id, idempotency_key,
            con_id, symbol, side, quantity_text, price_text, occurred_at,
            recorded_at, payload_fingerprint
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.fill_id,
            event.epoch_id,
            event.event_sequence,
            event.execution_id,
            event.idempotency_key,
            event.con_id,
            event.symbol,
            event.side.value,
            format(event.quantity, "f"),
            format(event.price, "f"),
            _utc_text(event.occurred_at),
            _utc_text(event.recorded_at),
            event.fingerprint(),
        ),
    )


def _snapshot_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_fixture_migration_is_exact_idempotent_and_foreign_keys_are_on(connection):
    assert_fifo_accounting_schema(connection)
    migrate_fifo_fixture_database(
        connection,
        expected_path=connection.execute("PRAGMA database_list").fetchone()[2],
    )
    assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert connection.execute(
        "SELECT version FROM fifo_schema_migrations WHERE component='fifo_accounting'"
    ).fetchall() == [(1,), (2,)]


def test_fixture_migration_rejects_production_style_filename(tmp_path):
    path = tmp_path / "trading_data.db"
    connection = sqlite3.connect(path)
    with pytest.raises(FifoFixtureMigrationError, match="must end with"):
        migrate_fifo_fixture_database(connection, expected_path=path)
    assert connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'fifo_%'"
    ).fetchone() == (0,)
    connection.close()


def test_fixture_migration_rejects_mismatched_connection(tmp_path):
    actual = tmp_path / "actual.fifo-fixture.sqlite3"
    expected = tmp_path / "expected.fifo-fixture.sqlite3"
    expected.touch()
    connection = sqlite3.connect(actual)
    with pytest.raises(FifoFixtureMigrationError, match="does not match"):
        migrate_fifo_fixture_database(connection, expected_path=expected)
    connection.close()


def test_fixture_migration_rejects_hard_link_alias_before_mutation(tmp_path):
    production_path = tmp_path / "trading_data.db"
    production = sqlite3.connect(production_path)
    production.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
    production.execute("INSERT INTO sentinel VALUES ('preserve')")
    production.commit()
    production.close()

    alias_path = tmp_path / "alias.fifo-fixture.sqlite3"
    os.link(production_path, alias_path)
    connection = sqlite3.connect(alias_path)
    with pytest.raises(FifoFixtureMigrationError, match="hard link"):
        migrate_fifo_fixture_database(connection, expected_path=alias_path)
    connection.close()

    production = sqlite3.connect(production_path)
    assert production.execute("SELECT value FROM sentinel").fetchone() == ("preserve",)
    assert production.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'fifo_%'"
    ).fetchone() == (0,)
    production.close()


def test_fixture_migration_rejects_copied_or_renamed_nonfixture_schema(tmp_path):
    path = tmp_path / "renamed-production.fifo-fixture.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
    connection.execute("INSERT INTO sentinel VALUES ('preserve')")
    connection.commit()

    with pytest.raises(FifoFixtureMigrationError, match="requires an empty database"):
        migrate_fifo_fixture_database(connection, expected_path=path)

    assert connection.execute("SELECT value FROM sentinel").fetchone() == ("preserve",)
    assert connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'fifo_%'"
    ).fetchone() == (0,)
    connection.close()


def test_fixture_migration_handles_sqlite_row_factory_exactly(tmp_path):
    path = tmp_path / "row-factory.fifo-fixture.sqlite3"
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    migrate_fifo_fixture_database(connection, expected_path=path)
    assert_fifo_accounting_schema(connection)
    migrate_fifo_fixture_database(connection, expected_path=path)
    assert_fifo_accounting_schema(connection)
    connection.close()


def test_exact_fifo_schema_plus_unrelated_table_is_not_a_fixture(tmp_path):
    path = tmp_path / "mixed-schema.fifo-fixture.sqlite3"
    connection = sqlite3.connect(path)
    migrate_fifo_fixture_database(connection, expected_path=path)
    connection.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
    connection.execute("INSERT INTO sentinel VALUES ('preserve')")
    connection.commit()

    with pytest.raises(FifoFixtureMigrationError, match="unexpected table sentinel"):
        assert_fifo_accounting_schema(connection)
    with pytest.raises(FifoFixtureMigrationError, match="unexpected table sentinel"):
        migrate_fifo_fixture_database(connection, expected_path=path)

    assert connection.execute("SELECT value FROM sentinel").fetchone() == ("preserve",)
    connection.close()


def test_interrupted_fixture_migration_rolls_back_all_schema_objects(tmp_path):
    path = tmp_path / "interrupted.fifo-fixture.sqlite3"
    connection = sqlite3.connect(path)

    def reject_triggers(action, _arg1, _arg2, _database, _source):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TRIGGER else sqlite3.SQLITE_OK

    connection.set_authorizer(reject_triggers)
    with pytest.raises(sqlite3.DatabaseError):
        migrate_fifo_fixture_database(connection, expected_path=path)
    connection.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
    assert connection.in_transaction is False
    assert connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'fifo_%'"
    ).fetchone() == (0,)

    migrate_fifo_fixture_database(connection, expected_path=path)
    assert_fifo_accounting_schema(connection)
    connection.close()


def test_financial_tables_are_append_only(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "1", "10", 1))
    ledger.record_fill(_fill(2, FillSide.SELL, "1", "11", 1))
    for table in (
        "fifo_schema_migrations",
        "fifo_accounting_epochs",
        "fifo_fills",
        "fifo_commissions",
        "fifo_lot_openings",
        "fifo_lot_matches",
        "fifo_position_snapshots",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(f"DELETE FROM {table}")
        connection.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("UPDATE fifo_fills SET quantity_text='2'")
    connection.rollback()


def test_insert_or_replace_cannot_rewrite_any_append_only_identity(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "2", "10", 1))
    ledger.record_fill(_fill(2, FillSide.SELL, "1", "11", 1))
    for table in (
        "fifo_schema_migrations",
        "fifo_accounting_epochs",
        "fifo_fills",
        "fifo_commissions",
        "fifo_lot_openings",
        "fifo_lot_matches",
        "fifo_position_snapshots",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="identity is append-only"):
            connection.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM {table} LIMIT 1")
        connection.rollback()

    alternate_identity_rows = {
        "fifo_accounting_epochs": "fepoch-" + "2" * 32,
        "fifo_fills": _identifier("ffill", 99),
        "fifo_commissions": _identifier("fcomm", 99),
        "fifo_lot_openings": _identifier("flot", 99),
        "fifo_lot_matches": _identifier("fmatch", 99),
        "fifo_position_snapshots": _identifier("fsnap", 99),
    }
    for table, replacement_primary_key in alternate_identity_rows.items():
        row = list(connection.execute(f"SELECT * FROM {table} LIMIT 1").fetchone())
        row[0] = replacement_primary_key
        placeholders = ",".join("?" for _ in row)
        with pytest.raises(sqlite3.IntegrityError, match="identity is append-only"):
            connection.execute(
                f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})",
                row,
            )
        connection.rollback()

    with pytest.raises(sqlite3.IntegrityError, match="identity is append-only"):
        connection.execute("""
            INSERT INTO fifo_schema_migrations(component, version, description, applied_at)
            SELECT component, version, 'forged', applied_at FROM fifo_schema_migrations
            WHERE true
            ON CONFLICT(component, version) DO UPDATE SET description='forged'
            """)
    connection.rollback()
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_schema_constraints_reject_bad_ids_and_orphan_fills(connection):
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO fifo_accounting_epochs VALUES(
                'fepoch-a', 1, 'paper', 'account', 'portfolio', 'EMPTY_LEDGER',
                ?, '2026-07-28T16:00:00.000000Z', '2026-07-28T16:00:00.000000Z'
            )
            """,
            (_digest("bad"),),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO fifo_accounting_epochs VALUES(
                ?, 1, 'paper', 'account', 'portfolio', 'EMPTY_LEDGER',
                ?, '2026-07-28T16:00:00.000000Z', '2026-07-28T16:00:00.000000Z'
            )
            """,
            ("fepoch-" + "z" * 32, _digest("bad hex")),
        )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        connection.execute(
            """
            INSERT INTO fifo_fills VALUES(
                ?, ?, 1, 'execution', 'idempotency', 1, 'AAPL', 'BUY', '1', '1',
                '2026-07-28T16:00:00.000000Z', '2026-07-28T16:00:00.000000Z', ?
            )
            """,
            (_identifier("ffill", 1), _identifier("fepoch", 2), _digest("payload")),
        )


def test_integer_minor_units_reject_fractional_real_storage(connection, ledger):
    event = _fill(1, FillSide.BUY, "1", "10")
    _insert_fill_row(connection, event)
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO fifo_commissions VALUES (?, ?, ?, 1.5, 'USD', 2, ?)",
            (event.commission_id, event.epoch_id, event.fill_id, _utc_text(event.recorded_at)),
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO fifo_lot_openings VALUES(
                ?, ?, ?, NULL, 0, ?, ?, 'LONG', '1', '10', 1.5, 1, ?
            )
            """,
            (
                _identifier("flot", 1),
                event.epoch_id,
                event.fill_id,
                event.con_id,
                event.symbol,
                _utc_text(event.occurred_at),
            ),
        )
    connection.rollback()


def test_temp_fifo_shadows_created_after_ledger_init_fail_every_operation(connection, ledger):
    tables = (
        "fifo_schema_migrations",
        "fifo_accounting_epochs",
        "fifo_fills",
        "fifo_commissions",
        "fifo_lot_openings",
        "fifo_lot_matches",
        "fifo_position_snapshots",
    )
    before = {
        table: connection.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
        for table in tables
    }
    for table in tables:
        connection.execute(f"CREATE TEMP TABLE {table} AS SELECT * FROM main.{table}")

    with pytest.raises(FifoFixtureMigrationError, match="temporary FIFO"):
        ledger.record_fill(_fill(1, FillSide.BUY, "1", "10"))
    with pytest.raises(FifoFixtureMigrationError, match="temporary FIFO"):
        ledger.verify_epoch_integrity(EPOCH_ID)
    with pytest.raises(FifoFixtureMigrationError, match="temporary FIFO"):
        ledger.create_epoch(
            AccountingEpoch(
                epoch_id="fepoch-" + "2" * 32,
                execution_domain_scope="paper-simulator-v2",
                account_scope="acct_v1_test_fixture",
                portfolio_id="other",
                source_fingerprint=_digest("shadowed epoch"),
                effective_at=NOW,
                created_at=NOW,
            )
        )

    assert {
        table: connection.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
        for table in tables
    } == before


def test_foreign_main_trigger_targeting_fifo_table_is_rejected(connection):
    connection.execute("CREATE TABLE unrelated_audit(value TEXT NOT NULL)")
    connection.execute("""
        CREATE TRIGGER unrelated_capture
        AFTER INSERT ON fifo_opening_balances
        BEGIN
            INSERT INTO unrelated_audit VALUES ('unexpected mutation');
        END
        """)

    with pytest.raises(FifoFixtureMigrationError, match="foreign triggers"):
        FifoLedger(connection, allow_other_objects=True)

    assert connection.execute("SELECT COUNT(*) FROM unrelated_audit").fetchone() == (0,)


def test_foreign_main_trigger_body_referencing_fifo_table_is_rejected(connection):
    connection.execute("CREATE TABLE unrelated_source(value TEXT NOT NULL)")
    connection.execute("""
        CREATE TRIGGER unrelated_fifo_injector
        AFTER INSERT ON unrelated_source
        BEGIN
            INSERT INTO fifo_opening_balances(opening_balance_id) VALUES ('forged');
        END
        """)

    with pytest.raises(FifoFixtureMigrationError, match="foreign triggers"):
        FifoLedger(connection, allow_other_objects=True)


def test_durable_trigger_injected_after_construction_blocks_fill_mutation(connection, ledger):
    operational_ledger = FifoLedger(connection, allow_other_objects=True)
    connection.execute("""
        CREATE TRIGGER post_construction_fifo_injector
        AFTER INSERT ON fifo_position_snapshots
        BEGIN
            SELECT 1;
        END
        """)

    with pytest.raises(FifoFixtureMigrationError, match="foreign triggers"):
        operational_ledger.record_fill(_fill(1, FillSide.BUY, "1", "10"))

    assert _table_count(connection, "fifo_fills") == 0
    assert _table_count(connection, "fifo_position_snapshots") == 0


def test_foreign_temp_trigger_targeting_fifo_table_is_rejected(connection):
    connection.execute("""
        CREATE TEMP TRIGGER unrelated_temp_capture
        BEFORE INSERT ON main.fifo_fills
        BEGIN
            SELECT 1;
        END
        """)

    with pytest.raises(FifoFixtureMigrationError, match="temporary FIFO"):
        FifoLedger(connection, allow_other_objects=True)


def test_uppercase_temp_shadow_is_rejected_with_case_sensitive_like(connection, ledger):
    connection.execute("PRAGMA case_sensitive_like=ON")
    connection.execute("CREATE TEMP TABLE FIFO_FILLS AS SELECT * FROM main.fifo_fills")

    with pytest.raises(FifoFixtureMigrationError, match="temporary FIFO"):
        ledger.record_fill(_fill(1, FillSide.BUY, "1", "10"))

    assert _table_count(connection, "fifo_fills") == 0


@pytest.mark.parametrize("bad_decimal", ["abc", "01", "1.0", "0", "-1", "1e2", ".5", "5."])
def test_schema_rejects_noncanonical_or_nonpositive_fill_decimals(connection, ledger, bad_decimal):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
        connection.execute(
            """
            INSERT INTO fifo_fills(
                fill_id, epoch_id, event_sequence, execution_id, idempotency_key,
                con_id, symbol, side, quantity_text, price_text, occurred_at,
                recorded_at, payload_fingerprint
            ) VALUES (?, ?, 1, 'execution', 'idempotency', 1, 'AAPL', 'BUY', ?, '1',
                      '2026-07-28T16:00:00.000000Z',
                      '2026-07-28T16:00:00.000000Z', ?)
            """,
            (_identifier("ffill", 1), EPOCH_ID, bad_decimal, _digest("payload")),
        )
    connection.rollback()


def test_epoch_replay_is_idempotent_and_conflict_fails(connection):
    ledger = FifoLedger(connection)
    epoch = AccountingEpoch(
        epoch_id=EPOCH_ID,
        execution_domain_scope="paper-simulator-v1",
        account_scope="acct_v1_test_fixture",
        portfolio_id="default",
        source_fingerprint=_digest("empty fixture epoch"),
        effective_at=NOW,
        created_at=NOW,
    )
    assert ledger.create_epoch(epoch) == epoch
    assert ledger.create_epoch(epoch) == epoch
    with pytest.raises(FifoAccountingConflict, match="different data"):
        ledger.create_epoch(replace(epoch, created_at=NOW + timedelta(seconds=1)))
    assert _table_count(connection, "fifo_accounting_epochs") == 1


def test_legacy_epoch_cannot_be_created_or_projected(connection):
    with pytest.raises(FifoAccountingValidationError, match="PR4B"):
        AccountingEpoch(
            epoch_id=EPOCH_ID,
            execution_domain_scope="paper-simulator-v1",
            account_scope="acct_v1_test_fixture",
            portfolio_id="default",
            source_fingerprint=_digest("legacy"),
            effective_at=NOW,
            created_at=NOW,
            origin_kind="LEGACY_AGGREGATE_OPENING_BALANCE",
        )


def test_fill_replay_returns_same_projection_without_new_rows(connection, ledger):
    event = _fill(1, FillSide.BUY, "2.5", "10.125", 3)
    first = ledger.record_fill(event)
    before = {
        table: _table_count(connection, table)
        for table in (
            "fifo_fills",
            "fifo_commissions",
            "fifo_lot_openings",
            "fifo_lot_matches",
            "fifo_position_snapshots",
        )
    }
    replay = ledger.record_fill(event)
    assert replay.replayed is True
    assert replay.snapshot == first.snapshot
    assert replay.opened_lot_id == first.opened_lot_id
    assert {table: _table_count(connection, table) for table in before} == before


@pytest.mark.parametrize(
    "replacement",
    [
        {"price": Decimal("11")},
        {"execution_id": "different-execution"},
        {"idempotency_key": "different-idempotency"},
        {"commission_minor": 99},
    ],
)
def test_fill_identity_conflicts_roll_back(connection, ledger, replacement):
    event = _fill(1, FillSide.BUY, "2", "10", 1)
    ledger.record_fill(event)
    with pytest.raises(FifoAccountingConflict):
        ledger.record_fill(replace(event, **replacement))
    assert _table_count(connection, "fifo_fills") == 1
    assert _table_count(connection, "fifo_commissions") == 1


def test_fill_sequence_and_event_time_are_strict(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "1", "10"))
    with pytest.raises(FifoAccountingOrderingError, match="extend"):
        ledger.record_fill(_fill(3, FillSide.BUY, "1", "11"))
    early = replace(
        _fill(2, FillSide.BUY, "1", "11"),
        occurred_at=NOW,
        recorded_at=NOW + timedelta(seconds=2),
    )
    with pytest.raises(FifoAccountingOrderingError, match="precedes"):
        ledger.record_fill(early)
    assert _table_count(connection, "fifo_fills") == 1


def test_fill_cannot_precede_epoch_effective_time(connection, ledger):
    event = replace(
        _fill(1, FillSide.BUY, "1", "10"),
        occurred_at=NOW - timedelta(microseconds=1),
    )
    with pytest.raises(FifoAccountingOrderingError, match="epoch effective_at"):
        ledger.record_fill(event)
    assert _table_count(connection, "fifo_fills") == 0


@pytest.mark.parametrize(
    "second",
    [
        _fill(2, FillSide.BUY, "1", "11", con_id=265598, symbol="MSFT"),
        _fill(2, FillSide.BUY, "1", "11", con_id=272093, symbol="AAPL"),
    ],
)
def test_contract_identifier_and_symbol_binding_cannot_drift(connection, ledger, second):
    ledger.record_fill(_fill(1, FillSide.BUY, "1", "10"))
    with pytest.raises(FifoAccountingConflict, match="binding changed"):
        ledger.record_fill(second)
    assert _table_count(connection, "fifo_fills") == 1


def test_long_fifo_partial_fills_match_oldest_lot(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "3", "10"))
    ledger.record_fill(_fill(2, FillSide.BUY, "2", "20"))
    result = ledger.record_fill(_fill(3, FillSide.SELL, "4", "30"))
    rows = connection.execute(
        """
        SELECT matched_quantity_text, opening_price_text, gross_pnl_text
        FROM fifo_lot_matches WHERE closing_fill_id = ? ORDER BY match_ordinal
        """,
        (result.fill_id,),
    ).fetchall()
    assert rows == [("3", "10", "60"), ("1", "20", "10")]
    assert result.snapshot.signed_quantity == Decimal("1")
    assert result.snapshot.open_cost == Decimal("20")
    assert result.snapshot.cumulative_realized_pnl == Decimal("70")
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_short_fifo_partial_cover_and_profit(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.SELL, "2", "50"))
    ledger.record_fill(_fill(2, FillSide.SELL, "1", "40"))
    result = ledger.record_fill(_fill(3, FillSide.BUY, "2.5", "30"))
    rows = connection.execute(
        """
        SELECT matched_quantity_text, opening_price_text, gross_pnl_text
        FROM fifo_lot_matches WHERE closing_fill_id = ? ORDER BY match_ordinal
        """,
        (result.fill_id,),
    ).fetchall()
    assert rows == [("2", "50", "40"), ("0.5", "40", "5")]
    assert result.snapshot.signed_quantity == Decimal("-0.5")
    assert result.snapshot.open_cost == Decimal("20.0")
    assert result.snapshot.cumulative_realized_pnl == Decimal("45")
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_zero_crossing_fill_closes_then_opens_opposite_lot(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "2", "10", 2))
    result = ledger.record_fill(_fill(2, FillSide.SELL, "5", "12", 5))
    assert len(result.match_ids) == 1
    assert result.opened_lot_id is not None
    assert result.snapshot.signed_quantity == Decimal("-3")
    assert result.snapshot.open_cost == Decimal("36")
    allocations = connection.execute(
        """
        SELECT closing_commission_minor FROM fifo_lot_matches WHERE closing_fill_id = ?
        UNION ALL
        SELECT opening_commission_minor FROM fifo_lot_openings WHERE opening_fill_id = ?
        """,
        (result.fill_id, result.fill_id),
    ).fetchall()
    assert sorted(value[0] for value in allocations) == [2, 3]
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_commission_allocation_conserves_every_cent_across_fifo_segments(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "1", "10", 1))
    ledger.record_fill(_fill(2, FillSide.BUY, "1", "11", 1))
    ledger.record_fill(_fill(3, FillSide.BUY, "1", "12", 1))
    result = ledger.record_fill(_fill(4, FillSide.SELL, "3", "20", 2))
    rows = connection.execute(
        """
        SELECT opening_commission_minor, closing_commission_minor, realized_pnl_text
        FROM fifo_lot_matches WHERE closing_fill_id = ? ORDER BY match_ordinal
        """,
        (result.fill_id,),
    ).fetchall()
    assert rows == [(1, 1, "9.98"), (1, 1, "8.98"), (1, 0, "7.99")]
    assert sum(row[1] for row in rows) == 2
    assert result.snapshot.cumulative_realized_pnl == Decimal("26.95")
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_partial_closes_allocate_opening_commission_cumulatively(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "3", "10", 2))
    ledger.record_fill(_fill(2, FillSide.SELL, "1", "11"))
    ledger.record_fill(_fill(3, FillSide.SELL, "1", "11"))
    ledger.record_fill(_fill(4, FillSide.SELL, "1", "11"))
    allocations = connection.execute("""
        SELECT m.opening_commission_minor
        FROM fifo_lot_matches m JOIN fifo_fills f ON f.fill_id=m.closing_fill_id
        ORDER BY f.event_sequence
        """).fetchall()
    assert allocations == [(0,), (1,), (1,)]
    assert sum(value[0] for value in allocations) == 2
    assert ledger.record_fill(_fill(4, FillSide.SELL, "1", "11")).replayed is True
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_negative_commission_rebate_is_allocated_exactly(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.SELL, "2", "10", -1))
    result = ledger.record_fill(_fill(2, FillSide.BUY, "2", "8", -1))
    row = connection.execute(
        """
        SELECT opening_commission_minor, closing_commission_minor, realized_pnl_text
        FROM fifo_lot_matches WHERE closing_fill_id = ?
        """,
        (result.fill_id,),
    ).fetchone()
    assert row == (-1, -1, "4.02")
    assert result.snapshot.cumulative_commission_minor == -2
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_fractional_quantities_and_subpenny_prices_remain_exact(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "0.125", "10.000001", 0))
    result = ledger.record_fill(_fill(2, FillSide.SELL, "0.125", "10.000009", 0))
    assert result.snapshot.signed_quantity == Decimal("0")
    assert result.snapshot.open_cost is None
    assert result.snapshot.cumulative_realized_pnl == Decimal("0.000001000")
    row = connection.execute(
        "SELECT gross_pnl_text FROM fifo_lot_matches WHERE closing_fill_id = ?",
        (result.fill_id,),
    ).fetchone()
    assert row == ("0.000001",)
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_maximum_input_scales_multiply_without_context_rounding(connection, ledger):
    ledger.record_fill(_fill(1, FillSide.BUY, "0.000000000000000001", "0.000000000000000001"))
    result = ledger.record_fill(
        _fill(2, FillSide.SELL, "0.000000000000000001", "0.000000000000000002")
    )
    assert result.snapshot.cumulative_realized_pnl == Decimal(
        "0.000000000000000000000000000000000001"
    )
    assert connection.execute(
        "SELECT gross_pnl_text FROM fifo_lot_matches WHERE closing_fill_id = ?",
        (result.fill_id,),
    ).fetchone() == ("0.000000000000000000000000000000000001",)
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_snapshot_chain_is_per_asset_and_globally_sequenced(connection, ledger):
    first = ledger.record_fill(_fill(1, FillSide.BUY, "1", "10"))
    second = ledger.record_fill(_fill(2, FillSide.BUY, "2", "20", con_id=272093, symbol="MSFT"))
    third = ledger.record_fill(_fill(3, FillSide.SELL, "1", "11"))
    assert first.snapshot.previous_snapshot_id is None
    assert second.snapshot.previous_snapshot_id is None
    assert third.snapshot.previous_snapshot_id == first.snapshot.snapshot_id
    assert third.snapshot.previous_state_fingerprint == first.snapshot.state_fingerprint
    assert [
        row[0]
        for row in connection.execute(
            "SELECT event_sequence FROM fifo_position_snapshots ORDER BY event_sequence"
        )
    ] == [1, 2, 3]
    ledger.verify_epoch_integrity(EPOCH_ID)


def test_integrity_verifier_replays_every_intermediate_snapshot(connection, ledger):
    first = ledger.record_fill(_fill(1, FillSide.BUY, "1", "10"))
    second = ledger.record_fill(_fill(2, FillSide.BUY, "1", "20"))

    first_payload = {
        "con_id": first.snapshot.con_id,
        "cumulative_commission_minor": 0,
        "cumulative_realized_pnl": "0",
        "epoch_id": EPOCH_ID,
        "event_sequence": 1,
        "open_cost": "90",
        "open_lot_count": 1,
        "previous_snapshot_id": None,
        "previous_state_fingerprint": None,
        "signed_quantity": "9",
        "snapshot_id": first.snapshot.snapshot_id,
        "source_fill_id": first.fill_id,
        "symbol": first.snapshot.symbol,
    }
    bad_first_fingerprint = _snapshot_fingerprint(first_payload)
    second_payload = {
        "con_id": second.snapshot.con_id,
        "cumulative_commission_minor": 0,
        "cumulative_realized_pnl": "0",
        "epoch_id": EPOCH_ID,
        "event_sequence": 2,
        "open_cost": "30",
        "open_lot_count": 2,
        "previous_snapshot_id": first.snapshot.snapshot_id,
        "previous_state_fingerprint": bad_first_fingerprint,
        "signed_quantity": "2",
        "snapshot_id": second.snapshot.snapshot_id,
        "source_fill_id": second.fill_id,
        "symbol": second.snapshot.symbol,
    }
    connection.execute("DROP TRIGGER fifo_position_snapshots_no_update")
    connection.execute(
        """
        UPDATE fifo_position_snapshots
        SET signed_quantity_text='9', open_cost_text='90', state_fingerprint=?
        WHERE snapshot_id=?
        """,
        (bad_first_fingerprint, first.snapshot.snapshot_id),
    )
    connection.execute(
        """
        UPDATE fifo_position_snapshots
        SET previous_state_fingerprint=?, state_fingerprint=?
        WHERE snapshot_id=?
        """,
        (
            bad_first_fingerprint,
            _snapshot_fingerprint(second_payload),
            second.snapshot.snapshot_id,
        ),
    )
    connection.execute("""
        CREATE TRIGGER fifo_position_snapshots_no_update
        BEFORE UPDATE ON fifo_position_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'fifo_position_snapshots is append-only');
        END
        """)
    connection.commit()

    with pytest.raises(FifoAccountingValidationError, match="snapshot diverges"):
        ledger.verify_epoch_integrity(EPOCH_ID)


def test_integrity_verifier_rejects_fill_without_commission_and_snapshot(connection, ledger):
    _insert_fill_row(connection, _fill(1, FillSide.BUY, "2", "10"))
    connection.commit()

    with pytest.raises(FifoAccountingValidationError, match="commission and one snapshot"):
        ledger.verify_epoch_integrity(EPOCH_ID)


def test_integrity_verifier_binds_opening_direction_to_source_fill(connection, ledger):
    event = _fill(1, FillSide.BUY, "2", "10")
    _insert_fill_row(connection, event)
    connection.execute(
        "INSERT INTO fifo_commissions VALUES (?, ?, ?, 0, 'USD', 2, ?)",
        (event.commission_id, event.epoch_id, event.fill_id, _utc_text(event.recorded_at)),
    )
    lot_id = _identifier("flot", 1)
    connection.execute(
        """
        INSERT INTO fifo_lot_openings VALUES(
            ?, ?, ?, NULL, 0, ?, ?, 'SHORT', '2', '10', 0, 1, ?
        )
        """,
        (
            lot_id,
            event.epoch_id,
            event.fill_id,
            event.con_id,
            event.symbol,
            _utc_text(event.occurred_at),
        ),
    )
    snapshot_id = _identifier("fsnap", 1)
    payload = {
        "con_id": event.con_id,
        "cumulative_commission_minor": 0,
        "cumulative_realized_pnl": "0",
        "epoch_id": event.epoch_id,
        "event_sequence": 1,
        "open_cost": "20",
        "open_lot_count": 1,
        "previous_snapshot_id": None,
        "previous_state_fingerprint": None,
        "signed_quantity": "-2",
        "snapshot_id": snapshot_id,
        "source_fill_id": event.fill_id,
        "symbol": event.symbol,
    }
    connection.execute(
        """
        INSERT INTO fifo_position_snapshots VALUES(
            ?, ?, ?, 1, ?, ?, '-2', '20', 1, '0', 0, NULL, NULL, ?, ?
        )
        """,
        (
            snapshot_id,
            event.epoch_id,
            event.fill_id,
            event.con_id,
            event.symbol,
            _snapshot_fingerprint(payload),
            _utc_text(event.recorded_at),
        ),
    )
    connection.commit()

    with pytest.raises(FifoAccountingValidationError, match="FIFO projection structure"):
        ledger.verify_epoch_integrity(EPOCH_ID)


def test_integrity_verifier_rejects_non_fifo_match_structure(connection, ledger):
    first = ledger.record_fill(_fill(1, FillSide.BUY, "1", "10"))
    second = ledger.record_fill(_fill(2, FillSide.BUY, "1", "20"))
    assert first.opened_lot_id is not None
    assert second.opened_lot_id is not None

    event = _fill(3, FillSide.SELL, "1", "30")
    _insert_fill_row(connection, event)
    connection.execute(
        "INSERT INTO fifo_commissions VALUES (?, ?, ?, 0, 'USD', 2, ?)",
        (event.commission_id, event.epoch_id, event.fill_id, _utc_text(event.recorded_at)),
    )
    connection.execute(
        """
        INSERT INTO fifo_lot_matches VALUES(
            ?, ?, ?, ?, 0, '1', '20', '30', 0, 0, '10', '10', ?
        )
        """,
        (
            _identifier("fmatch", 3),
            event.epoch_id,
            event.fill_id,
            second.opened_lot_id,
            _utc_text(event.occurred_at),
        ),
    )
    snapshot_id = _identifier("fsnap", 3)
    payload = {
        "con_id": event.con_id,
        "cumulative_commission_minor": 0,
        "cumulative_realized_pnl": "10",
        "epoch_id": event.epoch_id,
        "event_sequence": 3,
        "open_cost": "10",
        "open_lot_count": 1,
        "previous_snapshot_id": second.snapshot.snapshot_id,
        "previous_state_fingerprint": second.snapshot.state_fingerprint,
        "signed_quantity": "1",
        "snapshot_id": snapshot_id,
        "source_fill_id": event.fill_id,
        "symbol": event.symbol,
    }
    connection.execute(
        """
        INSERT INTO fifo_position_snapshots VALUES(
            ?, ?, ?, 3, ?, ?, '1', '10', 1, '10', 0, ?, ?, ?, ?
        )
        """,
        (
            snapshot_id,
            event.epoch_id,
            event.fill_id,
            event.con_id,
            event.symbol,
            second.snapshot.snapshot_id,
            second.snapshot.state_fingerprint,
            _snapshot_fingerprint(payload),
            _utc_text(event.recorded_at),
        ),
    )
    connection.commit()

    with pytest.raises(FifoAccountingValidationError, match="FIFO projection structure"):
        ledger.verify_epoch_integrity(EPOCH_ID)


def test_failed_snapshot_insert_rolls_back_fill_commission_matches_and_lots(connection, ledger):
    def reject_snapshot_insert(action, argument_one, _argument_two, _database, _trigger):
        if action == sqlite3.SQLITE_INSERT and argument_one == "fifo_position_snapshots":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(reject_snapshot_insert)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            ledger.record_fill(_fill(1, FillSide.BUY, "1", "10", 1))
    finally:
        connection.set_authorizer(lambda *_arguments: sqlite3.SQLITE_OK)
    for table in (
        "fifo_fills",
        "fifo_commissions",
        "fifo_lot_openings",
        "fifo_lot_matches",
        "fifo_position_snapshots",
    ):
        assert _table_count(connection, table) == 0


def test_validation_rejects_float_non_utc_and_nonpositive_values():
    base = _fill(1, FillSide.BUY, "1", "10")
    with pytest.raises(FifoAccountingValidationError, match="finite Decimal"):
        replace(base, price=10.0)
    with pytest.raises(FifoAccountingValidationError, match="positive"):
        replace(base, quantity=Decimal("0"))
    with pytest.raises(FifoAccountingValidationError, match="UTC"):
        replace(base, occurred_at=datetime(2026, 7, 28, 16, 0))


def test_input_precision_counts_expanded_integer_digits():
    base = _fill(1, FillSide.BUY, "1", "10")
    assert replace(base, quantity=Decimal("1E+37")).quantity == Decimal("1E+37")
    for field in ("quantity", "price"):
        with pytest.raises(FifoAccountingValidationError, match="input precision"):
            replace(base, **{field: Decimal("1E+38")})
        with pytest.raises(FifoAccountingValidationError, match="input precision"):
            replace(base, **{field: Decimal("1E+100")})
        with pytest.raises(FifoAccountingValidationError, match="input precision"):
            replace(base, **{field: Decimal("1E+100000000")})


def test_minor_unit_conversion_is_exact_under_low_ambient_decimal_precision(ledger):
    with localcontext() as context:
        context.prec = 2
        ledger.record_fill(_fill(1, FillSide.BUY, "1", "10", 1_000_000_000_000))
        result = ledger.record_fill(_fill(2, FillSide.SELL, "1", "11", -999_999_999_999))
        assert result.snapshot.cumulative_realized_pnl == Decimal("0.99")
        assert result.snapshot.cumulative_commission_minor == 1
        ledger.verify_epoch_integrity(EPOCH_ID)


def test_schema_drift_is_rejected_before_ledger_use(connection):
    connection.execute("DROP TRIGGER fifo_fills_no_delete")
    with pytest.raises(FifoFixtureMigrationError, match="trigger fifo_fills_no_delete"):
        FifoLedger(connection)
