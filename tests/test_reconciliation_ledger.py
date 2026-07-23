import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

from robo_trader.reconciliation.errors import IntegrityViolation, LedgerSafetyError
from robo_trader.reconciliation.integrity import (
    EvidenceIntegrityGuard,
    protected_evidence_paths,
)
from robo_trader.reconciliation.ledger import (
    ImmutableLedgerReader,
    validate_portfolio_ids,
)


def _create_ledger(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            market_price REAL,
            timestamp DATETIME
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            timestamp DATETIME
        );
        CREATE TABLE account (
            portfolio_id TEXT PRIMARY KEY,
            cash REAL NOT NULL,
            equity REAL NOT NULL
        );
        """)
    connection.execute("INSERT INTO account VALUES ('default', 1000, 1000)")
    connection.commit()
    connection.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_immutable_reader_does_not_modify_ledger(tmp_path):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO positions VALUES (1, 'default', 'AAPL', 3, 100.25, 101, '2026-01-01')"
    )
    connection.execute(
        "INSERT INTO trades VALUES (1, 'default', 'AAPL', 'BUY', 3, 100.25, '2026-01-01')"
    )
    connection.commit()
    connection.close()
    before = _sha256(database)

    snapshot = ImmutableLedgerReader(tmp_path, "ledger.db").read(["default"])

    assert _sha256(database) == before
    assert not Path(f"{database}-wal").exists()
    assert snapshot.active_portfolio_ids == ("default",)
    assert snapshot.aggregated_positions[0].quantity == 3
    assert snapshot.aggregated_positions[0].average_cost == pytest.approx(100.25)
    assert snapshot.recent_trades[0].local_trade_id == 1


def test_reader_connection_authorizer_denies_writes_and_attach(tmp_path):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    reader = ImmutableLedgerReader(tmp_path, "ledger.db")
    connection = reader._connect()
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("DELETE FROM positions")
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("ATTACH DATABASE ':memory:' AS extra")
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("-wal", "WAL"),
        ("-shm", "shared-memory"),
        ("-journal", "rollback journal"),
    ],
)
def test_reader_rejects_symlink_and_nonempty_sqlite_sidecars(tmp_path, suffix, expected):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    Path(f"{database}{suffix}").write_bytes(b"uncommitted")
    with pytest.raises(LedgerSafetyError, match=expected):
        ImmutableLedgerReader(tmp_path, "ledger.db")


def test_reader_rejects_symlink_database(tmp_path):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    symlink = tmp_path / "linked.db"
    symlink.symlink_to(database)
    with pytest.raises(LedgerSafetyError, match="symlink"):
        ImmutableLedgerReader(tmp_path, "linked.db")


def test_reader_rejects_duplicate_portfolio_symbol_before_aggregation(tmp_path):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO positions VALUES (?, 'default', 'AAPL', ?, 100, 100, ?)",
        [
            (1, 2, "2026-07-23T14:59:00"),
            (2, 0, "2026-07-23T14:59:01"),
        ],
    )
    connection.commit()
    connection.close()

    with pytest.raises(LedgerSafetyError, match="duplicate portfolio and symbol"):
        ImmutableLedgerReader(tmp_path, "ledger.db").read(["default"])


@pytest.mark.parametrize("table", ["positions", "trades"])
def test_reader_rejects_malformed_ledger_timestamps(tmp_path, table):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    connection = sqlite3.connect(database)
    if table == "positions":
        connection.execute(
            "INSERT INTO positions VALUES " "(1, 'default', 'AAPL', 3, 100, 100, 'not-a-timestamp')"
        )
    else:
        connection.execute(
            "INSERT INTO trades VALUES " "(1, 'default', 'AAPL', 'BUY', 3, 100, 'not-a-timestamp')"
        )
    connection.commit()
    connection.close()

    with pytest.raises(LedgerSafetyError, match=f"{table[:-1]} timestamp is malformed"):
        ImmutableLedgerReader(tmp_path, "ledger.db").read(["default"])


def test_reader_rejects_fractional_value_in_integer_quantity_schema(tmp_path):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO positions VALUES (1, 'default', 'AAPL', 1.5, 100, 100, '2026-01-01')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(LedgerSafetyError, match="fractional"):
        ImmutableLedgerReader(tmp_path, "ledger.db").read(["default"])


def test_multi_portfolio_aggregate_is_blocked_and_offsetting_net_zero_is_flagged(tmp_path):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO account VALUES ('other', 1000, 1000)")
    connection.executemany(
        "INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "default", "AAPL", 10, 100, 100, "2026-01-01"),
            (2, "other", "AAPL", -10, 110, 110, "2026-01-01"),
        ],
    )
    connection.commit()
    connection.close()

    snapshot = ImmutableLedgerReader(tmp_path, "ledger.db").read(["default", "other"])

    assert snapshot.aggregated_positions[0].quantity == 0
    assert snapshot.aggregated_positions[0].average_cost is None
    assert "AMBIGUOUS_MULTI_PORTFOLIO_BROKER_ALLOCATION" in snapshot.blockers
    assert "OFFSETTING_PORTFOLIO_POSITIONS:AAPL" in snapshot.blockers
    assert "NET_ZERO_MASKS_PORTFOLIO_EXPOSURE:AAPL" in snapshot.blockers


def test_unselected_active_portfolio_fails_closed_in_report_evidence(tmp_path):
    database = tmp_path / "ledger.db"
    _create_ledger(database)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO account VALUES ('other', 1000, 1000)")
    connection.execute("INSERT INTO positions VALUES (1, 'other', 'MSFT', 2, 50, 50, '2026-01-01')")
    connection.commit()
    connection.close()

    snapshot = ImmutableLedgerReader(tmp_path, "ledger.db").read(["default"])

    assert "UNSELECTED_ACTIVE_PORTFOLIOS" in snapshot.blockers
    assert snapshot.aggregated_positions == ()


def test_reader_rejects_missing_schema_and_unknown_portfolio(tmp_path):
    database = tmp_path / "ledger.db"
    sqlite3.connect(database).close()
    with pytest.raises(LedgerSafetyError, match="schema"):
        ImmutableLedgerReader(tmp_path, "ledger.db").read(["default"])

    database.unlink()
    _create_ledger(database)
    with pytest.raises(LedgerSafetyError, match="do not exist"):
        ImmutableLedgerReader(tmp_path, "ledger.db").read(["unknown"])


def test_portfolio_identity_rejects_account_shaped_selected_and_stored_values(tmp_path):
    with pytest.raises(LedgerSafetyError, match="portfolio IDs are invalid"):
        validate_portfolio_ids(["du1234567"])

    database = tmp_path / "ledger.db"
    _create_ledger(database)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO account VALUES ('desk-du1234567', 1000, 1000)")
    connection.commit()
    connection.close()

    with pytest.raises(LedgerSafetyError, match="ambiguous portfolio identity"):
        ImmutableLedgerReader(tmp_path, "ledger.db").read(["default"])


def test_evidence_guard_detects_content_appearance_and_disappearance(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.write_bytes(b"before")
    with pytest.raises(IntegrityViolation):
        with EvidenceIntegrityGuard([evidence]):
            evidence.write_bytes(b"after")

    absent = tmp_path / "absent"
    with pytest.raises(IntegrityViolation):
        with EvidenceIntegrityGuard([absent]):
            absent.write_bytes(b"appeared")

    with pytest.raises(IntegrityViolation):
        with EvidenceIntegrityGuard([evidence]):
            os.unlink(evidence)


def test_protected_paths_preserve_symlink_identity_and_include_rollback_journal(tmp_path):
    target = tmp_path / "first.db"
    target.write_bytes(b"same")
    alternate = tmp_path / "second.db"
    alternate.write_bytes(b"same")
    configured = tmp_path / "configured.db"
    configured.symlink_to(target)

    paths = protected_evidence_paths(tmp_path, {"RT_DB_PATH": "configured.db"})

    assert configured in paths
    assert target in paths
    assert Path(f"{configured}-journal") in paths
    with pytest.raises(IntegrityViolation):
        with EvidenceIntegrityGuard(paths):
            configured.unlink()
            configured.symlink_to(alternate)
