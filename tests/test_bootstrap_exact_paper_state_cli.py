from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.bootstrap_exact_paper_state as cli
from robo_trader.financial_state_bootstrap import ExactStateBootstrapError


def _candidate(database_path: Path, *, portfolio_id: str = "default") -> SimpleNamespace:
    return SimpleNamespace(
        bootstrap_id="pbs-test",
        database_path=str(database_path),
        database_identity="paper:test",
        legacy_snapshot_hash=(
            str(cli.inspect_legacy_state(database_path)["snapshot_hash"])
            if database_path.exists() and database_path.stat().st_size
            else "a" * 64
        ),
        reconciliation_snapshot_id="recon-test",
        reconciliation_report_hash="b" * 64,
        broker_snapshot_hash="c" * 64,
        execution_domain_scope="paper-simulator-v1",
        account_scope="acct_v1_" + "d" * 64,
        portfolio_id=portfolio_id,
        positions=(),
        fingerprint=lambda: "e" * 64,
    )


def _legacy_database(path: Path, *, wal: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    if wal:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.executescript("""
        CREATE TABLE portfolios (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE account (
            portfolio_id TEXT PRIMARY KEY, cash REAL, equity REAL,
            daily_pnl REAL, realized_pnl REAL, unrealized_pnl REAL,
            timestamp DATETIME
        );
        CREATE TABLE positions (
            portfolio_id TEXT, symbol TEXT, quantity INTEGER, avg_cost REAL,
            market_price REAL, timestamp DATETIME,
            PRIMARY KEY(portfolio_id, symbol)
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, portfolio_id TEXT, symbol TEXT, side TEXT,
            quantity INTEGER, price REAL, notional REAL, slippage REAL,
            commission REAL, pnl REAL, timestamp DATETIME
        );
        CREATE TABLE equity_history (
            id INTEGER PRIMARY KEY, portfolio_id TEXT, date TEXT, equity REAL,
            cash REAL, positions_value REAL, realized_pnl REAL,
            unrealized_pnl REAL, timestamp DATETIME
        );
        INSERT INTO account VALUES ('default',100000,100000,0,0,0,'2026-07-28');
        INSERT INTO portfolios VALUES ('default','Default');
        INSERT INTO trades VALUES
            (1,'default','AAPL','BUY',1,100,100,0,0,NULL,'2026-07-28');
        INSERT INTO equity_history VALUES
            (1,'default','2026-07-28',100000,99900,100,0,0,'2026-07-28');
    """)
    connection.commit()
    return connection


def test_online_backup_includes_active_wal_and_is_restorable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    writer = _legacy_database(source, wal=True)
    writer.execute("INSERT INTO trades(symbol) VALUES ('NVDA')")
    writer.commit()
    assert (tmp_path / "source.db-wal").exists()
    target = tmp_path / "backup.db"
    backup = cli._online_backup(source, target, _candidate(source))
    try:
        backup.assert_identity()
        assert target.stat().st_mode & 0o222 == 0
        with pytest.raises(OSError):
            os.write(backup.target_binding.descriptor, b"x")
        assert dict(backup.row_counts)["trades"] == 2
        with sqlite3.connect(target) as restored:
            assert restored.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert restored.execute("SELECT symbol FROM trades ORDER BY id").fetchall() == [
                ("AAPL",),
                ("NVDA",),
            ]
    finally:
        backup.close()
        writer.close()


def test_online_backup_rejects_source_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.db"
    _legacy_database(source).close()
    candidate = _candidate(source)
    original_connect = cli.sqlite3.connect
    replaced = False

    def substitute_then_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal replaced
        if not replaced and str(args[0]).startswith(source.as_uri()):
            replaced = True
            source.rename(tmp_path / "original.db")
            _legacy_database(source).close()
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(cli.sqlite3, "connect", substitute_then_connect)
    with pytest.raises(Exception, match="identit"):
        cli._online_backup(source, tmp_path / "backup.db", candidate)


def test_online_backup_rejects_target_substitution_and_leaves_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.db"
    _legacy_database(source).close()
    candidate = _candidate(source)
    target = tmp_path / "backup.db"
    moved = tmp_path / "reserved-original.db"
    original_connect = cli.sqlite3.connect
    replaced = False

    def substitute_then_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal replaced
        if not replaced and str(args[0]).startswith(target.as_uri()):
            replaced = True
            target.rename(moved)
            target.write_bytes(b"replacement must remain")
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(cli.sqlite3, "connect", substitute_then_connect)
    with pytest.raises(Exception, match="identit|file is not a database"):
        cli._online_backup(source, target, candidate)
    assert target.exists()
    assert target.read_bytes() == b"replacement must remain"
    assert moved.exists()


def test_backup_verification_failure_leaves_exclusive_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.db"
    _legacy_database(source).close()
    target = tmp_path / "failed-backup.db"
    monkeypatch.setattr(
        cli,
        "sqlite_table_evidence",
        lambda _connection: (_ for _ in ()).throw(RuntimeError("verification failed")),
    )
    with pytest.raises(RuntimeError, match="verification failed"):
        cli._online_backup(source, target, _candidate(source))
    assert target.exists()
    with pytest.raises(FileExistsError):
        os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)


def test_candidate_binding_detects_substitution(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}", encoding="utf-8")
    candidate_path.chmod(0o600)
    binding = cli._RegularFileBinding.open_readonly(
        candidate_path,
        label="candidate",
        owner_only=True,
    )
    try:
        candidate_path.rename(tmp_path / "original-candidate.json")
        candidate_path.write_text("{}", encoding="utf-8")
        candidate_path.chmod(0o600)
        with pytest.raises(ExactStateBootstrapError, match="changed"):
            binding.assert_identity()
    finally:
        binding.close()


def test_candidate_binding_detects_in_place_content_change_with_restored_mtime(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_bytes(b"original")
    candidate_path.chmod(0o600)
    binding = cli._RegularFileBinding.open_readonly(
        candidate_path,
        label="candidate",
        owner_only=True,
    )
    candidate_binding = cli._CandidateBinding(
        file=binding,
        candidate=object(),
        content_hash=cli.hashlib.sha256(b"original").hexdigest(),
    )
    try:
        metadata = candidate_path.stat()
        candidate_path.write_bytes(b"tampered")
        os.utime(
            candidate_path,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
        )
        with pytest.raises(ExactStateBootstrapError, match="content changed"):
            candidate_binding.assert_identity()
    finally:
        candidate_binding.close()


@pytest.mark.parametrize("active_kind", ["module-runner", "paper-listener", "live-listener"])
def test_assert_stopped_rejects_module_runner_and_broker_listeners(
    active_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if active_kind == "module-runner" and "robo_trader\\.runner_async" in command[-1]:
            return SimpleNamespace(returncode=0, stdout="123 python -m robo_trader.runner_async\n")
        if active_kind == "paper-listener" and "-iTCP:4001" in command:
            return SimpleNamespace(returncode=0, stdout="gateway listen\n")
        if active_kind == "live-listener" and "-iTCP:4002" in command:
            return SimpleNamespace(returncode=0, stdout="gateway listen\n")
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stopped|listener"):
        cli._assert_stopped()
    assert calls >= 2


def test_preview_scopes_position_coverage_to_candidate_portfolio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "ledger.db"
    candidate = _candidate(database_path, portfolio_id="alpha")
    candidate.positions = (SimpleNamespace(symbol="AAPL", quantity=2),)
    monkeypatch.setattr(
        cli,
        "inspect_legacy_state",
        lambda _path: {
            "snapshot_hash": candidate.legacy_snapshot_hash,
            "position_rows": [
                {"portfolio_id": "alpha", "symbol": "AAPL", "quantity": 2},
                {"portfolio_id": "beta", "symbol": "NVDA", "quantity": 7},
            ],
        },
    )
    report = cli.preview(candidate, database_path)
    assert report["position_count"] == 1


def test_confirmation_binds_candidate_database_and_backup(tmp_path: Path) -> None:
    database_path = tmp_path / "ledger.db"
    candidate = _candidate(database_path)
    runtime = SimpleNamespace(database_identity="paper:identity")
    confirmation = cli._required_confirmation(candidate, runtime, tmp_path / "backup.db")
    assert confirmation == (
        "APPLY_SEALED_EXACT_STATE_BOOTSTRAP "
        f"candidate={candidate.fingerprint()} database=paper:identity "
        f"backup={tmp_path / 'backup.db'}"
    )


@pytest.mark.asyncio
async def test_apply_passes_every_authority_to_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    result_receipt = SimpleNamespace(
        bootstrap_id="pbs-test",
        candidate_fingerprint="e" * 64,
        committed_at=SimpleNamespace(isoformat=lambda: "2026-07-28T00:00:00+00:00"),
        database_device=1,
        database_inode=2,
        operator_action_id="padmin-test",
    )

    class FakeDatabase:
        def __init__(self, path: Path) -> None:
            captured["path"] = path

        async def initialize(self) -> None:
            captured["initialized"] = True

        async def apply_exact_state_bootstrap(self, candidate: object, **kwargs: object) -> object:
            captured["candidate"] = candidate
            captured.update(kwargs)
            return result_receipt

        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "AsyncTradingDatabase", FakeDatabase)
    candidate = _candidate(tmp_path / "ledger.db")
    evidence = object()
    backup_receipt = object()
    runtime_contract = object()
    await cli._apply(
        candidate,
        tmp_path / "ledger.db",
        "reviewed offline bootstrap",
        evidence=evidence,
        backup_receipt=backup_receipt,
        runtime_contract=runtime_contract,
    )
    assert captured["evidence"] is evidence
    assert captured["backup_receipt"] is backup_receipt
    assert captured["runtime_contract"] is runtime_contract
    assert captured["operator_reason"] == "reviewed offline bootstrap"
    assert captured["closed"] is True


def test_wrong_typed_confirmation_blocks_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "ledger.db"
    database_path.touch()
    identity = "paper:" + cli.hashlib.sha256(str(database_path.resolve()).encode()).hexdigest()[:12]
    runtime = SimpleNamespace(database_path=str(database_path), database_identity=identity)
    candidate = _candidate(database_path)
    candidate.database_identity = identity
    binding = SimpleNamespace(
        candidate=candidate,
        assert_identity=lambda: None,
        close=lambda: None,
    )
    args = argparse.Namespace(
        command="apply",
        db_path=database_path,
        candidate=tmp_path / "candidate.json",
        reconciliation_evidence=tmp_path / "reconciliation.json",
        broker_snapshot=tmp_path / "broker.json",
        protective_marks=[tmp_path / "mark.json"],
        backup_path=tmp_path / "backup.db",
        reason="reviewed offline bootstrap",
        confirm="wrong confirmation",
        json=True,
    )
    parser = SimpleNamespace(parse_args=lambda _argv: args)
    monkeypatch.setattr(cli, "_parser", lambda: parser)
    monkeypatch.setattr(cli, "load_runtime_contract_from_env", lambda **_kwargs: runtime)
    monkeypatch.setattr(cli, "_open_candidate", lambda _path: binding)
    monkeypatch.setattr(cli, "load_exact_state_bootstrap_evidence", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "preview", lambda *_args: {"status": "ready"})
    called = False

    def forbidden_backup(*_args: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("backup must not run")

    monkeypatch.setattr(cli, "_online_backup", forbidden_backup)
    assert cli.main([]) == 2
    assert called is False
