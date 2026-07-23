import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from robo_trader.reconciliation.cli import (
    EXIT_BLOCKED,
    EXIT_CLEAN_QUANTITY_COST,
    build_parser,
    main,
)
from robo_trader.reconciliation.identity import (
    mask_account_identifier,
    validate_runtime_safety,
)
from robo_trader.reconciliation.models import (
    BrokerExecutionScope,
    BrokerPosition,
    BrokerSnapshot,
    ContractIdentity,
)

NOW = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
RAW_ACCOUNT = "DU1234567"


def _project(tmp_path: Path, *, valid_ibc: bool = True):
    (tmp_path / "config" / "ibc").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    ibc = "ReadOnlyApi=yes\nTradingMode=paper\n"
    if not valid_ibc:
        ibc = "ReadOnlyApi=no\nTradingMode=paper\n"
    (tmp_path / "config" / "ibc" / "config.ini").write_text(ibc)
    database = tmp_path / "ledger.db"
    connection = sqlite3.connect(database)
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
        INSERT INTO account VALUES ('default', 1000, 1000);
        INSERT INTO positions
        VALUES (1, 'default', 'AAPL', 3, 100, 101, '2026-07-23T14:59:00');
        """)
    connection.commit()
    connection.close()
    env = {
        "EXECUTION_MODE": "paper",
        "ENVIRONMENT": "dev",
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "4002",
        "IBKR_READONLY": "true",
        "IBKR_ACCOUNT": RAW_ACCOUNT,
        "IBKR_APPROVED_ACCOUNTS": RAW_ACCOUNT,
        "IBKR_ACCOUNT_TYPE": "paper",
        "RT_DB_PATH": "ledger.db",
        "RT_STATE_NAMESPACE": "paper",
        "BUILD_ID": "test-build",
        "MODEL_ARTIFACT_SET": "test-models",
        "LOG_FILE": "robo_trader.log",
    }
    return database, env


def _broker_snapshot() -> BrokerSnapshot:
    contract = ContractIdentity(
        con_id=1,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="AAPL",
    )
    return BrokerSnapshot(
        schema_version=1,
        account_alias=mask_account_identifier(RAW_ACCOUNT),
        broker_time_before=NOW - timedelta(seconds=1),
        broker_time_after=NOW,
        retrieved_at=NOW,
        execution_scope=BrokerExecutionScope(
            kind="bounded_execution_filter",
            start_at=NOW - timedelta(hours=24, seconds=1),
            end_at=NOW,
        ),
        positions=(BrokerPosition(contract, Decimal("3"), Decimal("100")),),
        balances={
            "NetLiquidation:USD": Decimal("1000"),
            "TotalCashValue:USD": Decimal("500"),
        },
    )


class FakeProvider:
    def __init__(self, snapshot=None, error=None):
        self.snapshot = snapshot
        self.error = error
        self.snapshot_calls = 0
        self.expected_accounts = []
        self.closed = False

    async def get_broker_snapshot(self, expected_account, *, max_age_seconds):
        self.snapshot_calls += 1
        self.expected_accounts.append(expected_account)
        assert max_age_seconds == 30.0
        if self.error:
            raise self.error
        return self.snapshot

    async def close(self):
        self.closed = True


def test_runtime_context_repr_and_validation_never_expose_raw_account(tmp_path):
    _, env = _project(tmp_path)
    context = validate_runtime_safety(tmp_path, env)
    assert RAW_ACCOUNT not in repr(context)
    assert context.account_alias == "***4567"
    context.verify_managed_accounts([RAW_ACCOUNT])

    try:
        context.verify_managed_accounts(["DU9999999"])
    except Exception as exc:
        assert RAW_ACCOUNT not in str(exc)


def test_invalid_ibc_blocks_before_provider_construction(tmp_path, capsys):
    _, env = _project(tmp_path, valid_ibc=False)
    constructed = []

    def factory(runtime):
        constructed.append(runtime)
        return FakeProvider(_broker_snapshot())

    result = main(
        ["--portfolio-id", "default", "--json"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=factory,
        now=NOW,
    )

    assert result == EXIT_BLOCKED
    assert constructed == []
    output = capsys.readouterr().out
    assert RAW_ACCOUNT not in output
    payload = json.loads(output)
    assert payload["mutated_state"] is False
    assert payload["authorizes_startup"] is False


def test_connection_gate_blocks_unsafe_port_readonly_and_zero_client_before_provider(
    tmp_path, capsys
):
    for key, value in (
        ("IBKR_PORT", "4001"),
        ("IBKR_READONLY", "false"),
        ("IBKR_CLIENT_ID", "0"),
    ):
        project = tmp_path / key.lower()
        _, env = _project(project)
        env[key] = value
        constructed = []

        result = main(
            ["--portfolio-id", "default", "--json"],
            project_root=project,
            process_environ=env,
            provider_factory=lambda runtime: constructed.append(runtime),
            now=NOW,
        )

        assert result == EXIT_BLOCKED
        assert constructed == []
        assert RAW_ACCOUNT not in capsys.readouterr().out


def test_connection_gate_rejects_non_loopback_host_before_provider(tmp_path, capsys):
    for index, host in enumerate(("192.0.2.10", "gateway.internal", "0.0.0.0")):
        project = tmp_path / f"remote-{index}"
        _, env = _project(project)
        env["IBKR_HOST"] = host
        constructed = []

        result = main(
            ["--portfolio-id", "default", "--json"],
            project_root=project,
            process_environ=env,
            provider_factory=lambda runtime: constructed.append(runtime),
            now=NOW,
        )

        assert result == EXIT_BLOCKED
        assert constructed == []
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "RUNTIME_SAFETY_BLOCK"
        assert "loopback" in payload["message"]


def test_connection_gate_accepts_named_ipv4_and_ipv6_loopback(tmp_path):
    for index, host in enumerate(("localhost", "127.0.0.1", "127.9.8.7", "::1")):
        project = tmp_path / f"loopback-{index}"
        _, env = _project(project)
        env["IBKR_HOST"] = host

        context = validate_runtime_safety(project, env)

        assert context.diagnostic_connection.host == host


def test_cli_does_not_expose_broker_or_mutation_overrides():
    option_strings = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert option_strings == {"-h", "--help", "--portfolio-id", "--json"}


def test_duplicate_ibc_safety_assignments_fail_closed_without_account_leak(tmp_path):
    _, env = _project(tmp_path)
    config = tmp_path / "config" / "ibc" / "config.ini"
    config.write_text("ReadOnlyApi=yes\nReadOnlyApi=yes\nTradingMode=paper\n")

    try:
        validate_runtime_safety(tmp_path, env)
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("ambiguous IBC configuration was accepted")

    assert RAW_ACCOUNT not in message
    assert "read-only paper session" in message


def test_success_is_non_mutating_masks_account_and_closes_provider(tmp_path, capsys):
    database, env = _project(tmp_path)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    provider = FakeProvider(_broker_snapshot())

    result = main(
        ["--portfolio-id", "default", "--json"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=lambda runtime: provider,
        now=NOW,
    )

    assert result == EXIT_CLEAN_QUANTITY_COST
    assert provider.snapshot_calls == 1
    assert provider.expected_accounts == [RAW_ACCOUNT]
    assert provider.closed is True
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert not (tmp_path / "robo_trader.log").exists()
    output = capsys.readouterr().out
    assert RAW_ACCOUNT not in output
    payload = json.loads(output)
    assert payload["account_alias"] == "***4567"
    assert payload["mutated_state"] is False
    assert payload["authorizes_startup"] is False
    assert payload["status"] == "QUANTITY_COST_COMPARABLE_ONLY"


def test_provider_error_is_redacted_and_cleanup_runs(tmp_path, capsys):
    _, env = _project(tmp_path)
    provider = FakeProvider(error=RuntimeError(f"credential {RAW_ACCOUNT} secret"))

    result = main(
        ["--portfolio-id", "default", "--json"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=lambda runtime: provider,
        now=NOW,
    )

    assert result == EXIT_BLOCKED
    assert provider.closed is True
    output = capsys.readouterr().out
    assert RAW_ACCOUNT not in output
    assert "credential" not in output
    assert json.loads(output)["error_code"] == "BROKER_EVIDENCE_BLOCK"


def test_provider_with_order_capability_is_rejected_without_calling_it(tmp_path, capsys):
    _, env = _project(tmp_path)

    class UnsafeProvider(FakeProvider):
        def place_order(self):
            raise AssertionError("must never be called")

    provider = UnsafeProvider(_broker_snapshot())
    result = main(
        ["--portfolio-id", "default", "--json"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=lambda runtime: provider,
        now=NOW,
    )

    assert result == EXIT_BLOCKED
    assert provider.snapshot_calls == 0
    assert provider.closed is True
    assert json.loads(capsys.readouterr().out)["error_code"] == "BROKER_EVIDENCE_BLOCK"


def test_provider_supplied_reconciliation_error_is_also_redacted(tmp_path, capsys):
    from robo_trader.reconciliation.errors import BrokerEvidenceError

    _, env = _project(tmp_path)
    provider = FakeProvider(error=BrokerEvidenceError(f"account={RAW_ACCOUNT}"))

    result = main(
        ["--portfolio-id", "default", "--json"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=lambda runtime: provider,
        now=NOW,
    )

    assert result == EXIT_BLOCKED
    output = capsys.readouterr().out
    assert RAW_ACCOUNT not in output
    assert "account=" not in output
