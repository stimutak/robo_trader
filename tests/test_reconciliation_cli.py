import asyncio
import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from robo_trader.reconciliation import cli as cli_module
from robo_trader.reconciliation.cli import (
    EXIT_BLOCKED,
    EXIT_CLEAN_QUANTITY_COST,
    EXIT_INTEGRITY_VIOLATION,
    build_parser,
    main,
    run_reconciliation,
)
from robo_trader.reconciliation.identity import (
    mask_account_identifier,
    validate_runtime_safety,
)
from robo_trader.reconciliation.models import (
    BrokerExecution,
    BrokerExecutionScope,
    BrokerOpenOrder,
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
        "IBKR_CLIENT_ID": "7",
        "IBKR_RECONCILIATION_CLIENT_ID": "997",
        "IBKR_ACCOUNT": RAW_ACCOUNT,
        "IBKR_APPROVED_ACCOUNTS": RAW_ACCOUNT,
        "IBKR_ACCOUNT_TYPE": "paper",
        "RT_DB_PATH": "ledger.db",
        "RT_STATE_NAMESPACE": "paper",
        "SAFETY_ACCOUNT_SCOPE": "acct_v1_" + ("0123456789abcdef" * 4),
        "SAFETY_JOURNAL_PATH": "safety-journal.db",
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


def test_env_change_during_resolution_blocks_before_provider(
    tmp_path,
    monkeypatch,
    capsys,
):
    _, env = _project(tmp_path)
    constructed = []
    real_resolve = cli_module.resolve_environment

    def resolve_then_replace(project_root, process_environ):
        resolved = real_resolve(project_root, process_environ)
        (project_root / ".env").write_text("RT_DB_PATH=other.db\n")
        return resolved

    monkeypatch.setattr(cli_module, "resolve_environment", resolve_then_replace)
    result = main(
        ["--portfolio-id", "default", "--json"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=lambda runtime: constructed.append(runtime),
        now=NOW,
    )

    assert result == EXIT_INTEGRITY_VIOLATION
    assert constructed == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "INTEGRITY_VIOLATION"
    assert payload["mutated_state"] is False
    assert payload["authorizes_startup"] is False


def test_symlinked_env_is_rejected_before_parsing_or_provider_construction(
    tmp_path,
    capsys,
):
    _, env = _project(tmp_path)
    raw_target = tmp_path / f"environment-{RAW_ACCOUNT}"
    raw_target.write_text("RT_DB_PATH=other.db\n")
    (tmp_path / ".env").symlink_to(raw_target)
    constructed = []

    result = main(
        ["--portfolio-id", "default", "--json"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=lambda runtime: constructed.append(runtime),
        now=NOW,
    )

    assert result == EXIT_INTEGRITY_VIOLATION
    assert constructed == []
    output = capsys.readouterr().out
    assert RAW_ACCOUNT not in output
    payload = json.loads(output)
    assert payload["error_code"] == "INTEGRITY_VIOLATION"
    assert payload["message"] == "runtime environment file must not be a symlink"
    assert payload["mutated_state"] is False
    assert payload["authorizes_startup"] is False


def test_connection_gate_blocks_unsafe_port_and_readonly_before_provider(tmp_path, capsys):
    for key, value in (
        ("IBKR_PORT", "4001"),
        ("IBKR_READONLY", "false"),
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


@pytest.mark.parametrize(
    ("evidence_kind", "expected_status"),
    [("open_order", "BLOCKED"), ("recent_execution", "INCOMPLETE")],
)
def test_unmatched_broker_activity_never_returns_success(
    tmp_path,
    capsys,
    evidence_kind,
    expected_status,
):
    _, env = _project(tmp_path)
    snapshot = _broker_snapshot()
    contract = snapshot.positions[0].contract
    if evidence_kind == "open_order":
        snapshot = replace(
            snapshot,
            open_orders=(
                BrokerOpenOrder(
                    order_id="101",
                    client_id=7,
                    contract=contract,
                    side="BUY",
                    quantity=Decimal("1"),
                    filled=Decimal("0"),
                    remaining=Decimal("1"),
                    order_type="LMT",
                    status="Submitted",
                    limit_price=Decimal("99"),
                    time_in_force="DAY",
                    last_status_at=NOW,
                ),
            ),
        )
    else:
        snapshot = replace(
            snapshot,
            recent_executions=(
                BrokerExecution(
                    execution_id="exec-101",
                    order_id="101",
                    contract=contract,
                    side="BUY",
                    quantity=Decimal("1"),
                    price=Decimal("99"),
                    executed_at=NOW - timedelta(minutes=1),
                    client_id=7,
                    execution_exchange="NASDAQ",
                ),
            ),
        )

    result = main(
        ["--portfolio-id", "default", "--json"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=lambda runtime: FakeProvider(snapshot),
        now=NOW,
    )

    assert result == EXIT_BLOCKED
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == expected_status
    assert payload["authorizes_startup"] is False


@pytest.mark.asyncio
async def test_snapshot_cancellation_closes_provider_before_propagating(tmp_path):
    _, env = _project(tmp_path)
    snapshot_started = asyncio.Event()
    closed = asyncio.Event()

    class CancellableProvider(FakeProvider):
        async def get_broker_snapshot(self, expected_account, *, max_age_seconds):
            del expected_account, max_age_seconds
            snapshot_started.set()
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True
            closed.set()

    provider = CancellableProvider(_broker_snapshot())
    task = asyncio.create_task(
        run_reconciliation(
            ["default"],
            project_root=tmp_path,
            process_environ=env,
            provider_factory=lambda runtime: provider,
            now=NOW,
        )
    )
    await snapshot_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed.is_set()
    assert provider.closed is True


@pytest.mark.asyncio
async def test_provider_close_is_shielded_from_cancellation(tmp_path):
    _, env = _project(tmp_path)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()

    class SlowCloseProvider(FakeProvider):
        async def close(self):
            close_started.set()
            await release_close.wait()
            self.closed = True
            close_finished.set()

    provider = SlowCloseProvider(_broker_snapshot())
    task = asyncio.create_task(
        run_reconciliation(
            ["default"],
            project_root=tmp_path,
            process_environ=env,
            provider_factory=lambda runtime: provider,
            now=NOW,
        )
    )
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert close_finished.is_set()
    assert provider.closed is True


@pytest.mark.asyncio
async def test_provider_close_has_no_outer_timeout(tmp_path, monkeypatch):
    """The production transport owns its bounded, multi-stage shutdown."""
    _, env = _project(tmp_path)
    observed_timeouts = []
    real_wait_for = asyncio.wait_for

    async def recording_wait_for(awaitable, *, timeout):
        observed_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(cli_module.asyncio, "wait_for", recording_wait_for)
    provider = FakeProvider(_broker_snapshot())

    await run_reconciliation(
        ["default"],
        project_root=tmp_path,
        process_environ=env,
        provider_factory=lambda runtime: provider,
        now=NOW,
    )

    assert observed_timeouts == [60.0]
    assert provider.closed is True


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


@pytest.mark.parametrize(
    "capability",
    [
        "place_order",
        "cancel_order",
        "modify_order",
        "replace_order",
        "exercise_options",
        "globalCancel",
        "req_global_cancel",
        "reqGlobalCancel",
    ],
)
def test_provider_with_order_capability_is_rejected_without_calling_it(
    tmp_path,
    capsys,
    capability,
):
    _, env = _project(tmp_path)
    env.update(
        {
            "IBKR_CLIENT_ID": "1",
            "IBKR_RECONCILIATION_CLIENT_ID": "71",
        }
    )

    provider = FakeProvider(_broker_snapshot())

    def forbidden_capability():
        raise AssertionError("must never be called")

    setattr(provider, capability, forbidden_capability)
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
