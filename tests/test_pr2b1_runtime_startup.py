import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.config import RuntimeContract
from robo_trader.runner_async import (
    _setup_continuous_runner,
    _start_paper_order_runtime,
    _start_paper_safety_runtime,
    main,
    run_continuous,
    run_once,
)
from robo_trader.safety import SafetyJournal
from robo_trader.safety.journal import JournalNotInitialized

ACCOUNT_SCOPE = "acct_v1_0123456789abcdef0123456789abcdef" "fedcba9876543210fedcba9876543210"


def _runtime_contract(tmp_path: Path) -> RuntimeContract:
    return RuntimeContract(
        environment="dev",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(tmp_path / "paper-ledger.db"),
        account_alias="***1234",
        account_type="paper",
        model_artifact_set="test-models",
        build_id="test-build",
        state_namespace="paper",
        safety_account_scope=ACCOUNT_SCOPE,
        safety_execution_domain_scope="paper-simulator-v1",
        safety_journal_path=str(tmp_path / "safety-journal.db"),
    )


def _cfg(tmp_path: Path):
    return SimpleNamespace(runtime_contract=_runtime_contract(tmp_path))


def test_startup_replays_existing_journal_without_initializing_it(tmp_path):
    cfg = _cfg(tmp_path)
    journal_path = Path(cfg.runtime_contract.safety_journal_path)
    SafetyJournal(journal_path).initialize(
        execution_domain_scope=cfg.runtime_contract.safety_execution_domain_scope,
        account_scope=cfg.runtime_contract.safety_account_scope,
    )

    coordinator = _start_paper_safety_runtime(cfg)

    assert coordinator.started
    assert journal_path.is_file()


def test_startup_missing_journal_fails_closed_without_creating_file(tmp_path):
    cfg = _cfg(tmp_path)
    journal_path = Path(cfg.runtime_contract.safety_journal_path)

    with pytest.raises(JournalNotInitialized):
        _start_paper_safety_runtime(cfg)

    assert not journal_path.exists()


def test_startup_requires_exact_validated_runtime_contract(tmp_path):
    cfg = SimpleNamespace(
        runtime_contract=SimpleNamespace(
            safety_execution_domain_scope="paper-simulator-v1",
            safety_account_scope=ACCOUNT_SCOPE,
            safety_journal_path=str(tmp_path / "journal.db"),
        )
    )

    with pytest.raises(RuntimeError, match="validated runtime contract"):
        _start_paper_safety_runtime(cfg)


@pytest.mark.asyncio
async def test_paper_order_runtime_remains_blocked_until_pr2b3():
    with pytest.raises(RuntimeError, match="PR 2B.3 terminal settlement"):
        await _start_paper_order_runtime(object(), object())


@pytest.mark.asyncio
async def test_partial_continuous_setup_always_cleans_generic_failure():
    runner = MagicMock()
    runner.setup = AsyncMock(side_effect=RuntimeError("setup failed"))
    runner._attach_health_monitor = AsyncMock()
    runner.cleanup = AsyncMock()

    with pytest.raises(RuntimeError, match="setup failed"):
        await _setup_continuous_runner(runner)

    runner.cleanup.assert_awaited_once()
    runner._attach_health_monitor.assert_not_awaited()


def test_main_replays_safety_journal_before_gateway_or_runner_work(tmp_path):
    cfg = SimpleNamespace(
        execution=SimpleNamespace(mode=SimpleNamespace(value="paper")),
        ibkr=SimpleNamespace(readonly=True, port=4002),
        runtime_contract=_runtime_contract(tmp_path),
    )
    startup_error = JournalNotInitialized("missing journal")

    with (
        patch.object(sys, "argv", ["robo-trader"]),
        patch("robo_trader.runner_async._enforce_preflight_or_exit"),
        patch("robo_trader.runner_async.load_config", return_value=cfg),
        patch(
            "robo_trader.runner_async._start_paper_safety_runtime",
            side_effect=startup_error,
        ) as replay,
        patch("robo_trader.runner_async.check_gateway_zombies") as gateway_check,
        patch("robo_trader.runner_async.asyncio.run") as run_async,
        patch("robo_trader.runner_async._write_exit_audit") as exit_audit,
        patch("robo_trader.runner_async._fire_runner_exit_alert") as exit_alert,
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 7
    replay.assert_called_once_with(cfg)
    gateway_check.assert_not_called()
    run_async.assert_not_called()
    exit_audit.assert_called_once()
    exit_alert.assert_called_once()


def test_main_refuses_runtime_until_terminal_settlement_is_ready(tmp_path):
    cfg = SimpleNamespace(
        execution=SimpleNamespace(mode=SimpleNamespace(value="paper")),
        ibkr=SimpleNamespace(readonly=True, port=4002),
        runtime_contract=_runtime_contract(tmp_path),
    )
    context = SimpleNamespace(runtime_contract=cfg.runtime_contract)

    with (
        patch.object(sys, "argv", ["robo-trader"]),
        patch("robo_trader.runner_async._enforce_preflight_or_exit"),
        patch("robo_trader.runner_async.load_config", return_value=cfg),
        patch(
            "robo_trader.runner_async._start_paper_safety_runtime",
            return_value=object(),
        ),
        patch(
            "robo_trader.runner_async.validate_runtime_safety",
            return_value=context,
        ),
        patch("robo_trader.runner_async.resolve_environment", return_value={}),
        patch("robo_trader.runner_async.check_gateway_zombies") as gateway_check,
        patch("robo_trader.runner_async.asyncio.run") as run_async,
        patch("robo_trader.runner_async._write_exit_audit") as exit_audit,
        patch("robo_trader.runner_async._fire_runner_exit_alert") as exit_alert,
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 9
    gateway_check.assert_not_called()
    run_async.assert_not_called()
    exit_audit.assert_called_once_with(
        "paper_terminal_settlement_not_ready",
        exit_code=9,
    )
    exit_alert.assert_called_once_with(
        "paper_terminal_settlement_not_ready",
        {"required_pr": "2B.3"},
    )


@pytest.mark.asyncio
async def test_run_once_retains_started_coordinator_for_runner(tmp_path):
    cfg = _cfg(tmp_path)
    SafetyJournal(Path(cfg.runtime_contract.safety_journal_path)).initialize(
        execution_domain_scope=cfg.runtime_contract.safety_execution_domain_scope,
        account_scope=cfg.runtime_contract.safety_account_scope,
    )
    coordinator = _start_paper_safety_runtime(cfg)
    runtime_context = object()
    resources = SimpleNamespace(database=object(), gateway=object())
    runner = MagicMock()
    runner.run = AsyncMock()
    runner.cleanup = AsyncMock()

    with (
        patch("robo_trader.runner_async.RuntimeSafetyContext", object),
        patch(
            "robo_trader.runner_async._start_paper_order_runtime",
            new_callable=AsyncMock,
            return_value=resources,
        ),
        patch(
            "robo_trader.runner_async._close_paper_order_runtime",
            new_callable=AsyncMock,
        ),
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner) as runner_class,
    ):
        await run_once(
            symbols=["AAPL"],
            safety_runtime=coordinator,
            runtime_context=runtime_context,
        )

    assert runner_class.call_args.kwargs["safety_runtime"] is coordinator
    assert runner_class.call_args.kwargs["shared_database"] is resources.database
    assert runner_class.call_args.kwargs["paper_reduction_gateway"] is resources.gateway
    runner.run.assert_awaited_once_with(["AAPL"])


@pytest.mark.asyncio
async def test_continuous_reuses_one_account_coordinator_for_portfolio_runners(tmp_path):
    cfg = _cfg(tmp_path)
    SafetyJournal(Path(cfg.runtime_contract.safety_journal_path)).initialize(
        execution_domain_scope=cfg.runtime_contract.safety_execution_domain_scope,
        account_scope=cfg.runtime_contract.safety_account_scope,
    )
    coordinator = _start_paper_safety_runtime(cfg)
    runtime_context = object()
    resources = SimpleNamespace(database=object(), gateway=object())
    runner = MagicMock()
    runner.recovery_in_progress = False
    runner._recovery_exhausted = False
    runner.health = None
    runner.run = AsyncMock()
    runner.teardown = AsyncMock()
    runner.cleanup = AsyncMock()
    portfolio = SimpleNamespace(
        id="default",
        name="Default",
        starting_cash=100000,
        symbols=["AAPL"],
        active=True,
    )

    with (
        patch("signal.signal"),
        patch("robo_trader.runner_async.RuntimeSafetyContext", object),
        patch(
            "robo_trader.runner_async._start_paper_order_runtime",
            new_callable=AsyncMock,
            return_value=resources,
        ),
        patch(
            "robo_trader.runner_async._close_paper_order_runtime",
            new_callable=AsyncMock,
        ),
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner) as runner_class,
        patch(
            "robo_trader.runner_async._setup_continuous_runner",
            new_callable=AsyncMock,
        ),
        patch("robo_trader.runner_async.is_trading_allowed", return_value=True),
        patch(
            "robo_trader.multiuser.portfolio_config.load_portfolio_configs",
            return_value=[portfolio],
        ),
        patch(
            "robo_trader.runner_async.sleep_unless_shutdown",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ),
        patch("robo_trader.runner_async._write_exit_audit"),
        patch("robo_trader.runner_async._fire_runner_exit_alert"),
    ):
        await run_continuous(
            symbols=["AAPL"],
            interval_seconds=1,
            safety_runtime=coordinator,
            runtime_context=runtime_context,
        )

    assert runner_class.call_args.kwargs["safety_runtime"] is coordinator
    assert runner_class.call_args.kwargs["shared_database"] is resources.database
    assert runner_class.call_args.kwargs["paper_reduction_gateway"] is resources.gateway
    runner.cleanup.assert_awaited_once()
