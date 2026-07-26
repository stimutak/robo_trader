import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import robo_trader.runner_async as runner_module
from robo_trader.config import RuntimeContract
from robo_trader.execution import PaperExecutor
from robo_trader.paper_reduction_gateway import PaperReductionGateway
from robo_trader.runner_async import (
    AsyncRunner,
    _close_paper_order_runtime_owned,
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
    runner._ensure_health_monitor_for_activation = AsyncMock()
    runner.cleanup = AsyncMock()

    with pytest.raises(RuntimeError, match="setup failed"):
        await _setup_continuous_runner(runner)

    runner.cleanup.assert_awaited_once()
    runner._ensure_health_monitor_for_activation.assert_not_awaited()


@pytest.mark.asyncio
async def test_continuous_health_attach_failure_never_registers_executor() -> None:
    runner = MagicMock()
    runner.setup = AsyncMock()
    runner._ensure_health_monitor_for_activation = AsyncMock(
        side_effect=RuntimeError("health attachment failed")
    )
    runner._activate_after_setup = MagicMock()
    runner.cleanup = AsyncMock()

    with pytest.raises(RuntimeError, match="health attachment failed"):
        await _setup_continuous_runner(runner)

    runner._activate_after_setup.assert_not_called()
    runner.cleanup.assert_awaited_once()
    assert runner._setup_complete is False


def test_runner_setup_cannot_publish_executor_before_post_setup_readiness() -> None:
    assert "register_paper_executor" not in inspect.getsource(AsyncRunner.setup)


@pytest.mark.asyncio
async def test_activation_health_gate_binds_live_monitor_to_exact_broker_client() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.ib = object()
    runner.health = None
    monitor_task = SimpleNamespace(done=lambda: False)

    async def attach_health() -> None:
        runner.health = SimpleNamespace(
            _ib_client=runner.ib,
            _monitor_task=monitor_task,
            status=runner_module.HealthStatus.HEALTHY,
        )

    runner._attach_health_monitor = AsyncMock(side_effect=attach_health)

    await runner._ensure_health_monitor_for_activation()

    runner._attach_health_monitor.assert_awaited_once_with()
    assert runner.health._ib_client is runner.ib
    assert runner.health._monitor_task is monitor_task


@pytest.mark.asyncio
async def test_continuous_activation_runs_only_after_health_attachment() -> None:
    order: list[str] = []
    runner = MagicMock()
    runner.setup = AsyncMock(side_effect=lambda: order.append("setup"))
    runner._ensure_health_monitor_for_activation = AsyncMock(
        side_effect=lambda: order.append("health")
    )
    runner._activate_after_setup = MagicMock(side_effect=lambda: order.append("activate"))
    runner.cleanup = AsyncMock()

    await _setup_continuous_runner(runner)

    assert order == ["setup", "health", "activate"]
    runner.cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_continuous_cleanup_drains_repeated_cancel_and_preserves_setup_error() -> (
    None
):
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def blocking_cleanup() -> None:
        cleanup_entered.set()
        await release_cleanup.wait()
        cleanup_finished.set()

    runner = MagicMock()
    runner.setup = AsyncMock()
    runner._ensure_health_monitor_for_activation = AsyncMock(
        side_effect=RuntimeError("health attachment failed")
    )
    runner._activate_after_setup = MagicMock()
    runner.cleanup = AsyncMock(side_effect=blocking_cleanup)

    owner = asyncio.create_task(_setup_continuous_runner(runner))
    await cleanup_entered.wait()
    owner.cancel()
    await asyncio.sleep(0)
    assert owner.done() is False
    owner.cancel()
    await asyncio.sleep(0)
    assert owner.done() is False
    release_cleanup.set()

    with pytest.raises(RuntimeError, match="health attachment failed"):
        await owner

    assert cleanup_finished.is_set()
    runner._activate_after_setup.assert_not_called()
    runner.cleanup.assert_awaited_once()
    assert runner._setup_complete is False


@pytest.mark.asyncio
async def test_one_shot_activation_orders_health_before_publish_and_rolls_back() -> None:
    order: list[str] = []
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._setup_complete = False
    runner.setup = AsyncMock(side_effect=lambda: order.append("setup"))
    runner._ensure_health_monitor_for_activation = AsyncMock(
        side_effect=lambda: order.append("health")
    )

    def fail_activation() -> None:
        order.append("activate")
        raise RuntimeError("activation failed")

    runner._activate_after_setup = MagicMock(side_effect=fail_activation)
    runner.cleanup = AsyncMock(side_effect=lambda: order.append("cleanup"))

    with pytest.raises(RuntimeError, match="activation failed"):
        await AsyncRunner.run(runner, [])

    assert order == ["setup", "health", "activate", "cleanup"]
    runner.cleanup.assert_awaited_once_with()
    assert runner._setup_complete is False


@pytest.mark.asyncio
async def test_one_shot_health_failure_never_activates_and_rolls_back() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._setup_complete = False
    runner.setup = AsyncMock()
    runner._ensure_health_monitor_for_activation = AsyncMock(
        side_effect=RuntimeError("health attachment failed")
    )
    runner._activate_after_setup = MagicMock()
    runner.cleanup = AsyncMock()

    with pytest.raises(RuntimeError, match="health attachment failed"):
        await AsyncRunner.run(runner, [])

    runner._activate_after_setup.assert_not_called()
    runner.cleanup.assert_awaited_once_with()
    assert runner._setup_complete is False


@pytest.mark.asyncio
async def test_one_shot_run_activates_recovery_pending_gateway_before_cycle() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._setup_complete = False
    runner.setup = AsyncMock()
    runner._ensure_health_monitor_for_activation = AsyncMock()
    runner.cleanup = AsyncMock()
    runner.executor = PaperExecutor()
    runner.portfolio_id = "portfolio-a"

    gateway = PaperReductionGateway.__new__(PaperReductionGateway)
    gateway._started = False
    gateway._diagnostic_recovery_required = True
    gateway.register_paper_executor = MagicMock()
    runner.paper_reduction_gateway = gateway

    activate = runner._activate_after_setup

    def activate_then_stop() -> None:
        activate()
        raise RuntimeError("stop after production activation")

    runner._activate_after_setup = MagicMock(side_effect=activate_then_stop)

    with (
        patch.object(runner_module, "_clear_exit_audit"),
        pytest.raises(RuntimeError, match="stop after production activation"),
    ):
        await AsyncRunner.run(runner, [])

    gateway.register_paper_executor.assert_called_once_with(
        "portfolio-a",
        runner.executor,
    )
    runner.cleanup.assert_awaited_once_with()
    assert runner._setup_complete is False


def test_activation_rejects_explicitly_closed_gateway() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._setup_complete = False
    runner.executor = PaperExecutor()
    runner.portfolio_id = "portfolio-a"

    gateway = PaperReductionGateway.__new__(PaperReductionGateway)
    gateway._started = False
    gateway._diagnostic_recovery_required = False
    gateway.register_paper_executor = MagicMock()
    runner.paper_reduction_gateway = gateway

    with pytest.raises(RuntimeError, match="started reduction gateway"):
        runner._activate_after_setup()

    gateway.register_paper_executor.assert_not_called()
    assert runner._setup_complete is False


@pytest.mark.asyncio
async def test_shared_runtime_startup_database_cleanup_drains_repeated_cancellation(
    tmp_path,
) -> None:
    class ExactSafetyRuntime:
        started = True

    context = SimpleNamespace(
        runtime_contract=SimpleNamespace(database_path=str(tmp_path / "paper-ledger.db"))
    )
    database = MagicMock()
    database.initialize = AsyncMock()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()

    async def blocking_close() -> None:
        close_entered.set()
        await release_close.wait()
        close_finished.set()

    database.close = AsyncMock(side_effect=blocking_close)
    gateway = MagicMock()
    gateway.start = AsyncMock(side_effect=RuntimeError("gateway startup failed"))

    with (
        patch.object(
            runner_module,
            "require_paper_terminal_settlement_ready",
        ),
        patch.object(
            runner_module,
            "assert_validated_runtime_safety_context",
            return_value=context,
        ),
        patch.object(runner_module, "SafetyRuntimeCoordinator", ExactSafetyRuntime),
        patch.object(runner_module, "AsyncTradingDatabase", return_value=database),
        patch.object(runner_module, "PaperReductionGateway", return_value=gateway),
    ):
        owner = asyncio.create_task(_start_paper_order_runtime(context, ExactSafetyRuntime()))
        await close_entered.wait()
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        release_close.set()

        with pytest.raises(RuntimeError, match="gateway startup failed"):
            await owner

    assert close_finished.is_set()
    database.close.assert_awaited_once()


def _install_minimal_cleanup_state(runner: AsyncRunner) -> None:
    runner._setup_complete = True
    runner.subprocess_monitor_task = None
    runner.risk_monitor_task = None
    runner.cleanup_task = None
    runner.stop_loss_monitor = None
    runner.use_advanced_risk = False
    runner.advanced_risk = None
    runner.ib = None
    runner.db = None
    runner.ws_client = None


def _tracking_cycle_manager(*, active_attribute: str, task_attribute: str):
    manager = SimpleNamespace(stop_calls=0, stopped=asyncio.Event())
    setattr(manager, active_attribute, True)
    background_task = asyncio.create_task(asyncio.Event().wait())
    setattr(manager, task_attribute, background_task)

    async def stop() -> None:
        manager.stop_calls += 1
        setattr(manager, active_attribute, False)
        background_task.cancel()
        await asyncio.gather(background_task, return_exceptions=True)
        manager.stopped.set()

    manager.stop = stop
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "early_cleanup_error",
    [RuntimeError("health cleanup failed"), asyncio.CancelledError("health cleanup cancelled")],
    ids=["runtime-error", "cancelled"],
)
async def test_partial_setup_cleanup_stops_both_cycle_managers_after_earlier_failure(
    early_cleanup_error: BaseException,
) -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _install_minimal_cleanup_state(runner)
    runner.setup = AsyncMock()
    runner._ensure_health_monitor_for_activation = AsyncMock(
        side_effect=ValueError("activation health gate failed")
    )
    runner._activate_after_setup = MagicMock()
    runner.health = SimpleNamespace(
        stop_monitoring=AsyncMock(side_effect=early_cleanup_error),
    )
    production = _tracking_cycle_manager(
        active_attribute="is_running",
        task_attribute="monitoring_task",
    )
    correlation = _tracking_cycle_manager(
        active_attribute="running",
        task_attribute="update_task",
    )
    runner.production_monitor = production
    runner.correlation_manager = correlation

    with pytest.raises(ValueError, match="activation health gate failed"):
        await _setup_continuous_runner(runner)

    runner._activate_after_setup.assert_not_called()
    assert production.stopped.is_set()
    assert correlation.stopped.is_set()
    assert production.stop_calls == 1
    assert correlation.stop_calls == 1
    assert runner.production_monitor is None
    assert runner.correlation_manager is None
    assert runner.health is None


@pytest.mark.asyncio
async def test_two_persistent_cycle_teardowns_preserve_managers_until_final_cleanup() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _install_minimal_cleanup_state(runner)
    runner.health = None
    production = _tracking_cycle_manager(
        active_attribute="is_running",
        task_attribute="monitoring_task",
    )
    correlation = _tracking_cycle_manager(
        active_attribute="running",
        task_attribute="update_task",
    )
    runner.production_monitor = production
    runner.correlation_manager = correlation

    await runner.teardown(full_cleanup=False)
    await runner.teardown(full_cleanup=False)

    assert production.stop_calls == 0
    assert correlation.stop_calls == 0
    assert production.is_running is True
    assert correlation.running is True
    assert production.monitoring_task.done() is False
    assert correlation.update_task.done() is False
    assert runner.production_monitor is production
    assert runner.correlation_manager is correlation

    await runner.teardown(full_cleanup=True)
    await runner.cleanup()

    assert production.stop_calls == 1
    assert correlation.stop_calls == 1
    assert runner.production_monitor is None
    assert runner.correlation_manager is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_kind", ["sync", "async"])
async def test_unproven_manager_stop_retains_reference_and_fails_cleanup(
    stop_kind: str,
) -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _install_minimal_cleanup_state(runner)
    runner.health = None
    pending_task = asyncio.create_task(asyncio.Event().wait())
    manager = SimpleNamespace(
        is_running=True,
        monitoring_task=pending_task,
        stop_calls=0,
    )

    def incomplete_sync_stop() -> None:
        manager.stop_calls += 1

    async def incomplete_async_stop() -> None:
        manager.stop_calls += 1

    manager.stop = incomplete_sync_stop if stop_kind == "sync" else incomplete_async_stop
    runner.production_monitor = manager
    runner.correlation_manager = None

    try:
        with pytest.raises(RuntimeError, match="production_monitor stop returned"):
            await runner.cleanup()

        assert manager.stop_calls == 1
        assert manager.is_running is True
        assert manager.monitoring_task.done() is False
        assert runner.production_monitor is manager
    finally:
        pending_task.cancel()
        await asyncio.gather(pending_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_manager_state_property_failure_is_isolated_and_retains_ownership() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _install_minimal_cleanup_state(runner)
    runner.health = None

    class UninspectableManager:
        is_running = True
        stop = MagicMock()

        @property
        def monitoring_task(self):
            raise RuntimeError("production state unreadable")

    production = UninspectableManager()
    correlation = _tracking_cycle_manager(
        active_attribute="running",
        task_attribute="update_task",
    )
    runner.production_monitor = production
    runner.correlation_manager = correlation
    runner.ib = SimpleNamespace(disconnect=AsyncMock(), stop=AsyncMock())
    runner.db = SimpleNamespace(close=AsyncMock())
    runner._owns_database = True

    with pytest.raises(RuntimeError, match="production state unreadable"):
        await runner.cleanup()

    production.stop.assert_not_called()
    assert runner.production_monitor is production
    assert correlation.stop_calls == 1
    assert runner.correlation_manager is None
    runner.ib.disconnect.assert_awaited_once_with()
    runner.ib.stop.assert_awaited_once_with()
    runner.db.close.assert_awaited_once_with()


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
@pytest.mark.parametrize(
    "cleanup_error",
    [RuntimeError("runner cleanup failed"), asyncio.CancelledError()],
    ids=["error", "cancelled"],
)
async def test_run_once_closes_shared_runtime_after_runner_cleanup_failure(
    tmp_path,
    cleanup_error: BaseException,
) -> None:
    cfg = _cfg(tmp_path)
    SafetyJournal(Path(cfg.runtime_contract.safety_journal_path)).initialize(
        execution_domain_scope=cfg.runtime_contract.safety_execution_domain_scope,
        account_scope=cfg.runtime_contract.safety_account_scope,
    )
    coordinator = _start_paper_safety_runtime(cfg)
    resources = SimpleNamespace(database=object(), gateway=object())
    runner = MagicMock()
    runner.run = AsyncMock()
    runner.cleanup = AsyncMock(side_effect=cleanup_error)

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
        ) as close_runtime,
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner),
    ):
        with pytest.raises(type(cleanup_error)):
            await run_once(
                symbols=["AAPL"],
                safety_runtime=coordinator,
                runtime_context=object(),
            )

    close_runtime.assert_awaited_once_with(resources)


@pytest.mark.asyncio
async def test_run_once_preserves_run_error_after_all_cleanup_attempts(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    SafetyJournal(Path(cfg.runtime_contract.safety_journal_path)).initialize(
        execution_domain_scope=cfg.runtime_contract.safety_execution_domain_scope,
        account_scope=cfg.runtime_contract.safety_account_scope,
    )
    coordinator = _start_paper_safety_runtime(cfg)
    resources = SimpleNamespace(database=object(), gateway=object())
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=ValueError("primary run failure"))
    runner.cleanup = AsyncMock(side_effect=RuntimeError("secondary cleanup failure"))

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
            side_effect=OSError("tertiary runtime close failure"),
        ) as close_runtime,
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner),
    ):
        with pytest.raises(ValueError, match="primary run failure"):
            await run_once(
                symbols=["AAPL"],
                safety_runtime=coordinator,
                runtime_context=object(),
            )

    runner.cleanup.assert_awaited_once()
    close_runtime.assert_awaited_once_with(resources)


@pytest.mark.asyncio
async def test_run_once_drains_runner_cleanup_through_repeated_cancellation(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path)
    SafetyJournal(Path(cfg.runtime_contract.safety_journal_path)).initialize(
        execution_domain_scope=cfg.runtime_contract.safety_execution_domain_scope,
        account_scope=cfg.runtime_contract.safety_account_scope,
    )
    coordinator = _start_paper_safety_runtime(cfg)
    resources = SimpleNamespace(database=object(), gateway=object())
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def blocking_cleanup() -> None:
        cleanup_entered.set()
        await release_cleanup.wait()
        cleanup_finished.set()

    runner = MagicMock()
    runner.run = AsyncMock()
    runner.cleanup = AsyncMock(side_effect=blocking_cleanup)

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
        ) as close_runtime,
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner),
    ):
        owner = asyncio.create_task(
            run_once(
                symbols=["AAPL"],
                safety_runtime=coordinator,
                runtime_context=object(),
            )
        )
        await cleanup_entered.wait()
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await owner

    assert cleanup_finished.is_set()
    runner.cleanup.assert_awaited_once()
    close_runtime.assert_awaited_once_with(resources)


@pytest.mark.asyncio
async def test_shared_runtime_close_survives_repeated_outer_cancellation() -> None:
    resources = SimpleNamespace(database=object(), gateway=object())
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()

    async def close_runtime(_resources) -> None:
        assert _resources is resources
        close_entered.set()
        await release_close.wait()
        close_finished.set()

    with patch(
        "robo_trader.runner_async._close_paper_order_runtime",
        side_effect=close_runtime,
    ):
        owner = asyncio.create_task(_close_paper_order_runtime_owned(resources))
        await close_entered.wait()
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await owner

    assert close_finished.is_set()


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


@pytest.mark.asyncio
async def test_continuous_closes_shared_runtime_after_runner_cleanup_cancellation(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path)
    SafetyJournal(Path(cfg.runtime_contract.safety_journal_path)).initialize(
        execution_domain_scope=cfg.runtime_contract.safety_execution_domain_scope,
        account_scope=cfg.runtime_contract.safety_account_scope,
    )
    coordinator = _start_paper_safety_runtime(cfg)
    resources = SimpleNamespace(database=object(), gateway=object())
    runner = MagicMock()
    runner.recovery_in_progress = False
    runner._recovery_exhausted = False
    runner.health = None
    runner.run = AsyncMock()
    runner.teardown = AsyncMock()
    runner.cleanup = AsyncMock(side_effect=asyncio.CancelledError)
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
        ) as close_runtime,
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner),
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
        with pytest.raises(asyncio.CancelledError):
            await run_continuous(
                symbols=["AAPL"],
                interval_seconds=1,
                safety_runtime=coordinator,
                runtime_context=object(),
            )

    runner.cleanup.assert_awaited_once()
    close_runtime.assert_awaited_once_with(resources)


@pytest.mark.asyncio
async def test_continuous_drains_every_runner_cleanup_through_repeated_cancellation(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path)
    SafetyJournal(Path(cfg.runtime_contract.safety_journal_path)).initialize(
        execution_domain_scope=cfg.runtime_contract.safety_execution_domain_scope,
        account_scope=cfg.runtime_contract.safety_account_scope,
    )
    coordinator = _start_paper_safety_runtime(cfg)
    resources = SimpleNamespace(database=object(), gateway=object())
    first_cleanup_entered = asyncio.Event()
    release_first_cleanup = asyncio.Event()
    first_cleanup_finished = asyncio.Event()
    second_cleanup_finished = asyncio.Event()

    async def first_cleanup() -> None:
        first_cleanup_entered.set()
        await release_first_cleanup.wait()
        first_cleanup_finished.set()

    async def second_cleanup() -> None:
        second_cleanup_finished.set()

    runners = []
    for cleanup in (first_cleanup, second_cleanup):
        runner = MagicMock()
        runner.recovery_in_progress = False
        runner._recovery_exhausted = False
        runner.health = None
        runner.run = AsyncMock()
        runner.teardown = AsyncMock()
        runner.cleanup = AsyncMock(side_effect=cleanup)
        runners.append(runner)

    portfolios = [
        SimpleNamespace(
            id=portfolio_id,
            name=portfolio_id.title(),
            starting_cash=100000,
            symbols=[symbol],
            active=True,
        )
        for portfolio_id, symbol in (("alpha", "AAPL"), ("beta", "MSFT"))
    ]

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
        ) as close_runtime,
        patch("robo_trader.runner_async.AsyncRunner", side_effect=runners),
        patch(
            "robo_trader.runner_async._setup_continuous_runner",
            new_callable=AsyncMock,
        ),
        patch("robo_trader.runner_async.is_trading_allowed", return_value=True),
        patch(
            "robo_trader.multiuser.portfolio_config.load_portfolio_configs",
            return_value=portfolios,
        ),
        patch(
            "robo_trader.runner_async.sleep_unless_shutdown",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ),
        patch("robo_trader.runner_async._write_exit_audit"),
        patch("robo_trader.runner_async._fire_runner_exit_alert"),
    ):
        owner = asyncio.create_task(
            run_continuous(
                symbols=["AAPL", "MSFT"],
                interval_seconds=1,
                safety_runtime=coordinator,
                runtime_context=object(),
            )
        )
        await first_cleanup_entered.wait()
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        release_first_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await owner

    assert first_cleanup_finished.is_set()
    assert second_cleanup_finished.is_set()
    runners[0].cleanup.assert_awaited_once()
    runners[1].cleanup.assert_awaited_once()
    close_runtime.assert_awaited_once_with(resources)
