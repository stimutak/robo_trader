"""Machine-enforced containment for the narrow Gate-A paper profile."""

from __future__ import annotations

import ast
import copy
import inspect
import pickle
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import robo_trader.execution as execution_module
import robo_trader.paper_execution_capability as capability_module
from robo_trader.config import Config
from robo_trader.core.engine import TradingEngine
from robo_trader.execution import Order, PaperExecutor
from robo_trader.gatea_containment import (
    ALLOWED_STRATEGIES,
    GateAContainmentError,
    assert_gate_a_runner_options,
    validate_gate_a_environment,
    validate_gate_a_order,
)
from robo_trader.multiuser.portfolio_config import PortfolioConfig, load_portfolio_configs
from robo_trader.order_manager import OrderManager, OrderStatus
from robo_trader.paper_execution_capability import (
    PaperExecutionCapabilityError,
    _issue_gateway_reduction_terminal_dispatch,
    _submit_gateway_reduction_once,
)
from robo_trader.runner.signal_generator import SignalGenerator
from robo_trader.runner.trade_executor import TradeExecutor
from robo_trader.runner_async import run_continuous
from tests.paper_execution_test_support import bind_gateway_reduction_harness


@pytest.mark.parametrize(
    "flag",
    [
        "AI_TRADING_ENABLED",
        "EXECUTION_SHORT_SELLING",
        "ML_ENHANCED_ENABLED",
        "ML_SELECTOR_ENABLED",
        "RISK_TAKE_PROFIT_ENABLED",
        "SMART_EXECUTION_ENABLED",
    ],
)
def test_gate_a_rejects_every_disabled_environment_capability(flag: str) -> None:
    with pytest.raises(GateAContainmentError, match=flag):
        validate_gate_a_environment({flag: "true"})


@pytest.mark.parametrize(
    "strategy_value",
    ["pairs_trading", "stat_arbitrage", "mean_reversion", "baseline_sma,pairs_trading"],
)
def test_gate_a_rejects_nonbaseline_strategy_configuration(strategy_value: str) -> None:
    with pytest.raises(GateAContainmentError, match="requires exactly baseline_sma"):
        validate_gate_a_environment({"STRATEGY_ENABLED": strategy_value})


def test_gate_a_rejects_contradictory_strategy_aliases() -> None:
    with pytest.raises(GateAContainmentError, match="contradict"):
        validate_gate_a_environment(
            {
                "STRATEGY_ENABLED": "baseline_sma",
                "STRATEGY_ENABLED_STRATEGIES": "pairs_trading",
            }
        )


def test_gate_a_safe_environment_has_one_strategy() -> None:
    profile = validate_gate_a_environment({})

    assert profile.enabled_strategies == ALLOWED_STRATEGIES


def test_example_environment_uses_the_only_admitted_strategy() -> None:
    example = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")

    assignments = [line for line in example.splitlines() if line.startswith("STRATEGY_ENABLED=")]
    assert assignments == ["STRATEGY_ENABLED=baseline_sma  # Gate-A permits exactly this strategy"]


@pytest.mark.parametrize(
    "override",
    [
        {"execution": {"enable_short_selling": True}},
        {"execution": {"use_smart_execution": True}},
        {"strategy": {"enabled_strategies": ["pairs_trading"]}},
        {"strategy": {"enable_ai_discovery": True}},
        {"strategy": {"enable_ml_selectors": True}},
        {"risk": {"enable_take_profit": True}},
    ],
)
def test_direct_config_contradictions_fail_closed(override: dict) -> None:
    with pytest.raises(ValidationError, match="Gate-A containment contradiction"):
        Config(**override)


def test_direct_config_rejects_invalid_explicit_portfolio_strategy() -> None:
    with pytest.raises(ValidationError, match="portfolio_configs.*baseline_sma"):
        Config(portfolio_configs=[{"id": "aggressive", "enabled_strategies": ["pairs_trading"]}])


def test_explicit_portfolio_strategy_never_falls_back(monkeypatch) -> None:
    monkeypatch.setenv(
        "PORTFOLIOS",
        '[{"id":"aggressive","enabled_strategies":"pairs_trading"}]',
    )

    with pytest.raises(GateAContainmentError, match="baseline_sma"):
        load_portfolio_configs()

    assert "using default" not in inspect.getsource(run_continuous)


def test_empty_explicit_portfolios_never_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("PORTFOLIOS", "")

    with pytest.raises(ValueError, match="must not be empty"):
        load_portfolio_configs()


def test_portfolio_strategy_override_accepts_only_baseline() -> None:
    portfolio = PortfolioConfig(
        id="baseline",
        name="Baseline",
        enabled_strategies=["BASELINE_SMA"],
    )

    assert portfolio.enabled_strategies == ["baseline_sma"]


@pytest.mark.parametrize(
    "options",
    [
        {"use_ml_strategy": True, "use_ml_enhanced": False, "use_smart_execution": False},
        {"use_ml_strategy": False, "use_ml_enhanced": True, "use_smart_execution": False},
        {"use_ml_strategy": False, "use_ml_enhanced": False, "use_smart_execution": True},
    ],
)
def test_runner_constructor_options_cannot_reenable_quarantined_paths(options: dict) -> None:
    with pytest.raises(GateAContainmentError):
        assert_gate_a_runner_options(env={}, **options)


@pytest.mark.parametrize(
    ("side", "take_profit", "expected_fragment"),
    [
        ("SELL_SHORT", None, "blocks new short exposure"),
        ("BUY", 110.0, "take-profit"),
    ],
)
def test_disabled_order_shapes_fail_before_capability_admission(
    side: str, take_profit: float | None, expected_fragment: str
) -> None:
    admitted, reason = validate_gate_a_order(
        side=side,
        take_profit=take_profit,
    )

    assert admitted is False
    assert expected_fragment in reason


@pytest.mark.parametrize("side", ["SELL", "BUY_TO_COVER"])
def test_semantic_reductions_remain_structurally_admitted(side: str) -> None:
    admitted, reason = validate_gate_a_order(
        side=side,
        take_profit=None,
    )

    assert admitted is True
    assert reason == "Gate-A semantic reduction"


def test_forgeable_intent_source_field_no_longer_exists() -> None:
    with pytest.raises(TypeError, match="intent_source"):
        Order("AAPL", 1, "BUY", 100.0, intent_source="baseline_sma")


def test_baseline_intent_and_terminal_capability_have_no_runtime_call_sites() -> None:
    package = Path(__file__).parents[1] / "robo_trader"
    issue_calls: list[Path] = []
    terminal_calls: list[Path] = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == (
                "issue_baseline_entry_intent"
            ):
                issue_calls.append(path)
            elif isinstance(node.func, ast.Name) and node.func.id == (
                "_submit_gateway_baseline_once"
            ):
                terminal_calls.append(path)

    assert issue_calls == []
    assert terminal_calls == []


def test_no_independently_bindable_baseline_authority_exists() -> None:
    assert not hasattr(capability_module, "_bind_paper_execution_authority")
    assert not hasattr(capability_module, "_bind_paper_reduction_execution_authority")
    authority = bind_gateway_reduction_harness(
        PaperExecutor(),
        "default",
    ).authority
    assert not hasattr(authority, "_submit_baseline_once")
    assert not hasattr(authority, "_submit_reduction_once")


def test_paper_executor_rejects_every_naked_submission(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_module,
        "os",
        SimpleNamespace(path=SimpleNamespace(exists=lambda _path: False)),
    )
    executor = PaperExecutor(slippage_bps=0)

    buy = executor.place_order(Order("AAPL", 1, "BUY", 100.0))
    sell = executor.place_order(Order("AAPL", 1, "SELL", 100.0))
    cover = executor.place_order(Order("AAPL", 1, "BUY_TO_COVER", 100.0))
    private = executor._place_simple_order(Order("AAPL", 1, "BUY", 100.0))

    for result in (buy, sell, cover, private):
        assert result.ok is False
        assert "submission capability" in result.message
    assert executor.fills == {}


def test_bound_authority_preserves_exact_reductions() -> None:
    executor = PaperExecutor()
    harness = bind_gateway_reduction_harness(executor, "default")

    sell = harness.submit(
        Order("AAPL", 1, "SELL", Decimal("100.0000")),
        pre_position_quantity=Decimal("1"),
    )
    cover = harness.submit(
        Order("MSFT", 2, "BUY_TO_COVER", Decimal("50.0000")),
        pre_position_quantity=Decimal("-2"),
    )

    assert sell.ok is True
    assert cover.ok is True
    assert len(executor.fills) == 2


@pytest.mark.parametrize(
    ("side", "quantity", "pre_position"),
    [
        ("SELL", 2, Decimal("1")),
        ("SELL", 1, Decimal("-1")),
        ("BUY_TO_COVER", 2, Decimal("-1")),
        ("BUY_TO_COVER", 1, Decimal("1")),
    ],
)
def test_reduction_capability_cannot_cross_zero(
    side: str, quantity: int, pre_position: Decimal
) -> None:
    executor = PaperExecutor()
    harness = bind_gateway_reduction_harness(executor, "default")

    result = harness.submit(
        Order("AAPL", quantity, side, Decimal("100.0000")),
        pre_position_quantity=pre_position,
    )

    assert result.ok is False
    assert "exposure" in result.message
    assert executor.fills == {}


def test_reduction_terminal_dispatch_never_calls_mutable_executor_method(monkeypatch) -> None:
    executor = PaperExecutor()
    harness = bind_gateway_reduction_harness(executor, "default")
    original = Order("AAPL", 1, "SELL", Decimal("100.0000"))
    dispatch = harness.issue(original, pre_position_quantity=Decimal("1"))
    monkeypatch.setattr(
        executor,
        "_place_simple_order",
        lambda *_args, **_kwargs: pytest.fail("mutable executor method must not receive authority"),
    )
    monkeypatch.setattr(
        capability_module,
        "_execute_sealed_paper_fill",
        lambda *_args, **_kwargs: pytest.fail("captured terminal sink must be immutable"),
    )
    monkeypatch.setattr(
        capability_module,
        "consume_paper_execution_capability",
        lambda *_args, **_kwargs: pytest.fail("captured consume primitive must be immutable"),
    )

    result = _submit_gateway_reduction_once(
        harness.authority,
        dispatch,
        submitter=harness.submitter_identity,
        order=original,
        pre_position_quantity=Decimal("1"),
    )

    assert result.ok is True
    assert len(executor.fills) == 1


def test_terminal_fill_direct_call_without_exact_authority_cannot_fill() -> None:
    executor = PaperExecutor()
    order = Order("AAPL", 1, "SELL", Decimal("100.0000"))

    with pytest.raises(PaperExecutionCapabilityError, match="exact submission capability"):
        capability_module._execute_sealed_paper_fill(executor, order, None)
    with pytest.raises(PaperExecutionCapabilityError, match="exact submission capability"):
        capability_module._execute_sealed_paper_fill(executor, order, object())
    with pytest.raises(PaperExecutionCapabilityError, match="exact consumed"):
        capability_module._apply_consumed_paper_fill(executor, order, None)
    with pytest.raises(PaperExecutionCapabilityError, match="exact consumed"):
        capability_module._apply_consumed_paper_fill(executor, order, object())

    assert executor.fills == {}


def test_terminal_authority_state_is_not_importable_or_directly_constructible() -> None:
    for name in (
        "_CAPABILITY_TOKEN",
        "_CAPABILITIES",
        "_CapabilityRecord",
        "_REGISTRY_LOCK",
        "_TERMINAL_DISPATCH_TOKEN",
        "_REDUCTION_DISPATCHES",
    ):
        assert not hasattr(capability_module, name)

    with pytest.raises(PaperExecutionCapabilityError, match="minted only"):
        capability_module._PaperExecutionCapability()

    executor = PaperExecutor()
    order = Order("AAPL", 1, "SELL", Decimal("100.0000"))
    unissued = object.__new__(capability_module._PaperExecutionCapability)
    with pytest.raises(PaperExecutionCapabilityError, match="unknown|already consumed"):
        capability_module._execute_sealed_paper_fill(executor, order, unissued)
    assert executor.fills == {}


def test_reduction_traceback_retains_only_burned_capability() -> None:
    executor = PaperExecutor()
    harness = bind_gateway_reduction_harness(executor, "default")
    original = Order("AAPL", 1, "SELL", Decimal("100.0000"))
    dispatch = harness.issue(original, pre_position_quantity=Decimal("1"))

    class RaisingFills(dict):
        def __setitem__(self, _key, _value):
            raise LookupError("injected post-consume fill failure")

    executor.fills = RaisingFills()
    with pytest.raises(LookupError, match="post-consume") as raised:
        _submit_gateway_reduction_once(
            harness.authority,
            dispatch,
            submitter=harness.submitter_identity,
            order=original,
            pre_position_quantity=Decimal("1"),
        )

    capability = None
    retained_records = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == "_submit_gateway_reduction_once":
            capability = traceback.tb_frame.f_locals.get("capability")
        record = traceback.tb_frame.f_locals.get("record")
        if record is not None and hasattr(record, "consumed"):
            retained_records.append(record)
        traceback = traceback.tb_next
    assert capability is not None
    for record in retained_records:
        with pytest.raises(AttributeError):
            record.consumed = False
        replace(record, consumed=False)
    with pytest.raises(PaperExecutionCapabilityError):
        copy.copy(capability)
    with pytest.raises(PaperExecutionCapabilityError):
        copy.deepcopy(capability)
    with pytest.raises(PaperExecutionCapabilityError):
        pickle.dumps(capability)
    with pytest.raises(TypeError):
        replace(capability)

    replay = PaperExecutor._place_simple_order(executor, original, _capability=capability)

    assert replay.ok is False
    assert "already consumed" in replay.message
    assert executor.fills == {}


def test_reduction_consume_uses_closure_captured_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = PaperExecutor()
    harness = bind_gateway_reduction_harness(executor, "default")
    original = Order("AAPL", 1, "SELL", Decimal("100.0000"))
    dispatch = harness.issue(original, pre_position_quantity=Decimal("1"))

    def forged_fingerprint(_order):
        pytest.fail("mutable module fingerprint must not replace the captured primitive")

    monkeypatch.setattr(capability_module, "_fingerprint_order", forged_fingerprint)
    result = _submit_gateway_reduction_once(
        harness.authority,
        dispatch,
        submitter=harness.submitter_identity,
        order=original,
        pre_position_quantity=Decimal("1"),
    )

    assert result.ok is True
    assert len(executor.fills) == 1


@pytest.mark.parametrize(
    ("side", "pre_position"),
    [("SELL", Decimal("1")), ("BUY_TO_COVER", Decimal("-1"))],
)
def test_direct_reduction_helpers_reject_unregistered_authority(
    side: str,
    pre_position: Decimal,
) -> None:
    executor = PaperExecutor()
    order = Order("AAPL", 1, side, Decimal("100.0000"))

    with pytest.raises(PaperExecutionCapabilityError, match="final allocation"):
        _issue_gateway_reduction_terminal_dispatch(
            object(),
            submitter=object(),
            executor=executor,
            coordinator=object(),
            final_allocation=object(),
            descriptor=object(),
            contract=object(),
            order=order,
            pre_position_quantity=pre_position,
        )
    with pytest.raises(PaperExecutionCapabilityError, match="dispatch is invalid"):
        _submit_gateway_reduction_once(
            object(),
            object(),
            submitter=object(),
            order=order,
            pre_position_quantity=pre_position,
        )

    assert executor.fills == {}


def test_reduction_terminal_dispatch_mismatch_burns_before_replay() -> None:
    executor = PaperExecutor()
    harness = bind_gateway_reduction_harness(executor, "default")
    admitted = Order("AAPL", 1, "SELL", Decimal("100.0000"))
    dispatch = harness.issue(
        admitted,
        pre_position_quantity=Decimal("1"),
    )

    with pytest.raises(PaperExecutionCapabilityError, match="does not match attempt"):
        _submit_gateway_reduction_once(
            harness.authority,
            dispatch,
            submitter=harness.submitter_identity,
            order=Order("MSFT", 1, "SELL", Decimal("100.0000")),
            pre_position_quantity=Decimal("1"),
        )
    with pytest.raises(PaperExecutionCapabilityError, match="already consumed"):
        _submit_gateway_reduction_once(
            harness.authority,
            dispatch,
            submitter=harness.submitter_identity,
            order=admitted,
            pre_position_quantity=Decimal("1"),
        )

    assert executor.fills == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL", "BUY_TO_COVER"])
async def test_order_manager_cannot_bypass_terminal_capability(side: str) -> None:
    executor = PaperExecutor()
    manager = OrderManager(max_retries=1)

    result = await manager.place_order("AAPL", 1, side, executor=executor)

    assert result.status is OrderStatus.ERROR
    assert "submission capability" in (result.error_message or "")
    assert executor.fills == {}


class _FakeOrderExecutor:
    def place_order(self, _order):
        return SimpleNamespace(ok=True, message="forged acceptance", order_id="fake")


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL", "BUY_TO_COVER"])
@pytest.mark.parametrize("executor", [None, object(), _FakeOrderExecutor()])
async def test_order_manager_rejects_missing_or_alternate_executor(side, executor) -> None:
    manager = OrderManager(max_retries=1)

    result = await manager.place_order("AAPL", 1, side, executor=executor)

    assert result.status is OrderStatus.ERROR
    assert "exact contained PaperExecutor" in (result.error_message or "")
    assert result.id not in manager.pending_orders
    assert result.id not in manager.active_orders
    assert result.id not in manager.monitoring_tasks


@pytest.mark.asyncio
async def test_smart_executor_cannot_be_constructed_or_reenabled() -> None:
    with pytest.raises(ValueError, match="smart execution"):
        PaperExecutor(use_smart_execution=True)
    with pytest.raises(ValueError, match="smart execution"):
        PaperExecutor(smart_executor=object())

    executor = PaperExecutor()
    executor.use_smart_execution = True
    order = Order("AAPL", 1, "BUY", 100.0)

    assert executor.place_order(order).ok is False
    assert (await executor.place_order_async(order)).ok is False
    assert executor._place_smart_order(order).ok is False
    assert (await executor._execute_smart_order_async(order)).ok is False
    assert (await executor.execute_order("AAPL", "BUY", 1, "market"))["executed_quantity"] == 0
    assert executor.fills == {}


@pytest.mark.parametrize(
    ("component", "construct"),
    [
        ("TradingEngine", lambda: TradingEngine(Config())),
        ("SignalGenerator", SignalGenerator),
        ("TradeExecutor", lambda: TradeExecutor(None, None, None, {})),
    ],
)
def test_alternate_entry_engines_are_quarantined(component: str, construct) -> None:
    with pytest.raises(GateAContainmentError, match=component):
        construct()
