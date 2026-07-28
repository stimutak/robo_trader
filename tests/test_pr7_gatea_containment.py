"""Machine-enforced containment for the narrow Gate-A paper profile."""

from __future__ import annotations

import ast
import inspect
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import robo_trader.execution as execution_module
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
    _bind_paper_execution_authority,
)
from robo_trader.runner.signal_generator import SignalGenerator
from robo_trader.runner.trade_executor import TradeExecutor
from robo_trader.runner_async import run_continuous


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


def test_baseline_intent_and_terminal_capability_have_single_runtime_call_sites() -> None:
    package = Path(__file__).parents[1] / "robo_trader"
    issue_calls: list[Path] = []
    terminal_calls: list[Path] = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "issue_baseline_entry_intent":
                issue_calls.append(path)
            elif node.func.attr == "_submit_baseline_once":
                terminal_calls.append(path)

    assert issue_calls == [package / "runner_async.py"]
    assert terminal_calls == [package / "paper_reduction_gateway.py"]


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
    authority = _bind_paper_execution_authority(executor, "default")

    sell = authority._submit_reduction_once(
        Order("AAPL", 1, "SELL", Decimal("100.0000")),
        pre_position_quantity=Decimal("1"),
    )
    cover = authority._submit_reduction_once(
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
    authority = _bind_paper_execution_authority(executor, "default")

    with pytest.raises(PaperExecutionCapabilityError, match="exposure"):
        authority._submit_reduction_once(
            Order("AAPL", quantity, side, Decimal("100.0000")),
            pre_position_quantity=pre_position,
        )

    assert executor.fills == {}


def test_capability_substitution_burns_authority(monkeypatch) -> None:
    executor = PaperExecutor()
    authority = _bind_paper_execution_authority(executor, "default")
    captured = []

    def capture(_order, *, _capability):
        captured.append(_capability)
        return SimpleNamespace(ok=False)

    monkeypatch.setattr(executor, "_place_simple_order", capture)
    original = Order("AAPL", 1, "SELL", Decimal("100.0000"))
    authority._submit_reduction_once(
        original,
        pre_position_quantity=Decimal("1"),
    )
    capability = captured[0]

    substituted = PaperExecutor._place_simple_order(
        executor,
        Order("MSFT", 1, "SELL", Decimal("100.0000")),
        _capability=capability,
    )
    replay = PaperExecutor._place_simple_order(
        executor,
        original,
        _capability=capability,
    )

    assert substituted.ok is False
    assert "does not match" in substituted.message
    assert replay.ok is False
    assert "already consumed" in replay.message
    assert executor.fills == {}


@pytest.mark.asyncio
async def test_order_manager_cannot_bypass_terminal_capability() -> None:
    executor = PaperExecutor()
    manager = OrderManager(max_retries=1)

    result = await manager.place_order("AAPL", 1, "BUY", executor=executor)

    assert result.status is OrderStatus.ERROR
    assert "submission capability" in (result.error_message or "")
    assert executor.fills == {}


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
