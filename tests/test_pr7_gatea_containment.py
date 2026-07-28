"""Machine-enforced containment for the narrow Gate-A paper profile."""

from __future__ import annotations

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
from robo_trader.runner.signal_generator import SignalGenerator
from robo_trader.runner.trade_executor import TradeExecutor


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
    ("side", "source", "take_profit", "expected_fragment"),
    [
        ("BUY", "pairs_trading", None, "not the baseline"),
        ("BUY", "stat_arbitrage", None, "not the baseline"),
        ("BUY", "ai_discovery", None, "not the baseline"),
        ("BUY", "ml_selector", None, "not the baseline"),
        ("BUY", "", None, "not the baseline"),
        ("SELL_SHORT", "baseline_sma", None, "blocks new short exposure"),
        ("BUY", "baseline_sma", 110.0, "take-profit"),
    ],
)
def test_disabled_paths_cannot_create_entry_intent(
    side: str,
    source: str,
    take_profit: float | None,
    expected_fragment: str,
) -> None:
    admitted, reason = validate_gate_a_order(
        side=side,
        intent_source=source,
        take_profit=take_profit,
    )

    assert admitted is False
    assert expected_fragment in reason


@pytest.mark.parametrize("side", ["SELL", "BUY_TO_COVER"])
def test_semantic_reductions_remain_admitted_without_entry_source(side: str) -> None:
    admitted, reason = validate_gate_a_order(
        side=side,
        intent_source="",
        take_profit=None,
    )

    assert admitted is True
    assert reason == "Gate-A semantic reduction"


def test_paper_executor_rechecks_containment_at_submission(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_module,
        "os",
        SimpleNamespace(path=SimpleNamespace(exists=lambda _path: False)),
    )
    executor = PaperExecutor(slippage_bps=0)

    rejected = executor.place_order(Order("AAPL", 1, "BUY", 100.0, intent_source="pairs_trading"))
    short = executor.place_order(
        Order("AAPL", 1, "SELL_SHORT", 100.0, intent_source="baseline_sma")
    )
    buy = executor.place_order(Order("AAPL", 1, "BUY", 100.0, intent_source="baseline_sma"))
    sell = executor.place_order(Order("AAPL", 1, "SELL", 100.0))
    cover = executor.place_order(Order("AAPL", 1, "BUY_TO_COVER", 100.0))

    assert rejected.ok is False
    assert short.ok is False
    assert buy.ok is True
    assert sell.ok is True
    assert cover.ok is True
    assert len(executor.fills) == 3


@pytest.mark.asyncio
async def test_smart_executor_cannot_be_constructed_or_reenabled() -> None:
    with pytest.raises(ValueError, match="smart execution"):
        PaperExecutor(use_smart_execution=True)
    with pytest.raises(ValueError, match="smart execution"):
        PaperExecutor(smart_executor=object())

    executor = PaperExecutor()
    executor.use_smart_execution = True
    order = Order("AAPL", 1, "BUY", 100.0, intent_source="baseline_sma")

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
