"""Runner policy selects one portfolio without mutating shared configuration."""

from decimal import Decimal

import pytest

from robo_trader.config import Config
from robo_trader.runner_async import AsyncRunner
from tests.risk.test_entry_runtime_policy import POLICY


def runner_for(config, portfolio_id="alpha", order=None, daily=None):
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = config
    runner.portfolio_id = portfolio_id
    runner.max_order_notional = order
    runner.max_daily_notional = daily
    runner.max_correlation = 0.7
    return runner


def config_for(portfolios=None):
    return Config(
        entry_risk_policy=POLICY,
        portfolio_configs=portfolios
        or [
            {"id": "alpha", "active": True, "max_position_pct": 0.01, "max_open_positions": 3},
            {"id": "beta", "active": True, "max_position_pct": 0.015, "max_open_positions": 7},
        ],
    )


def test_runner_resolves_portfolio_and_instance_limits_without_mutating_shared_config():
    config = config_for()
    before = config.model_dump()
    first = runner_for(config, order=Decimal("1200.01"), daily=5000)
    a = first._current_entry_risk_limits()
    b = runner_for(config, "beta")._current_entry_risk_limits()
    assert a.max_position_fraction == Decimal("0.01")
    assert a.max_open_positions == 3
    assert a.max_order_notional_usd == Decimal("1200.01")
    assert a.max_daily_notional_usd == Decimal("5000")
    assert b.max_position_fraction == Decimal("0.015")
    assert b.max_open_positions == 7
    assert b.max_order_notional_usd == Decimal("10000")
    assert config.model_dump() == before
    first.max_order_notional = 900
    assert first._current_entry_risk_limits().max_order_notional_usd == Decimal("900")


@pytest.mark.parametrize(
    "portfolios",
    [[], [{"id": "beta"}], [{"id": "alpha"}, {"id": "alpha"}], [{"id": "alpha", "active": False}]],
)
def test_runner_rejects_missing_ambiguous_or_inactive_portfolio(portfolios):
    config = config_for()
    config.portfolio_configs = portfolios
    with pytest.raises(ValueError, match="portfolio"):
        runner_for(config)._current_entry_risk_limits()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_order_notional", 0),
        ("max_daily_notional", 0),
        ("max_order_notional", False),
        ("max_daily_notional", float("nan")),
    ],
)
def test_runner_rejects_invalid_overrides_instead_of_falling_back(field, value):
    runner = runner_for(config_for())
    setattr(runner, field, value)
    with pytest.raises(ValueError):
        runner._current_entry_risk_limits()


@pytest.mark.parametrize(
    "field,value",
    [("max_position_pct", 0.03), ("max_open_positions", True), ("max_open_positions", 0)],
)
def test_runner_rejects_invalid_portfolio_entry_limits(field, value):
    config = config_for()
    config.portfolio_configs[0][field] = value
    with pytest.raises(ValueError):
        runner_for(config)._current_entry_risk_limits()


def test_none_portfolio_overrides_inherit_global_limits():
    config = config_for([{"id": "alpha", "max_position_pct": None, "max_open_positions": None}])
    limits = runner_for(config)._current_entry_risk_limits()
    assert limits.max_position_fraction == Decimal("0.02")
    assert limits.max_open_positions == 20


def test_runner_correlation_can_tighten_configured_limit():
    runner = runner_for(config_for())
    runner.max_correlation = 0.4
    assert runner._current_entry_risk_limits().max_absolute_correlation == Decimal("0.4")
    runner.max_correlation = 0.9
    assert runner._current_entry_risk_limits().max_absolute_correlation == Decimal("0.7")
    runner.max_correlation = False
    with pytest.raises(ValueError):
        runner._current_entry_risk_limits()
