"""Advanced-risk restart and signed short-accounting regression tests."""

from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.config import RuntimeContract
from robo_trader.paper_runtime_settlement import bounded_float_projection_matches
from robo_trader.portfolio import Portfolio
from robo_trader.risk.advanced_risk import AdvancedRiskManager
from robo_trader.runner_async import AsyncRunner, UnprotectedExistingPositionsError

TEST_RUNTIME_CONTRACT = RuntimeContract(
    environment="test",
    execution_mode="paper",
    execution_source="paper_simulator",
    ibkr_host="127.0.0.1",
    ibkr_port=4002,
    ibkr_readonly=True,
    database_path="/tmp/robotrader-advanced-risk-test.db",
    account_alias="***PER",
    account_type="paper",
    model_artifact_set="test",
    build_id="test-build",
    state_namespace="paper",
    safety_account_scope="acct_v1_" + ("0123456789abcdef" * 4),
    safety_execution_domain_scope="paper-simulator-v1",
)


def _risk_manager() -> AdvancedRiskManager:
    return AdvancedRiskManager(
        {"starting_capital": 100_000},
        enable_kelly=False,
        enable_correlation_limits=False,
        enable_kill_switch=True,
    )


def test_sell_short_creates_signed_risk_position() -> None:
    risk = _risk_manager()

    realized = risk.update_position("TSLA", 7, 250.0, "SELL_SHORT")

    assert realized == 0.0
    assert risk.positions["TSLA"]["quantity"] == -7
    assert risk.positions["TSLA"]["avg_price"] == 250.0
    assert risk.positions["TSLA"]["value"] == -1750.0
    assert risk.kill_switch.position_quantity["TSLA"] == -7


def test_partial_cover_preserves_short_cost_basis_and_records_profit() -> None:
    risk = _risk_manager()
    risk.seed_realized_pnl(total_pnl=125.0, daily_pnl=25.0)
    risk.seed_position("TSLA", -10, 250.0)

    realized = risk.update_position("TSLA", 4, 240.0, "BUY_TO_COVER")

    assert realized == 40.0
    assert risk.positions["TSLA"]["quantity"] == -6
    assert risk.positions["TSLA"]["avg_price"] == 250.0
    assert risk.positions["TSLA"]["value"] == -1440.0
    assert risk.total_pnl == 165.0
    assert risk.daily_pnl == 65.0
    assert risk.kill_switch.position_entry["TSLA"][0] == 250.0
    assert risk.kill_switch.position_quantity["TSLA"] == -6


def test_full_cover_removes_short_and_records_loss() -> None:
    risk = _risk_manager()
    risk.seed_position("TSLA", -5, 250.0)

    realized = risk.update_position("TSLA", 5, 255.0, "BUY_TO_COVER")

    assert realized == -25.0
    assert "TSLA" not in risk.positions
    assert "TSLA" not in risk.kill_switch.position_entry
    assert "TSLA" not in risk.kill_switch.position_quantity
    assert risk.total_pnl == -25.0
    assert risk.daily_pnl == -25.0


def test_float_projection_comparison_is_explicitly_bounded() -> None:
    assert bounded_float_projection_matches(
        Decimal("0.30000000000000004"),
        Decimal("0.3"),
    )
    assert not bounded_float_projection_matches(
        Decimal("0.30000001"),
        Decimal("0.3"),
    )


@pytest.mark.asyncio
async def test_restart_seeds_long_short_cost_basis_and_account_pnl() -> None:
    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {}
    runner.cfg = SimpleNamespace(runtime_contract=TEST_RUNTIME_CONTRACT)
    runner.db = SimpleNamespace(
        get_account_info=AsyncMock(
            return_value={
                "cash": 1.0,
                "realized_pnl": 0.0,
                "daily_pnl": 0.0,
                "cash_exact": Decimal("50000"),
                "realized_pnl_exact": Decimal("125.5"),
                "daily_pnl_exact": Decimal("-7.25"),
                "daily_pnl_baseline_exact": Decimal("18.75"),
                "daily_pnl_date_exact": datetime.utcnow().date(),
                "source_settlement_id": "pset-" + ("7" * 32),
                "bootstrap_lineage_valid": True,
            }
        ),
        get_positions=AsyncMock(
            return_value=[
                {
                    "symbol": "AAPL",
                    "quantity": 3,
                    "avg_cost": 190.25,
                    "market_price_exact": Decimal("191.25"),
                    "bootstrap_lineage_valid": True,
                },
                {
                    "symbol": "TSLA",
                    "quantity": -4,
                    "avg_cost": 251.75,
                    "market_price_exact": Decimal("250.75"),
                    "bootstrap_lineage_valid": True,
                },
            ]
        ),
        update_account=AsyncMock(),
    )
    runner.portfolio = Portfolio(100_000)
    runner.stop_loss_monitor = None
    runner.use_trailing_stop = False
    runner.stop_loss_percent = 0.02
    runner.trailing_stop_pct = 0.05
    runner.use_advanced_risk = True
    runner.advanced_risk = _risk_manager()
    runner.daily_pnl = 0.0
    runner.latest_prices = {}
    runner.latest_price_sources = {}
    runner.latest_price_times = {}
    runner._daily_pnl_date = None
    runner._starting_unrealized_today = 0.0
    runner._starting_unrealized_today_exact = Decimal("0")

    await runner.load_existing_positions()

    assert runner.portfolio.cash == Decimal("50000.0")
    assert runner.positions["AAPL"].quantity == 3
    assert runner.positions["TSLA"].quantity == -4
    assert runner.advanced_risk.positions["AAPL"]["quantity"] == 3
    assert runner.advanced_risk.positions["AAPL"]["avg_price"] == 190.25
    assert runner.advanced_risk.positions["TSLA"]["quantity"] == -4
    assert runner.advanced_risk.positions["TSLA"]["avg_price"] == 251.75
    assert runner.advanced_risk.total_pnl == 125.5
    assert runner.advanced_risk.daily_pnl == -7.25
    assert runner.daily_pnl == -7.25
    assert runner._starting_unrealized_today_exact == Decimal("18.75")
    assert runner._account_settlement_source_id == "pset-" + ("7" * 32)
    assert runner.advanced_risk.kill_switch.position_quantity == {"AAPL": 3, "TSLA": -4}


def _restart_runner(*, account_date, exact_mark):
    runner = object.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.positions = {}
    runner.cfg = SimpleNamespace(runtime_contract=TEST_RUNTIME_CONTRACT)
    runner.db = SimpleNamespace(
        get_account_info=AsyncMock(
            return_value={
                "cash_exact": Decimal("50000"),
                "realized_pnl_exact": Decimal("12.5"),
                "daily_pnl_exact": Decimal("-3"),
                "daily_pnl_baseline_exact": Decimal("15.5"),
                "daily_pnl_date_exact": account_date,
                "source_settlement_id": "pset-" + ("8" * 32),
                "bootstrap_lineage_valid": True,
            }
        ),
        get_positions=AsyncMock(
            return_value=[
                {
                    "symbol": "AAPL",
                    "quantity": 2,
                    "avg_cost": 100.0,
                    "market_price_exact": exact_mark,
                    "bootstrap_lineage_valid": True,
                }
            ]
        ),
        update_account=AsyncMock(),
    )
    runner.portfolio = Portfolio(100_000)
    runner.stop_loss_monitor = None
    runner.use_trailing_stop = False
    runner.stop_loss_percent = 0.02
    runner.trailing_stop_pct = 0.05
    runner.use_advanced_risk = True
    runner.advanced_risk = _risk_manager()
    runner.daily_pnl = 0.0
    runner.latest_prices = {}
    runner.latest_price_sources = {}
    runner.latest_price_times = {}
    runner._daily_pnl_date = None
    runner._starting_unrealized_today = 0.0
    runner._starting_unrealized_today_exact = Decimal("0")
    runner._account_settlement_source_id = None
    return runner


@pytest.mark.asyncio
async def test_restart_after_utc_midnight_resets_only_after_exact_mark_hydration() -> None:
    today = datetime.utcnow().date()
    runner = _restart_runner(
        account_date=today - timedelta(days=1),
        exact_mark=Decimal("105"),
    )

    await runner.load_existing_positions()

    assert runner.latest_prices == {"AAPL": 105.0}
    assert runner._daily_pnl_date == today
    assert runner._starting_unrealized_today_exact == Decimal("10")
    assert runner.daily_pnl == 12.5
    assert runner.advanced_risk.daily_pnl == 12.5
    runner.db.update_account.assert_awaited_once_with(
        cash=Decimal("50000"),
        equity=Decimal("50210.0"),
        daily_pnl=Decimal("12.5"),
        realized_pnl=Decimal("12.5"),
        unrealized_pnl=Decimal("10.0"),
        portfolio_id="default",
        daily_pnl_baseline=Decimal("10.0"),
        daily_pnl_date=today,
    )


@pytest.mark.asyncio
async def test_restart_refuses_new_day_reset_without_exact_position_mark() -> None:
    runner = _restart_runner(
        account_date=datetime.utcnow().date() - timedelta(days=1),
        exact_mark=None,
    )

    with pytest.raises(
        UnprotectedExistingPositionsError,
        match="position_load_failed",
    ):
        await runner.load_existing_positions()
