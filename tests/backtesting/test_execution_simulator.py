"""Deterministic and fail-closed tests for the backtest execution simulator."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from robo_trader.backtesting.execution_simulator import (
    ExecutionSimulator,
    MarketImpactModel,
    SimulatedOrder,
)

EVENT_TIME = pd.Timestamp("2026-07-27 09:30:00")


def _bars(
    *,
    index: pd.DatetimeIndex | None = None,
    open_price: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 101.0,
    volume: float = 1_000.0,
    **extra: float,
) -> pd.DataFrame:
    values = {
        "open": [open_price],
        "high": [high],
        "low": [low],
        "close": [close],
        "volume": [volume],
    }
    values.update({name: [value] for name, value in extra.items()})
    selected_index = index if index is not None else pd.DatetimeIndex([EVENT_TIME])
    return pd.DataFrame(values, index=selected_index)


def _simulator(**overrides: object) -> ExecutionSimulator:
    options: dict[str, object] = {
        "spread_model": "fixed",
        "market_impact_model": MarketImpactModel(0.0, 0.0),
        "slippage_factor": 0.0,
        "max_volume_participation": 1.0,
        "random_seed": 17,
    }
    options.update(overrides)
    return ExecutionSimulator(**options)


def _execute(simulator: ExecutionSimulator, **overrides: object) -> SimulatedOrder:
    options: dict[str, object] = {
        "symbol": "AAPL",
        "quantity": 10,
        "side": "buy",
        "order_type": "market",
        "price_data": _bars(),
        "timestamp": EVENT_TIME,
    }
    options.update(overrides)
    return simulator.simulate_execution(**options)


def test_owned_rng_is_repeatable_and_reset_per_run() -> None:
    simulator = _simulator(slippage_factor=0.01, random_seed=314)

    first_run = [_execute(simulator).fill_price for _ in range(3)]
    simulator.reset()
    second_run = [_execute(simulator).fill_price for _ in range(3)]

    assert first_run == second_run
    assert len(set(first_run)) == 3


def test_reset_can_select_a_new_remembered_seed() -> None:
    simulator = _simulator(slippage_factor=0.01, random_seed=1)
    seed_one_fill = _execute(simulator).fill_price

    simulator.reset(2)
    seed_two_fill = _execute(simulator).fill_price
    simulator.reset_random_state()

    assert seed_two_fill != seed_one_fill
    assert _execute(simulator).fill_price == seed_two_fill


def test_simulation_does_not_consume_numpy_global_rng() -> None:
    np.random.seed(12345)
    expected = np.random.random()
    np.random.seed(12345)

    _execute(_simulator(slippage_factor=0.01))

    assert np.random.random() == expected


def test_exact_timestamp_is_required_without_prior_or_future_fallback() -> None:
    index = pd.DatetimeIndex([EVENT_TIME, EVENT_TIME + pd.Timedelta(minutes=2)])
    bars = pd.concat([_bars(index=pd.DatetimeIndex([event])) for event in index], ignore_index=True)
    bars.index = index

    result = _execute(
        _simulator(),
        price_data=bars,
        timestamp=EVENT_TIME + pd.Timedelta(minutes=1),
    )

    assert not result.filled
    assert result.fill_price == 0.0
    assert result.requested_price == 0.0
    assert result.filled_quantity == 0
    assert result.remaining_quantity == 10


def test_equivalent_timezone_aware_timestamp_matches_exact_event() -> None:
    utc_time = pd.Timestamp("2026-07-27 13:30:00", tz="UTC")
    eastern_time = pd.Timestamp("2026-07-27 09:30:00", tz="America/New_York")
    result = _execute(
        _simulator(),
        price_data=_bars(index=pd.DatetimeIndex([utc_time])),
        timestamp=eastern_time,
    )

    assert result.filled


def test_timezone_awareness_mismatch_is_rejected() -> None:
    aware_index = pd.DatetimeIndex([EVENT_TIME.tz_localize("UTC")])
    with pytest.raises(ValueError, match="matching timezone awareness"):
        _execute(_simulator(), price_data=_bars(index=aware_index), timestamp=EVENT_TIME)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"symbol": ""}, "symbol"),
        ({"quantity": 0}, "quantity"),
        ({"quantity": -1}, "quantity"),
        ({"quantity": 1.5}, "quantity"),
        ({"quantity": True}, "quantity"),
        ({"side": "BUY"}, "side"),
        ({"order_type": "trailing"}, "order_type"),
        ({"timestamp": "2026-07-27"}, "timestamp"),
    ],
)
def test_invalid_order_fields_are_rejected(override: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _execute(_simulator(), **override)


@pytest.mark.parametrize("order_type", ["limit", "stop"])
def test_conditional_order_requires_its_trigger_price(order_type: str) -> None:
    expected = "limit_price" if order_type == "limit" else "stop_price"
    with pytest.raises(ValueError, match=expected):
        _execute(_simulator(), order_type=order_type)


@pytest.mark.parametrize("bad_price", [0.0, -1.0, float("nan"), float("inf")])
def test_nonfinite_or_nonpositive_trigger_price_is_rejected(bad_price: float) -> None:
    with pytest.raises(ValueError, match="limit_price"):
        _execute(_simulator(), order_type="limit", limit_price=bad_price)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns="volume"),
        lambda frame: frame.assign(open=np.nan),
        lambda frame: frame.assign(high=np.inf),
        lambda frame: frame.assign(low=0.0),
        lambda frame: frame.assign(close=-1.0),
        lambda frame: frame.assign(volume=-1.0),
        lambda frame: frame.assign(high=99.0),
        lambda frame: frame.assign(low=102.0),
        lambda frame: frame.assign(bid=100.0),
        lambda frame: frame.assign(bid=101.0, ask=100.0),
        lambda frame: frame.assign(volatility=-0.1),
    ],
)
def test_malformed_ohlcv_or_quote_data_is_rejected(mutator: object) -> None:
    malformed = mutator(_bars())
    with pytest.raises(ValueError):
        _execute(_simulator(), price_data=malformed)


def test_non_datetime_duplicate_and_unsorted_indexes_are_rejected() -> None:
    ordinary_index = _bars().reset_index(drop=True)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        _execute(_simulator(), price_data=ordinary_index)

    duplicate = pd.concat([_bars(), _bars()])
    with pytest.raises(ValueError, match="unique"):
        _execute(_simulator(), price_data=duplicate)

    unsorted = pd.concat(
        [
            _bars(index=pd.DatetimeIndex([EVENT_TIME + pd.Timedelta(minutes=1)])),
            _bars(),
        ]
    )
    with pytest.raises(ValueError, match="sorted"):
        _execute(_simulator(), price_data=unsorted)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spread_model": "unknown"},
        {"commission_per_share": float("nan")},
        {"min_commission": -1.0},
        {"slippage_factor": float("inf")},
        {"random_seed": -1},
        {"random_seed": True},
        {"max_volume_participation": 0.0},
        {"max_volume_participation": 1.1},
        {"use_real_spreads": "yes"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExecutionSimulator(**kwargs)


def test_market_orders_execute_at_opening_touch_for_each_side() -> None:
    simulator = _simulator()

    buy = _execute(simulator, side="buy")
    sell = _execute(simulator, side="sell")

    assert buy.fill_price == pytest.approx(100.005)
    assert sell.fill_price == pytest.approx(99.995)


def test_real_quotes_use_the_actual_ask_and_bid_not_a_synthetic_midpoint() -> None:
    bars = _bars(bid=99.80, ask=100.30)
    simulator = _simulator(use_real_spreads=True)

    buy = _execute(simulator, price_data=bars, side="buy")
    sell = _execute(simulator, price_data=bars, side="sell")

    assert buy.fill_price == pytest.approx(100.30)
    assert sell.fill_price == pytest.approx(99.80)


@pytest.mark.parametrize(
    ("side", "limit_price", "expected"),
    [
        ("buy", 101.0, 100.005),
        ("sell", 99.0, 99.995),
        ("buy", 98.0, 98.0),
        ("sell", 103.0, 103.0),
    ],
)
def test_limit_orders_follow_opening_then_intrabar_policy(
    side: str, limit_price: float, expected: float
) -> None:
    result = _execute(_simulator(), side=side, order_type="limit", limit_price=limit_price)

    assert result.filled
    assert result.fill_price == pytest.approx(expected)


@pytest.mark.parametrize(
    ("side", "limit_price"),
    [("buy", 94.0), ("sell", 106.0)],
)
def test_unreached_limit_does_not_fill(side: str, limit_price: float) -> None:
    result = _execute(_simulator(), side=side, order_type="limit", limit_price=limit_price)

    assert not result.filled


@pytest.mark.parametrize(
    ("side", "bars", "stop_price", "expected"),
    [
        ("buy", _bars(open_price=104.0, high=106.0), 102.0, 104.005),
        ("sell", _bars(open_price=96.0, low=94.0), 98.0, 95.995),
        ("buy", _bars(), 103.0, 103.005),
        ("sell", _bars(), 97.0, 96.995),
    ],
)
def test_stop_orders_follow_gap_then_intrabar_policy(
    side: str, bars: pd.DataFrame, stop_price: float, expected: float
) -> None:
    result = _execute(
        _simulator(),
        side=side,
        order_type="stop",
        stop_price=stop_price,
        price_data=bars,
    )

    assert result.filled
    assert result.fill_price == pytest.approx(expected)


def test_unreached_stop_does_not_fill() -> None:
    result = _execute(_simulator(), order_type="stop", stop_price=106.0)

    assert not result.filled


def test_volume_participation_produces_genuine_partial_fill() -> None:
    result = _execute(
        _simulator(max_volume_participation=0.25),
        quantity=100,
        price_data=_bars(volume=200),
    )

    assert result.filled
    assert not result.fully_filled
    assert result.quantity == 100
    assert result.partial_fill == 50
    assert result.filled_quantity == 50
    assert result.remaining_quantity == 50
    assert result.execution_cost.commission == pytest.approx(1.0)


def test_zero_or_subshare_capacity_returns_unfilled_order() -> None:
    result = _execute(
        _simulator(max_volume_participation=0.10),
        quantity=5,
        price_data=_bars(volume=9),
    )

    assert not result.filled
    assert result.filled_quantity == 0
    assert result.remaining_quantity == 5


def test_commission_is_separate_from_fill_price_and_total_cost_is_exact() -> None:
    simulator = _simulator(commission_per_share=5.0, min_commission=50.0)
    result = _execute(simulator, quantity=10)

    assert result.fill_price == pytest.approx(100.005)
    assert result.execution_cost.commission == pytest.approx(50.0)
    assert result.execution_cost.total_cost == pytest.approx(50.05)
    assert result.execution_cost.total_cost == pytest.approx(
        (
            result.execution_cost.spread_cost
            + result.execution_cost.market_impact
            + result.execution_cost.slippage
        )
        * result.filled_quantity
        + result.execution_cost.commission
    )


def test_limit_price_caps_adverse_impact_and_slippage() -> None:
    simulator = _simulator(
        market_impact_model=MarketImpactModel(1.0, 1.0),
        slippage_factor=1.0,
    )
    result = _execute(
        simulator,
        order_type="limit",
        limit_price=100.01,
        price_data=_bars(volume=10_000, volatility=0.5),
    )

    assert result.fill_price <= 100.01
    assert result.fill_price == pytest.approx(100.01)


def test_execution_analytics_use_filled_quantity_and_precomputed_total_once() -> None:
    simulator = _simulator(max_volume_participation=0.25)
    partial = _execute(simulator, quantity=100, price_data=_bars(volume=200))
    unfilled = _execute(simulator, quantity=10, price_data=_bars(volume=0))

    analytics = simulator.get_execution_analytics([partial, unfilled])

    assert analytics["fill_rate"] == pytest.approx(0.5)
    assert analytics["total_spread_cost"] == pytest.approx(partial.execution_cost.spread_cost * 50)
    assert analytics["total_commission"] == partial.execution_cost.commission
    assert analytics["total_execution_cost"] == partial.execution_cost.total_cost
    assert simulator.calculate_portfolio_turnover([partial]) == pytest.approx(
        (50 * partial.fill_price) / (partial.fill_price * 10_000)
    )


def test_legacy_public_fields_and_constructor_remain_available() -> None:
    result = _execute(_simulator())

    assert result.symbol == "AAPL"
    assert result.quantity == 10
    assert result.side == "buy"
    assert result.order_type == "market"
    assert result.timestamp == EVENT_TIME
    assert result.requested_price == 100.0
    assert result.execution_cost.fill_price == result.fill_price
    assert result.partial_fill == 10
