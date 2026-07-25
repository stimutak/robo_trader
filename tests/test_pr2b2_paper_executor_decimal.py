"""Exact-price behavior at the narrow PaperExecutor submission sink."""

from __future__ import annotations

from decimal import Decimal

import pytest

from robo_trader.execution import ExecutionResult, Order, PaperExecutor


def _order(*, side: str = "BUY", price: float | Decimal = Decimal("123.45")) -> Order:
    return Order(
        symbol="AAPL",
        quantity=3,
        side=side,
        price=price,
        order_ref="decimal-paper-test",
    )


def test_public_place_order_preserves_exact_decimal_into_simple_sink(monkeypatch) -> None:
    executor = PaperExecutor(slippage_bps=0)
    captured: list[Order] = []

    monkeypatch.setattr("robo_trader.execution.os.path.exists", lambda _path: False)

    def simple_sink(order: Order) -> ExecutionResult:
        captured.append(order)
        return ExecutionResult(True, "captured", fill_price=float(order.price))

    monkeypatch.setattr(executor, "_place_simple_order", simple_sink)
    exact_price = Decimal("123.4500")
    order = _order(price=exact_price)

    result = executor.place_order(order)

    assert result.ok is True
    assert captured == [order]
    assert type(captured[0].price) is Decimal
    assert captured[0].price is exact_price
    assert captured[0].price.as_tuple() == exact_price.as_tuple()


@pytest.mark.parametrize("slippage_bps", [0.0, 12.5])
def test_common_decimal_price_has_deterministic_finite_fill(slippage_bps: float) -> None:
    exact_price = Decimal("123.45")
    expected = float(exact_price + (exact_price * Decimal(str(slippage_bps)) / Decimal("10000")))

    first = PaperExecutor(slippage_bps=slippage_bps)._place_simple_order(_order(price=exact_price))
    second = PaperExecutor(slippage_bps=slippage_bps)._place_simple_order(_order(price=exact_price))

    assert first.ok is True
    assert second.ok is True
    assert first.fill_price == expected
    assert second.fill_price == expected


@pytest.mark.parametrize(
    ("side", "direction"),
    [("SELL", -1), ("BUY_TO_COVER", 1)],
)
def test_decimal_slippage_direction_for_reducing_sides(side: str, direction: int) -> None:
    price = Decimal("123.45")
    slippage_bps = Decimal("25")
    slip = price * slippage_bps / Decimal("10000")
    expected = float(price + (slip * direction))

    result = PaperExecutor(slippage_bps=float(slippage_bps))._place_simple_order(
        _order(side=side, price=price)
    )

    assert result.ok is True
    assert result.fill_price == expected
    assert result.fill_price < float(price) if side == "SELL" else result.fill_price > float(price)


@pytest.mark.parametrize(
    ("side", "slippage_bps"),
    [
        ("BUY", float("nan")),
        ("BUY", float("inf")),
        ("SELL", 20_000.0),
        ("BUY", -20_000.0),
    ],
)
def test_decimal_nonfinite_or_nonpositive_fill_fails_closed(
    side: str,
    slippage_bps: float,
) -> None:
    executor = PaperExecutor(slippage_bps=slippage_bps)

    result = executor._place_simple_order(_order(side=side))

    assert result.ok is False
    assert result.fill_price is None
    assert result.message == "Invalid paper execution fill"
    assert executor.fills == {}


@pytest.mark.parametrize(
    ("side", "expected"),
    [("BUY", 100.1), ("SELL", 99.9), ("BUY_TO_COVER", 100.1), ("SELL_SHORT", 99.9)],
)
def test_legacy_float_price_behavior_is_unchanged(side: str, expected: float) -> None:
    order = _order(side=side, price=100.0)
    executor = PaperExecutor(slippage_bps=10.0)

    result = executor._place_simple_order(order)

    assert result.ok is True
    assert result.fill_price == pytest.approx(expected)
    assert type(order.price) is float
    assert next(iter(executor.fills.values()))[1] is order


def test_public_kill_switch_gate_blocks_while_private_sink_remains_narrowly_callable(
    monkeypatch,
) -> None:
    probes: list[str] = []

    def kill_switch_exists(path: str) -> bool:
        probes.append(path)
        return path == "data/kill_switch.lock"

    monkeypatch.setattr("robo_trader.execution.os.path.exists", kill_switch_exists)
    executor = PaperExecutor(slippage_bps=0)
    order = _order(side="SELL")

    public_result = executor.place_order(order)

    assert public_result.ok is False
    assert public_result.message == "Kill switch active"
    assert executor.fills == {}
    assert probes == ["data/kill_switch.lock"]

    private_result = executor._place_simple_order(order)

    assert private_result.ok is True
    assert private_result.fill_price == 123.45
    assert len(executor.fills) == 1
    assert probes == ["data/kill_switch.lock"]
