"""Exact local-paper cost estimates share the executor's rounding policy."""

from decimal import ROUND_DOWN, Decimal, localcontext

import pytest

from robo_trader import paper_execution_cost as costs


@pytest.mark.parametrize(
    "reference,slippage,side,expected",
    [
        ("12.3456", "0.04131", "BUY", "12.3457"),
        ("100", "10", "BUY_TO_COVER", "100.1000"),
        ("100", "10", "SELL", "99.9000"),
        ("1.23445", "0", "BUY", "1.2344"),
        ("1.23455", "0", "BUY", "1.2346"),
    ],
)
def test_cost_rounds_exact_rational_fill_once(reference, slippage, side, expected):
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        actual = costs.exact_paper_fill_price(Decimal(reference), Decimal(slippage), side)
    assert actual == Decimal(expected)


def test_entry_cost_never_sizes_below_reference_even_when_tick_rounds_down():
    cost = costs.paper_entry_cost(Decimal("1.23445"), Decimal("0"))
    assert cost.modeled_fill_price_usd == Decimal("1.2344")
    assert cost.price_ceiling_usd == Decimal("1.23445")
    assert cost.commission_minor == 0


@pytest.mark.parametrize("slippage", ["-1", "10000", "NaN", "Infinity"])
def test_cost_rejects_invalid_slippage(slippage):
    with pytest.raises(ValueError):
        costs.paper_entry_cost(Decimal("100"), Decimal(slippage))
