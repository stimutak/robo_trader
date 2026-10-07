"""Settlement projections must not inherit caller rounding policy."""

from decimal import Decimal, Inexact, Rounded, localcontext

import pytest

from robo_trader.paper_terminal_settlement import PaperAccountSettlementState
from robo_trader.safety.models import OrderSide


@pytest.mark.parametrize("side", [OrderSide.SELL, OrderSide.BUY_TO_COVER])
@pytest.mark.parametrize("commission", [0, 12345, -12345])
@pytest.mark.parametrize("trapped", [False, True])
def test_reduction_projection_preserves_exact_commission_and_quantity(side, commission, trapped):
    state = PaperAccountSettlementState(
        "default",
        Decimal("100000"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        "2026-09-17",
        Decimal("100"),
        Decimal("100"),
        None,
    )
    request = dict(
        side=side,
        filled_quantity=Decimal("1"),
        fill_price=Decimal("101") if side is OrderSide.SELL else Decimal("99"),
        protective_mark_price=Decimal("101") if side is OrderSide.SELL else Decimal("99"),
        pre_position_quantity=Decimal("123456") if side is OrderSide.SELL else Decimal("-123456"),
        commission_minor=commission,
    )
    with localcontext() as context:
        context.prec = 64
        expected = state.post_values(**request)
    with localcontext() as context:
        context.prec = 4
        context.traps[Inexact] = trapped
        context.traps[Rounded] = trapped
        assert state.post_values(**request) == expected
