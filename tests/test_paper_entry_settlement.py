"""Opening paper fills must agree with exact FIFO and cash accounting."""

from dataclasses import replace
from decimal import Decimal, Inexact, Rounded, localcontext

import pytest

from robo_trader.paper_entry_settlement import project_flat_paper_buy
from robo_trader.paper_terminal_settlement import PaperAccountSettlementState
from robo_trader.safety.models import TerminalOrderStatus, ValidationError


def state():
    return PaperAccountSettlementState(
        "portfolio-a",
        Decimal("100000"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        "2026-09-17",
        None,
        None,
        None,
    )


def inputs():
    return dict(
        account=state(),
        pre_position_quantity=Decimal("0"),
        requested_quantity=Decimal("2"),
        filled_quantity=Decimal("2"),
        fill_price=Decimal("100.25"),
        mark_price=Decimal("100"),
        commission_minor=0,
        terminal_status=TerminalOrderStatus.FILLED,
    )


def test_opening_buy_agrees_with_actual_fifo_projection():
    from robo_trader.accounting.fifo import FillSide
    from robo_trader.accounting.fifo_runtime import append_runtime_fill_in_transaction
    from tests.accounting.test_fifo_runtime_settlement import _connection, _evidence

    connection, effective = _connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        fifo = append_runtime_fill_in_transaction(
            connection,
            _evidence(
                1,
                side=FillSide.BUY,
                quantity="2",
                price="100.25",
                commission_minor=0,
                occurred_at=effective,
            ),
        )
        result = project_flat_paper_buy(**inputs())
        assert result.position_quantity == fifo.signed_quantity == Decimal("2")
        assert result.position_cost_basis == fifo.average_cost == Decimal("100.25")
        assert result.realized_pnl == fifo.total_realized_pnl == Decimal("0")
        assert result.cash == Decimal("99799.50")
        assert result.daily_pnl == Decimal("-0.50")
        assert result.position_mark_price == Decimal("100")
        connection.rollback()
        assert connection.execute("SELECT count(*) FROM fifo_fills").fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "status",
    [TerminalOrderStatus.REJECTED, TerminalOrderStatus.CANCELLED, TerminalOrderStatus.EXPIRED],
)
def test_zero_fill_preserves_account_and_creates_no_position(status):
    values = inputs()
    values.update(terminal_status=status, filled_quantity=Decimal("0"), fill_price=None)
    result = project_flat_paper_buy(**values)
    assert result.cash == values["account"].cash
    assert result.daily_pnl == values["account"].daily_pnl
    assert result.position_quantity == 0
    assert result.position_cost_basis is None
    assert result.position_mark_price is None


@pytest.mark.parametrize("mark,expected", [("99", "-2.50"), ("100.25", "0.00"), ("101", "1.50")])
def test_entry_mark_to_market_is_independent_of_context(mark, expected):
    values = inputs()
    values["mark_price"] = Decimal(mark)
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = context.traps[Rounded] = True
        result = project_flat_paper_buy(**values)
    assert result.cash == Decimal("99799.50")
    assert result.daily_pnl == Decimal(expected)


@pytest.mark.parametrize(
    "changes",
    [
        {"pre_position_quantity": Decimal("1")},
        {"pre_position_quantity": Decimal("-1")},
        {"requested_quantity": Decimal("1.5")},
        {"filled_quantity": Decimal("1")},
        {"filled_quantity": Decimal("0")},
        {"commission_minor": 1},
        {"commission_minor": -1},
        {"commission_minor": False},
        {"terminal_status": "FILLED"},
        {"fill_price": Decimal("NaN")},
        {"mark_price": Decimal("Infinity")},
        {"fill_price": 100.25},
        {"requested_quantity": Decimal("0")},
    ],
)
def test_unsupported_or_ambiguous_entry_is_rejected(changes):
    values = inputs()
    values.update(changes)
    with pytest.raises(ValidationError):
        project_flat_paper_buy(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"position_cost_basis": Decimal("100")},
        {"position_mark_price": Decimal("100")},
        {"position_source_settlement_id": "pset-" + "a" * 32},
        {"cash": Decimal("200")},
    ],
)
def test_unavailable_flat_state_or_insufficient_cash_rejected(changes):
    values = inputs()
    values["account"] = replace(state(), **changes)
    with pytest.raises(ValidationError):
        project_flat_paper_buy(**values)


def test_other_positions_pnl_and_daily_baseline_are_preserved():
    values = inputs()
    values["account"] = replace(
        state(),
        realized_pnl=Decimal("12.34"),
        daily_pnl=Decimal("56.78"),
        daily_pnl_baseline=Decimal("9.87"),
    )
    result = project_flat_paper_buy(**values)
    assert result.realized_pnl == Decimal("12.34")
    assert result.daily_pnl == Decimal("56.28")
    assert result.daily_pnl_baseline == Decimal("9.87")
    assert result.daily_pnl_date == "2026-09-17"


def test_reentry_preserves_closed_history_until_a_new_fill():
    values = inputs()
    values["account"] = replace(
        state(),
        position_cost_basis=Decimal("80"),
        position_mark_price=Decimal("90"),
        position_source_settlement_id="pset-" + "a" * 32,
    )
    opened = project_flat_paper_buy(**values)
    assert opened.position_cost_basis == Decimal("100.25")
    values.update(
        terminal_status=TerminalOrderStatus.REJECTED, filled_quantity=Decimal("0"), fill_price=None
    )
    rejected = project_flat_paper_buy(**values)
    assert rejected.position_cost_basis == Decimal("80")
    assert rejected.position_mark_price == Decimal("90")


def test_buy_close_and_reentry_agree_with_fifo_realized_pnl():
    from datetime import timedelta

    from robo_trader.accounting.fifo import FillSide
    from robo_trader.accounting.fifo_runtime import append_runtime_fill_in_transaction
    from tests.accounting.test_fifo_runtime_settlement import _connection, _evidence

    connection, effective = _connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        for sequence, side, price in (
            (1, FillSide.BUY, "100"),
            (2, FillSide.SELL, "110"),
            (3, FillSide.BUY, "100.25"),
        ):
            fifo = append_runtime_fill_in_transaction(
                connection,
                _evidence(
                    sequence,
                    side=side,
                    quantity="2",
                    price=price,
                    commission_minor=0,
                    occurred_at=effective + timedelta(seconds=sequence),
                ),
            )
        values = inputs()
        values["account"] = replace(
            state(),
            cash=Decimal("100020"),
            realized_pnl=Decimal("20"),
            daily_pnl=Decimal("20"),
            position_cost_basis=Decimal("100"),
            position_mark_price=Decimal("110"),
            position_source_settlement_id="pset-" + "b" * 32,
        )
        result = project_flat_paper_buy(**values)
        assert result.realized_pnl == fifo.total_realized_pnl == Decimal("20")
        assert result.position_quantity == fifo.signed_quantity == Decimal("2")
        assert result.position_cost_basis == fifo.average_cost == Decimal("100.25")
        assert result.cash == Decimal("99819.50")
        assert result.daily_pnl == Decimal("19.50")
        connection.rollback()
    finally:
        connection.close()
