import pytest

from robo_trader.portfolio import Portfolio


@pytest.mark.asyncio
async def test_portfolio_pnl_and_equity():
    p = Portfolio(100_000)
    await p.update_fill("AAPL", "BUY", 10, 100)
    # cash 99,000; position 10 @ 100
    eq = await p.equity({"AAPL": 105})
    assert round(float(eq) - 100_000, 2) == 50.0  # unrealized 5 * 10 = 50

    await p.update_fill("AAPL", "SELL", 5, 110)
    # realized: (110-100)*5 = 50; remaining 5 @ 100
    eq2 = await p.equity({"AAPL": 110})
    # Simpler assertion: realized increased and equity grew
    assert float(p.realized_pnl) == 50
    assert eq2 > eq
