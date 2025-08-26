from robo_trader.risk import RiskManager, Position


def test_per_order_notional_cap_blocks_large_order():
    risk = RiskManager(1000, 0.02, 0.2, 2.0, max_order_notional=1500)
    ok, msg = risk.validate_order(
        symbol="AAPL",
        order_qty=20,
        price=100,
        equity=10_000,
        daily_pnl=0,
        current_positions={},
    )
    assert not ok and "per-order" in msg.lower()


def test_daily_notional_cap_blocks_when_exceeded():
    risk = RiskManager(1000, 0.02, 0.2, 2.0, max_daily_notional=2000)
    # Already executed 1500, now try to add 600 -> exceeds
    ok, msg = risk.validate_order(
        symbol="MSFT",
        order_qty=6,
        price=100,
        equity=10_000,
        daily_pnl=0,
        current_positions={},
        daily_executed_notional=1500,
    )
    assert not ok and "daily notional" in msg.lower()
