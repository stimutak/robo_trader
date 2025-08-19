from robo_trader.risk import RiskManager, Position


def test_position_size_basic():
    risk = RiskManager(1000, 0.02, 0.2, 2.0)
    size = risk.position_size(100_000, 200)
    assert size == 10  # 2% of 100k = 2k notionals, at $200 -> 10 shares


def test_validate_order_limits():
    risk = RiskManager(1000, 0.02, 0.2, 2.0)
    ok, msg = risk.validate_order(
        symbol="AAPL",
        order_qty=1000,
        price=100,
        equity=10_000,
        daily_pnl=0,
        current_positions={},
    )
    assert not ok and "exposure" in msg.lower()


def test_validate_order_leverage():
    risk = RiskManager(1000, 0.02, 0.2, 2.0)
    positions = {"AAPL": Position("AAPL", 50, 100)}  # 5k notional
    ok, msg = risk.validate_order(
        symbol="MSFT",
        order_qty=20,  # $2k notional equals 20% symbol exposure limit
        price=100,
        equity=10_000,
        daily_pnl=0,
        current_positions=positions,
    )
    # After order: 5k + 2k = 7k; leverage 0.7x within 2.0x; exposure at limit is allowed
    assert ok
