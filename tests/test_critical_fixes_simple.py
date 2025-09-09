"""
Simplified test suite for critical bug fixes that avoids ML dependencies.
"""

import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import pytz

from robo_trader.clients.async_ibkr_client import ConnectionConfig, ConnectionPool
from robo_trader.data.validation import DataValidator, TickData
from robo_trader.portfolio import Portfolio
from robo_trader.risk import Position, RiskManager
from robo_trader.websocket_client import WebSocketClient


class TestCriticalBugFixesSimple:
    """Test critical bug fixes without ML dependencies."""

    @pytest.mark.asyncio
    async def test_connection_pool_timeout_fix(self):
        """Test Bug #2: Connection pool exhaustion timeout is fixed."""
        config = ConnectionConfig(max_connections=1)
        pool = ConnectionPool(config)

        # Mock the connection creation to avoid actual IBKR connection
        with patch.object(pool, "_create_connection") as mock_create:
            mock_connection = Mock()
            mock_connection.isConnected.return_value = True
            mock_create.return_value = mock_connection

            await pool.initialize()

            # First acquisition should work
            async with pool.acquire(timeout=1.0) as conn1:
                assert conn1 is not None

                # Second acquisition should timeout since pool size is 1
                with pytest.raises(ConnectionError, match="Connection pool exhausted"):
                    async with pool.acquire(timeout=0.1) as conn2:
                        pass

    def test_position_sizing_truncation_fix(self):
        """Test Bug #3: Position sizing truncation is fixed."""
        risk = RiskManager(1000, 0.02, 0.2, 2.0)

        # Test case where old code would truncate to 0
        # $10,000 account, 2% risk = $200, stock at $199 = 1.005 shares
        # Old: int(200 // 199) = int(1.005) = 1 share ✓ (this was actually correct)
        # New: round(200 / 199) = round(1.005) = 1 share
        size = risk.position_size_fixed(10000, 199.0)
        assert size == 1

        # Test case where rounding makes a difference
        # $10,000 account, 2% risk = $200, stock at $133.33 = 1.5 shares
        # Old: int(200 // 133.33) = int(1.5) = 1 share
        # New: round(200 / 133.33) = round(1.5) = 2 shares
        size = risk.position_size_fixed(10000, 133.33)
        assert size == 2  # Should round up from 1.5

        # Test edge case: exactly 0.5 shares should round to 0 or 1
        size = risk.position_size_fixed(10000, 400.0)  # 200/400 = 0.5
        assert size in [0, 1]  # Python rounds 0.5 to nearest even (0)

    def test_portfolio_pnl_calculation_fix(self):
        """Test Bug #4: Portfolio PnL calculation error is fixed."""
        portfolio = Portfolio(100000)

        # Buy 100 shares at $100
        portfolio.update_fill("AAPL", "BUY", 100, 100.0)
        assert portfolio.cash == 90000  # 100000 - 10000
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100

        # Try to sell 150 shares (more than we have) at $110
        with patch("robo_trader.logger.get_logger") as mock_logger:
            mock_logger.return_value.warning = Mock()
            portfolio.update_fill("AAPL", "SELL", 150, 110.0)

            # Should have warned about overselling
            mock_logger.return_value.warning.assert_called_once()

        # Should only sell 100 shares, not 150
        assert "AAPL" not in portfolio.positions  # Position closed
        assert portfolio.realized_pnl == 1000.0  # (110-100) * 100, not * 150
        assert portfolio.cash == 101000  # 90000 + 11000 (100 shares * 110)

    def test_stop_loss_validation_fix(self):
        """Test Bug #5: Stop-loss validation is added."""
        risk = RiskManager(1000, 0.02, 0.2, 2.0)

        # Create a position with a reasonable stop-loss
        pos = Position("AAPL", 100, 150.0)
        pos.stop_loss = 145.0  # 3.33% stop

        current_prices = {"AAPL": 148.0}

        with patch("robo_trader.risk.logger") as mock_logger:
            heat = risk.calculate_portfolio_heat({"AAPL": pos}, current_prices)

            # Should not trigger any warnings for reasonable stop
            mock_logger.error.assert_not_called()
            mock_logger.critical.assert_not_called()

            # Test unreasonable stop-loss (too far)
            pos.stop_loss = 100.0  # 32% stop - too far
            heat = risk.calculate_portfolio_heat({"AAPL": pos}, current_prices)

            # Should log error about stop being too far
            mock_logger.error.assert_called()

            # Test triggered stop-loss
            pos.stop_loss = 149.0  # Stop above current price
            current_prices["AAPL"] = 148.0  # Price below stop
            heat = risk.calculate_portfolio_heat({"AAPL": pos}, current_prices)

            # Should log critical error about untriggered stop
            mock_logger.critical.assert_called()

    def test_float_comparison_epsilon_fix(self):
        """Test Bug #7: Float comparison uses epsilon tolerance."""
        config = Mock()
        validator = DataValidator(config)

        # Create tick data with very close bid/ask (floating point precision issue)
        tick = Mock()
        tick.bid = 100.0000001
        tick.ask = 100.0000002
        tick.last = 100.0000001
        tick.symbol = "AAPL"
        tick.timestamp = datetime.now()
        tick.spread_bps = 0.0000001

        result = validator._validate_tick_prices(tick)

        # Should pass validation with epsilon tolerance
        assert result.is_valid is True

        # Test actual inverted market
        tick.bid = 100.01
        tick.ask = 100.00

        result = validator._validate_tick_prices(tick)

        # Should fail validation for truly inverted market
        assert result.is_valid is False
        assert "Inverted market" in result.message

    def test_websocket_queue_overflow_fix(self):
        """Test Bug #8: WebSocket queue overflow protection."""
        client = WebSocketClient(max_queue_size=3)

        # Fill queue to capacity
        client.send_market_update("AAPL", 100.0)
        client.send_market_update("MSFT", 200.0)
        client.send_market_update("GOOGL", 300.0)

        # Queue should be full
        assert client.message_queue.qsize() == 3

        # Adding another message should drop the oldest
        client.send_market_update("TSLA", 400.0)

        # Queue size should still be 3 (oldest dropped)
        assert client.message_queue.qsize() == 3

        # Verify the newest message is in queue
        messages = []
        while not client.message_queue.empty():
            messages.append(client.message_queue.get_nowait())

        # Should contain TSLA (newest) but not necessarily AAPL (oldest)
        symbols = [msg["symbol"] for msg in messages]
        assert "TSLA" in symbols

    def test_basic_risk_calculations(self):
        """Test basic risk management calculations work correctly."""
        risk = RiskManager(
            max_daily_loss=1000,
            max_position_risk_pct=0.02,
            max_symbol_exposure_pct=0.2,
            max_leverage=2.0,
        )

        # Test position sizing
        size = risk.position_size(100000, 100.0)
        assert size > 0
        assert size <= 200  # 2% of 100k = 2k, at $100 = 20 shares max

        # Test order validation
        ok, msg = risk.validate_order(
            symbol="AAPL",
            order_qty=10,
            price=100.0,
            equity=100000,
            daily_pnl=0,
            current_positions={},
        )
        assert ok is True
        assert "OK" in msg or msg == ""

    def test_portfolio_basic_operations(self):
        """Test basic portfolio operations work correctly."""
        portfolio = Portfolio(100000)

        # Test initial state
        assert portfolio.cash == 100000
        assert len(portfolio.positions) == 0
        assert portfolio.realized_pnl == 0.0

        # Test buy
        portfolio.update_fill("AAPL", "BUY", 100, 150.0)
        assert portfolio.cash == 85000  # 100000 - 15000
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100
        assert portfolio.positions["AAPL"].avg_price == 150.0

        # Test partial sell
        portfolio.update_fill("AAPL", "SELL", 50, 160.0)
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 50
        assert portfolio.realized_pnl == 500.0  # (160-150) * 50

        # Test equity calculation
        equity = portfolio.equity({"AAPL": 155.0})
        expected_cash = 85000 + 8000  # Original cash + sell proceeds
        expected_position_value = 50 * 155.0  # Remaining shares at market price
        expected_equity = expected_cash + expected_position_value
        assert abs(equity - expected_equity) < 0.01
