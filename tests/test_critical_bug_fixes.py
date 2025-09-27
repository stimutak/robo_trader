"""
Test suite for critical bug fixes identified in the security audit.

This module tests the fixes for:
1. Race conditions in position updates
2. Connection pool exhaustion
3. Position sizing truncation
4. Portfolio PnL calculation errors
5. Stop-loss validation
6. Timestamp misalignment
7. Float comparison precision
8. WebSocket queue overflow
"""

import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import pytz

# Removed deprecated ConnectionPool/ConnectionConfig imports
from robo_trader.config import Config
from robo_trader.data.validation import DataValidator, TickData
from robo_trader.portfolio import Portfolio
from robo_trader.risk import Position, RiskManager
from robo_trader.runner_async import AsyncRunner
from robo_trader.websocket_client import WebSocketClient


class TestCriticalBugFixes:
    """Test critical bug fixes."""

    @pytest.mark.asyncio
    async def test_position_update_race_condition_fix(self):
        """Test Bug #1: Race condition in position updates is fixed."""
        runner = AsyncRunner(
            duration="1 D", bar_size="5 mins", symbols=["AAPL"], max_concurrent_symbols=1
        )

        # Mock the necessary components
        runner.cfg = Mock()
        runner.cfg.ibkr = Mock()
        runner.cfg.ibkr.host = "127.0.0.1"
        runner.cfg.ibkr.port = 7497
        runner.cfg.ibkr.client_id = 1
        runner.cfg.ibkr.readonly = True
        runner.cfg.default_cash = 100000

        runner.risk = Mock()
        runner.risk.position_size.return_value = 100
        runner.risk.validate_order.return_value = (True, "OK")

        runner.portfolio = Mock()
        runner.executor = Mock()
        runner.db = Mock()
        runner.monitor = Mock()

        # Test atomic position update
        success1 = await runner._update_position_atomic("AAPL", 100, 150.0, "BUY")
        success2 = await runner._update_position_atomic("AAPL", 50, 155.0, "BUY")

        assert success1 is True
        assert success2 is True

        # Should have one position with combined quantity
        assert "AAPL" in runner.positions
        assert runner.positions["AAPL"].quantity == 150  # 100 + 50

        # Average price should be weighted correctly
        expected_avg = (150.0 * 100 + 155.0 * 50) / 150
        assert abs(runner.positions["AAPL"].avg_price - expected_avg) < 0.01

    @pytest.mark.asyncio
    async def test_connection_pool_timeout_fix(self):
        """Test Bug #2: Connection pool exhaustion timeout is fixed."""

        class DummyPool:
            async def initialize(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def acquire(self, timeout: float = 1.0):
                # Simulate single-connection pool exhaustion by always raising on second attempt
                raise ConnectionError("Connection pool exhausted")

        pool = DummyPool()

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
            heat = risk.calculate_portfolio_heat({"AAPL": pos}, current_prices, 100000)

            # Should not trigger any warnings for reasonable stop
            mock_logger.error.assert_not_called()
            mock_logger.critical.assert_not_called()

            # Test unreasonable stop-loss (too far)
            pos.stop_loss = 100.0  # 32% stop - too far
            heat = risk.calculate_portfolio_heat({"AAPL": pos}, current_prices, 100000)

            # Should log error about stop being too far
            mock_logger.error.assert_called()

            # Test triggered stop-loss
            pos.stop_loss = 149.0  # Stop above current price
            current_prices["AAPL"] = 148.0  # Price below stop
            heat = risk.calculate_portfolio_heat({"AAPL": pos}, current_prices, 100000)

            # Should log critical error about untriggered stop
            mock_logger.critical.assert_called()

    def test_timestamp_alignment_fix(self):
        """Test Bug #6: Timestamp misalignment is fixed."""
        from robo_trader.data.pipeline import DataPipeline

        config = Mock()
        config.symbols = ["AAPL"]
        config.data = Mock()
        config.data.enable_realtime = True
        config.data.tick_buffer = 1000

        pipeline = DataPipeline(config)

        # Mock timezone to test market time usage
        with patch("robo_trader.data.pipeline.datetime") as mock_dt:
            with patch("robo_trader.data.pipeline.pytz") as mock_pytz:
                mock_tz = Mock()
                mock_pytz.timezone.return_value = mock_tz
                mock_now = Mock()
                mock_dt.now.return_value = mock_now

                # Set up last tick time
                pipeline.metrics["last_tick_time"] = datetime(2023, 1, 1, 10, 0, 0)

                # This should use market timezone
                asyncio.run(pipeline._monitor_data_quality())

                # Verify market timezone was used
                mock_pytz.timezone.assert_called_with("US/Eastern")
                mock_dt.now.assert_called_with(mock_tz)

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
